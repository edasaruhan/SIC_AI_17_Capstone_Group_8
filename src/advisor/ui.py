# pyright: reportMissingImports=false
"""Streamlit interface over the advisor: the visibility graph and the description audit.

The graph is the product; this page only drives it. Everything that costs money or
decides a result lives in ``service.py`` and ``audit.py``, shared with the CLI. The page
shows the call estimate and asks for confirmation before anything is paid, streams the
graph node by node, lets the user correct the extracted rivals and rerun from the cache,
and draws the measured numbers instead of only printing them.

    make advisor-ui
"""

from __future__ import annotations

import asyncio
import html
import re

import altair as alt
import pandas as pd
import streamlit as st

from advisor import assistants, audit, model, service
from advisor.advise import DIAGNOSES
from advisor.clients import require_keys
from advisor.render import DIAGNOSIS_NOTE

st.set_page_config(page_title="Marka Görünürlük Danışmanı", page_icon="🔎", layout="wide")

NAVY = "#1f275f"
SERIES = {"gemini": "#2a78d6", "cerebras": "#eb6834"}  # validated pair, direct-labelled
MUTED = "#b4b2a9"
VERDICT_BADGE = {
    "strong": ("good", "İki asistanda da kazandırıyor"),
    "mixed": ("flat", "Asistana bağlı"),
    "weak": ("weak", "Belirgin kazanç yok"),
    "risk": ("bad", "⚠ Risk"),
}
ADVICE_ICON = {
    "risk": "⚠️",
    "add": "➕",
    "differentiate": "↔️",
    "mixed": "◐",
    "replace": "✏️",
    "empty": "○",
}
DIAGNOSIS_TONE = {
    "absent": "bad",
    "low_rank": "flat",
    "ceiling": "flat",
    "leader": "good",
    "thin": "weak",
}

CSS = """
<style>
  .block-container { padding-top: 1.6rem; max-width: 1240px; }
  .hero { background: #1f275f; color: #fff; border-radius: 14px; padding: 26px 30px; margin-bottom: 18px; }
  .hero .kicker { font-size: 12px; letter-spacing: 1.6px; text-transform: uppercase; color: #a9aed6; }
  .hero h1 { color: #fff; font-size: 30px; margin: 4px 0 6px; padding: 0; line-height: 1.2; }
  .hero p { color: #d6d9ef; margin: 0; font-size: 15.5px; max-width: 880px; }
  .gates { display: flex; gap: 10px; margin-top: 16px; flex-wrap: wrap; }
  .gate { background: #2d3673; border-radius: 10px; padding: 10px 14px; flex: 1; min-width: 220px; }
  .gate b { color: #fff; display: block; font-size: 14.5px; }
  .gate span { color: #c9cde8; font-size: 13px; }
  .badge { display: inline-block; font-size: 12.5px; font-weight: 700; padding: 2px 9px; border-radius: 10px; white-space: nowrap; }
  .badge.good { color: #1a6b2e; background: #e5f1e7; }
  .badge.flat { color: #5c5b55; background: #ededea; }
  .badge.weak { color: #6b6a63; background: #f4f3ef; }
  .badge.bad { color: #b8322f; background: #f8e5e4; }
  .diag { border-radius: 12px; padding: 16px 20px; margin: 6px 0 14px; border-left: 6px solid #6b6a63; background: #f6f6f8; }
  .diag.bad { border-left-color: #b8322f; } .diag.flat { border-left-color: #2a78d6; }
  .diag.good { border-left-color: #1a6b2e; }
  .diag .k { font-size: 12px; letter-spacing: 1.2px; text-transform: uppercase; color: #6b6a63; }
  .diag .t { font-size: 22px; font-weight: 700; margin: 2px 0 6px; color: #14140f; }
  .diag .n { font-size: 15px; color: #3f3e38; }
  .chips { display: flex; gap: 8px; flex-wrap: wrap; margin: 4px 0 10px; }
  .chip { background: #f2f2f5; border-radius: 16px; padding: 4px 12px; font-size: 13.5px; color: #3f3e38; }
  .finding { border: 1px solid #e1e0d9; border-radius: 10px; padding: 10px 14px; margin-bottom: 8px; }
  .finding .top { display: flex; justify-content: space-between; gap: 10px; align-items: center; }
  .finding .lbl { font-weight: 700; font-size: 15px; }
  .finding .q { color: #52514e; font-size: 13.5px; margin-top: 4px; font-style: italic; }
  .advice { border-radius: 10px; padding: 10px 14px; margin-bottom: 8px; background: #f6f6f8; }
  .advice.risk { background: #fbeeed; }
  .advice b { display: block; margin-bottom: 2px; }
  .advice span { color: #52514e; font-size: 13.5px; }
</style>
"""


def badge(tone: str, text: str) -> str:
    return f'<span class="badge {tone}">{html.escape(text)}</span>'


@st.cache_resource(show_spinner="Öğrenilmiş sinyal modeli yükleniyor…")
def load_model() -> dict:
    return model.load()


def split(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


def pct(value: float | None) -> str:
    return "–" if value is None else f"%{100 * value:.0f}"


# --- charts ---------------------------------------------------------------------------


def share_chart(rows: list[dict], *, chance: float | None = 0.2) -> alt.LayerChart:
    """Win share per sentence type, two assistants side by side, chance as a dashed rule."""
    data = pd.DataFrame(
        [
            {
                "tür": row["label"],
                "asistan": assistants.label(name),
                "pay": row[name],
            }
            for row in rows
            for name in ("gemini", "cerebras")
        ]
    )
    domain = [assistants.label("gemini"), assistants.label("cerebras")]
    base = alt.Chart(data).encode(
        y=alt.Y("tür:N", sort=None, title=None, axis=alt.Axis(labelLimit=260, labelFontSize=13)),
        yOffset=alt.YOffset("asistan:N", sort=domain),
        x=alt.X(
            "pay:Q",
            title="Kazanma payı",
            scale=alt.Scale(domain=[0, 1]),
            axis=alt.Axis(format="%", grid=True, gridColor="#e1e0d9", tickCount=5),
        ),
        tooltip=[
            alt.Tooltip("tür:N", title="Cümle türü"),
            alt.Tooltip("asistan:N", title="Asistan"),
            alt.Tooltip("pay:Q", title="Kazanma payı", format=".0%"),
        ],
    )
    bars = base.mark_bar(cornerRadiusEnd=4, height=11).encode(
        color=alt.Color(
            "asistan:N",
            scale=alt.Scale(domain=domain, range=[SERIES["gemini"], SERIES["cerebras"]]),
            legend=alt.Legend(orient="top", title=None, labelFontSize=13),
        )
    )
    labels = base.mark_text(align="left", dx=5, fontSize=11.5, color="#3f3e38").encode(
        text=alt.Text("pay:Q", format=".0%")
    )
    layers = [bars, labels]
    if chance is not None:
        layers.append(
            alt.Chart(pd.DataFrame({"şans": [chance]}))
            .mark_rule(strokeDash=[5, 4], color="#898781")
            .encode(x="şans:Q")
        )
    return alt.layer(*layers).properties(height=max(120, 44 * len(rows)), width="container")


def measures_chart(per_assistant: dict[str, dict]) -> alt.Chart:
    rows = []
    for name, m in per_assistant.items():
        for key, text in (
            ("mention_off", "Arama kapalıyken anılma"),
            ("mention_on", "Arama açıkken anılma"),
            ("first_on", "İlk anılan marka"),
        ):
            rows.append({"ölçü": text, "asistan": assistants.label(name), "oran": m.get(key, 0)})
    data = pd.DataFrame(rows)
    names = list(per_assistant)
    domain = [assistants.label(n) for n in names]
    base = alt.Chart(data).encode(
        y=alt.Y("ölçü:N", sort=None, title=None, axis=alt.Axis(labelFontSize=13, labelLimit=240)),
        yOffset=alt.YOffset("asistan:N", sort=domain),
        x=alt.X("oran:Q", title=None, scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        tooltip=[
            alt.Tooltip("ölçü:N", title="Ölçü"),
            alt.Tooltip("asistan:N", title="Asistan"),
            alt.Tooltip("oran:Q", title="Oran", format=".0%"),
        ],
    )
    bars = base.mark_bar(cornerRadiusEnd=4, height=14).encode(
        color=alt.Color(
            "asistan:N",
            scale=alt.Scale(domain=domain, range=[SERIES.get(n, NAVY) for n in names]),
            legend=alt.Legend(orient="top", title=None) if len(names) > 1 else None,
        )
    )
    text = base.mark_text(align="left", dx=5, fontSize=12, color="#3f3e38").encode(
        text=alt.Text("oran:Q", format=".0%")
    )
    return (bars + text).properties(height=70 * 3 if len(names) > 1 else 150, width="container")


def rivals_chart(state: dict, brand: str) -> alt.Chart | None:
    counts: dict[str, int] = {}
    for row in state.get("observations") or []:
        if row.get("assistant", assistants.PRIMARY) != assistants.PRIMARY:
            continue
        for name in row.get("named_brands") or []:
            counts[name] = counts.get(name, 0) + 1
    if not counts:
        return None
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
    if brand in counts and brand not in dict(top):
        top.append((brand, counts[brand]))
    data = pd.DataFrame([{"marka": n, "yanıt": c, "sen": n == brand} for n, c in top])
    base = alt.Chart(data).encode(
        y=alt.Y("marka:N", sort="-x", title=None, axis=alt.Axis(labelFontSize=13)),
        x=alt.X("yanıt:Q", title="Andığı Gemini yanıtı", axis=alt.Axis(tickMinStep=1)),
        tooltip=[alt.Tooltip("marka:N", title="Marka"), alt.Tooltip("yanıt:Q", title="Yanıt")],
    )
    bars = base.mark_bar(cornerRadiusEnd=4, height=16).encode(
        color=alt.condition(alt.datum.sen, alt.value(SERIES["gemini"]), alt.value(MUTED))
    )
    text = base.mark_text(align="left", dx=5, fontSize=12, color="#3f3e38").encode(text="yanıt:Q")
    return (bars + text).properties(height=34 * len(top), width="container")


# --- audit output (shared by both tools) ----------------------------------------------


def show_audit(record: dict) -> None:
    rows = record.get("findings") or []
    advice = record.get("advice") or []
    counts = {kind: sum(r["verdict"] == kind for r in rows) for kind in VERDICT_BADGE}
    st.markdown(
        '<div class="chips">'
        f'<span class="chip">Kazandıran tür: <b>{counts["strong"]}</b></span>'
        f'<span class="chip">Asistana bağlı: <b>{counts["mixed"]}</b></span>'
        f'<span class="chip">Kazanç yok: <b>{counts["weak"]}</b></span>'
        f'<span class="chip">Risk: <b>{counts["risk"]}</b></span>'
        f'<span class="chip">Yöntem: <b>{"yapay zekâ" if record.get("method") == "ai" else "kurallar"}</b></span>'
        "</div>",
        unsafe_allow_html=True,
    )
    if counts["risk"]:
        st.error(
            "Açıklamada kaynağı gösterilmeyen bir kurum ya da klinik iddiası var. Deneyde bu tür "
            "cümle kazandırdı ama asistanlar onu sorgulamadan kullanıcıya aktardı; kaldırın ya da "
            "kaynağını verin.",
            icon="⚠️",
        )
    left, right = st.columns([1.05, 1], gap="large")
    with left:
        st.markdown("##### Açıklamada bulunanlar")
        if not rows:
            st.info("Deneyde ölçülen cümle türlerinin hiçbiri bulunamadı.")
        for row in rows:
            tone, text = VERDICT_BADGE[row["verdict"]]
            rival = (
                f' · rakiplerde {row["in_rivals"]}/{record["rivals"]}'
                if record.get("rivals")
                else ""
            )
            st.markdown(
                f'<div class="finding"><div class="top"><span class="lbl">{html.escape(row["label"])}</span>'
                f"{badge(tone, text)}</div>"
                f'<div class="q">“{html.escape(row["sentences"][0][:220])}”{html.escape(rival)}</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown("##### Ne yapmalı?")
        for item in advice:
            kind = item.get("kind", "")
            st.markdown(
                f'<div class="advice {"risk" if kind == "risk" else ""}">'
                f'<b>{ADVICE_ICON.get(kind, "•")} {html.escape(item["suggestion"])}</b>'
                f'<span>{html.escape(item["basis"])}</span></div>',
                unsafe_allow_html=True,
            )
    with right:
        st.markdown("##### Bu türler deneyde ne kadar kazandırdı?")
        if rows:
            st.altair_chart(share_chart(rows))
            st.caption(
                "Her kartta farklı bir cümle olduğunda o türün önerilen ürün olma payı; "
                "kesikli çizgi rastgele seçim (%20). Kaynak: açıklama deneyi, 2. tur."
            )
        with st.expander("Deneydeki bütün cümle türleri"):
            everything = [
                {
                    "label": audit.LABELS[key],
                    "gemini": audit.shares(key)[0],
                    "cerebras": audit.shares(key)[1],
                }
                for key in audit.WINS
            ]
            st.altair_chart(share_chart(everything))
    st.caption(audit.METHOD_NOTE.get(record.get("method", "rules"), ""))


# --- visibility tool ---------------------------------------------------------------


def run_with_progress(options: service.Options, boosters: dict) -> dict | None:
    """Run the graph inside a status box that lists each node as it finishes."""
    with st.status("Akış çalışıyor…", expanded=True) as status:
        if "cerebras" in options.assistants:
            status.write("gpt-oss-120b hız sınırıyla çalışır; çağrı başına ~25 sn sürer.")

        async def go() -> dict:
            final: dict = {}
            async for node, state in service.stream(options, boosters):
                label = service.STEP_LABELS.get(node)
                if label:
                    status.write(f"✓ {label}")
                final = state
            return final

        try:
            final = asyncio.run(go())
        except Exception as exc:  # noqa: BLE001 - the page must show every failure
            status.update(label="Akış durdu", state="error", expanded=True)
            st.error(f"{exc}")
            return None
        service.save(options, final)
        status.update(label="Tamamlandı", state="complete", expanded=False)
    return final


def show_result(options: service.Options, state: dict, boosters: dict) -> None:
    measures = state.get("measures") or {}
    scores = state.get("scores") or {}
    diagnosis = state.get("diagnosis", "thin")

    st.markdown(
        f'<div class="diag {DIAGNOSIS_TONE.get(diagnosis, "weak")}">'
        f'<div class="k">{html.escape(options.brand)} · {html.escape(options.sector)} · teşhis</div>'
        f'<div class="t">{html.escape(str(DIAGNOSES.get(diagnosis, diagnosis)))}</div>'
        f'<div class="n">{html.escape(DIAGNOSIS_NOTE.get(diagnosis, ""))}</div></div>',
        unsafe_allow_html=True,
    )
    columns = st.columns(4)
    tiles = [
        ("Arama sonuçlarında görünme", pct(measures.get("retrieval_presence"))),
        ("Arama açıkken anılma", pct(measures.get("mention_on"))),
        ("İlk anılan marka", pct(measures.get("first_on"))),
        (
            "Rakiplere göre sıra (tahmin)",
            f"{scores.get('rank', '–')} / {scores.get('candidates', '–')}" if scores else "–",
        ),
    ]
    for column, (label, value) in zip(columns, tiles, strict=True):
        with column.container(border=True):
            st.metric(label, value)

    tabs = st.tabs(["📊 Ölçüm", "🏷️ Rakipler", "📝 Ürün açıklaması", "✅ Öneriler", "📄 Tam rapor"])
    with tabs[0]:
        per = state.get("assistant_measures") or {assistants.PRIMARY: measures}
        st.altair_chart(measures_chart(per))
        if len(per) > 1:
            st.caption(
                "Teşhis Gemini'ye göre konur; ikinci asistan sonucun tek asistana özgü olup "
                "olmadığını gösterir."
            )
        rows = state.get("signals") or []
        if rows:
            st.markdown("##### Sektörler arasında taşınan sinyaller")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Sinyal": r["label"],
                            "Sen": round(r["brand"], 2),
                            "Rakipler": round(r["rivals"], 2),
                            "Durum": "geride" if r["lagging"] else "",
                        }
                        for r in rows
                    ]
                ),
                hide_index=True,
            )
    with tabs[1]:
        chart = rivals_chart(state, options.brand)
        if chart is not None:
            st.altair_chart(chart)
        rivals = [name for name in state.get("candidates") or [] if name != options.brand]
        with st.expander("Karşılaştırılan markaları düzelt", expanded=False):
            st.caption(
                "Liste arama sonuçlarından çıkarıldı. Yanlış olanları kaldırın, eksik olanları "
                "ekleyin. Düzeltilmiş koşu önbellekten okunur; yeni ücretli çağrı yapılmaz."
            )
            keep = st.multiselect("Rakipler", options=rivals, default=rivals)
            extra = st.text_input("Eksik rakipler (virgülle)", key="extra_rivals")
            if st.button("Düzeltmeyle yeniden hesapla"):
                corrected = service.Options(
                    **{
                        **options.__dict__,
                        "add_rivals": tuple(dict.fromkeys([*options.add_rivals, *split(extra)])),
                        "drop_rivals": tuple(
                            dict.fromkeys(
                                [*options.drop_rivals, *(r for r in rivals if r not in keep)]
                            )
                        ),
                    }
                )
                final = run_with_progress(corrected, boosters)
                if final is not None:
                    st.session_state["result"] = {"options": corrected, "state": final}
                    st.rerun()
    with tabs[2]:
        record = state.get("description_audit") or {}
        if record.get("findings") is None and not record.get("advice"):
            st.info("Markayı anlatan bir metin bulunamadı; açıklama denetimi yapılmadı.")
        else:
            source = (
                "Verdiğin ürün açıklaması denetlendi."
                if record.get("source") == "user"
                else "Açıklama verilmediği için arama sonuçlarında markanı anlatan özetler "
                "denetlendi: asistanın senin hakkında gördüğü metin."
            )
            if record.get("rival_names"):
                source += " Rakipler: " + ", ".join(record["rival_names"]) + "."
            st.caption(source)
            show_audit(record)
    with tabs[3]:
        for index, rec in enumerate(state.get("recommendations", []), 1):
            with st.container(border=True):
                st.markdown(f"**{index}. {rec['title']}**")
                st.write(rec["why"])
                st.markdown(f"**Yapılacak:** {rec['action']}")
                st.caption(rec["verdict"])
                if rec.get("targets"):
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "Alan adı": t["domain"],
                                    "Andığı rakipler": ", ".join(t["rivals"]),
                                    "Sıra": t.get("position"),
                                    "Sayfa": t["link"],
                                }
                                for t in rec["targets"]
                            ]
                        ),
                        hide_index=True,
                        column_config={"Sayfa": st.column_config.LinkColumn("Sayfa")},
                    )
    with tabs[4]:
        st.download_button(
            "Raporu indir (.md)",
            data=state.get("report", ""),
            file_name=f"{options.brand}-{options.sector}-gorunurluk.md",
            mime="text/markdown",
        )
        st.markdown(state.get("report", ""))


def visibility_tool() -> None:
    try:
        boosters = load_model()
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"Öğrenilmiş sinyal modeli yüklenemedi: {exc}")
        st.code("make advisor-train")
        return

    with st.container(border=True):
        first, second, third = st.columns([2, 2, 1])
        brand = first.text_input("Marka", placeholder="ör. Garanti BBVA", key="v_brand")
        sector = second.text_input("Sektör", placeholder="ör. bankacılık", key="v_sector")
        language = third.selectbox("Dil", ["tr", "en"], key="v_language")
        aliases = st.text_input(
            "Markanın diğer yazılışları (virgülle)", placeholder="ör. Garanti", key="v_aliases"
        )
        description = st.text_area(
            "Ürün açıklaman (isteğe bağlı)",
            placeholder="Boş bırakırsan arama sonuçlarında markanı anlatan özetler denetlenir.",
            height=90,
            key="v_description",
        )
        left, right = st.columns(2)
        second_assistant = left.toggle(
            "gpt-oss-120b ile de ölç",
            key="v_second",
            help="Aynı sorular Cerebras'taki gpt-oss-120b'ye de sorulur. Hız sınırı nedeniyle "
            "çağrı başına ~25 sn sürer; teşhis yine Gemini'ye göre konur.",
        )
        audit_ai = right.toggle(
            "Açıklamayı yapay zekâ ile sınıflandır",
            key="v_audit_ai",
            help="Cümle türlerini Gemini bulur (+1 çağrı); anahtar ifade kuralları yalnız "
            "deneyin iki kategorisini tanır.",
        )
        with st.expander("Gelişmiş ayarlar"):
            a, b, c = st.columns(3)
            queries = a.slider("Sorgu sayısı", 1, 5, 3)
            reps = b.slider("Tekrar", 1, 3, 2)
            max_calls = c.number_input("En fazla ücretli çağrı", 5, 120, 40)

    if not (brand.strip() and sector.strip()):
        st.info("Başlamak için bir marka ve sektör girin.")
    else:
        options = service.Options(
            brand=brand.strip(),
            sector=sector.strip(),
            language=language,
            brand_aliases=split(aliases),
            queries=int(queries),
            reps=int(reps),
            max_calls=int(max_calls),
            description=description.strip(),
            audit_ai=audit_ai,
            assistants=(
                (assistants.PRIMARY, "cerebras") if second_assistant else (assistants.PRIMARY,)
            ),
        )
        calls = service.estimated_calls(options)
        cached = (service.run_folder(options) / "report.md").exists()
        st.markdown(
            '<div class="chips">'
            f'<span class="chip">Tahmini ücretli çağrı: <b>{calls}</b></span>'
            f'<span class="chip">Bütçe: <b>{int(max_calls)}</b></span>'
            + (
                '<span class="chip">Bu analiz daha önce yapıldı: kayıtlı çağrılar önbellekten okunur</span>'
                if cached
                else ""
            )
            + "</div>",
            unsafe_allow_html=True,
        )
        over_budget = calls > int(max_calls)
        if over_budget:
            st.warning(
                "Tahmin bütçenin üstünde; sorgu veya tekrar sayısını azaltın ya da bütçeyi artırın."
            )
        confirmed = st.checkbox("Ücretli çağrıları onaylıyorum")
        if st.button("Analizi başlat", type="primary", disabled=over_budget or not confirmed):
            try:
                require_keys(options.assistants)
            except ValueError as exc:
                st.error(f"{exc} — anahtarı `.env` dosyasına ekleyin.")
                return
            final = run_with_progress(options, boosters)
            if final is not None:
                st.session_state["result"] = {"options": options, "state": final}

    result = st.session_state.get("result")
    if result:
        show_result(result["options"], result["state"], boosters)


# --- audit tool --------------------------------------------------------------------


def audit_tool() -> None:
    st.markdown(
        "Ürün listede olduğunda asistan **somut ürün bilgisine** göre seçiyor. Açıklamanı "
        "yapıştır; hangi cümle türlerini taşıdığını ve deneyde bu türlerin iki asistanda ne kadar "
        "kazandırdığını gör."
    )
    with st.container(border=True):
        left, right = st.columns(2)
        text = left.text_area(
            "Ürün açıklaman",
            height=200,
            key="a_text",
            placeholder="ör. 60'tan fazla ülkede sunucu, WireGuard ve AES-256 şifreleme…",
        )
        rivals_raw = right.text_area(
            "Rakip açıklamaları (isteğe bağlı; her birini boş bir satırla ayır)",
            height=200,
            key="a_rivals",
        )
        use_ai = st.toggle(
            "Yapay zekâ ile sınıflandır (Gemini, 1 ücretli çağrı; her sektörde çalışır)",
            key="a_ai",
            help="Kapalıyken ücretsiz anahtar ifade kuralları çalışır; kurallar deneyin iki "
            "kategorisindeki (güneş kremi, VPN) cümlelerle kuruldu.",
        )
        go = st.button("Denetle", type="primary", disabled=not text.strip(), key="a_go")
    if go:
        rivals = tuple(p.strip() for p in re.split(r"\n\s*\n", rivals_raw) if p.strip())
        if use_ai:
            try:
                with st.spinner("Gemini cümle türlerini sınıflandırıyor…"):
                    result = audit.audit_with_ai(text, rivals)
            except Exception as exc:  # noqa: BLE001 - fall back and say so
                st.warning(f"Yapay zekâ sınıflandırması alınamadı ({exc}); kurallar kullanıldı.")
                result = audit.audit(text, rivals)
        else:
            result = audit.audit(text, rivals)
        st.session_state["audit"] = audit.as_record(result)
    record = st.session_state.get("audit")
    if record:
        show_audit(record)


# --- about -----------------------------------------------------------------------


def about() -> None:
    st.markdown(
        """
#### Asistan neye göre öneriyor?

| | Bulgu | Kanıt |
|---|---|---|
| **1. Listeye girmek** | Bağımsız bir kaynakta, rakiplerle birlikte ve üstte görünmek | Karşılaştırma sayfası anılmayı %2 → %33; 1. sırada %65 (1.080 çağrı, Gemini) |
| **2. Seçilmek** | Ürüne özgü somut bilgi: teknik ayrıntı, fiyat avantajı | Kazanma payı teknik %35 / %67, fiyat %35 / %50; şans %20 (Gemini / gpt-oss-120b) |
| **Varsayılan** | İçerik sessizken tanınmış marka kazanır | Aynı kartlarda tanınmış %50–65, kurgusal %5–10 |
| **Risk** | Kaynaksız kurum iddiası kazandırır ve sorgulanmaz | %69–79 kullanıcıya aktarıldı; bu araç bunu hiçbir zaman önermez |

#### Akış

```
plan → ara → rakipleri çıkar → asistana sor → ölç ve teşhis et → açıklamayı denetle → öneriler → rapor
```

Ölçüm bir modelin görüşü değildir: anılma, ilk anılma ve sıra, marka adları eşleştirilerek
ölçülür. Rakipler arama sonuçlarından çıkarılır ve düzeltilebilir. Her ücretli çağrı makbuza
yazılır; aynı analiz tekrar ödenmez. Her öneri kodda tutulan etik filtreden geçer.

#### Sınırlar

- Etkiler iki asistanda ölçüldü; ChatGPT ve Claude gibi asistanlarda ölçülmedi.
- Kayıtlı olmayan bir sektörde skor bir ekstrapolasyondur.
- Kullanıcı testi yapılmadı.
"""
    )


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="hero"><div class="kicker">SIC Capstone · Grup 8</div>'
        "<h1>Yapay zekâ asistanlarında marka görünürlüğü</h1>"
        "<p>Markanın asistan yanıtlarında görünüp görünmediğini canlı ölçer, nedenini teşhis eder "
        "ve kontrollü deneylerde ölçülmüş etkilere dayanan öneriler verir.</p>"
        '<div class="gates">'
        '<div class="gate"><b>1 · Listeye gir</b><span>Bağımsız kaynak, rakiplerle, üstte</span></div>'
        '<div class="gate"><b>2 · Seçil</b><span>Somut ürün bilgisi, rakipte olmayan</span></div>'
        '<div class="gate"><b>⚠ Engelle</b><span>Kaynaksız iddia önerilmez</span></div>'
        "</div></div>",
        unsafe_allow_html=True,
    )
    visibility, description, how = st.tabs(
        ["🔎  Görünürlük analizi", "📝  Açıklama denetimi", "📚  Nasıl çalışır?"]
    )
    with visibility:
        visibility_tool()
    with description:
        audit_tool()
    with how:
        about()


main()
