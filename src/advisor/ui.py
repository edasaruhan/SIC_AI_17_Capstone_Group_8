# pyright: reportMissingImports=false
"""Streamlit interface over the advisor graph.

The graph is the product; this page only drives it. It shows the call estimate and
asks for confirmation before anything is paid, streams the graph node by node so the
workflow is visible, lets the user correct the extracted rivals and rerun from the
cache, and renders the same report the CLI writes. Everything that costs money or
decides a result lives in ``service.py``, shared with the CLI.

    make advisor-ui
"""

from __future__ import annotations

import asyncio
import re

import streamlit as st

from advisor import audit, model, service
from advisor.advise import DIAGNOSES
from advisor.clients import require_keys

st.set_page_config(page_title="Marka Görünürlük Danışmanı", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Öğrenilmiş sinyal modeli yükleniyor…")
def load_model() -> dict:
    return model.load()


def split(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


def pct(value: float | None) -> str:
    return "–" if value is None else f"%{100 * value:.0f}"


def run_with_progress(options: service.Options, boosters: dict) -> dict | None:
    """Run the graph inside a status box that lists each node as it finishes."""
    with st.status("Akış çalışıyor…", expanded=True) as status:

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

    st.header(f"{options.brand} · {options.sector}")
    columns = st.columns(4)
    columns[0].metric("Teşhis", DIAGNOSES.get(diagnosis, diagnosis))
    columns[1].metric("Arama sonuçlarında görünme", pct(measures.get("retrieval_presence")))
    columns[2].metric("Arama açıkken anılma", pct(measures.get("mention_on")))
    if scores:
        columns[3].metric(
            "Rakiplere göre sıra (tahmin)",
            f"{scores.get('rank', '–')} / {scores.get('candidates', '–')}",
        )

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
                        dict.fromkeys([*options.drop_rivals, *(r for r in rivals if r not in keep)])
                    ),
                }
            )
            final = run_with_progress(corrected, boosters)
            if final is not None:
                st.session_state["result"] = {"options": corrected, "state": final}
                st.rerun()

    st.markdown(state.get("report", ""))
    st.download_button(
        "Raporu indir (.md)",
        data=state.get("report", ""),
        file_name=f"{options.brand}-{options.sector}-gorunurluk.md",
        mime="text/markdown",
    )


def show_audit() -> None:
    """The free tool: no call, no model, only the rules measured in the experiment."""
    st.header("Ürün açıklaması denetimi")
    st.caption(
        "Ücretsizdir, API çağrısı yapmaz. Kurallar iki asistanla (Gemini 3.5 Flash Lite, "
        "gpt-oss-120b) yapılan kontrollü açıklama deneyine dayanır."
    )
    text = st.text_area("Ürün açıklamanız", height=160)
    rivals_raw = st.text_area(
        "Rakip açıklamaları (isteğe bağlı; her birini boş bir satırla ayırın)", height=160
    )
    if st.button("Denetle", type="primary", disabled=not text.strip()):
        rivals = tuple(part.strip() for part in re.split(r"\n\s*\n", rivals_raw) if part.strip())
        st.markdown(audit.render(audit.audit(text, rivals)))


def main() -> None:
    st.title("Yapay zekâ asistanlarında marka görünürlüğü danışmanı")
    with st.sidebar:
        tool = st.radio("Araç", ["Görünürlük analizi", "Açıklama denetimi"])
    if tool == "Açıklama denetimi":
        show_audit()
        return
    st.caption(
        "Herhangi bir marka ve sektör. Canlı Google araması ve Gemini ile ölçer, araştırmada "
        "öğrenilen sinyalle rakiplerinize göre konumlar; öneriler kontrollü testte ölçülmüş "
        "etkilere dayanır."
    )

    try:
        boosters = load_model()
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"Öğrenilmiş sinyal modeli yüklenemedi: {exc}")
        st.code("make advisor-train")
        st.stop()
        return

    with st.sidebar:
        st.header("Analiz")
        brand = st.text_input("Marka", placeholder="ör. Garanti BBVA")
        sector = st.text_input("Sektör", placeholder="ör. bankacılık")
        language = st.radio("Dil", ["tr", "en"], horizontal=True)
        aliases = st.text_input("Markanın diğer yazılışları (virgülle)", placeholder="ör. Garanti")
        with st.expander("Ayarlar"):
            queries = st.slider("Sorgu sayısı", 1, 5, 3)
            reps = st.slider("Tekrar", 1, 3, 2)
            max_calls = st.number_input("En fazla ücretli çağrı", 5, 60, 20)

    if not (brand.strip() and sector.strip()):
        st.info("Başlamak için soldan bir marka ve sektör girin.")
        result = st.session_state.get("result")
        if result:
            show_result(result["options"], result["state"], boosters)
        return

    options = service.Options(
        brand=brand.strip(),
        sector=sector.strip(),
        language=language,
        brand_aliases=split(aliases),
        queries=int(queries),
        reps=int(reps),
        max_calls=int(max_calls),
    )
    calls = service.estimated_calls(options)
    cached = (service.run_folder(options) / "report.md").exists()
    st.markdown(
        f"**Tahmini ücretli çağrı:** {calls}  ·  **bütçe:** {int(max_calls)}"
        + ("  ·  bu analiz daha önce yapıldı; önbellekten okunur" if cached else "")
    )
    over_budget = calls > int(max_calls)
    if over_budget:
        st.warning(
            "Tahmin bütçenin üstünde; sorgu veya tekrar sayısını azaltın ya da bütçeyi artırın."
        )
    confirmed = st.checkbox("Ücretli çağrıları onaylıyorum")

    if st.button("Analizi başlat", type="primary", disabled=over_budget or not confirmed):
        try:
            require_keys()
        except ValueError as exc:
            st.error(f"{exc} — anahtarı `.env` dosyasına ekleyin.")
            st.stop()
            return
        final = run_with_progress(options, boosters)
        if final is not None:
            st.session_state["result"] = {"options": options, "state": final}

    result = st.session_state.get("result")
    if result:
        show_result(result["options"], result["state"], boosters)


main()
