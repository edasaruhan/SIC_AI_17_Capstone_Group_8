# pyright: reportMissingImports=false
"""Brand visibility report: evidence_v2 tables in, a sourced brand report out.

Offline: reads the frozen evidence tables and the saved analysis outputs, makes no
API call. Start with ``make app``; Streamlit runs through ``uv run --with`` so the
lock file, and with it every evidence manifest, stays untouched.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from visibility import brand_profile as bp

ROOT = Path("data/processed/evidence_v2")
EFFECTS = Path("reports/intervention/effects.csv")
STABILITY = Path("reports/generalization/stability_matrix_y_mention.csv")
PAIR_COLUMNS = [
    "record_id",
    "category",
    "condition",
    "brand",
    "y_mention",
    "y_top",
    "in_search_results",
    "n_results_mentioning",
    "best_position",
    *[f"n_{kind}" for kind in bp.SOURCE_KINDS],
]
EVIDENCE_COLUMNS = ["category", "record_id", "domain", "source_type", "brand", "title", "link"]
STABILITY_TR = {"strong": "Güçlü", "moderate": "Orta", "weak": "Zayıf", "conflicting": "Çelişkili"}
SIGNALS = {
    "in_search_results": "Arama sonuçlarında görünmek",
    "n_results_mentioning": "Birden çok sonuçta anılmak",
    "best_position": "Üst sırada anılmak",
    "lex_superlative": "Üstünlük dili",
}


@st.cache_data(show_spinner="Kanıt tabloları doğrulanıyor…")
def load(track: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    from evidence_eval.workspace import verify

    verify(ROOT)  # never show numbers from altered or stale tables
    pairs = pd.read_parquet(ROOT / f"pairs_{track}.parquet", columns=PAIR_COLUMNS)
    evidence = pd.read_parquet(ROOT / f"evidence_{track}.parquet", columns=EVIDENCE_COLUMNS)
    return pairs, evidence


@st.cache_data
def optional_csv(path: str) -> pd.DataFrame | None:
    return pd.read_csv(path) if Path(path).exists() else None


def pct(value: float) -> str:
    return f"%{100 * value:.0f}"


def main() -> None:
    st.set_page_config(page_title="AI Marka Görünürlüğü", page_icon="🔎", layout="wide")
    st.title("Yapay zekâ asistanlarında marka görünürlüğü")
    st.caption(
        "Kaynaklı marka raporu · evidence_v2 · kayıtlı asistan yanıtları üzerinde; "
        "bu sayfa canlı API çağrısı yapmaz."
    )
    with st.sidebar:
        track = st.radio(
            "Veri seti",
            ["tr", "en"],
            format_func=lambda t: {"tr": "Türkçe (300 yanıt)", "en": "İngilizce (9.586 yanıt)"}[t],
        )
        sector = st.selectbox(
            "Sektör", bp.TRACK_SECTORS[track], format_func=lambda s: bp.SECTOR_TR[s]
        )
    try:
        pairs, evidence = load(track)
    except (ValueError, FileNotFoundError) as exc:
        st.error(f"Kanıt tabloları doğrulanamadı: {exc}. `make evidence-v2-prepare` çalıştırın.")
        st.stop()
        return
    board = bp.leaderboard(pairs, sector)
    with st.sidebar:
        brand = st.selectbox("Marka", list(board.index), index=min(len(board) - 1, 6))
    effects = optional_csv(str(EFFECTS))
    stability = optional_csv(str(STABILITY))

    row = board.loc[brand]
    rivals = bp.winners(board, brand)
    columns = st.columns(4)
    columns[0].metric(
        "Arama açıkken anılma",
        pct(row["mention_on"]),
        delta=f"{100 * (row['mention_on'] - board['mention_on'].mean()):+.0f} puan (sektör ort.)",
    )
    columns[1].metric(
        "Arama kapalıyken anılma",
        pct(row["mention_off"]),
        delta=f"{100 * (row['mention_on'] - row['mention_off']):+.0f} puan arama etkisi",
        delta_color="off",
    )
    columns[2].metric("Birincil öneri payı", pct(row["top_on"]))
    columns[3].metric(
        "Sektör sırası",
        f"{int(row['rank_on'])} / {len(board)}",
        help=f"Lider: {board.index[0]} ({pct(board['mention_on'].iloc[0])})",
    )

    profile = bp.retrieval_profile(pairs, sector)
    gap = bp.gaps(profile, brand, rivals)
    targets = bp.outreach_targets(evidence, sector, brand, rivals)

    st.header("Ne yapmalı?")
    st.caption(
        "Yalnız sektörler arasında tutarlı çıkan sinyallere dayanır; her önerinin yanında "
        "kontrollü testteki ölçülen etkisi yazar."
    )
    for rec in bp.recommendations(gap, targets, effects):
        with st.container(border=True):
            st.subheader(rec["title"])
            st.write(rec["why"])
            st.write(rec["action"])
            test = rec["test"]
            if test is None:
                st.info(rec["verdict"])
            elif test["effect_lo"] > 0:
                st.success(rec["verdict"])
            elif test["effect_hi"] < 0:
                st.error(rec["verdict"])
            else:
                st.warning(rec["verdict"])
            if rec["targets"]:
                table = pd.DataFrame(rec["targets"])
                table["source_type"] = table["source_type"].replace(bp.KIND_TR)
                st.dataframe(
                    table[["domain", "source_type", "rivals", "answers", "link"]],
                    column_config={
                        "domain": "Alan adı",
                        "source_type": "Kaynak tipi",
                        "rivals": "Andığı rakipler",
                        "answers": st.column_config.NumberColumn("Yanıt sayısı"),
                        "link": st.column_config.LinkColumn("Örnek sayfa"),
                    },
                    hide_index=True,
                    width="stretch",
                )

    st.header("Neden bu durumdasın?")
    left, right = st.columns([3, 2])
    with left:
        shown = gap.assign(
            brand=lambda f: f["brand"].where(f["metric"] == "results_per_answer", 100 * f["brand"]),
            winners=lambda f: f["winners"].where(
                f["metric"] == "results_per_answer", 100 * f["winners"]
            ),
        )
        st.dataframe(
            shown[["label", "brand", "winners"]],
            column_config={
                "label": "Ölçü (arama açık yanıtlar)",
                "brand": st.column_config.NumberColumn(brand, format="%.1f"),
                "winners": st.column_config.NumberColumn("En çok anılan 5 rakip", format="%.1f"),
            },
            hide_index=True,
            width="stretch",
        )
        st.caption("Oranlar yüzde; sonuç sayısı yanıt başına ortalama.")
    with right:
        kinds = [k for k in bp.SOURCE_KINDS if k != "unknown"]
        mix = pd.DataFrame(
            {
                brand: [
                    profile.at[brand, f"share_{k}"] if brand in profile.index else 0.0
                    for k in kinds
                ],
                "Rakipler": [profile.reindex(rivals)[f"share_{k}"].mean() for k in kinds],
            },
            index=pd.Index([bp.KIND_TR[k] for k in kinds]),
        )
        st.bar_chart(100 * mix, stack=False, y_label="Kaynak payı (%)")

    st.header("Kanıt: seni anan kaynaklar")
    sources = bp.brand_sources(evidence, sector, brand)
    if sources.empty:
        st.write("Kayıtlı arama sonuçlarında bu markayı anan kaynak yok.")
    else:
        sources = sources.assign(source_type=sources["source_type"].replace(bp.KIND_TR))
        st.dataframe(
            sources,
            column_config={
                "domain": "Alan adı",
                "source_type": "Kaynak tipi",
                "answers": "Yanıt sayısı",
                "title": "Örnek başlık",
                "link": st.column_config.LinkColumn("Örnek sayfa"),
            },
            hide_index=True,
            width="stretch",
        )

    with st.expander("Sektör sıralaması"):
        st.dataframe(
            (100 * board[["mention_off", "mention_on", "top_on"]]).round(1),
            column_config={
                "mention_off": "Aramasız anılma %",
                "mention_on": "Aramalı anılma %",
                "top_on": "Birincil öneri %",
            },
            width="stretch",
        )

    with st.expander("Yöntem ve sınırlılıklar"):
        st.markdown(
            "- Görünürlük, kayıtlı asistan yanıtlarından gözlemseldir; nedensel değildir.\n"
            "- Öneriler yalnız alan dışı (LODO) testte taşınan ve sektörler arasında aynı yönde "
            "çıkan sinyallere dayanır: `reports/generalization/`.\n"
            "- Test edilmiş etkiler, tek bir asistanla (Gemini 3.5 Flash Lite) kayıtlı arama bağlamına "
            "eklenen sentetik sayfalarla ölçülmüştür: `reports/intervention/`. Garanti değildir.\n"
            "- Türkçe veri 300 yanıttır; Türkçe sektör sonuçlarını yön gösterici okuyun."
        )
        if stability is not None:
            table = stability[stability["feature"].isin(SIGNALS)].assign(
                feature=lambda f: f["feature"].map(SIGNALS),
                stability=lambda f: f["stability"].map(STABILITY_TR),
                stability_matched=lambda f: f["stability_matched"].map(STABILITY_TR),
            )
            st.dataframe(
                table[["feature", "stability", "stability_matched"]],
                column_config={
                    "feature": "Sinyal",
                    "stability": "Sektörler arası kararlılık",
                    "stability_matched": "Hacim eşlenince",
                },
                hide_index=True,
            )


main()
