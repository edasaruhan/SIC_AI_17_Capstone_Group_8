"""Masking ablation across every sector the corpora cover.

M3 is a surrogate: it predicts which brand the assistant picked. Replacing every
brand name in the snippets with ``[BRAND]`` and retraining tells us how much of
*our model's* predictability rests on the name rather than on what the pages say.
Until now that was measured on VPN alone, so the finding could have been a
property of one sector.

This script pairs the named and masked runs of each sector on the same rows and
reports the difference with one shared query-cluster bootstrap -- the interval is
for the difference itself, not two overlapping marginal intervals. It reads the
parquet files ``scripts/run_m3.py`` writes; it trains nothing and calls nothing.

What the numbers do **not** say: masking deletes the name string, not the pattern
of which pages a brand appears on, and that pattern stays correlated with identity.
A delta is therefore not a causal share of "brand vs content", and it describes our
surrogate, not the assistant's own decision process.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")

from evidence_eval.io import write_json  # noqa: E402
from evidence_eval.metrics import summarise  # noqa: E402
from visibility import generalization as g  # noqa: E402

PAIRS = Path("data/processed/modeling")
TARGET = "y_top"
VARIANTS = ("named", "masked")
SECTOR_TR = {
    "vpn": "VPN",
    "hosting": "Hosting / bulut",
    "editors": "Kod editörleri",
    "travel": "Seyahat",
    "cosmetics": "Kozmetik",
}


def available(pairs: Path = PAIRS) -> list[tuple[str, str]]:
    """(track, category) pairs whose named and masked runs are both on disk."""
    found = []
    for path in sorted(pairs.glob(f"m3_*_{TARGET}_named.parquet")):
        stem = path.name[len("m3_") : -len(f"_{TARGET}_named.parquet")]
        track, _, category = stem.partition("_")
        if (pairs / f"m3_{track}_{category}_{TARGET}_masked.parquet").exists():
            found.append((track, category))
    return found


def paired(track: str, category: str, pairs: Path = PAIRS) -> pd.DataFrame:
    """Named and masked scores side by side on the same (response, brand) rows."""
    keep = ["record_id", "brand", "query_id", TARGET, "score_M3"]
    frames = {
        variant: pd.read_parquet(
            pairs / f"m3_{track}_{category}_{TARGET}_{variant}.parquet", columns=keep
        ).rename(columns={"score_M3": f"score_{variant}"})
        for variant in VARIANTS
    }
    merged = frames["named"].merge(
        frames["masked"].drop(columns=["query_id", TARGET]),
        on=["record_id", "brand"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(frames["named"]):
        raise ValueError(f"{track}/{category}: named ve masked satırları eşleşmedi")
    return merged


def rows(track: str, category: str) -> dict:
    frame = paired(track, category)
    delta = g.paired_delta(frame, "score_named", "score_masked", TARGET)
    row = {"track": track, "category": category, "n_pairs": len(frame)}
    for variant in VARIANTS:
        metrics = summarise(frame, f"score_{variant}", TARGET)
        for name in ("pr_auc", "top1", "ndcg@3"):
            row[f"{variant}_{name}"] = metrics[name]
    return {**row, **delta}


def fmt(value: float, digits: int = 3) -> str:
    return "–" if pd.isna(value) else f"{value:.{digits}f}"


def render(table: pd.DataFrame) -> str:
    lines = [
        "# Maskeleme ablasyonu: tahmin ne kadar marka adına dayanıyor?",
        "",
        "Bu sayfa `scripts/collect_masking.py` tarafından üretilir; elle düzenlemeyin.",
        "Yöntem ve sınırlılıklar betiğin başındadır.",
        "",
        "Snippet'lerdeki her marka adı `[BRAND]` ile değiştirilip M3 yeniden eğitildi.",
        "Fark, aynı satırlar üzerinde eşleştirilmiş ve tek bir sorgu-küme bootstrap'ı ile",
        "hesaplandı. Hedef: birincil öneri (`y_top`), seed 7, iki epoch.",
        "",
        "| Track | Sektör | Çift | İsimli PR-AUC | Maskeli PR-AUC | ΔPR-AUC [95% GA] | ΔTop-1 [95% GA] | Sorgu |",
        "|---|---|---:|---:|---:|---|---|---:|",
    ]
    for _, r in table.sort_values(["track", "category"]).iterrows():
        lines.append(
            f"| {r.track} | {SECTOR_TR.get(r.category, r.category)} | {int(r.n_pairs)} | "
            f"{fmt(float(r.named_pr_auc))} | {fmt(float(r.masked_pr_auc))} | "
            f"{fmt(float(r.delta_pr_auc))} "
            f"[{fmt(float(r.delta_pr_auc_lo))}, {fmt(float(r.delta_pr_auc_hi))}] | "
            f"{fmt(float(r['delta_top1']))} "
            f"[{fmt(float(r['delta_top1_lo']))}, {fmt(float(r['delta_top1_hi']))}] | "
            f"{int(r.independent_query_groups)} |"
        )
    lines += [
        "",
        "Pozitif ΔPR-AUC, marka adı silindiğinde tahminin **kötüleştiğini** gösterir:",
        "model o sektörde kimin kazandığını kısmen isimden biliyordu. Sıfıra yakın veya",
        "negatif bir fark, o sektörde tahminin isimden çok sayfa içeriğine dayandığı",
        "anlamına gelir.",
        "",
        "Güven aralığı sıfırı içeren satırlarda yön çağrılmaz.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=PAIRS)
    parser.add_argument("--out", type=Path, default=Path("reports/masking"))
    args = parser.parse_args()
    combos = available(args.pairs)
    if not combos:
        raise SystemExit(
            f"{args.pairs} altında eşleşmiş named/masked koşusu yok; önce scripts/run_m3.py"
        )
    table = pd.DataFrame([rows(track, category) for track, category in combos])
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "ablation.csv", index=False)
    (args.out / "results.md").write_text(render(table), encoding="utf-8")
    write_json(
        args.out / "run.json",
        {
            "target": TARGET,
            "sectors": [f"{track}/{category}" for track, category in combos],
            "bootstrap": "query clusters, shared across both variants (generalization.paired_delta)",
            "surrogate_warning": "measures our predictor, not the assistant's decision process",
        },
    )
    print(f"wrote {args.out}/results.md")
    print(table[["track", "category", "named_pr_auc", "masked_pr_auc", "delta_pr_auc"]].round(3))


if __name__ == "__main__":
    main()
