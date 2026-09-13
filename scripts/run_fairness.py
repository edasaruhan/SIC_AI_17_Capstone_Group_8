"""Concentration and fairness tables from the frozen evidence_v2 workspace.

Reads only; makes no API call and trains nothing. The measures and the rules
behind them are documented in ``src/visibility/fairness.py``; this script is the
I/O layer that writes the CSVs and the Turkish results page.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, cast

import pandas as pd

sys.path.insert(0, "src")

from evidence_eval.io import sha256, write_json  # noqa: E402
from evidence_eval.workspace import verify  # noqa: E402
from visibility import fairness as f  # noqa: E402

TRACKS = ("en", "tr")
CONDITION_TR = {"search_off": "arama kapalı", "search_on": "arama açık"}
TARGET_TR = {"y_mention": "anılma", "y_top": "birincil öneri"}
SECTOR_TR = {
    "vpn": "VPN",
    "hosting": "Hosting / bulut",
    "editors": "Kod editörleri",
    "travel": "Seyahat",
    "cosmetics": "Kozmetik",
}
ORIGIN_TR = {"tr": "Türkiye menşeli", "global": "Küresel"}


def fmt(value: Any, digits: int = 2) -> str:
    return "–" if pd.isna(value) else f"{value:.{digits}f}"


def interval(row: pd.Series, name: str, digits: int = 2) -> str:
    if bool(pd.isna(row[name])):
        return "–"
    low, high = float(row[f"{name}_lo"]), float(row[f"{name}_hi"])
    return f"{fmt(float(row[name]), digits)} [{fmt(low, digits)}, {fmt(high, digits)}]"


def points(value: float) -> str:
    return "–" if pd.isna(value) else f"{100 * float(value):+.1f}"


def gain_interval(row: pd.Series) -> str:
    if bool(pd.isna(row["retrieval_gain_lo"])):
        return "–"
    return f"[{points(float(row['retrieval_gain_lo']))}, {points(float(row['retrieval_gain_hi']))}]"


def concentration_tables(pairs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for track in TRACKS:
        for target in f.TARGETS:
            table = f.concentration(pairs[track], target=target)
            rows.append(table.assign(track=track))
    return pd.concat(rows, ignore_index=True)


def retrieval_effect(table: pd.DataFrame) -> pd.DataFrame:
    """Search-on minus search-off effective brand count, per (track, category, model)."""
    index = ["track", "category", "model_id", "target"]
    wide = table.pivot_table(index=index, columns="condition", values="n_eff")
    if "search_on" not in wide or "search_off" not in wide:
        return pd.DataFrame()
    wide = wide.assign(delta_n_eff=wide["search_on"] - wide["search_off"])
    return wide.reset_index().sort_values(index)


def recognition_tables(pairs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for track in TRACKS:
        for category in sorted(pairs[track]["category"].unique()):
            rows.append(f.recognition_terciles(pairs[track], str(category)).assign(track=track))
    return pd.concat(rows, ignore_index=True)


def origin_tables(pairs: dict[str, pd.DataFrame], labels: dict) -> pd.DataFrame:
    rows = []
    for track in TRACKS:
        for category in sorted(pairs[track]["category"].unique()):
            table = f.origin_groups(pairs[track], str(category), labels)
            if not table.empty:
                rows.append(table.assign(track=track))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def write_markdown(
    out: Path,
    concentration: pd.DataFrame,
    effect: pd.DataFrame,
    recognition: pd.DataFrame,
    origin: pd.DataFrame,
) -> None:
    lines = [
        "# Yoğunlaşma ve adalet (üretilmiş tablolar)",
        "",
        "Bu sayfa `scripts/run_fairness.py` tarafından üretilir; elle düzenlemeyin.",
        "Ölçülerin tanımı ve önceden sabitlenen kurallar `src/visibility/fairness.py`",
        "başındadır; yorum ve sınırlılıklar `README.md` içindedir.",
        "",
        "`N_eff = 1/HHI`: asistan, sektördeki *kaç* marka varmış gibi davranıyor.",
        "Registry'de VPN 24, hosting 36, editör 30, seyahat 27, kozmetik 74 marka var;",
        "N_eff bu sayıyla karşılaştırılarak okunur.",
        "",
        "## Yoğunlaşma · tüm asistanlar birlikte",
        "",
        "| Track | Sektör | Koşul | Ölçü | N_eff [95% GA] | İlk-3 payı [95% GA] | Hiç anılmayan | Sorgu |",
        "|---|---|---|---|---|---|---:|---:|",
    ]
    pooled = cast("pd.DataFrame", concentration[concentration["model_id"] == "ALL"])
    for _, r in pooled.sort_values(["track", "category", "target", "condition"]).iterrows():
        lines.append(
            f"| {r.track} | {SECTOR_TR.get(r.category, r.category)} | "
            f"{CONDITION_TR.get(r.condition, r.condition)} | {TARGET_TR[r.target]} | "
            f"{interval(r, 'n_eff')} | {interval(r, 'top3_share')} | "
            f"{int(r.n_never_named)}/{int(r.n_brands)} | {int(r.independent_query_groups)} |"
        )

    lines += [
        "",
        "## Arama açmak pazarı açıyor mu? (N_eff farkı, arama açık − kapalı)",
        "",
        "| Track | Sektör | Asistan | Ölçü | N_eff kapalı | N_eff açık | Fark |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for _, r in effect.iterrows():
        lines.append(
            f"| {r.track} | {SECTOR_TR.get(r.category, r.category)} | {r.model_id} | "
            f"{TARGET_TR[r.target]} | {fmt(r.search_off)} | {fmt(r.search_on)} | "
            f"{fmt(r.delta_n_eff)} |"
        )

    lines += [
        "",
        "## Aramadan kim kazanıyor? (tanınırlık tercilleri)",
        "",
        "Terciller arama-kapalı anılma oranına göre bir kez atanır; bootstrap yalnız",
        "sabit tercil içindeki kazancı değiştirir. Kazanç puan cinsindendir.",
        "",
        "| Track | Sektör | Tanınırlık | Marka | Aramasız % | Aramalı % | Kazanç [95% GA] |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for _, r in recognition.iterrows():
        lines.append(
            f"| {r.track} | {SECTOR_TR.get(r.category, r.category)} | {r.group} | "
            f"{int(r.n_brands)} | {100 * r.mention_off:.1f} | {100 * r.mention_on:.1f} | "
            f"{points(r.retrieval_gain)} {gain_interval(r)} |"
        )

    lines += ["", "## Yerli marka farkı", ""]
    if origin.empty:
        lines.append("Türkiye menşeli marka içeren sektör bulunamadı; tablo üretilmedi.")
    else:
        lines += [
            "Yalnız Türkiye menşeli markası bulunan sektörler. `unclear` etiketli markalar",
            "analizden çıkarılmıştır.",
            "",
            "| Track | Sektör | Menşe | Marka | Aramasız % | Aramalı % | Kazanç [95% GA] |",
            "|---|---|---|---:|---:|---:|---|",
        ]
        for _, r in origin.iterrows():
            lines.append(
                f"| {r.track} | {SECTOR_TR.get(r.category, r.category)} | "
                f"{ORIGIN_TR.get(r.group, r.group)} | {int(r.n_brands)} | "
                f"{100 * r.mention_off:.1f} | {100 * r.mention_on:.1f} | "
                f"{points(r.retrieval_gain)} {gain_interval(r)} |"
            )
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/processed/evidence_v2"))
    parser.add_argument("--out", type=Path, default=Path("reports/fairness"))
    args = parser.parse_args()
    manifest = verify(args.root)
    args.out.mkdir(parents=True, exist_ok=True)

    pairs = {track: pd.read_parquet(args.root / f"pairs_{track}.parquet") for track in TRACKS}
    labels = f.load_origin()

    concentration = concentration_tables(pairs)
    effect = retrieval_effect(concentration)
    recognition = recognition_tables(pairs)
    origin = origin_tables(pairs, labels)

    concentration.to_csv(args.out / "concentration.csv", index=False)
    effect.to_csv(args.out / "retrieval_effect.csv", index=False)
    recognition.to_csv(args.out / "recognition_terciles.csv", index=False)
    if not origin.empty:
        origin.to_csv(args.out / "origin_gap.csv", index=False)
    write_markdown(args.out, concentration, effect, recognition, origin)
    write_json(
        args.out / "run.json",
        {
            "experiment": manifest["identity"],
            "origin_labels": str(f.ORIGIN_PATH),
            "origin_sha256": sha256(f.ORIGIN_PATH),
            "origin_rule": labels["rule"],
            "measures": "n_eff = 1/HHI; top-3 share; normalised HHI; never-named count",
            "shares_over": "naming events, not responses; y_top over decided responses",
            "recognition_proxy": "retrieval-off mention rate (not company size)",
            "bootstrap": f"query clusters, seed {f.SEED}, {f.N_RESAMPLES} resamples, 2.5/97.5",
            "min_query_groups_for_direction": f.MIN_QUERY_GROUPS,
        },
    )
    print(f"wrote {args.out}/results.md")


if __name__ == "__main__":
    main()
