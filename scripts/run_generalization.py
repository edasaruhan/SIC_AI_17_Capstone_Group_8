"""M2-General, leave-one-domain-out transfer and the signal stability matrix.

Reads the frozen evidence_v1 tables only; no API, GPU or network. Writes CSVs, a
run manifest and a generated results page under ``reports/generalization/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, "src")

import pandas as pd  # noqa: E402

from evidence_eval.baselines import ESTIMATOR  # noqa: E402
from evidence_eval.io import read_json, write_json  # noqa: E402
from evidence_eval.metrics import summarise  # noqa: E402
from evidence_eval.workspace import verify  # noqa: E402
from modeling.features import select  # noqa: E402
from visibility import generalization as g  # noqa: E402

ARROW = {"up": "↑", "down": "↓", "flat": "→", "n/a": "·"}
STABILITY_TR = {
    "strong": "Güçlü",
    "moderate": "Orta",
    "weak": "Zayıf",
    "conflicting": "Çelişkili",
}
GAPS = {
    "transfer_gap": ("score_general_seen", "score_general_unseen"),
    "prior_dependence": ("score_M2_full", "score_general_seen"),
    "unseen_vs_position": ("score_general_unseen", "score_naive_position"),
    "invariant_transfer_gap": ("score_invariant_seen", "score_invariant_unseen"),
    "invariant_vs_general_unseen": ("score_invariant_unseen", "score_general_unseen"),
}
TR_GAPS = {
    "en_transfer_vs_tr_trained": ("score_general_en_transfer", "score_general_seen"),
    "en_transfer_vs_position": ("score_general_en_transfer", "score_naive_position"),
    "invariant_en_transfer_vs_position": ("score_invariant_en_transfer", "score_naive_position"),
    "invariant_vs_general_en_transfer": (
        "score_invariant_en_transfer",
        "score_general_en_transfer",
    ),
}


def attach_saved(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Add the production M2 (with priors) out-of-fold score from the v1 baselines."""
    saved = pd.read_parquet(path, columns=["record_id", "brand", "score_M2"])
    merged = frame.merge(saved, on=["record_id", "brand"], how="left", validate="one_to_one")
    if bool(merged["score_M2"].isna().any()):
        raise SystemExit(f"{path} does not cover the evaluation panel; rerun modeling-baselines")
    return merged.rename(columns={"score_M2": "score_M2_full"})


def metric_rows(frame: pd.DataFrame, track: str, target: str, gaps: dict) -> tuple[list, list]:
    scores = [c for c in frame.columns if c.startswith("score_")]
    metrics, deltas = [], []
    for domain in sorted(frame["category"].unique()):
        part = select(frame, frame["category"] == domain).reset_index(drop=True)
        for score in scores:
            row = summarise(part, score, target)
            metrics.append(
                {
                    "track": track,
                    "domain": domain,
                    "target": target,
                    "model": score.removeprefix("score_"),
                    **row,
                }
            )
        for name, (a, b) in gaps.items():
            deltas.append(
                {
                    "track": track,
                    "domain": domain,
                    "target": target,
                    "comparison": name,
                    "a": a.removeprefix("score_"),
                    "b": b.removeprefix("score_"),
                    **g.paired_delta(part, a, b, target),
                }
            )
        print(f"  scored {track}/{domain}/{target}", flush=True)
    return metrics, deltas


def signal_rows(frame: pd.DataFrame, track: str, target: str, *, with_model: bool) -> list:
    rows = []
    for domain in sorted(frame["category"].unique()):
        part = select(frame, frame["category"] == domain).reset_index(drop=True)
        shap = g.shap_profile(part, target, with_model=with_model).set_index("feature")
        for feature in g.MATRIX_FEATURES:
            effect = g.signal_effect(part, feature, target)
            # Presence and volume are the matching variable itself; they keep their effect.
            matched = (
                g.signal_effect(part, feature, target, matched_on=g.VOLUME)
                if feature in g.MATCHABLE_FEATURES
                else effect
            )
            rows.append(
                {
                    "domain": f"{track}/{domain}",
                    "target": target,
                    **effect,
                    **{
                        f"matched_{k}": matched[k]
                        for k in ("effect", "effect_lo", "effect_hi", "direction", "n_comparisons")
                    },
                    "shap_share": shap["shap_share"].get(feature, float("nan")),
                    "shap_direction": shap["shap_direction"].get(feature, float("nan")),
                }
            )
    return rows


def stability_tables(effects: pd.DataFrame, perms: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables = {}
    for target, part in effects.groupby("target"):
        wide = part.pivot(index="feature", columns="domain", values="direction")
        wide = wide.reindex(g.MATRIX_FEATURES)
        summary = []
        for feature in wide.index:
            directions = [str(d) for d in wide.loc[feature]]
            rows = select(part, part["feature"] == feature)
            called = select(rows, rows["direction"] != "n/a")
            drop = select(
                perms,
                (perms["target"] == target)
                & (perms["model"] == "general")
                & (perms["feature"] == feature),
            )
            summary.append(
                {
                    "feature": feature,
                    **{domain: ARROW[str(d)] for domain, d in wide.loc[feature].items()},
                    "up": directions.count("up"),
                    "down": directions.count("down"),
                    "flat": directions.count("flat"),
                    "mean_effect": called["effect"].mean(),
                    "median_shap_share": rows["shap_share"].median(),
                    "shap_sign_agrees": (
                        (called["effect"] * called["shap_direction"] > 0).mean()
                        if len(called)
                        else float("nan")
                    ),
                    "lodo_pr_auc_drop": drop["pr_auc_drop"].mean() if len(drop) else float("nan"),
                    "stability": g.classify(directions),
                    "mean_matched_effect": select(rows, rows["matched_direction"] != "n/a")[
                        "matched_effect"
                    ].mean(),
                    "stability_matched": g.classify([str(d) for d in rows["matched_direction"]]),
                }
            )
        tables[str(target)] = pd.DataFrame(summary)
    return tables


def fmt(value: Any, digits: int = 3) -> str:
    return "–" if pd.isna(value) else f"{value:.{digits}f}"


def write_markdown(
    out: Path, metrics: pd.DataFrame, deltas: pd.DataFrame, tables: dict[str, pd.DataFrame]
) -> None:
    lines = [
        "# M2-General · alan dışı genelleme ve sinyal kararlılığı (üretilmiş tablolar)",
        "",
        "Bu sayfa `scripts/run_generalization.py` tarafından üretilir; elle düzenlemeyin.",
        "Yorum ve sınırlılıklar için `README.md`.",
        "",
    ]
    for target in g.TARGETS:
        lines += [
            f"## Performans · `{target}`",
            "",
            "| Track | Domain | Model | Pozitif oran | PR-AUC [95% GA] | Top-1 [95% GA] | NDCG@3 |",
            "|---|---|---|---:|---|---|---:|",
        ]
        for _, r in metrics[metrics["target"] == target].iterrows():
            # Top-1 is undefined for multi-winner mention answers; its bootstrap is noise.
            top1 = (
                f"{fmt(r.top1)} [{fmt(r.top1_lo)}, {fmt(r.top1_hi)}]" if pd.notna(r.top1) else "–"
            )
            lines.append(
                f"| {r.track} | {r.domain} | {r.model} | {fmt(r.positive_rate)} | "
                f"{fmt(r.pr_auc)} [{fmt(r.pr_auc_lo)}, {fmt(r.pr_auc_hi)}] | "
                f"{top1} | {fmt(r['ndcg@3'])} |"
            )
        lines += [
            "",
            f"### Eşleştirilmiş farklar · `{target}` (a − b, sorgu-küme bootstrap)",
            "",
            "| Track | Domain | Karşılaştırma | ΔPR-AUC [95% GA] | ΔTop-1 [95% GA] | ΔNDCG@3 [95% GA] |",
            "|---|---|---|---|---|---|",
        ]
        for _, r in deltas[deltas["target"] == target].iterrows():
            lines.append(
                f"| {r.track} | {r.domain} | {r.comparison} | "
                f"{fmt(r.delta_pr_auc)} [{fmt(r.delta_pr_auc_lo)}, {fmt(r.delta_pr_auc_hi)}] | "
                f"{fmt(r.delta_top1)} [{fmt(r.delta_top1_lo)}, {fmt(r.delta_top1_hi)}] | "
                f"{fmt(r['delta_ndcg@3'])} [{fmt(r['delta_ndcg@3_lo'])}, {fmt(r['delta_ndcg@3_hi'])}] |"
            )
        lines.append("")
    for target, table in tables.items():
        domains = [c for c in table.columns if "/" in c]
        lines += [
            f"## Sinyal kararlılık matrisi · `{target}`",
            "",
            "↑/↓: kazanan–kaybeden farkı 95% GA ile sıfırdan ayrık; →: ayrık değil; "
            f"·: < {g.MIN_QUERY_GROUPS} bağımsız sorgu, yön çağrılmadı.",
            "",
            "| Sinyal | "
            + " | ".join(domains)
            + " | Ort. etki | SHAP payı (medyan) | SHAP yön uyumu | LODO ΔPR-AUC | Kararlılık"
            + " | Hacim-eşli ort. etki | Hacim-eşli kararlılık |",
            "|---|" + "---|" * len(domains) + "---:|---:|---:|---:|---|---:|---|",
        ]
        for _, r in table.iterrows():
            lines.append(
                f"| `{r.feature}` | "
                + " | ".join(str(r[d]) for d in domains)
                + f" | {fmt(r.mean_effect)} | {fmt(r.median_shap_share)} | "
                f"{fmt(r.shap_sign_agrees, 2)} | {fmt(r.lodo_pr_auc_drop, 4)} | "
                f"{STABILITY_TR[r.stability]} | {fmt(r.mean_matched_effect)} | "
                f"{STABILITY_TR[r.stability_matched]} |"
            )
        lines.append("")
    (out / "results.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/processed/evidence_v2"))
    parser.add_argument("--out", type=Path, default=Path("reports/generalization"))
    args = parser.parse_args()
    manifest = verify(args.root)
    args.out.mkdir(parents=True, exist_ok=True)

    en_pairs = pd.read_parquet(args.root / "pairs_en.parquet")
    tr_pairs = pd.read_parquet(args.root / "pairs_tr.parquet")
    folds_en = read_json(args.root / "folds_en.json")
    folds_tr = read_json(args.root / "folds_tr.json")
    results_per_response = {
        track: pd.read_parquet(args.root / f"sources_{track}.parquet", columns=["record_id"])[
            "record_id"
        ].value_counts()
        for track in ("en", "tr")
    }
    invariant = g.INVARIANT_FEATURES

    metrics, deltas, perms, effects = [], [], [], []
    for target in g.TARGETS:
        en = g.add_relative(g.panel(en_pairs, target), results_per_response["en"])
        en["score_naive_position"] = en["in_search_results"] / en["best_position"]
        en["score_general_seen"] = g.seen_domain_scores(en, folds_en, target)
        en["score_general_unseen"], importances = g.unseen_domain_scores(en, target)
        perms += [{"target": target, "model": "general", **row} for row in importances]
        en["score_invariant_seen"] = g.seen_domain_scores(en, folds_en, target, features=invariant)
        en["score_invariant_unseen"], importances = g.unseen_domain_scores(
            en, target, features=invariant
        )
        perms += [{"target": target, "model": "invariant", **row} for row in importances]
        en = attach_saved(en, args.root / "baselines" / f"scores_en_{target}.parquet")
        g.tie_break(en, [c for c in en.columns if c.startswith("score_")])
        rows, gaps = metric_rows(en, "en", target, GAPS)
        metrics += rows
        deltas += gaps

        # Cross-lingual: the generator models differ between corpora, so model_id is
        # dropped on both TR columns to keep them comparable.
        tr = g.add_relative(g.panel(tr_pairs, target), results_per_response["tr"])
        tr["score_naive_position"] = tr["in_search_results"] / tr["best_position"]
        tr["score_general_seen"] = g.seen_domain_scores(tr, folds_tr, target, with_model=False)
        _, _, tr["score_general_en_transfer"] = g.fit_score(en, tr, target, with_model=False)
        _, _, tr["score_invariant_en_transfer"] = g.fit_score(
            en, tr, target, with_model=False, features=invariant
        )
        tr = attach_saved(tr, args.root / "baselines" / f"scores_tr_{target}.parquet")
        g.tie_break(tr, [c for c in tr.columns if c.startswith("score_")])
        rows, gaps = metric_rows(tr, "tr", target, TR_GAPS)
        metrics += rows
        deltas += gaps

        effects += signal_rows(en, "en", target, with_model=True)
        effects += signal_rows(tr, "tr", target, with_model=True)

    metrics_df, deltas_df = pd.DataFrame(metrics), pd.DataFrame(deltas)
    perms_df, effects_df = pd.DataFrame(perms), pd.DataFrame(effects)
    tables = stability_tables(effects_df, perms_df)
    metrics_df.to_csv(args.out / "performance.csv", index=False)
    deltas_df.to_csv(args.out / "paired_deltas.csv", index=False)
    perms_df.to_csv(args.out / "lodo_permutation.csv", index=False)
    effects_df.to_csv(args.out / "signal_effects.csv", index=False)
    for target, table in tables.items():
        table.to_csv(args.out / f"stability_matrix_{target}.csv", index=False)
    write_markdown(args.out, metrics_df, deltas_df, tables)
    write_json(
        args.out / "run.json",
        {
            "experiment": manifest["identity"],
            "estimator": ESTIMATOR,
            "features": g.GENERAL_FEATURES + ["model_id"],
            "invariant_features": g.INVARIANT_FEATURES + ["model_id"],
            "excluded_features": g.EXCLUDED_FEATURES,
            "panel": "condition == search_on, confidence != low; y_top: decided responses",
            "min_query_groups_for_direction": g.MIN_QUERY_GROUPS,
            "tie_noise": g.TIE_NOISE,
            "stability_rule": "conflicting if any opposite significant directions; strong if "
            ">=4 domains agree and >=2/3 of called domains; moderate if >=2; else weak",
        },
    )
    print(f"\nwrote {args.out}/results.md")


if __name__ == "__main__":
    main()
