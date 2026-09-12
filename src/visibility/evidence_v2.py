"""evidence_v2: the v1 tables rebuilt with a completed source taxonomy.

v1 shipped empty editorial and affiliate domain lists, so about 88% of retrieved
results were ``unknown`` and the affiliate hypothesis could not be tested. v2
changes only the source rules. Responses, candidates, labels, folds and features
are built by the unchanged v1 code, so any v1/v2 difference is the taxonomy.

v1 stays verifiable: its manifest hashes the files that existed when it was built,
and this module, the v2 rules file and the v2 root are all new.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from evidence_eval import workspace
from evidence_eval.evidence import host, retrieved_sources
from evidence_eval.io import digest, read_json, sha256, write_json
from modeling import datasets, splits
from modeling.brands import load_registry

VERSION = "evidence_v2"
MODULE = Path("src/visibility/evidence_v2.py")
RULES = Path("configs/modeling/source_domains_v2.json")
DEFAULT_ROOT = Path("data/processed/evidence_v2")
CATEGORIES = ("vpn", "hosting", "editors", "travel", "cosmetics")
LISTED_KINDS = ("retailer", "forum", "affiliate", "editorial")


def _matches(hostname: str, domain: str) -> bool:
    return hostname == domain or hostname.endswith("." + domain)


def source_type(link: str, brand: str | None, rules: dict) -> str:
    """Classify one result for one brand; the longest matching domain wins.

    Longest-match keeps ``aws.amazon.com`` a vendor site rather than a retailer and
    ``pages.github.com`` a vendor site rather than a forum. Without a brand, a
    brand's own site is ``vendor``; for a brand it is ``official`` when it is that
    brand's site and ``vendor_other`` when it belongs to a competitor.
    """
    hostname = host(link)
    if not hostname:
        return "unknown"
    if brand and any(_matches(hostname, d) for d in rules["official"].get(brand, [])):
        return "official"
    vendor = "vendor" if brand is None else "vendor_other"
    candidates = [(d, kind) for kind in LISTED_KINDS for d in rules[kind]]
    candidates += [(d, vendor) for b, ds in rules["official"].items() if b != brand for d in ds]
    best, kind = "", "unknown"
    for domain, label in candidates:
        if _matches(hostname, domain) and len(domain) > len(best):
            best, kind = domain, label
    return kind


def validate_rules(rules: dict) -> None:
    brands = {b for category in CATEGORIES for b in load_registry(category).brands}
    unknown = sorted(set(rules["official"]) - brands)
    if unknown:
        raise ValueError(f"Official domains for brands outside the registries: {unknown}")
    seen: dict[str, str] = {}
    listed = [(d, kind) for kind in LISTED_KINDS for d in rules[kind]]
    listed += [(d, "official") for ds in rules["official"].values() for d in ds]
    for domain, kind in listed:
        if domain != domain.strip().lower() or "/" in domain or domain.startswith("www."):
            raise ValueError(f"Domain must be a bare lowercase host: {domain!r}")
        if kind != "official" and seen.get(domain, kind) != kind:
            raise ValueError(f"{domain} is listed as both {seen[domain]} and {kind}")
        if kind != "official":
            seen[domain] = kind


def associations(sources: pd.DataFrame, rules: dict) -> pd.DataFrame:
    rows = []
    for row in sources.to_dict("records"):
        for brand in row["matched_brands"]:
            rows.append(
                {**row, "brand": brand, "source_type": source_type(row["link"], brand, rules)}
            )
    columns = list(dict.fromkeys([*sources.columns, "source_type", "brand"]))
    return pd.DataFrame(rows).reindex(columns=columns)


def contract(english: Path, turkish: Path) -> dict:
    identity = workspace.contract(english, turkish)
    identity["version"] = VERSION
    identity["source_rules"] = str(RULES)
    identity["implementation"][str(MODULE)] = sha256(MODULE)
    return identity


def prepare(root: Path, english: Path, turkish: Path) -> dict:
    rules = read_json(RULES)
    validate_rules(rules)
    quality = workspace.validate_inputs(english, turkish)
    identity = contract(english, turkish)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        old = read_json(manifest_path)
        if old["contract"] != identity:
            raise ValueError("Inputs/config/code changed; use a new --root, keep the old run")
        if old.get("status") == "completed":
            return workspace.verify(root)
    elif root.exists() and any(root.iterdir()):
        raise ValueError("Refusing to adopt an existing directory without a manifest")
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "preparing",
        "contract": identity,
        "identity": digest(identity),
        "quality": quality,
    }
    write_json(manifest_path, manifest)
    previous = datasets.EN_PARQUET, datasets.TR_PARQUET
    artifacts = []
    try:
        datasets.EN_PARQUET, datasets.TR_PARQUET = english, turkish
        for track in ("en", "tr"):
            frame = datasets.load_track(track).frame
            folds = splits.load_frozen(track)
            if set(frame.query_id) != set(folds):
                raise ValueError("Frozen query split differs from the input corpus")
            write_json(
                root / f"responses_{track}.json",
                json.loads(str(frame.to_json(orient="records", force_ascii=False))),
            )
            write_json(root / f"folds_{track}.json", folds)
            sources = retrieved_sources(frame, rules)
            sources["source_type"] = [source_type(link, None, rules) for link in sources["link"]]
            evidence = associations(sources, rules)
            pairs = workspace.pair_table(frame, evidence)
            for name, table in (("sources", sources), ("evidence", evidence), ("pairs", pairs)):
                table.to_parquet(root / f"{name}_{track}.parquet", index=False)
                artifacts.append(f"{name}_{track}.parquet")
            artifacts += [f"responses_{track}.json", f"folds_{track}.json"]
            print(f"prepared {track}: {len(pairs):,} pairs, {len(evidence):,} evidence", flush=True)
    finally:
        datasets.EN_PARQUET, datasets.TR_PARQUET = previous
    manifest.update(status="completed", artifacts={name: sha256(root / name) for name in artifacts})
    write_json(manifest_path, manifest)
    return manifest


def taxonomy_shift(v1: Path, v2: Path) -> pd.DataFrame:
    """Share of evidence rows per source type, v1 next to v2."""
    rows = []
    for track in ("en", "tr"):
        for version, root in (("v1", v1), ("v2", v2)):
            table = pd.read_parquet(root / f"evidence_{track}.parquet", columns=["source_type"])
            for kind, share in table["source_type"].value_counts(normalize=True).items():
                rows.append({"track": track, "version": version, "kind": kind, "share": share})
    return (
        pd.DataFrame(rows)
        .pivot_table(index=["track", "kind"], columns="version", values="share", fill_value=0.0)
        .reset_index()
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--english", type=Path, default=Path("data/interim/reference.parquet"))
    parser.add_argument("--turkish", type=Path, default=Path("data/interim/turkish_raw.parquet"))
    parser.add_argument("--compare-to", type=Path, default=workspace.DEFAULT_ROOT)
    args = parser.parse_args(argv)
    manifest = prepare(args.root, args.english, args.turkish)
    print(json.dumps({"status": manifest["status"], "identity": manifest["identity"]}))
    if (args.compare_to / "manifest.json").exists():
        print(taxonomy_shift(args.compare_to, args.root).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
