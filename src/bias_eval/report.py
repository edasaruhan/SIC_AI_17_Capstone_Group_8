"""Descriptive English-reference versus Turkish-VPN comparison."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .logging import logger

REFERENCE_QUERY_IDS = {"vpn_01", "vpn_02", "vpn_03", "vpn_04", "vpn_08"}


def _load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return pq.read_table(path).to_pylist()


def _parsed(row: dict[str, Any], key: str) -> dict[str, Any]:
    value = row.get(key)
    if isinstance(value, str):
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    if isinstance(value, dict):
        return value
    prefix = f"{key}__"
    return {
        str(column)[len(prefix) :]: item
        for column, item in row.items()
        if str(column).startswith(prefix)
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    core = [_parsed(row, "core") for row in rows]
    deterministic = [_parsed(row, "deterministic") for row in rows]
    picks = Counter(
        str(item.get("top_recommendation")) for item in core if item.get("top_recommendation")
    )
    return {
        "rows": len(rows),
        "conditions": dict(sorted(Counter(str(row.get("condition")) for row in rows).items())),
        "single_winner_rate": (
            round(sum(bool(item.get("has_single_winner")) for item in core) / len(core), 4)
            if core
            else None
        ),
        "mean_response_words": (
            round(
                sum(int(item.get("response_length_words", 0)) for item in deterministic)
                / len(deterministic),
                2,
            )
            if deterministic
            else None
        ),
        "top_recommendations": dict(picks.most_common(10)),
    }


def build_report(
    turkish_path: Path,
    reference_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    logger.info(
        "report_started turkish_path={} reference_path={} output_path={}",
        turkish_path,
        reference_path,
        output_path,
    )
    turkish = [row for row in _load(turkish_path) if row.get("category") == "vpn"]
    english = [
        row
        for row in _load(reference_path)
        if row.get("category") == "vpn" and row.get("query_id") in REFERENCE_QUERY_IDS
    ]
    summary = {"turkish_vpn": _summary(turkish), "english_reference_vpn": _summary(english)}
    lines = [
        "# Türkçe–İngilizce VPN Betimsel Karşılaştırması",
        "",
        (
            "Bu rapor yalnız betimseldir. Diller arasındaki farklar model, tarih, sağlayıcı ve "
            "örnekleme koşullarından da kaynaklanabileceği için nedensel dil etkisi olarak yorumlanmamalıdır."
        ),
        "",
        "| Ölçüm | Türkçe VPN | İngilizce referans VPN |",
        "|---|---:|---:|",
        f"| Satır | {summary['turkish_vpn']['rows']} | {summary['english_reference_vpn']['rows']} |",
        (
            "| Tek kazanan oranı | "
            f"{summary['turkish_vpn']['single_winner_rate']} | "
            f"{summary['english_reference_vpn']['single_winner_rate']} |"
        ),
        (
            "| Ortalama yanıt kelimesi | "
            f"{summary['turkish_vpn']['mean_response_words']} | "
            f"{summary['english_reference_vpn']['mean_response_words']} |"
        ),
        "",
        "## En sık top recommendation",
        "",
        "```json",
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(
        "report_completed output_path={} turkish_rows={} english_rows={}",
        output_path,
        summary["turkish_vpn"]["rows"],
        summary["english_reference_vpn"]["rows"],
    )
    return summary
