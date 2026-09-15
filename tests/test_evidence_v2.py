import pandas as pd
import pytest

from evidence_eval.io import read_json
from visibility import evidence_v2 as v2

RULES = {
    "official": {
        "AWS": ["aws.amazon.com"],
        "Vercel": ["vercel.com"],
        "GitHub Pages": ["pages.github.com"],
    },
    "retailer": ["amazon.com"],
    "forum": ["github.com", "reddit.com"],
    "affiliate": ["vpnmentor.com"],
    "editorial": ["pcmag.com"],
}


@pytest.mark.parametrize(
    ("link", "brand", "expected"),
    [
        ("https://aws.amazon.com/lambda", "AWS", "official"),
        ("https://aws.amazon.com/lambda", "Vercel", "vendor_other"),
        ("https://aws.amazon.com/lambda", None, "vendor"),
        ("https://www.amazon.com/dp/1", "Vercel", "retailer"),
        ("https://pages.github.com/", None, "vendor"),
        ("https://github.com/vercel/next.js", "Vercel", "forum"),
        ("https://www.pcmag.com/picks/best-vpn", "Vercel", "editorial"),
        ("https://tr.vpnmentor.com/review", None, "affiliate"),
        ("https://notamazon.com/", None, "unknown"),
        ("javascript:alert(1)", None, "unknown"),
    ],
)
def test_longest_matching_domain_decides_the_source_type(link, brand, expected) -> None:
    assert v2.source_type(link, brand, RULES) == expected


def test_associations_classify_each_brand_on_the_same_result_separately() -> None:
    sources = pd.DataFrame(
        {"link": ["https://vercel.com/pricing"], "matched_brands": [["Vercel", "AWS"]]}
    )
    rows = v2.associations(sources, RULES).set_index("brand")["source_type"].to_dict()
    assert rows == {"Vercel": "official", "AWS": "vendor_other"}


def test_shipped_rules_are_valid_and_cover_editorial_and_affiliate() -> None:
    rules = read_json(v2.RULES)
    v2.validate_rules(rules)
    assert len(rules["editorial"]) >= 20 and len(rules["affiliate"]) >= 20
    assert v2.source_type("https://www.pcmag.com/x", "NordVPN", rules) == "editorial"
    assert v2.source_type("https://www.security.org/vpn/", "NordVPN", rules) == "affiliate"
    assert v2.source_type("https://nordvpn.com/pricing", "NordVPN", rules) == "official"
    assert v2.source_type("https://nordvpn.com/pricing", "Surfshark", rules) == "vendor_other"


def test_rules_validation_rejects_conflicts_and_unknown_brands() -> None:
    conflicting = {**RULES, "official": {}, "forum": ["pcmag.com"]}
    with pytest.raises(ValueError, match="both"):
        v2.validate_rules(conflicting)
    with pytest.raises(ValueError, match="outside the registries"):
        v2.validate_rules({**RULES, "official": {"Not A Brand": ["x.com"]}})
    with pytest.raises(ValueError, match="bare lowercase"):
        v2.validate_rules({**RULES, "official": {}, "editorial": ["www.pcmag.com"]})
