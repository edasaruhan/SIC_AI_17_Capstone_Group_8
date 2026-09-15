import pandas as pd

from visibility import intervention as iv

RESULTS = [
    {"title": f"t{i}", "snippet": f"s{i}", "link": f"https://site{i}.com/x"} for i in range(10)
]


def test_comparison_page_lists_the_leaders_and_the_target_once() -> None:
    peers = ["NordVPN", "Proton VPN"]
    page = iv.arm_results(RESULTS, "Windscribe", "vpn", "neutral_p5", peers)[4]
    assert "NordVPN, Proton VPN, Windscribe" in page["snippet"]
    assert "best" not in page["snippet"].casefold()
    superlative = iv.arm_results(RESULTS, "Windscribe", "vpn", "superlative_p5", peers)[4]
    assert "best" in superlative["snippet"] and superlative["link"] == page["link"]
    # A target that is itself a leader is listed once, not twice.
    own = iv.arm_results(RESULTS, "NordVPN", "vpn", "neutral_p1", peers)[0]
    assert own["snippet"].count("NordVPN") == 2  # "Our picks" list + the target sentence


def test_peers_are_the_most_named_brands_on_the_context_queries() -> None:
    rows = []
    for brand, rate in (("NordVPN", 0.9), ("Proton VPN", 0.7), ("Windscribe", 0.1)):
        for i in range(10):
            rows.append(
                {
                    "query_id": "vpn_01",
                    "condition": "search_on",
                    "brand": brand,
                    "y_mention": int(i < rate * 10),
                }
            )
    rows.append({"query_id": "vpn_09", "condition": "search_on", "brand": "IVPN", "y_mention": 1})
    contexts = [{"sector": "vpn", "query_id": "vpn_01"}]
    for sector in ("hosting", "travel"):
        contexts.append({"sector": sector, "query_id": f"{sector}_01"})
    peers = iv.select_peers(pd.DataFrame(rows), contexts)
    assert peers["vpn"] == ["NordVPN", "Proton VPN"]
