"""Whose site is this? A rule that also works in sectors nobody curated.

The source taxonomy lists official domains only for the five research sectors. In any
other sector a rival's own site would pass as an "independent page" -- the first live
banking run suggested isbank.com.tr as a place for Garanti BBVA to get listed. So a
domain is matched against the run's own candidate brands instead: the name part of the
host ("isbank" in www.isbank.com.tr) against each brand folded to ASCII ("İş Bankası"
-> "isbankasi"). Domains are ASCII; Turkish brand names are not, so folding is the
whole trick.

Platforms that host everyone's pages -- app stores, social networks, marketplaces --
are never outreach targets either: there is no editor to ask.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from urllib.parse import urlparse

PLATFORM_HOSTS = (
    "play.google.com",
    "apps.apple.com",
    "youtube.com",
    "facebook.com",
    "instagram.com",
    "x.com",
    "twitter.com",
    "linkedin.com",
    "tiktok.com",
    "wikipedia.org",
    "reddit.com",
    "amazon.com",
    "amazon.com.tr",
    "trendyol.com",
    "hepsiburada.com",
    "n11.com",
)
# Second-level public suffixes common in the markets this tool serves.
SECOND_LEVEL = ("com.tr", "net.tr", "org.tr", "gov.tr", "edu.tr", "gen.tr", "co.uk", "org.uk")
_FOLD = str.maketrans({"ı": "i", "İ": "i", "ş": "s", "Ş": "s", "ğ": "g", "Ğ": "g"})


def ascii_key(text: str) -> str:
    """Lowercase ASCII letters and digits only: 'İş Bankası' -> 'isbankasi'."""
    decomposed = unicodedata.normalize("NFKD", text.translate(_FOLD))
    plain = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]", "", plain.casefold())


def host(link: str) -> str:
    return urlparse(link or "").netloc.casefold().removeprefix("www.")


def host_label(link: str) -> str:
    """The registrable name of a host: www.isbank.com.tr -> isbank, a.nordvpn.com -> nordvpn."""
    name = host(link) if "://" in (link or "") else (link or "").casefold().removeprefix("www.")
    if not name:
        return ""
    for suffix in SECOND_LEVEL:
        if name.endswith(f".{suffix}"):
            return ascii_key(name[: -len(suffix) - 1].split(".")[-1])
    parts = name.split(".")
    return ascii_key(parts[-2] if len(parts) >= 2 else parts[0])


def is_platform(link: str) -> bool:
    name = host(link)
    return any(name == p or name.endswith(f".{p}") for p in PLATFORM_HOSTS)


def names_match_label(name: str, label: str) -> bool:
    """Does a brand name and a domain label refer to the same company?

    Exact match needs three characters (qnb); containment needs four on both sides, so
    short common strings do not collide.
    """
    key = ascii_key(name)
    if not key or not label:
        return False
    if key == label:
        return len(key) >= 3
    return min(len(key), len(label)) >= 4 and (label in key or key in label)


def owned_by(link: str, brands: Iterable[str]) -> str | None:
    """The candidate brand whose own site this is, if any."""
    label = host_label(link)
    return next((brand for brand in brands if names_match_label(brand, label)), None)
