"""Canonical brand identity shared by the English and Turkish tracks.

The two corpora spell brands differently. The English reference preserves case
("IVPN", "Mullvad VPN"); the Turkish collection pipeline applied Turkish
casefolding to every extracted name, which maps "I" to "ı" and turned "IVPN"
into "ıvpn" and "Private Internet Access" into "private ınternet access".

`comparison_key` folds I, İ, ı and i to a single "i" so both spellings land on
the same key, then strips everything that is not alphanumeric. Curated registries
under `configs/modeling/brands/` map keys to a display name and list the surface
forms that are not consumer brands in that category.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import yaml

REGISTRY_DIR = Path("configs/modeling/brands")
_I_FORMS = {"I": "i", "İ": "i", "ı": "i", "i": "i"}


def comparison_key(name: str) -> str:
    """Language-agnostic identity key for a brand surface form.

    Folds every dotted/dotless I to "i" before casefolding, so Turkish-casefolded
    English names ("ıvpn") and original-case names ("IVPN") share a key.
    """
    # Strip the trademark marks before NFKC: normalisation expands "™" to "TM",
    # which would otherwise survive as letters and change the key.
    text = name.replace("™", "").replace("®", "").replace("©", "")
    text = unicodedata.normalize("NFKC", text)
    text = "".join(_I_FORMS.get(char, char) for char in text)
    text = text.casefold()
    return "".join(char for char in text if char.isalnum())


@dataclass(frozen=True)
class BrandRegistry:
    """Curated canonical names and exclusions for one category."""

    category: str
    display_by_key: Mapping[str, str]
    excluded_keys: frozenset[str]
    # Original spellings as written in the registry, needed to rewrite brand
    # names out of free text for the masking ablation.
    surface_forms: Mapping[str, tuple[str, ...]]

    def resolve(self, name: str | None) -> str | None:
        """Return the canonical display name, or None if unknown or excluded."""
        if not name:
            return None
        key = comparison_key(name)
        if not key or key in self.excluded_keys:
            return None
        return self.display_by_key.get(key)

    def resolve_all(self, names: Iterable[str] | None) -> list[str]:
        """Resolve a mention list, dropping unknowns and de-duplicating."""
        if names is None:
            return []
        seen: dict[str, None] = {}
        for name in names:
            display = self.resolve(name)
            if display is not None:
                seen.setdefault(display, None)
        return list(seen)

    @property
    def brands(self) -> list[str]:
        return sorted(set(self.display_by_key.values()))


def _build(category: str, document: Mapping[str, object]) -> BrandRegistry:
    canonical = document.get("canonical") or {}
    if not isinstance(canonical, Mapping):
        raise ValueError(f"{category}: 'canonical' must be a mapping")
    display_by_key: dict[str, str] = {}
    surface_forms: dict[str, tuple[str, ...]] = {}
    for display, aliases in canonical.items():
        forms = [str(display), *(str(alias) for alias in (aliases or []))]
        surface_forms[str(display)] = tuple(dict.fromkeys(forms))
        for form in forms:
            key = comparison_key(form)
            if not key:
                continue
            previous = display_by_key.get(key)
            if previous is not None and previous != display:
                raise ValueError(
                    f"{category}: surface form {form!r} maps to both {previous!r} and {display!r}"
                )
            display_by_key[key] = str(display)

    excluded: set[str] = set()
    groups = document.get("exclude") or {}
    if isinstance(groups, Mapping):
        for terms in groups.values():
            for term in terms or []:
                key = comparison_key(str(term))
                if key:
                    excluded.add(key)
    # A curated canonical name always wins over an exclusion list entry, so that
    # "Cloudflare Pages" survives even though "cloudflare" is excluded.
    excluded -= set(display_by_key)
    return BrandRegistry(category, display_by_key, frozenset(excluded), surface_forms)


@cache
def load_registry(category: str, registry_dir: str | None = None) -> BrandRegistry:
    """Load and cache the curated registry for one category."""
    base = Path(registry_dir) if registry_dir else REGISTRY_DIR
    path = base / f"{category}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No brand registry for category {category!r} at {path}")
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if document.get("category") != category:
        raise ValueError(f"{path} declares category {document.get('category')!r}")
    return _build(category, document)
