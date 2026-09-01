"""Turkish-aware case handling.

Python's ``str.lower`` and ``str.upper`` are wrong for Turkish: ``"I".lower()``
gives ``"i"`` where Turkish needs ``"ı"``, and ``"SIRALAMA".lower()`` therefore
does not round-trip back to ``"SIRALAMA"``. A parser that folds case with the
default methods silently loses responses whose marker was written in lower
case - exactly the kind of loss that is only noticed weeks later.
"""

from __future__ import annotations

import unicodedata

_TO_UPPER = str.maketrans({"i": "İ", "ı": "I"})
_TO_LOWER = str.maketrans({"I": "ı", "İ": "i"})

# Folding for marker matching only: every Turkish letter collapses to its
# closest ASCII form, so SIRALAMA / Sıralama / sıralama / SİRALAMA all match.
_FOLD = str.maketrans(
    {
        "ı": "I",
        "i": "I",
        "İ": "I",
        "ş": "S",
        "Ş": "S",
        "ğ": "G",
        "Ğ": "G",
        "ü": "U",
        "Ü": "U",
        "ö": "O",
        "Ö": "O",
        "ç": "C",
        "Ç": "C",
        "â": "A",
        "Â": "A",
        "î": "I",
        "Î": "I",
        "û": "U",
        "Û": "U",
    }
)


def turkish_upper(text: str) -> str:
    """Uppercase the Turkish way: i becomes İ and ı becomes I."""

    return text.translate(_TO_UPPER).upper()


def turkish_lower(text: str) -> str:
    """Lowercase the Turkish way: I becomes ı and İ becomes i."""

    return text.translate(_TO_LOWER).lower()


def fold(text: str) -> str:
    """Collapse Turkish letters to ASCII upper case for tolerant matching.

    Use this to compare markers and labels, never to store or display a value:
    it deliberately destroys the distinction between ı and i.
    """

    normalized = unicodedata.normalize("NFC", text)
    return normalized.translate(_FOLD).upper()
