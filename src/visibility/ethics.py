"""The manipulation filter: what this product refuses to recommend, in code.

The literature the project is built on shows that fabricated authority -- invented
clinical citations, expert endorsements that do not exist -- is among the most
effective ways to raise a brand's visibility in assistant answers. A tool that
optimised visibility naively would surface exactly that as its top advice, which
would make it an instrument for deceiving the reader rather than for competing.

The concept note promised that the constraint would be *structural rather than
advisory*: not a line in a prompt asking the model to behave, but a list kept in
code, applied to output, and unit-tested. This module is that promise.

Where it applies:

* ``brand_demo.core.advice`` -- the only place free LLM text reaches a user. It is
  already the step validator, so raising here stops the receipt from being written:
  a blocked suggestion never enters the cache and never reaches the report.
* ``visibility.brand_profile.recommendations`` -- the static pool. Nothing there is
  generated, so the filter cannot fire in production; running it anyway means the
  pool itself is checked by the same rule, and a future edit that smuggles in a
  manipulative template fails the test suite.

What the list is *not*: a safety classifier or a claim about the truth of any
sentence. It matches advice that tells a brand to manufacture evidence, buy
endorsement, impersonate independent voices, promise a ranking it cannot promise,
or tamper with someone else's site. Legitimate advice that merely mentions reviews,
comparisons or rankings is left alone, which is why every pattern below needs an
action verb next to the object.

Prompt-level instructions stay where they are (``brand_demo.core.ADVICE_SYSTEM``);
this is the layer that does not depend on a model choosing to comply.
"""

from __future__ import annotations

import re

# (rule, pattern). Turkish and English, because advice reaches users in both.
# Each pattern pairs an action with its object; "inceleme" alone is ordinary advice,
# "sahte inceleme yaz" is not.
BLOCKED_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        "fabricated_reviews",
        r"(sahte|yapay|kurgusal|uydurma)\s+(yorum|inceleme|değerlendirme|kullanıcı)"
        r"|fake\s+(review|testimonial|comment)"
        r"|(yorum|inceleme|değerlendirme)\s+(satın\s*al|satınal|sipariş\s*et)"
        r"|buy\s+(reviews|ratings|testimonials)",
    ),
    (
        "paid_placement_as_editorial",
        r"(ücretli|paralı|sponsorlu)\s+(liste\w*|yerleşim|sıralama|giriş)"
        r"|(liste\w*|sıra\w*|yerleşim)[^.!?]{0,40}?(ücret|para)\s*öde"
        r"|(ücret|para)\s*öde[^.!?]{0,40}?(liste\w*|sıra\w*|yerleşim)"
        r"|pay\s+(for|to)[^.!?]{0,30}?(placement|listing|ranking|spot)"
        r"|gizli\s+sponsorlu|undisclosed\s+sponsorship",
    ),
    (
        "fabricated_authority",
        r"(uydur\w+|olmayan|sahte|kurgusal)\s+(klinik|bilimsel|araştırma|çalışma|test|sertifika|ödül)"
        r"|(klinik|bilimsel)\s+(çalışma|araştırma|kanıt)\s+(uydur|icat\s*et)"
        r"|(fabricate|invent|make\s*up)\s+(a\s+)?(clinical|scientific|study|citation|certification|award)"
        r"|(sahte|olmayan)\s+(uzman|doktor|dermatolog)\s+(onay|görüş|tavsiye)"
        r"|fake\s+(expert|doctor|dermatologist)\s+(endorsement|approval)",
    ),
    (
        "guaranteed_ranking",
        r"(garantili|garanti\s+ed\w+)\s+(sıra|sıralama|birincilik|listeye\s*giriş|görünürlük)"
        r"|(ilk\s*sıra|birinci\s*sıra|bir\s*numara)\w*\s+garanti"
        r"|guarantee\w*\s+(the\s+)?(ranking|first\s+place|top\s+spot|visibility)",
    ),
    (
        "mass_spam",
        r"(toplu|otomatik)\s+(spam|yorum|mesaj|gönderi|paylaşım)\s*(gönder|at|yay|bırak)"
        r"|(forum|blog|yorum)\s*(spam|bombardıman)"
        r"|mass[- ]?(post|comment|spam)|comment\s+spam|bot\s+(accounts?|network)",
    ),
    (
        "tampering_with_others",
        r"(rakip\w*|başkasının|üçüncü\s*taraf\w*)\s+(site|sayfa|içerik)\w*\s*(değiştir|düzenle|sil|kaldır)"
        r"|edit\s+(the\s+)?(competitor|third[- ]party)('?s)?\s+(site|page|content)"
        r"|(rakip\w*|rakib\w+)\s+(hakkında\s+)?(olumsuz|kötü)\s+(yorum|içerik)\w*\s*(yaz|yay)",
    ),
    (
        "impersonation",
        r"(gerçek\s+)?kullanıcı\w*\s+gibi\s+(davran|görün|yaz)"
        r"|(bağımsız|tarafsız)\s+(görünen|süsü\s*veren)[^.!?]{0,20}?(sayfa|site|inceleme)"
        r"|pose\s+as\s+(a\s+)?(real\s+)?(user|customer|reviewer)"
        r"|astroturf\w*",
    ),
)

_COMPILED = tuple((rule, re.compile(pattern, re.IGNORECASE)) for rule, pattern in BLOCKED_PATTERNS)


def explain(text: str) -> list[str]:
    """Names of the rules this text trips; empty when it is clean."""
    return [rule for rule, pattern in _COMPILED if pattern.search(text or "")]


def screen(text: str, *, where: str = "öneri") -> None:
    """Raise on manipulative advice; return silently otherwise."""
    broken = explain(text)
    if broken:
        raise ValueError(
            f"{where} etik filtreye takıldı ({', '.join(broken)}); "
            "doğrulanabilir kanıt ve kaynak çeşitliliği dışındaki taktikler önerilmez."
        )


def screen_actions(actions: list[dict], *, field: str = "suggestion") -> list[dict]:
    """Screen every suggestion in an action list, returning it unchanged when clean."""
    for index, action in enumerate(actions, 1):
        screen(str(action.get(field, "")), where=f"{index}. öneri")
    return actions
