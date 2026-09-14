"""The run's report as a PDF, in the page's own look.

Built from the result the page shows (``app.result_of``); nothing here measures or decides.
Assistant answers arrive as Markdown, often with tables, and are set as real tables. Every
string is escaped before a mark is added, so answer text can never become markup.

The typefaces are the page's (Google Fonts); without a network the PDF falls back to the
same fallback stack the page uses.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import date
from html import escape

from advisor.advise import limits
from advisor.audit import VERDICT_TR
from advisor.render import pct

FONTS = (
    "https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;"
    "0,6..72,500;1,6..72,400&family=Schibsted+Grotesk:wght@400;500;700;800&display=swap"
)
MONTHS = (
    "Ocak",
    "Şubat",
    "Mart",
    "Nisan",
    "Mayıs",
    "Haziran",
    "Temmuz",
    "Ağustos",
    "Eylül",
    "Ekim",
    "Kasım",
    "Aralık",
)
RULE_ROW = re.compile(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?$")
BOLD = re.compile(r"(\*\*[^*]+\*\*)")
BREAK = re.compile(r"<br\s*/?>", re.IGNORECASE)
HEADING = re.compile(r"^#{1,6}\s+(.*)$")
BULLET = re.compile(r"^(?:[*\-•]|\d+[.)])\s+(.*)$")


def _lower(text: str) -> str:
    """Turkish lower case that keeps every character in place, so indexes line up."""
    out = []
    for char in text:
        low = "ı" if char == "I" else "i" if char == "İ" else char.lower()
        out.append(low if len(low) == 1 else char)
    return "".join(out)


def marked(text: str, own: Iterable[str] = (), rivals: Iterable[str] = ()) -> str:
    """Escaped text with the brand's spellings and rival names wrapped in <mark>."""
    own = [name for name in own if name]
    names = sorted(
        [(name, "own") for name in own] + [(n, "rival") for n in rivals if n and n not in own],
        key=lambda item: -len(item[0]),
    )
    lower = _lower(text)
    out, i = [], 0
    while i < len(text):
        hit = None
        for name, cls in names:
            at = lower.find(_lower(name), i)
            if at != -1 and (hit is None or at < hit[0]):
                hit = (at, len(name), cls)
        if hit is None:
            out.append(escape(text[i:]))
            break
        at, size, cls = hit
        out.append(escape(text[i:at]))
        out.append(f'<mark class="{cls}">{escape(text[at:at + size])}</mark>')
        i = at + size
    return "".join(out)


def inline(text: str, own: Iterable[str] = (), rivals: Iterable[str] = ()) -> str:
    """**bold** and <br> kept; any other Markdown mark dropped."""
    own, rivals = list(own), list(rivals)
    lines = []
    for chunk in BREAK.split(text):
        parts = []
        for part in BOLD.split(chunk):
            if len(part) > 4 and part.startswith("**") and part.endswith("**"):
                parts.append(f"<strong>{marked(part[2:-2], own, rivals)}</strong>")
            else:
                parts.append(marked(re.sub(r"\*\*|__|`", "", part), own, rivals))
        lines.append("".join(parts))
    return "<br>".join(lines)


def _cells(line: str) -> list[str]:
    line = line.strip()
    line = line[1:] if line.startswith("|") else line
    line = line[:-1] if line.endswith("|") else line
    return [cell.strip() for cell in line.split("|")]


def _table(rows: list[str], own: list[str], rivals: list[str]) -> str:
    parsed = [(bool(RULE_ROW.match(row)), _cells(row)) for row in rows]
    has_head = len(parsed) > 1 and parsed[1][0]
    body = [cells for rule, cells in parsed if not rule]
    if not body:
        return ""
    width = max(len(cells) for cells in body)

    def row(cells: list[str], tag: str) -> str:
        filled = cells + [""] * (width - len(cells))
        return (
            "<tr>" + "".join(f"<{tag}>{inline(c, own, rivals)}</{tag}>" for c in filled) + "</tr>"
        )

    head = f"<thead>{row(body[0], 'th')}</thead>" if has_head else ""
    rest = body[1:] if has_head else body
    return f'<table class="md-table">{head}<tbody>{"".join(row(c, "td") for c in rest)}</tbody></table>'


def markdown(text: str, own: Iterable[str] = (), rivals: Iterable[str] = ()) -> str:
    """An assistant's Markdown answer as HTML: headings, lists, tables, paragraphs."""
    own, rivals = list(own), list(rivals)
    # Quote marks ("> ") carry no meaning in an answer card; the text under them does.
    lines = [
        re.sub(r"^>\s?", "", line.strip()).strip() for line in text.replace("\r", "").split("\n")
    ]
    blocks: list[str] = []
    bullets: list[str] = []

    def flush() -> None:
        if bullets:
            blocks.append("<ul>" + "".join(f"<li>{item}</li>" for item in bullets) + "</ul>")
            bullets.clear()

    def next_filled(i: int) -> str:
        while i < len(lines) and not lines[i]:
            i += 1
        return lines[i] if i < len(lines) else ""

    i = 0
    while i < len(lines):
        line = lines[i]
        if not line:
            flush()
            i += 1
            continue
        if line.startswith("|"):
            flush()
            rows = []
            # Models often leave a blank line between a table's rows; it is still one table.
            while i < len(lines):
                if lines[i].startswith("|"):
                    rows.append(lines[i])
                elif not (not lines[i] and next_filled(i).startswith("|")):
                    break
                i += 1
            blocks.append(_table(rows, own, rivals))
            continue
        heading, bullet = HEADING.match(line), BULLET.match(line)
        if heading:
            flush()
            blocks.append(f'<p class="md-h">{inline(heading[1], own, rivals)}</p>')
        elif re.fullmatch(r"[-*_]{3,}", line):
            flush()
        elif bullet:
            bullets.append(inline(bullet[1], own, rivals))
        else:
            flush()
            blocks.append(f"<p>{inline(line, own, rivals)}</p>")
        i += 1
    flush()
    return "".join(blocks)


def _date(day: date) -> str:
    return f"{day.day} {MONTHS[day.month - 1]} {day.year}"


def _figure(value: str, label: str) -> str:
    return f'<div class="figure"><b>{escape(value)}</b><span>{escape(label)}</span></div>'


def _section(title: str, lead: str, body: str, cls: str = "") -> str:
    intro = f'<p class="lead">{escape(lead)}</p>' if lead else ""
    return f'<section class="section {cls}"><h2>{escape(title)}</h2>{intro}{body}</section>'


def _link(url: str, text: str) -> str:
    if not url.startswith(("http://", "https://")):
        return escape(text)
    return f'<a href="{escape(url, quote=True)}">{escape(text)}</a>'


def _recommendations(result: dict) -> str:
    items = []
    for rec in result.get("recommendations") or []:
        targets = ""
        if rec.get("targets"):
            rows = "".join(
                f"<tr><td>{_link(t.get('link', ''), t['domain'])}</td>"
                f"<td>{escape(', '.join(t['rivals']))}</td>"
                f"<td class=\"num\">{escape(str(t.get('position') or '–'))}</td></tr>"
                for t in rec["targets"]
            )
            targets = (
                "<table><thead><tr><th>Sayfa</th><th>Andığı rakipler</th>"
                f'<th class="num">Sıra</th></tr></thead><tbody>{rows}</tbody></table>'
            )
        items.append(
            f'<li class="rec"><h3>{escape(rec["title"])}</h3><p>{escape(rec["why"])}</p>'
            f'<p class="todo">{escape(rec["action"])}</p>'
            f'<p class="evidence">{escape(rec["verdict"])}</p>{targets}</li>'
        )
    if not items:
        return ""
    return _section(
        "Listeye girmek için",
        "Her öneri kontrollü testte ölçülmüş etkisiyle birlikte.",
        f'<ol class="recs">{"".join(items)}</ol>',
    )


def _bar(share: float, cls: str) -> str:
    width = max(0.0, min(1.0, share)) * 78
    return (
        f'<div class="bar {cls}"><i style="width:{width:.1f}%"></i><span>{pct(share)}</span></div>'
    )


def _description(result: dict) -> str:
    record = result.get("description_audit") or {}
    if not record.get("findings") and not record.get("advice"):
        return ""
    source = (
        "Sitendeki ürün cümleleri incelendi."
        if record.get("source") == "user"
        else "Sitende ürün cümlesi bulunamadığı için arama sonuçlarında markanı anlatan "
        "özetler incelendi."
    )
    body = []
    marks = result.get("description_marks") or []
    if marks:
        spans = " ".join(
            (
                f'<span class="s {escape(m["verdict"], quote=True)}">{escape(m["text"])}'
                f'<span class="tag">{escape(", ".join(m["labels"]))}</span></span>'
                if m.get("verdict")
                else escape(m["text"])
            )
            for m in marks
        )
        legend = "".join(
            f'<span><i class="{key}"></i>{escape(VERDICT_TR[key])}</span>'
            for key in ("strong", "mixed", "weak", "risk")
        )
        body.append(f'<p class="copy">{spans}</p><p class="legend">{legend}</p>')
    advice = record.get("advice") or []
    if advice:
        body.append(
            '<ul class="advice">'
            + "".join(
                f'<li class="{escape(item.get("kind", ""), quote=True)}">'
                f'<b>{escape(item["suggestion"])}</b><span>{escape(item["basis"])}</span></li>'
                for item in advice
            )
            + "</ul>"
        )
    findings = record.get("findings") or []
    if findings:
        rows = "".join(
            f'<tr><td class="label">{escape(row["label"])}</td>'
            f'<td>{_bar(row["gemini"], "s1")}{_bar(row["cerebras"], "s2")}</td></tr>'
            for row in findings
        )
        body.append(
            '<h3 class="minor">Bulunan cümle türleri deneyde ne kazandırdı</h3>'
            '<p class="series"><span><i class="s1"></i>Gemini 3.5 Flash Lite</span>'
            '<span><i class="s2"></i>gpt-oss-120b</span></p>'
            f'<table class="bars">{rows}</table>'
            '<p class="hint">Her kartta farklı bir cümle olduğunda o türün önerilen ürün olma '
            "payı. Rastgele seçim %20.</p>"
        )
    return _section(
        "Seçilmek için: açıklaman",
        f"Marka listedeyken asistan somut ürün bilgisine göre seçiyor. {source}",
        "".join(body),
    )


def _named(result: dict) -> str:
    per = result.get("named_measures") or {}
    if not per:
        return ""
    labels = result.get("assistant_labels") or {}
    figures = "".join(
        _figure(
            pct(m.get("picked")),
            f"{labels.get(name, name)} seni önerdi"
            + (f"; yerine en çok {m['rivals_picked'][0][0]}" if m.get("rivals_picked") else ""),
        )
        for name, m in per.items()
    )
    questions = "".join(f"<li>{escape(q)}</li>" for q in result.get("named_queries") or [])
    return _section(
        "Adın geçtiğinde",
        "Markanı ve ürününü anan sorular. Asistan seni mi, rakibini mi öneriyor? "
        "Yanıtların tamamı ekte.",
        f'<div class="figures">{figures}</div><ul class="plain">{questions}</ul>',
    )


def _rivals(result: dict) -> str:
    counts: dict[str, int] = {}
    for row in result.get("answers") or []:
        if row.get("kind") != "named" and row.get("assistant") == "gemini":
            for name in row.get("named_brands") or []:
                counts[name] = counts.get(name, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:8]
    if not ranked:
        return ""
    top = ranked[0][1]
    rows = "".join(
        f'<tr><td class="label">{escape(name)}</td><td><div class="bar '
        f'{"own" if name == result.get("brand") else "rival"}">'
        f'<i style="width:{count / top * 78:.1f}%"></i><span>{count}</span></div></td></tr>'
        for name, count in ranked
    )
    return _section(
        "Yanıtlarda kimler var",
        "Arama sonuçlarıyla verilen Gemini yanıtlarında markaların kaç kez anıldığı.",
        f'<table class="bars">{rows}</table>',
    )


def _assistants(result: dict) -> str:
    per = result.get("assistant_measures") or {}
    if len(per) < 2:
        return ""
    labels = result.get("assistant_labels") or {}
    rows = "".join(
        f"<tr><td>{escape(labels.get(name, name))}</td><td class=\"num\">{pct(m.get('mention_off'))}</td>"
        f"<td class=\"num\">{pct(m.get('mention_on'))}</td><td class=\"num\">{pct(m.get('first_on'))}</td>"
        f"<td class=\"num\">{m.get('n_observations', 0)}</td></tr>"
        for name, m in per.items()
    )
    return _section(
        "Asistanlar arasında",
        "Aynı sorular, aynı arama sonuçları. Teşhis Gemini'nin yanıtlarından okunur; ikinci "
        "asistan sonucun tek bir asistana özgü olup olmadığını gösterir.",
        '<table><thead><tr><th>Asistan</th><th class="num">Arama kapalıyken anılma</th>'
        '<th class="num">Arama açıkken anılma</th><th class="num">İlk anılan marka</th>'
        f'<th class="num">Yanıt</th></tr></thead><tbody>{rows}</tbody></table>',
    )


def _signal(result: dict) -> str:
    scores = result.get("scores") or {}
    if not scores:
        return ""
    brand = escape(result.get("brand") or "")
    body = (
        f'<table><thead><tr><th></th><th class="num">{brand}</th>'
        '<th class="num">Karşılaştırılan rakipler</th></tr></thead><tbody>'
        f"<tr><td>Anılma olasılığı (tahmin)</td><td class=\"num\">{pct(scores.get('score_mention'))}</td>"
        f"<td class=\"num\">{pct(scores.get('rival_score_mention'))}</td></tr>"
        f"<tr><td>Birincil öneri olasılığı (tahmin)</td><td class=\"num\">{pct(scores.get('score_top'))}</td>"
        f"<td class=\"num\">{pct(scores.get('rival_score_top'))}</td></tr></tbody></table>"
        f'<p class="hint">{scores.get("candidates", 0)} aday marka içinde anılma tahmininde '
        f'{escape(str(scores.get("rank", "–")))}. sıradasın. Karşılaştırılan rakipler: '
        f'{escape(", ".join(scores.get("compared_with") or []) or "–")}.</p>'
    )
    signals = result.get("signals") or []
    if signals:
        rows = "".join(
            f"<tr><td>{escape(row['label'])}</td><td class=\"num\">{row['brand']:.2f}</td>"
            f"<td class=\"num\">{row['rivals']:.2f}</td><td>{'geride' if row['lagging'] else '–'}</td></tr>"
            for row in signals
        )
        body += (
            "<table><thead><tr><th>Sektörler arasında taşınan sinyal</th>"
            '<th class="num">Sen</th><th class="num">Rakipler</th><th>Durum</th></tr></thead>'
            f"<tbody>{rows}</tbody></table>"
        )
    return _section(
        "Öğrenilmiş sinyal: rakiplerine göre konumun",
        "Kayıtlı yanıtlardan öğrenilen model (M2-Invariant) marka kimliğine ve sektöre bakmaz; "
        "yalnız aynı arama sonuçlarında rakiplerine göre nerede durduğuna bakar.",
        body,
    )


def _answer(row: dict, own: list[str], brand: str) -> str:
    if row.get("kind") == "named":
        picked = row.get("picked")
        meta = (
            ("Seni önerdi" if picked == brand else f"{picked} önerdi")
            if picked
            else "Net bir öneri yok"
        )
    elif row.get("mentioned"):
        meta = "Seni ilk sırada andı" if row.get("first") else "Seni andı, ama ilk sırada değil"
    else:
        meta = "Seni anmadı"
    rivals = [name for name in row.get("named_brands") or [] if name != brand]
    return (
        f'<article class="answer"><p class="q">{escape(row.get("query") or "")}</p>'
        f'<p class="meta">{escape(meta)}</p>'
        f'<div class="a">{markdown(row.get("answer") or "", own, rivals)}</div></article>'
    )


def _answers(result: dict, own: list[str]) -> str:
    rows = result.get("answers") or []
    if not rows:
        return ""
    brand = result.get("brand") or ""
    labels = result.get("assistant_labels") or {}
    parts = []
    for name in dict.fromkeys(row.get("assistant") for row in rows):
        mine = [row for row in rows if row.get("assistant") == name]
        parts.append(f'<h3 class="asked">{escape(labels.get(name or "", name or ""))}</h3>')
        parts += [_answer(row, own, brand) for row in mine if row.get("kind") != "named"]
        named = [row for row in mine if row.get("kind") == "named"]
        if named:
            parts.append('<p class="sub">Markanı ve ürününü anan sorular</p>')
            parts += [_answer(row, own, brand) for row in named]
    return _section(
        "Ek: Asistanlar ne dedi",
        "Arama sonuçlarıyla verilen yanıtların tam metni, her soru için ilk tekrar. Senin "
        "markan sarıyla, rakipler griyle işaretli.",
        "".join(parts),
        cls="appendix",
    )


def _closing(result: dict) -> str:
    items = "".join(
        f"<li>{inline(item)}</li>"
        for item in limits(
            result.get("sector") or "", result.get("language") or "tr", result.get("curated", True)
        )
    )
    notes = "".join(f"<li>{escape(note)}</li>" for note in result.get("notes") or [])
    names = result.get("candidates") or []
    compared = (
        f'<p class="hint">Karşılaştırılan {len(names)} marka: {escape(", ".join(names))}.</p>'
        if names
        else ""
    )
    return _section(
        "Bu rapor ne söylemiyor",
        "",
        f'<ul class="plain">{items}</ul><h3 class="minor">Koşu notları</h3>'
        f'<ul class="plain small">{notes}</ul>{compared}'
        '<p class="hint">Ölçülen etkiler projenin kontrollü testinden gelir; bu koşuda yeniden '
        "ölçülmemiştir.</p>",
    )


def report_html(result: dict, aliases: Iterable[str] = (), made: date | None = None) -> str:
    brand = result.get("brand") or ""
    own = [brand, *aliases]
    m = result.get("measures") or {}
    scores = result.get("scores") or {}
    rank = f"{scores['rank']}/{scores['candidates']}" if scores.get("rank") else "–"
    figures = "".join(
        [
            _figure(pct(m.get("retrieval_presence")), "arama sonuçlarında görünme"),
            _figure(pct(m.get("mention_on")), "asistan yanıtlarında anılma"),
            _figure(pct(m.get("first_on")), "ilk önerilen marka olma"),
            _figure(rank, "rakiplere göre tahmini sıra"),
        ]
    )
    body = "".join(
        [
            '<header class="top"><span class="wordmark"><span class="wm">kısa</span>liste</span>'
            f'<span class="stamp">Görünürlük raporu, {_date(made or date.today())}</span></header>',
            f'<section class="verdict"><p class="who">{escape(brand)}, '
            f'{escape(result.get("sector") or "")}</p>'
            f'<h1>{escape(result.get("diagnosis_title") or "")}</h1>'
            f'<p>{inline(result.get("diagnosis_note") or "")}</p></section>',
            f'<div class="figures lead-figures">{figures}</div>',
            _recommendations(result),
            _description(result),
            _named(result),
            _rivals(result),
            _assistants(result),
            _signal(result),
            _closing(result),
            _answers(result, own),
        ]
    )
    return (
        f'<!doctype html><html lang="tr"><head><meta charset="utf-8">'
        f"<title>{escape(brand)} görünürlük raporu</title>"
        f'<link rel="stylesheet" href="{escape(FONTS, quote=True)}">'
        f"<style>{CSS}</style></head><body>{body}</body></html>"
    )


def report_pdf(result: dict, aliases: Iterable[str] = ()) -> bytes:
    from weasyprint import HTML  # pyright: ignore[reportMissingImports]

    return HTML(string=report_html(result, aliases)).write_pdf()


CSS = """
:root {
  --paper: #f6f7f9; --ink: #101b3b; --ink-2: #414b63; --muted: #6c7489; --rule: #dce0e8;
  --mark: #ffe45c; --mark-rival: #e8ebf1; --good: #1e6b3a; --good-bg: #dcefe1;
  --mixed-bg: #e3ecfb; --weak-bg: #eceef2; --risk: #b8322f; --risk-bg: #f8dcda;
  --s1: #2a78d6; --s2: #eb6834;
  --sans: "Schibsted Grotesk", "Carlito", "Liberation Sans", sans-serif;
  --serif: "Newsreader", "Noto Serif", "Liberation Serif", Georgia, serif;
}
@page {
  size: A4; margin: 18mm 17mm 20mm;
  @bottom-left { content: "kısaliste"; font: 800 9pt "Schibsted Grotesk", "Carlito", sans-serif; color: #101b3b; }
  @bottom-right { content: counter(page) " / " counter(pages); font: 9pt "Schibsted Grotesk", "Carlito", sans-serif; color: #6c7489; }
}
@page :first { @bottom-left { content: none; } }
* { box-sizing: border-box; }
body { margin: 0; font-family: var(--sans); font-size: 9.8pt; line-height: 1.5; color: var(--ink); background: #fff; }
h1, h2, h3 { margin: 0; font-weight: 800; letter-spacing: -0.02em; break-after: avoid; }
p { margin: 0; }
a { color: var(--ink); }
.top { display: flex; justify-content: space-between; align-items: baseline; padding-bottom: 12pt; }
.wordmark { font-weight: 800; font-size: 17pt; letter-spacing: -0.6pt; }
.wm { background: linear-gradient(transparent 52%, var(--mark) 52%, var(--mark) 92%, transparent 92%); }
.stamp { font-size: 9pt; color: var(--muted); }
.verdict { padding: 22pt 0 18pt; border-bottom: 1.4pt solid var(--ink); }
.verdict .who { font-size: 10pt; color: var(--muted); }
.verdict h1 { font-size: 32pt; line-height: 1.04; letter-spacing: -0.035em; margin-top: 4pt; max-width: 15em; }
.verdict p:last-child { margin-top: 10pt; font-size: 11.5pt; color: var(--ink-2); max-width: 36em; }
.figures { display: flex; border-bottom: 0.8pt solid var(--rule); break-inside: avoid; }
.figure { flex: 1; padding: 12pt 10pt 12pt 0; }
.figure + .figure { padding-left: 10pt; border-left: 0.8pt solid var(--rule); }
.figure b { display: block; font-size: 24pt; font-weight: 800; letter-spacing: -0.03em; line-height: 1; }
.figure span { display: block; margin-top: 5pt; font-size: 8.8pt; color: var(--ink-2); }
.section { padding-top: 22pt; }
.section > h2 { font-size: 16pt; line-height: 1.2; }
.section > .lead { color: var(--ink-2); margin: 4pt 0 12pt; max-width: 40em; }
.appendix { break-before: page; }
.minor { font-size: 10.5pt; font-weight: 700; margin: 16pt 0 6pt; }
.hint { font-size: 8.8pt; color: var(--muted); margin-top: 6pt; }
.recs { list-style: none; padding: 0; margin: 0; counter-reset: rec; }
.rec { counter-increment: rec; position: relative; padding-left: 28pt; margin-top: 12pt; }
.rec::before { content: counter(rec); position: absolute; left: 0; top: 0; width: 19pt; height: 19pt; border-radius: 50%;
  background: var(--ink); color: #fff; text-align: center; line-height: 19pt; font-weight: 700; font-size: 9.5pt; }
.rec h3 { font-size: 11.5pt; line-height: 1.3; padding-top: 2pt; }
.rec p { margin-top: 4pt; color: var(--ink-2); }
.rec .todo { color: var(--ink); }
.rec .evidence { font-size: 8.8pt; color: var(--muted); }
table { border-collapse: collapse; width: 100%; margin-top: 8pt; font-size: 9pt; }
th { text-align: left; font-weight: 700; color: var(--ink-2); padding: 5pt 8pt 5pt 0; border-bottom: 1.2pt solid var(--ink); font-size: 8.6pt; vertical-align: bottom; }
td { padding: 5pt 8pt 5pt 0; border-bottom: 0.7pt solid var(--rule); vertical-align: top; }
tr { break-inside: avoid; }
.num { text-align: right; }
mark { background: none; color: inherit; }
mark.own { background: linear-gradient(transparent 12%, var(--mark) 12%, var(--mark) 88%, transparent 88%); font-weight: 600; }
mark.rival { background: linear-gradient(transparent 12%, var(--mark-rival) 12%, var(--mark-rival) 88%, transparent 88%); }
.copy { font-family: var(--serif); font-size: 12pt; line-height: 1.8; background: var(--paper); border-radius: 10pt; padding: 14pt 16pt; }
.copy .s { border-radius: 2pt; padding: 1pt 2pt; box-decoration-break: clone; }
.copy .strong { background: var(--good-bg); border-bottom: 1.5pt solid var(--good); }
.copy .mixed { background: var(--mixed-bg); border-bottom: 1.5pt solid var(--s1); }
.copy .weak { background: var(--weak-bg); border-bottom: 1.5pt solid #9aa1b1; }
.copy .risk { background: var(--risk-bg); border-bottom: 1.5pt solid var(--risk); }
.copy .tag { font-family: var(--sans); font-size: 7.5pt; font-weight: 700; margin-left: 3pt; color: var(--ink-2); }
.legend { margin-top: 8pt; font-size: 8.6pt; color: var(--ink-2); }
.legend span { margin-right: 12pt; white-space: nowrap; }
.legend i { display: inline-block; width: 8pt; height: 8pt; border-radius: 2pt; margin-right: 4pt; vertical-align: -0.5pt; }
.legend .strong { background: var(--good); } .legend .mixed { background: var(--s1); }
.legend .weak { background: #9aa1b1; } .legend .risk { background: var(--risk); }
.advice { list-style: none; padding: 0; margin: 12pt 0 0; }
.advice li { padding-left: 9pt; border-left: 2.2pt solid var(--rule); margin-top: 8pt; break-inside: avoid; }
.advice li.risk { border-left-color: var(--risk); }
.advice li.add, .advice li.differentiate { border-left-color: var(--ink); }
.advice b { display: block; }
.advice span { color: var(--ink-2); font-size: 9pt; }
.series { font-size: 8.8pt; color: var(--ink-2); }
.series span { margin-right: 12pt; }
.series i { display: inline-block; width: 8pt; height: 8pt; border-radius: 2pt; margin-right: 4pt; }
.series .s1 { background: var(--s1); } .series .s2 { background: var(--s2); }
table.bars { margin-top: 6pt; }
table.bars td { border-bottom: 0; padding: 3pt 8pt 3pt 0; vertical-align: middle; }
table.bars .label { width: 38%; font-size: 9pt; }
.bar { display: flex; align-items: center; font-size: 8pt; color: var(--ink-2); margin: 1.5pt 0; }
.bar i { display: block; height: 7pt; min-width: 1.5pt; border-radius: 0 2.5pt 2.5pt 0; background: var(--s1); margin-right: 5pt; }
.bar.s2 i { background: var(--s2); }
.bar.own i { background: var(--ink); }
.bar.rival i { background: #aab2c3; }
.plain { padding-left: 13pt; margin: 6pt 0 0; color: var(--ink-2); }
.plain li + li { margin-top: 3pt; }
.plain.small { font-size: 8.8pt; }
.asked { font-size: 12pt; margin: 16pt 0 2pt; }
.sub { font-weight: 700; color: var(--ink-2); margin: 14pt 0 0; }
.answer { background: var(--paper); border-radius: 9pt; padding: 11pt 13pt; margin-top: 8pt; }
.answer .q { font-weight: 700; font-size: 10pt; break-after: avoid; }
.answer .meta { font-size: 8.6pt; color: var(--muted); break-after: avoid; }
.answer .a { font-family: var(--serif); font-size: 10.5pt; line-height: 1.55; margin-top: 6pt; }
.answer .a > * + * { margin-top: 5pt; }
.answer .a ul { padding-left: 13pt; margin-bottom: 0; }
.answer .md-h { font-family: var(--sans); font-weight: 700; font-size: 9.6pt; margin-top: 9pt; break-after: avoid; }
.answer .md-table { font-family: var(--sans); font-size: 8.4pt; line-height: 1.4; background: #fff; }
.answer .md-table th, .answer .md-table td { padding: 4pt 6pt; overflow-wrap: anywhere; }
.answer .md-table th:first-child, .answer .md-table td:first-child { padding-left: 5pt; }
"""
