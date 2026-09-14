"""Read a brand's website: the text an assistant could also read about it.

The home page and up to three product pages on the same site are fetched, stripped to
readable text with the standard library's HTML parser (no new dependency, ``uv.lock``
stays untouched), and cached, so a second analysis of the same site costs nothing.

Only public web addresses are fetched. Every hop of a redirect is resolved and refused if
it points at a private, loopback or link-local address: the interface takes a URL from a
user, and a server that fetches whatever it is given is an open door into the machine's
own network.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx

from evidence_eval.io import digest, read_json, write_json

CACHE = Path("data/processed/advisor_sites")
USER_AGENT = "Mozilla/5.0 (compatible; kisaliste/1.0; SIC Capstone research)"
MAX_BYTES = 1_500_000
MAX_TEXT = 12_000
MAX_REDIRECTS = 5
PRODUCT_HINTS = (
    "urun",
    "ürün",
    "product",
    "hizmet",
    "service",
    "kart",
    "kredi",
    "plan",
    "fiyat",
    "pricing",
    "feature",
    "ozellik",
    "özellik",
    "koleksiyon",
    "collection",
    "shop",
    "magaza",
    "mağaza",
    "paket",
    "tarife",
    "abonelik",
    "cozum",
    "çözüm",
    "solution",
)
SKIP_TAGS = {"script", "style", "noscript", "svg", "template", "iframe"}
BLOCK_TAGS = {
    "p",
    "li",
    "h1",
    "h2",
    "h3",
    "h4",
    "td",
    "dd",
    "dt",
    "blockquote",
    "figcaption",
    "section",
    "article",
    "div",
    "br",
}
URL_LIKE = re.compile(r"^(https?://)?([\w-]+\.)+[a-z]{2,}(/\S*)?$", re.IGNORECASE)


def as_url(value: str) -> str | None:
    """A URL when the input looks like one ("garantibbva.com.tr" too), else None."""
    candidate = value.strip()
    if " " in candidate or not URL_LIKE.match(candidate):
        return None
    return (
        candidate
        if candidate.lower().startswith(("http://", "https://"))
        else f"https://{candidate}"
    )


def public_host(url: str) -> bool:
    """True only when every address the host resolves to is a public one."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    try:
        infos = socket.getaddrinfo(parsed.hostname, None)
    except OSError:
        return False
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
            or address.is_multicast
            or address.is_unspecified
        ):
            return False
    return bool(infos)


class _Reader(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.skip = 0
        self.in_title = False
        self.title = ""
        self.meta: dict[str, str] = {}
        self.headings: list[str] = []
        self.heading: list[str] | None = None
        self.blocks: list[str] = []
        self.buffer: list[str] = []
        self.links: list[tuple[str, str]] = []
        self.anchor: tuple[str, list[str]] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {key: value or "" for key, value in attrs}
        if tag in SKIP_TAGS:
            self.skip += 1
            return
        if tag == "title":
            self.in_title = True
        if tag == "meta":
            key = (attributes.get("name") or attributes.get("property") or "").lower()
            content = attributes.get("content", "").strip()
            if key in {"description", "og:description", "og:title", "og:site_name"} and content:
                self.meta[key] = content
        if tag in BLOCK_TAGS:
            self._flush()
        if tag in {"h1", "h2", "h3"}:
            self.heading = []
        if tag == "a" and attributes.get("href"):
            self.anchor = (attributes["href"], [])

    def handle_endtag(self, tag: str) -> None:
        if tag in SKIP_TAGS:
            self.skip = max(0, self.skip - 1)
            return
        if tag == "title":
            self.in_title = False
        if tag in {"h1", "h2", "h3"} and self.heading is not None:
            text = " ".join("".join(self.heading).split())
            if text:
                self.headings.append(text)
            self.heading = None
        if tag == "a" and self.anchor is not None:
            self.links.append((self.anchor[0], " ".join("".join(self.anchor[1]).split())))
            self.anchor = None
        if tag in BLOCK_TAGS:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self.skip:
            return
        if self.in_title:
            self.title += data
            return
        self.buffer.append(data)
        if self.heading is not None:
            self.heading.append(data)
        if self.anchor is not None:
            self.anchor[1].append(data)

    def _flush(self) -> None:
        text = " ".join("".join(self.buffer).split())
        self.buffer = []
        if len(text) >= 3:
            self.blocks.append(text)


@dataclass
class Page:
    url: str
    title: str
    description: str
    headings: list[str]
    text: str


@dataclass
class Site:
    url: str
    pages: list[Page]
    # When the pages were fetched, and -- after a forced reread -- whether the text differs
    # from the copy it replaced (None when there was nothing to compare).
    fetched_at: str = ""
    changed: bool | None = None

    def text(self, limit: int = MAX_TEXT) -> str:
        parts = []
        for page in self.pages:
            parts.append("\n".join(p for p in (page.title, page.description, page.text) if p))
        return "\n\n".join(parts)[:limit]


def parse(html: str, url: str) -> tuple[Page, list[tuple[str, str]]]:
    reader = _Reader()
    reader.feed(html)
    reader.close()
    reader._flush()
    blocks = list(dict.fromkeys(reader.blocks))
    page = Page(
        url=url,
        title=" ".join(reader.title.split()) or reader.meta.get("og:title", ""),
        description=reader.meta.get("description") or reader.meta.get("og:description", ""),
        headings=reader.headings[:20],
        text="\n".join(blocks)[:MAX_TEXT],
    )
    return page, reader.links


def _host(url: str) -> str:
    return (urlparse(url).hostname or "").removeprefix("www.")


def product_links(links: list[tuple[str, str]], base_url: str, limit: int = 3) -> list[str]:
    """Same-site links whose address or anchor text looks like a product or service page."""
    host = _host(base_url)
    scored: dict[str, int] = {}
    for href, anchor in links:
        url = urljoin(base_url, href).split("#")[0]
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or _host(url) != host:
            continue
        if url.rstrip("/") == base_url.rstrip("/"):
            continue
        blob = f"{parsed.path} {anchor}".casefold()
        score = sum(hint in blob for hint in PRODUCT_HINTS)
        if score:
            scored[url] = max(scored.get(url, 0), score)
    ranked = sorted(scored.items(), key=lambda kv: (-kv[1], len(kv[0])))
    return [url for url, _ in ranked[:limit]]


async def fetch_html(http: httpx.AsyncClient, url: str) -> tuple[str, str]:
    """Follow redirects by hand, checking every hop is public. Returns (html, final url)."""
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        if not public_host(current):
            raise ValueError("Bu adres herkese açık bir web sitesine gitmiyor.")
        async with http.stream(
            "GET",
            current,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
            follow_redirects=False,
        ) as response:
            if response.is_redirect and response.headers.get("location"):
                current = urljoin(current, response.headers["location"])
                continue
            if response.status_code >= 400:
                raise ValueError(f"Site {response.status_code} yanıtı döndürdü.")
            if "html" not in response.headers.get("content-type", "").lower():
                raise ValueError("Adres bir web sayfası döndürmedi.")
            chunks: list[bytes] = []
            size = 0
            async for chunk in response.aiter_bytes():
                chunks.append(chunk)
                size += len(chunk)
                if size > MAX_BYTES:
                    break
            body = b"".join(chunks)
            return body.decode(response.encoding or "utf-8", errors="replace"), current
    raise ValueError("Site çok fazla yönlendirme yaptı.")


def _cached(data: dict) -> Site:
    return Site(
        url=data["url"],
        pages=[Page(**page) for page in data["pages"]],
        fetched_at=data.get("fetched_at", ""),
    )


async def read_site(
    url: str,
    http: httpx.AsyncClient,
    *,
    cache: Path = CACHE,
    pages: int = 3,
    refresh: bool = False,
) -> Site:
    """The site from the cache, or fetched and cached. ``refresh`` fetches it again even
    when a copy is kept -- a site changes, and the cache never expires on its own."""
    path = cache / f"{digest(url)[:16]}.json"
    old = read_json(path) if path.exists() else None
    if old and not refresh:
        return _cached(old)
    html, final = await fetch_html(http, url)
    home, links = parse(html, final)
    extra: list[Page] = []
    for link in product_links(links, final, pages):
        try:
            page_html, page_url = await fetch_html(http, link)
        except (ValueError, httpx.HTTPError):
            continue
        extra.append(parse(page_html, page_url)[0])
    fresh = Site(url=final, pages=[home, *extra])
    site = Site(
        url=fresh.url,
        pages=fresh.pages,
        fetched_at=datetime.now(UTC).isoformat(timespec="seconds"),
        changed=None if old is None else fresh.text() != _cached(old).text(),
    )
    write_json(
        path,
        {
            "url": site.url,
            "pages": [asdict(page) for page in site.pages],
            "fetched_at": site.fetched_at,
        },
    )
    return site
