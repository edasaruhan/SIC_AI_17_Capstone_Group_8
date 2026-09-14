import asyncio
import json

import pytest

from advisor import audit, measure, nodes, profile, service, site
from advisor.candidates import build_registry

HTML = """<!doctype html><html><head>
<title> Bonus Kart | Garanti BBVA </title>
<meta name="description" content="Garanti BBVA Bonus kart ile alışverişin keyfini çıkar.">
<script>var tracking = "gizli";</script><style>.x{}</style>
</head><body>
<nav><a href="/bireysel/kartlar/bonus-kart">Bonus Kart</a> <a href="/iletisim">İletişim</a>
<a href="https://baska-site.com/urun">Dış ürün</a> <a href="/krediler#faiz">Krediler</a></nav>
<h1>Bonus Kart</h1>
<p>Yıllık aidat ödemezsin.</p>
<p>Başvuru mobil uygulamadan 5 dakikada tamamlanır.</p>
<div>Alışverişlerde %2 bonus kazanırsın.</div>
</body></html>"""


def test_parse_reads_title_description_headings_and_text_but_not_scripts():
    page, links = site.parse(HTML, "https://www.garantibbva.com.tr/")
    assert page.title == "Bonus Kart | Garanti BBVA"
    assert page.description.startswith("Garanti BBVA Bonus kart")
    assert "Bonus Kart" in page.headings
    assert "Yıllık aidat ödemezsin." in page.text
    assert "gizli" not in page.text
    assert ("/iletisim", "İletişim") in links


def test_product_links_stay_on_the_site_and_prefer_product_pages():
    _, links = site.parse(HTML, "https://www.garantibbva.com.tr/")
    chosen = site.product_links(links, "https://www.garantibbva.com.tr/")
    assert set(chosen) == {
        "https://www.garantibbva.com.tr/bireysel/kartlar/bonus-kart",
        "https://www.garantibbva.com.tr/krediler",
    }
    assert all("baska-site" not in url and "iletisim" not in url for url in chosen)


def test_urls_are_recognised_and_private_hosts_refused():
    assert site.as_url("garantibbva.com.tr") == "https://garantibbva.com.tr"
    assert site.as_url("https://example.com/x") == "https://example.com/x"
    assert site.as_url("Garanti BBVA") is None
    assert not site.public_host("http://127.0.0.1:8600/")
    assert not site.public_host("http://localhost/")
    assert not site.public_host("file:///etc/passwd")


SITE_TEXT = (
    "Garanti BBVA Bonus kart ile alışverişin keyfini çıkar. Yıllık aidat ödemezsin. "
    "Başvuru mobil uygulamadan 5 dakikada tamamlanır. GarantiBBVA mobil uygulaması."
)


def test_profile_keeps_only_what_the_text_supports():
    reply = json.dumps(
        {
            "brand": "Garanti BBVA",
            "aliases": ["GarantiBBVA", "Garanti Bankası"],
            "sector": "bankacılık",
            "language": "tr",
            "products": [
                {
                    "name": "Bonus kart",
                    "category": "kredi kartı",
                    "sentences": ["Yıllık aidat ödemezsin.", "Sınırsız nakit iade verir."],
                },
                {"name": "", "category": "boş"},
            ],
        }
    )
    parsed = profile.parse_profile(f"```json\n{reply}\n```", SITE_TEXT)
    assert parsed["aliases"] == ["GarantiBBVA"]
    assert parsed["products"] == [
        {"name": "Bonus kart", "category": "kredi kartı", "sentences": ["Yıllık aidat ödemezsin."]}
    ]
    assert profile.description(parsed) == "Yıllık aidat ödemezsin."
    with pytest.raises(ValueError):
        profile.parse_profile(json.dumps({"brand": "", "sector": "x"}), SITE_TEXT)


PROFILE = {
    "brand": "Garanti BBVA",
    "aliases": ["Garanti"],
    "sector": "bankacılık",
    "language": "tr",
    "products": [{"name": "Bonus kart", "category": "kredi kartı", "sentences": []}],
}


def test_aliases_are_spellings_not_suffixed_forms_or_products():
    text = "Garanti BBVA'da kart. Garanti BBVA Mobil uygulaması. GarantiBBVA ve Garanti."
    reply = json.dumps(
        {
            "brand": "Garanti BBVA",
            "sector": "bankacılık",
            "aliases": ["Garanti BBVA'da", "Garanti BBVA Mobil", "GarantiBBVA", "Garanti"],
        }
    )
    assert profile.parse_profile(reply, text)["aliases"] == ["GarantiBBVA", "Garanti"]


def test_discovery_questions_must_ask_for_a_choice_between_companies():
    reply = json.dumps(
        {
            "discovery": [
                "Ev sahibi olmak için ne tür finansman seçeneği daha avantajlı?",
                "Konut kredisi için hangi banka daha avantajlı?",
            ]
        }
    )
    planned = profile.parse_queries(reply, PROFILE)
    assert planned["discovery"] == ["Konut kredisi için hangi banka daha avantajlı?"]


def test_discovery_questions_never_name_the_brand_and_named_ones_must():
    reply = json.dumps(
        {
            "discovery": [
                "Aidatsız kredi kartı önerir misin?",
                "Garanti mi daha iyi yoksa diğer bankalar mı?",
                "Bonus kart alınır mı?",
                "En uygun ihtiyaç kredisi hangi bankada?",
            ],
            "named": ["Garanti BBVA Bonus kart mı Axess mi?", "Hangi kart daha iyi?"],
        }
    )
    planned = profile.parse_queries(reply, PROFILE, n_discovery=3, n_named=2)
    assert planned["discovery"] == [
        "Aidatsız kredi kartı önerir misin?",
        "En uygun ihtiyaç kredisi hangi bankada?",
    ]
    assert planned["named"] == ["Garanti BBVA Bonus kart mı Axess mi?"]
    with pytest.raises(ValueError):
        profile.parse_queries(json.dumps({"discovery": ["Garanti iyi mi?"]}), PROFILE)


class _Receipts:
    def __init__(self, replies: dict[str, dict]):
        self.replies = replies
        self.calls: list[tuple[str, str]] = []

    async def call(self, key, service_name, payload, validator=None):
        self.calls.append((key, service_name))
        prefix = next(p for p in self.replies if key.startswith(p))
        result = self.replies[prefix]
        if validator:
            validator(result)
        return result


def test_build_profile_from_a_name_searches_then_reads_the_brands_own_site(monkeypatch):
    read = []

    async def fake_read_site(url, http, **kwargs):
        read.append(url)
        return site.Site(url=url, pages=[site.Page(url, "Garanti BBVA", "", [], SITE_TEXT)])

    monkeypatch.setattr(site, "read_site", fake_read_site)
    receipts = _Receipts(
        {
            "profile_search_": {
                "organic": [
                    {
                        "title": "Karşılaştırma",
                        "snippet": "…",
                        "link": "https://www.hangikredi.com/",
                    },
                    {
                        "title": "Garanti BBVA",
                        "snippet": "…",
                        "link": "https://www.garantibbva.com.tr/",
                    },
                ]
            },
            "profile_": {
                "text": json.dumps(
                    {"brand": "Garanti BBVA", "sector": "bankacılık", "products": []}
                )
            },
        }
    )
    found = asyncio.run(profile.build_profile("Garanti BBVA", receipts, None))  # type: ignore[arg-type]
    assert read == ["https://www.garantibbva.com.tr/"]
    assert found["website"] == "https://www.garantibbva.com.tr/" and found["source"] == "site"
    assert [service for _, service in receipts.calls] == ["serper", "gemini"]


def test_profile_questions_change_the_run_but_an_ordinary_run_keeps_its_folder():
    base = service.Options(
        brand="Garanti BBVA", sector="bankacılık", language="tr", brand_aliases=("Garanti",)
    )
    assert service.run_id(base) == "bbfd57540f7d01b3"
    custom = service.Options(
        brand="Garanti BBVA",
        sector="bankacılık",
        language="tr",
        brand_aliases=("Garanti",),
        custom_queries=("Aidatsız kredi kartı önerir misin?",),
        named_queries=("Garanti BBVA Bonus kart mı Axess mi?",),
    )
    assert service.run_id(custom) != service.run_id(base)
    assert service.estimated_calls(custom) == 0 + 2 + 1 + 2 * 2 * 2 + 0


def test_plan_takes_profile_questions_without_a_paid_call():
    runtime = nodes.Runtime(
        receipts=_Receipts({}),  # type: ignore[arg-type]
        custom_queries=("Aidatsız kredi kartı önerir misin?",),
        named_queries=("Garanti BBVA Bonus kart mı Axess mi?",),
    )
    state = {"brand": "Garanti BBVA", "sector": "bankacılık", "language": "tr", "max_calls": 10}
    out = asyncio.run(nodes.make_plan(runtime, 10)(state))  # type: ignore[arg-type]
    assert out["queries"] == ["Aidatsız kredi kartı önerir misin?"]
    assert out["named_queries"] == ["Garanti BBVA Bonus kart mı Axess mi?"]
    assert runtime.spent == []


class _Answers:
    def __init__(self):
        self.keys: list[str] = []

    async def call(self, key, service_name, payload, validator=None):
        self.keys.append(key)
        if key.startswith("named"):
            return {"text": "İkisi de iyi ama size **Akbank Axess** öneririm. Garanti BBVA da var."}
        return {"text": "Garanti BBVA iyi bir seçenek, Akbank da var."}


def test_named_questions_are_asked_under_their_own_keys_and_kept_out_of_visibility():
    registry = build_registry("Garanti BBVA", ["Garanti"], ["Akbank"], "bankacılık")
    runtime = nodes.Runtime(receipts=_Answers(), reps=1)  # type: ignore[arg-type]
    runtime.registry = registry
    results = [
        {"title": "Garanti BBVA ve Akbank", "snippet": "Kartlar.", "link": "https://x.com/a"}
    ]
    state = {
        "brand": "Garanti BBVA",
        "sector": "bankacılık",
        "language": "tr",
        "max_calls": 20,
        "queries": ["Aidatsız kart?"],
        "named_queries": ["Garanti BBVA Bonus mı Axess mi?"],
        "search": [
            {"query": "Aidatsız kart?", "results": results},
            {"query": "Garanti BBVA Bonus mı Axess mi?", "results": results},
        ],
    }
    out = asyncio.run(nodes.make_interrogate(runtime, 20)(state))  # type: ignore[arg-type]
    assert set(runtime.receipts.keys) == {  # type: ignore[attr-defined]
        "ask_0_search_off_r0",
        "ask_0_search_on_r0",
        "named_0_search_off_r0",
        "named_0_search_on_r0",
    }
    named = [o for o in out["observations"] if o["kind"] == "named"]
    assert named and all(o["picked"] == "Akbank" for o in named)
    assert all(o["answer"] for o in out["observations"])

    rates = nodes.named_rates(out["observations"], "Garanti BBVA")
    assert rates["gemini"]["picked"] == 0.0 and rates["gemini"]["rivals_picked"] == [("Akbank", 2)]
    discovery = [o for o in out["observations"] if o["kind"] == "discovery"]
    assert measure.rates(discovery)["n_observations"] == 2


def test_marked_sentences_carry_their_types_and_strongest_verdict():
    text = "Yıllık aidat yok. Türkiye'nin en iyi kartı. Harvard çalışmasında %98 kanıtlanmıştır."
    record = audit.as_record(
        audit.audit(
            text,
            found={"price": ["Yıllık aidat yok."], "superlative": ["Türkiye'nin en iyi kartı."]},
            method="ai",
        )
    )
    marks = audit.marked(text, record)
    assert [m["verdict"] for m in marks] == ["strong", "weak", None]
    assert marks[0]["labels"] == ["Fiyat avantajı"]


def test_web_api_serves_the_page_the_audit_and_an_estimate():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient  # pyright: ignore[reportMissingImports]

    from advisor.web import app as web

    client = TestClient(web.app)
    page = client.get("/")
    assert page.status_code == 200 and "kısaliste" in page.text
    audited = client.post("/api/audit", json={"text": "WireGuard ve AES-256 şifreleme kullanır."})
    body = audited.json()
    assert audited.status_code == 200 and body["record"]["method"] == "rules"
    assert body["marks"][0]["keys"] == ["technical"]
    plan = {
        "brand": "Garanti BBVA",
        "sector": "bankacılık",
        "aliases": ["Garanti"],
        "discovery": ["Aidatsız kredi kartı önerir misin?"],
        "named": [],
    }
    estimate = client.post("/api/estimate", json=plan).json()
    assert estimate["calls"] == 1 + 1 + 1 * 2 * 2 + 1
    assert client.get("/api/runs/yok").status_code == 404
    assert len(client.get("/api/evidence").json()["types"]) == len(audit.WINS)


def test_answer_tables_become_tables_even_with_blank_lines_between_rows():
    from advisor.web import pdf

    answer = (
        "İşte öneriler:\n\n| Marka / Ürün | Özellikleri |\n\n|--------------|-------------|\n"
        "| **Asperox** Sarı Güç | Yağı söker<br>Limon kokulu |\n| Cif | <script>x</script> |\n"
    )
    html = pdf.markdown(answer, own=["Asperox"], rivals=["Cif"])
    assert html.count("<table") == 1 and "|" not in html and "---" not in html
    assert "<th>Marka / Ürün</th>" in html and html.count("<tr>") == 3
    assert '<strong><mark class="own">Asperox</mark></strong> Sarı Güç' in html
    assert "Yağı söker<br>Limon kokulu" in html and "&lt;script&gt;" in html


def test_brand_marks_fold_turkish_capitals_without_shifting_the_text():
    from advisor.web import pdf

    assert pdf.marked("İŞ BANKASI ve Işık", own=["iş bankası"]) == (
        '<mark class="own">İŞ BANKASI</mark> ve Işık'
    )


def test_pdf_report_carries_every_section_and_escapes_answers():
    from datetime import date

    from advisor.web import pdf

    result = {
        "brand": "Asperox",
        "sector": "temizlik ürünleri",
        "language": "tr",
        "curated": False,
        "diagnosis_title": "Aramada görünmüyorsun",
        "diagnosis_note": "Sorun **bulunabilirlik**.",
        "measures": {"retrieval_presence": 0.0, "mention_on": 0.25, "first_on": 0.0},
        "scores": {"rank": 1, "candidates": 7, "compared_with": ["Cif"]},
        "recommendations": [
            {
                "title": "Karşılaştırma sayfalarına gir",
                "why": "w",
                "action": "a",
                "verdict": "v",
                "targets": [
                    {
                        "domain": "ornek.com",
                        "link": "https://ornek.com/x",
                        "rivals": ["Cif"],
                        "position": 2,
                    }
                ],
            }
        ],
        "answers": [
            {
                "query": "Hangi yağ çözücü?",
                "assistant": "gemini",
                "kind": "discovery",
                "mentioned": True,
                "first": False,
                "named_brands": ["Cif", "Asperox"],
                "answer": "| A | B |\n|---|---|\n| Cif | <b>kalın</b> |",
            }
        ],
        "assistant_labels": {"gemini": "Gemini 3.5 Flash Lite"},
        "notes": ["40 asistan yanıtı ölçüldü."],
        "candidates": ["Asperox", "Cif"],
    }
    html = pdf.report_html(result, aliases=["Asperox Sarı Güç"], made=date(2026, 9, 15))
    for text in (
        "Aramada görünmüyorsun",
        "Listeye girmek için",
        "Yanıtlarda kimler var",
        "Ek: Asistanlar ne dedi",
        "Bu rapor ne söylemiyor",
        "15 Eylül 2026",
        'href="https://ornek.com/x"',
        "<strong>bulunabilirlik</strong>",
        "Seni andı",
    ):
        assert text in html
    assert "<b>kalın</b>" not in html and "&lt;b&gt;kalın&lt;/b&gt;" in html


def test_downloads_are_named_after_the_brand_whatever_its_letters():
    pytest.importorskip("fastapi")
    from advisor.web import app as web

    header = web.attachment("Türk Telekom görünürlük raporu.pdf")["Content-Disposition"]
    header.encode("latin-1")
    assert 'filename="Turk-Telekom-gorunurluk-raporu.pdf"' in header
    assert "filename*=UTF-8''T%C3%BCrk%20Telekom" in header


def test_a_kept_site_is_read_from_the_cache_until_a_reread_is_asked_for(tmp_path, monkeypatch):
    served = ["<html><body><p>Sarı Güç yağı söker.</p></body></html>"]
    fetched = []

    async def fake_fetch(http, url):
        fetched.append(url)
        return served[-1], url

    monkeypatch.setattr(site, "fetch_html", fake_fetch)
    url = "https://www.asperox.com.tr/"
    first = asyncio.run(site.read_site(url, None, cache=tmp_path))  # type: ignore[arg-type]
    again = asyncio.run(site.read_site(url, None, cache=tmp_path))  # type: ignore[arg-type]
    assert len(fetched) == 1 and again.fetched_at == first.fetched_at and first.changed is None
    same = asyncio.run(site.read_site(url, None, cache=tmp_path, refresh=True))  # type: ignore[arg-type]
    assert len(fetched) == 2 and same.changed is False
    served.append("<html><body><p>Yeni formül limon kokulu.</p></body></html>")
    moved = asyncio.run(site.read_site(url, None, cache=tmp_path, refresh=True))  # type: ignore[arg-type]
    assert moved.changed is True and "limon" in moved.text()
    kept = asyncio.run(site.read_site(url, None, cache=tmp_path))  # type: ignore[arg-type]
    assert len(fetched) == 3 and "limon" in kept.text()


def test_a_repeated_search_sets_the_old_receipt_aside_instead_of_deleting_it(tmp_path):
    from brand_demo.workflow import Receipts

    answers = [
        {"organic": [{"title": "Asperox", "snippet": "yağ çözücü", "link": "https://x.com/"}]}
    ]
    calls = []

    async def client(service_name, payload):
        calls.append(service_name)
        return answers[-1]

    receipts = Receipts(tmp_path, client, retry_failed=False)
    first = asyncio.run(profile.source_text("Asperox", receipts, None))  # type: ignore[arg-type]
    asyncio.run(profile.source_text("Asperox", receipts, None))  # type: ignore[arg-type]
    assert calls == ["serper"] and first["changed"] is None
    same = asyncio.run(profile.source_text("Asperox", receipts, None, refresh=True))  # type: ignore[arg-type]
    assert calls == ["serper", "serper"] and same["changed"] is False
    assert len(list((tmp_path / "replaced").glob("profile_search_*.json"))) == 1
    answers.append(
        {"organic": [{"title": "Asperox", "snippet": "yeni ürün", "link": "https://x.com/"}]}
    )
    moved = asyncio.run(profile.source_text("Asperox", receipts, None, refresh=True))  # type: ignore[arg-type]
    assert moved["changed"] is True and "yeni ürün" in moved["text"]
