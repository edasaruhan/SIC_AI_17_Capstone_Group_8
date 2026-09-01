# Toplama Boru Hattı (S1-6, S1-7, S2-3)

Türkçe sorguları üç asistana soran, yanıtları ham JSONL olarak kaydeden ve katı
çıktı formatını ayrıştıran boru hattı. Sorumlu: Kişi 1.

- Kod: [`src/collect/`](../src/collect), [`src/parse/`](../src/parse)
- Ayarlar: [`configs/collect.yaml`](../configs/collect.yaml)
- Testler: `tests/test_collect_*.py`, `tests/test_parse_protocol.py`

## 1. Ne yapıyor

```
configs/queries_tr.yaml ─┐
configs/brands_tr.yaml  ─┼─► plan.py ──► CallSpec listesi ──► runner.py ──► data/raw/<run-id>.jsonl
configs/protocols.yaml  ─┤                (call_id ile)         (httpx +          (append-only)
configs/variants.yaml   ─┘                                       tenacity)              │
                                                                                        ▼
                                                              parse/protocol.py ──► ParseResult
                                                                                        │
                                                              collect/monitor.py ──► günlük tablo
```

`plan` ve `status` hiçbir çağrı yapmaz, her an güvenle çalıştırılabilir. `run`
para harcar.

## 2. Tasarım dosyalarından beklenen şema

S1-1 – S1-4 çıktıları aşağıdaki biçimde olmalı. Bu, boru hattının okuyabildiği
tek şekildir; farklı bir yapı tercih edilirse `src/collect/plan.py` güncellenir.

**`configs/queries_tr.yaml`**

```yaml
version: 1
sectors:
  - id: kozmetik
    label: Kozmetik ve kişisel bakım
    queries:
      - {id: kozmetik_01, text: "En iyi nemlendirici hangisi?", intent: genel_oneri}
```

**`configs/brands_tr.yaml`** — `type` yalnızca şu dördünden biri olabilir:
`kuresel`, `yerel`, `kucuk_yerel`, `kurgusal`.

```yaml
version: 1
sectors:
  - id: kozmetik
    brands:
      - {id: koz_m01, name: "Marka Adı", type: kuresel}
```

**`configs/protocols.yaml`** — persona ve sıcaklık burada sabitlenir (Kural 5).
Şablonlar `str.format` ile doldurulur.

```yaml
version: 1
persona: "Sen Türkçe konuşan bir alışveriş danışmanısın."
temperature: 0.0
repetitions: 20
candidate_list_size: 8
protocols:
  alpha:
    layer: A
    marker: SIRALAMA
    template: |
      Soru: {query}
      Adaylar:
      {candidate_block}
      Yalnızca son satırda {marker}: [{candidate_labels}] biçiminde sırala.
```

Şablonda kullanılabilen alanlar:

| Katman | Alanlar |
|---|---|
| A (`alpha`, `beta`) | `{query}`, `{intent}`, `{sector}`, `{marker}`, `{candidate_block}`, `{candidate_labels}` |
| B (`gamma`) | `{brand}`, `{sector}`, `{marker}`, `{control_text}`, `{variant_text}`, `{option_a}`, `{option_b}` |

**`configs/variants.yaml`** (Katman B)

```yaml
version: 1
brands:
  - brand_id: koz_m01
    control: "Dokunulmamış kontrol açıklaması."
    variants:
      - {id: v01, text: "Sayısal kanıt içeren varyant."}
```

## 3. Kodda zorunlu kılınan kurallar

Aşağıdakiler kişinin dikkatine bırakılmadı; plan üretilirken uygulanıyor ve
testle korunuyor.

| Kural | Nerede | Test |
|---|---|---|
| 1 · Tek çağrıda 8 aday marka | `candidate_list_size` | `test_layer_a_plan_has_the_expected_shape` |
| 2 · Her tekrarda sıra karıştırılır | `_cell_rng` + `rotate_candidates` | `test_presentation_order_is_reshuffled_across_repetitions` |
| 3 · Aynı markanın iki varyantı aynı listede olmaz | `CallSpec.__post_init__` | `test_layer_a_plan_has_the_expected_shape` |
| 4 · Model sürümü ve zaman damgası her satırda | `CallRecord` | `test_successful_call_records_version_cost_and_timestamps` |
| 5 · Persona ve sıcaklık sabit | `protocols.yaml` → `DesignConfig` | `test_layer_a_plan_has_the_expected_shape` |
| 6 · Ham veri değiştirilmez | `RawLogWriter` yalnızca `"a"` kipinde açar | `test_an_interrupted_run_resumes_where_it_stopped` |

Marka kapsaması `rotate_candidates` ile dönerek dengelenir: 12 markadan 8'i her
çağrıda gösterildiğinde rastgele örnekleme bazı markaları az ölçerdi ve bu,
sonradan marka etkisi gibi görünen bir örnekleme yanlılığı üretirdi.

## 4. Ham satır şeması

`data/raw/<run-id>.jsonl` içindeki her satır bir çağrıdır. S2-4 (çift üretimi)
bu alanları okur.

| Alan | Tip | Açıklama |
|---|---|---|
| `call_id` | str | Çağrının kimliği; tasarım değişirse değişir |
| `run_id` | str | Koşu kimliği, dosya adıyla aynı |
| `layer`, `protocol` | str | `A`/`B`, `alpha`/`beta`/`gamma` |
| `sector`, `query_id` | str | Sorgu havuzundan |
| `condition` | str | `search_on` / `search_off` |
| `model_key`, `provider`, `model_requested` | str | Yapılandırmadaki model |
| `model_version` | str | **Sağlayıcının döndürdüğü gerçek sürüm** (Kural 4) |
| `repetition` | int | 0'dan başlar |
| `persona`, `temperature` | str, float | Sabit tutulan değişkenler |
| `prompt` | str | Gönderilen istemin tamamı |
| `candidates` | list[str] | **Gösterim sırasıyla** marka kimlikleri; `A` = `candidates[0]` |
| `variant_ids` | list[str] | Katman B'de gösterim sırasıyla `control` ve varyant |
| `status` | str | `ok` / `error` |
| `response_text` | str \| null | Yanıt metni |
| `search_results` | list | Normalleştirilmiş `{url, title, snippet}` |
| `stop_reason` | str \| null | Sağlayıcının durma nedeni |
| `usage` | dict | `input_tokens`, `output_tokens`, `search_requests` |
| `cost_usd` | float \| null | Bu çağrının maliyeti |
| `attempts`, `latency_ms` | int | Yeniden deneme sayısı ve süre |
| `started_at`, `completed_at` | str | ISO 8601, UTC |
| `error_type`, `error_message` | str \| null | Başarısız çağrılarda |
| `raw_response` | dict \| null | Sağlayıcının ham gövdesi |
| `schema_version` | int | Şu an `1` |

Dosya **yalnızca eklenir**. Başarısız bir çağrı yeniden denendiğinde aynı
`call_id` için ikinci bir satır yazılır; işleme aşaması `status == "ok"` olanı
alır.

## 5. Koşuyu çalıştırma

```bash
cp .env.example .env          # ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY doldurulur
                              # .env.example şablondur ve boş kalır; anahtarlar .env'e yazılır
make collect-plan RUN_ID=pilot-01 ARGS="--limit 20 --sample"   # önce plana bakılır
make collect-run  RUN_ID=pilot-01 ARGS="--limit 20"            # 20 çağrılık deneme
make collect-run  RUN_ID=katman-a-01                           # tam koşu
```

`make` yoksa (Windows) doğrudan:

```powershell
$env:PYTHONPATH="src"; uv run --env-file .env python -m collect plan --run-id pilot-01 --limit 20
```

**Kesinti.** Ctrl+C ile durdurulabilir; biten her çağrı zaten diskte olduğu için
aynı `--run-id` ile yeniden başlatmak kaldığı yerden devam eder. Yeni bir
`--run-id` yeni bir koşudur ve her şeyi baştan toplar.

**Harcama tavanı.** Günlük tavana ulaşılınca yeni çağrı planlanmaz, uçuştakiler
tamamlanır ve koşu temiz biter. Ertesi gün aynı `--run-id` ile sürdürülür.

## 6. Günlük izleme (S2-3)

```bash
make collect-monitor RUN_ID=katman-a-01                  # tablo
make collect-monitor RUN_ID=katman-a-01 ARGS="--format line"   # gruba yazılacak tek satır
```

Tek satır özet nöbetçinin kontrol listesindeki dört soruyu da yanıtlar: çağrı
sayısı, hata oranı, ayrıştırma oranı ve günlük harcama. Tavan aşıldıysa satırda
uyarı çıkar ve komut sıfırdan farklı bir kod döner.

Tabloda **hata oranı** ile **ayrıştırma oranı** ayrı tutulur: birincisi çağrının
başarısız olması, ikincisi yanıtın gelip katı formata uymamasıdır. S1-8 pilotunda
ölçülecek "ayrıştırma başarısı, model bazında" değeri ikincisidir.

## 7. Ayrıştırma (S1-7)

`parse_response`, protokolün işaretini (`SIRALAMA:`, `ONERI:`, `SECIM:`) yanıtın
son satırından geriye doğru arar ve etiketleri marka kimliklerine çevirir.

Okunamayan yanıt **sessizce atlanmaz**; `status="failed"` ve bir gerekçe ile
döner: `no_response_text`, `marker_missing`, `empty_value`, `unknown_label`,
`duplicate_label`, `incomplete_ranking`, `expected_single_value`.

Türkçe tuzağı: `"SIRALAMA".lower()` Türkçede `"sıralama"` vermez. Bu yüzden
işaret eşleştirmesi `parse.turkish.fold` ile yapılır; `SIRALAMA`, `Sıralama`,
`sıralama` ve `SİRALAMA` aynı kabul edilir. Modeller son satırı sık sık
kalınlaştırdığı için `**SIRALAMA:**` biçimi de kabul edilir.

## 8. Bilinen sınırlar

- `configs/collect.yaml` içindeki model kimlikleri pilot öncesi sağlayıcı
  belgelerinden doğrulanmalı. Yanlış kimlik 400 döndürür ve yeniden denenmez;
  `attempts=1` ve `FatalProviderError` olarak kaydedilir.
- Katman B karşılaştırması varyantı her zaman kontrol açıklamasıyla eşler.
  Varyantların birbiriyle karşılaştırılması gerekirse `build_layer_b_specs`
  genişletilmeli.
- Sağlayıcı SDK'ları yerine ortak bir httpx katmanı kullanılıyor. Üç asistana
  aynı hız sınırlama ve yeniden deneme davranışını uygulamak ölçüm tutarlılığı
  için gerekli; SDK'ların kendi retry mantıkları bunu bozardı.
