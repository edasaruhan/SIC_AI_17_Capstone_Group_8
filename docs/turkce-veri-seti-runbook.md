# Referansla Uyumlu Türkçe Veri Seti Runbook'u

Bu altyapı, `3RAIN/brand-bias-evaluations` deney yapısını izleyerek Türkçe marka
önerisi yanıtları toplar. Repository'ye gerçek veri veya anahtar eklemez. İlk API
isteği ancak siz `dataset-preflight`, `dataset-search-test`, `dataset-pilot`,
`dataset-collect` veya `dataset-judge` hedeflerinden birini çalıştırdığınızda yapılır.

## Sabit deney tasarımı

```text
2 domain × 5 sorgu × 3 model × 2 koşul × 5 tekrar = 300 yanıt
```

| Boyut | Değer |
|---|---|
| Domain | `vpn`, `cosmetics` |
| Dil | Türkçe (`tr`) |
| Koşul | `search_off`, `search_on` |
| Tekrar | 5 |
| Sıcaklık | 0.7 |
| Maksimum yanıt tokenı | 1024 |
| Maksimum arama turu | 5 |
| Search | Serper Google Search API, Türkiye/Türkçe, 10 organik sonuç |
| Judge | Cerebras `gpt-oss-120b`, sıcaklık 0, reasoning `low` |

Her domain 150, her model 100, her koşul 150 ve her sorgu 30 satır üretir.
`bias-eval plan` bu dağılımları ağ bağlantısı veya anahtar olmadan doğrular.

Generation sağlayıcıları tek bir OpenAI uyumlu adapter kullanır; model ve endpoint
eşlemeleri [`models.yaml`](../configs/evaluation/models.yaml) dosyasında sabittir.

| Sağlayıcı | Model kimliği | Base URL | Ortam değişkeni |
|---|---|---|---|
| Google | `gemini-3.5-flash-lite` | `https://generativelanguage.googleapis.com/v1beta/openai` | `GEMINI_API_KEY` |
| MiniMax | `MiniMax-M2.7` | `https://api.minimax.io/v1` | `MINIMAX_API_KEY` |
| Abliteration | `abliterated-model-large-v2` | `https://api.abliteration.ai/v1` | `ABLITERATION_API_KEY` |

Sağlayıcı model kataloğu değişirse model kimliğini sessizce başka bir modele
çevirmeyin. Preflight hatasını kaydedin ve deney yapılandırmasını ekip kararıyla
sürümleyin. İlgili sağlayıcı belgeleri: [Gemini](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite),
[MiniMax](https://platform.minimax.io/docs/guides/text-generation),
[Abliteration](https://docs.abliteration.ai/api/openai-compatibility).

NVIDIA modelleri preview alias'ında HTTP 410, Pro sürümünde 502 ve Flash sürümünde
uzun timeout sorunları verdiği için 4 Eylül 2026 tarihinde deney tasarımından
çıkarılmıştır. NVIDIA'ya ait güncel ve arşivlenmiş ham kayıtlar silinmiş, plan 300
satırlık üç model tasarımına geçirilmiştir. Judge yalnız bu güncel 300 kaydı işler.

Google kolu düşük gecikmeli ve maliyet odaklı kararlı
`gemini-3.5-flash-lite` modelini kullanır. Eski `gemini-3.5-flash` kayıtları yeni
modele aktarılmamıştır ve deney temizliği sırasında kalıcı olarak silinmiştir.
Flash Lite için 100 bağımsız hücre tamamlanmıştır.

## Kurulum ve anahtarlar

```bash
make setup
cp .env.example .env
```

`.env` içindeki beş boş alanı kendi anahtarlarınızla doldurun:

```dotenv
GEMINI_API_KEY=
MINIMAX_API_KEY=
ABLITERATION_API_KEY=
SERPER_API_KEY=
CEREBRAS_API_KEY=
```

`.env` Git tarafından izlenmez. Anahtarı komut satırında yazmayın, loglara
kopyalamayın ve gerçek `.env` dosyasını paylaşmayın. Pipeline Authorization ve API
key başlıklarını ham kayıtlara yazmaz; hata yanıtlarının yalnız temizlenmiş kısa
mesajını tutar.

## Loglama ve canlı hata takibi

Her `bias-eval` komutu Loguru ile hem konsola hem `logs/bias-eval.log` dosyasına
yazar. Dosya logu `DEBUG` seviyesindedir; hücre/model kimliği, araç turu, cache
hit/miss, HTTP durum kodu, istek süresi, retry beklemesi, token sayısı ve devre
kesici nedenini içerir. Beklenmeyen uygulama hatalarında stack trace de dosyaya
yazılır. Ham request/response gövdeleri, kullanıcı yanıtları, API anahtarları ve
Authorization başlıkları loglanmaz. Ortam değişkenlerindeki key/token/secret
değerleri ayrıca yazılmadan önce `[REDACTED]` ile değiştirilir.

```bash
make dataset-logs
make dataset-follow-logs
```

İkinci komut canlı takipte kalır; `Ctrl-C` yalnız log takibini kapatır. Dosya 10 MB
olunca döner ve son beş arşiv tutulur. Farklı dosya/seviye için
`BIAS_EVAL_LOG_FILE` ve `BIAS_EVAL_LOG_LEVEL` kullanılabilir.

Ücretsiz kalmak için çalıştırmadan önce tüm sağlayıcı panellerinde otomatik ödeme
ve bakiye yüklemeyi kapatın; mümkünse en düşük harcama hard limitini ayarlayın.
Kod, aynı anahtar altındaki ücretsiz ve ücretli bakiyeyi güvenilir biçimde ayıramaz.
Kota biterse model değiştirilmez ve eksik satır tamamlanmış sayılmaz.

## Çalıştırma sırası

Tüm Türkçe veri setini tek komutla oluşturmak için:

```bash
make dataset-all
```

Bu hedef sırasıyla preflight, collect, generation status, judge, judge status,
export ve validation çalıştırır. `dataset-search-test` ile pilotu çağırmaz; böylece
final 300 hücrenin dışında kota tüketmez. Bir aşama sıfırdan farklı kodla biterse
sonraki aşamaya geçmez. Sorunu düzelttikten sonra aynı komutu yeniden çalıştırmak
tamamlanmış generation ve judgment kayıtlarını atlayarak eksiklerden devam eder.

Önce tamamen offline kontrolleri çalıştırın:

```bash
make check
make dataset-plan
```

`dataset-plan` çıktısında `total` ve `unique_record_ids` değerleri 300 olmalıdır.
Ardından anahtar/bağlantı kontrolü ve final plana ait 12 satırlık pilot gelir:

```bash
make dataset-preflight
make dataset-pilot
make dataset-status
```

Preflight sağlayıcıların model katalog endpoint'lerini kontrol eder; generation
yanıtı üretmez. Serper için yalnız anahtar varlığını kontrol eder. İsterseniz ayrıca
`make dataset-search-test` çalıştırabilirsiniz; bu komut **bir gerçek Serper sorgusu
ve bir ücretsiz kota kredisi** tüketir.

Pilot, iki domain × üç model × iki koşul matrisindeki ilk sorgu/ilk tekrar
hücrelerini üretir. Bunlar final 300 hücrenin parçasıdır ve collect aşamasında
yeniden çağrılmaz. Pilot başarılıysa kalan 288 hücreyi tamamlayın:

```bash
make dataset-collect
make dataset-status
make dataset-judge
make dataset-judge-status
make dataset-export
make dataset-validate
```

Generation 300/300 tamamlandıysa sonraki beş komutun kısa karşılığı:

```bash
make dataset-finish
```

Bu hedef status, Cerebras judge, judge status, Parquet export ve validation
aşamalarını sırayla çalıştırır. Generation endpoint'lerine istek göndermez.

`dataset-status` içinde `invalid` görülürse önce şu hedef çalıştırılır:

```bash
make dataset-repair-generation
```

Bu veri sürümünde boş cevapların tamamı GLM kolundadır. Hedef yalnız
`glm_5_3_abliteration` modelini açar, geçerli tamamlanmış GLM hücrelerini atlar ve
yalnız boş nihai cevapları yeniden üretir. Onarım çağrılarında Abliteration'ın
`thinking: false` seçeneği kullanılır; bu özel parametre ham kaydın
`request_parameters` alanına yazılır. Gemini ve MiniMax endpoint'lerine istek
göndermez. Boş veya yalnız boşluk içeren cevaplar artık `completed` sayılmaz.

Son olarak sabit İngilizce Hugging Face verisini hazırlayıp betimsel dil raporunu
oluşturun:

```bash
make reference-data
make dataset-report
```

Rapor `reports/tr_vpn_language_comparison.md` dosyasına yazılır. İngilizce tarafta
yalnız `vpn_01`, `vpn_02`, `vpn_03`, `vpn_04` ve `vpn_08` sorguları seçilir. Rapor
nedensel dil etkisi iddia etmez; model/tarih/sağlayıcı farklarını açık bir sınırlılık
olarak korur.

## Search davranışı

`search_off` yalnız kullanıcı sorgusunu yollar; sistem istemi ve araç tanımı yoktur.
Ham kayıtta `tool_calls` ile `search_results` boş kalır.

`search_on`, Türkçe sistem istemi ve `web_search` fonksiyonunu `tool_choice=auto`
ile sunar. Modelin ürettiği sorgu Serper'a `num=10`, `gl=tr`, `hl=tr`,
`location=Turkey` ile gönderilir. Başlık, snippet ve URL numaralı metin olarak modele
geri verilir. Beş ardışık araç turundan sonra hücre hata olur. Model hiç arama
yapmadan cevap verirse kayıt tamamlanır ve `quality_flags.search_not_used=true` olur.

Aynı sorgu/ülke/dil/konum/sonuç sayısı kombinasyonu `data/search_cache` altında
saklanır. Cache kaydı sorguyu, çekilme zamanını, locale'i ve ham Serper cevabını
içerir. Bu klasör Git tarafından izlenmez.

İlk generation koşusu ayrıca `data/raw/.collection-lock.json` oluşturur. Sorgu,
model, prompt, sıcaklık, locale veya diğer generation ayarları sonradan değişirse
aynı klasörde yeni API isteği yapılmadan hata verilir. Yeni deney sürümü yeni
`suite_id` ve yeni raw/judged/cache/processed yolları kullanmalıdır; ayrıntılar
[geliştirici rehberindedir](veri-seti-gelistirici-rehberi.md).

## Resume, retry ve dosyalar

Her hücre atomik bir JSON dosyasıdır:

```text
data/raw/<experiment>/<model>/<condition>/<query_id>_<run_index>.json
data/judged/<experiment>/<model>/<condition>/<query_id>_<run_index>.json
```

`completed` generation hücreleri yeniden çalıştırıldığında atlanır. Hatalı/bozuk
hücreler yeniden denenir. `408`, `429` ve `5xx` durumları `Retry-After` dikkate
alınarak en fazla üç denemeye tabidir. Yetki, bilinmeyen model ve bakiye hataları
kalıcı kaydedilir. Bir generation sağlayıcısının kotası tükenirse yalnız o model
durdurulur; diğer sağlayıcılar devam eder. Cerebras kota bitiminde judge kuyruğunun
kalanı tamamlanmış sayılmaz.

Yalnız yeni Gemini Flash Lite hücrelerini güvenli pilotla üretmek için:

```bash
make dataset-retry-gemini
```

Bu hedef önce model kataloğunu kontrol eder, dört final pilot hücreyi çalıştırır ve
pilot başarılıysa kalan 96 Gemini hücresine geçer. MiniMax ve Abliteration
generation endpoint'lerine istek göndermez.

Gemini 3 OpenAI uyumluluk katmanında araç yanıtı ikinci tura döndürülürken asistanın
`content: null` alanı gönderilmez ve ilk yanıttaki sağlayıcıya özgü tool-call bağlamı
(thought signature dahil) kayıpsız taşınır. Google'ın liste biçimindeki hata gövdesi
de anahtarları sızdırmadan ayrıştırılır; tekrar hata olursa gerçek kısa neden logda
görünür.

Judge kaydı generation içeriğinin SHA-256 değerini ve prompt/şema hash'ini taşır.
Yanıt, prompt veya şema değişirse kayıt bayat kabul edilir ve sonraki judge koşusunda
yenilenir. Deterministik özellikler LLM kullanılmadan hesaplanır.

Export çıktıları:

```text
data/processed/tr_brand_bias_v1/
  vpn/train.parquet
  cosmetics/train.parquet
  all/train.parquet
  manifest.json
  alias_candidates.json
```

Parquet, upstream'in 19 kolonunu aynı sırayla korur ve yalnız `language` kolonunu
ekler. `tool_calls`, `search_results`, `core`, `domain`, `search_aware` ve
`deterministic` JSON string olarak saklanır; `search_off` için `search_aware` gerçek
null değeridir. Ham sağlayıcı cevapları yalnız `data/raw`/`data/judged` altında kalır.

Markalar NFKC, Türkçe case-folding ve noktalama/boşluk standardizasyonundan geçer.
Bilinen VPN alias'ları birleştirilir; fuzzy yakın eşleşmeler otomatik birleştirilmez,
`alias_candidates.json` içinde `manual_review` olarak raporlanır.

## Kota bütçesi ve hata giderme

- En kötü Serper üst sınırı `150 search_on × 5 tur = 750` sorgudur. Cache tekrarları
  azaltabilir; bu sayı garanti edilen gerçek tüketim değildir.
- Abliteration ücretsiz/promosyon kredisi 100 satıra yetmezse ilgili model eksik kalır.
- Cerebras judge 300 başarılı generation kaydını ayrı ayrı işler ve deneme kredisine
  bağlıdır.
- Cerebras çağrıları tek işçiyle ve istek başlangıçları arasında en az 20 saniye
  bırakılarak yapılır. Sunucunun `Retry-After` değeri daha uzunsa o süre uygulanır.
- `search_on` hücresinde generation modeli arama aracını hiç kullanmamışsa ve judge
  `search_aware=null` döndürürse dört arama-atıf alanı deterministik olarak `false`
  ve kaynak listesi boş olarak kaydedilir; judge promptu ve şeması değiştirilmez.
- `dataset-preflight` sıfırdan farklı biterse toplama başlatmayın. Anahtar, model
  erişimi ve hesap kotasını sağlayıcı panelinden düzeltin.
- `dataset-status` hata gösteriyorsa aynı collect komutu güvenle yeniden çalıştırılır.
- `dataset-judge-status` içindeki `stale`, prompt/şema veya generation içeriği
  değiştiğini gösterir; judge komutu yalnız gerekenleri yeniden işler.
- `dataset-export` 300 güncel judgment yoksa çıktı üretmeden sıfırdan farklı biter.
- `dataset-validate`, 300 generation, 300 güncel judgment, beklenen dağılımlar,
  benzersiz kimlikler, JSON alanları ve 20 kolonlu Parquet şartlarından biri
  bozulursa sıfırdan farklı biter.

## Referans ve provenance

Pipeline tasarımı, geçici klonda şu upstream commit incelenerek uyarlanmıştır:

- GitHub: `3RAIN/brand-bias-evaluations`
- Commit: `cc42677a42bbbf92f6ef4c376abda6528f0463ea`
- Hugging Face revision: `400da04eced51d3afe52b6d20c0207fd613f8a4a`
- Lisans: MIT; tam metin
  [`brand-bias-evaluations-LICENSE`](references/brand-bias-evaluations-LICENSE)

Upstream `.git` dizini, ham veri ve analiz çıktıları bu projeye kopyalanmamıştır.
Manifest, iki revision değerini, config/query hash'lerini, dağılımları, eksik/hatalı
hücreleri, collection fingerprint'i, toplama zamanlarını, judge hash'ini ve her
Parquet SHA-256 değerini kaydeder.
