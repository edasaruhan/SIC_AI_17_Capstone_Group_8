# Veri Seti Geliştirici Rehberi

Bu belge, repoya yeni katılan bir contributor'ın Türkçe veri seti pipeline'ını
değiştirmeden önce bilmesi gereken sözleşmeyi özetler. Operasyonel komutların ve
hata çözümlerinin tamamı [runbook'ta](turkce-veri-seti-runbook.md) bulunur.

## Pipeline yaşam döngüsü

```text
configs/evaluation/*.yaml
          │
          ▼
       plan (offline)
          │
          ▼
 generation ── Serper cache
          │
          ▼
   raw JSON kayıtları
          │
          ▼
 Cerebras judge + deterministik özellikler
          │
          ▼
  judged JSON kayıtları
          │
          ▼
 export → Parquet + manifest → validate
```

Kaynak kod sorumlulukları:

| Dosya/modül | Sorumluluk |
|---|---|
| `configs/evaluation/suite.yaml` | Deney matrisi ve ortak üretim ayarları |
| `models.yaml` | Model kimliği, endpoint ve anahtar ortam değişkeni |
| `queries_*.yaml` | Sürümlenen kullanıcı sorguları |
| `records.py` | Hücre planı ve kararlı `record_id` üretimi |
| `runner.py` | Resume, retry ve generation/tool döngüsü |
| `search.py` | Serper isteği ve locale-duyarlı cache |
| `judge.py`, `judge_schema.py` | Cerebras extraction ve şema sözleşmesi |
| `exporter.py` | 20 kolonlu Parquet ve provenance manifesti |
| `validation.py` | Eksik, bayat veya karışmış release'i reddeden kontroller |

## Değişmez veri sözleşmesi

Aktif `tr_brand_bias_v1` sürümü tam olarak şunu üretir:

```text
2 domain × 5 sorgu × 3 model × 2 koşul × 5 tekrar = 300 satır
```

- Her `record_id` benzersizdir.
- `search_off` satırlarında araç izi yoktur ve `search_aware` null'dır.
- `search_on` satırlarında judge tarafından doldurulmuş `search_aware` bulunur.
- JSON içeren Parquet kolonları geçerli JSON string olmalıdır.
- Ham sağlayıcı gövdeleri Parquet'e girmez.
- Export; 150 VPN, 150 kozmetik ve model başına 100 satır içermelidir.
- Tam 300 generation ve 300 güncel judgment olmadan release yayınlanamaz.

Bu kuralları değiştiren çalışma yeni bir veri seti sürümüdür; mevcut `v1` adıyla
sessizce yayınlanmaz.

## Collection lock ve yeni sürüm açma

İlk generation komutu `data/raw/.collection-lock.json` oluşturur. Bu dosya model,
sorgu, sistem istemi, sıcaklık, arama locale'i ve diğer generation ayarlarının
fingerprint'ini taşır. Aynı ham veri klasöründe bu ayarlardan biri değişirse pipeline
API çağrısından **önce** durur. Böylece eski ve yeni koşullar aynı release içinde
karışmaz.

Deney tasarımı değişecekse:

1. `suite_id` değerini artırın; örneğin `tr_brand_bias_v2`.
2. `raw_dir`, `judged_dir`, `search_cache_dir` ve `processed_dir` için yeni,
   sürüme özel yollar kullanın.
3. Eski veri klasörlerini silmeyin veya yeni sürüme kopyalamayın.
4. `make dataset-plan` çıktısını ve dağılımları güncelleyin.
5. Beklenen satır sayılarını kod/test/dokümantasyonda birlikte güncelleyin.
6. Judge promptu ya da şeması değiştiyse prompt sürümünü artırın.
7. `make check` ile offline kontrolleri tamamlayın.
8. Önce pilot, sonra tam toplama çalıştırın.

Aynı sorgularla daha güncel web sonuçları toplamak da yeni bir release sayılmalıdır.
Yeni `search_cache_dir` kullanılmazsa eski Serper cache cevapları bilinçli olarak
yeniden kullanılabilir.

## Güvenli geliştirme döngüsü

API harcamadan önce:

```bash
make setup
make format
make check
make dataset-plan
```

Testlerde gerçek URL'lere çıkılmaz; tüm provider, Serper ve Cerebras cevapları mock
transport veya fixture ile sağlanır. Bir adapter değişikliğinde en az şu davranışlar
korunmalıdır:

- Kimlik doğrulama değerleri loga veya hata metnine girmemeli.
- `429`, `408` ve `5xx` retry sınırını aşmamalı.
- Kalıcı hata yalnız ilgili sağlayıcının devresini açmalı.
- Tamamlanan hücreler yeniden çağrılmamalı.
- Atomik yazma yarım JSON bırakmamalı.
- Provider'a özgü alanlar yalnız ilgili provider'a gönderilmeli.

Gerçek pilot çalıştırmak ayrıca yetki ve kota gerektirir:

```bash
make dataset-preflight
make dataset-pilot
make dataset-status
```

## Release kabul kriterleri

Bir veri release'i ancak aşağıdakilerin tümü sağlandığında hazırdır:

- `make check` başarılı.
- `make dataset-status`: 300/300, sıfır hata.
- `make dataset-judge-status`: 300 current, sıfır stale/error.
- `make dataset-export` başarılı.
- `make dataset-validate`: `ok: true`.
- Manifestte collection fingerprint, config/query hash'leri ve üç Parquet hash'i var.
- `alias_candidates.json` manuel gözden geçirilmiş.

Pipeline planı ve provenance deterministiktir; dış LLM'ler ve web sonuçları nedeniyle
üretilen doğal dil yanıtlarının byte düzeyinde aynı olması garanti edilmez.
