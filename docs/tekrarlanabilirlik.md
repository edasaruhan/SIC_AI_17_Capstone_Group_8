# Temiz klondan tekrarlanabilirlik

Bu belge, depoyu ilk kez klonlayan birinin hangi sonucu hangi komutla yeniden
üretebileceğini söyler: hangi adım ağ ister, hangisi anahtar ister, hangisi GPU
ister ve hangisi hiçbirini istemez.

**Son doğrulama: 12 Eylül 2026**, commit `67c1ea7`, ayrı bir dizine yapılmış temiz
klon üzerinde.

## Kurulum ve testler — ağ dışında hiçbir şey gerekmez

```bash
git clone <depo> && cd <depo>
uv sync --group dev --locked      # veya: make setup
.venv/bin/pytest -q               # veya: make check
```

Doğrulandı: **252 test geçti**, ~10 saniye. Anahtar, GPU veya indirilmiş veri
gerekmez; testler sentetik tablolarla çalışır. `make check` ayrıca ruff, black ve
pyright çalıştırır.

> 12 Eylül'den önce bu adım temiz klonda kırıktı: `configs/modeling/source_domains_v2.json`
> ve `src/visibility/` commit edilmemişti, bu yüzden kanıt hattı klonda hiç başlamıyordu.
> Bu dosyalar artık depodadır.

## Veri hazırlığı — ağ gerekir, anahtar gerekmez

```bash
make modeling-data       # reference-data + turkish-data + scripts/build_pairs.py
```

İki yayımlanmış veri setini sabitlenmiş revizyonlarından indirir ve içerik
hash'lerine karşı doğrular; uyuşmazlıkta durur:

| Kaynak | Sürüm kapısı |
|---|---|
| `3RAIN/brand-bias-evaluations` (İngilizce) | `EN_REVISION`, içerik SHA-256 |
| `furkankarli/turkish-brand-bias-evaluations` (Türkçe) | `REVISION`, `SOURCE_SHA256`, `CONTENT_SHA256` |

Üretir: `data/interim/{reference,turkish_raw}.parquet` ve
`data/processed/modeling/{pairs,evidence,splits}_{en,tr}`. Sorgu fold'ları
(`splits_*.json`) depoda sürümlüdür, yeniden bölünmez.

**Bu adım atlanırsa** sonraki komutlar `FileNotFoundError: data/interim/...` verir.
Bu beklenen davranıştır; eksik olan kod değil veridir.

**Ağ yoksa:** `data/interim/*.parquet` elinizde varsa indirmeye gerek yoktur,
doğrudan `PYTHONPATH=src .venv/bin/python scripts/build_pairs.py` yeterlidir.

## Analiz zinciri — CPU, çevrimdışı, anahtarsız

Sırayla çalıştırılmalıdır; her adım bir öncekinin çıktısını okur.

| Komut | Üretir | Not |
|---|---|---|
| `make evidence-v2-prepare` | `data/processed/evidence_v2/` + manifest | Kaynak taksonomisini uygular, girdi hash'lerini dondurur |
| `make evidence-v2-baselines` | `evidence_v2/baselines/` | LightGBM + SHAP |
| `make modeling-generalization` | `reports/generalization/` | LODO, M2-Invariant, kararlılık matrisi |
| `make modeling-fairness` | `reports/fairness/` | N_eff yoğunlaşma, tanınırlık tercilleri, menşe farkı |
| `uv run python scripts/run_family.py` | `reports/modeling/model_family.csv`, `scored_*.parquet` | M0–M2 ailesi |

`evidence_v2` manifesti `configs/modeling/**`, `src/modeling/*.py`,
`src/evidence_eval/*.py`, `src/turkish_data.py` ve `uv.lock` dosyalarının
SHA-256'sını dondurur. Bunlardan biri değişirse — `make format` bile yeter —
`workspace.verify` hattı durdurur. Analiz kodu bu yüzden `src/visibility/` altındadır.

**Kayan nokta uyarısı:** `run_family.py` yeniden çalıştırıldığında
`model_family.csv` son basamaklarda (1e-16 düzeyinde) oynayabilir. Sayısal sonuç
aynıdır; `git diff` gösteriyorsa gürültüdür.

## GPU gerektiren adımlar — teslim için zorunlu değil

| Komut | Süre (RTX 4050 6 GB) |
|---|---|
| `scripts/run_m3.py --tracks en --category <sektör>` | sektör başına ~10 dk (isimli + maskeli) |
| `scripts/run_m3.py --tracks tr --category cosmetics` | ~1,5 dk |
| `make final-model-train` | değişken |

Çıktıları (`reports/modeling/m3_*.csv`, `reports/masking/`) **depoda sürümlüdür**,
böylece GPU'su olmayan biri raporu doğrulayabilir. Eğitim yeniden koşulmadan
`scripts/collect_masking.py` çalıştırılamaz; o, koşuların parquet çıktılarını okur.

İlk çalıştırmada encoder'ların indirilmiş olması gerekir. Önbellek belirli bir
revizyona sabitlendiğinden `HF_HOME` altındaki `refs/main` dosyası yoksa çevrimdışı
çözümleme başarısız olur; bu durumda ya ağ açılır ya da snapshot hash'i `refs/main`
içine yazılır.

## API anahtarı gerektiren adımlar — teslim için zorunlu değil

Bunlar veri toplama ve canlı deneylerdir; çıktıları depoda ya da yayımlanmış veri
setindedir, yeniden koşulması gerekmez.

| Komut | Anahtar |
|---|---|
| `make dataset-*` (Türkçe veri toplama) | GEMINI, MINIMAX, ABLITERATION, SERPER, CEREBRAS |
| `make intervention-run` (kontrollü test) | GEMINI |
| `make brand-demo` (canlı CLI demosu) | MINIMAX, SERPER |

`make brand-demo-offline`, `make ai-brand-report`, `make brand-report` ve
`make app` anahtarsız ve ağsız çalışır.

## Değerlendirici için en kısa yol

Rapordaki sayıları GPU ve anahtar olmadan doğrulamak için:

```bash
uv sync --group dev --locked
make check                      # 252 test
make modeling-data              # ağ
make evidence-v2-prepare && make evidence-v2-baselines
make modeling-generalization && make modeling-fairness
```

Bu zincirin sonunda `reports/generalization/` ve `reports/fairness/` klasörleri
depodakiyle aynı sayılarla yeniden üretilmiş olur. Kontrollü test
(`reports/intervention/`) ve maskeleme (`reports/masking/`) sonuçları ücretli
çağrı ve GPU gerektirdiği için yeniden üretilmez; her ikisinin de ham makbuzları
ve koşu parametreleri (`run.json`) depoda veya yerel veri klasöründedir.
