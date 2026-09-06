# Yapay Zekâ Asistanlarında Marka Görünürlüğü

Samsung Innovation Campus capstone projesi. Amaç, yapay zekâ asistanlarının marka
önerilerini ölçmek ve marka ön bilgisi ile içerik etkisini ayrıştırmaktır.

[Fikir önerisi (PDF)](docs/references/fikir-onerisi.pdf)

Ödev 1 teslimindeki üç raporun Markdown/DOCX sürümleri ve güncellenmiş görselleri
[`reports/assignment-1/`](reports/assignment-1/) klasöründedir.

## Sprint 0 · 10–16 Ağustos

| Görev | Sorumlu | Çıktı |
|---|---|---|
| S0-1 Repo iskeleti | [@furkankarli](https://github.com/furkankarli) | Çalışan minimal repo |
| S0-2 Ortam ve bağımlılıklar | [@furkankarli](https://github.com/furkankarli) | Python 3.11 + `uv` |
| S0-3 Referans veri setini hazırlama | [@muratmertkucuk](https://github.com/muratmertkucuk) | `reference.parquet` |
| S0-4 İki bulguyu yeniden üretme | [@muratmertkucuk](https://github.com/muratmertkucuk) | Notebook ve kısa rapor |
| S0-5 Literatür tablosu | [@kubragzc](https://github.com/kubragzc) | Sekiz çalışmalık özet |
| S0-6 Teknoloji ve maliyet incelemesi | [@kubragzc](https://github.com/kubragzc) | Tarihli maliyet tablosu |
| S0-7 Kullanıcı görüşmeleri | [@zeynepsinal](https://github.com/zeynepsinal) | En az beş görüşme notu |
| S0-8 Görüşme sentezi | [@zeynepsinal](https://github.com/zeynepsinal) | Ürün gereksinimleri |

## Kurulum

Gereksinimler: Git ve [`uv`](https://docs.astral.sh/uv/).

```bash
cp .env.example .env
make setup
make check
```

Notebook'lar Colab/Kaggle üzerinde veya ekip üyesinin tercih ettiği yerel notebook
ortamında çalıştırılabilir.

Referans veri setini hazırlamak için:

```bash
make reference-data
```

Komut, sabitlenmiş `3RAIN/brand-bias-evaluations` sürümünün yalnızca `all` alt
kümesini işler ve `data/interim/reference.parquet` dosyasını üretir. Parquet türetilmiş
veridir ve Git'e eklenmez; ekip aynı dosyayı komutla yeniden oluşturur.

## Türkçe marka yanlılığı veri seti

Referansın deney ve export yapısıyla uyumlu Türkçe toplama altyapısı iki domain,
üç model ve aramalı/aramasız iki koşul için tam 300 hücre planlar. Kod gerçek
API çağrısı yapmadan hazırlanmış ve mock testleriyle doğrulanmıştır; veri toplama
ancak sizin `.env` anahtarlarını ekleyip ilgili Make hedefini çalıştırmanızla başlar.

Tüm Türkçe veri setini tek komutla üretmek için:

```bash
make dataset-all
```

Bu hedef preflight, eksik generation hücreleri, judge, export ve validation
aşamalarını sırayla çalıştırır; hata durumunda durur ve yeniden çalıştırıldığında
tamamlanmış hücreleri tekrar çağırmaz.

Generation modelleri Gemini Flash Lite, MiniMax M2.7 ve GLM-5.3/Abliteration'dır.
NVIDIA, tekrarlanan timeout ve endpoint hataları nedeniyle deneyden çıkarılmış;
ilgili ham ve arşiv kayıtları temizlenmiştir. Tamamlanan generation sonrasında
judge, export ve validation aşamalarını tek komutla çalıştırmak için
`make dataset-finish` kullanılır.

Boş nihai cevapla kalan GLM hücreleri `make dataset-repair-generation` ile
onarılır. Bu hedef yalnız eksik GLM hücrelerinde Abliteration düşünmesini kapatır;
tamamlanmış kayıtları ve diğer generation sağlayıcılarını yeniden çağırmaz.

Her CLI komutu, ekrandaki kısa durum mesajlarının yanında ayrıntılı ve dönen bir
`logs/bias-eval.log` dosyası üretir. Son logları `make dataset-logs`, canlı akışı
ayrı bir terminalden `make dataset-follow-logs` ile izleyebilirsiniz. Anahtarlar ve
Authorization değerleri log yazılmadan önce maskelenir.

```bash
make dataset-plan
make dataset-preflight
make dataset-pilot
make dataset-status
make dataset-collect
make dataset-judge
make dataset-judge-status
make dataset-export
make dataset-validate
make reference-data
make dataset-report
```

Kurulum, kota güvenliği, resume davranışı, dosya şemaları ve hata giderme adımları
için [Türkçe veri seti runbook'una](docs/turkce-veri-seti-runbook.md) bakın.
Pipeline'ı değiştirecek ekip üyeleri önce
[veri seti geliştirici rehberini](docs/veri-seti-gelistirici-rehberi.md) okumalıdır.

İki referans bulguyu yeniden üretip notebook'u çalıştırmak için:

```bash
make reference-report
```

Notebook araçları bu komutta geçici olarak kurulur; kalıcı proje bağımlılıklarına
eklenmez. Çalıştırılmış analiz `notebooks/S0-4-reference-validation.ipynb`, kısa sonuç
özeti ise `reports/referans_dogrulama.md` altında tutulur.

## Modelleme ve bulgular

`src/modeling/` altındaki paket, İngilizce referans ile Türkçe veri setini **aynı
kodla iki kez** işler; hiçbir yerde dile özel ayrı bir hat yoktur. İki korpus 282.450
(yanıt, aday marka) çiftine açılır ve iki hedef modellenir: markanın yanıtta anılması
(satır düzeyinde AI Share of Voice) ve markanın tek birincil öneri olması.

Model ailesi kümülatiftir, çünkü anlamlı olan tek bir skor değil aralarındaki farktır:
iki naive temel → **M0** yalnız marka prior'ı → **M1** + arama konumu ve kaynak tipi →
**M2** + snippet dil özellikleri (LightGBM, SHAP ile öneri üretir) → **M3** cross-encoder
(BERTurk / BERT, maskeleme ablation'ı için).

**Üç bulgu.**

1. **Arama, öneriyi değiştiriyor.** Web araması açıldığında mahremiyet itibarıyla
   tanınan markalar görünürlük kaybediyor, ticari pazarlama yapanlar kazanıyor:
   NordVPN Türkçe'de +30,7 / İngilizce'de +25,9 puan, Mullvad Türkçe'de −24,0.
   Kozmetikte de aynı yapı (L'Oréal Paris +18,7, CeraVe −17,3). Örüntü iki dil ve
   iki sektörde, farklı modellerle tekrarlanıyor.
2. **Tahmin modeli temelleri aşıyor.** İngilizce VPN'de M2, "en sık kazananı söyle"
   temelini +0,27 PR-AUC geçiyor (0,801 vs 0,535) ve top-1 doğruluğu %64,1'den
   %72,4'e çıkıyor. Görünürlük hedefinde M2 altı domain-dil kombinasyonunun dördünde
   kazanıyor.
3. **Tanınırlık ve içerik ayrıştırıldı.** Snippet'lerdeki her marka adı `[BRAND]` ile
   değiştirildiğinde İngilizce'de PR-AUC yarıya iniyor (0,714 → 0,362, üç seed'de de
   aynı yönde). Kararın kabaca yarısı içeriğin ne dediğinden değil, hangi markanın adı
   olduğundan geliyor. Türkçe'de aynı ölçüm 57 karara bağlanmış yanıtla yapılamıyor.

Ayrıntılı yöntem, beş domainin tamamındaki sonuç tabloları, SHAP atfı, sınırlılıklar
ve sızıntı önlemleri için [Sprint 2 raporuna](reports/sprint-2/Sprint2_Modelleme_Raporu.pdf)
bakın. Rakamların ham hâli `reports/modeling/` altındaki CSV'lerde.

Sırayla çalıştırmak için:

```bash
make reference-data
uv run python scripts/run_family.py       # pair'ler, fold'lar, M0-M2, iki hedef
uv run python scripts/score_tables.py     # metrik tabloları, yeniden eğitmeden
uv run python scripts/run_m3.py           # M3 + maskeleme ablation'ı
uv run python scripts/compare_tracks.py   # VPN yan yana tablosu
uv run python scripts/attribution.py      # SHAP + ASoV temeli
```

M3 bir CUDA GPU'su ister; `scripts/run_m3.py --tracks tr` yalnız Türkçe hattını
çalıştırır ve birkaç dakika sürer. Fold atamaları `data/processed/modeling/splits_*.json`
altında dondurulmuş ve sürümlenmiştir — aynı bölme olmadan hiçbir skor yeniden
üretilemez. Beş domainin marka kayıtları ve dil sözlükleri `configs/modeling/` altında
denetlenebilir YAML olarak durur.

## Klasörler

```text
configs/       Ortak deney ayarları
data/raw/      Değiştirilmeyen ham veri
data/interim/  Ara çıktılar
data/processed/ Analize hazır veri
docs/          Proje referansları
notebooks/     Keşif ve doğrulama çalışmaları
reports/       Sprint çıktıları
src/           Tekrar kullanılabilir kod
tests/         Testler
```

Veri dosyaları Git'e eklenmez; yalnızca onları üreten kod ve raporlar paylaşılır.
Her görev `feat/S0-<no>-<konu>` dalında geliştirilir ve pull request ile birleştirilir.
Dal, veri ve PR kurallarının tamamı için [katkı rehberine](CONTRIBUTING.md) bakın.
