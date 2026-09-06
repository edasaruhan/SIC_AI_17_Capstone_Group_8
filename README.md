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
