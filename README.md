# Yapay Zekâ Asistanlarında Marka Görünürlüğü

Samsung Innovation Campus capstone projesi. Amaç, yapay zekâ asistanlarının marka
önerilerini ölçmek ve marka ön bilgisi ile içerik etkisini ayrıştırmaktır.

[Fikir önerisi (PDF)](docs/references/fikir-onerisi.pdf)

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
