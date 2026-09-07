# İlk Türkçe listwise eğitim koşusu

Tarih: 2026-09-07. Dal: `feat/evidence-backed-brand-reports`.
Bu not gerçek GPU eğitimi ve kaydedilmiş sonuçlara dayanır; smoke sonuçları
başarı metriği olarak kullanılmaz. Generation, Serper veya judge API'si çağrılmadı.

## Ne eğitildi?

- Encoder: `dbmdz/bert-base-turkish-cased` (BERTurk).
- Model revision: `b6e1de16c983e0f2c70664591ea3f22810072608`.
- Veri/uygulama kimliği:
  `ffe314b787bb7b85489d4b790cb9bc67fa4606b9316b0784220dbaa878de9dbe`.
- Hedef: bir yanıtın sabit aday havuzu içinde tek kazananını listwise softmax
  ile tahmin etmek. Üretici LLM yeniden eğitilmedi; bu ayrı bir vekil modeldir.
- Her domain için isimli ve maskeli varyant, seed 7, iki epoch,
  learning rate `2e-5`, max length 128, candidate batch 4.
- Sorgu grupları eğitim/test arasında ayrıldı. Aynı sorgunun tekrarları
  bağımsız test grupları olarak sayılmadı.
- Donanım: NVIDIA GeForce RTX 4050 Laptop GPU, 6 GB.
- Ortam: PyTorch `2.14.0+cu130`, Transformers `5.16.1`;
  bağımlılıklar `uv sync --extra m3 --group dev --locked` ile kuruldu.

| Domain | Arama açık yanıt | Dahil / hariç | Bağımsız sorgu | Aday/yanıt | Tam koşu kimliği |
|---|---:|---:|---:|---:|---|
| VPN | 75 | 57 / 18 | 5 | 24 | `9dd1e55ee84de53b` |
| Kozmetik | 75 | 24 / 51 | 3 | 74 | `1c5aadf2b8963237` |

Kalan yanıtlar silinmedi; tek kazananlı listwise hedefe uygun olmadıkları için
bu deneyin dışında kaldılar. Türkçe kaynak veri hâlâ 300 yanıttır.
Kozmetikte uygun yanıtlar yalnız 3 sorgudan geldi; 5 bağımsız sorgu test edilmiş
gibi raporlanmaz. Her fold'da yukarıdaki toplamın yalnız eğitim alt kümesi öğrenilir.

Tam koşuların ilk `config.json`–`results.json` dosya zamanları arasındaki yaklaşık
süre: VPN **8,8 dakika**, kozmetik **4,3 dakika**. Bunlar paket/model indirmesini,
smoke testlerini ve sonraki denetimleri içermez. İlk zaman aralıkları UTC olarak
VPN 13:56:01–14:04:49, kozmetik 14:04:59–14:09:19'dur; tamamlanmış koşu tekrar
çağrıldığında config/results dosya zamanları yenilenebileceğinden bu süreler ilk
çağrı sırasında kaydedilmiştir, bir performans garantisi değildir.

## Aynı test panelinde sonuçlar

PR-AUC (yüksek daha iyi), sorgu-dışı tahminler:

| Domain | Frekans | M0 | M1 | M2 | M3 isimli | M3 maskeli |
|---|---:|---:|---:|---:|---:|---:|
| VPN | 0,182 | 0,160 | 0,238 | 0,267 | 0,165 | 0,278 |
| Kozmetik | 0,103 | 0,160 | 0,238 | 0,167 | 0,125 | 0,077 |

Tek kazananı ilk sıraya koyma oranı:

| Domain | Frekans | M0 | M1 | M2 | M3 isimli | M3 maskeli |
|---|---:|---:|---:|---:|---:|---:|
| VPN | %28,1 | %15,8 | %21,1 | %19,3 | %14,0 | %19,3 |
| Kozmetik | %33,3 | %29,2 | %16,7 | %16,7 | %29,2 | %12,5 |

Tüm modeller **aynı test yanıtı/aday/etiket panelinde** karşılaştırıldı. Bununla
birlikte eğitim kapsamları farklıdır: CPU modelleri Türkçe track'in tüm
kategorilerini/iki koşulu, M3 yalnız ilgili domain/arama-açık yanıtları kullanır.
Bu tablo yalnız mimariyi değiştiren kontrollü bir deney değildir. Önceki CPU
raporundaki iki koşullu sonuçlarla bu tablonun paydaları farklıdır.

M3 PR-AUC için sorgu-bootstrap %95 aralıkları:

- VPN isimli: 0,080–0,369; maskeli: 0,198–0,381.
- Kozmetik isimli: 0,108–0,265; maskeli: 0,063–0,258.

Yorum: VPN maskeli M3'ün PR-AUC'si M2'ye yakın; kozmetikte M1'in nokta tahmini
iki M3 varyantından yüksek. Tek kazanan metriğinde basit frekans yöntemi her iki
domain'de bütün eğitimli modellerden yüksek nokta tahminine sahip. Dolayısıyla
M3'ün ürüne net ek faydası bu koşuyla gösterilmiş değildir. Az sorgu ve tek seed
nedeniyle istatistiksel üstünlük veya domain genellemesi iddia edilmez. İsimli–
maskeli fark nedensel marka/içerik katkı yüzdesi veya modelin gizli düşüncesi değildir.

## Teknik doğrulama ve dosyalar

- İki domain'in isimli/maskeli küçük GPU testleri tamamlandı.
- Tam deneyde 10 VPN + 6 kozmetik = **16 fold checkpoint'i** tamamlandı.
- Checkpoint/skor hash'leri, aday ve etiket paneli, sorgu fold'ları, sonlu ve
  grup toplamı 1 olan olasılıklar kontrol edildi.
- **16 checkpoint'in her biri yeniden yüklendi**; her birinden bir yanıt yeniden
  skorlandı. Kontrol edilen tahminlerde maksimum mutlak fark **0,0** idi.
  Bu yeniden-skorlama kontrolü bütün yanıtları kapsıyor gibi sunulmaz.
- İki tam koşu aynı ayarlarla tekrar çağrıldı; tamamlanmış fold'lar atlandı.
  Toplam **32 checkpoint/skor dosyasının** değiştirilme zamanı aynı kaldı.
  Bu kontrol tamamlanmış koşunun yeniden kullanımını doğrular; yarım epoch
  ortasında kesinti uygulanıp optimizer devamı ayrıca denenmiş değildir.
- `make check`: **120 test**, ruff, black ve pyright başarılı.

Yerel çıktı kökü `data/processed/evidence_v1/`:

```text
listwise/9dd1e55ee84de53b/       # VPN, tam koşu
listwise/1c5aadf2b8963237/       # kozmetik, tam koşu
  config.json
  results.json
  audit.json
  named_fold*/
    checkpoint.pt             # model + optimizer + tamamlanan epoch
    tokenizer/
    scores.parquet
    state.json
  masked_fold*/
listwise/0e04046e7bdfc3b4/      # VPN smoke; tam deney değil
listwise/4748d1469e1786e2/      # kozmetik smoke; tam deney değil
training_logs/                # ilk eğitim, denetim ve devam kontrolleri
```

Checkpoint'ler optimizer durumları dahil tutulduğu için tüm listwise çıktıları
yaklaşık 25 GB yer kaplar. Git tarafından izlenmez; bu commit model ağırlığı veya
veri seti yüklemez. `.env` okunmadı, yayımlanmış veri yeniden toplanmadı.

## Hazır olan ile kalan işin ayrımı

Artık yalnız eğitim kodu değil, **gerçekten eğitilmiş ve tekrar yüklenmiş Türkçe
model checkpoint'leri** var. Bunlar çapraz doğrulama modelleridir; tüm veride
eğitilmiş seçili final model, servis veya canlı marka analiz arayüzü değildir.

Henüz yapılmayanlar: İngilizce M3 eğitimi; seed 13/21 tekrarları; insan incelemesi
(0/30); yeni markalara genelleme; final model seçimi/tam-veri eğitimi/inference
paketi; canlı arama ve LLM arayüzü. Eğitim sonuçları bu doğrulama kapılarının
yerine geçmez. Eğitim, devam ve denetim komutları
[çalıştırma rehberindedir](../docs/kaynakli-marka-analizi.md).
