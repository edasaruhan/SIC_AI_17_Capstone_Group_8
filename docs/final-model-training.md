# Nihai deneysel model paketi v1

Bu aşama yalnız analiz kodu hazırlamak değildir: seçilen modeller uygun verinin
tamamıyla yeniden eğitilir, ağırlıkları kaydedilir ve diskten yeniden yükleme
tahminleri karşılaştırılır. Bir sohbet LLM'si sıfırdan eğitilmez. Paket, asistanların
tek kazanan marka tercihini tahmin eden üç ayrı sıralama modelinden oluşur.
Web arama, ürün arayüzü ve kaynaklı aksiyon önerisi bu paketin dışındadır.

## Seçim ve veri kapsamı

Seçim, mevcut sorgu-dışı çapraz doğrulamada **PR-AUC** esas alınarak yapıldı.
Bu, test sonuçları görüldükten sonra yapılmış deneysel seçimdir; bağımsız bir
final-model başarı ölçümü değildir. Farklı eğitim kapsamları nedeniyle karşılaştırma
yalnız mimarinin etkisini de ölçmez.

| Kullanım | Seçilen model | Tam veri eğitimi | Önceki CV PR-AUC | Önceki CV top-1 |
|---|---|---|---:|---:|
| Türkçe VPN | M3 maskeli BERTurk | 57 yanıt × 24 aday | 0,2777 | %19,30 |
| Türkçe kozmetik | M1 LightGBM | Her iki Türkçe domain/koşuldan 136 yanıt, 5.364 aday satırı | 0,2383 | %16,67 |
| İngilizce VPN | M3 isimli BERT | 975 yanıt × 24 aday | 0,6932 | %71,69 |

Tablodaki metrikler eski CV model ailesinin aynı aramalı değerlendirme panelindeki
sonuçlarıdır; yeni tam-veri ağırlıklarının test skoru değildir. Kozmetik M1 eğitim
kapsamı önceki M1 deneyiyle aynı tutulur, ürün yönlendirmesinde yalnız Türkçe
kozmetik için kullanılır. BERT modelleri yalnız `search_on`, tek kazananı bulunan,
düşük güvenli olmayan kayıtlarla eğitilir. Türkçe VPN'de 75 aramalı yanıtın 18'i,
İngilizce VPN'de 1.188 yanıtın 213'ü bu filtreden dolayı dışarıda kalır.

Türkçe frekans taban çizgisi top-1'de hâlâ daha iyidir: VPN %28,07, kozmetik
%33,33. İngilizce sonuçlar daha güçlü olsa da yalnız az sayıdaki sorgu grubuna
dayanır. Türkçe/İngilizce model ve veri farkları yüzünden bunlardan saf dil etkisi
sonucu çıkarılamaz. Sonraki ürün denemelerinde frekans karşılaştırması korunmalıdır.

Eğitim hedefi mevcut veri setinin/judge'ın etiketidir. `review/ai/annotations.csv`
okunmaz; AI incelemesi insan doğrulaması veya altın standart eğitim etiketi sayılmaz.

## Çalıştırma

Mevcut `.venv`, hazırlanan `data/processed/evidence_v1`, tamamlanan üç CV deneyi
ve yerel encoder cache'i gereklidir. Bu hedefler bağımlılık/model indirmez,
`.env` okumaz ve provider, Serper veya judge çağrısı yapmaz.

```bash
make final-model-plan
make final-model-train
make final-model-status
```

`train` önce CPU üzerinde M1'i, sonra GPU üzerinde Türkçe ve İngilizce BERT'i
çalıştırır. BERT eğitimi için CUDA zorunludur; uzun CPU eğitimine sessiz geçiş yoktur.
API kredisi kullanılmaz; yerel GPU/elektrik tüketilir.

- BERT: 2 epoch, seed 7, AdamW `2e-5`, en çok 128 token, 4 adaylık hesaplama
  parçaları; kayıp her yanıtın tüm adayları üzerinde tek listwise softmax'tır.
- Başlangıç, eski CV checkpoint'i değil sabit revision'daki temel encoderdır.
- M1: önceki 120 ağaçlık LightGBM ayarları; eğitim öncülleri sorgu-grubu dışından,
  inference öncülleri bütün geçmiş uygun veriden hesaplanır.
- `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `local_files_only=True` kullanılır.
- Encoder cache: `data/processed/evidence_v1/hf_cache/hub`.
- BERTurk revision: `b6e1de16c983e0f2c70664591ea3f22810072608`.
- İngilizce BERT revision: `86b5e0934494bd15c9632b12f734a8a67f723594`.

Kesinti sonrası aynı komut tamamlanan ve checksum'u doğrulanan modelleri atlar.
BERT son tamamlanan **epoch** checkpoint'inden devam eder; yarıda kesilen epoch
baştan çalışır. Aynı sürüm için eşzamanlı ikinci eğitim dosya kilidiyle reddedilir.
Durum/Loguru logu her 25 yanıtta ve epoch sınırında güncellenir.

## Çıktı ve sürümleme

`data/processed/final_models_v1/<release_id>/` altında:

```text
manifest.json                 # seçim, eski CV metrikleri, veri/kod/paket kimliği
training.log                  # Loguru ilerleme ve hata kaydı
training.lock
tr_cosmetics/
  model.txt                   # yerel LightGBM biçimi, pickle değil
  priors.parquet              # dondurulmuş geçmiş marka öncülleri
  preprocessing.json          # sütunlar ve kategorik değer sıraları
  state.json                  # tamamlanma, SHA-256, reload karşılaştırması
tr_vpn/ ve en_vpn/
  encoder/                    # safetensors ağırlıkları, config, tokenizer
  checkpoint.pt               # epoch-resume için optimizer ve eğitim ağırlıkları
  reload_check.json           # bir eğitim yanıtında teknik eşitlik kontrolü
  state.json
```

Model yalnız dosyaya kaydetme ve tekrar yükleme tahmin eşitliği başarılı olduğunda
`completed` olur. Bu teknik kontrol yeni test skoru değildir. `status` tamamlanmış
inference dosyalarının SHA-256 değerlerini tekrar doğrular. Resume checkpoint'i
inference paketinin checksum listesinde bulunmaz; yalnız yerel resume içindir.

Eski `evidence_v1` manifesti, 26 CV checkpoint'i ve raporlar değiştirilmez.
Yeni sürüm kimliği final eğitim kodu, paket sürümleri, dondurulmuş deney kimliği ve
seçim bilgilerini kapsar. Final kodu değişince yeni klasör oluşur. Büyük ağırlıklar
Git tarafından izlenmez; bu çalışma bunları bir yere yüklemez.

## Kaydedilmiş modelden tahmin

Inference mevcut repodaki dondurulmuş ön işleme sözleşmesine bağlıdır; henüz
bağımsız dağıtılabilir web servisi değildir. Girdi yalnız sorgu ve **kaydedilmiş
arama sonucu başlığı/snippet/URL/pozisyonudur**. Asistanın son cevabı girdi değildir.
Kod URL'yi ziyaret etmez. `sources: []` desteklenir ama kaynak yokluğu modelin
marka ön bilgisini yansıtabilir; kaynak kanıtı oluşturmaz.

Örnek istek dosyası (generator kimliğini ilgili manifestteki `generator_ids`
listesinden aynen seçin):

```json
{
  "language": "tr",
  "category": "vpn",
  "condition": "search_on",
  "model_id": "MANIFESTTEKI_GENERATOR_ID",
  "query_text": "Gizlilik için hangi VPN'i kullanmalıyım?",
  "sources": [
    {
      "title": "Kaydedilmiş arama sonucu başlığı",
      "snippet": "Kaydedilmiş metin",
      "link": "https://example.org/recorded-page",
      "position": 1
    }
  ]
}
```

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src .venv/bin/python -m final_model predict \
  --release data/processed/final_models_v1/RELEASE_ID \
  --request /tam/yol/istek.json --device cpu
```

Çıktı tam sabit aday listesinin sıralaması ve sınırlamalardır. M1 skorları kalibre
edilmemiş ikili sınıflandırma skorlarıdır; BERT skorları sabit aday paneli üzerindeki
koşullu softmax'tır. İkisi de “görünürlük şu kadar artar” olasılığı değildir.
M1 çıktısında `frequency_comparator` da yer alır. Yeni domain, generator ve marka
kapsamı değerlendirme/veri eklemeden sessizce genişletilmez.

Üç modelin her birinde üç tarihsel kaydın ön işleme eşitliğini ve bir kaydın CPU
inference'ını yeniden kontrol etmek için:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
  .venv/bin/python scripts/verify_final_models.py \
  --release data/processed/final_models_v1/RELEASE_ID
```

Bu komut `validation/` altına gerçek yerel kayıtlardan örnek istek/tahmin JSON'ları
ve `integration.json` üretir. Yeni veri toplamaz. Seçilen örnekler eğitim verisinden
olduğu için çıktılar başarı ölçümü veya yeni kullanıcı önerisi olarak sunulmaz.

## Nihai ürüne geçiş

Bu paket, arayüzde bir marka için **mevcut kanıtlar altında tercih sıralaması**
üreten deneysel bileşendir. Nihai ürün için ayrıca güncel veri toplama, kaynak
doğrulama, kaynaklara bağlı öneri üretimi, frekans taban çizgisiyle karşılaştırma ve
yeni sorgularla bağımsız değerlendirme gerekir. Bir içerik değişikliğinin gerçekten
görünürlüğü artırdığı iddiası kontrollü önce/sonra deneyleri olmadan kurulmaz.

## Bu çalışma alanındaki tamamlanan koşu — 9 Eylül 2026

- Sürüm: `8a0b2f09072b4506`.
- Üç modelin durumu `completed`; gerçek yerel eğitim yaklaşık 17 dakika sürdü.
- Üçünde de kaydetme/yeniden yükleme kontrolündeki en büyük mutlak tahmin farkı `0.0`.
- Her modelde üç tarihsel kayıt için ön işleme eşitliği, bir kayıt için CPU inference
  doğrulandı. Kanıt: `data/processed/final_models_v1/8a0b2f09072b4506/validation/integration.json`.
- `make check`: Ruff, Black ve Pyright başarılı; **154 test geçti**.
- Önceki 26 CV checkpoint'inin kayıtlı checksum'ları ve insan inceleme CSV'si korundu.
- API çağrısı, yeni veri toplama, AI inceleme etiketlerini eğitime katma veya push yapılmadı.

Bu tamamlanma kaydı teknik eğitim teslimidir; bağımsız kalite onayı değildir.
