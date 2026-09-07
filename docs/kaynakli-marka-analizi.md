# Kaynaklı marka analizi: evidence_v1

Bu aşama, nihai LLM arayüzlü görünürlük yazılımının **analiz motorudur**.
İnternette yeni marka araştırmaz, provider anahtarlarını okumaz, yeni yanıt veya
judge etiketi üretmez. Yayımlanmış veriden gözlem, kaynak izi, vekil model skoru ve
inceleme gerektiren eylem hipotezi üretir. Bunlar ayrı alanlardır.

## Kaynaklar ve eski deneylerle ilişki

- İngilizce: `3RAIN/brand-bias-evaluations`, revision
  `400da04eced51d3afe52b6d20c0207fd613f8a4a`, 9.586 yanıt.
- Türkçe: Furkan Karlı'nın `furkankarli/turkish-brand-bias-evaluations` veri seti,
  revision `4d274b954be7d9b0abbfe3314b2cbc387dfd800f`, 300 yanıt.
- Türkçe kaynak `all/train.parquet` SHA-256:
  `bd7319afb3f717449b8fe0baa28fe8a4008da15f59e3af1a6b2cf2bde2801a71`.
- Türkçe tasarım: 2 domain x 5 soru x 3 model x 2 koşul x 5 tekrar.
- Yayımlanan kaynakların lisansları ayrıdır; referansın MIT lisansı Türkçe veriye
  otomatik uygulanmaz. Veri yeniden yayımlanmaz.

`reports/modeling/` içindeki S2 skorları tarihsel sonuçlardır. Yeni aday kümesi,
eşleştirme, prior ve öğrenici ayarları nedeniyle v1 skorlarının aynı olması
beklenmez. Yeni çıktılar `data/processed/evidence_v1/` altında, Git dışında kalır.
S2 PDF'sinin yalnız nedensel yorum paragrafı düzeltilmiştir; tablolar değiştirilmedi.
Bu düzeltmenin betiği `scripts/correct_sprint2_pdf.py`; eski PDF `ab7a3de` commit'inde
korunur. Orijinal raporun HTML kaynağı repoda olmadığından bu betik kapsamı dar
ve açık bir düzeltme uygular; genel rapor tasarımını yeniden üretmez.

## Çalıştırma sırası

Normal geliştirme bağımlılıkları yeterlidir; `.env` gerekmez:

```bash
make setup
make check
```

Veri dosyaları bu bilgisayarda yoksa yalnız yayımlanan verileri indir:

```bash
make reference-data
make turkish-data
```

Bu iki komut internete bağlanır; Serper veya generation/judge API'sini çağırmaz.
Mevcut dosyalar varsa aşağıdaki offline aşamaya doğrudan geçilebilir:

```bash
make modeling-validate
make modeling-prepare
make evidence-sample
make brand-report BRAND="NordVPN" DOMAIN=vpn LANGUAGE=tr
```

İnceleme dosyaları:

- `review/pilot.md`: önce birlikte okunacak üç gerçek yanıt.
- `review/sample.md`: 30 yanıt ve onlara gösterilen kaynaklar.
- `review/sample.json`: seçilen kayıtlar ve deney kimliği.
- `review/annotations.csv`: **başlangıçta boş insan etiketleri**.
- `reports/tr_vpn_nordvpn.md` ve `.json`: marka raporu.

Yollar, `data/processed/evidence_v1/` köküne göredir. Marka raporu model eğitimi
olmadan da gözlemsel bölümlerini üretir; model durumunu `not_trained` gösterir.
Kozmetik örneği:

```bash
make brand-report BRAND="CeraVe" DOMAIN=cosmetics LANGUAGE=tr
```

## İnsan incelemesi

Örneklem 10 EN VPN, 10 TR VPN, 10 TR kozmetiktir; her grupta beş soru niyetinden
ikişer yanıt seçilir, modeller dengelenir. Seed 42 ve kayıt listesi sürümlüdür.
Üç pilot kayıt bu 30 kaydın içindedir, ek yanıt değildir.

Önce `pilot.md` okunur. CSV'de her kayıt için en az bir inceleme satırı tamamlanır;
bir yanıtta birden çok iddia varsa satır çoğaltılır. Çözümlenmemiş `unreviewed`
satır bırakılmaz. Bir kaydın incelendi sayılması, tüm iddiaların nesnel olarak
doğru olduğuna dair otomatik garanti değildir; inceleyen kişi kapsamı not etmelidir.

| Alan | Yazılacak içerik |
|---|---|
| `record_id` | Hazır gelen kayıt kimliği; değiştirme |
| `brand` | İlgili marka |
| `claim` | Modelin son yanıtından birebir iddia alıntısı |
| `source_ids` | İncelenen kaynak kimlikleri; birden çoksa `;` ile ayır |
| `label` | `supported`, `contradicted`, `unverifiable` veya `no_claim` |
| `reviewer` | İncelemeyi yapan kişinin adı/kimliği |
| `note` | Neyi kontrol ettiğin ve etiketin gerekçesi |

`supported`: kaydedilen başlık/snippet iddiayı destekliyor. `contradicted`:
kaydedilen kaynak iddiayla çelişiyor. `unverifiable`: eldeki snippet'lerden
doğrulanamıyor; iddianın yanlış olduğu söylenmiyor. `no_claim`: incelenecek marka
iddiası yok; bunun açıklaması notta bulunmalı.

`supported` ve `contradicted` için aynı yanıta ait kaynak kimlikleri zorunludur.
Etiketlemeyi bir AI'ın doldurması insan doğrulaması sayılmaz. Bağımsız çift
etiketleyici veya inter-rater agreement yapılmış gibi raporlanmaz.

```bash
make evidence-review-check
```

30 kayıt tamamlanmadıysa bu komut bilerek hata verir. Kısmi durumu görmek için:

```bash
PYTHONPATH=src .venv/bin/python -m evidence_eval review-check
```

## Kaynak sözleşmesi

- `sources_<dil>.parquet`: marka eşleşmese bile **bütün** organik sonuçlar.
- `evidence_<dil>.parquet`: `(record_id, brand, source_id)` eşleşmeleri.
- `pairs_<dil>.parquet`: sabit marka kaydındaki tüm adayların model özellikleri.
- `responses_<dil>.json`: kullanıcı sorusu, son yanıt ve gösterilen kaynaklar.
- `folds_<dil>.json`: önceki sürümlü sorgu bölmelerinin kopyası.

Kaynak kimliği kayıt + sonuç indeksi + içerikten türetilir. Aynı URL'nin farklı
arama turlarında gösterilmesi korunur. Eşleşme, destek/alıntı/etki kanıtı değildir.
Resmî kaynaklar açık domain haritasıyla tanınır; `nordvpn.com.attacker.org` resmî
sayılmaz. Satış siteleri `retailer`, doğrulanmamış kaynaklar `unknown` olur.
Affiliate/editoryal listeleri doğrulanmış kayıt eklenene kadar boştur; editoryal
başlıklı bir sayfa otomatik bağımsız kaynak sayılmaz. Mevcut resmî domain haritası
kısmi olup bütün kozmetik markalarını kapsamıyor; `unknown` bu durumda doğru etikettir.

`manifest.json`, gerçek girdi dosyası hash'lerini, kaynak revision'larını,
hazırlama kodunu, ayarları ve çıktı hash'lerini içerir. İngilizce hazırlanmış
içerik ve Türkçe kaynak içeriği sabit parmak izleriyle doğrulanır. Dosyanın ham
SHA-256'sı ile kanonik içerik parmak izi birbirinin yerine kullanılmaz.

Kod/config/veri değişirse mevcut analiz kökü benimsenmez veya üzerine yazılmaz:

```bash
make modeling-prepare EVIDENCE_ROOT=data/processed/evidence_v2
make evidence-sample EVIDENCE_ROOT=data/processed/evidence_v2
```

Yeni root seçmek yeni veri sürümünü otomatik onaylamaz. Yeni veri için kaynak
revision/hash/hücre sözleşmesi de bilinçli olarak güncellenmelidir. Üretim veri
setinin `.env`, raw, judged veya export dosyalarını bunun için değiştirme.

## CPU modelleri ve açıklamalar

```bash
make modeling-baselines
make brand-report BRAND="NordVPN" DOMAIN=vpn LANGUAGE=tr
```

M0/M1/M2 aynı LightGBM ayarlarını ve ortak kategori/model/koşul girdilerini kullanır.
M0 prior; M1 prior + kaynak yapısı; M2 bunlara snippet dil özelliklerini ekler.
Aday havuzu test yanıtlarının markalarından türetilmez. Prior'lar kategori,
üretici model ve marka bazında, eğitim satırları için iç sorgu bölmeleriyle;
test satırları için yalnız dış eğitim bölmesiyle hesaplanır.

Anılma baseline'ı `prior_mention_off`, tek kazanan baseline'ı `prior_top_off`
kullanır. Düşük extraction confidence satırları model eğitiminden çıkarılır,
gözlemsel raporun paydasından çıkarılmaz. `y_top` yalnız tek kazananlı yanıtları
değerlendirir; rapordaki görünürlük ise tüm yanıtları payda alır.

`baselines/` altında iki dil ve iki hedef için skorlar, metrikler, durum ve
sorgu-dışı M2 SHAP katkıları saklanır. Başarıyla biten hedef yeniden eğitilmez.
Hatalı hedef yeniden başlatılır; mevcut diğer hedeflerin sonuçları korunur.
SHAP birimi log-odds'tur, katkı veya görünürlük yüzdesi değildir. Model olasılıkları
kalibre edilmiş gerçek dünya görünürlük garantisi değildir.

PR-AUC, top-1 ve NDCG@3; sorgu grubu bootstrap aralıklarıyla raporlanır. Tekil
tekrarlar bağımsız örnek sayılmaz. EN/TR VPN karşılaştırması aynı beş niyettedir;
üretici modeller, tarih ve arama locale'leri eşit olmadığı için saf dil etkisi denmez.

## Kontrollü GPU aşaması

Bu komutlar hiçbir hazırlama/rapor komutunun bağımlılığı değildir. CUDA yoksa
uzun eğitim CPU'ya düşmez. Önce ayrı ortamda veya bu ortamda `uv sync --extra m3`
ile isteğe bağlı paketler kurulur; paket/model indirmesi büyük olabilir.

Encoder: EN `bert-base-uncased`, TR `dbmdz/bert-base-turkish-cased`. Kullanılacak
modelin HF sürüm geçmişinden **40 karakterli commit SHA'sı** seçilir ve kaydedilir.
`ENCODER_COMMIT` aşağıdaki komutlarda bu gerçek değerle değiştirilir:

```bash
make modeling-m3-smoke M3_ARGS="--track tr --model-revision ENCODER_COMMIT --allow-download"
make modeling-m3-train M3_ARGS="--track tr --model-revision ENCODER_COMMIT --seed 7"
```

İlk indirme için yalnız açık `--allow-download` kullanılır; sonrasında cache
kullanılır. EN için `--track en` ve o modelin kendi revision'ı kullanılır.
Deney seed'leri 7, 13, 21'dir; her biri ayrı komutla çalıştırılır. Aynı seed'de
isimli ve maskeli varyantlar otomatik aynı adaylar ve sorgu bölmelerinde çalışır.

Listwise grup, tek yanıtın tüm adaylarıdır; skorlar grup içinde softmax ile tek
kazanana doğru eğitilir. Yalnız `search_on`, tek kazananlı ve düşük confidence
olmayan yanıtlar kullanılır. Kapsama alınan/dışarıda kalan yanıt sayıları saklanır.
İsim maskelemesi `[TARGET_BRAND]` ve farklı `[OTHER_BRAND_n]` işaretleri kullanır;
URL/domain metni encoder girdisine eklenmez. Eşleşmeyen alias'lar için maskeleme
kusursuzluğu iddia edilmez; sabit kayıt kapsamındaki varyantlar test edilir.

Epoch sonu model/optimizer checkpoint'i, tokenizer, revision, ayarlar, skorlar ve
hata durumu `listwise/<config-hash>/` altında kalır. Kesinti son başarılı epoch'tan
devam eder; farklı ayarlar ayrı run oluşturur. Smoke yalnız iki eğitim yanıtı ve
iki test yanıtını bir fold'da çalıştırır; sonuçları tam deney metriği diye sunulmaz.
4 GB GPU'ya sığacağı garanti edilmez; bellek yetmezse küçük candidate batch veya
max length ile **ayrı run** açılır. Otomatik ayar/model değişimi yapılmaz.

## Nihai ürüne geçiş kapısı

Bu sürümden sonra eklenecekler: yeni marka ve site girdisi, canlı arama ve tam
sayfa denetimi, LLM ile kaynaklı açıklama, kullanıcı arayüzü ve önerinin etkisini
ayrı deneyle ölçme. Canlı kaynaklar geçmiş snippet'lerle aynı gözlem gibi ele alınmaz.

E-GEO veri/lisans/hedef uyumu henüz değerlendirilmedi. Yeni marka ve yeni domain'e
genelleme, LLM-only ile LLM+vekil model karşılaştırması ve önerilerin gerçek etkisi
gelecek doğrulama işleridir; bu sürüm bunları tamamlanmış kabul etmez.

İlk kontrol kapısı: `make check`, sabit veri doğrulaması, 3/30 inceleme dosyaları,
gerçek kaynaklı marka raporu. İnsan etiketleme tamamlanmadan `review complete`,
GPU eğitimi yapılmadan `trained listwise model` denmez. Eksik aşamalar hata veya
durum alanlarıyla görünürdür; örnek skor/etiket uydurulmaz.
