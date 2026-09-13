# Görünürlük danışmanı (LangGraph)

Projenin fikri: **örnek veriden görünürlük sinyalini öğren, sonra hiç görmediğin bir
sektöre uygula.** Danışman bunu yapar. Herhangi bir marka ve herhangi bir sektör
verilir; akış canlı arama yapar, rakip markaları sonuçlardan çıkarır, asistanın
cevabını ölçer, kayıtlı beş sektörde öğrenilen sinyal modeliyle markayı rakiplerine
göre skorlar, markanın hangi durumda olduğunu teşhis eder ve o duruma özgü öneriyi
kontrollü testte ölçülmüş etkisiyle birlikte verir.

```bash
make advisor-train                                             # bir kez: modeli eğit (CPU, ~1 sn)
make advisor-plan BRAND="Garanti BBVA" DOMAIN="bankacılık" LANGUAGE=tr   # ücretsiz tahmin
make advisor      BRAND="Garanti BBVA" DOMAIN="bankacılık" LANGUAGE=tr   # ÜCRETLİ
```

## Arayüz

```bash
make advisor-ui
```

Streamlit sayfası LangGraph akışını sürer; akışın kendisi değişmez. Soldan marka, sektör
ve dil girilir. Sayfa ücretli çağrı tahminini ve bütçeyi gösterir, analiz daha önce
yapıldıysa önbellekten okunacağını söyler, onay kutusu işaretlenmeden başlatmaz. Koşu
sırasında grafiğin her düğümü bittikçe listelenir (sorular → arama → rakipler → yapay
zekâya sorma → ölçüm ve teşhis → öneriler → rapor). Sonuçta teşhis, görünme, anılma ve
rakiplere göre sıra metrik olarak, ardından rapor gösterilir; rapor indirilebilir.

"Karşılaştırılan markaları düzelt" bölümünde yanlış rakipler kaldırılır, eksikler
eklenir ve "Düzeltmeyle yeniden hesapla" ile yeniden çalıştırılır. Düzeltme hiçbir ücretli
isteği değiştirmediği için koşu önbellekten okunur.

Arayüz ve komut satırı aynı `service.py` modülünü kullanır: koşu kimliği, çağrı
tahmini, bütçe ve kayıt ikisinde birebir aynıdır.

## Komut satırı

Rakip listesi yanlışsa düzeltip yeniden çalıştırın; önbellekteki çağrılar yeniden
ödenmez:

```bash
make advisor BRAND="Garanti BBVA" DOMAIN="bankacılık" LANGUAGE=tr \
  ADVISOR_ARGS='--brand-alias Garanti --add-rival "QNB" --drop-rival "Yanlış Ad"'
```

Anahtarlar `.env` içinde: `GEMINI_API_KEY`, `SERPER_API_KEY`. Varsayılan koşu
(3 sorgu, 2 tekrar) yaklaşık 17 çağrıdır: gerekirse 1 sorgu üretimi + 3 arama +
1 rakip çıkarımı + 12 asistan yanıtı. `--max-calls 20` bütçesini aşan bir koşu hiç
başlamaz.

## Örnek veriden sinyal, başka sektöre

Taşınan sinyal **M2-Invariant** modelidir (`reports/generalization/`). Marka
kimliğine, sektöre ve üretici modele bakmaz. Yalnız aynı cevaptaki adaylara göre
göreli sinyallere bakar:

- arama sonuçlarında görünmek,
- sonuçların ne kadarında anıldığı ve lidere oranı,
- en üstte çıkan aday olup olmadığı ve rakiplere göre sırası,
- kaynak karışımı ve metin özelliklerinin rakiplere göre yüzdeliği.

Bu model **leave-one-domain-out** testinde hiç görmediği sektörde, sektöre özgü
modeli dört İngilizce sektörün dördünde geçti. Örneğin VPN'de PR-AUC 0,566'ya karşı
0,256, kod editörlerinde 0,675'e karşı 0,348. Yalnız İngilizce veriyle eğitilip
Türkçe VPN'e de taşındı. Danışmanın kullandığı sürüm, kayıtlı **beş sektörün tamamında
ve iki dilde** eğitilir (`make advisor-train`). Özellikler eğitimdeki kodla birebir
aynı fonksiyonlarla üretilir (`add_relative`, `snippet_features`, v2 `source_type`),
çünkü aktarım ancak böyle geçerli olur.

Kaydedilen model `data/processed/advisor_model/manifest.json` içinde özellik listesi,
eğitim sektörleri, kanıt manifestinin kimliği ve dosya hash'leriyle tutulur. Dosya
değişmişse danışman modeli yüklemez.

## Akış

```
plan → search → discover → interrogate → analyse ─┬─ absent   ─┐
                                                  ├─ low_rank ─┤
                                                  ├─ ceiling  ─┼→ report
                                                  ├─ leader   ─┤
                                                  └─ thin     ─┘
```

| Düğüm | Ne yapar | Ücret |
|---|---|---|
| `plan` | Kayıtlı sektörde sorguları korpustan alır; diğer sektörlerde asistana ürettirir | 0 veya 1 |
| `search` | Her sorgu için Serper, 10 organik sonuç | Sorgu başına 1 |
| `discover` | Sonuçlardan rakip marka adlarını çıkarır; kullanıcı düzeltmesini uygular | 1 |
| `interrogate` | Gemini'ye soruyu arama kapalı ve arama sonuçlarıyla açık olarak sorar | Sorgu × 2 × tekrar |
| `analyse` | Ölçer, modeli uygular, geride kalınan sinyalleri bulur, teşhis koyar | 0 |
| `advise_*` | Teşhise özgü öneri; etkiler `reports/intervention/effects.csv`'den | 0 |
| `report` | Markdown rapor; her öneri etik filtreden geçmiştir | 0 |

## Rakip markalar nereden geliyor

Kayıtlı olmayan bir sektörde elle yazılmış marka listesi yoktur; liste her koşuda
kurulur:

1. Asistan, arama başlık ve özetlerinden marka adlarını **çıkarır**. Bu bir çıkarım
   işidir, görünürlük yargısı değildir. Veri setinin etiketleri de böyle üretildi.
2. **Metinde gerçekten geçmeyen her ad atılır.** Model bir marka uydurursa listeye
   giremez.
3. Kullanıcı `--add-rival` ile eksik adı ekler, `--drop-rival` ile yanlışı çıkarır,
   `--brand-alias` ile kendi markasının diğer yazılışlarını verir.
4. Sektör kayıtlıysa (VPN, hosting, seyahat, kod editörleri, kozmetik) elle doğrulanmış
   yazılış biçimleri listeye eklenir.

## Teşhis dallanması

| Teşhis | Anlamı | Öneri | Dayanak |
|---|---|---|---|
| **absent** | Arama sonuçlarında yoksun | Rakiplerini anan bağımsız karşılaştırma sayfalarına gir | +31 puan, ölçüldü |
| **low_rank** | Sonuçlarda varsın ama alt sıralarda | Girdiğin sayfalarda üst sıraya çık | +31 puan, ölçüldü |
| **ceiling** | Anılıyorsun ama asla ilk değilsin | **Bunu içerikle çözemezsin**; bütçeyi listeye girmeye ayır | 1.080 çağrıda 0 |
| **leader** | Zaten öndesin | Konumu koru | — |

Teşhis ölçülen oranlardan konur. Öğrenilmiş sinyal skoru bunun yanında durur ve
markanın **neden** o durumda olduğunu gösterir: hangi taşınabilir sinyalde rakiplerin
belirgin biçimde gerisinde kaldığını. Eşikler (`advise.py`) ürün kararıdır, ölçüm
değildir.

## Tasarım kararları

- **Sorgular marka önerisi sorar.** Kayıtlı olmayan bir sektörde sorgular üretilir; istem,
  yanıtı marka önermek olan sorular ister ve korpustaki gerçek soruları ("Şu anda en iyi
  VPN hizmeti hangisi?") biçim örneği olarak verir. İlk bankacılık denemesi bu kural
  olmadan "nasıl tasarruf ederim" gibi genel sorular üretti; hiçbir banka anılmadığı için
  her marka yok görünüyordu.
- **Pazar yoksa teşhis yok.** Arama sonuçlarında ve yanıtlarda 2'den az rakip marka
  geçiyorsa danışman "aramada görünmüyorsun" demez; teşhis koymaz ve nedenini yazar.
- **Ölçüm bir modelin görüşü değildir.** Anılma, ilk anılma, arama sonucunda görünme
  ve sıra, koşunun marka listesiyle metin eşleştirerek ölçülür. "Bu marka görünür mü?"
  diye bir LLM'e sormak döngüsel olurdu.
- **Etki büyüklükleri uydurulmaz.** Raporda geçen her etki kontrollü testten gelir.
- **Makbuzlar korunur.** Ücretli her çağrı `brand_demo.workflow.Receipts` ile yazılır.
  Geçersiz bir çıkarım veya sorgu üretimi doğrulayıcıdan geçemez ve önbelleğe girmez.
- **Aynı asistan.** Canlı çağrılar Gemini 3.5 Flash Lite ile yapılır, etkiler de bu
  asistanda ölçüldü.
- **Rakip siteleri ve platformlar outreach hedefi değildir.** Kaynak taksonomisi resmî
  alan adlarını yalnız araştırmanın beş sektörü için bilir; diğer sektörlerde alan adının
  isim kısmı koşunun aday markalarıyla eşleştirilir. Türkçe harfler ASCII'ye katlanır:
  `isbank.com.tr` "İş Bankası"na, `qnb.com.tr` "QNB"ye bağlanır. Uygulama mağazaları ve
  sosyal ağlar da elenir. Karşılaştırma siteleri (ör. hangikredi.com) bağımsız hedef
  olarak kalır; kontrollü testin ölçtüğü tam olarak bu tür bir sayfadır.
- **Aday çıkarımı kategoriye sınırlıdır ve alan adını görür.** Çıkarım modeline her
  sonuç "alan adı — başlık — özet" olarak verilir. Bir bankanın sayfası başlıkta adını
  hiç anmayabilir; alan adında görünen marka da aday sayılır. Hisse senetleri, holdingler,
  piyasa adları ve karşılaştırma siteleri rakip olarak çıkarılmaz; alt ürün yerine ana
  marka yazılır.
- **Etik filtre.** Her öneri `visibility.ethics.screen`'den geçer.
- **`uv.lock` değişmez.** `langgraph` yalnız `graph.py` içinde kullanılır ve çalışma
  anında `uv run --with langgraph` ile kurulur. Diğer bütün modüller normal test
  paketinde sınanır.

## Bu rapor ne söylemez

- Kayıtlı olmayan bir sektörde skor bir **ekstrapolasyondur**. Leave-one-domain-out
  testi bunun işe yaradığını gösterir, ama o sektörde ayrıca doğrulanmadı.
- Etkiler tek asistanda, İngilizce bağlamda ve üç sektörde ölçüldü. Yön taşınıyor,
  büyüklük ölçülmedi.
- Rakip listesi çıkarıma dayanır; eksik veya fazla ad sonuçları değiştirir.
- Serper sonuçları, asistanın gerçekte getireceği sayfalarla aynı olmayabilir. Eğitim
  verisinde yanıt başına ortanca sonuç sayısı İngilizce'de 20, Türkçe'de 11'dir; canlı
  koşu 10 sonuç kullanır.
- "İlk anılan marka", birincil önerinin yaklaşık bir vekilidir.
- Varsayılan koşu az sorgu ve tekrarla çalışır; oranlar yön gösterir.

## Örnek: modelin hiç görmediği bir sektör

`Garanti BBVA`, bankacılık, Türkçe, 13 Eylül 2026. Bankacılık araştırmanın beş sektöründen
biri değil; sorgular, rakipler ve outreach hedefleri tamamen koşu sırasında kuruldu.

| | Değer |
|---|---|
| Aday markalar | 16, hepsi banka (QNB, Yapı Kredi, DenizBank, Enpara, ING, Ziraat Bankası, Türkiye İş Bankası…) |
| Arama sonuçlarında görünme | %100, ortanca sıra 6 |
| Arama açıkken / kapalıyken anılma | %33 / %50 |
| Teşhis | Aramada görünüyorsun ama alt sıralarda |
| Öğrenilmiş sinyal | 16 aday içinde tahmini anılmada 1.; geride kalınan sinyal: en üstte çıkan aday olmak |
| Outreach hedefleri | enuygunfinans.com, hangikredi.com |

Bu sonuca üç başarısız koşudan geçilerek varıldı, ve her biri bir tasarım hatası gösterdi:

1. **Kesilen yanıt.** 1024 tokenlık sınır uzun banka cevaplarını kesti; tek bir reddedilen
   yanıt bütün koşuyu düşürdü. → Sınır 2048 oldu, tekil hatalar makbuza yazılıp koşu devam
   ediyor.
2. **Pazar bulamayan sorgular.** Üretilen sorular "nasıl tasarruf ederim" türündendi; hiçbir
   banka anılmadı ve danışman markayı yanlışlıkla "aramada yok" sandı. → Sorgular marka
   önerisi soruyor, korpustaki gerçek sorular örnek veriliyor; pazar yoksa teşhis konmuyor.
3. **Kategori dışına kayma ve kirli liste.** Bir sorgu borsa uygulamalarına kaydı (Apple,
   Tesla); çıkarım QNB ve Yapı Kredi'yi kaçırdı çünkü alan adını görmüyordu; outreach
   listesine uygulama mağazaları ve bir rakibin kendi sitesi girdi. → Sorgu ve çıkarım
   kategoriye sınırlandı, alan adı çıkarıma verildi, `domains.py` rakip sitelerini ve
   platformları her sektörde eliyor.

Kayıtlı beş sektör aynı kodu kullanır. Bu düzeltmelerden sonra onlar için ayrıca ücretli
bir koşu yapılmadı; ölçüm, aday listesi ve özellik üretimi çevrimdışı testlerle sınanıyor.
