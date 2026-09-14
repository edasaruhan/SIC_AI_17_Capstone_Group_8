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

Tarayıcıda http://localhost:8600 açılır (FastAPI arka uç, tek sayfalık ön yüz;
`src/advisor/web/`). Kullanıcı yalnız markanın **web sitesini ya da adını** girer:

1. **Marka.** Site okunur (ana sayfa ve en fazla üç ürün sayfası, yalnız herkese açık
   adresler). Ad girildiyse önce aranır ve markanın kendi sitesi bulunur. Gemini markayı,
   diğer yazılışlarını, sektörü, dili ve ürünleri çıkarır; ürün cümleleri siteden birebir
   alıntı olmak zorundadır.
2. **Sorular.** Profilden iki tür soru yazılır: markayı anmayan keşif soruları (görünürlük
   bunlarla ölçülür) ve markayı ile ürününü anan sorular (asistan seni mi rakibini mi
   öneriyor). Kullanıcı profili ve soruları düzeltir; çağrı tahmini canlı güncellenir.
3. **Analiz.** Akışın her düğümü bittikçe listelenir.
4. **Sonuç.** Teşhis, dört ölçü, asistan yanıtları (markan sarı, rakipler gri işaretli),
   ölçülmüş etkisiyle öneriler, sitedeki ürün cümlelerinin türüne göre renklendirildiği
   açıklama incelemesi, markanı anan soruların sonucu, rakipler ve indirilebilir rapor.

Açıklama denetimi ayrı bir sayfada da kullanılabilir. Model ve site metni sayfaya yalnız
düz metin olarak yazılır.

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

## İkinci asistan

`--second-assistant` (arayüzde "gpt-oss-120b ile de ölç") aynı soruları aynı arama
sonuçlarıyla Cerebras'taki gpt-oss-120b'ye de sorar. Teşhis ve öneriler yine Gemini'nin
yanıtlarından okunur, çünkü etkiler onda ölçüldü; rapordaki "Asistanlar arasında" tablosu
sonucun tek asistana özgü olup olmadığını gösterir. Cerebras'ın hız sınırı nedeniyle çağrı
başına ~25 saniye sürer. Anahtar: `CEREBRAS_API_KEY`.

İkinci asistan, ürün açıklaması ve yapay zekâ sınıflandırması koşu kimliğini değiştirmez:
çağrıları kendi adımlarına yazılır (`ask_…__cerebras`, `audit_<özet>`). Açmak aynı klasöre
yeni makbuz ekler, eski çağrıları yeniden ödetmez.

## Açıklama denetimi

Görünürlük analizi "asistan markamı buluyor ve anıyor mu?" sorusunu yanıtlar. Denetim
ikinci soruyu yanıtlar: **ürün listede olduğunda asistan onu neye göre seçiyor, ve benim
açıklamam o bilgiyi taşıyor mu?** Varsayılan hali çağrı yapmaz ve model yüklemez; yapay zekâ
sınıflandırması tek bir Gemini çağrısıdır.

```bash
make advisor-audit AUDIT_ARGS='--file aciklama.txt --rival-file rakip1.txt --rival-file rakip2.txt'
```

Arayüzde **Açıklama denetimi** sekmesinden açılır; görünürlük analizinde de her koşunun
`describe` düğümü aynı denetimi yapar.

Kurallar açıklama deneyinin 2. turundan gelir (`reports/description_lab/*/round2/`):
beş kurgusal marka, her kartta farklı bir cümle, iki asistan (Gemini 3.5 Flash Lite,
gpt-oss-120b). Bir cümle türünün **kazanma payı**, göründüğü çağrılarda önerilen ürün
olma oranıdır; rastgele seçimde %20'dir.

| Cümle türü | Gemini | Cerebras | Denetimin kararı |
|---|---:|---:|---|
| Ürüne özgü teknik ayrıntı | %35 | %67 | İki asistanda da kazandırıyor → yoksa ekle |
| Fiyat avantajı | %35 | %50 | İki asistanda da kazandırıyor → varsa yaz |
| Sayısal ölçüm / istatistik | %37 | %22 | Asistana bağlı → tek başına güvenme |
| Bağımsız test veya denetim raporu | %32 | %3 | Asistana bağlı → tek başına güvenme |
| Puan ve kullanıcı sayısı | %23 | %3 | Belirgin kazanç yok |
| Otorite ifadesi | %13 | %27 | Belirgin kazanç yok |
| Sertifika | %7 | %23 | Belirgin kazanç yok |
| Uzman alıntısı | %10 | %15 | Belirgin kazanç yok |
| Üstünlük ifadesi ("en iyi") | %20 | %13 | Belirgin kazanç yok → yerine özellik yaz |
| Duygusal dil | %3 | %3 | Belirgin kazanç yok |
| Kaynağı gösterilmeyen kurum/klinik iddiası | %45 | %30 | **Risk** → kaldır ya da kaynağını ver |

Karar kuralı sonuçlardan önce değil sonra konmuştur ve basittir: iki asistanda da en az
%30 ise "kazandırıyor", yalnız birinde ise "asistana bağlı", hiçbirinde değilse "belirgin
kazanç yok". Kaynaksız iddia kazandırsa bile hiçbir zaman önerilmez: asistanlar iddiayı
gösterildiği yanıtların %69–79'unda kullanıcıya aktardı ve neredeyse hiç uyarmadı.

Rakip açıklamaları verilirse denetim, kazandıran bilginin rakiplerde de olup olmadığına
bakar. Deneyde bütün kartlar aynı bilgiyi taşıdığında seçimi marka tanınırlığı ve liste
sırası belirledi; herkeste olan bilgi ayırt etmez.

Cümle türleri iki yoldan bulunur:

- **Kurallar (ücretsiz).** Anahtar ifadeler; deneyde kullanılan her cümlenin kendi türüyle
  tanındığı ve temel ürün özelliklerinin hiçbir türe sayılmadığı testle doğrulanır. Ama
  kurallar deneyin iki kategorisinin cümleleriyle kuruldu: bir bankacılık açıklamasındaki
  "yıllık aidat yok" ya da "5 dakikada başvuru" onlara görünmez.
- **Yapay zekâ sınıflandırması (1 çağrı).** Gemini her cümleyi türlerden birine koyar.
  Her karar metinden birebir alıntıya dayanmak zorundadır; metinde geçmeyen cümle ya da
  bilinmeyen tür atılır. Kaynaksız iddia kuralı ayrıca uygulanır, model kaçırsa bile risk
  kaybolmaz. Çağrı hata verirse kurallara dönülür. `make advisor-audit AUDIT_ARGS='--file
  aciklama.txt --ai --yes'`.

Tablodaki sayıların raporlarla aynı olduğu testtedir. Denetim bir iddianın doğru olup
olmadığını sınamaz; her önerisi etik filtreden geçer. Yapay zekâ sınıflandırması türleri
deneyin iki kategorisinden geniş yorumlar; kazanma payları o iki kategoride ölçüldü.

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
plan → search → discover → interrogate → analyse → describe ─┬─ absent   ─┐
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
| `interrogate` | Soruyu arama kapalı ve arama sonuçlarıyla açık olarak sorar; Gemini ve istenirse gpt-oss-120b | Sorgu × 2 × tekrar × asistan |
| `analyse` | Ölçer, modeli uygular, geride kalınan sinyalleri bulur, teşhisi Gemini'ye göre koyar, her asistanın oranlarını ayrıca tutar | 0 |
| `describe` | Markanın açıklamasını (verilmediyse arama özetlerini) rakiplerin özetleriyle birlikte açıklama deneyinin kurallarıyla denetler | 0, yapay zekâ sınıflandırmasıyla 1 |
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
| **ceiling** | Anılıyorsun ama asla ilk değilsin | Aramada birinciliği içerik tek başına getirmiyor; listede seçilmek için açıklamayı güçlendir | Arama bağlamında 1.080 çağrıda 0; ürün listesinde açıklama deneyi |
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
- **Birincil asistan Gemini.** Teşhis ve öneriler Gemini 3.5 Flash Lite'ın yanıtlarından
  okunur, etkiler de bu asistanda ölçüldü; gpt-oss-120b isteğe bağlı karşılaştırmadır.
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
