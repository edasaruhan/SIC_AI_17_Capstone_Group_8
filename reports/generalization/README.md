# M2-General ve M2-Invariant: sektör dışı genelleme ve sinyal kararlılık matrisi

Komutlar (CPU, offline; API/GPU/ağ yok):

```bash
make evidence-v2-prepare     # tamamlanmış kaynak taksonomisiyle evidence_v2
make evidence-v2-baselines   # v2 üzerinde M0–M2 (M2_full karşılaştırması için)
make modeling-generalization # bu rapordaki bütün tablolar
```

Girdi: `data/processed/evidence_v2` (manifest hash'leri doğrulanır). Üretilen tablolar
[`results.md`](results.md) ve aynı klasördeki CSV'lerde.

## Soru

Mevcut değerlendirme sorgu düzeyinde fold kullanıyor. Aynı sorgu hem eğitimde hem
testte yer almıyor, ama model test ettiği sektörü eğitimde yine görüyor. Bu yüzden
şu soru cevapsızdı:

> VPN'de öğrendiğimiz şey VPN'e mi özgü, yoksa LLM marka görünürlüğünün genel bir
> özelliği mi?

**Leave-one-domain-out (LODO)** bunun kanıtını, **sinyal kararlılık matrisi**
açıklamasını verir. **M2-Invariant** ise sinyallerin sektörden bağımsız bir biçimde
kurulup kurulamayacağını test eder.

## evidence_v2: neden yeni veri sürümü?

evidence_v1'de `configs/modeling/source_domains.json` içindeki `editorial` ve
`affiliate` listeleri boştu. Arama sonuçlarının %88'i `unknown` sayılıyordu; pcmag,
cnet, security.org, top10vpn gibi en sık görülen kaynaklar bile. Sunumdaki affiliate
hipotezi bu haliyle test edilemiyordu.

evidence_v2 yalnız kaynak kurallarını değiştirir (`configs/modeling/source_domains_v2.json`,
`src/visibility/evidence_v2.py`). Yanıtlar, adaylar, etiketler, fold'lar ve özellik
kodu v1 ile aynıdır. v1 olduğu gibi doğrulanmaya devam eder; eğitilmiş modeller ve
v1 raporları etkilenmez.

İki kural önemli:
- **En uzun eşleşen domain kazanır.** Böylece `aws.amazon.com` bir satıcı değil, AWS'nin
  kendi sitesi sayılır.
- **Rakibin kendi sitesindeki sonuç** (`vendor_other`) resmî ya da bilinmeyen sayılmaz.

| Kaynak tipi (kanıt satırı payı) | EN v1 | EN v2 | TR v1 | TR v2 |
|---|---:|---:|---:|---:|
| editoryal | %0 | %19 | %0 | %36 |
| affiliate / karşılaştırma | %0 | %8 | %0 | %11 |
| forum / kullanıcı içeriği | %10 | %28 | %7 | %12 |
| resmî site | %2 | %8 | %2 | %2 |
| satıcı | %0 | %0 | %1 | %15 |
| rakip sitesi | – | %4 | – | %0 |
| bilinmeyen | %88 | %33 | %90 | %24 |

## Tasarım (sonuçlara bakmadan önce sabitlendi)

| Karar | Gerekçe |
|---|---|
| **Prior yok**: `prior_top_*`, `prior_mention_*` çıkarıldı | Prior'lar marka kimliğidir; görülmemiş sektörde tanımsızdırlar, görülmüş sektörde SHAP'ı domine ederler. |
| **`category` yok** | Görülmemiş sektör için bu değişkenin bir seviyesi yoktur. |
| **Yalnız arama açık yanıtlar** | Arama kapalıyken bütün retrieval özellikleri sıfırdır; prior'suz model tüm adayları aynı görür. |
| **Aynı öğrenici** | `baselines.ESTIMATOR`. Değişen yalnız özellikler ve eğitim sektörleridir. |
| **Eşitlikler rastgele kırılır** | Tek bir sabit gürültü vektörü bütün modellere aynı şekilde eklenir. Aksi halde top-1, tabloda önce gelen markayı ödüllendirir. |
| **Güven aralıkları sorgu-küme bootstrap ile** | Tekrarlar bağımsız değildir. Farklar için eşleştirilmiş bootstrap kullanılır. |

### İki özellik seti

- **M2-General:** ham retrieval ve içerik sinyalleri. Kaç sonuçta anıldığı, en iyi
  sıra, kaynak tipi sayıları, snippet dil özellikleri, üretici model.
- **M2-Invariant:** aynı bilgi, **aynı yanıttaki adaylara göre** ifade edilir:
  - sonuç payı (markayı anan sonuç / yanıttaki tüm sonuçlar)
  - lidere oran ve hacim yüzdeliği
  - en iyi sırayı tutuyor mu, sıra yüzdeliği
  - kaynak karışımı oranları
  - içerik sinyallerinin aramada görünen adaylar arasındaki yüzdeliği

  "7 sonuçta anıldı" bilgisi, 10 sonuç getiren bir sektörde ve 40 sonuç getiren bir
  sektörde farklı anlama gelir; payı ve rakiplere göre sırası ise gelmez.
  Özellikler yalnız aynı yanıttaki adayların arama verisinden hesaplanır, etiket
  kullanılmaz.

### Karşılaştırılan skorlar (aynı test paneli)

| Skor | Anlamı |
|---|---|
| `naive_position` | Eğitimsiz taban çizgisi: en üst arama sonucunun markası. |
| `M2_full` | Mevcut üretim M2: prior'lı, sektörü görmüş. |
| `general_seen` / `invariant_seen` | Sorgu fold'lu: sektör eğitimde var, sorgu yok. |
| `general_unseen` / `invariant_unseen` | LODO: sektör eğitimde hiç yok. |
| `*_en_transfer` | Yalnız TR: dört İngilizce sektörde eğitilip Türkçe VPN/kozmetikte test (`model_id` olmadan; üretici modeller farklı). |

### Sinyal kararlılık matrisi

Her sinyal ve her sektör için:

1. **Etki yönü ve büyüklüğü.** Aynı yanıttaki kazanan ile kaybeden adaylar arasındaki
   rank-biserial korelasyon. İçerik sinyalleri yalnız aramada görünen adaylar
   arasında karşılaştırılır.
2. **Yön kararı.** 95% güven aralığı sıfırı dışlıyorsa ↑/↓, yoksa →. En az 5 bağımsız
   sorgu yoksa yön çağrılmaz (·).
3. **SHAP payı** (betimsel) ve **LODO ΔPR-AUC** (görülmemiş sektörde sinyal karıştırılınca
   kaybedilen PR-AUC).
4. **Hacim-eşli kontrol.** İçerik sayaçları ve "yıl var mı" bilgisi, sonuç sayısıyla
   birlikte artar. Bu yüzden her içerik sinyali bir de yalnız aynı `n_results_mentioning`
   değerine sahip kazanan–kaybeden çiftleri arasında test edilir.

Kararlılık sınıfının kuralı önceden sabitlendi:
- **Çelişkili:** anlamlı zıt yönler var
- **Güçlü:** en az 4 sektörde ve çağrılanların en az 2/3'ünde aynı yön
- **Orta:** en az 2 sektörde aynı yön
- **Zayıf:** diğer durumlar

## Bulgular

### 1. Sektörden bağımsız sinyaller kurulabiliyor: transfer açığı kapanıyor

Birincil öneri (`y_top`), İngilizce, **görülmemiş sektör**:

| Sektör | M2-General PR-AUC | **M2-Invariant PR-AUC** | M2_full (prior'lı, görmüş) | M2-Invariant top-1 | M2_full top-1 |
|---|---:|---:|---:|---:|---:|
| editors | 0,348 | **0,675** | 0,848 | %67,5 | %89,2 |
| hosting | 0,192 | **0,290** | 0,269 | %34,6 | %34,6 |
| travel | 0,249 | **0,361** | 0,554 | %49,5 | %75,6 |
| vpn | 0,256 | **0,566** | 0,688 | %62,2 | %61,4 |

- M2-Invariant görülmemiş sektörde M2-General'i dört sektörün dördünde anlamlı biçimde
  geçiyor: ΔPR-AUC +0,10 ile +0,33, hepsinde güven aralığı sıfırın üstünde.
- Sektörü görmek ile görmemek arasındaki top-1 farkı sıfıra yakın (en fazla +2,3
  puan). M2-General'de bu fark editors'ta 20, VPN'de 30 puandı.
- Hosting ve VPN'de marka kimliği bilmeyen, sektörü hiç görmemiş model, prior'lı üretim
  modelinin top-1 başarısına ulaşıyor.

**Sonuç:** Ham sayılarla öğrenilen kural sektöre özgü görünüyordu. Aynı bilgi yanıt
içi göreli biçimde ifade edilince taşınıyor. Önceki raporda "top-1 kısmen sektöre özgü"
dediğimiz farkın büyük kısmı sinyalin kendisinden değil, sayıların ölçeğinden geliyormuş.

### 2. Görünürlük (anılma) sinyalleri zaten taşınıyordu, göreli hali biraz daha iyi

Anılma hedefinde görülmüş/görülmemiş farkı M2-General'de de küçüktü (en fazla 0,031).
M2-Invariant'ta bu fark en fazla 0,017 ve dört sektörün dördünde M2-General'den iyi.
Hosting'de M2_full'dan ayırt edilemiyor.

### 3. İngilizce → Türkçe: dil ve üretici model de değişiyor

| TR sektör / hedef | Konum kuralı | M2-Invariant (EN'de eğitilmiş) | M2_full (TR'de eğitilmiş, prior'lı) |
|---|---:|---:|---:|
| VPN · birincil öneri, top-1 | %15,8 | **%42,1** | %22,8 |
| VPN · anılma, PR-AUC | 0,599 | **0,794** | 0,809 |
| Kozmetik · anılma, PR-AUC | 0,409 | **0,485** | 0,510 |

Türkçe VPN'de yalnız İngilizce veriyle eğitilen göreli model konum kuralını anlamlı
biçimde geçiyor: birincil öneride +0,27 PR-AUC [0,04, 0,48], anılmada +0,19 [0,13, 0,22].
Birincil öneride nokta tahmini Türkçe veriyle eğitilen prior'lı modelin de üstünde, ama
ikisi arasında eşleştirilmiş test yapılmadı ve aralıklar örtüşüyor. Türkçe veri küçük
(VPN'de 57 kazanan, 5 sorgu). Kozmetik birincil öneride yalnız
3 sorgu var, kanıt değildir.

### 4. Hangi sinyal genel? Kararlılık matrisi (hacim-eşli)

| Sinyal | Marjinal | Hacim-eşli | Yorum |
|---|---|---|---|
| Aramada görünmek | Güçlü | Güçlü | Her sektörde ve iki dilde belirleyici. |
| Kaç sonuçta anıldığı | Güçlü | Güçlü | En büyük transfer katkısı (LODO ΔPR-AUC 0,10–0,12). |
| Snippet uzunluğu | Orta/Güçlü | Orta | Hacimden bağımsız kalan tek içerik sinyali. |
| Üstünlük dili | Orta/Çelişkili | Çelişkili | Çoğu sektörde **ters** yönde. Genel bir "üstünlük iddiası kazandırır" kuralı yok. |
| En iyi sıra | Güçlü | Zayıf | Eşit hacimli rakipler arasında sıra tutarlı biçimde ayırt etmiyor; göreli sıra (M2-Invariant) ise transferde işe yarıyor. |
| Affiliate kaynak sayısı | Güçlü | Zayıf | Marjinal ilişki hacimden geliyor. |
| Editoryal kaynak sayısı | Orta/Çelişkili | Orta (anılmada ters yön) | Eşit hacimde editoryal kaynak payı anılmayla ters ilişkili; editoryal listeler çok marka sayıyor olabilir. |
| Otorite/sosyal kanıt/özgüllük dili, yıl | çoğu Güçlü | Zayıf/Orta/Çelişkili | Marjinal etkinin çoğu "daha çok sonuç, daha çok metin" etkisi. |

**Sonuç:** Sektörler ve diller arasında taşınan genel sinyal içerik dili değil,
**retrieval'daki göreli konum**. Kaç bağımsız sonuçta, rakiplere göre ne kadar sık ve
ne kadar üstte anıldığı belirleyici. Kaynak tipinin hacimden bağımsız bir etkisi
görülmedi. Affiliate hipotezi bu gözlemsel veriyle **desteklenmiyor**: affiliate
kaynakların etkisi, markayı anan sonuç sayısının etkisinden ayrılamıyor.

### Yan bulgu: konum taban çizgisinin eski top-1 değerleri sıra artefaktı

Arama açık yanıtların %78'inde birden fazla aday aynı en iyi konumu paylaşıyor. Eski
değerlendirme bu eşitlikleri tablo sırasıyla (`idxmax`) kırıyordu. Bu raporda
eşitlikler bütün modellerde aynı sabit tohumlu gürültüyle rastgele kırılır. Sprint 2
ve evidence_v1 raporlarındaki `naive_position` top-1 değerleri bu nedenle güvenilir
değildir.

## Sınırlılıklar

- **Az bağımsız sorgu.** İngilizce sektör başına 10, Türkçe 5 sorgu. Aralıklar geniş.
- **Etkiler gözlemseldir.** Hacim-eşli karşılaştırma bile nedensel değildir: bir markanın
  çok sonuçta anılması zaten tanınmış olmasının sonucu olabilir. Nedensel test
  [`reports/intervention/`](../intervention/) altındadır.
- **`*_seen` kimliği kısmen öğrenebilir.** Prior olmasa da aynı sektörün diğer
  sorgularında aynı markaların sayfalarını görür. Transfer açığı bu yüzden bir üst
  sınırdır.
- **Kaynak taksonomisi kural tabanlıdır.** En sık domainler elle sınıflandırıldı;
  kanıt satırlarının %33'ü (EN) hâlâ `unknown`. İki bağımsız işaretleyiciyle
  doğrulanmadı.
