# Yoğunlaşma ve adalet: kim vitrine çıkabiliyor?

Tablolar [`results.md`](results.md) içinde, `make modeling-fairness` ile üretilir.
Ölçülerin tanımı ve sonuçlara bakılmadan sabitlenen kurallar
`src/visibility/fairness.py` başındadır. Bu sayfa o tabloların elle yazılmış yorumudur.

**Ana ölçü `N_eff = 1/HHI`:** asistan, sektörde *kaç* marka varmış gibi davranıyor.
Registry'de kaç marka olduğuyla karşılaştırılarak okunur. Ham HHI sektörler arasında
karşılaştırılamaz (alt sınırı `1/N`), Gini ise burada asistan davranışını değil bizim
marka listemizin uzunluğunu ölçerdi; ikisi de bu yüzden başlık sayısı değildir.

## 1. Birincil öneri neredeyse tek markaya gidiyor

| Sektör (registry markası) | N_eff · aramasız | N_eff · aramalı | İlk-3 payı |
|---|---:|---:|---:|
| Kod editörleri (30) | **1,14** | **1,25** | %97 |
| VPN · EN (24) | 1,32 | 2,45 | %98 |
| Seyahat (27) | 1,48 | 1,71 | %90 |
| Kozmetik · TR (74) | 4,15 | 3,60 | %79 |
| VPN · TR (24) | 3,88 | 3,95 | %81 |
| Hosting (36) | 4,62 | 5,44 | %67 |

Kod editörlerinde 30 markalık bir listeden asistan fiilen **bir** marka varmış gibi
davranıyor; birincil önerilerin %97'si ilk üç markaya gidiyor. Sunumdaki "on bağlantı
yerine üç isim" tespiti, birincil öneri düzeyinde üçten de dar: çoğu sektörde tek isim.

Anılma düzeyinde tablo daha yumuşak — VPN'de 24 markadan ~6,5, hosting'de 36'dan ~19,
kozmetikte 74'ten ~29 — ama "anılmak" ile "önerilmek" arasındaki bu fark, projenin
kontrollü testindeki bulguyla birebir örtüşüyor: içerik ve kaynak stratejisi markayı
listeye sokuyor, birinciliğe taşımıyor.

## 2. Arama pazarı daraltmıyor, hafifçe genişletiyor

Beklenti, ticari içerikle dolu aramanın yoğunlaşmayı artırması yönündeydi. Veri bunu
göstermiyor: arama açıldığında etkin marka sayısı ya sabit kalıyor ya artıyor
(EN VPN birincil öneride 1,32 → 2,45; TR kozmetik anılmada 28,9 → 35,9; hosting
18,9 → 20,8). Hiçbir sektörde anlamlı bir daralma yok.

Bu, "arama her şeyi bozuyor" cümlesini desteklemiyor. Aramanın etkisi seçici:
*kimin* kazandığı sektöre göre değişiyor (§3), ama *kaç markanın* göründüğü
genel olarak artıyor.

## 3. Aramadan kim kazanıyor? Sektöre göre değişiyor

Markalar arama-kapalı anılma oranına (tanınırlık vekili) göre üçe bölündü; terciller
bir kez, tam örneklemden atandı.

| Sektör | Az tanınan tercil | Çok tanınan tercil |
|---|---|---|
| Hosting | **+2,7** [+1,2 · +4,2] | **−11,8** [−14,5 · −9,1] |
| Kozmetik · TR | **+5,3** [+1,8 · +9,3] | −1,2 [−5,2 · +3,2] |
| Kod editörleri | +0,8 [−1,0 · +2,8] | −4,0 [−9,5 · +2,1] |
| Seyahat | +1,8 [−0,8 · +4,0] | +2,8 [−1,0 · +7,5] |
| VPN · EN | +0,4 [−0,0 · +1,1] | +3,8 [−1,0 · +9,7] |
| VPN · TR | **+1,2** [+0,5 · +2,0] | **+9,5** [+5,5 · +14,8] |

İki zıt örüntü var. Hosting ve Türkçe kozmetikte arama **yeniden dağıtıyor**: az
tanınan markalar kazanıyor, yerleşikler kaybediyor — hosting'de lider tercil 11,8 puan
geriliyor. Türkçe VPN'de ise tam tersi: arama **yerleşiği büyütüyor**, lider tercil
9,5 puan kazanıyor.

Fark muhtemelen sektörün kaynak yapısından geliyor: hosting ve kozmetikte çok sayıda
karşılaştırma ve inceleme sayfası uzun kuyruğu getiriyor; VPN aramasında ise aynı
birkaç liderin bulunduğu listeler dönüyor. Bu bir hipotezdir; test edilmedi.

**"Az tanınan" şirket büyüklüğü demek değildir.** Elimizdeki tek tanınırlık ölçüsü
modelin arama kapalıyken markayı anma oranıdır; ciro, pazar payı veya çalışan sayısı
verisi yok. İki kavramı karıştırmayın.

## 4. Yerli marka farkı: ölçülebilir bir dezavantaj görünmüyor

| Menşe (TR kozmetik) | Marka | Aramasız | Aramalı | Aramadan kazanç |
|---|---:|---:|---:|---|
| Türkiye menşeli | 9 | %9,5 | %11,0 | +1,5 [−0,6 · +3,9] |
| Küresel | 65 | %10,8 | %13,0 | +2,3 [+0,4 · +4,3] |

Türkiye menşeli markalar aramasız koşulda küresel markalarla neredeyse aynı oranda
anılıyor ve aramadan biraz daha az kazanıyor; aralıklar geniş ölçüde örtüşüyor.
**Ne dezavantaj ne eşitlik iddia edilebilir** — 9 marka ve 5 bağımsız sorgu ile bu
büyüklükte bir farkı ayırt edecek güç yok. Bulgu "fark bulunamadı"dır, "fark yoktur"
değil.

Menşe kırılımı yalnız Türkçe kozmetikte mümkün: Türkçe VPN registry'sindeki 24
markanın hiçbiri Türkiye menşeli değil, İngilizce korpusta da yok. Etiketler
[`configs/visibility/brand_origin.yaml`](../../configs/visibility/brand_origin.yaml)
dosyasındadır ve analiz kodu yazılmadan önce ayrı bir commit'te mühürlenmiştir;
`fairness.py` dosyanın hash'ini sabit tutar, değişirse rapor üretilmez.

## Sınırlılıklar

- **Gözlemsel.** Buradaki hiçbir sayı nedensel değildir. Tek nedensel kanıt
  [`reports/intervention/`](../intervention/) altındadır ve o da menşe değil,
  kaynak/sıra müdahalesi ölçer.
- **Marka evreni bizim seçimimiz.** Registry'de olmayan marka hiçbir zaman anılmamış
  görünür; "hiç anılmayan marka" sayıları bu yüzden betimseldir.
- **Tanınırlık ≠ şirket büyüklüğü** (§3).
- **Menşe etiketleri insan doğrulaması bekliyor.** Model bilgisinden taslaklandı;
  `confidence: low` satırları önce doğrulanmalı. Marka kökeni ile bugünkü sahiplik
  ayrı alanlarda: Flormar Türkiye'de kurulmuştur ama 2012'den beri Fransız bir gruba
  aittir; analiz kökeni esas alır, dosya iki tanımla da yeniden üretilebilir.
- **Dil karşılaştırması yapılmadı.** EN ve TR korpusları farklı asistanlar, farklı
  sorgular ve farklı toplama pencereleriyle üretildi; aradaki fark dil etkisi değildir.
- **Türkçe aralıklar 5 bağımsız sorgudan** gelir; genişlikleri gerçek belirsizliği
  alttan tahmin ediyor olabilir.
- **Etiketler judge modelinin çıkarımına dayanır.** 30 kayıtlık insan incelemesi
  tamamlanmadı; bu belirsizlik nicelenmedi.
- **Tek zaman penceresi.** Sağlayıcı modelleri güncellendiğinde sonuçların kayıp
  kaymadığı ölçülmedi.
