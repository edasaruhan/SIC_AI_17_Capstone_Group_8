# Maskeleme ablasyonu: tahmin ne kadar marka adına dayanıyor?

Bu sayfa `scripts/collect_masking.py` tarafından üretilir; elle düzenlemeyin.
Yöntem ve sınırlılıklar betiğin başındadır.

Snippet'lerdeki her marka adı `[BRAND]` ile değiştirilip M3 yeniden eğitildi.
Fark, aynı satırlar üzerinde eşleştirilmiş ve tek bir sorgu-küme bootstrap'ı ile
hesaplandı. Hedef: birincil öneri (`y_top`), seed 7, iki epoch.

| Track | Sektör | Çift | İsimli PR-AUC | Maskeli PR-AUC | ΔPR-AUC [95% GA] | ΔTop-1 [95% GA] | Sorgu |
|---|---|---:|---:|---:|---|---|---:|
| en | Kod editörleri | 32610 | 0.841 | 0.427 | 0.414 [0.213, 0.589] | 0.463 [0.310, 0.593] | 10 |
| en | Hosting / bulut | 26460 | 0.210 | 0.138 | 0.072 [0.017, 0.132] | -0.007 [-0.103, 0.132] | 10 |
| en | Seyahat | 20763 | 0.642 | 0.558 | 0.084 [-0.128, 0.317] | 0.179 [0.053, 0.321] | 10 |
| en | VPN | 18525 | 0.735 | 0.389 | 0.347 [0.136, 0.491] | 0.232 [0.087, 0.396] | 10 |
| tr | Kozmetik | 1776 | 0.055 | 0.072 | -0.017 [-0.018, 0.164] | 0.125 [0.000, 0.429] | 3 |
| tr | VPN | 1083 | 0.146 | 0.120 | 0.026 [-0.080, 0.149] | -0.018 [-0.213, 0.180] | 5 |

Pozitif ΔPR-AUC, marka adı silindiğinde tahminin **kötüleştiğini** gösterir:
model o sektörde kimin kazandığını kısmen isimden biliyordu. Sıfıra yakın veya
negatif bir fark, o sektörde tahminin isimden çok sayfa içeriğine dayandığı
anlamına gelir.

Güven aralığı sıfırı içeren satırlarda yön çağrılmaz.
