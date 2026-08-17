# Referans veri seti doğrulama raporu

**Genel değerlendirme:** Ready to share
**Doğrulama tarihi:** 17 Ağustos 2026

## Amaç ve kaynak

Bu çalışma, [3RAIN/brand-bias-evaluations](https://huggingface.co/datasets/3RAIN/brand-bias-evaluations)
veri setinde raporlanan bir güçlü arama etkisini ve bir kontrol bulgusunu yeniden
üretir. `all/train` verisi
`400da04eced51d3afe52b6d20c0207fd613f8a4a` revision'ına sabitlenmiştir. Kaynağın
analiz kodu ve yöntem bağlamı
[ThreeRiversAINexus/brand-bias-evaluations](https://github.com/ThreeRiversAINexus/brand-bias-evaluations)
deposundadır.

## Yöntem

Analiz birimi tek model yanıtıdır. Marka öneri oranının paydası, ilgili kategori ve
arama koşulundaki **tüm yanıtlardır**; `top_recommendation` null olan satırlar da
paydada tutulmuştur. Veri hazırlama sırasında şu alias'lar uygulanmıştır:

- `Mullvad VPN → Mullvad`
- `Visual Studio Code → VS Code`
- `ProtonVPN → Proton VPN`

Yüzde puan değişimi `search_on − search_off` olarak hesaplanmıştır. Düşük confidence
satırları elenmemiştir.

## Veri kalitesi

| Kontrol | Sonuç |
|---|---:|
| Satır | 9.586 |
| Benzersiz `record_id` | 9.586 |
| Benzersiz sorgu | 40 |
| VPN / travel / hosting / editors | 2.388 / 2.400 / 2.398 / 2.400 |
| search-off / search-on | 4.800 / 4.786 |
| Confidence high / medium / low | 9.576 / 7 / 3 |
| JSON ayrıştırma hatası | 0 |

GLM-5/search-on altında dört hücre 30 tekrardan azdır ve olduğu gibi korunmuştur:

| Kategori | Sorgu | Satır |
|---|---|---:|
| hosting | `hosting_04` | 29 |
| hosting | `hosting_08` | 29 |
| vpn | `vpn_06` | 20 |
| vpn | `vpn_08` | 28 |

High olmayan 10 extraction etiketi (`medium=7`, `low=3`) yalnızca `travel_08`
sorgusundadır; doğrulanan VPN ve editors bulgularının hücrelerinde high olmayan
etiket yoktur. Bu satırlar kalite/sensitivity takibi için ayrılmış, kaynak bulguları
yeniden üretilirken paydadan çıkarılmamıştır.

## Bulgular ve hesap kontrolleri

| Bulgu | search-off | search-on | Değişim |
|---|---:|---:|---:|
| Mullvad, VPN | 589/1.200 = %49,08 | — | — |
| NordVPN, VPN | 63/1.200 = %5,25 | 397/1.188 = %33,42 | +28,17 yüzde puan |
| VS Code, editors | 975/1.200 = %81,25 | 970/1.200 = %80,83 | −0,42 yüzde puan |

Kesin sayımlar ve paydalar kaynak değerlerle birebir eşleşmiştir. VPN'de NordVPN
öneri payındaki güçlü artış yeniden üretilmiştir. Editör kontrolünde VS Code payı iki
koşul arasında hemen hemen sabittir.

## Sınırlamalar

- VPN search-on paydası dört eksik GLM-5 hücresi nedeniyle 1.188'dir; eksik
  tekrarlar doldurulmamış veya analizden ayrıca çıkarılmamıştır.
- Marka alanları kaynak LLM-judge çıkarımlarına dayanır; burada bağımsız elle
  etiketleme denetimi yapılmamıştır.
- Düşük confidence etiketleri ana eğitim verisinden çıkarma kararı modelleme
  sprintine aittir; eğitim/test ayrımı tekrar satırlarında değil sorgu düzeyinde
  gruplanmalıdır.
- Alias listesi belirtilen üç yazım varyantıyla sınırlıdır.
- Bulgular bu sabit veri sürümünü betimler; yeni model sürümleri veya farklı
  kullanıcı grupları için nedensel/genel bir sonuç değildir.

## Sonuç

Hedeflenen beş sayım, payda ve iki yüzde puan değişimi tam eşleştiğinden sonuç
**Ready to share** olarak değerlendirilmiştir. Çalıştırılmış hesaplar ve sınırlı
çıktılar `notebooks/S0-4-reference-validation.ipynb` dosyasındadır.
