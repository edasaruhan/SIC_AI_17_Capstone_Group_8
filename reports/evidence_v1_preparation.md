# evidence_v1 hazırlık ve doğrulama notu

Tarih: 2026-09-07. Başlangıç: birleşmiş S2 commit'i `ab7a3de`.
Yeni dal: `feat/evidence-backed-brand-reports`. Üretim veri seti yeniden toplanmadı.
Hiçbir Serper, generation veya Cerebras judge çağrısı yapılmadı.

**Sonraki aşama güncellemesi:** Aşağıdaki GPU durumu bu hazırlık notunun ilk
yazıldığı ana aittir. Aynı gün Türkçe listwise eğitimleri tamamlandı;
[gerçek eğitim sonuçları ve kalan kapılar](listwise_tr_seed7.md) ayrı nottadır.

## Uygulananlar

- Türkçe veri indirmesi revision ve kaynak hash'ine sabitlendi. Tam 300 deney
  hücresi, sorgu metni, tekrar ve dil denetleniyor. Her iki girdinin kanonik
  içerik parmak izi de sürümlü analiz hattında doğrulanıyor.
- S2 PDF'sindeki nedensel katkı alt/üst sınırı yorumu düzeltildi. Eski skor
  tabloları korunuyor; kaynak PDF `ab7a3de` commit'inden geri alınabilir.
- Aday havuzu kayıtlı markaların tamamı; test yanıtlarıyla daraltılmıyor.
- Eşleşmeyenler dahil bütün arama sonuçları ayrı tabloda; sorgu, tur, sonuç
  sırası, URL ve snippet kaybolmuyor. Eşleşme destek kanıtı sayılmıyor.
- Sürümü sabit 3 pilot / 30 kayıtlık insan inceleme paketi oluşturuldu.
- Kaynaklı Markdown/JSON marka raporları ve sorgu-dışı model açıklamaları hazır.
- Aynı öğrenicili M0/M1/M2, iç sorgu bölmeli prior ve hedefe uygun frekans
  baseline'ı eklendi. Listwise eğitim ayrı ve açık komutla çalışır.

## Gerçek veriyle üretilen tablolar

| Dil | Yanıt | Bütün organik sonuçlar | Marka-kaynak eşleşmesi | Aday çiftleri |
|---|---:|---:|---:|---:|
| EN | 9.586 | 121.733 | 172.379 | 280.440 |
| TR | 300 | 1.440 | 2.299 | 14.700 |

Kaynak sonuçları tekrar gösterimleri içerir; bu sayılar benzersiz URL sayısı
değildir. Yeni eşleştirme ve aday politikası nedeniyle eski S2 sayılarıyla
aynı olmaları beklenmez. Tablolar ve manifest `data/processed/evidence_v1/`
altında kalır, Git'e eklenmez.

## Testler ve ilk model kontrolü

`make check`: **109 test başarılı**, ruff, black ve pyright başarılı.
Yeni testler ağ bağlantısını engelleyerek fixture verisi kullanır; eski toplama
testleri de mock tabanlıdır. Örnek raporlar gerçek kayıtlarla NordVPN ve CeraVe
için oluşturuldu. Sonuçlar garanti edilmiş görünürlük artışı içermez.

EN ve TR için hem `y_top` hem `y_mention` CPU koşuları tamamlandı. Türkçe komut
tekrar çalıştırıldığında tamamlanmış skor/SHAP dosyalarının değiştirilmediği
dosya zamanlarıyla doğrulandı; yeniden eğitim yapılmadı.

Türkçe sorgu-dışı PR-AUC (yuvarlanmış, v1 hedef/kapsam tanımıyla):

| Domain / hedef | Frekans baseline | M1 | M2 |
|---|---:|---:|---:|
| VPN / tek kazanan | 0,212 | 0,263 | 0,278 |
| Kozmetik / tek kazanan | 0,068 | 0,176 | 0,149 |
| VPN / anılma | 0,572 | 0,722 | 0,725 |
| Kozmetik / anılma | 0,249 | 0,389 | 0,387 |

M2 her durumda en iyi değildir. Bu tablo nedensel etki veya istatistiksel
üstünlük iddiası değildir. Tam metrikler, sorgu sayıları ve sorgu-bootstrap
aralıkları yerel `baselines/metrics_*.json` çıktılarındadır. İnsan incelemesi
tamamlanmadığı için kaynak etiketlerinin doğruluğu henüz onaylanmış değildir.

## Henüz tamamlanmayan doğrulama kapıları

- **İnsan incelemesi: 0/30.** CSV boş ve `unreviewed`; etiketler uydurulmadı.
  Tamamlama kontrolünün hata vermesi doğrulandı. İlk iş `review/pilot.md`
  içindeki üç kaydı birlikte incelemek.
- **GPU eğitimi çalıştırılmadı.** İsteğe bağlı torch/transformers kurulumu yok;
  model indirilmedi. İsimli/maskeli listwise giriş panellerinin aynı aday/etiket
  satırlarını taşıdığı gerçek veride doğrulandı: EN VPN 975 dahil / 213 hariç,
  TR VPN 57 dahil / 18 hariç. Bunlar arama-açık yanıt sayılarıdır.
- Listwise kayıp/grup/maskeleme mantığı offline test edildi; gerçek transformer
  backprop, GPU bellek kullanımı ve epoch-checkpoint devamı GPU smoke aşamasında
  ayrıca doğrulanmalıdır. Hazır kod, yapılmış eğitim gibi sunulmaz.
- Yeni marka/canlı site analizi, LLM arayüzü, E-GEO, LLM-only karşılaştırması ve
  önerilerin gerçek görünürlük etkisi sonraki ürün/doğrulama aşamalarıdır.

Çalıştırma sırası ve dosya sözleşmeleri:
[Kaynaklı marka analizi rehberi](../docs/kaynakli-marka-analizi.md).
