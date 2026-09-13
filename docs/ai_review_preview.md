# AI incelemesiyle deneysel marka raporu

Bu akış, AI ön incelemesinin kullanıcı tarafından çalışma taslağı olarak kabul
edilmesi üzerine eklendi. Bağımsız insan değerlendirmesi yapıldığı anlamına gelmez.

## Girdi ve kapsam

`data/processed/evidence_v1/review/ai/annotations.csv` dosyası 30 kayıt için 60
seçilmiş iddia içerir. Her yanıttan iki iddia vardır; tüm iddiaların envanteri değildir.
26 etiket `supported`, 34 etiket `unverifiable` olarak aktarılmıştır.
Bu sayılar modelin doğruluk oranı değildir.

CSV aynı yedi alanı korur:
`record_id,brand,claim,source_ids,label,reviewer,note`.
`reviewer` alanı `AI:` ile başlar. Her not AI kökenini ve bekleyen insan
kontrolünü açıkça belirtir. Kaynak kimlikleri noktalı virgülle ayrılır.
UTF-8, virgül ayraç, çift tırnakla kaçış kullanılır. Excel'de doğrudan açılış
yerine Veri > Metin/CSV'den içe aktarma ile UTF-8 ve virgül seçilebilir.

Yanındaki `manifest.json` deney kimliğini, CSV SHA-256 değerini, kaynak JSON
hash'ini, kapsamı ve AI kökenini kaydeder. Dosyalar Git tarafından izlenmeyen
yerel verilerdir. Ekipte kullanım için CSV ve manifest birlikte paylaşılmalıdır.
Eksikse örnek/sahte etiket üretilmez. Kaynak AI JSON'u
`output/ai_review_v1/ai_review.json` dosyasıdır.

**Bu CSV'yi `review/annotations.csv` üzerine kopyalamayın.** Mevcut v1 insan
raporlayıcısı o yolu insan kanıtı kabul eder. AI dosyaları ayrı komutla işlenir.

## Komutlar

Repo kökünde, hazırlanmış veri ve `.venv` mevcutken:

```bash
make ai-review-check
make ai-brand-report BRAND="Proton VPN" DOMAIN=vpn LANGUAGE=tr
make ai-brand-report BRAND="Proton VPN" DOMAIN=vpn LANGUAGE=en
make ai-brand-report BRAND="Clinique" DOMAIN=cosmetics LANGUAGE=tr
```

Bu komutlar `.env` okumaz, API veya web isteği yapmaz, eğitim başlatmaz.
Yerel veri, mevcut taban çizgisi sonuçları ve SHAP çıktıları okunur.
Çıktılar `data/processed/evidence_v1/reports/ai_preview/` altında Markdown ve JSON'dur.
Önceki raporlar ve insan CSV'si değiştirilmez. Aynı marka/dil/domain için tekrar
çalıştırma yalnız aynı AI önizleme çıktısını yeniler.

Farklı yerel girdi için `AI_ANNOTATIONS=.../annotations.csv` verilebilir.
Yanındaki `manifest.json` aynı deneye ait ve CSV hash'i güncel olmalıdır.
Girdi değiştiğinde inceleme kapsamı ve kaynak eşleşmeleri yeniden kontrol edilmeden
sadece hash güncellenerek doğrulanmış veri gibi sunulmamalıdır.

## Rapor ne sağlar?

- Kayıtlı yanıtlarda marka anılma ve tek tercih sayıları.
- Mevcut sorgu-dışı taban çizgisi skorları/SHAP verileri, varsa.
- Marka ve dile uyan seçilmiş AI iddiaları, kaynak kimlikleri ve gerekçeleri.
- Açık `ai_reviewed_evidence_not_human_validated` durumu. İnsan inceleme durumu
  ayrı kalır; AI kaydı bunu tamamlamaz.

`supported`, kaydedilmiş başlık/snippet'in iddiayı desteklemesi demektir.
Kaynağın gerçek dünyadaki doğruluğu, klinik etkinlik, tüm ürünlere genelleme veya
görünürlük artışı kanıtlanmış değildir. `unverifiable` kesin yanlış demek değildir.
Çok markalı/bileşik alıntılar korunur; raporda aynı iddia ilgili markalarda görünebilir.
Bu yüzden marka raporlarındaki iddia sayılarını toplayıp benzersiz iddia sayısı gibi kullanmayın.

## Sonraki ürün adımı

Bu JSON, marka görünürlük arayüzü için çevrimdışı örnek sözleşmedir.
Önce marka seçimi, gözlenen görünürlük, kanıt ve öneri/belirsizlik bölümleriyle
prototip yapılabilir. Henüz yeni marka/site denetimi, canlı arama, son kullanıcı
LLM akışı veya nihai üretim modeli yoktur. Müşterinin sitesini değiştirme,
dışarıya yayınlama veya ücretli çağrı bu komutların parçası değildir.

İnsan kontrolü yapıldığında özgün insan dosyasına gerçek değerlendiricinin
kararları girilir; sonra `make evidence-review-check` kullanılır. AI önizlemesi
insan kapısını geçirmez. Eğitim etiketleri bu aşamada değiştirilmemiştir.

Yeni rapor betiği donmuş v1 kaynak koduna dokunmaz. Böylece mevcut eğitim
manifestinin hash doğrulaması korunur; gelecek deney değişiklikleri ayrı sürümlenmelidir.
