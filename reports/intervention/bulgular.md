# Kontrollü öneri testi: bulgular ve yorum

Tablolar [`README.md`](README.md) içinde, `make intervention-analyze` ile üretilir. Bu
sayfa o tabloların elle yazılmış yorumudur.

**Kapsam.** Planlanan 1.080 çağrının **1.080'i tamamlandı**; hatalı veya eksik adım yok.
Otuz hücrenin (sorgu × hedef marka) hepsinde 8 tekrar var, her kolda 240 gözlem.
Koşu üç oturuma yayıldı: ilk ikisi Gemini anahtarının günlük kotasında (HTTP 429)
durdu, üçüncüsü kalanı tamamladı. İkinci oturumdan itibaren çağrılar tekrar-öncelikli
sırada gönderildi (önce bütün hücrelerin 1. tekrarı, sonra 2. tekrarı…), böylece kota
kesintileri sektörleri değil yalnız son tekrarları eksik bıraktı.

## Ne test edildi?

Uygulamanın önerileri, tek bir asistanda (Gemini 3.5 Flash Lite) ölçüldü. Kayıtlı
gerçek bir arama bağlamına bir değişiklik eklendi ve aynı soru 8 kez soruldu. Hedefler,
asistanın bu sorularda tanıdığı ama aramanın nadiren getirdiği markalardır (önceden
sabit kural):
- **VPN:** Private Internet Access, Windscribe
- **Hosting:** Supabase, Coolify
- **Seyahat:** The Flight Deal, Secret Flying

## Sonuçlar (anılma olasılığı, puan; 95% güven aralığı)

| Öneri | Tümü | VPN | Hosting | Seyahat |
|---|---|---|---|---|
| Rakipleri anan bir karşılaştırma sayfasında yer almak (kontrole göre) | **+31,2** [+19,2 · +44,2] | **+33,8** [+12,5 · +57,5] | **+51,2** [+28,7 · +73,8] | **+8,8** [+1,2 · +17,5] |
| Aynı sayfanın 5. yerine 1. sırada olması | **+31,2** [+19,2 · +44,2] | **+35,0** [+12,5 · +58,8] | **+20,0** [+5,0 · +38,8] | **+38,8** [+16,2 · +61,3] |
| Aynı sayfada üstünlük dili ("en iyi", "#1") | +6,2 [−0,4 · +14,2] | +16,2 [−2,5 · +36,2] | +3,8 [−1,2 · +8,8] | −1,2 [−5,0 · +2,5] |
| Aynı metnin ikinci bir sitede de olması | +6,2 [−0,4 · +13,8] | +8,8 [−2,5 · +23,8] | +0,0 [−6,2 · +5,0] | +10,0 [−1,2 · +26,2] |
| Markanın **ilk anılan** marka olması (bütün kollarda) | 0 | 0 | 0 | 0 |

Kontrol kolunda hedefler yanıtların %2,1'inde anılıyordu. Karşılaştırma sayfası 5.
sıradayken bu oran %33'e, 1. sıradayken %65'e çıktı.

## Yorum

1. **"Rakiplerini anan karşılaştırma sayfalarında yer al" önerisi işe yarıyor.**
   Asistanın tanıdığı ama aramada görünmeyen bir marka, tek bir bağımsız karşılaştırma
   sayfasına girince çok daha sık anılıyor. Gözlemsel analizin en güçlü sinyali olan
   "aramada görünmek ve çok sonuçta anılmak", kontrollü koşulda da nedensel bir etki
   olarak tekrar ediyor.
2. **Sıra en az girmek kadar önemli.** Aynı sayfanın 1. sırada olması anılmayı %33'ten
   %65'e çıkarıyor; etkisi "hiç görünmemekten 5. sıraya çıkmak" ile aynı büyüklükte.
   Bu, M2-Invariant'ın göreli sıra sinyalinin transferdeki başarısıyla uyumlu.
3. **Sektörler aynı tepkiyi vermiyor.** Hosting'de tek bir karşılaştırma sayfası 5.
   sırada bile anılmayı %5'ten %56'ya çıkardı (+51 puan). Seyahatte aynı sayfanın 5.
   sıradaki etkisi çok küçük (+8,8 puan); asıl etki sayfa 1. sıraya alınınca ortaya
   çıkıyor (+38,8). Yani "listelere gir" tavsiyesi her sektörde tek başına yeterli
   değil; asistanın yerleşik lider listesi güçlü olduğunda üst sıra şart oluyor.
4. **Görünmek birinci önerilmek değildir.** 1.080 çağrının hiçbirinde hedef marka ilk
   anılan marka olmadı; asistan her seferinde önce liderleri saydı. Bu, gözlemsel
   analizdeki "birincil önerinin büyük kısmını marka kimliği taşıyor" bulgusuyla
   tutarlı. İçerik ve kaynak stratejisi listeye girmeyi sağlıyor, liderliği sağlamıyor.
5. **Üstünlük dili: ölçülebilir bir kazanç yok.** Tam örneklemde etki +6,2 puan ve güven
   aralığı sıfırı içeriyor; seyahatte işaret ters. Koşunun VPN ağırlıklı ilk yarısındaki
   "+13 [+2 · +27]" tahmini, diğer iki sektör tamamlanınca ayakta kalmadı — küçük
   örneklemde anlamlı görünen bir farkın nasıl kaybolabileceğinin örneği. Sonuç artık
   gözlemsel bulguyla da aynı yönde: üstünlük dili sektörler arasında taşınmıyor.
   Arayüzün "tek başına strateji yapma" uyarısı yerinde.
6. **Kopyalamak işe yaramıyor.** Aynı metnin ikinci bir sitede de bulunması anlamlı bir
   ek etki yaratmadı (+6,2 puan, aralık sıfırı içeriyor). Arayüzün "tek bir bültenin
   kopyaları yerine ayrı içerik" uyarısı bununla uyumlu, ama farklı içerikli ikinci bir
   kaynak test edilmedi.
7. **Pilot bulgusu.** Tek markalı bir inceleme sayfası 27 çağrının hiçbirinde hedefi
   anılır yapmadı. Yalnızca kendi hakkında yazılmış bir sayfa, bu asistanın liste
   yanıtlarını değiştirmiyor.

## Sınırlılıklar

- Tek asistan ve İngilizce bağlam; sonuçlar başka modellere kendiliğinden genellenmez.
- Sektör başına 10 hücre var; sektör kırılımlarının güven aralıkları geniştir ve
  yukarıdaki sektör farkları yön göstericidir, kesin büyüklük değildir.
- Sayfalar kayıtlı arama sonuçlarına eklendi. Test "sayfa aramada gelirse ne olur"
  sorusunu yanıtlar; gerçek web'de bir sayfada yer almak aramanın onu getireceğini
  garanti etmez.
- Sonuçlar marka adı eşleştirmesiyle ölçüldü. "İlk anılan marka", birincil önerinin
  yaklaşık bir vekilidir.
- Eklenen sayfalar `.example` alan adlarında. Modelin bunlara güveni gerçek yayınlardan
  düşük olabilir; etkiler bu açıdan muhafazakârdır.
- Hedefler ve kollar pilotlardan sonra, tam koşudan önce sabitlendi. Tek markalı sayfa
  pilotu sonrasındaki tasarım değişikliği yukarıda ve kodun başında belgelenmiştir.
