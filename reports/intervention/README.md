# Kontrollü öneri testi: öneriler asistanın cevabını değiştiriyor mu?

Bu sayfa `make intervention-analyze` ile kaydedilmiş çağrılardan üretilir; elle düzenlemeyin. Tasarım ve gerekçe `src/visibility/intervention.py` başındadır; yorum ve sınırlılıklar [`bulgular.md`](bulgular.md) içindedir.

## Tasarım

- Asistan: `gemini-3.5-flash-lite`, sıcaklık 0.7, her hücre 8 tekrar.
- 15 kayıtlı arama bağlamı (sektör başına 5 sorgu, İngilizce korpustan, ilk 10 sonuç).
- Hedef markalar (önceden sabit kural: aramasız yanıtlarda %5–40 anılan, aramada en az görünen): VPN: Private Internet Access, Windscribe; Hosting: Supabase, Coolify; Seyahat: The Flight Deal, Secret Flying.
- Eklenen sayfa, sektörün iki liderini ve hedefi listeleyen bağımsız bir karşılaştırma sayfası; uygulamanın verdiği öneri ('rakiplerini anan karşılaştırma sayfalarında yer al') tam olarak budur. Liderler: VPN: Proton VPN, Mullvad; Hosting: Render, Railway; Seyahat: Kayak, Skyscanner.
- Eklenen sayfalar ayrılmış `.example` alan adlarında; hiçbir gerçek yayın taklit edilmedi. Metin ve alan adı kollar arasında sabit.
- Pilot: önce tek markalı bir inceleme sayfası denendi; 27 çağrının hiçbirinde hedef anılmadı, asistan her seferinde aynı liderleri saydı. Tam koşudan önce sayfa, verilen öneriyle uyumlu karşılaştırma biçimine getirildi. O pilot, 'tek başına bir sayfa yazmak' önerisinin bu asistanda işe yaramadığını gösterir.
- Tamamlanan çağrı: 1080/1080. Token: 1,170,920 girdi + 497,948 çıktı.

## Kollara göre oranlar (sorgu×marka hücreleri üzerinden)

| Kol | Anılma | İlk anılan marka | Gözlem |
|---|---:|---:|---:|
| Kontrol: kayıtlı arama bağlamı | %2.1 | %0.0 | 240 |
| +1 karşılaştırma sayfası, 5. sıra | %33.3 | %0.0 | 240 |
| +1 karşılaştırma sayfası, 1. sıra | %64.6 | %0.0 | 240 |
| +1 üstünlük dilli karşılaştırma sayfası, 5. sıra | %39.6 | %0.0 | 240 |
| +2 karşılaştırma sayfası (iki site), 5. ve 8. sıra | %39.6 | %0.0 | 240 |

## Önerilerin etkisi (eşleştirilmiş fark, puan; 95% hücre-bootstrap GA)

| Kapsam | Değişiklik | Ölçü | Etki | 95% GA | Hücre |
|---|---|---|---:|---|---:|
| Tümü | Bağımsız bir sonuçta görünmek (0 → 1) | Anılma | +31.2 | [+19.2, +44.2] | 30 |
| Tümü | Bağımsız bir sonuçta görünmek (0 → 1) | İlk anılan marka | +0.0 | [+0.0, +0.0] | 30 |
| Tümü | Sıra: 5 → 1 | Anılma | +31.2 | [+19.2, +44.2] | 30 |
| Tümü | Sıra: 5 → 1 | İlk anılan marka | +0.0 | [+0.0, +0.0] | 30 |
| Tümü | Dil: üstünlük iddiası vs tarafsız | Anılma | +6.2 | [-0.4, +14.2] | 30 |
| Tümü | Dil: üstünlük iddiası vs tarafsız | İlk anılan marka | +0.0 | [+0.0, +0.0] | 30 |
| Tümü | Hacim: 1 → 2 sonuç | Anılma | +6.2 | [-0.4, +13.8] | 30 |
| Tümü | Hacim: 1 → 2 sonuç | İlk anılan marka | +0.0 | [+0.0, +0.0] | 30 |
| VPN | Bağımsız bir sonuçta görünmek (0 → 1) | Anılma | +33.8 | [+12.5, +57.5] | 10 |
| VPN | Bağımsız bir sonuçta görünmek (0 → 1) | İlk anılan marka | +0.0 | [+0.0, +0.0] | 10 |
| VPN | Sıra: 5 → 1 | Anılma | +35.0 | [+12.5, +58.8] | 10 |
| VPN | Sıra: 5 → 1 | İlk anılan marka | +0.0 | [+0.0, +0.0] | 10 |
| VPN | Dil: üstünlük iddiası vs tarafsız | Anılma | +16.2 | [-2.5, +36.2] | 10 |
| VPN | Dil: üstünlük iddiası vs tarafsız | İlk anılan marka | +0.0 | [+0.0, +0.0] | 10 |
| VPN | Hacim: 1 → 2 sonuç | Anılma | +8.8 | [-2.5, +23.8] | 10 |
| VPN | Hacim: 1 → 2 sonuç | İlk anılan marka | +0.0 | [+0.0, +0.0] | 10 |
| Hosting | Bağımsız bir sonuçta görünmek (0 → 1) | Anılma | +51.2 | [+28.7, +73.8] | 10 |
| Hosting | Bağımsız bir sonuçta görünmek (0 → 1) | İlk anılan marka | +0.0 | [+0.0, +0.0] | 10 |
| Hosting | Sıra: 5 → 1 | Anılma | +20.0 | [+5.0, +38.8] | 10 |
| Hosting | Sıra: 5 → 1 | İlk anılan marka | +0.0 | [+0.0, +0.0] | 10 |
| Hosting | Dil: üstünlük iddiası vs tarafsız | Anılma | +3.8 | [-1.2, +8.8] | 10 |
| Hosting | Dil: üstünlük iddiası vs tarafsız | İlk anılan marka | +0.0 | [+0.0, +0.0] | 10 |
| Hosting | Hacim: 1 → 2 sonuç | Anılma | +0.0 | [-6.2, +5.0] | 10 |
| Hosting | Hacim: 1 → 2 sonuç | İlk anılan marka | +0.0 | [+0.0, +0.0] | 10 |
| Seyahat | Bağımsız bir sonuçta görünmek (0 → 1) | Anılma | +8.8 | [+1.2, +17.5] | 10 |
| Seyahat | Bağımsız bir sonuçta görünmek (0 → 1) | İlk anılan marka | +0.0 | [+0.0, +0.0] | 10 |
| Seyahat | Sıra: 5 → 1 | Anılma | +38.8 | [+16.2, +61.3] | 10 |
| Seyahat | Sıra: 5 → 1 | İlk anılan marka | +0.0 | [+0.0, +0.0] | 10 |
| Seyahat | Dil: üstünlük iddiası vs tarafsız | Anılma | -1.2 | [-5.0, +2.5] | 10 |
| Seyahat | Dil: üstünlük iddiası vs tarafsız | İlk anılan marka | +0.0 | [+0.0, +0.0] | 10 |
| Seyahat | Hacim: 1 → 2 sonuç | Anılma | +10.0 | [-1.2, +26.2] | 10 |
| Seyahat | Hacim: 1 → 2 sonuç | İlk anılan marka | +0.0 | [+0.0, +0.0] | 10 |

## Nasıl okunur

- *Bağımsız bir sonuçta görünmek*: kontrol ile 5. sıradaki karşılaştırma sayfası farkı. "Rakiplerini anan karşılaştırma sayfalarında yer al" önerisinin testi.
- *Sıra*: aynı sayfanın 1. sırada olması ile 5. sırada olması farkı.
- *Dil*: aynı sayfanın üstünlük iddialı hali ile tarafsız hali farkı.
- *Hacim*: ikinci bağımsız sonuç eklemenin tek sonuca göre farkı.

## Sınırlılıklar

- Tek asistan ve İngilizce bağlam. Sonuç başka modellere kendiliğinden genellenmez.
- Sayfalar kayıtlı arama sonuçlarına eklendi. Gerçek web'de bir sayfada yer almak, aramanın o sayfayı getireceğini garanti etmez; test yalnız 'getirilirse ne olur' sorusunu yanıtlar.
- Sonuç, marka adı eşleştirmesiyle ölçüldü. 'İlk anılan marka', birincil önerinin yaklaşık bir vekilidir.
- `.example` alan adları modelin güvenini düşürebilir. Etkiler bu yüzden muhafazakâr (alt sınıra yakın) okunmalı.
