# Açıklama deneyi, 2. tur: hangi cümle kazanıyor? — Gemini 3.5 Flash Lite

Bu sayfa `python -m description_lab analyze --round 2` ile üretilir; elle düzenlemeyin.
Tasarım `src/description_lab/round2.py` başındadır.

- Asistan: `gemini-3.5-flash-lite`, sıcaklık 0.7.
- Tamamlanan çağrı: 256/256.
- Önerilen marka 1. turun kuralıyla okunur; kural bulamayıp ilk anılana düşülen çağrı: 0.

## 1. Turnuva: her kartta farklı bir cümle

Beş kurgusal marka, her kartta 13 cümleden farklı biri. Her cümle her sıraya eşit sayıda konur. Şans payı %20.

**Tümü**

| Cümle | Görünme | Kazanma | Kazanma payı [95% GA] |
|---|---:|---:|---|
| Uydurma klinik/denetim iddiası | 60 | 27 | %45 [%32 · %58] |
| Sayısal kanıt / istatistik | 60 | 22 | %37 [%25 · %49] |
| Teknik detay | 60 | 21 | %35 [%23 · %47] |
| Fiyat avantajı | 60 | 21 | %35 [%23 · %47] |
| Bağımsız test raporu gösterme | 60 | 19 | %32 [%20 · %44] |
| Sosyal kanıt (puan, kullanıcı sayısı) | 60 | 14 | %23 [%13 · %35] |
| Üstünlük dili ('en iyi', 'bir numara') | 60 | 12 | %20 [%11 · %31] |
| Otorite dili | 60 | 8 | %13 [%5 · %23] |
| Uzman alıntısı | 60 | 6 | %10 [%3 · %18] |
| Sertifika | 60 | 4 | %7 [%1 · %13] |
| Duygusal dil | 60 | 2 | %3 [%0 · %8] |
| Kontrol (yalnız temel özellikler) | 60 | 0 | %0 [%0 · %0] |
| Bilgisiz dolgu cümlesi (uzunluk kontrolü) | 60 | 0 | %0 [%0 · %0] |

**Güneş kremi**

| Cümle | Görünme | Kazanma | Kazanma payı [95% GA] |
|---|---:|---:|---|
| Teknik detay | 30 | 18 | %60 [%42 · %77] |
| Sosyal kanıt (puan, kullanıcı sayısı) | 30 | 12 | %40 [%23 · %58] |
| Sayısal kanıt / istatistik | 30 | 12 | %40 [%23 · %57] |
| Fiyat avantajı | 30 | 11 | %37 [%21 · %55] |
| Uydurma klinik/denetim iddiası | 30 | 10 | %33 [%17 · %50] |
| Üstünlük dili ('en iyi', 'bir numara') | 30 | 6 | %20 [%7 · %36] |
| Uzman alıntısı | 30 | 4 | %13 [%3 · %26] |
| Bağımsız test raporu gösterme | 30 | 2 | %7 [%0 · %17] |
| Otorite dili | 30 | 2 | %7 [%0 · %17] |
| Sertifika | 30 | 1 | %3 [%0 · %11] |
| Kontrol (yalnız temel özellikler) | 30 | 0 | %0 [%0 · %0] |
| Duygusal dil | 30 | 0 | %0 [%0 · %0] |
| Bilgisiz dolgu cümlesi (uzunluk kontrolü) | 30 | 0 | %0 [%0 · %0] |

**VPN**

| Cümle | Görünme | Kazanma | Kazanma payı [95% GA] |
|---|---:|---:|---|
| Bağımsız test raporu gösterme | 30 | 17 | %57 [%38 · %74] |
| Uydurma klinik/denetim iddiası | 30 | 17 | %57 [%39 · %74] |
| Sayısal kanıt / istatistik | 30 | 10 | %33 [%17 · %52] |
| Fiyat avantajı | 30 | 10 | %33 [%17 · %52] |
| Otorite dili | 30 | 6 | %20 [%7 · %35] |
| Üstünlük dili ('en iyi', 'bir numara') | 30 | 6 | %20 [%7 · %35] |
| Sertifika | 30 | 3 | %10 [%0 · %21] |
| Teknik detay | 30 | 3 | %10 [%0 · %22] |
| Uzman alıntısı | 30 | 2 | %7 [%0 · %17] |
| Duygusal dil | 30 | 2 | %7 [%0 · %16] |
| Sosyal kanıt (puan, kullanıcı sayısı) | 30 | 2 | %7 [%0 · %17] |
| Kontrol (yalnız temel özellikler) | 30 | 0 | %0 [%0 · %0] |
| Bilgisiz dolgu cümlesi (uzunluk kontrolü) | 30 | 0 | %0 [%0 · %0] |

## 2. Dolgu cümlesi: fark mı, bilgi mi?

Hedef kurgusal markanın tek cümlesi bilgisiz dolgu. 1. turun aynı asistandaki kontrolü (cümlesiz) ve on bir cümlenin ortalamasıyla karşılaştırılır.

| Kapsam | n | Dolgu ile birinci önerilme | 1. tur kontrol | 1. tur cümleler | Dolgu − kontrol (puan) | Cümleler − dolgu (puan) |
|---|---:|---|---:|---:|---|---|
| Tümü | 20 | %15 [%0 · %30] | %10 | %99 | +5.0 [-15.0 · +25.0] | +83.6 [+67.3 · +98.6] |
| Güneş kremi | 10 | %20 [%0 · %50] | %10 | %99 | +10.0 [-20.0 · +40.0] | +79.1 [+50.0 · +100.0] |
| VPN | 10 | %10 [%0 · %30] | %10 | %98 | +0.0 [-30.0 · +30.0] | +88.2 [+67.3 · +100.0] |

## 3. Çaprazlama: cümle, tanınan markayı yenebilir mi?

Aynı listede küresel yerleşik marka (cümlesiz) ve kurgusal marka (cümleyle), yanında üç gerçek rakip.

| Kapsam | Kurgusal markanın cümlesi | n | Kurgusal önerildi | Yerleşik önerildi |
|---|---|---:|---|---|
| Tümü | Kontrol (yalnız temel özellikler) | 20 | %10 [%0 · %25] | %55 [%35 · %75] |
| Tümü | Sayısal kanıt / istatistik | 20 | %100 [%100 · %100] | %0 [%0 · %0] |
| Tümü | Üstünlük dili ('en iyi', 'bir numara') | 20 | %100 [%100 · %100] | %0 [%0 · %0] |
| Tümü | Uydurma klinik/denetim iddiası | 20 | %95 [%85 · %100] | %0 [%0 · %0] |
| Güneş kremi | Kontrol (yalnız temel özellikler) | 10 | %10 [%0 · %30] | %40 [%10 · %70] |
| Güneş kremi | Sayısal kanıt / istatistik | 10 | %100 [%100 · %100] | %0 [%0 · %0] |
| Güneş kremi | Üstünlük dili ('en iyi', 'bir numara') | 10 | %100 [%100 · %100] | %0 [%0 · %0] |
| Güneş kremi | Uydurma klinik/denetim iddiası | 10 | %90 [%70 · %100] | %0 [%0 · %0] |
| VPN | Kontrol (yalnız temel özellikler) | 10 | %10 [%0 · %30] | %70 [%40 · %100] |
| VPN | Sayısal kanıt / istatistik | 10 | %100 [%100 · %100] | %0 [%0 · %0] |
| VPN | Üstünlük dili ('en iyi', 'bir numara') | 10 | %100 [%100 · %100] | %0 [%0 · %0] |
| VPN | Uydurma klinik/denetim iddiası | 10 | %100 [%100 · %100] | %0 [%0 · %0] |

## 4. Uydurma iddia: seçiliyor mu, tekrarlanıyor mu, sorgulanıyor mu?

İddianın gösterildiği çağrılar. 'Tekrar' = yanıt iddianın kaynağını (Harvard/MIT) anıyor. 'Uyarı' anahtar kelime eşleşmesidir, üst sınırdır.

| Blok | n | İddialı kart önerildi | İddia tekrarlandı | Önerilmediğinde bile tekrar | Uyarı |
|---|---:|---|---|---|---|
| Tümü | 80 | %57 [%46 · %68] | %79 [%70 · %88] | %56 [%38 · %74] | %0 [%0 · %0] |
| Çaprazlama | 20 | %95 [%85 · %100] | %100 [%100 · %100] | %100 [– · –] | %0 [%0 · %0] |
| Turnuva | 60 | %45 [%33 · %58] | %72 [%60 · %83] | %55 [%36 · %70] | %0 [%0 · %0] |

## 5. Turnuvada önerilen kartın sırası

| Kapsam | Sıra | Önerilme payı [95% GA] |
|---|---:|---|
| Tümü | 1 | %37 [%29 · %45] |
| Tümü | 2 | %24 [%18 · %31] |
| Tümü | 3 | %12 [%7 · %17] |
| Tümü | 4 | %14 [%9 · %20] |
| Tümü | 5 | %13 [%8 · %19] |
| Güneş kremi | 1 | %35 [%24 · %45] |
| Güneş kremi | 2 | %26 [%17 · %36] |
| Güneş kremi | 3 | %13 [%5 · %21] |
| Güneş kremi | 4 | %14 [%6 · %22] |
| Güneş kremi | 5 | %13 [%6 · %21] |
| VPN | 1 | %40 [%29 · %50] |
| VPN | 2 | %23 [%14 · %32] |
| VPN | 3 | %10 [%4 · %17] |
| VPN | 4 | %14 [%8 · %22] |
| VPN | 5 | %13 [%6 · %21] |

## Sınırlılıklar

- Kategori başına tek ürün seti, iki soru kalıbı ve sabit cümle metinleri: bir cümle türünün sonucu o türün bu metnine aittir.
- Turnuvada cümle başına 30 görünme (kategori başına); yakın paylar arasında sıra çağrılmaz.
- Dolgu cümlesi uzunluğu eşler ama bir metnin 'hiç bilgi taşımadığı' tam sağlanamaz.
- Uyarı ölçüsü anahtar kelimedir; uyarının varlığını abartabilir, yokluğunu doğru gösterir.
- Uydurma iddialar yalnız kurgusal markalara eklendi; ürün bunları önermez, risk olarak raporlar.
