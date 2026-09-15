# Açıklama deneyi, 2. tur: hangi cümle kazanıyor? — gpt-oss-120b (Cerebras)

Bu sayfa `python -m description_lab analyze --round 2` ile üretilir; elle düzenlemeyin.
Tasarım `src/description_lab/round2.py` başındadır.

- Asistan: `gpt-oss-120b`, sıcaklık 0.7.
- Tamamlanan çağrı: 256/256.
- Önerilen marka 1. turun kuralıyla okunur; kural bulamayıp ilk anılana düşülen çağrı: 0.

## 1. Turnuva: her kartta farklı bir cümle

Beş kurgusal marka, her kartta 13 cümleden farklı biri. Her cümle her sıraya eşit sayıda konur. Şans payı %20.

**Tümü**

| Cümle | Görünme | Kazanma | Kazanma payı [95% GA] |
|---|---:|---:|---|
| Teknik detay | 60 | 40 | %67 [%54 · %78] |
| Fiyat avantajı | 60 | 30 | %50 [%38 · %63] |
| Uydurma klinik/denetim iddiası | 60 | 18 | %30 [%19 · %42] |
| Otorite dili | 60 | 16 | %27 [%15 · %38] |
| Sertifika | 60 | 14 | %23 [%13 · %34] |
| Sayısal kanıt / istatistik | 60 | 13 | %22 [%11 · %32] |
| Uzman alıntısı | 60 | 9 | %15 [%7 · %25] |
| Üstünlük dili ('en iyi', 'bir numara') | 60 | 8 | %13 [%6 · %23] |
| Bağımsız test raporu gösterme | 60 | 2 | %3 [%0 · %8] |
| Duygusal dil | 60 | 2 | %3 [%0 · %9] |
| Sosyal kanıt (puan, kullanıcı sayısı) | 60 | 2 | %3 [%0 · %9] |
| Bilgisiz dolgu cümlesi (uzunluk kontrolü) | 60 | 2 | %3 [%0 · %8] |
| Kontrol (yalnız temel özellikler) | 60 | 0 | %0 [%0 · %0] |

**Güneş kremi**

| Cümle | Görünme | Kazanma | Kazanma payı [95% GA] |
|---|---:|---:|---|
| Teknik detay | 30 | 22 | %73 [%56 · %89] |
| Fiyat avantajı | 30 | 12 | %40 [%23 · %58] |
| Sertifika | 30 | 11 | %37 [%20 · %56] |
| Otorite dili | 30 | 9 | %30 [%13 · %47] |
| Uzman alıntısı | 30 | 8 | %27 [%11 · %44] |
| Sayısal kanıt / istatistik | 30 | 6 | %20 [%7 · %38] |
| Uydurma klinik/denetim iddiası | 30 | 3 | %10 [%0 · %22] |
| Sosyal kanıt (puan, kullanıcı sayısı) | 30 | 2 | %7 [%0 · %17] |
| Duygusal dil | 30 | 2 | %7 [%0 · %16] |
| Bilgisiz dolgu cümlesi (uzunluk kontrolü) | 30 | 2 | %7 [%0 · %17] |
| Bağımsız test raporu gösterme | 30 | 1 | %3 [%0 · %11] |
| Kontrol (yalnız temel özellikler) | 30 | 0 | %0 [%0 · %0] |
| Üstünlük dili ('en iyi', 'bir numara') | 30 | 0 | %0 [%0 · %0] |

**VPN**

| Cümle | Görünme | Kazanma | Kazanma payı [95% GA] |
|---|---:|---:|---|
| Fiyat avantajı | 30 | 18 | %60 [%41 · %77] |
| Teknik detay | 30 | 18 | %60 [%42 · %77] |
| Uydurma klinik/denetim iddiası | 30 | 15 | %50 [%32 · %69] |
| Üstünlük dili ('en iyi', 'bir numara') | 30 | 8 | %27 [%11 · %44] |
| Sayısal kanıt / istatistik | 30 | 7 | %23 [%10 · %39] |
| Otorite dili | 30 | 7 | %23 [%9 · %39] |
| Sertifika | 30 | 3 | %10 [%0 · %22] |
| Bağımsız test raporu gösterme | 30 | 1 | %3 [%0 · %11] |
| Uzman alıntısı | 30 | 1 | %3 [%0 · %11] |
| Kontrol (yalnız temel özellikler) | 30 | 0 | %0 [%0 · %0] |
| Sosyal kanıt (puan, kullanıcı sayısı) | 30 | 0 | %0 [%0 · %0] |
| Duygusal dil | 30 | 0 | %0 [%0 · %0] |
| Bilgisiz dolgu cümlesi (uzunluk kontrolü) | 30 | 0 | %0 [%0 · %0] |

## 2. Dolgu cümlesi: fark mı, bilgi mi?

Hedef kurgusal markanın tek cümlesi bilgisiz dolgu. 1. turun aynı asistandaki kontrolü (cümlesiz) ve on bir cümlenin ortalamasıyla karşılaştırılır.

| Kapsam | n | Dolgu ile birinci önerilme | 1. tur kontrol | 1. tur cümleler | Dolgu − kontrol (puan) | Cümleler − dolgu (puan) |
|---|---:|---|---:|---:|---|---|
| Tümü | 20 | %0 [%0 · %0] | %5 | %76 | -5.0 [-15.0 · +0.0] | +75.9 [+70.0 · +80.9] |
| Güneş kremi | 10 | %0 [%0 · %0] | %10 | %68 | -10.0 [-30.0 · +0.0] | +68.2 [+60.0 · +77.3] |
| VPN | 10 | %0 [%0 · %0] | %0 | %84 | +0.0 [+0.0 · +0.0] | +83.6 [+76.4 · +90.0] |

## 3. Çaprazlama: cümle, tanınan markayı yenebilir mi?

Aynı listede küresel yerleşik marka (cümlesiz) ve kurgusal marka (cümleyle), yanında üç gerçek rakip.

| Kapsam | Kurgusal markanın cümlesi | n | Kurgusal önerildi | Yerleşik önerildi |
|---|---|---:|---|---|
| Tümü | Kontrol (yalnız temel özellikler) | 20 | %0 [%0 · %0] | %30 [%10 · %50] |
| Tümü | Sayısal kanıt / istatistik | 20 | %90 [%75 · %100] | %10 [%0 · %25] |
| Tümü | Üstünlük dili ('en iyi', 'bir numara') | 20 | %85 [%70 · %100] | %10 [%0 · %25] |
| Tümü | Uydurma klinik/denetim iddiası | 20 | %70 [%50 · %90] | %10 [%0 · %25] |
| Güneş kremi | Kontrol (yalnız temel özellikler) | 10 | %0 [%0 · %0] | %40 [%10 · %70] |
| Güneş kremi | Sayısal kanıt / istatistik | 10 | %80 [%50 · %100] | %20 [%0 · %50] |
| Güneş kremi | Üstünlük dili ('en iyi', 'bir numara') | 10 | %70 [%40 · %100] | %20 [%0 · %50] |
| Güneş kremi | Uydurma klinik/denetim iddiası | 10 | %40 [%10 · %70] | %20 [%0 · %50] |
| VPN | Kontrol (yalnız temel özellikler) | 10 | %0 [%0 · %0] | %20 [%0 · %50] |
| VPN | Sayısal kanıt / istatistik | 10 | %100 [%100 · %100] | %0 [%0 · %0] |
| VPN | Üstünlük dili ('en iyi', 'bir numara') | 10 | %100 [%100 · %100] | %0 [%0 · %0] |
| VPN | Uydurma klinik/denetim iddiası | 10 | %100 [%100 · %100] | %0 [%0 · %0] |

## 4. Uydurma iddia: seçiliyor mu, tekrarlanıyor mu, sorgulanıyor mu?

İddianın gösterildiği çağrılar. 'Tekrar' = yanıt iddianın kaynağını (Harvard/MIT) anıyor. 'Uyarı' anahtar kelime eşleşmesidir, üst sınırdır.

| Blok | n | İddialı kart önerildi | İddia tekrarlandı | Önerilmediğinde bile tekrar | Uyarı |
|---|---:|---|---|---|---|
| Tümü | 80 | %40 [%29 · %50] | %69 [%59 · %79] | %48 [%33 · %62] | %11 [%5 · %19] |
| Çaprazlama | 20 | %70 [%50 · %90] | %80 [%60 · %95] | %33 [%0 · %67] | %5 [%0 · %15] |
| Turnuva | 60 | %30 [%18 · %42] | %65 [%53 · %77] | %50 [%36 · %64] | %13 [%5 · %23] |

## 5. Turnuvada önerilen kartın sırası

| Kapsam | Sıra | Önerilme payı [95% GA] |
|---|---:|---|
| Tümü | 1 | %30 [%23 · %37] |
| Tümü | 2 | %16 [%10 · %22] |
| Tümü | 3 | %20 [%14 · %26] |
| Tümü | 4 | %19 [%13 · %26] |
| Tümü | 5 | %15 [%9 · %21] |
| Güneş kremi | 1 | %38 [%28 · %49] |
| Güneş kremi | 2 | %14 [%6 · %22] |
| Güneş kremi | 3 | %23 [%14 · %33] |
| Güneş kremi | 4 | %14 [%6 · %22] |
| Güneş kremi | 5 | %10 [%4 · %18] |
| VPN | 1 | %22 [%13 · %31] |
| VPN | 2 | %18 [%10 · %27] |
| VPN | 3 | %17 [%9 · %26] |
| VPN | 4 | %24 [%15 · %33] |
| VPN | 5 | %19 [%12 · %28] |

## Sınırlılıklar

- Kategori başına tek ürün seti, iki soru kalıbı ve sabit cümle metinleri: bir cümle türünün sonucu o türün bu metnine aittir.
- Turnuvada cümle başına 30 görünme (kategori başına); yakın paylar arasında sıra çağrılmaz.
- Dolgu cümlesi uzunluğu eşler ama bir metnin 'hiç bilgi taşımadığı' tam sağlanamaz.
- Uyarı ölçüsü anahtar kelimedir; uyarının varlığını abartabilir, yokluğunu doğru gösterir.
- Uydurma iddialar yalnız kurgusal markalara eklendi; ürün bunları önermez, risk olarak raporlar.
