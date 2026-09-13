# Açıklama deneyi: yapay zekâ bir ürünü neye göre öneriyor?

Bu sayfa `python -m description_lab analyze` ile üretilir; elle düzenlemeyin.
Tasarım ve önceden sabitlenen kurallar `src/description_lab/design.py` başındadır.

- Asistan: `gemini-3.5-flash-lite`, sıcaklık 0.7.
- Tamamlanan çağrı: 280/280.
- Her çağrıda aynı temel özelliklere sahip 5 ürün kartı; her hücrede hedef kart her sıraya eşit sayıda konur, rakiplerin sırası çağrı başına karışır.
- Ölçüm: gösterilen 5 markanın ad eşleştirmesi. 'Birinci önerilme' = yanıtın önerdiği marka hedef mi (önce kalın yazılan tek marka, yoksa 'öner/tercih' cümlesindeki tek marka, yoksa ilk anılan).
- Önerilen marka kuralla bulunamayıp ilk anılana düşülen çağrı: 0.

## 1. Marka adı: aynı ürün, farklı marka

Hedefin açıklaması kontrol metni; yalnız markası değişiyor. Taban: kurgusal marka.

| Kapsam | Kol | n | Birinci önerilme | Fark (puan) [95% GA] | Anılma | Fark (puan) [95% GA] |
|---|---|---:|---:|---|---:|---|
| Tümü | Küresel yerleşik | 20 | %45 | +35.0 [+10.0 · +60.0] | %85 | +0.0 [-25.0 · +25.0] |
| Tümü | Küçük gerçek marka | 20 | %15 | +5.0 [-15.0 · +25.0] | %90 | +5.0 [-15.0 · +25.0] |
| Tümü | Kurgusal marka | 20 | %10 | taban | %85 | taban |
| Güneş kremi | Küresel yerleşik | 10 | %40 | +30.0 [-10.0 · +60.0] | %80 | -10.0 [-40.0 · +20.0] |
| Güneş kremi | Küçük gerçek marka | 10 | %20 | +10.0 [-20.0 · +40.0] | %100 | +10.0 [+0.0 · +30.0] |
| Güneş kremi | Kurgusal marka | 10 | %10 | taban | %90 | taban |
| VPN | Küresel yerleşik | 10 | %50 | +40.0 [+0.0 · +70.0] | %90 | +10.0 [-20.0 · +40.0] |
| VPN | Küçük gerçek marka | 10 | %10 | +0.0 [-30.0 · +30.0] | %80 | +0.0 [-30.0 · +30.0] |
| VPN | Kurgusal marka | 10 | %10 | taban | %80 | taban |

## 2. Açıklama içeriği: aynı kurgusal marka, farklı metin

Hedef kurgusal marka; yalnız açıklamasına eklenen cümle değişiyor. Taban: kontrol metni.

| Kapsam | Kol | n | Birinci önerilme | Fark (puan) [95% GA] | Anılma | Fark (puan) [95% GA] |
|---|---|---:|---:|---|---:|---|
| Tümü | Kontrol (yalnız temel özellikler) | 20 | %10 | taban | %85 | taban |
| Tümü | Sayısal kanıt / istatistik | 20 | %100 | +90.0 [+75.0 · +100.0] | %100 | +15.0 [+0.0 · +30.0] |
| Tümü | Bağımsız test raporu gösterme | 20 | %90 | +80.0 [+60.0 · +95.0] | %100 | +15.0 [+0.0 · +35.0] |
| Tümü | Uzman alıntısı | 20 | %95 | +85.0 [+65.0 · +100.0] | %100 | +15.0 [+0.0 · +30.0] |
| Tümü | Otorite dili | 20 | %100 | +90.0 [+75.0 · +100.0] | %100 | +15.0 [+0.0 · +35.0] |
| Tümü | Sosyal kanıt (puan, kullanıcı sayısı) | 20 | %85 | +75.0 [+55.0 · +95.0] | %100 | +15.0 [+0.0 · +30.0] |
| Tümü | Sertifika | 20 | %95 | +85.0 [+65.0 · +100.0] | %100 | +15.0 [+0.0 · +30.1] |
| Tümü | Üstünlük dili ('en iyi', 'bir numara') | 20 | %100 | +90.0 [+75.0 · +100.0] | %100 | +15.0 [+0.0 · +30.0] |
| Tümü | Duygusal dil | 20 | %85 | +75.0 [+50.0 · +95.0] | %95 | +10.0 [-10.0 · +30.0] |
| Tümü | Teknik detay | 20 | %100 | +90.0 [+75.0 · +100.0] | %100 | +15.0 [+0.0 · +30.0] |
| Tümü | Fiyat avantajı | 20 | %95 | +85.0 [+65.0 · +100.0] | %100 | +15.0 [+0.0 · +30.0] |
| Tümü | Uydurma klinik/denetim iddiası | 20 | %100 | +90.0 [+75.0 · +100.0] | %100 | +15.0 [+0.0 · +30.0] |
| Güneş kremi | Kontrol (yalnız temel özellikler) | 10 | %10 | taban | %90 | taban |
| Güneş kremi | Sayısal kanıt / istatistik | 10 | %100 | +90.0 [+70.0 · +100.0] | %100 | +10.0 [+0.0 · +30.0] |
| Güneş kremi | Bağımsız test raporu gösterme | 10 | %100 | +90.0 [+70.0 · +100.0] | %100 | +10.0 [+0.0 · +30.0] |
| Güneş kremi | Uzman alıntısı | 10 | %90 | +80.0 [+50.0 · +100.0] | %100 | +10.0 [+0.0 · +30.0] |
| Güneş kremi | Otorite dili | 10 | %100 | +90.0 [+70.0 · +100.0] | %100 | +10.0 [+0.0 · +30.0] |
| Güneş kremi | Sosyal kanıt (puan, kullanıcı sayısı) | 10 | %90 | +80.0 [+50.0 · +100.0] | %100 | +10.0 [+0.0 · +30.0] |
| Güneş kremi | Sertifika | 10 | %90 | +80.0 [+50.0 · +100.0] | %100 | +10.0 [+0.0 · +30.0] |
| Güneş kremi | Üstünlük dili ('en iyi', 'bir numara') | 10 | %100 | +90.0 [+70.0 · +100.0] | %100 | +10.0 [+0.0 · +30.0] |
| Güneş kremi | Duygusal dil | 10 | %90 | +80.0 [+50.0 · +100.0] | %100 | +10.0 [+0.0 · +30.0] |
| Güneş kremi | Teknik detay | 10 | %100 | +90.0 [+70.0 · +100.0] | %100 | +10.0 [+0.0 · +30.0] |
| Güneş kremi | Fiyat avantajı | 10 | %100 | +90.0 [+70.0 · +100.0] | %100 | +10.0 [+0.0 · +30.0] |
| Güneş kremi | Uydurma klinik/denetim iddiası | 10 | %100 | +90.0 [+70.0 · +100.0] | %100 | +10.0 [+0.0 · +30.0] |
| VPN | Kontrol (yalnız temel özellikler) | 10 | %10 | taban | %80 | taban |
| VPN | Sayısal kanıt / istatistik | 10 | %100 | +90.0 [+70.0 · +100.0] | %100 | +20.0 [+0.0 · +50.0] |
| VPN | Bağımsız test raporu gösterme | 10 | %80 | +70.0 [+40.0 · +100.0] | %100 | +20.0 [+0.0 · +50.0] |
| VPN | Uzman alıntısı | 10 | %100 | +90.0 [+70.0 · +100.0] | %100 | +20.0 [+0.0 · +50.0] |
| VPN | Otorite dili | 10 | %100 | +90.0 [+70.0 · +100.0] | %100 | +20.0 [+0.0 · +50.0] |
| VPN | Sosyal kanıt (puan, kullanıcı sayısı) | 10 | %80 | +70.0 [+30.0 · +100.0] | %100 | +20.0 [+0.0 · +50.0] |
| VPN | Sertifika | 10 | %100 | +90.0 [+70.0 · +100.0] | %100 | +20.0 [+0.0 · +50.0] |
| VPN | Üstünlük dili ('en iyi', 'bir numara') | 10 | %100 | +90.0 [+70.0 · +100.0] | %100 | +20.0 [+0.0 · +50.0] |
| VPN | Duygusal dil | 10 | %80 | +70.0 [+40.0 · +100.0] | %90 | +10.0 [-20.0 · +40.0] |
| VPN | Teknik detay | 10 | %100 | +90.0 [+70.0 · +100.0] | %100 | +20.0 [+0.0 · +50.0] |
| VPN | Fiyat avantajı | 10 | %90 | +80.0 [+50.0 · +100.0] | %100 | +20.0 [+0.0 · +50.0] |
| VPN | Uydurma klinik/denetim iddiası | 10 | %100 | +90.0 [+70.0 · +100.0] | %100 | +20.0 [+0.0 · +50.0] |

## 3. Liste sırası

Bütün çağrılar, hedef kartın listedeki yerine göre. Taban: 5. sıra.

| Kapsam | Kol | n | Birinci önerilme | Fark (puan) [95% GA] | Anılma | Fark (puan) [95% GA] |
|---|---|---:|---:|---|---:|---|
| Tümü | 1. sıra | 56 | %95 | +17.9 [+5.4 · +30.4] | %100 | +3.6 [+0.0 · +8.9] |
| Tümü | 2. sıra | 56 | %77 | +0.0 [-16.1 · +16.1] | %98 | +1.8 [-3.6 · +7.1] |
| Tümü | 3. sıra | 56 | %73 | -3.6 [-19.6 · +10.7] | %95 | -1.8 [-10.7 · +5.4] |
| Tümü | 4. sıra | 56 | %77 | +0.0 [-16.1 · +16.1] | %95 | -1.8 [-8.9 · +5.4] |
| Tümü | 5. sıra | 56 | %77 | taban | %96 | taban |
| Güneş kremi | 1. sıra | 28 | %96 | +17.9 [+0.0 · +35.7] | %100 | +3.6 [+0.0 · +10.7] |
| Güneş kremi | 2. sıra | 28 | %75 | -3.6 [-25.0 · +17.9] | %100 | +3.6 [+0.0 · +10.7] |
| Güneş kremi | 3. sıra | 28 | %75 | -3.6 [-25.0 · +17.9] | %96 | +0.0 [-10.7 · +10.7] |
| Güneş kremi | 4. sıra | 28 | %79 | +0.0 [-21.4 · +21.4] | %96 | +0.0 [-10.7 · +10.7] |
| Güneş kremi | 5. sıra | 28 | %79 | taban | %96 | taban |
| VPN | 1. sıra | 28 | %93 | +17.9 [+0.0 · +35.7] | %100 | +3.6 [+0.0 · +10.7] |
| VPN | 2. sıra | 28 | %79 | +3.6 [-17.9 · +25.0] | %96 | +0.0 [-10.7 · +10.7] |
| VPN | 3. sıra | 28 | %71 | -3.6 [-28.6 · +17.9] | %93 | -3.6 [-14.3 · +7.1] |
| VPN | 4. sıra | 28 | %75 | +0.0 [-21.4 · +25.0] | %93 | -3.6 [-14.3 · +7.1] |
| VPN | 5. sıra | 28 | %75 | taban | %96 | taban |

## 4. Hangi sıradaki kart öneriliyor? (bütün kartlar)

Her çağrıda önerilen ürünün listedeki sırası. Rastgele seçimde her sıra %20 olurdu.

| Kapsam | Sıra | Önerilme payı [95% GA] | Çağrı |
|---|---:|---|---:|
| Tümü | 1 | %34 [%28 · %40] | 280 |
| Tümü | 2 | %17 [%13 · %22] | 280 |
| Tümü | 3 | %15 [%11 · %20] | 280 |
| Tümü | 4 | %16 [%11 · %20] | 280 |
| Tümü | 5 | %18 [%14 · %22] | 280 |
| Güneş kremi | 1 | %34 [%26 · %41] | 140 |
| Güneş kremi | 2 | %16 [%11 · %23] | 140 |
| Güneş kremi | 3 | %15 [%9 · %21] | 140 |
| Güneş kremi | 4 | %16 [%10 · %22] | 140 |
| Güneş kremi | 5 | %19 [%13 · %26] | 140 |
| VPN | 1 | %34 [%26 · %42] | 140 |
| VPN | 2 | %18 [%11 · %24] | 140 |
| VPN | 3 | %16 [%10 · %21] | 140 |
| VPN | 4 | %16 [%10 · %22] | 140 |
| VPN | 5 | %16 [%11 · %23] | 140 |

## Sınırlılıklar

- Tek asistan (Gemini 3.5 Flash Lite); önerinin 'en az iki model' koşulu karşılanmadı.
- Kategori başına tek ürün seti ve iki soru kalıbı; sonuç başka ürün setlerine kendiliğinden genellenmez.
- Kol başına az tekrar; güven aralıkları geniştir, sıfırı içeren farklarda yön çağrılmaz.
- Varyant cümleleri açıklamaya bilgi ve uzunluk ekler; 'içerik' etkisi uzunluktan tam ayrıştırılmadı.
- Kimlik ve içerik tam çaprazlanmadı: içeriğin yerleşik markayı yenip yenemediği bu tasarımda doğrudan ölçülmedi.
- Uydurma iddialar yalnız kurgusal markaya eklendi; ürün bunları önermez, yalnız risk olarak raporlar.
