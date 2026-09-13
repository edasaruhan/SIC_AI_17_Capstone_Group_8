# Açıklama deneyi: yapay zekâ bir ürünü neye göre öneriyor? — gpt-oss-120b (Cerebras)

Bu sayfa `python -m description_lab analyze` ile üretilir; elle düzenlemeyin.
Tasarım ve önceden sabitlenen kurallar `src/description_lab/design.py` başındadır.

- Asistan: `gpt-oss-120b`, sıcaklık 0.7.
- Tamamlanan çağrı: 280/280.
- Her çağrıda aynı temel özelliklere sahip 5 ürün kartı; her hücrede hedef kart her sıraya eşit sayıda konur, rakiplerin sırası çağrı başına karışır.
- Ölçüm: gösterilen 5 markanın ad eşleştirmesi. 'Birinci önerilme' = yanıtın önerdiği marka hedef mi (tablo satırları dışında: önce 'öneririm/önerim' cümlesindeki tek marka, yoksa kalın yazılan tek marka, yoksa ilk anılan).
- Önerilen marka kuralla bulunamayıp ilk anılana düşülen çağrı: 0.

## 1. Marka adı: aynı ürün, farklı marka

Hedefin açıklaması kontrol metni; yalnız markası değişiyor. Taban: kurgusal marka.

| Kapsam | Kol | n | Birinci önerilme | Fark (puan) [95% GA] | Anılma | Fark (puan) [95% GA] |
|---|---|---:|---:|---|---:|---|
| Tümü | Küresel yerleşik | 20 | %65 | +60.0 [+35.0 · +85.0] | %95 | +15.0 [-5.0 · +35.0] |
| Tümü | Küçük gerçek marka | 20 | %0 | -5.0 [-15.0 · +0.0] | %75 | -5.0 [-30.0 · +20.0] |
| Tümü | Kurgusal marka | 20 | %5 | taban | %80 | taban |
| Güneş kremi | Küresel yerleşik | 10 | %70 | +60.0 [+20.0 · +90.0] | %100 | +20.0 [+0.0 · +50.0] |
| Güneş kremi | Küçük gerçek marka | 10 | %0 | -10.0 [-30.0 · +0.0] | %80 | +0.0 [-30.0 · +40.0] |
| Güneş kremi | Kurgusal marka | 10 | %10 | taban | %80 | taban |
| VPN | Küresel yerleşik | 10 | %60 | +60.0 [+30.0 · +90.0] | %90 | +10.0 [-20.0 · +40.0] |
| VPN | Küçük gerçek marka | 10 | %0 | +0.0 [+0.0 · +0.0] | %70 | -10.0 [-50.0 · +30.0] |
| VPN | Kurgusal marka | 10 | %0 | taban | %80 | taban |

## 2. Açıklama içeriği: aynı kurgusal marka, farklı metin

Hedef kurgusal marka; yalnız açıklamasına eklenen cümle değişiyor. Taban: kontrol metni.

| Kapsam | Kol | n | Birinci önerilme | Fark (puan) [95% GA] | Anılma | Fark (puan) [95% GA] |
|---|---|---:|---:|---|---:|---|
| Tümü | Kontrol (yalnız temel özellikler) | 20 | %5 | taban | %80 | taban |
| Tümü | Sayısal kanıt / istatistik | 20 | %85 | +80.0 [+60.0 · +95.0] | %100 | +20.0 [+5.0 · +35.0] |
| Tümü | Bağımsız test raporu gösterme | 20 | %55 | +50.0 [+25.0 · +75.0] | %85 | +5.0 [-20.0 · +30.0] |
| Tümü | Uzman alıntısı | 20 | %70 | +65.0 [+40.0 · +85.0] | %85 | +5.0 [-20.0 · +30.0] |
| Tümü | Otorite dili | 20 | %75 | +70.0 [+50.0 · +90.0] | %90 | +10.0 [-10.0 · +30.0] |
| Tümü | Sosyal kanıt (puan, kullanıcı sayısı) | 20 | %90 | +85.0 [+65.0 · +100.0] | %100 | +20.0 [+5.0 · +40.0] |
| Tümü | Sertifika | 20 | %95 | +90.0 [+75.0 · +100.0] | %95 | +15.0 [-5.0 · +35.0] |
| Tümü | Üstünlük dili ('en iyi', 'bir numara') | 20 | %85 | +80.0 [+60.0 · +95.0] | %95 | +15.0 [-5.0 · +35.0] |
| Tümü | Duygusal dil | 20 | %15 | +10.0 [-10.0 · +30.0] | %75 | -5.0 [-30.0 · +20.0] |
| Tümü | Teknik detay | 20 | %90 | +85.0 [+70.0 · +100.0] | %90 | +10.0 [-10.0 · +30.0] |
| Tümü | Fiyat avantajı | 20 | %95 | +90.0 [+75.0 · +100.0] | %100 | +20.0 [+5.0 · +40.0] |
| Tümü | Uydurma klinik/denetim iddiası | 20 | %80 | +75.0 [+55.0 · +95.0] | %85 | +5.0 [-20.0 · +30.0] |
| Güneş kremi | Kontrol (yalnız temel özellikler) | 10 | %10 | taban | %80 | taban |
| Güneş kremi | Sayısal kanıt / istatistik | 10 | %90 | +80.0 [+50.0 · +100.0] | %100 | +20.0 [+0.0 · +50.0] |
| Güneş kremi | Bağımsız test raporu gösterme | 10 | %10 | +0.0 [-30.0 · +30.0] | %70 | -10.0 [-50.0 · +30.0] |
| Güneş kremi | Uzman alıntısı | 10 | %70 | +60.0 [+20.0 · +90.0] | %70 | -10.0 [-50.0 · +30.0] |
| Güneş kremi | Otorite dili | 10 | %50 | +40.0 [+0.0 · +80.0] | %80 | +0.0 [-30.0 · +30.0] |
| Güneş kremi | Sosyal kanıt (puan, kullanıcı sayısı) | 10 | %90 | +80.0 [+50.0 · +100.0] | %100 | +20.0 [+0.0 · +50.0] |
| Güneş kremi | Sertifika | 10 | %100 | +90.0 [+70.0 · +100.0] | %90 | +10.0 [-20.0 · +40.0] |
| Güneş kremi | Üstünlük dili ('en iyi', 'bir numara') | 10 | %70 | +60.0 [+20.0 · +90.0] | %90 | +10.0 [-20.0 · +40.0] |
| Güneş kremi | Duygusal dil | 10 | %20 | +10.0 [-20.0 · +40.0] | %70 | -10.0 [-50.0 · +30.0] |
| Güneş kremi | Teknik detay | 10 | %100 | +90.0 [+70.0 · +100.0] | %80 | +0.0 [-40.0 · +40.0] |
| Güneş kremi | Fiyat avantajı | 10 | %90 | +80.0 [+50.0 · +100.0] | %100 | +20.0 [+0.0 · +50.0] |
| Güneş kremi | Uydurma klinik/denetim iddiası | 10 | %60 | +50.0 [+10.0 · +80.0] | %70 | -10.0 [-50.0 · +30.0] |
| VPN | Kontrol (yalnız temel özellikler) | 10 | %0 | taban | %80 | taban |
| VPN | Sayısal kanıt / istatistik | 10 | %80 | +80.0 [+50.0 · +100.0] | %100 | +20.0 [+0.0 · +50.0] |
| VPN | Bağımsız test raporu gösterme | 10 | %100 | +100.0 [+100.0 · +100.0] | %100 | +20.0 [+0.0 · +50.0] |
| VPN | Uzman alıntısı | 10 | %70 | +70.0 [+40.0 · +90.0] | %100 | +20.0 [+0.0 · +50.0] |
| VPN | Otorite dili | 10 | %100 | +100.0 [+100.0 · +100.0] | %100 | +20.0 [+0.0 · +50.0] |
| VPN | Sosyal kanıt (puan, kullanıcı sayısı) | 10 | %90 | +90.0 [+70.0 · +100.0] | %100 | +20.0 [+0.0 · +50.0] |
| VPN | Sertifika | 10 | %90 | +90.0 [+70.0 · +100.0] | %100 | +20.0 [+0.0 · +50.0] |
| VPN | Üstünlük dili ('en iyi', 'bir numara') | 10 | %100 | +100.0 [+100.0 · +100.0] | %100 | +20.0 [+0.0 · +50.0] |
| VPN | Duygusal dil | 10 | %10 | +10.0 [+0.0 · +30.0] | %80 | +0.0 [-30.0 · +30.2] |
| VPN | Teknik detay | 10 | %80 | +80.0 [+50.0 · +100.0] | %100 | +20.0 [+0.0 · +50.0] |
| VPN | Fiyat avantajı | 10 | %100 | +100.0 [+100.0 · +100.0] | %100 | +20.0 [+0.0 · +50.0] |
| VPN | Uydurma klinik/denetim iddiası | 10 | %100 | +100.0 [+100.0 · +100.0] | %100 | +20.0 [+0.0 · +50.0] |

## 3. Liste sırası

Bütün çağrılar, hedef kartın listedeki yerine göre. Taban: 5. sıra.

| Kapsam | Kol | n | Birinci önerilme | Fark (puan) [95% GA] | Anılma | Fark (puan) [95% GA] |
|---|---|---:|---:|---|---:|---|
| Tümü | 1. sıra | 56 | %80 | +12.5 [-3.6 · +28.6] | %91 | +0.0 [-10.7 · +10.7] |
| Tümü | 2. sıra | 56 | %54 | -14.3 [-32.1 · +5.4] | %84 | -7.1 [-19.6 · +5.4] |
| Tümü | 3. sıra | 56 | %52 | -16.1 [-33.9 · +1.8] | %89 | -1.8 [-12.5 · +8.9] |
| Tümü | 4. sıra | 56 | %70 | +1.8 [-16.1 · +17.9] | %91 | +0.0 [-10.7 · +10.7] |
| Tümü | 5. sıra | 56 | %68 | taban | %91 | taban |
| Güneş kremi | 1. sıra | 28 | %82 | +14.3 [-7.1 · +35.7] | %82 | -10.7 [-28.6 · +7.1] |
| Güneş kremi | 2. sıra | 28 | %46 | -21.4 [-46.4 · +3.6] | %79 | -14.3 [-32.1 · +3.6] |
| Güneş kremi | 3. sıra | 28 | %39 | -28.6 [-53.6 · -3.6] | %82 | -10.7 [-28.6 · +7.1] |
| Güneş kremi | 4. sıra | 28 | %61 | -7.1 [-32.1 · +17.9] | %86 | -7.1 [-25.0 · +10.7] |
| Güneş kremi | 5. sıra | 28 | %68 | taban | %93 | taban |
| VPN | 1. sıra | 28 | %79 | +10.7 [-10.8 · +35.7] | %100 | +10.7 [+0.0 · +21.4] |
| VPN | 2. sıra | 28 | %61 | -7.1 [-32.1 · +17.9] | %89 | +0.0 [-17.9 · +14.3] |
| VPN | 3. sıra | 28 | %64 | -3.6 [-28.6 · +21.4] | %96 | +7.1 [-3.7 · +21.4] |
| VPN | 4. sıra | 28 | %79 | +10.7 [-10.7 · +32.1] | %96 | +7.1 [-3.6 · +21.4] |
| VPN | 5. sıra | 28 | %68 | taban | %89 | taban |

## 4. Hangi sıradaki kart öneriliyor? (bütün kartlar)

Her çağrıda önerilen ürünün listedeki sırası. Rastgele seçimde her sıra %20 olurdu.

| Kapsam | Sıra | Önerilme payı [95% GA] | Çağrı |
|---|---:|---|---:|
| Tümü | 1 | %34 [%29 · %41] | 279 |
| Tümü | 2 | %14 [%10 · %18] | 279 |
| Tümü | 3 | %14 [%10 · %18] | 279 |
| Tümü | 4 | %19 [%14 · %23] | 279 |
| Tümü | 5 | %19 [%14 · %23] | 279 |
| Güneş kremi | 1 | %46 [%38 · %55] | 139 |
| Güneş kremi | 2 | %12 [%6 · %17] | 139 |
| Güneş kremi | 3 | %10 [%6 · %15] | 139 |
| Güneş kremi | 4 | %16 [%10 · %22] | 139 |
| Güneş kremi | 5 | %17 [%11 · %23] | 139 |
| VPN | 1 | %23 [%16 · %30] | 140 |
| VPN | 2 | %16 [%11 · %23] | 140 |
| VPN | 3 | %19 [%12 · %25] | 140 |
| VPN | 4 | %21 [%14 · %28] | 140 |
| VPN | 5 | %21 [%14 · %28] | 140 |

## Sınırlılıklar

- Tek asistan sayfası; öteki asistanın sonucu kendi klasöründedir. İki asistan aynı yönü göstermedikçe sonuç modele özgü sayılır.
- Kategori başına tek ürün seti ve iki soru kalıbı; sonuç başka ürün setlerine kendiliğinden genellenmez.
- Kol başına az tekrar; güven aralıkları geniştir, sıfırı içeren farklarda yön çağrılmaz.
- Varyant cümleleri açıklamaya bilgi ve uzunluk ekler; 'içerik' etkisi uzunluktan tam ayrıştırılmadı.
- Kimlik ve içerik tam çaprazlanmadı: içeriğin yerleşik markayı yenip yenemediği bu tasarımda doğrudan ölçülmedi.
- Uydurma iddialar yalnız kurgusal markaya eklendi; ürün bunları önermez, yalnız risk olarak raporlar.
