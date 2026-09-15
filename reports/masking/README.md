# Maskeleme ablasyonu: marka adı mı, içerik mi?

Tablolar [`results.md`](results.md) içinde, `make modeling-masking` ile üretilir.
Yöntem ve sınırlılıklar `scripts/collect_masking.py` başındadır. Bu sayfa o tablonun
elle yazılmış yorumudur.

**Deney:** snippet'lerdeki her marka adı `[BRAND]` ile değiştirilip M3 cross-encoder
yeniden eğitildi. İsimli ve maskeli koşular aynı satırlar üzerinde eşleştirilip tek bir
sorgu-küme bootstrap'ı ile karşılaştırıldı. Hedef birincil öneri (`y_top`), seed 7,
iki epoch, 128 token; encoder İngilizce'de `bert-base-uncased`, Türkçe'de BERTurk.

## Sonuç: bulgu sektöre bağlı

| Sektör | İsimli PR-AUC | Maskeli PR-AUC | Fark [95% GA] | Bağımsız sorgu |
|---|---:|---:|---|---:|
| Kod editörleri | 0,841 | 0,427 | **+0,414** [+0,213 · +0,589] | 10 |
| VPN · EN | 0,735 | 0,389 | **+0,347** [+0,136 · +0,491] | 10 |
| Seyahat | 0,642 | 0,558 | +0,084 [−0,128 · +0,317] | 10 |
| Hosting | 0,210 | 0,138 | **+0,072** [+0,017 · +0,132] | 10 |
| VPN · TR | 0,146 | 0,120 | +0,026 [−0,080 · +0,149] | 5 |
| Kozmetik · TR | 0,055 | 0,072 | −0,017 [−0,018 · +0,164] | 3 · yön çağrılmaz |

Pozitif fark, marka adı silindiğinde tahminin **kötüleştiği** anlamına gelir: model o
sektörde kimin kazandığını kısmen isimden biliyordu.

**Üç farklı rejim görünüyor:**

1. **Kimlik ağırlıklı (kod editörleri, İngilizce VPN).** Marka adı silinince tahmin
   gücü yarıdan fazla düşüyor. Bu sektörlerde kazananı büyük ölçüde markanın kendisi
   belirliyor; sayfa içeriği tek başına kimin önerileceğini açıklamıyor.
2. **Küçük ama ölçülebilir (hosting).** Fark +0,072 ve aralık sıfırın üstünde, ama
   etkinin büyüklüğü birinci gruptakinin onda biri. Hosting zaten en düşük tahmin
   edilebilirliğe sahip sektör (isimli PR-AUC 0,210); orada kazananı ne isim ne içerik
   güçlü biçimde belirliyor.
3. **Etki yok (seyahat, Türkçe VPN, Türkçe kozmetik).** Aralıklar sıfırı içeriyor.
   Seyahatte nokta tahmini pozitif ama aralık geniş; Türkçe kozmetikte yalnız 3
   bağımsız sorgu var, bu kararlılık eşiğinin (5) altındadır ve yön çağrılmaz.

## Neden bu genişletme yapıldı

Önceki raporlarda maskeleme yalnız VPN'de ölçülmüş ve "marka kimliği tahmin
edilebilirliğin büyük kısmını taşıyor" sonucu buradan yazılmıştı. Altı sektöre
yayıldığında bunun **evrensel değil, sektöre özgü** olduğu görülüyor: VPN ve kod
editörleri kimlik ağırlıklı, seyahat ve Türkçe sektörler değil. Tek sektörle genelleme
yapılmış olsaydı rapor yanlış bir iddia taşıyacaktı.

İngilizce VPN koşusu bu turda yeniden yapıldı ve önceki sonucu bağımsız biçimde
doğruladı (0,735 → 0,389; önceki seed'lerde 0,714 → 0,362).

## Ne söylemiyoruz

- **M3 bir vekil modeldir.** Ölçtüğümüz şey, *bizim tahmin modelimizin* neye dayandığıdır;
  asistanın karar mekanizması değildir. Asistanın içinde ne olduğunu bu deney görmez.
- **Maskeleme yalnız ad dizgisini siler**, bir markanın hangi sayfalarda göründüğünü
  silmez. O örüntü kimlikle ilişkili kalır. Fark bu yüzden "marka payı / içerik payı"
  diye okunamaz; ne alt ne üst sınır verir.
- **Nedensel değildir.** Nedensel kanıt yalnız [`reports/intervention/`](../intervention/)
  altındadır.
- **Tek seed.** Yeni sektörler seed 7 ile bir kez koşuldu; VPN'de üç seed'lik kararlılık
  kontrolü var, diğerlerinde yok. Sektörler arası sıralama tek koşuya dayanıyor.
- **Türkçe koşular küçüktür.** Kozmetikte 24 yanıt / 3 sorgu, VPN'de 57 yanıt / 5 sorgu;
  bu satırlar yön göstermez, yalnız kapsamı belgeler.
- **Eğitim ve değerlendirme donmuş sorgu fold'ları üzerindedir**; aynı sorgu hem eğitimde
  hem testte yer almaz.
