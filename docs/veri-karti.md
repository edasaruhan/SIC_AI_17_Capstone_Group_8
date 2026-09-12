# Veri kartı — Türkçe marka yanlılığı değerlendirmeleri

**Veri seti:** [`furkankarli/turkish-brand-bias-evaluations`](https://huggingface.co/datasets/furkankarli/turkish-brand-bias-evaluations)
**Sürüm (revision):** `4d274b954be7d9b0abbfe3314b2cbc387dfd800f`
**Lisans:** CC BY 4.0 — atıf zorunlu, ticari kullanım serbest.
**Dil:** Türkçe · **Boyut:** 300 yanıt · **Biçim:** Parquet (`all/train.parquet`)

Bu kart, veri setinin ne olduğunu, nasıl toplandığını, neyi ölçtüğünü ve **neyi
ölçmediğini** anlatır. Buradaki her sayı depodaki bir komutla yeniden üretilebilir;
son bölüme bakın.

## Ne içerir

Yapay zekâ asistanlarına Türkçe marka önerisi soruları soruldu ve verdikleri yanıtlar,
arama açık ve kapalı olmak üzere iki koşulda kaydedildi. Veri seti bu yanıtların
kendisidir: asistanın metni, kullandığı arama sonuçları ve yapılandırılmış marka
çıkarımları.

Amaç, bir markanın asistan yanıtlarında görünürlüğünün ne kadarının **marka
tanınırlığından**, ne kadarının **hakkında yazılmış içerikten** geldiğini ayırmaktır.

## Toplama tasarımı (sonuçlara bakılmadan sabitlendi)

```
2 sektör × 5 sorgu × 3 model × 2 koşul × 5 tekrar = 300 yanıt
```

| Boyut | Değer |
|---|---|
| Sektör | `vpn`, `cosmetics` (her biri 150 satır) |
| Koşul | `search_off` (yalnız modelin kendi bilgisi), `search_on` (web sonuçları verildi) — her biri 150 satır |
| Model | `gemini-3.5-flash-lite`, `MiniMax-M2.7`, `abliterated-model-large-v2` — her biri 100 satır |
| Tekrar | 5 · **Sıcaklık** 0,7 · **Maks. yanıt tokenı** 1024 |
| Arama | Serper Google Search API, Türkiye/Türkçe, 10 organik sonuç, en fazla 5 tur |
| Etiketleyici (judge) | Cerebras `gpt-oss-120b`, sıcaklık 0 |

NVIDIA modelleri sağlayıcı tarafındaki HTTP 410/502 ve timeout hataları nedeniyle
4 Eylül 2026'da tasarımdan çıkarıldı; ham kayıtları silindi ve plan üç modele
düşürüldü. Bu, sonuçlara bakıldıktan sonra yapılmış bir seçim değildir.

**Proje planından sapma:** ödev planı üç Türkçe sektör öngörüyordu; gerçekleşen iki
sektördür (`vpn`, `cosmetics`). Üçüncü sektör API bütçesi ve takvim nedeniyle
toplanmadı.

## Alanlar

| Alan | Anlamı |
|---|---|
| `record_id` | Yanıtın tekil kimliği |
| `experiment_id` | Donmuş deney tasarımının kimliği |
| `model_id`, `provider` | Yanıtı üreten asistan ve sağlayıcısı |
| `condition` | `search_off` / `search_on` |
| `category`, `language` | Sektör ve dil (`tr`) |
| `query_id`, `query_text` | Sorgu kimliği ve sorulan soru |
| `run_index` | Aynı hücredeki tekrar numarası (0–4) — **bağımsız gözlem değildir** |
| `temperature` | Üretim sıcaklığı (0,7) |
| `final_response` | Asistanın kullanıcıya verdiği nihai metin |
| `tool_calls`, `search_results` | Arama çağrıları ve dönen organik sonuçlar (JSON) |
| `core` | Judge çıkarımı: anılan markalar, birincil öneri, ilk anılan marka, marka sayısı (JSON) |
| `domain` | Sektöre özgü yapılandırılmış alanlar (JSON) |
| `search_aware` | Aramaya bağlı alanlar: kaç sonuç döndü, hangi kaynaklar (JSON) |
| `deterministic` | Kural tabanlı türetilmiş alanlar: yanıt uzunluğu vb. (JSON) |
| `judge_model`, `judge_prompt_version` | Etiketleyici modeli ve prompt sürümü |

Şema `src/turkish_data.py:EXPECTED_COLUMNS` içinde sabittir; JSON alanları
`JSON_COLUMNS` listesiyle işaretlenir.

## Amaçlanan kullanım

- Asistan yanıtlarında marka görünürlüğünün ölçülmesi ve modellenmesi.
- Arama açık/kapalı karşılaştırmasıyla retrieval etkisinin incelenmesi.
- Marka adı maskeleme gibi atıf (attribution) deneyleri.

**Amaçlanmayan kullanım:** bir markanın gerçek pazar payı, kalitesi veya kullanıcı
tercihinin göstergesi olarak kullanmak; bir asistanın bugünkü davranışını tahmin
etmek (veri tek bir zaman penceresinden gelir); kişiselleştirilmiş öneri üretmek.

## Bilinen sınırlılıklar ve yanlılıklar

- **Etiketler doğrulanmadı.** Marka çıkarımları bir judge modelinden gelir. Planlanan
  30 kayıtlık insan incelemesi **tamamlanmamıştır**; `review/ai/annotations.csv`
  dosyası AI kökenlidir ve insan onayı sayılmaz. Etiket gürültüsü nicelenmemiştir.
- **Marka evreni bizim seçimimizdir.** `configs/modeling/brands/*.yaml` içindeki
  listede olmayan bir marka, yanıtta geçse bile "anılmamış" görünür.
- **Sorgu havuzu küçüktür:** sektör başına 5 bağımsız sorgu. Güven aralıkları bu
  nedenle geniştir ve sorgu seçimi sonuçları şekillendirir.
- **`confidence` alanı tamamen `high`.** Türkçe kayıtlarda düşük güven filtresi
  hiçbir satırı elemez; İngilizce referansta eler. İki korpusu karşılaştırırken bu
  fark hatırlanmalıdır.
- **Tekrarlar bağımsız değildir.** `run_index` aynı hücrenin tekrarıdır; istatistik
  yaparken küme birimi `query_id` olmalıdır.
- **Arama sonuçları bir zaman anlık görüntüsüdür** (Türkiye/Türkçe, 2026 Eylül).
  Aynı sorgu bugün farklı sonuçlar döndürebilir.
- **Sağlayıcı modelleri güncellenebilir.** Kayıtlar iki haftalık bir pencerede
  toplandı; sonraki model sürümlerinde davranış değişebilir, bu ölçülmedi.
- **Kozmetikte uzun kuyruk eksiktir.** Registry 74 marka içerir; eşiğin altındaki
  küçük markalar listeye girmemiştir.

## Etik ve kişisel veri

Veri seti **hiçbir kişisel veri içermez.** Girdiler sentetik sorulardır; çıktılar
model üretimidir. Kullanıcı, müşteri veya çalışan verisi toplanmamıştır. Sağlayıcı
kullanım şartları toplama sırasında geçerli olan sürümleriyle belgelenmiştir.

Veri seti marka yanlılığını **ölçmek** için yayımlanmıştır. Görünürlük manipülasyonu
için kullanılması amaçlanmamıştır; projenin öneri katmanı bu nedenle kod düzeyinde
kısıtlanmıştır (`src/visibility/ethics.py`).

## Provenance ve bütünlük

| Kontrol | Değer |
|---|---|
| HF revision | `4d274b954be7d9b0abbfe3314b2cbc387dfd800f` |
| Kaynak dosya SHA-256 | `bd7319afb3f717449b8fe0baa28fe8a4008da15f59e3af1a6b2cf2bde2801a71` |
| İçerik SHA-256 (kolon/satır sırasından bağımsız) | `1b77bc902487212e9be13aa846a16d27ae20d987e54f257252716faf4d43a9e4` |

`make turkish-data` indirdiği dosyanın hash'ini bu sabitlere karşı doğrular ve
uyuşmazlıkta durur; sessizce farklı bir sürümle çalışmaz.

## Yeniden üretim

```bash
make turkish-data        # veriyi indir ve hash'ini doğrula
make modeling-validate   # 300 satır, sektör/koşul/model dağılımları, şema
```

Bu kartın bütün sayıları (300 satır, 150/150 sektör, 100/100/100 model, 20 kolon)
`src/turkish_data.py` içindeki `EXPECTED_*` sabitlerinden gelir ve `check()`
tarafından her indirmede doğrulanır.

## Atıf

```
Samsung Innovation Campus Capstone Grup 8 (2026).
Turkish Brand Bias Evaluations.
https://huggingface.co/datasets/furkankarli/turkish-brand-bias-evaluations
Lisans: CC BY 4.0
```

Ekip: [@furkankarli](https://github.com/furkankarli),
[@muratmertkucuk](https://github.com/muratmertkucuk),
[@kubragzc](https://github.com/kubragzc),
[@zeynepsinal](https://github.com/zeynepsinal).
Yayımlamadan önce atıf satırındaki adları ekip kendi tercih ettiği yazımla
tamamlamalıdır.

Veri seti, İngilizce `3RAIN/brand-bias-evaluations` referansının deney ve export
yapısıyla uyumlu olacak şekilde tasarlanmıştır; o referansın MIT lisansı bu veri
setine otomatik olarak uygulanmaz.
