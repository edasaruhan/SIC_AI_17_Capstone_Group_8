# MiniMax ile marka analizi CLI demosu

Bu bir ürün prototipidir; veri setini yeniden üretmez ve model eğitmez. Kullanıcı
markasını/sektörünü girer; uygulama MiniMax yanıtlarında anılmasını sayar, Google
arama snippet'lerinden kaynaklı inceleme önerileri üretir. Henüz LangGraph veya
web arayüzü yoktur.

## Marka anılmıyorsa: somut iş planı (v2)

Demo artık yalnız anılma tablosu veya MiniMax'in boş `actions` sonucuyla bitmez.
Kaydedilmiş yanıtlardan ve arama sonuçlarından ayrı bir **kural tabanlı iş planı**
üretilir. Bu bölüm MiniMax'in ürettiği bir cevap veya kanıtlanmış etki değildir.

- Genel sorguların snippet'lerindeki anılma ile marka aramasındaki bulunmayı ayırır.
- Desteklenen sektörlerde sabit marka listesinden, diğer sektörlerde kullanıcı
  tarafından girilen rakip isimlerinden yanıtlarda anılanları gösterir. Tüm pazarın
  keşfedildiği iddia edilmez. Kaynak eşleşmesi tercih nedeni olarak sunulmaz.
- Ürünün sorguya uygunluğu, doğrulanabilir bilgi/SSS içeriği, gerekirse teknik
  bulunabilirlik kontrolü ve ilgili yayıncı kaynaklarının incelenmesini önceliklendirir.
- Her iş için **dayanak, sorumlu, adımlar, teslim edilecek çıktı ve tamamlanma
  kontrolü** verir. Kaynak yoksa kaynaklı iddia yerine açık araştırma/doğrulama işleri verir.
- Yeniden ölçüm planında sorgu, model, bölge ve tekrar sayısını sabit tutar;
  kaynakta anılma ve yanıtta anılma ayrı izlenir. Otomatik yeni toplama yapmaz.

MiniMax öneri istemine de bu tanısal bağlam eklenmiştir; sıfır anılma tek başına
öneriden vazgeçme nedeni değildir. Yeni canlı çalışmada normal çağrı sınırı yine
7 MiniMax + 4 Serper'dır.

**Mevcut raporunuz varsa API'leri tekrar çalıştırmayın:**

```bash
make brand-demo-actions DEMO_RUN=data/processed/brand_demo/MEVCUT_CALISMA_ID
```

Bu komut sıfır API çağrısıyla `report.action-plan.v2.md` ve
`report.action-plan.v2.json` oluşturur. `report.json`, `state.json` ve ücretli adım
kayıtları değiştirilmez. Girdi raporunun tamamlanması ve içerik hash'i doğrulanır;
yeni dosya eski raporun SHA-256 değeri ve aksiyon kodunun hash'ini taşır. `.env`
okunmaz, yerel model yeniden çalıştırılmaz. Tekrar çağırırsanız yalnız bu iki v2
çıktısı yenilenir. Canlı demo kodu değiştiği için `make brand-demo` yeni çalışma
kimliği oluşturabilir; sırf yeni rapor görünümünü almak için onu kullanmayın.

Genel yöntem dayanakları: [Google AI features rehberi](https://developers.google.com/search/docs/appearance/ai-features)
ve [yararlı/güvenilir içerik rehberi](https://developers.google.com/search/docs/fundamentals/creating-helpful-content).
Bunlar Google araması içindir; MiniMax'in bir markayı seçeceğini garanti eden
kurallar değildir. Teknik kontrol işleri de sitede gerçekten sorun bulunduğu
iddiası değildir.

## Tek komutla başlatma

Mevcut `.venv` kuruluysa ve `.env` içinde `MINIMAX_API_KEY` ile `SERPER_API_KEY`
tanımlıysa:

```bash
make brand-demo
```

Marka ve sektör sorulur. Örneğin `Proton VPN` / `vpn`, `Clinique` / `kozmetik`
veya kendi markanız / `kahve`. **Marka adını sektör alanına da yazmayın.**
Sorgular, çıktı klasörü ve çağrı sınırı gösterilir; `evet` yazmadan API çağrılmaz.
MiniMax ve Serper dışında anahtar gerekmez. `.env` shell olarak çalıştırılmaz;
yalnız bu iki anahtarın düz veya tırnaklı değerleri okunur. Komut çalıştırma,
değişken genişletme ve `.env` içinde satır sonu yorumları desteklenmez.

Kredisiz deneme:

```bash
make brand-demo-offline
```

Bu mod açıkça **sentetik** yanıt/kaynak kullanır. `.env` okumaz, internete çıkmaz,
pilot model çağırmaz. Gösterilen anılma oranı gerçek bir marka sonucu değildir.

Yalnız planı görmek (dosya/API/anahtar işlemi yok):

```bash
PYTHONPATH=src .venv/bin/python -m brand_demo --brand "Proton VPN" --sector vpn
```

Karşılaştırma markaları ve sabit çalışma kimliğiyle:

```bash
PYTHONPATH=src .venv/bin/python -m brand_demo --live \
  --brand "Proton VPN" --sector vpn --competitor Mullvad --competitor NordVPN \
  --run-id ilk-demo
```

En çok beş karşılaştırma markası kabul edilir. Karşılaştırma isimleri tarafsız
üretim sorgularına eklenmez; yalnız yanıtların sonradan sayımında kullanılır.
Otomasyon için `--yes` onayı atlar ve **canlı çağrıların maliyetini kabul eder**.

## Akış ve neyi ölçtüğü

```text
Marka + sektör
  → Sektöre ait 3 marka adı içermeyen sorgu
  → Her sorgu için MiniMax aramasız yanıtı
  → Aynı sorgu için Serper sonuçları → MiniMax kaynak bağlamlı yanıtı
  → Yanıtlarda hedef/karşılaştırma markalarının anılma sayımı
  → Ayrı marka araması (yalnız kaynak incelemesi için)
  → MiniMax kaynak alıntılı öneri hipotezleri
  → Uygunsa yerel geçmiş veri ve pilot tahmini (ayrı bölüm)
  → Terminal özeti + report.md + report.json
```

Aramalı koşulda aramayı uygulama zorunlu yapar ve modele en çok beş organik
sonucun başlık/snippet/URL bilgisini sunar. Bu, referans araştırmadaki modelin
`tool_choice=auto` ile kendi aramasını seçmesiyle **aynı deney değildir**.
Demo çıktıları eğitim veri setine veya referans karşılaştırmalarına karıştırılmaz.

Marka sorgusu sonradan, yalnız öneri için çalışır. Bu aramanın sonuçları, tarafsız
görünürlük sorgularına verilmez ve o sorguların anılma paydasına katılmaz.
İki koşulda da payda üç yanıt; anılma olumlu öneri, birincilik veya kalite demek
değildir. Düşünme içeriği sayılmaz. Açıkça kullanıcı tarafından belirtilmeyen
rakipleri bu sürüm kendiliğinden keşfetmez.

Önerideki `source_id` ve alıntının ilgili başlık/snippet içinde birebir bulunduğu
kodla doğrulanır. Geçersiz öneri rapora alınmaz. Ancak doğru alıntı, önerinin doğru
olduğunu veya işe yarayacağını kanıtlamaz: öneriler insan kontrolü bekleyen
hipotezlerdir. Tam sayfa indirilmez, sitenin eksikleri hakkında kanıtlanmış denetim
iddiası kurulmaz. Dış metinler istemde güvenilmeyen veri olarak sınırlandırılır;
bağlantılar takip edilmez ve MiniMax'e yan etkili araç verilmez.

## Kota ve hata davranışı

- Normal, önbelleksiz çalışma: **en fazla 7 MiniMax + 4 Serper isteği**.
- MiniMax: `MiniMax-M2.7`, sıcaklık `0.3`, çağrı başına en çok `4096`
  `max_completion_tokens`; yedi başarılı çağrı için istenen çıktı tavanı `28.672`
  token. Bu dolar maliyeti değildir; input tokenları ve hatalı/belirsiz istekler
  ayrıca maliyete yol açabilir. Gerçek kullanılabilir kredi model sağlayıcısına bağlıdır.
- M2.7 düşünmesi kapatılamaz; `reasoning_split=True` yalnız çıktı ayrımıdır.
  Düşünme içeriği rapor, anılma sayımı ve kayıt dosyalarına alınmaz.
- Timeout: toplam HTTP timeout ayarı 120 saniye, bağlantı 20 saniye; çağrılar sıralıdır.
- Otomatik HTTP tekrar, JSON tamir çağrısı, model değiştirme, bakiye yükleme yoktur.
- 401/402/429/5xx ve timeout'ta durur; HTTP kodu ve varsa sayısal `Retry-After`
  bildirilir. 429'un günlük kota mı anlık sınır mı olduğu tek başına bu demodan
  kesinleştirilemez; sağlayıcı panelini kontrol edin.
- Kaynak yoksa final öneri çağrısı atlanır; kaynak/öneri uydurulmaz.
- Kesilmiş veya boş MiniMax yanıtı başarılı sayılmaz. Kaynak alıntısı/JSON
  doğrulaması başarısızsa yalnız öneri adımı hata olur; önceki 10 adım korunur.
- API ücretli bakiyeyi ücretsiz/promosyon bakiyesinden kesin ayıramaz. Ücretsiz
  çalışma garantisi yoktur; sağlayıcı panelindeki ödeme/harcama sınırlarını siz yönetin.

Başarılı adımlar atomik JSON ve içerik hash'iyle saklanır. Aynı marka, sektör,
karşılaştırmalar, kod ve `--run-id` ile yeniden çalıştırmak tamamlanan çağrıları
tekrarlamaz. Aynı çalışma eşzamanlı ikinci süreçte açılamaz.

Varsayılan `--run-id` **UTC tarihidir**; gün değişince yeni çalışma oluşur. Gece
yarısını aşan devam işlemlerinde önceki `plan.json` içindeki run_id'yi açıkça verin.
Önbellek süresizdir; eski sorgu aynı kimlikle yeniden canlı arama yapmaz. Güncel yeni
ölçüm için yeni `--run-id` seçin; bunun yeni çağrı ve ücret demek olduğunu unutmayın.
Kod/istem değişikliği de yeni çalışma kimliği üretir; aynı eski çalışma gibi devam
ediyormuş görüntüsü verilmez.

Hata nedenini çözdükten sonra aynı parametrelere açık tekrar izni ekleyin:

```bash
PYTHONPATH=src .venv/bin/python -m brand_demo --live \
  --brand "Proton VPN" --sector vpn --competitor Mullvad --competitor NordVPN \
  --run-id ilk-demo --retry-failed
```

İstek sağlayıcıda işlenip yanıt kaybolmuş olabilir; tekrar ücretlenebilir.
Her adım için tüm devam denemeleri boyunca en fazla üç istek hakkı vardır. Bu
nedenle açık tekrarlarla toplam deneme sayısı normal 7+4 sınırını aşabilir; mutlak
çalışma sınırı 21 MiniMax + 12 Serper denemesidir. Tamamlananlar yine atlanır.

## Yerel modellerin rolü

Türkçe VPN/kozmetik ve sabit marka listesiyle uyum varsa:

- Kayıtlı veri setindeki geçmiş anılma oranları ayrı tutulur.
- Tamamlanmış yerel model paketi ilk tarafsız sorgu ve onun arama sonuçları için
  CPU'da deneysel aday sırası/skoru üretir.
- Bu skor gerçek zamanlı gözlenen anılma değildir ve final MiniMax öneri istemine
  nedensel kanıt gibi eklenmez. Sıra rakipler arasında kullanıcı memnuniyeti veya
  Google sıralaması anlamına gelmez.

Farklı sektör/marka için genel MiniMax+Serper demosu çalışır; eğitilmemiş yerel
modelden skor üretilmez. Yerel veri/ağırlık/opsiyonel torch eksikse veya checksum
doğrulanamazsa bu bölüm açıkça kullanılamıyor olarak işaretlenir; API toplama yeniden
başlatılmaz. Yerel bileşeni tamamen kapatmak için `--no-local-model` kullanılabilir.

## Dosyalar ve geliştirici yapısı

Çıktılar `data/processed/brand_demo/<id>/` altında, Git dışında kalır:

```text
plan.json       # sorgular, model, locale, çalışma/kod kimliği
steps/*.json    # tek istek sonucu, zaman, deneme sayısı, hash, token kullanımı
demo.log        # Loguru, anahtarlar ve provider hata gövdeleri olmadan
state.json      # completed/error
report.md       # okunabilir rapor, yanıt önizlemeleri
report.json     # tüm nihai yanıtlar, kaynaklar, sayımlar ve ayrı yerel analiz
```

`report.json` içindeki toplama zaman aralığı önbellekteki adımların gerçek kayıt
zamanıdır; raporun tekrar yazılma zamanı güncel arama yapıldığı anlamına gelmez.
`successful_completion_tokens` yalnız kaydı başarıyla tamamlanan çağrıları kapsar,
faturanın tamamı olarak yorumlanmaz. Kimlik doğrulama başlıkları ve MiniMax'in ham
düşünme alanları kaydedilmez. Raporlar yerelde kalır; fakat canlı çalışmada sorgu ve
kaynaklar ilgili API sağlayıcılarına gönderilir. Gizli müşteri bilgisi girmeyin.

- `src/brand_demo/core.py`: saf planlama, isim eşleştirme, öneri doğrulama, render.
- `clients.py`: MiniMax/Serper HTTP adapter'ı ve açıkça sentetik fixture adapter'ı.
- `workflow.py`: adımlar ve atomik devam kayıtları.
- `actions.py`: sıfır anılmada da kanıt/varsayım ayrımlı iş planı; eski raporu ağsız zenginleştirme.
- `local.py`: dondurulmuş geçmiş veri ve pilot modeline salt-okunur bağlantı.
- `__main__.py`: terminal soruları, maliyet onayı ve iki anahtarın okunması.
- `tests/test_brand_demo.py`: ağ yasaklı fixture/MockTransport testleri.

LangGraph'a geçişte bu fonksiyonlar düğüm olabilir; yeni bir agent çerçevesi kurmak
için veri hazırlama, HTTP bağlantısı veya kaynak doğrulamasını yeniden yazmak gerekmez.
Bu demodaki adım önbelleği LangGraph checkpoint formatı değildir; geçişte adapter
gerekir. Önce gerçek kullanıcı akışının işe yararlılığı ve kaynak güvenilirliği
değerlendirilmeli, sonra çoklu sağlayıcı/tam sayfa denetimi/arayüz eklenmelidir.

Bağlantı sözleşmesi 9 Eylül 2026'da [MiniMax OpenAI uyumluluk belgesi](https://platform.minimax.io/docs/api-reference/text-openai-api)
ve [Chat Completions alanları](https://platform.minimax.io/docs/api-reference/text-chat-openai)
ile kontrol edildi. Yerel testler canlı sağlayıcı entegrasyonunun yerine geçmez;
ilk gerçek çalışmayı kullanıcı onayıyla yapmak gerekir.
