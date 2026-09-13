# İngilizce VPN: uzun listwise eğitim işi

Bu akış yalnız mevcut İngilizce referansın **VPN** domain'ini eğitir. Türkçe
koşuların ayarları korunur: isimli/maskeli varyant, seed 7, iki epoch, 128 token,
candidate batch 4. Encoder `bert-base-uncased`, sabit HF revision'ı
`86b5e0934494bd15c9632b12f734a8a67f723594`.

Mevcut evidence_v1 verisinde 1.188 arama-açık VPN yanıtının 975'i tek kazananlı
hedefe uygundur; 213'ü bu hedefin dışında kalır. Veri silinmez. Her yanıtta 24
aday, toplam 10 sorgu ve 5 sorgu fold'u vardır. İki varyant toplam 10 eğitim
checkpoint'i üretir. Bu, bütün İngilizce domain'lerin veya bir sohbet LLM'inin
eğitimi değildir.

## İlk kurulum ve indirme

**Aktif eğitim varken `uv sync`, `make setup`, branch değiştirme veya eğitim
kod/config dosyalarını değiştirme.** Gereken paketler çalıştırmadan önce kurulur:

```bash
uv sync --extra m3 --group dev --locked
export HF_HOME="$PWD/data/processed/evidence_v1/hf_cache"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1

HF_HUB_OFFLINE=0 HF_HUB_DISABLE_XET=1 .venv/bin/python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("bert-base-uncased", revision="86b5e0934494bd15c9632b12f734a8a67f723594", allow_patterns=["config.json", "tokenizer_config.json", "tokenizer.json", "vocab.txt", "model.safetensors"]))'
```

İlk indirme internete çıkar fakat generation/search/judge API'si kullanmaz.
2026-09-07'de eski `bert-base-uncased` adı için Xet token ucu 404 verdi;
`HF_HUB_DISABLE_XET=1` ile aynı model/sürüm standart HTTP yolundan indirildi.
Model cache'de mevcutsa yeniden indirme gerekmez. `EVIDENCE_ROOT` değişiyorsa
cache'in yolu da o kökün `hf_cache` dizini olmalıdır.

## Başlatma, durum ve loglar

Repo kökünden, `.env` dosyasını yüklemeden:

```bash
make modeling-en-train
```

Bu komut Linux'ta **ön planda** çalışır ve sırasıyla küçük GPU testi, tam eğitim,
checkpoint denetimi yapar. Ağ erişimi için `HF_HUB_OFFLINE=1` uygular; model
eksikse kendiliğinden indirmez. Aynı işi ikinci kez başlatma girişimi kilitle
engellenir. Terminalden bağımsız arka plan işi istenirse:

```bash
mkdir -p data/processed/evidence_v1/training_jobs/en_vpn_seed7
nohup setsid env PYTHONPATH=src .venv/bin/python -u scripts/run_english_listwise.py \
  --root data/processed/evidence_v1 \
  >> data/processed/evidence_v1/training_jobs/en_vpn_seed7/launcher.log 2>&1 < /dev/null &
```

Durumu yeni eğitim başlatmadan görmek için:

```bash
make modeling-en-status
tail -f data/processed/evidence_v1/training_jobs/en_vpn_seed7/launcher.log
```

İş dizinindeki dosyalar:

- `status.json`: PID, worker PID, heartbeat, aşama, geçen süre, tamamlanmış ve
  çalışan fold'lar. Atomik yazılır; heartbeat tek başına öğrenmenin ilerlediğini
  kanıtlamaz. Gerçek epoch mesajları `training.log` dosyasındadır.
- `smoke.log`: iki yanıtlık isimli/maskeli GPU testi.
- `training.log`: tam eğitim, epoch mesajları ve metrikler.
- `audit.log`: checkpoint yeniden-yükleme ve aynı test panelinde baseline karşılaştırması.
- `launcher.log`: arka plan başlangıcı, aşama geçişleri, dakika başına liveness kaydı.
- `job.lock`: aynı işin eşzamanlı iki kez başlamasını engelleyen işletim sistemi kilidi.

`job_active: true` iş kilidinin hâlâ tutulduğunu gösterir. Kalıcı durum `running`
olduğu hâlde kilit serbestse durum komutu `interrupted` gösterir; tamamlanmış
gibi sunmaz. Kilit çocuğa da aktarılır: ana süreç beklenmedik şekilde ölürken
eğitim çocuğu çalışıyorsa ikinci eğitim başlamaz. Heartbeat bu durumda bayatlar.

**`status: completed` yalnız smoke + eğitim + audit başarılı olduktan sonra
yazılır.** Herhangi biri başarısızsa `error` ve ilgili log dosyası gösterilir;
sonraki aşama çalışmaz. Komutun başlaması eğitim tamamlandı demek değildir.

## Çıktı ve devam

Mevcut evidence_v1 kimliği için tam koşu dizini:

```text
data/processed/evidence_v1/listwise/ff0c305a3479fcad/
```

`run_dir` alanı gerçek çıktının yerini gösterir. Farklı deney kimliği farklı
koşu dizini üretir; kimlikler elle taşınmaz. Başarılı koşuda `results.json`,
`audit.json` ve isimli/maskeli fold checkpoint'leri bulunur. Checkpoint denetimi
her fold'dan bir yanıtı yeniden skorlar; bütün veri yeniden skorlanmış gibi
raporlanmaz. Dosyalar Git dışında kalır, Türkçe sonuçların üzerine yazılmaz.

Kesinti sonrası aynı `make modeling-en-train` komutu kullanılabilir. Tamamlanan
fold'lar hash kontrolüyle atlanır; yarım fold son kaydedilmiş epoch'tan devam eder.
Henüz kaydedilmemiş epoch yeniden çalışır. Hata kalıcıysa yalnız tekrar başlatmak
yerine ilgili log incelenmelidir. Devam sırasında model/parametre değiştirilmez.

Ön planda Ctrl+C veya `status.json` içindeki **doğrulanmış bu işe ait PID'ye**
SIGTERM göndermek çocuğu da durdurur ve checkpoint'leri korur. Geniş `pkill`
komutları kullanma; başka işler durdurulmamalıdır. Arka plan işi bilgisayar
kapanmasına veya uykuya karşı koruma sağlamaz.

RTX 4050 üzerindeki Türkçe VPN koşusuna ve yaklaşık 17 kat yanıt sayısına göre
ilk kaba tahmin 2–3 saattir; süre ölçülmüş İngilizce sonuç değildir. Termal durum,
diğer GPU işleri ve girdi uzunluğu etkileyebilir. Epoch/model kaydı ve denetim
için yeterli disk alanı bırakılmalıdır; optimizer checkpoint'leri de tutulur.

## Bilimsel sınır

İngilizce tam VPN eğitimi 10 sorgu, Türkçe VPN eğitimi 5 sorgu kullanır. Ayrıca
üretici modeller, tekrar sayıları, toplama zamanı ve arama locale'i farklıdır.
Genel EN/TR skor farkı saf dil etkisi değildir; ortak beş niyette ayrıca betimsel
karşılaştırma yapılmalıdır. Bu koşu final servis modeli, insan incelemesi veya
yeni markalara genelleme doğrulaması yerine geçmez. Seed 13/21 otomatik başlamaz.
