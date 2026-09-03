# Katkı Rehberi

Bu repo dört kişilik ekip tarafından kısa ömürlü görev dalları ve pull request'lerle
geliştirilir. `main` dalına doğrudan commit veya push yapılmaz.

## 1. Göreve başlama

Önce `main` dalını güncelleyin, ardından görev numarasıyla yeni dal açın:

```bash
git switch main
git pull --ff-only
git switch -c feat/S0-3-reference-data
```

Dal adları:

- Yeni çalışma: `feat/S0-<no>-<konu>`
- Düzeltme: `fix/S0-<no>-<konu>`
- Yalnızca belge: `docs/S0-<no>-<konu>`

Her dal tek bir görevi kapsamalıdır. Başka sprintlerin boş klasörleri veya henüz
kullanılmayan bağımlılıklar eklenmez.

## 2. Dosyaları doğru yere koyma

- `configs/`: Ortak ve sürümlenen ayarlar
- `data/raw/`: Değiştirilmeyen ham veri
- `data/interim/`: Ara dönüşüm çıktıları
- `data/processed/`: Analize hazır veri
- `notebooks/`: Keşif ve doğrulama notebook'ları
- `reports/`: Literatür, maliyet, görüşme ve doğrulama raporları
- `src/`: Birden fazla yerde kullanılacak Python kodu
- `tests/`: `src/` kodunun testleri

Veri dosyaları Git'e gönderilmez. Veriyi indiren veya üreten kod, kullanılan kaynak
ve sonuç özeti paylaşılır. `.env`, API anahtarları ve kişisel görüşme bilgileri commit
edilmez.

## 3. Değişikliği kontrol etme

Commit öncesinde:

```bash
make format
make check
git status
git add path/to/file.py path/to/test_file.py
```

Commit mesajı kısa ve görevle ilişkili olmalıdır:

```bash
git commit -m "feat(S0-3): prepare reference dataset"
```

### Veri seti pipeline'ı değişiklikleri

Model, sorgu, prompt, koşul, tekrar sayısı, locale veya export şeması değişiklikleri
normal kod değişikliği değildir; veri release sözleşmesini etkiler. Böyle bir PR:

- yeni `suite_id` ve sürüme özel data yolları kullanmalı;
- eski raw/judged/cache dosyalarını yeni sürüme taşımamalı;
- plan, manifest, validation ve ilgili testleri birlikte güncellemeli;
- gerçek API çalıştırmadan önce `make check` ve `make dataset-plan` sonuçlarını sunmalı;
- harcama yapacak pilot/tam koşuları PR açıklamasında açıkça belirtmeli;

Mimari, veri sözleşmesi ve release kontrol listesi için
[`docs/veri-seti-gelistirici-rehberi.md`](docs/veri-seti-gelistirici-rehberi.md)
okunmalıdır.

## 4. Pull request

Dalı gönderin ve hedef dalı `main` olan bir pull request açın:

```bash
git push -u origin feat/S0-3-reference-data
```

PR açıklamasında görev numarası, yapılan değişiklik, üretilen çıktı ve çalıştırılan
kontroller bulunmalıdır. Birleştirme için:

- GitHub Actions `quality` kontrolü başarılı olmalı.
- PR sahibinden farklı en az bir ekip üyesi onay vermeli.
- Açık inceleme yorumu kalmamalı.
- Birleştirme `Squash and merge` ile yapılmalı.
- Birleşen görev dalı silinmeli.

Önerilen sıra, “PR sahibi → gözden geçiren” biçiminde: Furkan → Murat → Kübra →
Zeynep → Furkan. Gerektiğinde müsait başka bir ekip üyesi de inceleyebilir; PR sahibi
kendi PR'ını onaylayamaz.

## Bitti tanımı

Bir görev; beklenen çıktı doğru klasördeyse, `make check` geçiyorsa, README/rapor
gerekiyorsa güncellenmişse ve bir ekip üyesi PR'ı onaylamışsa tamamlanmıştır.
