# Yoğunlaşma ve adalet (üretilmiş tablolar)

Bu sayfa `scripts/run_fairness.py` tarafından üretilir; elle düzenlemeyin.
Ölçülerin tanımı ve önceden sabitlenen kurallar `src/visibility/fairness.py`
başındadır; yorum ve sınırlılıklar `README.md` içindedir.

`N_eff = 1/HHI`: asistan, sektördeki *kaç* marka varmış gibi davranıyor.
Registry'de VPN 24, hosting 36, editör 30, seyahat 27, kozmetik 74 marka var;
N_eff bu sayıyla karşılaştırılarak okunur.

## Yoğunlaşma · tüm asistanlar birlikte

| Track | Sektör | Koşul | Ölçü | N_eff [95% GA] | İlk-3 payı [95% GA] | Hiç anılmayan | Sorgu |
|---|---|---|---|---|---|---:|---:|
| en | Kod editörleri | arama kapalı | anılma | 14.68 [12.10, 16.66] | 0.31 [0.28, 0.38] | 0/30 | 10 |
| en | Kod editörleri | arama açık | anılma | 14.77 [11.85, 16.78] | 0.31 [0.28, 0.38] | 0/30 | 10 |
| en | Kod editörleri | arama kapalı | birincil öneri | 1.14 [1.00, 1.37] | 0.97 [0.94, 1.00] | 24/30 | 10 |
| en | Kod editörleri | arama açık | birincil öneri | 1.25 [1.00, 1.63] | 0.97 [0.94, 1.00] | 24/30 | 10 |
| en | Hosting / bulut | arama kapalı | anılma | 18.92 [14.10, 22.73] | 0.24 [0.20, 0.30] | 1/36 | 10 |
| en | Hosting / bulut | arama açık | anılma | 20.76 [15.38, 22.97] | 0.22 [0.19, 0.30] | 0/36 | 10 |
| en | Hosting / bulut | arama kapalı | birincil öneri | 4.62 [2.33, 5.85] | 0.71 [0.64, 0.92] | 18/36 | 10 |
| en | Hosting / bulut | arama açık | birincil öneri | 5.44 [2.55, 6.87] | 0.67 [0.55, 0.87] | 18/36 | 10 |
| en | Seyahat | arama kapalı | anılma | 12.05 [7.67, 15.66] | 0.42 [0.31, 0.55] | 0/27 | 10 |
| en | Seyahat | arama açık | anılma | 11.77 [8.18, 14.43] | 0.43 [0.34, 0.54] | 0/27 | 10 |
| en | Seyahat | arama kapalı | birincil öneri | 1.48 [1.00, 2.46] | 0.98 [0.96, 1.00] | 22/27 | 10 |
| en | Seyahat | arama açık | birincil öneri | 1.71 [1.06, 3.08] | 0.90 [0.84, 1.00] | 17/27 | 10 |
| en | VPN | arama kapalı | anılma | 6.53 [5.98, 6.97] | 0.57 [0.53, 0.62] | 7/24 | 10 |
| en | VPN | arama açık | anılma | 6.29 [5.73, 6.83] | 0.58 [0.54, 0.61] | 8/24 | 10 |
| en | VPN | arama kapalı | birincil öneri | 1.32 [1.08, 1.70] | 1.00 [0.99, 1.00] | 19/24 | 10 |
| en | VPN | arama açık | birincil öneri | 2.45 [1.65, 2.64] | 0.98 [0.96, 0.99] | 19/24 | 10 |
| tr | Kozmetik | arama kapalı | anılma | 28.93 [16.64, 31.27] | 0.22 [0.19, 0.33] | 13/74 | 5 |
| tr | Kozmetik | arama açık | anılma | 35.88 [19.86, 36.85] | 0.17 [0.15, 0.28] | 6/74 | 5 |
| tr | Kozmetik | arama kapalı | birincil öneri | 4.15 [1.38, 5.54] | 0.78 [0.67, 1.00] | 67/74 | 2 |
| tr | Kozmetik | arama açık | birincil öneri | 3.60 [1.32, 5.04] | 0.79 [0.73, 1.00] | 66/74 | 3 |
| tr | VPN | arama kapalı | anılma | 6.96 [5.77, 7.50] | 0.55 [0.52, 0.63] | 9/24 | 5 |
| tr | VPN | arama açık | anılma | 6.97 [5.63, 7.41] | 0.56 [0.53, 0.68] | 7/24 | 5 |
| tr | VPN | arama kapalı | birincil öneri | 3.88 [2.22, 4.39] | 0.81 [0.74, 1.00] | 19/24 | 5 |
| tr | VPN | arama açık | birincil öneri | 3.95 [1.81, 4.46] | 0.81 [0.72, 1.00] | 19/24 | 5 |

## Arama açmak pazarı açıyor mu? (N_eff farkı, arama açık − kapalı)

| Track | Sektör | Asistan | Ölçü | N_eff kapalı | N_eff açık | Fark |
|---|---|---|---|---:|---:|---:|
| en | Kod editörleri | ALL | anılma | 14.68 | 14.77 | 0.09 |
| en | Kod editörleri | ALL | birincil öneri | 1.14 | 1.25 | 0.11 |
| en | Kod editörleri | claude-opus-4-6 | anılma | 13.58 | 15.03 | 1.45 |
| en | Kod editörleri | claude-opus-4-6 | birincil öneri | 1.04 | 1.22 | 0.18 |
| en | Kod editörleri | gpt-5.4 | anılma | 14.63 | 14.59 | -0.04 |
| en | Kod editörleri | gpt-5.4 | birincil öneri | 1.13 | 1.26 | 0.13 |
| en | Kod editörleri | grok-4.20-0309-reasoning | anılma | 14.35 | 13.49 | -0.86 |
| en | Kod editörleri | grok-4.20-0309-reasoning | birincil öneri | 1.33 | 1.40 | 0.07 |
| en | Kod editörleri | zai-org/GLM-5 | anılma | 14.04 | 13.78 | -0.26 |
| en | Kod editörleri | zai-org/GLM-5 | birincil öneri | 1.08 | 1.12 | 0.05 |
| en | Hosting / bulut | ALL | anılma | 18.92 | 20.76 | 1.84 |
| en | Hosting / bulut | ALL | birincil öneri | 4.62 | 5.44 | 0.83 |
| en | Hosting / bulut | claude-opus-4-6 | anılma | 17.74 | 22.45 | 4.70 |
| en | Hosting / bulut | claude-opus-4-6 | birincil öneri | 4.82 | 5.39 | 0.58 |
| en | Hosting / bulut | gpt-5.4 | anılma | 17.37 | 16.30 | -1.06 |
| en | Hosting / bulut | gpt-5.4 | birincil öneri | 3.04 | 3.80 | 0.77 |
| en | Hosting / bulut | grok-4.20-0309-reasoning | anılma | 18.89 | 20.45 | 1.56 |
| en | Hosting / bulut | grok-4.20-0309-reasoning | birincil öneri | 4.68 | 4.96 | 0.28 |
| en | Hosting / bulut | zai-org/GLM-5 | anılma | 19.00 | 20.19 | 1.18 |
| en | Hosting / bulut | zai-org/GLM-5 | birincil öneri | 4.42 | 6.30 | 1.88 |
| en | Seyahat | ALL | anılma | 12.05 | 11.77 | -0.28 |
| en | Seyahat | ALL | birincil öneri | 1.48 | 1.71 | 0.23 |
| en | Seyahat | claude-opus-4-6 | anılma | 13.11 | 14.20 | 1.10 |
| en | Seyahat | claude-opus-4-6 | birincil öneri | 1.83 | 1.34 | -0.49 |
| en | Seyahat | gpt-5.4 | anılma | 10.05 | 8.90 | -1.15 |
| en | Seyahat | gpt-5.4 | birincil öneri | 1.48 | 1.44 | -0.05 |
| en | Seyahat | grok-4.20-0309-reasoning | anılma | 11.22 | 12.37 | 1.14 |
| en | Seyahat | grok-4.20-0309-reasoning | birincil öneri | 1.43 | 2.26 | 0.83 |
| en | Seyahat | zai-org/GLM-5 | anılma | 11.37 | 10.23 | -1.14 |
| en | Seyahat | zai-org/GLM-5 | birincil öneri | 1.30 | 1.78 | 0.48 |
| en | VPN | ALL | anılma | 6.53 | 6.29 | -0.24 |
| en | VPN | ALL | birincil öneri | 1.32 | 2.45 | 1.13 |
| en | VPN | claude-opus-4-6 | anılma | 5.44 | 6.50 | 1.06 |
| en | VPN | claude-opus-4-6 | birincil öneri | 1.04 | 2.10 | 1.06 |
| en | VPN | gpt-5.4 | anılma | 5.98 | 5.72 | -0.26 |
| en | VPN | gpt-5.4 | birincil öneri | 1.13 | 2.68 | 1.55 |
| en | VPN | grok-4.20-0309-reasoning | anılma | 7.11 | 6.42 | -0.69 |
| en | VPN | grok-4.20-0309-reasoning | birincil öneri | 1.02 | 1.94 | 0.92 |
| en | VPN | zai-org/GLM-5 | anılma | 6.05 | 5.91 | -0.14 |
| en | VPN | zai-org/GLM-5 | birincil öneri | 2.16 | 2.25 | 0.09 |
| tr | Kozmetik | ALL | anılma | 28.93 | 35.88 | 6.95 |
| tr | Kozmetik | ALL | birincil öneri | 4.15 | 3.60 | -0.55 |
| tr | Kozmetik | MiniMax-M2.7 | anılma | 23.37 | 32.98 | 9.61 |
| tr | Kozmetik | MiniMax-M2.7 | birincil öneri | 1.80 | 3.60 | 1.80 |
| tr | Kozmetik | abliterated-model-large-v2 | anılma | 21.55 | 29.74 | 8.19 |
| tr | Kozmetik | abliterated-model-large-v2 | birincil öneri | 3.00 | 2.28 | -0.72 |
| tr | Kozmetik | gemini-3.5-flash-lite | anılma | 29.10 | 26.12 | -2.98 |
| tr | Kozmetik | gemini-3.5-flash-lite | birincil öneri | 3.86 | 2.33 | -1.52 |
| tr | VPN | ALL | anılma | 6.96 | 6.97 | 0.02 |
| tr | VPN | ALL | birincil öneri | 3.88 | 3.95 | 0.07 |
| tr | VPN | MiniMax-M2.7 | anılma | 8.07 | 8.43 | 0.36 |
| tr | VPN | MiniMax-M2.7 | birincil öneri | 2.00 | 2.96 | 0.96 |
| tr | VPN | abliterated-model-large-v2 | anılma | 4.29 | 5.69 | 1.40 |
| tr | VPN | abliterated-model-large-v2 | birincil öneri | 2.03 | 3.97 | 1.94 |
| tr | VPN | gemini-3.5-flash-lite | anılma | 6.19 | 5.83 | -0.36 |
| tr | VPN | gemini-3.5-flash-lite | birincil öneri | 3.26 | 3.48 | 0.22 |

## Aramadan kim kazanıyor? (tanınırlık tercilleri)

Terciller arama-kapalı anılma oranına göre bir kez atanır; bootstrap yalnız
sabit tercil içindeki kazancı değiştirir. Kazanç puan cinsindendir.

| Track | Sektör | Tanınırlık | Marka | Aramasız % | Aramalı % | Kazanç [95% GA] |
|---|---|---|---:|---:|---:|---|
| en | Kod editörleri | düşük | 10 | 4.3 | 5.1 | +0.8 [-1.0, +2.8] |
| en | Kod editörleri | orta | 10 | 14.4 | 16.3 | +1.9 [-0.5, +4.2] |
| en | Kod editörleri | yüksek | 10 | 59.9 | 55.9 | -4.0 [-9.5, +2.1] |
| en | Hosting / bulut | düşük | 12 | 3.4 | 6.1 | +2.7 [+1.2, +4.2] |
| en | Hosting / bulut | orta | 12 | 15.7 | 15.6 | -0.2 [-4.2, +2.9] |
| en | Hosting / bulut | yüksek | 12 | 52.4 | 40.6 | -11.8 [-14.5, -9.1] |
| en | Seyahat | düşük | 9 | 4.4 | 6.2 | +1.8 [-0.8, +4.0] |
| en | Seyahat | orta | 9 | 12.5 | 11.6 | -0.8 [-3.0, +1.6] |
| en | Seyahat | yüksek | 9 | 49.7 | 52.4 | +2.8 [-1.0, +7.5] |
| en | VPN | düşük | 8 | 0.0 | 0.4 | +0.4 [-0.0, +1.1] |
| en | VPN | orta | 8 | 1.8 | 1.1 | -0.7 [-1.3, +0.1] |
| en | VPN | yüksek | 8 | 50.1 | 53.8 | +3.8 [-1.0, +9.7] |
| tr | Kozmetik | düşük | 24 | 1.0 | 6.3 | +5.3 [+1.8, +9.3] |
| tr | Kozmetik | orta | 25 | 6.2 | 8.7 | +2.5 [+0.9, +4.1] |
| tr | Kozmetik | yüksek | 25 | 24.3 | 23.0 | -1.2 [-5.2, +3.2] |
| tr | VPN | düşük | 8 | 0.0 | 1.2 | +1.2 [+0.5, +2.0] |
| tr | VPN | orta | 8 | 2.3 | 3.5 | +1.2 [-1.0, +3.3] |
| tr | VPN | yüksek | 8 | 41.5 | 51.0 | +9.5 [+5.5, +14.8] |

## Yerli marka farkı

Yalnız Türkiye menşeli markası bulunan sektörler. `unclear` etiketli markalar
analizden çıkarılmıştır.

| Track | Sektör | Menşe | Marka | Aramasız % | Aramalı % | Kazanç [95% GA] |
|---|---|---|---:|---:|---:|---|
| tr | Kozmetik | Küresel | 65 | 10.8 | 13.0 | +2.3 [+0.4, +4.3] |
| tr | Kozmetik | Türkiye menşeli | 9 | 9.5 | 11.0 | +1.5 [-0.6, +3.9] |
