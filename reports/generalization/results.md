# M2-General · alan dışı genelleme ve sinyal kararlılığı (üretilmiş tablolar)

Bu sayfa `scripts/run_generalization.py` tarafından üretilir; elle düzenlemeyin.
Yorum ve sınırlılıklar için `README.md`.

## Performans · `y_top`

| Track | Domain | Model | Pozitif oran | PR-AUC [95% GA] | Top-1 [95% GA] | NDCG@3 |
|---|---|---|---:|---|---|---:|
| en | editors | naive_position | 0.033 | 0.228 [0.185, 0.291] | 0.319 [0.235, 0.418] | 0.526 |
| en | editors | general_seen | 0.033 | 0.666 [0.529, 0.774] | 0.683 [0.530, 0.812] | 0.823 |
| en | editors | general_unseen | 0.033 | 0.348 [0.291, 0.444] | 0.481 [0.388, 0.589] | 0.706 |
| en | editors | invariant_seen | 0.033 | 0.717 [0.570, 0.872] | 0.690 [0.555, 0.800] | 0.825 |
| en | editors | invariant_unseen | 0.033 | 0.675 [0.542, 0.812] | 0.675 [0.545, 0.782] | 0.808 |
| en | editors | M2_full | 0.033 | 0.848 [0.659, 1.000] | 0.892 [0.770, 1.000] | 0.944 |
| en | hosting | naive_position | 0.028 | 0.163 [0.137, 0.198] | 0.195 [0.142, 0.257] | 0.388 |
| en | hosting | general_seen | 0.028 | 0.190 [0.139, 0.299] | 0.283 [0.174, 0.436] | 0.485 |
| en | hosting | general_unseen | 0.028 | 0.192 [0.139, 0.316] | 0.227 [0.146, 0.349] | 0.451 |
| en | hosting | invariant_seen | 0.028 | 0.281 [0.187, 0.472] | 0.343 [0.206, 0.540] | 0.536 |
| en | hosting | invariant_unseen | 0.028 | 0.290 [0.184, 0.474] | 0.346 [0.206, 0.543] | 0.536 |
| en | hosting | M2_full | 0.028 | 0.269 [0.201, 0.365] | 0.346 [0.220, 0.481] | 0.562 |
| en | travel | naive_position | 0.037 | 0.199 [0.127, 0.347] | 0.299 [0.148, 0.511] | 0.496 |
| en | travel | general_seen | 0.037 | 0.287 [0.137, 0.494] | 0.446 [0.218, 0.627] | 0.604 |
| en | travel | general_unseen | 0.037 | 0.249 [0.127, 0.389] | 0.430 [0.210, 0.604] | 0.601 |
| en | travel | invariant_seen | 0.037 | 0.385 [0.180, 0.689] | 0.519 [0.288, 0.713] | 0.684 |
| en | travel | invariant_unseen | 0.037 | 0.361 [0.176, 0.639] | 0.495 [0.266, 0.665] | 0.670 |
| en | travel | M2_full | 0.037 | 0.554 [0.241, 0.978] | 0.756 [0.475, 0.966] | 0.818 |
| en | vpn | naive_position | 0.042 | 0.193 [0.164, 0.222] | 0.266 [0.214, 0.315] | 0.486 |
| en | vpn | general_seen | 0.042 | 0.532 [0.415, 0.650] | 0.618 [0.483, 0.739] | 0.769 |
| en | vpn | general_unseen | 0.042 | 0.256 [0.221, 0.301] | 0.323 [0.228, 0.417] | 0.592 |
| en | vpn | invariant_seen | 0.042 | 0.701 [0.583, 0.801] | 0.626 [0.497, 0.739] | 0.807 |
| en | vpn | invariant_unseen | 0.042 | 0.566 [0.435, 0.724] | 0.622 [0.497, 0.744] | 0.796 |
| en | vpn | M2_full | 0.042 | 0.688 [0.609, 0.757] | 0.614 [0.504, 0.711] | 0.832 |
| tr | cosmetics | naive_position | 0.014 | 0.121 [0.062, 0.279] | 0.167 [0.083, 0.286] | 0.235 |
| tr | cosmetics | general_seen | 0.014 | 0.147 [0.052, 0.269] | 0.125 [0.000, 0.286] | 0.277 |
| tr | cosmetics | general_en_transfer | 0.014 | 0.156 [0.041, 0.352] | 0.208 [0.000, 0.429] | 0.329 |
| tr | cosmetics | invariant_en_transfer | 0.014 | 0.214 [0.090, 0.519] | 0.250 [0.083, 0.429] | 0.381 |
| tr | cosmetics | M2_full | 0.014 | 0.241 [0.159, 0.428] | 0.167 [0.000, 0.400] | 0.387 |
| tr | vpn | naive_position | 0.042 | 0.211 [0.160, 0.303] | 0.158 [0.060, 0.250] | 0.411 |
| tr | vpn | general_seen | 0.042 | 0.246 [0.159, 0.488] | 0.246 [0.077, 0.431] | 0.460 |
| tr | vpn | general_en_transfer | 0.042 | 0.297 [0.252, 0.389] | 0.228 [0.122, 0.317] | 0.482 |
| tr | vpn | invariant_en_transfer | 0.042 | 0.485 [0.324, 0.664] | 0.421 [0.211, 0.589] | 0.615 |
| tr | vpn | M2_full | 0.042 | 0.239 [0.175, 0.414] | 0.228 [0.094, 0.400] | 0.492 |

### Eşleştirilmiş farklar · `y_top` (a − b, sorgu-küme bootstrap)

| Track | Domain | Karşılaştırma | ΔPR-AUC [95% GA] | ΔTop-1 [95% GA] | ΔNDCG@3 [95% GA] |
|---|---|---|---|---|---|
| en | editors | transfer_gap | 0.318 [0.213, 0.393] | 0.201 [0.108, 0.287] | 0.117 [0.079, 0.156] |
| en | editors | prior_dependence | 0.182 [0.060, 0.283] | 0.210 [0.114, 0.300] | 0.121 [0.056, 0.177] |
| en | editors | unseen_vs_position | 0.120 [0.049, 0.193] | 0.162 [0.017, 0.309] | 0.180 [0.074, 0.268] |
| en | editors | invariant_transfer_gap | 0.041 [-0.019, 0.088] | 0.015 [0.004, 0.028] | 0.017 [0.009, 0.026] |
| en | editors | invariant_vs_general_unseen | 0.328 [0.220, 0.439] | 0.194 [0.108, 0.275] | 0.102 [0.067, 0.140] |
| en | hosting | transfer_gap | -0.002 [-0.021, 0.017] | 0.056 [0.022, 0.099] | 0.034 [0.010, 0.054] |
| en | hosting | prior_dependence | 0.079 [-0.007, 0.155] | 0.063 [-0.109, 0.232] | 0.077 [-0.017, 0.157] |
| en | hosting | unseen_vs_position | 0.029 [-0.017, 0.145] | 0.033 [-0.068, 0.156] | 0.063 [-0.027, 0.182] |
| en | hosting | invariant_transfer_gap | -0.008 [-0.037, 0.025] | -0.003 [-0.017, 0.011] | 0.000 [-0.009, 0.012] |
| en | hosting | invariant_vs_general_unseen | 0.098 [0.028, 0.190] | 0.118 [0.055, 0.207] | 0.084 [0.046, 0.124] |
| en | travel | transfer_gap | 0.038 [-0.028, 0.178] | 0.016 [-0.157, 0.169] | 0.003 [-0.117, 0.090] |
| en | travel | prior_dependence | 0.267 [0.101, 0.589] | 0.309 [0.136, 0.529] | 0.214 [0.081, 0.371] |
| en | travel | unseen_vs_position | 0.049 [-0.060, 0.186] | 0.131 [-0.063, 0.328] | 0.105 [-0.030, 0.243] |
| en | travel | invariant_transfer_gap | 0.024 [-0.006, 0.113] | 0.023 [-0.018, 0.070] | 0.014 [-0.008, 0.039] |
| en | travel | invariant_vs_general_unseen | 0.113 [0.030, 0.298] | 0.065 [-0.022, 0.183] | 0.069 [0.020, 0.129] |
| en | vpn | transfer_gap | 0.276 [0.185, 0.380] | 0.295 [0.190, 0.416] | 0.178 [0.119, 0.233] |
| en | vpn | prior_dependence | 0.156 [-0.005, 0.266] | -0.004 [-0.167, 0.151] | 0.063 [-0.052, 0.172] |
| en | vpn | unseen_vs_position | 0.063 [0.018, 0.110] | 0.057 [-0.058, 0.164] | 0.106 [-0.006, 0.207] |
| en | vpn | invariant_transfer_gap | 0.135 [0.061, 0.193] | 0.004 [-0.042, 0.034] | 0.010 [-0.013, 0.029] |
| en | vpn | invariant_vs_general_unseen | 0.310 [0.205, 0.433] | 0.298 [0.211, 0.410] | 0.204 [0.157, 0.249] |
| tr | cosmetics | en_transfer_vs_tr_trained | 0.009 [-0.011, 0.156] | 0.083 [-0.083, 0.400] | 0.052 [-0.042, 0.274] |
| tr | cosmetics | en_transfer_vs_position | 0.035 [-0.027, 0.118] | 0.042 [-0.083, 0.200] | 0.094 [-0.083, 0.326] |
| tr | cosmetics | invariant_en_transfer_vs_position | 0.094 [0.025, 0.362] | 0.083 [0.000, 0.200] | 0.147 [0.011, 0.326] |
| tr | cosmetics | invariant_vs_general_en_transfer | 0.059 [-0.016, 0.244] | 0.042 [0.000, 0.083] | 0.053 [0.000, 0.094] |
| tr | vpn | en_transfer_vs_tr_trained | 0.050 [-0.124, 0.136] | -0.018 [-0.135, 0.125] | 0.023 [-0.088, 0.155] |
| tr | vpn | en_transfer_vs_position | 0.086 [-0.013, 0.201] | 0.070 [-0.063, 0.220] | 0.071 [-0.047, 0.214] |
| tr | vpn | invariant_en_transfer_vs_position | 0.274 [0.044, 0.481] | 0.263 [0.019, 0.500] | 0.204 [0.076, 0.346] |
| tr | vpn | invariant_vs_general_en_transfer | 0.188 [0.026, 0.319] | 0.193 [0.036, 0.333] | 0.133 [0.038, 0.198] |

## Performans · `y_mention`

| Track | Domain | Model | Pozitif oran | PR-AUC [95% GA] | Top-1 [95% GA] | NDCG@3 |
|---|---|---|---:|---|---|---:|
| en | editors | naive_position | 0.258 | 0.611 [0.555, 0.673] | – | 0.745 |
| en | editors | general_seen | 0.258 | 0.691 [0.630, 0.763] | – | 0.876 |
| en | editors | general_unseen | 0.258 | 0.684 [0.629, 0.749] | – | 0.847 |
| en | editors | invariant_seen | 0.258 | 0.726 [0.673, 0.782] | – | 0.873 |
| en | editors | invariant_unseen | 0.258 | 0.720 [0.663, 0.777] | – | 0.858 |
| en | editors | M2_full | 0.258 | 0.761 [0.704, 0.817] | – | 0.910 |
| en | hosting | naive_position | 0.207 | 0.647 [0.574, 0.705] | – | 0.786 |
| en | hosting | general_seen | 0.207 | 0.719 [0.677, 0.759] | – | 0.898 |
| en | hosting | general_unseen | 0.207 | 0.707 [0.660, 0.752] | – | 0.902 |
| en | hosting | invariant_seen | 0.207 | 0.733 [0.678, 0.782] | – | 0.894 |
| en | hosting | invariant_unseen | 0.207 | 0.731 [0.676, 0.780] | – | 0.901 |
| en | hosting | M2_full | 0.207 | 0.728 [0.667, 0.774] | – | 0.889 |
| en | travel | naive_position | 0.235 | 0.658 [0.584, 0.725] | – | 0.734 |
| en | travel | general_seen | 0.235 | 0.693 [0.618, 0.752] | – | 0.777 |
| en | travel | general_unseen | 0.235 | 0.689 [0.599, 0.756] | – | 0.748 |
| en | travel | invariant_seen | 0.235 | 0.715 [0.653, 0.763] | – | 0.768 |
| en | travel | invariant_unseen | 0.235 | 0.706 [0.640, 0.762] | – | 0.749 |
| en | travel | M2_full | 0.235 | 0.779 [0.696, 0.836] | – | 0.913 |
| en | vpn | naive_position | 0.184 | 0.682 [0.624, 0.745] | – | 0.784 |
| en | vpn | general_seen | 0.184 | 0.896 [0.847, 0.931] | – | 0.957 |
| en | vpn | general_unseen | 0.184 | 0.865 [0.811, 0.903] | – | 0.938 |
| en | vpn | invariant_seen | 0.184 | 0.912 [0.854, 0.948] | – | 0.963 |
| en | vpn | invariant_unseen | 0.184 | 0.895 [0.840, 0.932] | – | 0.957 |
| en | vpn | M2_full | 0.184 | 0.926 [0.886, 0.958] | – | 0.974 |
| tr | cosmetics | naive_position | 0.128 | 0.409 [0.309, 0.524] | – | 0.499 |
| tr | cosmetics | general_seen | 0.128 | 0.439 [0.342, 0.579] | – | 0.588 |
| tr | cosmetics | general_en_transfer | 0.128 | 0.453 [0.341, 0.568] | – | 0.521 |
| tr | cosmetics | invariant_en_transfer | 0.128 | 0.485 [0.366, 0.603] | – | 0.541 |
| tr | cosmetics | M2_full | 0.128 | 0.510 [0.408, 0.630] | – | 0.692 |
| tr | vpn | naive_position | 0.186 | 0.599 [0.528, 0.697] | – | 0.632 |
| tr | vpn | general_seen | 0.186 | 0.724 [0.675, 0.816] | – | 0.742 |
| tr | vpn | general_en_transfer | 0.186 | 0.745 [0.684, 0.826] | – | 0.753 |
| tr | vpn | invariant_en_transfer | 0.186 | 0.794 [0.742, 0.852] | – | 0.753 |
| tr | vpn | M2_full | 0.186 | 0.809 [0.737, 0.900] | – | 0.834 |

### Eşleştirilmiş farklar · `y_mention` (a − b, sorgu-küme bootstrap)

| Track | Domain | Karşılaştırma | ΔPR-AUC [95% GA] | ΔTop-1 [95% GA] | ΔNDCG@3 [95% GA] |
|---|---|---|---|---|---|
| en | editors | transfer_gap | 0.007 [-0.011, 0.025] | – [–, –] | 0.029 [0.013, 0.046] |
| en | editors | prior_dependence | 0.070 [0.008, 0.120] | – [–, –] | 0.034 [-0.024, 0.096] |
| en | editors | unseen_vs_position | 0.073 [0.054, 0.098] | – [–, –] | 0.102 [0.068, 0.142] |
| en | editors | invariant_transfer_gap | 0.006 [-0.000, 0.014] | – [–, –] | 0.015 [0.004, 0.028] |
| en | editors | invariant_vs_general_unseen | 0.036 [0.023, 0.048] | – [–, –] | 0.011 [-0.005, 0.025] |
| en | hosting | transfer_gap | 0.012 [-0.003, 0.029] | – [–, –] | -0.005 [-0.018, 0.006] |
| en | hosting | prior_dependence | 0.009 [-0.039, 0.043] | – [–, –] | -0.009 [-0.042, 0.021] |
| en | hosting | unseen_vs_position | 0.060 [0.033, 0.093] | – [–, –] | 0.116 [0.072, 0.164] |
| en | hosting | invariant_transfer_gap | 0.002 [-0.006, 0.009] | – [–, –] | -0.007 [-0.023, 0.006] |
| en | hosting | invariant_vs_general_unseen | 0.024 [0.011, 0.037] | – [–, –] | -0.001 [-0.011, 0.007] |
| en | travel | transfer_gap | 0.004 [-0.009, 0.022] | – [–, –] | 0.029 [0.012, 0.046] |
| en | travel | prior_dependence | 0.085 [0.004, 0.168] | – [–, –] | 0.137 [0.017, 0.249] |
| en | travel | unseen_vs_position | 0.032 [-0.015, 0.078] | – [–, –] | 0.014 [-0.045, 0.076] |
| en | travel | invariant_transfer_gap | 0.009 [-0.001, 0.023] | – [–, –] | 0.019 [0.006, 0.036] |
| en | travel | invariant_vs_general_unseen | 0.016 [-0.007, 0.049] | – [–, –] | 0.001 [-0.025, 0.020] |
| en | vpn | transfer_gap | 0.031 [0.017, 0.048] | – [–, –] | 0.019 [0.003, 0.045] |
| en | vpn | prior_dependence | 0.030 [0.012, 0.051] | – [–, –] | 0.017 [-0.001, 0.046] |
| en | vpn | unseen_vs_position | 0.184 [0.142, 0.222] | – [–, –] | 0.154 [0.102, 0.198] |
| en | vpn | invariant_transfer_gap | 0.017 [0.010, 0.026] | – [–, –] | 0.005 [-0.002, 0.015] |
| en | vpn | invariant_vs_general_unseen | 0.029 [0.013, 0.047] | – [–, –] | 0.019 [0.004, 0.043] |
| tr | cosmetics | en_transfer_vs_tr_trained | 0.014 [-0.038, 0.028] | – [–, –] | -0.067 [-0.145, 0.011] |
| tr | cosmetics | en_transfer_vs_position | 0.044 [0.022, 0.059] | – [–, –] | 0.022 [-0.004, 0.047] |
| tr | cosmetics | invariant_en_transfer_vs_position | 0.076 [0.028, 0.122] | – [–, –] | 0.042 [-0.013, 0.093] |
| tr | cosmetics | invariant_vs_general_en_transfer | 0.031 [-0.002, 0.066] | – [–, –] | 0.020 [-0.058, 0.096] |
| tr | vpn | en_transfer_vs_tr_trained | 0.021 [-0.017, 0.033] | – [–, –] | 0.011 [-0.005, 0.037] |
| tr | vpn | en_transfer_vs_position | 0.146 [0.093, 0.163] | – [–, –] | 0.122 [0.070, 0.158] |
| tr | vpn | invariant_en_transfer_vs_position | 0.194 [0.131, 0.219] | – [–, –] | 0.121 [0.074, 0.161] |
| tr | vpn | invariant_vs_general_en_transfer | 0.049 [0.023, 0.062] | – [–, –] | -0.001 [-0.016, 0.012] |

## Sinyal kararlılık matrisi · `y_mention`

↑/↓: kazanan–kaybeden farkı 95% GA ile sıfırdan ayrık; →: ayrık değil; ·: < 5 bağımsız sorgu, yön çağrılmadı.

| Sinyal | en/editors | en/hosting | en/travel | en/vpn | tr/cosmetics | tr/vpn | Ort. etki | SHAP payı (medyan) | SHAP yön uyumu | LODO ΔPR-AUC | Kararlılık | Hacim-eşli ort. etki | Hacim-eşli kararlılık |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---|---:|---|
| `in_search_results` | ↑ | ↑ | ↑ | ↑ | ↑ | ↑ | 0.614 | 0.041 | 0.67 | 0.0431 | Güçlü | 0.614 | Güçlü |
| `n_results_mentioning` | ↑ | ↑ | ↑ | ↑ | ↑ | ↑ | 0.571 | 0.151 | 1.00 | 0.1160 | Güçlü | 0.571 | Güçlü |
| `best_position` | ↑ | ↑ | ↑ | ↑ | → | ↑ | 0.386 | 0.078 | 0.67 | -0.0012 | Güçlü | 0.028 | Zayıf |
| `n_official` | ↑ | ↑ | ↑ | ↑ | → | ↑ | 0.140 | 0.009 | 0.83 | 0.0019 | Güçlü | 0.014 | Orta |
| `n_editorial` | ↑ | → | → | ↑ | ↓ | ↑ | 0.187 | 0.018 | 0.83 | -0.0040 | Çelişkili | -0.091 | Orta |
| `n_affiliate` | ↑ | ↑ | → | ↑ | → | ↑ | 0.192 | 0.008 | 0.50 | -0.0015 | Güçlü | -0.009 | Zayıf |
| `n_forum` | ↑ | ↑ | ↑ | ↑ | ↑ | ↑ | 0.352 | 0.019 | 1.00 | -0.0004 | Güçlü | 0.042 | Çelişkili |
| `n_retailer` | → | → | ↑ | → | ↑ | → | 0.033 | 0.000 | 0.17 | 0.0000 | Orta | 0.003 | Zayıf |
| `n_unknown` | ↑ | ↑ | ↑ | ↑ | ↑ | ↑ | 0.367 | 0.038 | 0.83 | 0.0002 | Güçlü | 0.044 | Zayıf |
| `lex_authority` | ↑ | ↑ | → | ↑ | → | ↑ | 0.093 | 0.001 | 0.17 | -0.0006 | Güçlü | -0.004 | Orta |
| `lex_social_proof` | ↑ | ↑ | ↑ | ↑ | → | → | 0.040 | 0.003 | 0.67 | 0.0013 | Güçlü | 0.005 | Zayıf |
| `lex_specificity` | ↑ | ↑ | ↑ | ↑ | → | ↑ | 0.107 | 0.004 | 0.50 | -0.0024 | Güçlü | 0.029 | Orta |
| `lex_hedging` | ↑ | ↑ | ↑ | ↑ | → | → | 0.056 | 0.001 | 0.50 | -0.0005 | Güçlü | 0.004 | Zayıf |
| `lex_superlative` | ↓ | ↓ | → | ↓ | ↑ | ↓ | -0.154 | 0.016 | 0.83 | -0.0007 | Çelişkili | -0.132 | Çelişkili |
| `snippet_words` | → | ↑ | → | ↑ | → | ↑ | 0.195 | 0.241 | 0.83 | 0.0757 | Orta | 0.154 | Orta |
| `snippet_has_year` | ↑ | ↑ | ↑ | ↑ | → | ↑ | 0.151 | 0.007 | 0.67 | -0.0007 | Güçlü | 0.011 | Çelişkili |
| `snippet_digit_share` | ↓ | ↓ | → | → | → | ↑ | 0.015 | 0.033 | 0.50 | -0.0004 | Çelişkili | -0.030 | Çelişkili |

## Sinyal kararlılık matrisi · `y_top`

↑/↓: kazanan–kaybeden farkı 95% GA ile sıfırdan ayrık; →: ayrık değil; ·: < 5 bağımsız sorgu, yön çağrılmadı.

| Sinyal | en/editors | en/hosting | en/travel | en/vpn | tr/cosmetics | tr/vpn | Ort. etki | SHAP payı (medyan) | SHAP yön uyumu | LODO ΔPR-AUC | Kararlılık | Hacim-eşli ort. etki | Hacim-eşli kararlılık |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---|---:|---|
| `in_search_results` | ↑ | ↑ | ↑ | ↑ | · | ↑ | 0.641 | 0.000 | 0.00 | 0.0000 | Güçlü | 0.641 | Güçlü |
| `n_results_mentioning` | ↑ | ↑ | ↑ | ↑ | · | ↑ | 0.749 | 0.145 | 1.00 | 0.1012 | Güçlü | 0.749 | Güçlü |
| `best_position` | ↑ | ↑ | ↑ | ↑ | · | ↑ | 0.504 | 0.174 | 1.00 | 0.0187 | Güçlü | 0.184 | Zayıf |
| `n_official` | → | ↑ | ↓ | ↑ | · | ↑ | 0.125 | 0.008 | 0.60 | -0.0016 | Çelişkili | -0.077 | Orta |
| `n_editorial` | ↑ | → | → | ↑ | · | ↑ | 0.370 | 0.016 | 0.80 | -0.0149 | Orta | -0.007 | Zayıf |
| `n_affiliate` | ↑ | ↑ | ↑ | ↑ | · | ↑ | 0.283 | 0.006 | 0.80 | -0.0012 | Güçlü | -0.027 | Zayıf |
| `n_forum` | ↑ | ↑ | ↑ | ↑ | · | → | 0.588 | 0.030 | 1.00 | 0.0310 | Güçlü | 0.154 | Zayıf |
| `n_retailer` | → | → | → | → | · | → | 0.014 | 0.000 | 0.20 | -0.0000 | Zayıf | -0.012 | Zayıf |
| `n_unknown` | ↑ | ↑ | ↑ | ↑ | · | ↑ | 0.512 | 0.020 | 0.80 | 0.0036 | Güçlü | -0.062 | Zayıf |
| `lex_authority` | ↑ | → | → | ↑ | · | ↑ | 0.150 | 0.005 | 0.60 | -0.0030 | Orta | 0.004 | Zayıf |
| `lex_social_proof` | ↑ | → | → | → | · | → | 0.094 | 0.001 | 0.20 | -0.0165 | Zayıf | -0.000 | Zayıf |
| `lex_specificity` | ↑ | ↑ | → | ↑ | · | → | 0.109 | 0.004 | 0.20 | 0.0017 | Orta | -0.006 | Zayıf |
| `lex_hedging` | ↑ | → | ↑ | ↑ | · | → | 0.071 | 0.002 | 0.20 | -0.0014 | Orta | 0.044 | Zayıf |
| `lex_superlative` | → | ↓ | ↓ | → | · | ↓ | -0.164 | 0.031 | 0.60 | -0.0076 | Orta | -0.082 | Çelişkili |
| `snippet_words` | ↑ | → | ↑ | ↑ | · | ↑ | 0.257 | 0.050 | 1.00 | 0.0316 | Güçlü | 0.288 | Orta |
| `snippet_has_year` | ↑ | ↑ | ↑ | ↑ | · | ↑ | 0.182 | 0.005 | 0.40 | 0.0026 | Güçlü | 0.033 | Zayıf |
| `snippet_digit_share` | → | → | → | → | · | → | -0.019 | 0.058 | 1.00 | 0.0114 | Zayıf | -0.022 | Zayıf |
