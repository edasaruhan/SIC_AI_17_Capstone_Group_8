# Görünürlük danışmanı (LangGraph)

Bir marka ve sektör verilir; akış canlı arama yapar, asistana soruyu arama açık ve
kapalı olarak sorar, markanın görünürlüğünü **ölçer**, üç durumdan hangisinde
olduğunu **teşhis eder** ve o duruma özgü öneriyi, kontrollü testte **ölçülmüş
etkisiyle** birlikte verir.

```bash
make advisor-plan BRAND="Windscribe" DOMAIN=vpn LANGUAGE=tr   # ücretsiz: çağrı tahmini
make advisor      BRAND="Windscribe" DOMAIN=vpn LANGUAGE=tr   # ÜCRETLİ: ~15 çağrı
```

Anahtarlar `.env` içinde: `GEMINI_API_KEY`, `SERPER_API_KEY`. Varsayılan koşu 3 sorgu ×
(1 arama + 2 koşul × 2 tekrar) = 15 çağrıdır ve `--max-calls 20` bütçesiyle sınırlıdır;
tahmin bütçeyi aşarsa hiçbir çağrı yapılmaz. `ADVISOR_ARGS="--queries 5 --reps 3
--max-calls 40"` ile genişletilebilir.

## Akış

```
plan → search → interrogate → analyse ─┬─ absent   ─┐
                                       ├─ low_rank ─┤
                                       ├─ ceiling  ─┼→ report
                                       ├─ leader   ─┤
                                       └─ thin     ─┘
```

| Düğüm | Ne yapar | Ücret |
|---|---|---|
| `plan` | Sektör/dil için sorguları donmuş korpustan alır; korpusta yoksa asistana ürettirir | Korpusta varsa yok |
| `search` | Her sorgu için Serper, Türkiye/Türkçe, 10 organik sonuç | Sorgu başına 1 |
| `interrogate` | Gemini'ye soruyu arama kapalı ve arama sonuçlarıyla açık olarak sorar | Sorgu × 2 koşul × tekrar |
| `analyse` | Marka registry'si ile ölçer, teşhis koyar, rakipleri ve outreach hedeflerini bulur | Yok |
| `advise_*` | Teşhise özgü öneri; etkiler `reports/intervention/effects.csv`'den | Yok |
| `report` | Markdown rapor; her öneri etik filtreden geçmiştir | Yok |

## Neden grafik: teşhis dallanması

Kontrollü test üç şey ölçtü ve bunlar üç ayrı duruma karşılık geliyor:

| Teşhis | Anlamı | Öneri | Dayanak |
|---|---|---|---|
| **absent** | Arama sonuçlarında yoksun | Rakiplerini anan bağımsız karşılaştırma sayfalarına gir | +31 puan, ölçüldü |
| **low_rank** | Sonuçlarda varsın ama alt sıralarda | Girdiğin sayfalarda üst sıraya çık | +31 puan, ölçüldü |
| **ceiling** | Anılıyorsun ama asla ilk değilsin | **Bunu içerikle çözemezsin**; bütçeyi listeye girmeye ayır | 1.080 çağrıda 0 |
| **leader** | Zaten öndesin | Konumu koru | — |

Piyasadaki araçlar üç duruma da "içerik üretin" der. Üçüncü duruma dürüst cevap
verebilmek ürünün farkıdır.

Teşhis eşikleri (`advise.py` başındaki `PRESENCE_FLOOR`, `RANK_FLOOR` vb.) ürün
kararıdır, ölçüm değildir: hangi önerinin gösterileceğini belirler, etkinin ne kadar
olduğunu değil.

## Tasarım kararları

- **Ölçüm modelin görüşü değildir.** Anılma, ilk anılma, arama sonucunda görünme ve
  sıra; hepsi küratörlü marka registry'siyle metin eşleştirerek ölçülür
  (`measure.py`). "Bu marka görünür mü?" diye bir LLM'e sormak döngüsel olurdu.
- **Etki büyüklükleri uydurulmaz.** Raporda geçen her etki kontrollü testten gelir;
  bu koşuda yeniden ölçülmez ve rapor bunu söyler.
- **Makbuzlar korunur.** Ücretli her çağrı `brand_demo.workflow.Receipts` ile yazılır.
  Aynı marka yeniden koşulduğunda önbellekten okunur, yeniden ödenmez; hata sessizce
  tekrar denenmez. LangGraph'ın kendi checkpointer'ı bunun yerine kullanılmadı: hash
  doğrulamalı makbuz, ücretli çağrının iki kez yapılmadığını denetlenebilir kılar.
- **Aynı asistan.** Canlı çağrılar Gemini 3.5 Flash Lite ile yapılır; etkileri ölçtüğümüz
  asistanla aynıdır.
- **Rakip siteleri hedef değildir.** Outreach listesi, rakipleri anan ama markayı anmayan
  **bağımsız** sayfalardır; bir rakibin kendi sitesi kaynak taksonomisindeki resmî alan
  adı listesiyle elenir.
- **Etik filtre.** Her öneri `visibility.ethics.screen`'den geçer.
- **`uv.lock` değişmez.** `langgraph` yalnız `graph.py` içinde kullanılır ve çalışma
  anında `uv run --with langgraph` ile kurulur, çünkü `uv.lock` kanıt manifestlerinde
  hash'lidir. Ölçüm, teşhis ve öneri modülleri LangGraph'a bağımlı değildir ve normal
  test paketinde sınanır.

## Bu rapor ne söylemez

- Etkiler tek asistanda, İngilizce bağlamda ve üç sektörde ölçüldü. Başka bir sektör ve
  dilde yön aynı olsa da büyüklük ölçülmedi.
- Serper sonuçları, asistanın gerçekte getireceği sayfalarla aynı olmayabilir.
- "İlk anılan marka", birincil önerinin yaklaşık bir vekilidir.
- Varsayılan koşu az sorgu ve tekrarla çalışır; oranlar yön gösterir, kesin değildir.

## Örnek

`Windscribe`, VPN, Türkçe, 15 çağrı (12 Eylül 2026): arama sonuçlarının %100'ünde
görünüyor ama ortanca sırası 6 → teşhis **low_rank**. Rakipleri anan bağımsız hedefler:
01net, DonanımHaber, Technopat, Webtekno, Siberatay.
