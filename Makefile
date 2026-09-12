.DEFAULT_GOAL := help
export UV_CACHE_DIR := .uv-cache

.PHONY: help setup format check reference-data reference-report \
	dataset-plan dataset-preflight dataset-search-test dataset-pilot dataset-collect \
	dataset-status dataset-judge dataset-judge-status dataset-export dataset-validate dataset-report \
	dataset-all dataset-gemini-pilot dataset-retry-gemini dataset-logs dataset-follow-logs \
	dataset-finish dataset-repair-generation

PYTHON := .venv/bin/python
RUFF := .venv/bin/ruff
PYRIGHT := .venv/bin/pyright
PYTEST := .venv/bin/pytest
BIAS_EVAL := PYTHONPATH=src $(PYTHON) -m bias_eval
ENV_RUN := set -a; [ ! -f .env ] || . ./.env; set +a;
TRAINING_SCRIPTS := scripts/audit_listwise_run.py scripts/run_english_listwise.py scripts/verify_final_models.py
REVIEW_SCRIPTS := scripts/ai_review_report.py scripts/run_generalization.py scripts/run_fairness.py
DATASET_PYTHON_PATHS := src/bias_eval src/evidence_eval src/final_model src/brand_demo src/visibility tests $(TRAINING_SCRIPTS) $(REVIEW_SCRIPTS)

help: ## Kullanılabilir komutları göster
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "%-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Ortamı kur ve Git kontrollerini etkinleştir
	uv sync --group dev
	uv run pre-commit install

format: ## Kodu biçimlendir
	$(RUFF) check --fix src tests $(TRAINING_SCRIPTS) $(REVIEW_SCRIPTS)
	find $(DATASET_PYTHON_PATHS) -type f -name '*.py' -exec .venv/bin/black --quiet {} \;

check: ## Kod ve iskelet kontrollerini çalıştır
	$(RUFF) check src tests $(TRAINING_SCRIPTS) $(REVIEW_SCRIPTS)
	find $(DATASET_PYTHON_PATHS) -type f -name '*.py' -exec .venv/bin/black --quiet --check {} \;
	$(PYRIGHT) -p pyrightconfig.json
	$(PYTEST) -q

reference-data: ## Sabit sürümdeki İngilizce referans verisini hazırla
	uv run python src/reference_data.py --config configs/base.yaml

turkish-data: ## Yayınlanmış Türkçe veri setini modelleme için indir
	uv run python src/turkish_data.py

modeling-data: reference-data turkish-data ## Her iki korpusu indir ve pair/kanıt tablolarını üret
	uv run python scripts/build_pairs.py

# Offline analysis v1. Never source .env or invoke collection from these targets.
EVIDENCE_ROOT ?= data/processed/evidence_v1
EVIDENCE := PYTHONPATH=src $(PYTHON) -m evidence_eval --root "$(EVIDENCE_ROOT)"
LANGUAGE ?= tr
DOMAIN ?= vpn
M3_ARGS ?=

.PHONY: turkish-data modeling-data modeling-prepare modeling-validate evidence-sample \
	evidence-review-check modeling-baselines modeling-m3-smoke modeling-m3-train brand-report

modeling-prepare: ## Yerel veri kopyalarından sürümlü yeni analiz tablolarını hazırla (offline)
	$(EVIDENCE) prepare

modeling-validate: ## Türkçe 300 hücreyi ve varsa analiz manifestini doğrula (offline)
	$(EVIDENCE) validate

evidence-sample: ## 3 pilot ve 30 inceleme kaydı; boş insan etiketleri (offline)
	$(EVIDENCE) sample

evidence-review-check: ## 30 kaydın insan incelemesi tamamlanmadıysa hata ver
	$(EVIDENCE) review-check --require-complete

modeling-baselines: ## CPU üzerinde sorgu-dışı taban çizgileri ve SHAP üret
	$(EVIDENCE) baselines

.PHONY: evidence-v2-prepare evidence-v2-baselines modeling-generalization modeling-fairness
EVIDENCE_V2_ROOT ?= data/processed/evidence_v2
evidence-v2-prepare: ## Tamamlanmış kaynak taksonomisiyle evidence_v2 tablolarını üret; v1'e dokunmaz (offline)
	PYTHONPATH=src $(PYTHON) -m visibility.evidence_v2 --root "$(EVIDENCE_V2_ROOT)"

evidence-v2-baselines: ## evidence_v2 üzerinde M0–M2 taban çizgileri ve SHAP (CPU, offline)
	PYTHONPATH=src $(PYTHON) -m evidence_eval --root "$(EVIDENCE_V2_ROOT)" baselines

modeling-generalization: ## Prior'suz M2-General, alan-dışı (LODO) test ve sinyal kararlılık matrisi (CPU, offline)
	PYTHONPATH=src $(PYTHON) scripts/run_generalization.py --root "$(EVIDENCE_V2_ROOT)"

modeling-fairness: ## Yoğunlaşma (N_eff) ve adalet tabloları: kim anılıyor, aramadan kim kazanıyor (CPU, offline)
	PYTHONPATH=src $(PYTHON) scripts/run_fairness.py --root "$(EVIDENCE_V2_ROOT)"

.PHONY: app intervention-plan intervention-pilot intervention-run intervention-analyze
app: ## Marka görünürlük arayüzü (Streamlit, offline; uv.lock ve evidence manifestleri değişmez)
	PYTHONPATH=src uv run --with streamlit streamlit run src/visibility/app.py

INTERVENTION := PYTHONPATH=src $(PYTHON) -m visibility.intervention --root "$(EVIDENCE_V2_ROOT)"
INTERVENTION_ARGS ?=
intervention-plan: ## Kontrollü öneri testi: hedefler, kollar ve çağrı sayısı (API çağrısı yok)
	$(INTERVENTION) plan

intervention-pilot: ## 27 Gemini çağrılık pilot ve token ölçümü (ÜCRETLİ; --yes olmadan çağrı yok)
	$(INTERVENTION) pilot $(INTERVENTION_ARGS)

intervention-run: ## Tam kontrollü test, 1.080 Gemini çağrısı; kaldığı yerden devam eder (ÜCRETLİ)
	$(INTERVENTION) run $(INTERVENTION_ARGS)

intervention-analyze: ## Tamamlanan çağrılardan etkileri, güven aralıklarını ve raporu üret (API yok)
	$(INTERVENTION) analyze
	PYTHONPATH=src $(PYTHON) -m visibility.intervention_report --root "$(EVIDENCE_V2_ROOT)"

.PHONY: team-report-pdf
TEAM_REPORT_PDF := output/pdf/AI_Marka_Gorunurlugu_Bulgular.pdf
team-report-pdf: ## Ekip bulgu raporunu PDF'e bas (offline; uv.lock değişmez)
	@mkdir -p $(dir $(TEAM_REPORT_PDF))
	uv run --with weasyprint python -c "from weasyprint import HTML; \
	  HTML('docs/ekip-bulgu-raporu.html').write_pdf('$(TEAM_REPORT_PDF)')"
	@echo "$(TEAM_REPORT_PDF)"

modeling-m3-smoke: ## Açık revision ve çalışan GPU ile küçük listwise eğitim kontrolü
	$(EVIDENCE) m3-smoke $(M3_ARGS)

modeling-m3-train: ## Açık revision ve çalışan GPU ile kontrollü listwise eğitim
	$(EVIDENCE) m3-train $(M3_ARGS)

.PHONY: modeling-en-train modeling-en-status
modeling-en-train: ## İngilizce VPN seed 7: offline smoke, tam eğitim ve checkpoint denetimi
	PYTHONPATH=src $(PYTHON) scripts/run_english_listwise.py --root "$(EVIDENCE_ROOT)"

modeling-en-status: ## İngilizce uzun eğitimin durumu; GPU/API çağrısı yapmaz
	PYTHONPATH=src $(PYTHON) scripts/run_english_listwise.py --root "$(EVIDENCE_ROOT)" --status

.PHONY: final-model-plan final-model-train final-model-status
FINAL_MODELS_ROOT ?= data/processed/final_models_v1
FINAL_MODEL := HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src $(PYTHON) -m final_model
final-model-plan: ## Seçilen 3 nihai deneysel model ve tam veri eğitim kapsamı (offline)
	$(FINAL_MODEL) plan --root "$(EVIDENCE_ROOT)"

final-model-train: ## 3 modeli uygun verinin tamamıyla eğit ve yeniden yükleyerek doğrula (GPU, offline)
	$(FINAL_MODEL) train --root "$(EVIDENCE_ROOT)" --output "$(FINAL_MODELS_ROOT)"

final-model-status: ## Nihai paket ilerlemesi ve tamamlanan ağırlıkların SHA-256 kontrolü
	$(FINAL_MODEL) status --output "$(FINAL_MODELS_ROOT)"

.PHONY: brand-demo brand-demo-offline
brand-demo: ## Marka/sektör sor, maliyet onayından sonra MiniMax + Serper CLI demosu
	PYTHONPATH=src $(PYTHON) -m brand_demo --live

brand-demo-offline: ## Anahtarsız ve ağsız SENTETİK CLI örneği; gerçek ölçüm değildir
	PYTHONPATH=src $(PYTHON) -m brand_demo --offline --brand "Proton VPN" --sector vpn

.PHONY: brand-demo-actions
DEMO_RUN ?=
brand-demo-actions: ## DEMO_RUN klasöründeki mevcut rapora API kullanmadan somut aksiyon planı ekle
	@test -n "$(DEMO_RUN)" || (echo 'DEMO_RUN=data/processed/brand_demo/<id> gerekli'; exit 1)
	PYTHONPATH=src $(PYTHON) -m brand_demo --rebuild-report "$(DEMO_RUN)"

brand-report: ## Marka için kaynaklı Markdown ve JSON rapor oluştur (offline)
	$(EVIDENCE) report --brand "$(BRAND)" --domain "$(DOMAIN)" --language "$(LANGUAGE)"

.PHONY: ai-review-check ai-brand-report
AI_ANNOTATIONS ?= $(EVIDENCE_ROOT)/review/ai/annotations.csv
ai-review-check: ## AI CSV ve provenance kontrolü; insan incelemesini tamamlanmış saymaz (offline)
	PYTHONPATH=src $(PYTHON) scripts/ai_review_report.py --root "$(EVIDENCE_ROOT)" --annotations "$(AI_ANNOTATIONS)" --check

ai-brand-report: ## AI etiketleri ayrı tutulan deneysel marka raporu (offline)
	PYTHONPATH=src $(PYTHON) scripts/ai_review_report.py --root "$(EVIDENCE_ROOT)" --annotations "$(AI_ANNOTATIONS)" --brand "$(BRAND)" --domain "$(DOMAIN)" --language "$(LANGUAGE)"

reference-report: reference-data ## Referans doğrulama notebook'unu baştan sona çalıştır
	uv run --with jupyter --with matplotlib python -m jupyter nbconvert \
		--execute --to notebook --inplace \
		--ExecutePreprocessor.timeout=180 \
		notebooks/S0-4-reference-validation.ipynb

dataset-plan: ## API çağrısı yapmadan 300 hücrelik planı göster
	$(BIAS_EVAL) plan

dataset-preflight: ## Anahtarları ve model katalog bağlantılarını kontrol et
	$(ENV_RUN) $(BIAS_EVAL) preflight

dataset-search-test: ## Bir Serper kredisiyle arama bağlantısını ayrıca test et
	$(ENV_RUN) $(BIAS_EVAL) search-test

dataset-pilot: ## Final plana ait 12 hücrelik pilotu topla
	$(ENV_RUN) $(BIAS_EVAL) run --pilot

dataset-collect: ## Eksik generation hücrelerini kaldığı yerden tamamla
	$(ENV_RUN) $(BIAS_EVAL) run

dataset-repair-generation: ## Geçersiz GLM hücrelerini düşünme kapalıyken yeniden üret
	$(ENV_RUN) $(BIAS_EVAL) run --model glm_5_3_abliteration --disable-abliteration-thinking
	$(BIAS_EVAL) status

dataset-gemini-pilot: ## Yalnız dört final Gemini Flash Lite pilot hücresini çalıştır
	$(ENV_RUN) $(BIAS_EVAL) preflight --model gemini_3_5_flash_lite --skip-judge
	$(ENV_RUN) $(BIAS_EVAL) run --pilot --model gemini_3_5_flash_lite

dataset-retry-gemini: dataset-gemini-pilot ## Pilottan sonra yalnız Gemini Flash Lite hücrelerini tamamla
	$(ENV_RUN) $(BIAS_EVAL) run --model gemini_3_5_flash_lite
	$(BIAS_EVAL) status

dataset-logs: ## Son 200 ayrıntılı pipeline logunu göster
	@tail -n 200 logs/bias-eval.log

dataset-follow-logs: ## Pipeline loglarını canlı takip et (Ctrl-C ile çık)
	@tail -f logs/bias-eval.log

dataset-status: ## Generation ilerlemesini göster
	$(BIAS_EVAL) status

dataset-judge: ## Başarılı generation kayıtlarını Cerebras ile işle
	$(ENV_RUN) $(BIAS_EVAL) judge

dataset-judge-status: ## Güncel, bayat ve eksik judge sonuçlarını göster
	$(BIAS_EVAL) judge-status

dataset-export: ## Referans uyumlu Parquet dosyalarını oluştur
	$(BIAS_EVAL) export

dataset-validate: ## Tam 300 generation + 300 judgment sonucunu zorunlu tut
	$(BIAS_EVAL) validate

dataset-report: ## İngilizce referans ile Türkçe VPN sonuçlarını betimsel karşılaştır
	$(BIAS_EVAL) report

dataset-all: ## Türkçe veri setini tek komutla topla, judge et, dışa aktar ve doğrula
	$(MAKE) --no-print-directory dataset-preflight
	$(MAKE) --no-print-directory dataset-collect
	$(MAKE) --no-print-directory dataset-status
	$(MAKE) --no-print-directory dataset-judge
	$(MAKE) --no-print-directory dataset-judge-status
	$(MAKE) --no-print-directory dataset-export
	$(MAKE) --no-print-directory dataset-validate

dataset-finish: ## Tamamlanan generation sonrası judge, export ve validation çalıştır
	$(MAKE) --no-print-directory dataset-status
	$(MAKE) --no-print-directory dataset-judge
	$(MAKE) --no-print-directory dataset-judge-status
	$(MAKE) --no-print-directory dataset-export
	$(MAKE) --no-print-directory dataset-validate
