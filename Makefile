.DEFAULT_GOAL := help
export UV_CACHE_DIR := .uv-cache

.PHONY: help setup format check reference-data reference-report \
        collect-plan collect-run collect-status collect-monitor require-run-id

help: ## Kullanılabilir komutları göster
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "%-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Ortamı kur ve Git kontrollerini etkinleştir
	uv sync --group dev
	uv run pre-commit install

format: ## Kodu biçimlendir
	uv run ruff check --fix src tests
	find src tests -type f -name '*.py' -exec .venv/bin/black --quiet {} \;

check: ## Kod ve iskelet kontrollerini çalıştır
	uv run ruff check src tests
	find src tests -type f -name '*.py' -exec .venv/bin/black --quiet --check {} \;
	uv run pyright -p pyrightconfig.json
	uv run pytest -q

reference-data: ## Sabit sürümdeki İngilizce referans verisini hazırla
	uv run python src/reference_data.py --config configs/base.yaml

reference-report: reference-data ## Referans doğrulama notebook'unu baştan sona çalıştır
	uv run --with jupyter --with matplotlib python -m jupyter nbconvert \
		--execute --to notebook --inplace \
		--ExecutePreprocessor.timeout=180 \
		notebooks/S0-4-reference-validation.ipynb

# --- Toplama boru hattı (S1-6, S1-7, S2-3) ---------------------------------
# RUN_ID zorunludur; aynı RUN_ID ile tekrar çalıştırmak kaldığı yerden sürdürür.
# Ek argümanlar ARGS ile geçilir, örn: make collect-run RUN_ID=pilot-01 ARGS="--limit 20"

# .env varsa API anahtarlarını ortama yükle; yoksa uv hata vermesin diye atla.
ENV_FILE := $(if $(wildcard .env),--env-file .env,)
COLLECT := PYTHONPATH=src uv run $(ENV_FILE) python -m collect

require-run-id:
	@test -n "$(RUN_ID)" || { echo "RUN_ID gerekli, örn: make $(MAKECMDGOALS) RUN_ID=pilot-01"; exit 2; }

collect-plan: require-run-id ## Toplama planını göster; hiçbir çağrı yapmaz
	$(COLLECT) plan --run-id $(RUN_ID) $(ARGS)

collect-run: require-run-id ## Toplamayı başlat veya kaldığı yerden sürdür
	$(COLLECT) run --run-id $(RUN_ID) $(ARGS)

collect-status: require-run-id ## Koşunun ne kadarının toplandığını göster
	$(COLLECT) status --run-id $(RUN_ID) $(ARGS)

collect-monitor: require-run-id ## Günlük izleme tablosu ve tek satır özet
	$(COLLECT) monitor --run-id $(RUN_ID) $(ARGS)
