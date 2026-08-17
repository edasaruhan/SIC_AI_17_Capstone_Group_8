.DEFAULT_GOAL := help
export UV_CACHE_DIR := .uv-cache

.PHONY: help setup format check reference-data reference-report

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
