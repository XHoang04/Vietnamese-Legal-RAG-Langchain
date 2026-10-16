.PHONY: install lint test test-model inspect data chunks embed index golden review calibrate ablation cleanup api ui up down

PYTHON ?= python

install:
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check src tests scripts

test:
	$(PYTHON) -m pytest -m "not slow and not model"

test-model:
	$(PYTHON) -m pytest -m "model"

inspect:
	$(PYTHON) scripts/00_inspect_dataset.py

data:
	$(PYTHON) scripts/01_download_filter.py

chunks:
	$(PYTHON) scripts/02_build_chunks.py

embed:
	$(PYTHON) scripts/03_embed_chunks.py

index:
	$(PYTHON) scripts/04_build_index.py

golden:
	$(PYTHON) scripts/05_make_golden_set.py

review:
	$(PYTHON) scripts/06_review_golden_set.py

calibrate:
	$(PYTHON) scripts/07_calibrate_thresholds.py

ablation:
	$(PYTHON) scripts/08_run_ablation.py

cleanup:
	$(PYTHON) scripts/09_cleanup_uploads.py

api:
	$(PYTHON) -m uvicorn apps.api.main:app --reload --port 8000

ui:
	$(PYTHON) -m streamlit run apps/ui/app.py

up:
	docker compose up -d

down:
	docker compose down
