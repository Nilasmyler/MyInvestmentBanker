PYTHON ?= python3
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip
VENV_UVICORN := $(VENV)/bin/uvicorn

.PHONY: setup check-venv run verify tunnel

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PIP) install -r requirements.txt

check-venv:
	@test -x $(VENV_PYTHON) || (echo "Missing $(VENV_PYTHON). Run 'make setup' first." && exit 1)

run: check-venv
	$(VENV_UVICORN) main:app --reload --port 8000

verify: check-venv
	$(VENV_PYTHON) test_pipeline.py

tunnel:
	npx ngrok http 8000
