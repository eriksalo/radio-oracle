.PHONY: install lint test run run-voice clean

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

install:
	python3 -m venv $(VENV)
	$(PIP) install -e ".[dev]"

lint:
	$(VENV)/bin/ruff check oracle/ tests/ config/
	$(VENV)/bin/ruff format --check oracle/ tests/ config/

format:
	$(VENV)/bin/ruff check --fix oracle/ tests/ config/
	$(VENV)/bin/ruff format oracle/ tests/ config/

test:
	$(PYTHON) -m pytest

run:
	$(PYTHON) -m oracle

run-voice:
	$(PYTHON) -m oracle --mode voice

clean:
	rm -rf $(VENV) *.egg-info dist build
