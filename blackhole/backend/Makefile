.PHONY: install dev test migrate

install:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -e ".[dev]"

dev:
	uvicorn app.main:app --reload --port 8080

test:
	pytest

migrate:
	alembic upgrade head

