SHELL := /bin/bash

APP ?= thevoid-local
HOST ?= 127.0.0.1
PORT ?= 8080
REGION ?= sjc
VOLUME ?= thevoid_data
VOLUME_SIZE_GB ?= 10
DB_APP ?= $(APP)-db
API_BASE_URL ?= http://127.0.0.1:8080
IDENTITY_TOKEN ?= dev-ingest
DURATION_SECONDS ?= 120
LOCAL_DATE ?= $(shell date +%F)
AUDIO ?=
MSG ?=
COMPOSE_FILE ?= ../docker-compose.yml
COMPOSE := docker compose -f $(COMPOSE_FILE)

VENV_DIR ?= .venv
PYTHON ?= python3

.PHONY: help venv deps api worker stack stack-lan test migrate migration-new ingest compose-build compose-up compose-down compose-logs compose-ps compose-migrate fly-bootstrap fly-mpg-create fly-mpg-attach fly-deploy fly-status fly-logs fly-health fly-secrets

help:
	@echo "backend make targets"
	@echo "  make deps                      # create .venv + install deps"
	@echo "  make api HOST=127.0.0.1 PORT=8080"
	@echo "  make worker"
	@echo "  make stack PORT=8080            # dockerized local stack"
	@echo "  make stack-lan PORT=8080"
	@echo "  make test"
	@echo "  make migrate                   # alembic upgrade head"
	@echo "  make migration-new MSG='add xyz'"
	@echo "  make ingest AUDIO=/abs/path/file.m4a [IDENTITY_TOKEN=dev-user]"
	@echo "  make compose-build"
	@echo "  make compose-up                # db + migrations + api + worker"
	@echo "  make compose-down"
	@echo "  make compose-logs"
	@echo "  make compose-ps"
	@echo "  make compose-migrate"
	@echo "  make fly-bootstrap APP=thevoid-local REGION=sjc VOLUME=thevoid_data VOLUME_SIZE_GB=10"
	@echo "  make fly-mpg-create APP=thevoid-local DB_APP=thevoid-local-db REGION=sjc"
	@echo "  make fly-mpg-attach APP=thevoid-local DB_APP=thevoid-local-db"
	@echo "  make fly-deploy APP=thevoid-local"
	@echo "  make fly-status APP=thevoid-local"
	@echo "  make fly-logs APP=thevoid-local"
	@echo "  make fly-health APP=thevoid-local"
	@echo "  make fly-secrets APP=thevoid-local JWT_SECRET=... APPLE_ALLOWED_AUDIENCES=... OPENAI_API_KEY=... ADMIN_USERNAME=... ADMIN_PASSWORD=..."

venv:
	@if [[ ! -d "$(VENV_DIR)" ]]; then $(PYTHON) -m venv "$(VENV_DIR)"; fi

deps: venv
	@source "$(VENV_DIR)/bin/activate" && pip install -r requirements.txt

api:
	@./scripts/dev_api.sh "$(HOST)" "$(PORT)"

worker:
	@./scripts/dev_worker.sh

stack:
	@COMPOSE_FILE="$(COMPOSE_FILE)" API_PORT="$(PORT)" ./scripts/dev_stack.sh "$(HOST)" "$(PORT)"

stack-lan:
	@$(MAKE) stack HOST=0.0.0.0 PORT="$(PORT)"

test: deps
	@source "$(VENV_DIR)/bin/activate" && pytest -q

migrate:
	@./scripts/migrate.sh

migration-new:
	@if [[ -z "$(MSG)" ]]; then \
		echo "MSG is required. Example: make migration-new MSG='add table xyz'"; \
		exit 1; \
	fi
	@./scripts/migration_new.sh "$(MSG)"

ingest:
	@if [[ -z "$(AUDIO)" ]]; then \
		echo "AUDIO is required. Example: make ingest AUDIO=/abs/path/reflection.m4a"; \
		exit 1; \
	fi
	@./scripts/ingest_reflection.py \
		--api-base-url "$(API_BASE_URL)" \
		--identity-token "$(IDENTITY_TOKEN)" \
		--audio "$(AUDIO)" \
		--duration-seconds "$(DURATION_SECONDS)" \
		--local-date "$(LOCAL_DATE)"

compose-build:
	@$(COMPOSE) build api worker migrate

compose-up: compose-build
	@$(COMPOSE) up -d db
	@$(COMPOSE) run --rm migrate
	@$(COMPOSE) up -d api worker

compose-down:
	@$(COMPOSE) down

compose-logs:
	@$(COMPOSE) logs -f --tail=200 db api worker

compose-ps:
	@$(COMPOSE) ps

compose-migrate:
	@$(COMPOSE) run --rm migrate

fly-bootstrap:
	@./scripts/fly_bootstrap.sh "$(APP)" "$(REGION)" "$(VOLUME)" "$(VOLUME_SIZE_GB)"

fly-mpg-create:
	@fly mpg create --name "$(DB_APP)" --region "$(REGION)"

fly-mpg-attach:
	@fly mpg attach "$(DB_APP)" -a "$(APP)"

fly-deploy:
	@./scripts/fly_deploy.sh "$(APP)"

fly-status:
	@fly status -a "$(APP)"

fly-logs:
	@fly logs -a "$(APP)"

fly-health:
	@curl -fsSL "https://$(APP).fly.dev/health" && echo

fly-secrets:
	@: "$${JWT_SECRET:?Set JWT_SECRET=...}"
	@: "$${APPLE_ALLOWED_AUDIENCES:?Set APPLE_ALLOWED_AUDIENCES=...}"
	@: "$${OPENAI_API_KEY:?Set OPENAI_API_KEY=...}"
	@: "$${ADMIN_USERNAME:?Set ADMIN_USERNAME=...}"
	@: "$${ADMIN_PASSWORD:?Set ADMIN_PASSWORD=...}"
	@fly secrets set -a "$(APP)" \
		JWT_SECRET="$$JWT_SECRET" \
		APPLE_ALLOWED_AUDIENCES="$$APPLE_ALLOWED_AUDIENCES" \
		OPENAI_API_KEY="$$OPENAI_API_KEY" \
		ADMIN_USERNAME="$$ADMIN_USERNAME" \
		ADMIN_PASSWORD="$$ADMIN_PASSWORD"
