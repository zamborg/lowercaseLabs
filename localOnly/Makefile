.PHONY: dev-up dev-down dev-logs migrate web-harness

dev-up:
	docker compose up -d postgres redis
	cd backend && uv sync && uv run db-upgrade

dev-down:
	docker compose down

dev-logs:
	docker compose logs -f postgres redis

migrate:
	cd backend && uv run db-upgrade

web-harness:
	cd web-harness && python3 -m http.server 4173
