.PHONY: dev build clean

IMAGE := roundrobin
CONTROL_PORT ?= 8080

# Build docker image + start local control plane
dev: build
	CONTROL_PORT=$(CONTROL_PORT) bun src/control/server.ts

# Build the agent docker image
build:
	bun install --frozen-lockfile 2>/dev/null || bun install
	docker build -t $(IMAGE) .

# Delete run artifacts (workspaces, chat DBs, compiled dist) — keeps tasks/souls/prompts
clean:
	rm -rf runs/ dist/
	@echo "Cleaned run artifacts."
