# 02: App Chart Spec

This document tells coding agents exactly how to structure an app repository for ingestion.

## Required Files

```text
<repo-root>/
  Dockerfile
  fly.toml
  README.md
  app/ or src/
```

## Dockerfile Rules

- Must produce a single runnable container.
- Must expose one HTTP port.
- Must run without interactive prompts.
- Must not require build-time secrets.

## fly.toml Rules

Minimal expected structure:

```toml
app = "replace-with-fly-app-name"
primary_region = "sjc"

[metadata]
networking_mode = "proxy" # or "direct"

[http_service]
internal_port = 8080
force_https = true
```

Alternative service blocks are allowed, but `internal_port` must be discoverable.

## Parser Behavior (Current)

The platform parser currently extracts:

- `Dockerfile` presence
- `fly.toml` presence
- `app` name
- `networking_mode` from metadata
- `internal_port`
- volume requirement via `mounts`

## Ingestion Endpoints

### Parse only

`POST /apps/charts/parse`

Body:

```json
{
  "repo_path": "/abs/path/to/repo"
}
```

or

```json
{
  "repo_url": "https://github.com/org/repo",
  "ref": "main"
}
```

### Parse + register app

`POST /apps/from-chart`

Body:

```json
{
  "slug": "echo-web",
  "name": "Echo Web",
  "docker_image": "registry.example.com/echo-web:latest",
  "repo_url": "https://github.com/org/repo",
  "ref": "main",
  "fly_app_name": "optional-override"
}
```

## Authoring Guidance for Agents

- Keep file paths conventional and shallow.
- Keep startup command explicit and deterministic.
- Include README with run/build/publish instructions.
- Avoid dynamic runtime that requires hidden steps.
