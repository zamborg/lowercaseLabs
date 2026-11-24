# ZOS — Architectural & Implementation Decisions

Log any choice not settled by `PLAN.md`, `SCHEMA.md`, or `BUILD.md`, plus spike findings and any deviations.

Format: `## YYYY-MM-DD — <Short title>` with **Decision**, **Why**, and **Impact**.

---

## 2026-11-23 — Python package and dependency baseline (Task 0.1)

- **Decision:** Pin `requires-python = ">=3.12,<3.13"` with `starlette` + `uvicorn[standard]` as the ASGI stack, `psycopg[binary,pool]` for Postgres 3 async pooling, `argon2-cffi` for owner passphrase hashing, and `hatchling` as build backend.
- **Why:** Matches `BUILD.md` stack (Python 3.12, Starlette/FastAPI, official `mcp` SDK, psycopg 3, Typer, cryptography, httpx). Starlette avoids redundant FastAPI overhead since the MCP Streamable HTTP transport and minimal HTML forms mount cleanly as Starlette ASGI routes.
- **Impact:** Both `zos` and `zos-actuator` target Python 3.12 via `uv`.
