FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.2 /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
COPY migrations ./migrations
COPY src ./src

RUN uv sync --frozen --no-dev

ENV HOST=0.0.0.0
ENV PORT=8080
ENV ZOS_MIGRATIONS_DIR=/app/migrations

EXPOSE 8080

CMD ["sh", "-c", "uv run --no-sync zos migrate && exec uv run --no-sync python -m zos.app"]
