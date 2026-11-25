FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.2 /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
COPY src ./src

RUN uv sync --frozen --no-dev

ENV MCP_PORT=8081
ENV MCP_PATH=/mcp
ENV UI_PORT=8082

EXPOSE 8081 8082

CMD ["uv", "run", "--no-sync", "python", "-m", "zos_inbox.main"]
