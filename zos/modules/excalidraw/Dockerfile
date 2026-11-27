FROM node:22-bookworm-slim AS editor
WORKDIR /build/editor
COPY editor/package*.json ./
RUN npm ci --no-audit --no-fund
COPY editor/ ./
RUN npm run build

FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.9.2 /uv /bin/
WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
COPY src ./src
COPY --from=editor /build/src/zos_excalidraw/static ./src/zos_excalidraw/static
RUN uv sync --frozen --no-dev
ENV MCP_PORT=8081 MCP_PATH=/mcp UI_PORT=8082
EXPOSE 8081 8082
CMD ["uv", "run", "--no-sync", "python", "-m", "zos_excalidraw.main"]
