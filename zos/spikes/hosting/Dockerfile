FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.2 /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
COPY src ./src
COPY spikes/hosting ./spikes/hosting

RUN uv sync --frozen --no-dev

EXPOSE 8080
CMD ["uv", "run", "--no-sync", "python", "spikes/hosting/server.py"]
