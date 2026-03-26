FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ripgrep \
    && python -m pip install --no-cache-dir pytest==8.4.1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY project/ /app/

ENV PYTHONUNBUFFERED=1
