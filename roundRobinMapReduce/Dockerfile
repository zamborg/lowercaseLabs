FROM node:22-bookworm-slim

ENV NPM_CONFIG_UPDATE_NOTIFIER=false \
    NPM_CONFIG_FUND=false

RUN apt-get update && apt-get install -y \
  bash \
  ca-certificates \
  coreutils \
  findutils \
  git \
  python3 \
  procps \
  tmux \
  && rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash claude

RUN npm install -g \
  @anthropic-ai/claude-code \
  @google/gemini-cli \
  @openai/codex

WORKDIR /app
COPY package.json ./
COPY src ./src

EXPOSE 3000

CMD ["node", "src/runtime.js", "/run/runspec.json"]
