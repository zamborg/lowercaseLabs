FROM oven/bun:1 AS base

RUN apt-get update && apt-get install -y \
  tmux \
  curl \
  git \
  nodejs \
  npm \
  sudo \
  && rm -rf /var/lib/apt/lists/*

# Install Claude Code CLI
RUN npm install -g @anthropic-ai/claude-code

# Create non-root user (claude-code refuses --dangerously-skip-permissions as root)
RUN useradd -m -s /bin/bash agent \
  && echo "root ALL=(agent) NOPASSWD: ALL" >> /etc/sudoers

WORKDIR /app
COPY package.json bun.lockb* ./
RUN bun install --frozen-lockfile 2>/dev/null || bun install

COPY . .
RUN chmod +x bin/roundrobin

# Pre-compile channel server into a standalone binary — eliminates Bun JIT on each agent start
RUN bun build --compile src/channel/server.ts --outfile dist/channel

RUN chown -R agent:agent /app

EXPOSE 9090

CMD ["bun", "src/hub/server.ts", "--port", "9090"]
