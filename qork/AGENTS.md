# qork Agent Documentation

This document describes the current architecture, behavior, and extension points of the `qork` CLI agent.

## Overview (current state)

- Backend: OpenAI Responses API (non‑streaming by default; optional streaming)
- Optional global thread mode: stores only a single `previous_response_id`
- Clean terminal output: plaintext or styled with Rich

## How it Works

The agent is built using the following key components:

-   **`openai`**: Used for the Responses API pathway.
-   **`rich`**: For readable output panels and markdown rendering.
-   **`typer`**: Command-line parsing for flags (`--stream`, `--model`, `--plaintext`, `--debug`).

The core logic is contained within the `qork` package:

-   **`main.py`**: CLI entry point. Responses API only. Handles streaming/plaintext/debug rendering.
-   **`config.py`**: Reads `OPENAI_API_KEY` and `QORK_MODEL`.
-   **`utils.py`**: Responses API helpers (`get_response`, `stream_response`, `response_text`).
-   **`session.py`**: Minimal global thread storage of only `previous_response_id` at `~/.qork/history/session.id` (clobbers across shells).
-   **`ask.py`**: Python API mirroring CLI behavior.

## System Prompt

To ensure concise and accurate answers suitable for a developer audience, the following system prompt is sent with every user request:

> You are a commandline assistant. The user is a sophisticated developer looking for a FAST and ACCURATE answer to their question. You should be concise and to the point. Prioritize answers, and explanations ONLY when requested.

## Dependencies
- `openai`
- `rich`
- `typer`

## Configuration

The agent can be configured using environment variables:

-   **`OPENAI_API_KEY`**: Your API key (required).
-   **`QORK_MODEL`**: Default model (optional).
    - Note: CLI `--profile` overrides this.

## Usage

CLI examples:

```bash
qork "Default Responses API path"
qork -m gpt-5-mini "Responses with explicit model"
qork --stream "Streaming output"
qork --profile nano "Preset model"
qork --profile high "High reasoning effort"
qork -pt "Plaintext output"
qork -d "Debug info when available"
```

For more details on flags, session behavior, and Python API usage, see `README.md`.
