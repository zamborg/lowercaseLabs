# qork Agent Documentation

This document describes the current architecture, behavior, and extension points of the `qork` CLI agent.

## Overview (current state)

- Default backend: OpenAI Responses API (non‑streaming)
- Optional backend: Chat Completions (streaming by default)
- Per‑shell session memory: stores only a single `previous_response_id` used by the Responses path
- Clean terminal output: plaintext or styled with Rich

## How it Works

The agent is built using the following key components:

-   **`openai`**: Used for the Responses API pathway.
-   **`litellm`**: Used for Chat Completions with a unified interface.
-   **`rich`**: For readable output panels and markdown rendering.
-   **`argparse`**: Command-line parsing for flags (`--chat`, `--no-stream`, `--model`, `--plaintext`, `--debug`).

The core logic is contained within the `qork` package:

-   **`main.py`**: CLI entry point and router. Default to Responses; `--chat` selects Chat. Handles streaming/plaintext/debug rendering.
-   **`config.py`**: Reads `OPENAI_API_KEY` and `QORK_MODEL`.
-   **`utils.py`**: `get_response` (Responses API with optional `previous_response_id`), `get_completion` (Chat), `get_token_count` (debug).
-   **`session.py`**: Minimal per-shell session storage of only `previous_response_id`, keyed by a session key derived from TTY (fallback PPID).
-   **`ask.py`**: Python API mirroring CLI behavior.
-   **`models.py`**: `TokenCount` for streaming token estimates.
-   **`printer.py`**: Printer abstractions (Rich/plain); the CLI often prints directly.

## System Prompt

To ensure concise and accurate answers suitable for a developer audience, the following system prompt is sent with every user request:

> You are a commandline assistant. The user is a sophisticated developer looking for a FAST and ACCURATE answer to their question. You should be concise and to the point. Prioritize answers, and explanations ONLY when requested.

## Dependencies
- `openai`
- `litellm`
- `rich`
- `tiktoken`

## Configuration

The agent can be configured using environment variables:

-   **`OPENAI_API_KEY`**: Your API key (required).
-   **`QORK_MODEL`**: Default model for Chat Completions (optional).

## Usage

CLI examples:

```bash
qork "Default Responses API path"
qork -m gpt-5-mini "Responses with explicit model"
qork --chat "Chat Completions (streaming)"
qork --chat --no-stream "Chat Completions (non‑streaming)"
qork -pt "Plaintext output"
qork -d "Debug info when available"
```

For more details on flags, session behavior, and Python API usage, see `README.md`.
