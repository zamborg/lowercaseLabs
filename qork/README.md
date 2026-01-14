# qork

A simple, beautiful CLI for asking LLMs questions from your terminal. Fast defaults, clean output, and optional conversational threading.

## Highlights
- Backend: OpenAI Responses API
- Optional streaming output (`--stream`)
- Optional global thread mode (`--thread`) that stores a single `previous_response_id`
- Plaintext or pretty Rich output

## Install
```bash
pip install qork
```

## Prerequisites
- Environment: `OPENAI_API_KEY` must be set
- Models: you can pass `-m/--model` at call time; otherwise defaults apply (see below)

## Quick start
- Default (Responses API, non-streaming):
```bash
qork "Say hello in one short sentence."
```
- Responses API with explicit model:
```bash
qork -m gpt-5-mini "Give me a five-word poem."
```
- Presets:
```bash
qork --profile nano "Explain this in one sentence."
qork --profile high "Think carefully and give the best answer."
```
- Streaming:
```bash
qork --stream "List 3 colors."
```
- Plaintext output (easier to copy/paste):
```bash
qork -pt "Plain output please."
```
- Debug info (tokens/cost where available):
```bash
qork -d "How many seconds in a day?"
```

## Backends and defaults
- Responses API (only backend)
  - Non‑streaming by default; use `--stream` to enable streaming
  - Default model if not specified: `QORK_MODEL` or `gpt-5-mini`

## Thread mode (simple and explicit)
qork can optionally reuse a single global `previous_response_id` to continue one thread across invocations.
- Enable with `-t/--thread`
- State file lives at: `~/.qork/history/session.id`
- This is global and will clobber across shells (intentionally simple)
- To reset: delete `~/.qork/history/session.id`

## CLI flags
- `-m, --model`         Set model name
- `--profile`           Preset: `nano|mini|large|high` (high sets `reasoning.effort=high`)
- `--stream/--no-stream` Stream output tokens
- `-t, --thread`        Continue a single global thread (stores `previous_response_id`)
- `-pt, --plaintext`    Plain stdout (no rich panels/markdown)
- `-d, --debug`         Show token usage/cost when available

## Python API
Call from notebooks and scripts using the same behavior as the CLI.
```python
from qork.ask import ask

# Responses API (non-streaming)
text = ask("One short sentence.", stream=False, plaintext=True, return_text=True)

# Responses API (streaming)
text = ask("Print three facts.", stream=True, plaintext=True, return_text=True)
```

Parameters you’ll likely use:
- `prompt: str` (required)
- `model: Optional[str]`
- `stream: bool`
- `plaintext: bool` (stdout formatting)
- `debug: bool` (token/cost info)
- `previous_response_id: Optional[str]` (continue a thread)
- `return_text: bool` (return text value in addition to printing)

## Examples
- Continue a single global thread:
```bash
qork -t "Start a thread in one sentence."
qork -t "Continue in one sentence."
```

## Tests (end‑to‑end)
These tests hit live APIs (no mocks). Set your key first.
```bash
export OPENAI_API_KEY=sk-...
pytest -q tests/test_e2e_cli.py::test_cli_responses_session_persistence
pytest -q tests/test_e2e_cli.py::test_cli_plaintext_non_stream
pytest -q tests/test_e2e_cli.py::test_cli_plaintext_stream
pytest -q tests/test_e2e_python_api.py
```
You can select a model for tests with `QORK_E2E_MODEL` or rely on defaults.

## Troubleshooting
- “API key not set”: ensure `OPENAI_API_KEY` is exported in your shell
- No thread carry‑over: ensure you used `-t/--thread`; check `~/.qork/history/session.id` (delete it to reset)
- Streaming: use `--stream` if you prefer incremental output

---
Designed for fast, accurate answers from the terminal with minimal ceremony. 
