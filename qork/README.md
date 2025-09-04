# qork

A simple, beautiful CLI for asking LLMs questions from your terminal. Fast defaults, clean output, and per-shell conversational context.

## Highlights
- Default backend: OpenAI Responses API (non‑streaming), zero config beyond your API key
- Optional Chat Completions path with streaming
- Per‑shell state: remembers the last conversation id only for the current shell session
- Plaintext or pretty Rich output

## Install
```bash
pip install qork
```

## Prerequisites
- Environment: `OPENAI_API_KEY` must be set
- Models: you can pass `-m/--model` at call time; otherwise defaults apply (see below)

## Quick start
- Default (Responses API):
```bash
qork "Say hello in one short sentence."
```
- Responses API with explicit model:
```bash
qork -m gpt-5-mini "Give me a five-word poem."
```
- Chat Completions (streaming by default):
```bash
qork --chat "List 3 colors."
```
- Chat non‑streaming:
```bash
qork --chat --no-stream "Summarize: qork is..."
```
- Plaintext output (easier to copy/paste):
```bash
qork -pt "Plain output please."
qork --chat -pt "Also works for chat."
```
- Debug info (tokens/cost where available):
```bash
qork -d "How many seconds in a day?"
qork --chat -d --no-stream "Explain Big-O of binary search."
```

## Backends and defaults
- Responses API (default)
  - Selected unless you pass `--chat`
  - Non‑streaming
  - Default model if not specified: `gpt-5-mini`
- Chat Completions (`--chat`)
  - Streaming by default; use `--no-stream` to disable
  - Model comes from `-m/--model` or from `QORK_MODEL` env var, else falls back to library default

## Per‑shell session behavior (simple and automatic)
qork keeps only a single `previous_response_id` per shell session to thread your Responses API calls.
- A shell session is identified by TTY when available, otherwise by parent PID
- State file lives at: `~/.qork/sessions/<session_key>.json`
- File format:
```json
{
  "previous_response_id": "resp_abc123",
  "updated_at": "2025-09-04T12:34:56Z"
}
```
- New shell → new session file → no prior context
- Each `qork -r` (default path) uses the stored id (if any) and overwrites it with the latest id
- To reset: simply open a new shell, or delete the file in `~/.qork/sessions/`

Note: Session state is only used for the Responses API path. The Chat Completions path does not maintain threads.

## CLI flags
- `-m, --model`         Set model name
- `-r, --responses`     Use Responses API (default)
- `--chat`              Use Chat Completions backend
- `-ns, --no-stream`    Disable streaming (chat only)
- `-pt, --plaintext`    Plain stdout (no rich panels/markdown)
- `-d, --debug`         Show token usage/cost when available

## Python API
Call from notebooks and scripts using the same behavior as the CLI.
```python
from qork.ask import ask

# Default pathway (Responses API)
text = ask("One short sentence.", responses=True, plaintext=True, return_text=True)

# Chat Completions (non-streaming)
text = ask("Explain merge sort in one paragraph.", stream=False, responses=False, return_text=True)

# Chat Completions (streaming)
text = ask("Print three facts.", stream=True, responses=False, return_text=True)
```

Parameters you’ll likely use:
- `prompt: str` (required)
- `model: Optional[str]`
- `responses: bool` (True for Responses API; False for Chat)
- `stream: Optional[bool]` (Chat only; default True)
- `plaintext: bool` (stdout formatting)
- `debug: bool` (token/cost info)
- `return_text: bool` (return text value in addition to printing)

## Examples
- Continue a short thread in the same shell (Responses API is default):
```bash
qork "Start a thread in one sentence."
qork "Continue in one sentence."
```
- Switch to Chat Completions with streaming:
```bash
qork --chat "Stream five words only."
```

## Tests (end‑to‑end)
These tests hit live APIs (no mocks). Set your key first.
```bash
export OPENAI_API_KEY=sk-...
pytest -q tests/test_e2e_cli.py::test_cli_responses_session_persistence
pytest -q tests/test_e2e_cli.py::test_cli_chat_non_stream_plaintext
pytest -q tests/test_e2e_cli.py::test_cli_chat_stream_plaintext
pytest -q tests/test_e2e_python_api.py
```
You can select a model for tests with `QORK_E2E_MODEL` or rely on defaults.

## Troubleshooting
- “API key not set”: ensure `OPENAI_API_KEY` is exported in your shell
- No session carry‑over: you likely opened a new shell (that’s expected); check `~/.qork/sessions/`
- Switch backends: `--chat` for Chat Completions, Responses is the default

---
Designed for fast, accurate answers from the terminal with minimal ceremony. 