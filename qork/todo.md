## Refactor and Extensibility Notes

- zubin make sure that this is refactored in ways that is more extensible

Details and ideas:

- Core API layering
  - Extract a thin core client module (transport + adapters) decoupled from CLI and notebook UX.
  - Unify interfaces for Chat Completions vs Responses API via a small adapter layer.
  - Centralize model/provider configuration (env, defaults, overrides) in one place.

- Output/printing abstraction
  - Consolidate printing into a `Printer` interface (rich/plain), selectable via flag or parameter.
  - Add a Markdown-to-plain converter that preserves code blocks for better copy/paste in plaintext mode.
  - Consider a JSON output option for tooling (`--json` or `format="json"`).

- Public Python API
  - Keep `qork.ask()` minimal; move advanced options into typed dataclasses or kwargs object for future growth.
  - Add sync/async variants (e.g., `ask_async`) for asyncio notebooks/servers.
  - Provide result objects with `.text`, `.usage`, `.cost` to avoid reaching into response internals.

- Streaming
  - Normalize streaming chunk handling; push chunk parsing into a helper that tolerates provider differences.
  - Optional callbacks/hooks: `on_token`, `on_error`, `on_finish` for UI integrations.

- Error handling
  - Standardize exceptions (e.g., `QorkError`, `ProviderError`) and map provider errors to them.
  - Provide `--verbose`/`debug` levels and structured logs when `QORK_DEBUG=1`.

- Configuration
  - Add `pyproject.toml` config section or `qork.toml` for local defaults.
  - Support per-project `.env` discovery and a `qork config` subcommand.

- Packaging/CLI
  - Split CLI (`qork/main.py`) into subcommands if features grow (`qork chat`, `qork resp`, `qork run`).
  - Keep CLI thin; delegate all logic to core modules.

- Testing
  - Add unit tests for `ask()` with mocked providers; snapshot streaming behavior.
  - Integration test matrix for plaintext vs rich, streaming vs non-streaming, Responses vs Completions.

- Docs
  - Update README with Python API examples and flags.
  - Add examples notebook demonstrating streaming, debug, and Responses API.
