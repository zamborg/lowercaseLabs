# Controllr

For Controllr Voice or actuator work, use the project-local [controllr-shortcuts skill](.agents/skills/controllr-shortcuts/SKILL.md).

## Provider parity

- Keep OpenAI, Anthropic, and OpenRouter in sync when changing conversations, tool calls, or the agent loop. Put shared behavior in ControllrCore, provider-specific API details in the adapters, and voice playback and dictation in the shortcut.
- Preserve the same behavior for ordinary replies, mixed text and tool calls, matching tool results, follow-up turns, and action confirmations. Keep saved threads resumable across provider or model changes.
- Keep each provider's API key in Keychain and its selected model in a separate setting. The active provider and model come from Setup.
- Update shared evals and the affected adapter tests with changes to the agent contract. Use fixtures for all three providers and record live checks separately when credentials are available.
