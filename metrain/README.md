# iMessage Export for Fine-Tuning

This folder holds a small utility that pulls messages from your macOS `chat.db`
(typically `~/Library/Messages/chat.db`) and generates a JSON Lines file in the
chat-style format OpenAI expects for fine-tuning (each example now has a
`"messages"` array).

Use cases:

- Build a dataset of **just your messages** (default) so that other people’s replies
  stay private.
- Flip on `--include-context` if you want to pair each of your replies with the
  most recent incoming message (or messages, via `--context-depth`) as the
  preceding context.

## Usage

```bash
python3 imessage_exporter.py [path/to/chat.db]
```

### Common flags

- `-o, --output`: destination JSONL file (default `imessage_export.jsonl`).
- `--include-context`: prepend the most recent non-you reply to each example.
- `--context-depth N`: when context is enabled, include up to `N` previous replies.
- `--limit N`: stop after exporting `N` of your messages (handy for quick checks).

## What the export looks like

Each line is a JSON object formatted for chat fine-tuning:

- `messages`: an array of `{ "role": ..., "content": ... }` dictionaries.
  Context (other people’s replies) is emitted with `role` set to `user` and the
  handle name prefixed in the content. Your own message is always emitted as
  the final entry with `role` set to `assistant`.

Example line (with context):

```json
{
  "messages": [
    {
      "role": "user",
      "content": "alex@example.com: Sure, bring the drafts tonight."
    },
    {
      "role": "assistant",
      "content": "Got it, I’ll be there by 7."
    }
  ]
}
```

## Next steps

1. Inspect the generated `imessage_export.jsonl` to confirm messages and context look sensible.
2. Upload it to OpenAI via the CLI / web UI, e.g.
   `openai api fine_tunes.create -t imessage_export.jsonl -m gpt-4o-mini`.
3. Share the resulting job ID with any systems that need to interact with your custom model.

If you don’t want to run a fine-tune programmatically, go to https://platform.openai.com/fine-tuning
and upload the generated JSONL under “Create file” before creating a fine-tune job manually.
