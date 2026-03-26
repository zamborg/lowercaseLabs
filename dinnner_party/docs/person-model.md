# Person Model

Each person lives in their own directory under `people/<id>/`.

## Required files

- `person.json`: small structured metadata used by the app and runner.
- `soul.md`: the canonical prompt-like description of the person's preferences and style.

## Current schema

`person.json` contains:

- `id`: stable identifier used in APIs and chat.
- `display_name`: human-readable label.
- `preferred_model`: optional default model hint.
- `calendar_identity`: the calendar principal or lookup key we will eventually use with MCP.
- `calendar_mcp_server`: the MCP server name the person expects for calendar access.
- `role`: short label for the person's place in the dinner-planning simulation.

## Why this shape

- `soul.md` stays unstructured, expressive, and easy to rewrite.
- `person.json` gives the app just enough stable structure for UI and automation.
- The directory gives us a clean place to add future files such as `channels.json`, `contacts.json`, or connector-specific notes.
