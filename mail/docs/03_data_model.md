# Data model

Use Postgres. Store raw email safely + normalized metadata.

# Tables (minimum)

## users

- id (uuid), email_address, created_at

## accounts

- id, user_id, provider (imap), imap_host, imap_user, imap_encrypted_pass, smtp_host, smtp_user, smtp_encrypted_pass

## emails

- id (uuid)
- account_id
- message_id (RFC)
- thread_key (string; provider-specific)
- from, to, cc, bcc (json)
- subject
- date
- snippet
- body_text (nullable; store after fetch)
- body_html (nullable)
- labels (json)
- is_read (bool)
- ingested_at

## triage_events (append-only audit)

- id
- user_id
- email_id
- action (archive|delete|snooze|remember|delegate_reply|break_glass_reply|mark_read|label)
- payload (json: snooze_until, category_id, etc.)
- created_at

## categories

- id, user_id, name, description
- matching_rules (json)  // e.g., from domain, keywords, thread heuristics
- default_policy_id (nullable)

## policies

- id, user_id, name
- autonomy_level (0..3)
- allowed_actions (json)
- reply_template (text)
- style_profile (json: tone sliders, signatures)
- escalation_rules (json)

## agent_jobs

- id, user_id, type (summarize|classify|draft_reply|notes_update|snooze_suggest)
- status (queued|running|succeeded|failed)
- input_refs (json: email_ids, category_id)
- output (json)
- error (text)
- created_at, updated_at

## notes_documents

- id, user_id, title
- content_markdown
- updated_at

## notes_items

- id
- notes_document_id
- email_id
- summary
- tags (json)
- citations (json: {email_id, message_id, snippet_offsets?})
- created_at
