# AI layer spec

This is the heart.

## Agent responsibilities (MVP)

1. Classify emails into categories (suggested) with confidence.
2. Snooze suggestions: propose an until + reason label.
3. Draft replies from instruction: user gives intent, agent composes email in user style.
4. Notes ingestion: summarize remembered emails with citations; update notes documents.
5. Safety + policy compliance: obey category policies and autonomy levels.

## Non-goals (MVP)

- Fully autonomous sending without approval (unless autonomy=3 and explicitly enabled).
- Calendar integrations (can be stubbed; later plugin).

## Tool interface (internal)

Agent can call:

- get_email(email_id) -> email + thread context
- search_similar_emails(query) -> prior examples for style consistency
- get_policy(category_id) -> policy constraints
- create_draft(email_id, draft_text, metadata)
- update_notes(email_id, summary, tags, citations)
- propose_snooze(email_id, until, reason)

## Autonomy levels

- 0: suggest only (never send)
- 1: draft + tag + snooze suggestions
- 2: draft + apply labels/snooze automatically; still requires send approval
- 3: can auto-send for category if policy allows (still logs + reversible)

## Output format

Every agent job output must be JSON with:

- primary_artifact (draft_text / summary / suggestion)
- rationale_short (1-3 sentences)
- citations (email IDs referenced)
- confidence (0..1)
- policy_checks (pass/fail list)
