# REST API (simple)

All endpoints under /v1.

## Auth

- Dev: Authorization: Bearer <DEV_TOKEN>
- Later: OAuth/Gmail, etc. (not required for MVP)

## Inbox

- GET /v1/inbox?view=unread|all&limit=50
  - returns list of email summaries + computed loop status + agent hints (if available)
- GET /v1/email/{id}
  - returns full email + thread context (N previous messages)

## Triage

- POST /v1/email/{id}/archive
- POST /v1/email/{id}/delete
- POST /v1/email/{id}/snooze body: { until: iso8601, reason: string }
- POST /v1/email/{id}/remember body: { note_category?: string, why?: string }

## Delegated reply

- POST /v1/email/{id}/delegate body:

  {
    instruction: string,
    category_id?: string,
    autonomy_override?: 0..3
  }

  - returns job_id for draft generation

- GET /v1/drafts/{job_id}
  - returns draft text + rationale summary + citations

- POST /v1/drafts/{job_id}/send
  - sends via SMTP and records audit event

## Categories and policies

- GET/POST /v1/categories
- PUT /v1/categories/{id}
- GET/POST /v1/policies
- PUT /v1/policies/{id}

## Notes

- GET /v1/notes
- GET /v1/notes/{id}
- POST /v1/notes/rebuild body: { scope: "all" | "category:<id>" }
