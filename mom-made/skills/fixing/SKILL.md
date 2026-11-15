---
name: fixing
description: Diagnose a blank page, an error, or something that does not work, and report honestly.
---

# When something is broken

1. Look before guessing: app_page_text, app_console, app_logs, and a screenshot if available.
2. Find the cause, then make the smallest change that fixes it. Do not rewrite working parts.
3. Check a database error against migrations/: a missing table or column means a new migration is needed, not a change inside a request handler.
4. Reload and repeat the exact action that failed. Confirm the console is clean.
5. Report in plain words: what was wrong, what you changed, what you checked. If you could not fix it or could not verify it, say so and say what you would try next.
6. If the problem is outside the app (assistant sign-in, Cloudflare account, internet), say so simply and point to Settings or Publish instead of editing code.
