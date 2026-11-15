---
name: data
description: Store and show an app's information (lists, entries, settings) in D1 through the Hono Worker.
---

# Saving information in mom-made

- Anything that must still be there tomorrow (entries, lists, settings, scores) goes in the D1 database through the Worker. Never use localStorage as the only copy, and never keep data only in memory.
- Add tables with a new numbered file in migrations/ (0002_*.sql). Use CREATE TABLE IF NOT EXISTS, an id, and created_at. Never edit a migration that already exists.
- Write routes in worker/index.ts under /api/*. Use prepared statements with bound values (`c.env.DB.prepare("... WHERE id = ?").bind(id)`). Never build SQL by joining strings.
- Validate input in the Worker: check types, trim text, cap lengths, and return a clear 400 message the page can show.
- Give every list an empty state ("Nothing here yet, add the first one"), a loading state, and a visible error with a retry.
- After a change, update the screen straight away and keep it correct if the request fails.
- Anything the person would be upset to lose (deleting a record) asks first, or offers undo.
- Local preview data is separate from the published app's data. Tell the person if they ask: published apps start empty.
- Do not store passwords, card numbers or other highly sensitive data. See the auth guidance for sign-in.
