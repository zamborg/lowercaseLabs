# pia-pastry design

## Goal

Reach Butter & Crumble's Hotplate final order review with as little latency and
selector fragility as practical, while making an accidental or over-budget
purchase difficult.

## Live facts verified on July 28, 2026

- The storefront is `https://www.hotplate.com/butterandcrumble`.
- The page says pastry preorders drop on Tuesdays.
- No next drop is currently scheduled.
- The July 23–26 event exposed `Small Pastry Box` ($32) and `Large Pastry Box`
  ($68) buttons on a per-drop URL.
- A sold-out product opens a dialog, which makes the old event useful as a
  non-purchasing selector rehearsal.

These are observations, not durable API contracts. `npm run rehearse` is the
canary for UI drift.

## Architecture

```text
release clock
    │
    ▼
persistent Playwright profile ── reload+jitter ── Hotplate storefront
    │                                                  │
    │                                        next-drop card appears
    │                                                  ▼
    └────────────────────────────── product preference → pre-armed Add click
                                                       │
                                                       ▼
                                             random date/time fallback
                                                       │
                                                       ▼
                                              guarded checkout resolver
                                                       │
                         ┌─────────────────────────────┴──────────────┐
                         ▼                                            ▼
                   human handoff                           explicit auto gate
             CAPTCHA/auth/unknown UI              ack + total found + under cap
                                                                      │
                                                                      ▼
                                                               final click
```

The drop-to-cart loop is ordinary code. An LLM round-trip in that loop would add
latency and variance. Codex/browser control is best used to inspect the current
page, repair changed selectors, run the rehearsal, and supervise a live run.

## Purchase modes

- `dry-run`: detects a live product but does not click it. The pinned sold-out
  rehearsal is safe to open because it cannot create an order.
- `assist` (default): advances until the final purchase control, rings, saves a
  screenshot, and asks the operator to review/place the order.
- `auto`: clicks the final purchase control only when all three gates pass:
  `PIA_MODE=auto`, `PIA_AUTO_PURCHASE_ACK=BUY_PASTRIES`, and a displayed order
  total at or below `PIA_MAX_TOTAL_DOLLARS`.

CAPTCHA, verification-code, missing-payment, ambiguous-control, and unverified
total states always hand off. There is no CAPTCHA bypass or direct payment-data
storage.

## Speed choices

- A warm, persistent browser profile avoids a cold login/session start.
- The runner sleeps until shortly before a configured release timestamp.
- While the modal says `Available Soon`, Playwright starts waiting on the future
  `Add to Cart` button. Its actionability wait clicks as soon as the button
  exists and is enabled; there is no generic-control scan at the release edge.
- Polling reuses one page and adds light jitter; the minimum interval is 1.5s.
- Accessible names and roles avoid generated Hotplate CSS classes.
- Product selection is deterministic. Pickup preferences win when configured;
  otherwise a random compact date and time control is chosen immediately.

## Payment data

Payment credentials never belong in a project JSON file, `.env`, fixture, log,
or screenshot. The supported path is the dedicated persistent browser profile
with Hotplate/Stripe's tokenized saved-payment or Link session. A future
programmatic fallback should retrieve a secret from macOS Keychain at runtime
and keep it out of process logs and repository files.

## August 11 live-drop lesson

The first live run reached the product modal before noon, but the generic
resolver lost the sub-second transition from `Available Soon` to `Add to Cart`.
The revised flow pre-arms the exact Add locator and bypasses generic intent
resolution until the cart reservation is obtained. After that, short controls
that look like dates and times are selected before generic checkout actions.

## Failure posture

Every meaningful transition is appended to a JSONL run log. Drop discovery,
handoff, failure, and purchase confirmation also save full-page screenshots
under `artifacts/`. Unknown UI stops for a human rather than guessing.
