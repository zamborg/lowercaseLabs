# pia-pastry operating rules

- Treat `src/automation.mjs` as the speed-critical path. Keep it deterministic; do not add an LLM call between drop detection and checkout.
- Never add CAPTCHA bypassing, stealth plugins, direct payment-card storage, or aggressive request flooding.
- Keep `assist` as the default mode.
- Preserve the three-part auto-purchase gate: `PIA_MODE=auto`, the exact acknowledgement token, and a verified displayed total below the configured cap.
- Pre-arm the exact `Add to Cart` locator while the product modal says `Available Soon`; do not route that release-edge click through the generic checkout resolver.
- Never persist card numbers, CVVs, passwords, OTPs, or other payment credentials in JSON, `.env`, fixtures, logs, or screenshots. Prefer the dedicated browser profile and tokenized provider sessions.
- Test selector changes with `npm run rehearse` against the pinned sold-out drop before using `npm run watch`.
- Hotplate is an external UI. Prefer accessible roles and visible text over generated CSS classes.
