---
name: auth
description: Protect a mom-made app or add sign-in, accounts and per-person data on Cloudflare Workers + D1.
---

# Sign-in and privacy in mom-made

Published apps are public at their workers.dev address. If the person wants an app private, or wants people to have their own data, pick the lightest option that fits and explain it in plain words.

## Pick one
1. **Only certain people can open it (family, friends, a team).** Use Cloudflare Access. It puts an email-code login in front of the whole app, needs no code, and is free for up to 50 people. mom-made does not set it up yet: tell the person to open Cloudflare, go to Zero Trust, Access, Applications, add the app's workers.dev address, and allow the email addresses they want. Do not claim it is on until they say they did it.
2. **One shared passcode.** Fine for low-stakes pages. Keep it in D1 (hashed), never in frontend code.
3. **People sign up and see their own things.** Build accounts in the Worker (below).

## Accounts in the Worker
- Table `users(id, email, password_hash, salt, created_at)` and `sessions(token_hash, user_id, expires_at)` added through a new file in migrations/.
- Hash passwords with Web Crypto PBKDF2 (SHA-256, 100000+ iterations, random per-user salt). Never store or log plain passwords.
- On sign-in create a random token (crypto.getRandomValues, 32 bytes), store only its SHA-256, and set it in a cookie with HttpOnly, Secure, SameSite=Lax, Path=/. Sessions expire (30 days is fine).
- Check the session in a Hono middleware on every /api/* route except sign-in and sign-up. Return 401, and have the frontend show the sign-in screen on 401.
- Scope every query by the signed-in user's id (WHERE user_id = ?). Never trust a user id sent from the browser.
- Rate-limit sign-in attempts per email (a small D1 counter is enough). Use the same error for a wrong email and a wrong password.
- Do not rely on secrets the person must paste in. Everything above works with D1 only.
- If a Cloudflare Access login is in front, the Worker can read the visitor's email from the Cf-Access-Authenticated-User-Email header. Only trust it once the person confirms Access is on, since without Access anyone can forge it.

## Always
- Test the signed-out, signed-in and wrong-password paths in the preview before reporting back, and say what you checked.
- Keep the sign-in screen as simple and warm as the rest of the app: one clear action, clear errors.
- Tell the person plainly if something is only a light lock (shared passcode) and not real security.
