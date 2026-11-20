# pia-pastry 🥐

A low-latency, guarded Hotplate runner for a Butter & Crumble pastry box.

It uses a dedicated persistent Playwright browser profile. Codex is useful for
rehearsal and UI-drift repair, but there is no model call in the drop-to-cart
hot path.

At release, the product modal is opened early and an exact `Add to Cart` click
is armed before the button exists. Playwright fires it as soon as Hotplate
enables it. After the cart reservation, configured pickup preferences are tried
first; otherwise the runner chooses a random available date and time control.

## Quick start

Requirements: Node.js 20+.

```bash
cd pia-pastry
npm install
npm run install-browser
cp .env.example .env
npm run rehearse
npm run setup
```

Edit `.env` before drop day. In particular, choose the desired products, pickup
preference order, maximum total, and the known release timestamp.

Start the runner several minutes early:

```bash
npm run watch
```

The default `assist` mode reaches the final review, rings, saves a screenshot,
and waits for you. Keep the terminal and headed browser visible.

## Auto mode

Auto mode can make a real purchase. It is intentionally three-keyed:

```dotenv
PIA_MODE=auto
PIA_MAX_TOTAL_DOLLARS=40
PIA_AUTO_PURCHASE_ACK=BUY_PASTRIES
```

It still stops for CAPTCHA, OTP/account input, missing payment data, an unknown
control, a total it cannot verify, or a total over the cap.

Do not put phone numbers, verification codes, passwords, or payment card data in
`.env`. Prepare those in the dedicated browser profile with `npm run setup`.

## Product and pickup policy

The conservative default buys only one `Small Pastry Box`. To accept a large
box as a fallback, explicitly change:

```dotenv
PIA_PRODUCTS=Small Pastry Box,Large Pastry Box
PIA_MAX_TOTAL_DOLLARS=75
```

Pickup preferences are evaluated left-to-right, not by whatever order Hotplate
renders:

```dotenv
PIA_PICKUP_PREFERENCES=Thursday,Thu,Friday,Fri,Saturday,Sat,Sunday,Sun
PIA_PICKUP_TIME_PREFERENCES=10:00 AM,10:30 AM,11:00 AM
```

Leave time preferences blank if any time is acceptable or the UI does not ask
for time until checkout. Random fallback is enabled by default:

```dotenv
PIA_RANDOM_PICKUP=true
```

## Payment details

Do not create a JSON file containing card data. Keep payment in the dedicated
browser profile using Hotplate/Stripe's saved or tokenized session:

```bash
npm run setup
```

The runner deliberately hands off for missing payment, OTP, or CAPTCHA. If a
future programmatic fallback is needed, use macOS Keychain retrieval at runtime,
never a repository file or `.env`.

## Commands

```bash
npm run setup
npm run rehearse
npm run watch
npm test
npm run pia -- help
```

`npm run rehearse` targets the pinned July 23–26 sold-out event and validates
that the current storefront/product/modal structure is still recognizable.

Read [DESIGN.md](./DESIGN.md) for the state machine and safety boundaries.
