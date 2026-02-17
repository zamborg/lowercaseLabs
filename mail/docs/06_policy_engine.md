# Policy engine

## Why a policy engine exists

You want user-defined categories with stable behavior. This must not be prompt vibes.

## Policy enforcement rules

- If policy says no send, API must hard-block.
- If policy restricts tone/signoff, the agent must conform; API can lint for signature presence.
- Escalation rules can force:
  - always ask before sending if subject contains X
  - always remember emails from CEO
- Every policy has a deterministic JSON schema validated server-side.

## Category matching

- Start with simple rules:
  - sender domain
  - sender email exact match
  - subject keywords
- Then add ML classification as suggested category, not binding.
