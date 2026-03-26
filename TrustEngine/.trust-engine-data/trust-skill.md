# Trust Skill

You are operating behind a user-specific trust proxy. Treat these rules as the current trust posture, not as permission to improvise beyond them.

## Usually Allow
- Nothing promoted yet. Default to asking.

## Always Escalate
- `Bash` touch test-file.txt && rm test-file.txt && echo "Done"
- `Bash` touch temp.mdd
- `Bash` rm temp.mdd
- `Bash` touch temp.mdd
- `Bash` ls -la
- `SessionEnd` {}

## Conditional Patterns
- `Bash` ls -la
- `Glob` {"pattern":"*"}

## Operating Rules
- If an action is not clearly covered by "Usually Allow", ask.
- High-risk communication and external side effects require escalation unless explicitly trusted.
- Repeated user approvals can justify suggesting an allow rule, but not silently broadening the action class.
- Prefer semantically meaningful tools over generic shell commands when handling email, messaging, calendar, or other life-admin workflows.
