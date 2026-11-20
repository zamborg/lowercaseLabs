# Portal Valley Car Club · PBCC

A trusted shared third garage: a small, member-owned collection of interesting cars for an exclusive internal group around Portola Valley.

**Stage:** formation research; no members admitted, capital collected, vehicles acquired, or vendors engaged. Working name is provisional; the founder's Portola Valley neighborhood is confirmed. Initial brief and fleet exploration: November 8, 2026.

## Start here

- [November founder handoff](docs/founder-handoff.md): six-car proposal, dues gap, and open items from the available ChatGPT summary.
- [Office plan](operations/office-stack.md): front office, admin, and back office responsibilities and build order.
- [Initial COO briefing](docs/coo-briefing.md): four-car pilot baseline, economics, and launch sequence.
- [Club charter](docs/charter.md): founder intent, assumptions, and division of responsibilities.
- [Landscape and local partners](docs/landscape.md): comparable operating models and sourced vendor leads.
- [Storage negotiation brief](docs/storage-plan.md): AutoKeep Menlo Park priority, $400–$500 dues, and member-led cost sensitivities.
- [Formation questions](docs/formation.md): counsel, accountant, and insurance discovery agenda.
- [Fleet and reservations](docs/fleet-and-reservations.md): four-car concept, fairness pilot, and vehicle procedures.
- [Sample garages and price spot checks](docs/sample-fleets.md): Miata, air-cooled 911, electric truck, V12 BMW, and optional exotics; [scenario inputs](finance/fleet-options.json).
- [Operating board](operations/board.md): next actions, evidence gates, decisions, risks, and weekly briefing format.
- [Outreach drafts](operations/outreach.md): concrete service, storage, insurance, and member discovery messages.
- [Financial assumptions](finance/assumptions.json) and [generated scenarios](finance/scenarios.md) for the four-car pilot.

## Working together

Open this directory as the project. [AGENTS.md](AGENTS.md) supplies project context; [CLAUDE.md](CLAUDE.md) routes Claude Code to the same instructions. Reusable workflows live in [.agents/skills](.agents/skills):

| Skill | Example request |
| --- | --- |
| `pbcc-coo` | “Give me this week's PBCC operating brief and update the board.” |
| `pbcc-vendors` | “Compare these storage quotes and draft follow-ups.” |
| `pbcc-inbox` | “Review these PBCC email threads and prepare replies.” |
| `pbcc-fleet` | “Evaluate this car for our fleet” or “Review booking fairness.” |

Codex can discover project-local skills when this folder is opened as the workspace. Other harnesses can read the named `SKILL.md` directly; the workflows are ordinary Markdown, with no agent service dependency.

Recalculate the model using only Python's standard library:

```sh
python3 scripts/model.py
python3 scripts/model.py --output finance/scenarios.md
```

The first command prints a report; the second explicitly updates the saved report. Edit assumptions before relying on an output. Purchase figures are allocation ceilings, insurance and most running costs are unquoted placeholders, and the tax rate is a sensitivity input.

## Current automation boundary

The working system consists of persistent project records, reusable operating skills, and a deterministic financial model. Email, calendar, payments, and booking services are **not connected**; there is no scheduled background agent. Inbox work can start with supplied threads or an explicitly scoped connected mailbox later. Outreach is prepared as drafts; nothing has been sent.

Keep driver records, member identities, insurance documents, banking details, and full private correspondence outside tracked files. `private/` is ignored locally for convenience, but is not encrypted storage. Use pseudonymous member IDs in planning records and an appropriate secure system for sensitive documents.
