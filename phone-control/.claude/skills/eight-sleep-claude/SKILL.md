---
name: eight-sleep-claude
description: Use Claude's computer-use tools and a mirrored iPhone to read or set Eight Sleep Pod temperature, apply a temperature profile, or run authorized overnight checks.
---

# Eight Sleep in Claude

Read [iPhone Mirroring in Claude](../iphone-mirroring-claude/SKILL.md) and [Eight Sleep control](../../../skills/eight-sleep-control/SKILL.md). The Mirroring adapter loads the shared Apple workflow and provides the Claude computer-use calls. Use Eight Sleep control for app navigation, profile decisions, and result verification.

A read-only check opens Eight Sleep and reports the temperature card's displayed target, or **Off**. It leaves the dial, schedule, and Pod power untouched. Write only when the user has approved an absolute target.

For a scheduled read or write, load Eight Sleep control's [overnight profile reference](../../../skills/eight-sleep-control/references/overnight-profile.md) and use it for the individual check. The scheduled run must use the local Mac that has access to the user's iPhone and must have its own computer-use access.
