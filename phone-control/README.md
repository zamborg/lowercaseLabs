# Phone control

Use Apple's iPhone Mirroring to operate apps on a nearby iPhone from a Mac. The first app workflow is Eight Sleep.

| Skill | Role |
| --- | --- |
| [iPhone Mirroring](skills/iphone-mirroring/SKILL.md) | Shared Apple setup, navigation, and reconnection workflow |
| [Eight Sleep control](skills/eight-sleep-control/SKILL.md) | Shared Pod controls, profiles, and overnight checks |
| [iPhone Mirroring in Codex](.agents/skills/iphone-mirroring-codex/SKILL.md) | Codex Computer Use adapter for the shared mirroring workflow |
| [Eight Sleep in Codex](.agents/skills/eight-sleep-codex/SKILL.md) | Codex entry point combining the adapter and shared Eight Sleep workflow |

Open `phone-control` as a Codex project to use `$iphone-mirroring-codex` or `$eight-sleep-codex`. From another project, point Codex to the corresponding `SKILL.md` directly.

For Claude Code or another agent, put its macOS computer-control calls in its own adapter. Its Eight Sleep entry point can load that adapter and the same two shared skills. The adapter needs to select the Mirroring window, observe the phone screen, and send input to it.

The interactive Codex test opened Eight Sleep on a real iPhone and verified that its temperature card showed **Off**. The [overnight workflow](skills/eight-sleep-control/references/overnight-profile.md) describes how to test scheduled reconnection with the Mac and phone in their intended overnight conditions.
