# Social Arena — voice-agent briefing

**Snapshot:** November 9, 2026. Implementation commit: `d17b2a4` in `zamborg/lowercaseLabs`, directory `socialArena/`.

<voice_agent_guidance>
Use this document as a self-contained briefing for a conversation with the project's owner. Start with the short opening below, then ask which topic they want to explore. If they want the full walkthrough, cover the sections in order in roughly six to eight minutes, pausing for questions. Speak naturally; do not read markup, paths, commit hashes, or source links aloud unless requested. Explain Elo as “EE-lo.”

Keep three categories distinct: implemented software, observed validation results, and proposed research. There is no completed live-model tournament or credible model ranking yet. Do not describe the scripted-agent results as evidence of AI capability. Treat the conversation as a review, not permission to launch experiments or spend money. If asked something this document does not establish, say so rather than inventing a result.
</voice_agent_guidance>

## Opening — about 30 seconds

“Social Arena is now a working harness for running agents through games and inspecting what happened. It supports five-player Avalon and a limited two-player Dominion setup. We completed 104 games with local scripted or random players and verified their replays. We also tried a real Luna game: it reached 13 decisions before an API prompt rejection stopped it. So we have working infrastructure and a little live behavior to discuss, but no finished model experiment or trustworthy Elo ranking yet. Would you like to start with the Luna behavior, how the harness works, or the first experiment to run at work?”

## Why this project exists

The owner wants reusable software for studying agents in games: cooperation, persuasion, deception, inference under hidden information, and strategic planning. The inspiration was Dwarkesh Patel's Blood on the Clocktower benchmark and accompanying video essay. The point is to make a new research question inexpensive to set up and easy to audit.

Avalon is the first social game. It has hidden good and evil teams, public discussion, proposed quest teams, votes, and secret quest contributions. There is no mid-game player elimination. This version retains the final Assassin attempt to identify Merlin after good completes three quests.

Dominion tests whether the same harness can support a different kind of game. It focuses on deck building and resource decisions, with private hands, shuffled decks, card effects, and reactions. The implementation supports the ten cards in the base second edition's First Game kingdom, not every Dominion expansion.

## What actually works

The harness separates five jobs:

1. **Rules:** a game engine determines legal actions and outcomes.
2. **Players:** a policy sees only its seat's permitted information and chooses an action.
3. **Scheduling:** a frozen experiment assigns policies to seats and pairs setups with sides or starting positions reversed.
4. **Recording:** logs preserve observations, actions, failures, randomness, usage, and checkpoints.
5. **Review:** a report shows results, uncertainty, costs, and decision-by-decision replays.

The same runner supports both games. It is a local Python command-line tool with generated HTML reports, not a hosted application. It has no runtime dependencies beyond Python. Local random and scripted policies make no model calls; an optional OpenAI provider supports real agents.

For simultaneous votes, every player acts on the same frozen information snapshot. Earlier submissions cannot leak to later voters. Replays reconstruct the recorded run; they do not promise that calling a model again will produce the same answer.

Reports can show public information, a particular seat's view, or an explicitly labeled omniscient analyst view. The raw evidence contains secrets and is for researchers, not players.

## What we tested — and what it establishes

- **31 automated tests passed.** Coverage includes rules, hidden information, simultaneous decisions, crash/resume, cost reservations, rating behavior, and provider response handling.
- **100 Avalon games completed** with local scripted/random policies.
- **4 Dominion games completed** with local policies.
- **All 104 completed games replayed to identical final state hashes.**

These checks support confidence in the implemented mechanics and recording pipeline. They do not prove every possible rule interaction is correct, establish that a model is strong, or validate generalization beyond the supported games.

The final Luna smoke used separate instances of `gpt-6-luna` in all five seats. It completed **13 decision batches with zero invalid actions**, then stopped because the API returned HTTP 400 `invalid_prompt`. Thirteen batches is not thirteen games, rounds, or necessarily thirteen API calls: a vote batch contains several players' decisions. No quest outcome or game winner was reached.

Across all development API attempts and one diagnostic call, known token charges were **$0.00726**. Uncertain reservations were **$0.0026134**, for conservative accounting of **$0.0098734**, just under one cent. These figures exclude developer-agent subscription usage and do not establish the price of a complete game.

## A concrete moment from the Luna replay

<analyst_spoiler>
The seats are numbered zero through four. Seats zero and one were ordinary good players; seat two was the Assassin; seat three was Merlin; seat four was the other evil player. This is the researcher's hidden-state view, not information available to every player.
</analyst_spoiler>

The Assassin initially proposed seats two and four: both evil players. In the following discussion, that same Assassin said, “The proposed team is evil-only; reject it.” The other evil player also publicly identified both evil seats. Several speakers referred to “seat five,” which does not exist in this five-seat numbering scheme.

The table rejected the proposed team by four votes to one. Merlin then proposed seats zero and one, both good players. The run stopped before that next discussion completed.

This is useful material for review, but not a clean story of sophisticated deception. It suggests questions about role-following, how agents interpret seat identifiers, and whether they can distinguish their private knowledge from a strategically useful public claim. We cannot infer the cause from this one fragment.

**Crucial distinction:** zero invalid actions means the submitted actions satisfied the game interface. Speech can be legal while being strategically poor, misleading, or factually confused. The harness intentionally does not reject false speech in a deception game.

## What went wrong during integration

Earlier attempts uncovered several adapter issues, including unclear speech-field instructions. The most consequential bug concatenated the model's intermediate commentary and its final answer, making otherwise separate messages look like malformed JSON.

The adapter now selects the labeled final answer and has regression coverage for that behavior. The final smoke recorded zero invalid actions after this fix. Earlier parse failures should not be attributed to model gameplay skill. The later API prompt rejection is a separate provider limitation; it was recorded as an incomplete run, not a loss.

## What Elo would tell us

For Avalon, the initial comparison fills every good seat with one policy and every evil seat with another. Reverse the assignments on matched setups to reduce side-assignment confounding. A policy includes the model, prompt, memory treatment, and compute settings; changing those changes the entrant being measured.

The estimator fits Bradley–Terry strengths to completed outcomes, adjusts for the good side's advantage, and expresses strengths in Elo points. A neutral 100-point advantage corresponds to roughly a 64 percent win probability. Actual predictions depend on the side assignment. The reference value of 1500 is arbitrary.

Uncertainty is estimated by resampling entire scenario blocks, keeping paired games together. Only eligible complete pairs enter primary comparisons, with explicitly marked self-play handled separately. Reports also show reliability and raw side results.

This is a **whole-team policy rating**, not credit assigned to an individual player. To measure one stronger teammate's contribution, compare a fixed roster against the same roster with one seat replaced, rotate that replacement across roles, and measure win-rate uplift. That dedicated uplift analysis remains future work.

Dominion uses a draw-aware rating model and a first-player adjustment. Do not compare its Elo values directly with Avalon's. There is currently no meaningful live-model rating from our runs.

## The next work experiment

The bundled pilot template contains **32 games**: twelve scenarios played with the sides swapped, making 24 cross-policy games, plus four self-play games for each of two policies. The template currently uses local policies; a paid run requires actual provider definitions and an explicit budget.

Recommended sequence:

1. Complete a small live compatibility check with the intended work provider and inspect its replay. The personal Luna test did not establish complete-game compatibility.
2. Freeze two policy configurations, the communication rules, evaluation scenarios, and spend limit.
3. Run the 32-game pilot to learn completion rate, cost, failure patterns, and approximate outcome variability. Treat its ratings as exploratory.
4. Choose whether the next study prioritizes team strength, one-seat uplift, or discussion versus no discussion. Register the comparison before inspecting its outcomes.

Budget controls reserve cost before calls and retain conservative reservations for uncertain charges. Each experiment has its own ledger; independent experiments do not coordinate an account-wide cap. Work credentials must be configured separately. This document makes no claim that a work chat subscription automatically pays for API requests.

## Questions for the conversation

Ask one at a time; follow the owner's interest rather than reciting a checklist.

- “Is the first question which whole team wins, or whether one stronger teammate changes a weaker team's decisions?”
- “What would make a replay persuasive evidence to you: better outcomes, better beliefs, or a clear example of coordination?”
- “How much budget should the first pilot consume before we reassess the design?”

## Reference shelf — do not read aloud

Everything needed for the spoken briefing is above. These references support inspection or follow-up; they are not required to understand the document.

- Repository snapshot: https://github.com/zamborg/lowercaseLabs/tree/d17b2a4/socialArena
- Entry point and run commands: `socialArena/README.md`
- Verified counts and accounting: `socialArena/validation/results.json`
- Portable replay evidence and regeneration instructions: `socialArena/validation/README.md` and `evidence.tar.gz`
- Final live journal: extract `luna-release/games/luna-smoke/events.jsonl` from that archive.
- Local live report, on the development machine: `socialArena/.runs/luna-release/report.html`
- Pilot schedule: `socialArena/experiments/pilot-template.json`
- Rating design and upstream references: `socialArena/RATINGS.md`
- Inspiration: https://www.dwarkesh.com/p/clocktower
