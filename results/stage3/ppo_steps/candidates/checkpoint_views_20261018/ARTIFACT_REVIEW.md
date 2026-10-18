# Checkpoint-view artifact review

Four new views of the same saved DPPO estimates; no jobs or new statistics. Each figure uses common four-decade axes. Point estimates alone do not show the uncertainty, which remains in the existing table.

| ID | What it answers / limitation | Suggested curated name (PNG and PDF, with caption) |
|---|---|---|
| A | DPPO gradient noise across 20 steps after 32 updates | `dppo-per-step-noise-32-updates` |
| B | The same quantities after 63 updates | `dppo-per-step-noise-63-updates` |
| C | The same quantities after 95 updates | `dppo-per-step-noise-95-updates` |
| D | Compare all three checkpoints on identical axes | `dppo-per-step-noise-checkpoint-grid` |

[Figures and axis-limit note](../../figs/PER_CHECKPOINT_NOTE.md). Approved on 2026-10-18: "keep curated", covering A–D under the proposed names. All four PNG/PDF pairs and captions are in the [curated index](../../curated/INDEX.md#per-checkpoint-views); the existing curated figures are unchanged. [Curation decision](../../runs/curation_checkpoint_views_20261018/decision.json).
