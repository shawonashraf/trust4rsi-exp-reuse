# P23: DeepSeek guide — token-counting probe (SPIKE budget)

Notebook/raw: `quick_runs/results/P23/trace_deepseek.jsonl` (guide trace) and `ledger_deepseek.jsonl` (per-call
tokens). Code: the `OpenAILLM` backend in `reveries/llm.py` and the G3ds arms in `quick_runs/p13/guide.py`. The key and
base URL are in the gitignored `.env` (`SPIKE_API_KEY`, `SPIKE_BASE_URL`); nothing secret is committed.

**Purpose.** Measure how many tokens a guide experiment needs when the guide is DeepSeek-V41-Flash-1017-FP8 on the
SPIKE cluster (OpenAI-compatible endpoint), so the SPIKE token budget can be sized. Secondary: a fifth guide family.

**One-line answer.** A full single-model guide experiment (70 runs over the three stores) costs **0.24M prompt and
0.50M completion tokens**. The weekly SPIKE budget is 110M prompt and 11M completion, so one guide experiment uses **0.2% of the prompt and 4.5% of the completion budget**; an agent run costs about the same in completion. Completion is the binding side: DeepSeek reasons heavily
(~7,100 completion tokens per guide run, 2-3x Claude). The current budget runs this experiment end to end many times
over.

## What was run

The guide is DeepSeek-V41-Flash-1017-FP8 through the SPIKE gateway; no local GPU. Original P13 prompt, retry budget
R = 3, three stores: clean (G3ds, 6 tasks), P13 wrong lessons (G3ds-p, 4 tasks), agreeing wrong lessons (G3ds-pc, 4
tasks). 70 guide runs, 5 per task. The agent stage (Gemma, local) was not run here; it bills no SPIKE tokens.

## Token counts (measured)

| | Prompt tokens | Completion tokens | Total |
| --- | --- | --- | --- |
| Whole experiment (70 guide runs, 174 API calls) | 240,406 | 500,230 | 740,636 |
| Per guide run (mean) | 3,434 | 7,146 | 10,580 |
| Per API call (mean; 2.49 calls/run) | 1,382 | 2,875 | 4,257 |
| Per guide run (min / max completion) | | 879 / 15,099 | |

- Completion dominates: DeepSeek-V41-Flash emits about twice as many completion tokens as prompt tokens on this task,
  because it reasons at length before the JSON reply. For comparison, Claude Opus 5.5 used ~2,200 completion / run and
  Fable 5.1 ~3,400 / run on the identical arms; DeepSeek uses ~7,100.
- All 174 calls finished with stop (no truncation at 16k max tokens) and all 174 returned parseable JSON, so these
  counts are for complete, usable guide runs, not retries or failures.
- Per-run completion varies widely (879 to 15,099): a few runs reason much more than the mean.

## Against the SPIKE budget

Weekly budget (light tier, resets Monday 00:00 CET): **110,000,000 prompt**, **11,000,000 completion** (10M weekly +
100M / 10M extra that expires 2026-12-31).

| Workload | Prompt | Completion | % of weekly prompt | % of weekly completion |
| --- | --- | --- | --- | --- |
| This experiment (70 guide runs) | 0.24M | 0.50M | 0.2% | 4.5% |
| A 140-run run (e.g. two seeds, or 6 arms) | ~0.48M | ~1.0M | 0.4% | 9.1% |
| Replicating every guide arm in the paper (~900 runs) | ~3.1M | ~6.4M | 2.8% | 58% |
| Weekly ceiling at the mean rate | | | 32,000 runs | **1,539 runs** |
| Weekly ceiling at the worst-case rate (15.1k compl./run) | | | | 728 runs |

- **Completion tokens are the binding constraint**, not prompt. The budget allows roughly **1,500 guide runs per
  week** at the mean rate, or ~730 even if every run reasoned at the observed maximum.
- The current budget is therefore more than enough to run this DeepSeek-guide experiment end to end, with headroom to
  repeat it for more seeds or add arms within one week.
- If a much larger study is planned (every arm, many seeds, several models), the ~6.4M completion for a full
  single-pass replication still fits one week; a request for a larger completion budget is only needed to run several
  such replications in the same week or to add long-context arms.

## Agent-stage token counts (DeepSeek as the agent)

A later probe ran DeepSeek-V41-Flash as the **agent** (the ReAct data-science loop), not the guide. Measured over 81
agent runs (`quick_runs/results/PROBE/ledger_ds_agent.jsonl`):

| | Prompt tokens | Completion tokens |
| --- | --- | --- |
| Per agent run (mean; ~4.7 calls incl. the extractor) | 9,368 | 7,050 |
| Per guide run (for comparison) | 3,434 | 7,146 |

The agent uses more prompt (the transcript grows each step) but about the same completion as the guide. Completion is
the binding side for both.

## Combined weekly capacity (completion-bound)

Weekly budget: 110M prompt, 11M completion (completion binds). At the measured rates:

| Workload (per run) | Completion/run | Runs/week within the 11M completion budget |
| --- | --- | --- |
| Guide only | 7,146 | ~1,540 |
| Agent only | 7,050 | ~1,560 |
| Guide + agent (a full task-run) | ~14,200 | ~775 |

So the current budget supports, in one week, on the order of **1,500 guide or agent runs, or ~775 full guide+agent
task-runs** — far more than any single experiment here (70-140 runs). A larger budget is only needed to run several
such studies per week or to add long-context arms. For reference, DeepSeek's completion per run is ~2-3x Claude's
(Opus 2.2k, Fable 3.4k per guide run), because it reasons at length; a less verbose model would stretch the budget
further.

## The guide also behaved like the others (free by-product)

| Store | DeepSeek-Flash | Gemma | Opus 5.5 | Fable 5.1 |
| --- | --- | --- | --- | --- |
| Clean: TF03 fix shown / keep-`column_x` (of 5) | 1 / 5 | 1 / 4 | 1 / 4 | 1 / 4 |
| Clean: TF05 fix shown (of 5) | 0 | 0 | 0 | 0 |
| Contradicting wrong lessons shown (of 20) | 1 | 5 | 0 | 0 |
| Agreeing wrong lessons shown (of 20) | 20 | 20 | 1 | 9 |

- On the clean store DeepSeek rejects the TF03/TF05 fixes exactly like every other judge: the instruction-ranking is
  universal across five families now (Gemma, Qwen, Opus, Fable, DeepSeek).
- It passes all 20 agreeing wrong lessons, like Gemma, unlike Opus (1) and Fable (9). So a reasoning-heavy model is
  not automatically resistant to the agreeing trap: among the models tried, only Opus caught them. This supports the
  paper's revised claim that the agreeing-lesson result is model-specific, not purely a function of scale or
  reasoning.

## Status for the paper

The guide experiment ran end to end within budget, and the result is consistent and clean (no truncation, all JSON
parsed). The guide-stage tokens are a small fraction of the weekly budget, so a DeepSeek arm can be included in the
paper as a fifth guide family; this needs the Gemma agent stage (local, no SPIKE tokens) to produce scores. If it is
not included, this file stands as the token-counting probe for the budget request.

## Limits

- One model, 70 guide runs, one pass; completion-token use has high per-run variance.
- The counts are the guide stage only; the agent runs are Gemma and bill no SPIKE tokens.
- SPIKE is a shared cluster; wall-clock time is not reported here, only token use.
