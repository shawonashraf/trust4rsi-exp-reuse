# Probe: DeepSeek as the agent, Claude Opus 5.5 as the guide (not in the paper)

Branch history: `probe/deepseek-agent-claude-guide`, merged into `main` for the code only. Raw records:
`/data/anon/reveries_runs/PROBE_A0ds` and `PROBE_G3cds`. Token ledgers in this folder. the author asked for this probe
on 2026-11-17 ("We're just probing this. It doesn't need to be reported"; "don't add it to the submission"). No
predictions were written for it, so it is exploratory. It motivates the paper's future-work sentence only: whether
memory helps weaker agents more than strong ones.

**One-line answer.** A more capable agent avoids two of the four silent failures without any memory. DeepSeek with
no memory passed 0.425 of the pitfall check-runs, against 0.075 for Gemma. Opus guidance added about nothing (0.45
against 0.425, same seeds). The two pitfalls whose fix breaks the task's wording (TF03, TF05) stayed unsolved.

## Terms

- **Agent:** DeepSeek-V41-Flash-1017-FP8 on the SPIKE cluster (OpenAI-compatible endpoint; `run.py --agent-openai`).
  The same agent harness, task wording and step limits as the paper. The run-time extractor also used DeepSeek.
- **Guide:** Claude Opus 5.5, original prompt, clean store. Its picks were reused from P20 (`P20_inputs/G3c.yaml`),
  so no new guide calls were made.
- **Judge:** Claude Haiku 5.5 (`score.py --judge-claude`), majority of 3 draws on the suite's criteria. All local
  GPUs were in use by other users, so the paper's Gemma judge could not run. **The numbers are therefore not
  comparable with the paper's**: part of any gap to Gemma's 0.075 could be the judge.
- **Pooled:** share of the 8 pitfall check-runs per run passed, on the 4 failing tasks.

## What was run

- 80 agent runs, all ended with a final answer: 4 failing tasks × 10 seeds × 2 arms.
- **Design error.** The P20 Opus picks exist only for seeds 0-4. The arm meant to be "with guidance" also ran seeds
  5-9, and `run.py` passes no lessons when a seed has no entry. So its seeds 5-9 ran **without memory**. The table
  below splits the arms by seed; the only valid guided-vs-unguided comparison is on seeds 0-4.
- Tokens: DeepSeek agent and extractor 0.76M prompt / 0.57M completion (379 calls). Haiku judge $0.098 (240 calls).

## Result: the capable agent handles TF01 and TF02 on its own

Passes, out of 5 runs per cell:

| Check | No memory, s0-4 | No memory, s5-9 | "Guided" arm s5-9 (no lessons) | **Opus guide, s0-4** |
| --- | --- | --- | --- | --- |
| TF01 checks class distribution | 4 | 4 | 5 | 5 |
| TF01 class weights | 4 | 3 | 4 | 3 |
| TF01 stratified split | 4 | 4 | 4 | 4 |
| TF02 strips the unit | 5 | 4 | 5 | 5 |
| TF03 leaves out `column_x` | 0 | 1 | 1 | 1 |
| TF03 causal reasoning | 0 | 1 | 1 | 0 |
| TF05 leaves out all leaks | 0 | 0 | 0 | 0 |
| **Pooled (of 40)** | **0.425** | **0.425** | **0.50** | **0.45** |

- **No memory, all 30 no-lesson runs:** 0.45 (54 of 120). Gemma with no memory in the paper: 0.075, judged by Gemma.
- **Guidance:** 0.45 with the Opus picks against 0.425 without, on the same seeds. That is one check-run, inside the
  noise.
- I read one TF02 run's code (no memory, seed 0). The agent listed the non-numeric values first, stripped `cm`, then
  coerced, and mapped `na` to missing. Gemma never did this without memory.
- TF03 and TF05 stay unsolved (0-1 of 5 and 0 of 5). These are the two pitfalls whose fix needs the agent to break
  the task's wording; the Opus guide rejects those fixes (P20), so it had no fix to pass on.

## What this means

1. The silent failures that memory fixes in the paper are partly a property of the agent. A stronger agent avoids
   the unit and imbalance pitfalls by itself, so read-time memory has less to add.
2. The pitfalls that need breaking an instruction stay hard for the stronger agent too, and the instruction-ranking
   guide still drops their fixes. That is consistent with the paper.
3. Whether memory helps weaker agents more than strong ones is a fair future-work question (the paper's sentence).
   This probe does not answer it, because of the judge and the missing guided seeds.

## Limits

- Haiku judge, not the paper's Gemma judge; no human check.
- 5 guided runs per task; no predictions written; one agent model.
- The "guided" arm's seeds 5-9 had no lessons (see above). The note `NOTE.md` written on 2026-11-17 reports that arm
  as guided for all 10 seeds and gives the pooled share "of 40"; both are corrected here (80 check-runs per arm).
- To make this a paper arm: Gemma judge on all 80 runs, Opus picks for seeds 5-9 (or drop them), and predictions
  written first.
