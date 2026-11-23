# P19: review round 4 controls

Notebook: [docs/notebooks/quick_P19.ipynb](../../../docs/notebooks/quick_P19.ipynb). Predictions (committed before the
runs): [quick_runs/predictions/P19.md](../../predictions/P19.md). Numbers come from `summary.json`. The nfd lessons are in
`lessons_nfd.jsonl`; their labels (committed before the agent runs) are in `quick_runs/p19/labels_nfd.yaml`.

**One-line answer.** A data probe at write time recovers most of what the evaluator's outcome gave the store: with all
own lessons, 0.60 against 0.025 without any check (and 0.75 with the outcome). An escalating guide passes the fixes and
the contradicting wrong lessons to the user, but still gives the agent every wrong lesson that agrees with the
instructions. The one-sentence cut holds on each task alone.

## Terms

- **nfd lessons.** The P17 writer (no evaluator outcome) given the P16 data profile of the task's files instead, with
  "Use the profile to check what the agent assumed about the data." Same 40 source runs.
- **Escalating guide (G3x).** The P13 guide prompt plus: a note whose use would mean departing from the task's
  instructions is flagged ("conflict": true), not given to the agent, and shown to the user. **Flagged:** in a run, the
  guide flagged that lesson.
- Fix lessons, wrong lessons, reinforcing lessons, shown, pooled: as in P16-P18.

## What was run

- Guide only: 120 runs (G3s and G3s-pl, 10 more each per task; G3x, G3x-p, G3x-pc and G3n-nf, 20 each).
- Writer: 40 calls, 119 lessons, 0 parse errors.
- Agent: C1-nfd and E-nfd (30 each), A0h (20). All 80 ended with a final answer.
- Two TP=2 Gemma servers on GPUs 4-5 and 2-3, idle at launch. Wall time: 1 h 02 min, plus the labelling pause.

## Result 1: the one-sentence cut holds on each task

| Task | Fix shown, cut (of 15) | Fix shown, placebo (of 15) | Fisher p |
| --- | --- | --- | --- |
| TF03 | 15 | 1 | < 0.0001 |
| TF05 | 14 | 4 | 0.0005 |

## Result 2: escalation surfaces the fixes and the contradicting wrong lessons, not the agreeing ones

Runs (of 20; fixes: of 10, on TF03 and TF05):

| | G3x (clean) | G3x-p (P13 wrong lessons) | G3x-pc (agreeing wrong lessons) |
| --- | --- | --- | --- |
| A fix lesson flagged | 10 | 10 | 5 |
| A fix lesson shown to the agent | 0 | 0 | 0 |
| A wrong lesson flagged | | **18** | **1** |
| A wrong lesson shown to the agent | | 2 | **20** |
| Lessons flagged per run | 3.5 | 4.2 | 2.8 |

- The flag reasons are the same instruction reasons as before, now routed to the user: "The note suggests identifying
  and removing leakage features (like 'majorcards'), but the task explicitly instructs to 'Use all available feature
  columns'."
- The guide also flags lessons that do not conflict with the task. On TF01 it flagged stratification ("adding
  'stratify' changes the resulting training and validation sets, which may conflict with the intended split").
- In G3x-pc the TF05 fixes were not flagged in any run. The guide had accepted the agreeing lesson ("the spending columns
  are recorded before the decision") first.

## Result 3: the open prompt also keeps the no-checker lessons

G3n-nf showed a reinforcing lesson in 10 of the 10 TF02 and TF03 runs, and none elsewhere. Every arm run on that store now passes them:
embedding, all own lessons, and the guide with the original, data-profile and open prompts.

## Result 4: a data probe at write time gives fix lessons, but not for every pitfall

| | Fix lessons | Reinforcing lessons |
| --- | --- | --- |
| No checker (P17) | 0 | 10 (TF02 5, TF03 5) |
| Data probe (nfd) | **7** (TF01 2, TF03 5) | 7 (TF02 5, TF03 2) |

- TF03: every run produced "Inspect features for data leakage … 'column_x' had a +1.00 correlation in training but
  -0.04 in testing".
- TF01: 2 of 5 runs produced a class-weight lesson.
- TF02: the probe showed '183cm'. The writer still wrote "use pd.to_numeric(errors='coerce')" in 5 of 5 runs.
- TF05: nothing about leakage.

## Result 5: the probed store helps when all own lessons are given, little through retrieval

Targeted checks on the 4 failing tasks (passes out of 5):

| Check | A0 | C1 (outcome) | C1-nf (none) | C1-nfd (probe) | E (outcome) | E-nf | E-nfd | A0h (hint, no memory) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TF01 class distribution | 0 | 5 | 0 | 4 | 5 | 1 | 2 | 5 |
| TF01 class weights | 0 | 5 | 0 | 5 | 5 | 0 | 1 | 0 |
| TF01 stratified split | 3 | 5 | 1 | 5 | 5 | 0 | 3 | 5 |
| TF02 strips the unit | 0 | 4 | 0 | 0 | 4 | 0 | 0 | 5 |
| TF03 leaves out `column_x` | 0 | 5 | 0 | 5 | 0 | 0 | 0 | 5 |
| TF03 causal reasoning | 0 | 5 | 0 | 5 | 0 | 0 | 0 | 4 |
| TF05 leaves out all 4 leaks | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 5 |
| **Pooled (of 40)** | **0.075** | **0.75** | **0.025** | **0.60** | **0.475** | **0.025** | **0.15** | **0.725** |

- C1-nf against C1-nfd: p < 0.0001. C1-nfd against C1: p = 0.23. E-nf against E-nfd: p = 0.11.
- In C1-nfd the TF03 agent was shown the 5 fix lessons and the 2 reinforcing ones together, and it left `column_x` out
  in 5 of 5 runs. In P18 a fix shown with "keep `column_x`" was followed in 0 of 16 runs. The difference: these
  reinforcing lessons are generic ("use select_dtypes"), while lesson 130 names `column_x`.
- Embedding retrieval showed no nfd fix lesson in any run. On TF03 it showed a reinforcing lesson in all 5
  runs.
- With the hinted wording and no memory, the agent passed 29 of 40 check-runs (0.725). The hint does what memory at
  its best does, without retrieval.
- No harm: TF04 1.0; TF06 within 0.0007 of no memory in C1-nfd and E-nfd.

## Predictions (written before the runs)

| # | Prediction | Result |
| --- | --- | --- |
| 1 | Cut ≥ 11 of 15 and placebo ≤ 5 of 15 on each task, each p < 0.05 | **Held.** TF03 15/1, TF05 14/4. |
| 2 | G3x flags a fix in ≥ 7 of 10 | **Held.** 10 of 10. |
| 3 | G3x-p flags a wrong lesson in ≥ 12 of 20 | **Held.** 18 of 20. |
| 4 | G3x-pc flags ≤ 4 of 20 and shows ≥ 14 of 20 | **Held.** 1 and 20. |
| 5 | G3n-nf shows a reinforcing lesson in ≥ 8 of 10 | **Held.** 10 of 10. |
| 6 | nfd: ≥ 3 fix (TF01-TF03), ≤ 1 fix (TF05), ≤ 5 reinforcing | **Failed** (the third part). 7, 0, 7. |
| 7 | C1-nfd pooled ≥ 0.20 | **Held.** 0.60. |
| 8 | E-nfd pooled ≥ 0.10 | **Held.** 0.15. |
| 9 | A0h pooled ≥ 0.35 | **Held.** 0.725. |
| 10 | TF04 / TF06 within 0.01 of A0 (C1-nfd, E-nfd) | **Held.** |

9 held, 1 failed. Running total across P1-P19: 45 held, 17 failed, 1 not testable (of 63).

## What this means

1. Checking at the write step works where the data shows the problem. A probe of the files, given to the writer
   instead of the evaluator's outcome, produced the TF03 and TF01 fixes. It missed the unit suffix and leakage, as the
   read-time profile did. This is an upper bound: the profile includes test-set numbers.
2. A store only partly recovered by a probe still needs retrieval to find the fixes. Embedding retrieval reached
   none of them.
3. Escalation is a better use of an instruction check than rejection. Both the fixes and the contradicting wrong
   lessons go to a person. But a person who sees only the flagged notes never sees the agreeing wrong lessons, and
   those reach the agent in every run. A flag also costs the user 3-4 notes per run, some of which do not conflict.
4. Generic reinforcing lessons do not block a fix the way a lesson that names the column does.

## Limits

- 5 runs per agent cell. The escalation arms were measured at the guide only, with no user.
- The probe includes test-set numbers, so it is an upper bound on a write-time data check.
- One write generation; my labels.
