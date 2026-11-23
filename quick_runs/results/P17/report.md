# P17: a store written without a checker

Notebook: [docs/notebooks/quick_P17.ipynb](../../../docs/notebooks/quick_P17.ipynb). Predictions (written before the
runs): [quick_runs/predictions/P17.md](../../predictions/P17.md). Numbers come from `summary.json`. The lessons are in
`lessons_nf.jsonl`, and their labels (made before any guide or agent run) are in `quick_runs/p17/labels_nf.yaml`.

**One-line answer.** The agent's own lessons turned silent failures into rules. With the evaluator's outcome
removed, the agent wrote no lesson that fixes any of the 4 pitfalls. It wrote 10 lessons that encode the failure on
TF02 and TF03. Every guide, the data-profile guide included, passed them in every run. Memory then scored at or
below no memory (0.0-0.025, against 0.075; with the evaluated store, 0.475-0.75).

## Terms

- **nf lessons.** Lessons written by the same writer (P12, Gemma 4 31B, temperature 0) from the same 40 runs, but
  without the "Evaluation of this run" section. The writer sees only the task and what the agent did.
- **nf store.** The 708-lesson store with its 99 Silent Failures lessons replaced by the 116 nf lessons. BLADE and
  DSBench lessons are unchanged.
- **Reinforcing lesson.** A lesson that tells the agent to do what fails the pitfall check. The rule for this was
  fixed in the predictions.
- **Arms.** C1-nf (all of a task's own nf lessons), E-nf (embedding top-3), G3-nf (the P13 guide), G3d-nf (the P16
  data-profile guide). Each ran 6 tasks × 5 runs.

## Result 1: without the outcome, the lessons encode the failure

| Task | nf lessons | Fix | Reinforcing | Example of a reinforcing lesson |
| --- | --- | --- | --- | --- |
| TF01 | 13 | 0 | 0 | (scaling, ID columns, metric choice) |
| TF02 | 15 | 0 | 5 | "use pd.to_numeric with errors='coerce'" (turns '183cm' into a missing value) |
| TF03 | 15 | 0 | 5 | "Strict Adherence to Feature Selection Instructions … include all numerical columns" (keeps `column_x`) |
| TF05 | 14 | 0 | 0 | (all about encoding and target mapping) |

None of the 116 lessons mentions class balance, unit suffixes, `column_x` or leakage. The writer called its own
runs correct: "The agent correctly implemented a new `scaler_full`…". Four of the five TF03 reinforcing lessons are titled
"strict adherence" to the task's feature instructions; the fifth is "Automate Numerical Feature Selection"
(`select_dtypes(include=[np.number])`).

## Result 2: every guide passed them, even with the data in view

| | C1-nf | E-nf | G3-nf | G3d-nf |
| --- | --- | --- | --- | --- |
| Failing-task runs with a reinforcing lesson among candidates | 10 | 10 | 11 | 10 |
| … and shown to the agent | 10 | 10 | 10 | 10 |

The reinforcing lessons were among the candidates only on TF02 and TF03, where they were shown in 5 of 5 runs in
every arm. The data-profile guide accepted the TF02 coerce lesson because of the very data that refutes it:
"Extremely relevant because 'height_in_cm' and 'international_caps' are listed as strings and contain non-numeric
characters (e.g., '183cm', 'na'), which must be handled before imputation". It accepted the TF03 lesson because it
is "consistent with the prompt's constraints".

## Result 3: memory without a checker did not help, and slightly hurt

Targeted checks on the 4 failing tasks (passes out of 5):

| Check | A0 none | C1 own | E emb | G3 | G3d | C1-nf | E-nf | G3-nf | G3d-nf |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TF01 class distribution | 0 | 5 | 5 | 4 | 2 | 0 | 1 | 0 | 0 |
| TF01 class weights | 0 | 5 | 5 | 5 | 5 | 0 | 0 | 0 | 0 |
| TF01 stratified split | 3 | 5 | 5 | 5 | 5 | 1 | 0 | 1 | 0 |
| TF02 strips the unit | 0 | 4 | 4 | 4 | 3 | 0 | 0 | 0 | 0 |
| TF03 leaves out `column_x` | 0 | 5 | 0 | 1 | 5 | 0 | 0 | 0 | 0 |
| TF03 causal reasoning | 0 | 5 | 0 | 1 | 4 | 0 | 0 | 0 | 0 |
| TF05 leaves out all 4 leaks | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Pooled (of 40)** | **0.075** | **0.75** | **0.475** | **0.50** | **0.60** | **0.025** | **0.025** | **0.025** | **0.0** |

- The difference between each evaluated-store arm and its nf counterpart is significant (p < 0.0001 for C1, E and
  G3).
- C1-nf against no memory is not significant (p = 0.62).
- The only check that got worse than no memory is TF01 stratification (3 of 5 runs, down to 0 or 1).
- On TF02 and TF03 the reinforcing lessons say what the no-memory agent already does. There, they lock in the
  failure rather than cause a new one.
- TF04 scored 1.0 in every arm. TF06 stayed within 0.002 of A0.
- All 120 runs ended with a final answer.

## Predictions (written before the runs)

| # | Prediction | Result |
| --- | --- | --- |
| 1 | At most 2 fix lessons for TF03 + TF05; at most half fix for TF01 / TF02 | **Held.** 0 fix lessons on all 4 tasks. |
| 2 | At least 4 reinforcing lessons for the 4 failing tasks | **Held.** 10. |
| 3 | C1-nf pooled at most 0.25 | **Held.** 0.025. |
| 4 | G3-nf shows a reinforcing lesson in at least half of the runs where one is a candidate | **Held.** 10 of 11. |
| 5 | G3d-nf shows a reinforcing lesson in at least 3 fewer runs than G3-nf | **Failed.** 10 against 10. |
| 6 | TF04 / TF06 within 0.01 of A0 in every nf arm | **Held.** |

## What this means

1. Without an external check, the agent's memory records its silent failures as good practice, in the task's own
   terms ("strict adherence to the instructions"). This is the agreeing wrong lesson from P16, written by the agent
   itself.
2. The trust problem therefore arises without an attacker. An instruction-ranking guide keeps these lessons, as P16
   predicted.
3. The data profile did not help here. The lessons are generic ("convert strings to numbers") and match the prompt.
   The profile even supplied the guide's reason to accept one. A profile helps against a lesson that makes a
   specific claim the numbers contradict, not against a generic procedure that silently fails.
4. All the usefulness in the earlier experiments came from the evaluator's outcome, given to the writer. The store
   is only as good as the check that wrote it.

## Limits

- 1 writing generation from 40 no-memory runs. The store does not update during the agent runs.
- One writer model at temperature 0.
- The labels are mine, made by the pre-set rule before any guide or agent run.
- BLADE and DSBench lessons in the store still had their evaluator outcome.
