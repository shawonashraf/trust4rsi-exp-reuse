# P1b divergence census on DSBench (ground-truth tasks, no memory, no hint)

Notebook: [docs/notebooks/quick_P1b.ipynb](../../../docs/notebooks/quick_P1b.ipynb). Predictions:
[quick_runs/predictions/P1b.md](../../predictions/P1b.md). Numbers are from `summary.json`; per-run tables in
`analysis.csv`, `modeling.csv` and `behaviour.csv`.

## What was run

34 tasks x 5 runs = 170 runs. Same agent, prompt style, sampling (temperature 1.0) and 25-step budget as P1.
- 26 DSBench data-analysis questions (Excel workbooks from the ModelOff financial-modelling contest; 20
  option-letter answers, 6 whole-number answers). Score: exact match (numbers within 1).
- 8 DSBench data-modelling tasks (tabular Kaggle competitions). Score: the *relative gap*, the share of the
  distance from DSBench's baseline score to the perfect score that the run closes. 0 = baseline, 1 = perfect.

How the runs ended: 166 with a final answer, 3 at the step limit (all on question 00000029/1), 1 with a
server error (00000020/35, seed3; HTTP 400 from SGLang). Found later (P12 setup): that server's KV cache held
only 23,840 tokens in total (all requests together), far below its 64k window, so a long run could exceed it. The
largest P1b run peaked at about 23.8k tokens. The error is most likely this limit.

**A benchmark label error.** DSBench's answer key for competition 00000020 is shifted. Question 35 is a letter
question (options A to I), but its stored gold answer is the number 14978914492, which belongs to a later
question. So question 35 cannot be scored. All 4 runs that finished answered "D". Accuracy is given below
without it (25 questions). The raw table in the notebook keeps it.

## Result 1: analysis questions are mostly right, and when runs agree they are right

1. **Accuracy 0.896** over 25 questions (125 runs). Letter questions 0.89 (20 questions), whole-number
   questions 0.92 (5 questions).
2. **Answer consistency 0.844** (share of run pairs with the same answer). Letter 0.845, whole-number 0.84.
3. **18 questions got the same answer in all 5 runs. All 18 were right.** No question got the same wrong
   answer in all 5 runs. So, unlike Silent Failures in P1, there are no consistent failures here.
4. The wrong answers are scattered single runs, with three exceptions:
   - 00000043/1: answers C, C, A, A, B (gold B). Only 1 of 5 right.
   - 00000017/29: answers C, E, E, E, C (gold E).
   - 00000029/1: 3 runs hit the step limit. They had computed 4,829.07 (option A, the right answer). But the
     question says "Something isn't working" in the sheet, so they kept looking for the error and never
     answered. The 2 runs that answered chose A.

## Result 2: modelling runs all submit, and the scores barely move between runs

1. **40 of 40 runs wrote a valid submission.**
2. **Score spread is small.** The standard deviation of the relative gap across the 5 runs is above 0.02 on
   only 1 of 8 tasks (playground-series-s3e12: 0.021, range 0.69 to 0.74). On the other 7 it is 0.0003 to
   0.009.
3. Mean relative gap per task ranges from 0.49 (s3e5) to 0.94 (bike-sharing-demand).
4. **The model is almost always XGBoost** (30 of 40 runs). Random forest in 6 runs (4 of them on s3e12).
5. **Validation is where modelling runs split** (agreement 0.60). They choose between k-fold, stratified
   k-fold and a holdout split. bike-sharing-demand used a holdout in all 5 runs; s4e3 used three different
   schemes across its 5 runs.
6. **bike-sharing-demand:** no run used `casual` or `registered` as features (they add up to the target and are
   missing from the test file). 3 runs dropped them explicitly; 2 never mentioned them. (The extractor listed
   `registered` as a variable in one run; the code shows it was dropped. An extractor mistake.)

## Result 3: decision agreement

Mean over tasks, lowest first:

| Decision type | All 34 | Analysis (26) | Modelling (8) |
| --- | --- | --- | --- |
| transformation | 0.21 | 0.22 | 0.19 |
| final_selection | 0.25 | 0.25 | 0.26 |
| variables | 0.29 | 0.19 | 0.49 |
| framing.unit | 0.54 | 0.51 | 0.64 |
| framing.target | 0.60 | 0.43 | 0.94 |
| framing.population | 0.79 | 0.73 | 1.00 |
| metric | 0.82 | 0.78 | 0.92 |
| validation | 0.91 | 1.00 | 0.60 |
| model.family | 0.92 | 1.00 | 0.71 |

1. The decision types were built for modelling. On the analysis questions most of them do not fit: there is
   no model ("other"), no validation ("none"). There, the target, unit and variables are free text such as
   "Total volume for low value widgets" or "sheet:Inputs". So low agreement on analysis questions is largely
   wording, not different choices. 18 of 26 questions had all 5 answers identical while decision agreement
   stayed low.
2. On questions where runs disagreed on the answer, agreement was lower on target (0.12 vs 0.51), population
   (0.49 vs 0.84) and variables (0.00 vs 0.25) than on questions where they agreed. This fits "they read
   different cells or set up the calculation differently", but with 8 such questions and free-text values it
   is a weak signal.
3. On modelling tasks, runs agree on target and population. They differ on the feature set, transformations
   and validation. First divergence was framing.unit on 4 of 8 tasks.

## Result 4: how the agent works

| | Analysis (130 runs) | Modelling (40 runs) |
| --- | --- | --- |
| Steps per run (mean, median, max) | 6.2, 4, 25 | 6.9, 7, 11 |
| Code cells per run (mean) | 4.2 | 5.7 |
| Cells that raised an error, per run | 0.2 | 0.8 |
| Decision types stated / made silently, per run | 3.6 / 8.0 | 8.5 / 1.0 |

1. On analysis questions at least a quarter of runs answer in 2 steps (one code cell). Others read the
   workbook for 10 to 20 cells.
2. On modelling tasks it states nearly every decision it makes (only 1 silent type per run). The modelling
   prompt asks for a pipeline, so the decision list fits.
3. Cell errors in modelling are mostly library API mismatches (for example, `early_stopping_rounds` passed to
   `XGBRegressor.fit`), fixed in the next cell.
4. Wall-clock minutes (41 and 38 per run) are inflated: up to 16 runs shared each server. They are not a
   cost estimate.

## Against the predictions

| Prediction | Outcome |
| --- | --- |
| 1. Analysis accuracy below 0.5; letters better than whole numbers | Wrong. Accuracy 0.896. Whole-number questions (0.92) did slightly better than letters (0.89). |
| 2. Answer consistency higher on letters than on whole numbers | Held only weakly: 0.845 vs 0.84. |
| 3. At least 2 of 26 questions with the same wrong answer in all 5 runs | Wrong: 0. (The only all-wrong question is the label error.) |
| 4. Disagreeing runs differ mostly in transformation | Not supported. They differ in target, population and variables. The values are free text, so this is weak either way. |
| 5. At least 80% valid submissions | Held: 100%. |
| 6. Relative gap SD above 0.02 on at least half the tasks | Wrong: 1 of 8. |
| 7. Lowest agreement at variables and transformation; family in the middle; validation varies | Held. Transformation 0.19, variables 0.49, family 0.71, validation 0.60. |
| 8. bike-sharing: at least one run tries `casual`/`registered` | Wrong. No run used them. |

## What it suggests (for the author's decision)

1. **DSBench tasks, as selected, are too easy for this model to show the memory problem.** Accuracy is near
   0.9, there are no consistent failures, and modelling scores barely vary. A memory has little to fix here.
   Silent Failures (P1), with its planted pitfalls, is the better test bed.
2. **Contamination is a likely part of the reason.** ModelOff questions and answers are published, and the 8
   Kaggle competitions have public notebooks. P10 did not test DSBench recall; this was not checked.
3. **One useful behaviour pattern.** On 00000029/1 the agent found the right number, but the question's
   hint that "something is wrong" sent it searching until it ran out of steps. Text in the prompt can override
   the agent's own correct result. This is a small, concrete case of the agent weighing prompt text over its
   own evidence, which is the opposite of "ignoring memories". It is worth a closer look in P3/P4.
4. If DSBench is used again, take harder questions: the ones DSBench reports as low-accuracy, or the
   remaining questions of the competitions where runs split here (00000043, 00000017).
5. Caveats. 5 runs per task. 25 scorable analysis questions. One DSBench label error found by hand; others may
   exist. The decision types fit modelling tasks, not workbook questions.
