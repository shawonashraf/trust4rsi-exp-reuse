# P1b predictions: divergence census on DSBench (ground-truth tasks)

Written 2026-11-11, before any P1b run. Added on the author's request: hypothesis-search-style tasks have
no ground truth, so the baseline should rest on tasks with known answers.

Run: 34 tasks x 5 runs = 170 runs, same agent, sampling and step budget as P1.
- 26 DSBench data-analysis questions (20 option-letter answers, 6 whole-number answers), selected by
  `quick_runs/select_dsbench_analysis.py`. Score: exact match (letters equal; numbers within 1).
- 8 DSBench data-modelling tasks (tabular Kaggle, under 20 MB), selected by
  `quick_runs/select_dsbench_modeling.py`. Score: the competition metric on held-out answers, turned into
  DSBench's relative gap, (score - baseline) / (perfect - baseline). Checked: submitting the held-out
  answers gives the perfect score on all 8.

## Expected direction

Analysis questions (judge-free answers):
1. Accuracy is low: below 0.5 pooled. Letter questions do better than whole-number questions.
2. Answer consistency (share of run pairs with the same answer) is higher on letter questions than on
   whole-number questions.
3. Some questions get the same wrong answer in all 5 runs (consistent failures). At least 2 of 26.
4. Runs that disagree on the answer differ mostly in transformation (how the workbook is read and the
   calculation set up); decision types like model.family mostly do not apply.

Modelling tasks:
5. Most runs write a valid submission (at least 80% of the 40 runs).
6. Score spread is real: on at least half of the 8 tasks, the SD of the relative gap across the 5 runs is
   above 0.02.
7. Lowest decision agreement at variables (feature sets) and transformation; model.family agreement in
   the middle (runs pick among xgboost / lightgbm / random forest / gradient boosting); validation varies
   (holdout vs k-fold).
8. bike-sharing-demand: train.csv has `casual` and `registered` (they sum to the target) but test.csv
   does not. At least one run tries to use them and has to recover, or fails.
