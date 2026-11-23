# P1 divergence census (no memory, no hint)

Notebook: [docs/notebooks/quick_P1.ipynb](../../../docs/notebooks/quick_P1.ipynb). Predictions:
[quick_runs/predictions/P1.md](../../predictions/P1.md). Numbers are from `summary.json`; per-run tables in
`checks.csv`, `forks.csv` and `behaviour.csv`.

## What was run

15 tasks x 5 runs = 75 runs. No memory, no hint, temperature 1.0 (the server ignores the seed, so the runs
differ by sampling and serving noise). All 75 runs ended with a final answer.

Terms:
- *Decision agreement* at a decision type: the share of run pairs that executed the same value (from the
  post-hoc extractor, reading the code). 1.0 = all 5 runs did the same thing.
- *Answer consistency*: the share of run pairs with the same final answer. For Silent Failures, the test
  metric to 3 significant digits (TF06: the selection metric and the chosen model). For BLADE, the direction
  and the yes/no verdict.
- *Consistent failure*: all 5 runs fail the same check.

Silent Failures (SF) is the main result: it has a known right way to handle each task. BLADE is secondary:
it has expert annotations but no single right answer (the author, 2026-11-11).

## Result 1: on Silent Failures, the runs mostly agree, and they agree on the wrong thing

**Answer consistency on SF is 0.81 pooled.** 6 of the 8 SF tasks give the same answer in all 5 runs. The
two exceptions:
- TF01 (0.1): the 5 runs report macro F1 between 0.646 and 0.795. They pick different feature columns.
- TF06-cue (0.4): all runs select on ROC AUC, but they split between XGBoost (3) and random forest (2).

**8 checks fail in every run** (all 5 of 5):

| Task | Planted problem | What all 5 runs did |
| --- | --- | --- |
| TF01 | 5% positive label | Never looked at the class distribution, no class weights, no resampling (and 4 of 5 no stratified split) |
| TF02 | heights like '185cm' | Turned them into missing values instead of stripping 'cm' (all 5 runs report the identical RMSE) |
| TF03 | spurious `column_x` | Kept it as a feature; no causal reasoning about features |
| TF05 | 4 columns leak the label | Used them (test accuracy 0.966 in every run) |

What the agent did get right: TF02's 'na' strings were handled in 5 of 5 runs. TF04 built torque features in
all 10 runs (both the leak-free and the cue variant). TF06 selected on ROC AUC, not accuracy, in every run the
judge could read.

The judge (gemma-4-31B, 3 draws per run) gave the same verdict in all 3 draws for 79 of 80 run-checks.

## Result 2: where runs differ, it is in the details, not the main choices

Decision agreement, mean over tasks (lowest first):

| Decision type | All 15 | SF (8) | BLADE (7) |
| --- | --- | --- | --- |
| transformation | 0.27 | 0.29 | 0.25 |
| model.controls | 0.29 | - (no controls in SF) | 0.29 |
| final_selection | 0.35 | 0.64 | 0.01 |
| variables | 0.52 | 0.49 | 0.56 |
| framing.unit | 0.73 | 0.79 | 0.67 |
| model.family | 0.83 | 0.83 | 0.83 |
| framing.target | 0.85 | 1.00 | 0.69 |
| framing.population | 0.90 | 1.00 | 0.79 |
| verdict | 0.94 | - | 0.94 |
| metric | 0.95 | 0.91 | 1.00 |
| validation | 0.97 | 0.95 | 1.00 |

1. Runs differ most in the small steps (transformation, the feature list, the control set). They mostly agree
   on the model family, the validation scheme and the metric.
2. transformation and final_selection are free-text lists, so some of that disagreement is wording (see the
   hand check). variables and model.controls are column lists and are more reliable.
3. First divergence (the first decision type where two runs differ) is `variables` in 7 of 15 tasks, then
   framing.target (3), framing.unit (2), transformation (2), framing.population (1).
4. 62 forks in total (a fork = a task and decision type where runs executed more than one value). Main-choice
   forks are few, for example: fish model family (negative binomial 4, Poisson 1), teachingratings (OLS 4,
   mixed effects 1), TF04-cue (random forest 2, decision tree 2, other 1), TF01 validation (holdout 4, given
   test set only 1).

## Result 3 (secondary): BLADE

- Answer consistency 0.86 pooled. 5 of 7 datasets give the same direction and verdict in all 5 runs.
  Exceptions: reading (0.4: "positive" in 2 runs, "no effect" in 3) and caschools (0.6: one run says "negative
  effect" but "not supported").
- crofoot: all 5 runs concluded "no effect, not supported". In P9 the 4 greedy runs of the same task reached
  3 different conclusions. So agreement here does not mean the answer is stable.
- Expert match: the outcome, main predictor and model family match an expert choice in nearly every run. Two
  exceptions. fish: 4 of 5 runs used negative binomial, which no expert listed. reading: all 5 runs used
  `speed` / `log_speed` as the outcome, while the experts built it from `adjusted_running_time` and
  `num_words`. That is likely the same quantity, so this "consistent failure" is probably a limit of the
  column matching, not a wrong choice.

## Result 4: how the agent works (behaviour)

| | SF (40 runs) | BLADE (35 runs) |
| --- | --- | --- |
| Steps per run (mean, min-max) | 3.2 (2-6) | 6.1 (3-10) |
| Code cells per run | 2.2 | 4.9 |
| Cells that raised an error, per run | 0.08 | 0.37 |
| Model families fitted per run | 1.8 | 1.3 |
| Minutes per run (mean, max) | 3.8 (15.3) | 17.4 (26.9) |

1. On SF the agent writes the whole pipeline in one or two cells and answers. It rarely looks at the data
   first. That is how the planted problems slip through: nothing in its process would show it a 5% label,
   a '185cm' string or a leaking column.
2. Decisions made but never announced, per run: 3.0 decision types on SF, 1.5 on BLADE. Most often silent:
   validation (28 runs), framing.unit (27), framing.population (26), transformation (25), metric (18).
3. Stated vs executed model family: 0 mismatches in 75 runs. (A first count gave 4; all 4 were my keyword map
   reading a list of candidate models, or "xgboost regression", as a different family. Fixed and tested.)

## Against the predictions

| Prediction | Outcome |
| --- | --- |
| 1. Disagreement concentrates; lowest at transformation, controls, variables; highest at target | Held. |
| 2. At least half of tasks first diverge at one of the first two stated types | Wrong: 4 of 15. Runs agree on the framing they state first and split later, at variables. |
| 3. Consistent failures on TF03 and TF05; TF01 resampling/class weights mostly fail; TF02 split | Held for TF03, TF05, TF01. Wrong for TF02: no split, all 5 runs failed the unit suffix the same way. |
| 4. *_cue variants fail less than leak-free on TF04 and TF06 | Not testable: both variants passed in every scorable run. |
| 5. BLADE answer consistency at least 0.6 while controls agreement is low | Held: 0.86 vs 0.29. |
| 6. SF numeric answers agree less often than BLADE verdicts | Held only weakly (0.81 vs 0.86), driven by TF01 and TF06-cue. |
| 7. Stated vs executed mismatch on at least one type per task | Wrong for model.family (0 of 75). Other types not measured this way. |

## What it suggests (for the author's decision)

1. This is the "little disagreement, many consistent failures" case in quick_runs.md §5. On the
   ground-truth tasks, the problem is not runs disagreeing; it is runs agreeing on a wrong choice. That points
   to reframing the memory around consistent failures (G08) rather than forks (G05).
2. A memory that only makes runs agree would add little here. To help, it has to change the choice: get the
   agent to look at the class balance, strip the unit, drop the spurious or leaking column. P2 (oracle hint)
   tests whether a correct hint does that. P4 (wrong hint) and P3 (hint at the decision vs in the prompt) test
   whether the agent follows it at all. Together they test the "agents ignore memories" observation.
3. The agent's process (whole pipeline in one cell, no look at the data) is part of the failure. A check run on
   the data before the model is fit (P3 mode 3, `before_execute`) may matter more than any prompt text.
4. Caveats. 5 runs per task. The SF checks are judged by the same model that ran the agent (it agreed with
   itself in 79 of 80 run-checks). The BLADE expert match is approximate (see reading). P9 showed greedy
   reruns differ too, so part of any disagreement is serving noise.
