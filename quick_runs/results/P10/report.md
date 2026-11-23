# P10 recall probe

Notebook: [docs/notebooks/quick_P10.ipynb](../../../docs/notebooks/quick_P10.ipynb). Predictions:
[quick_runs/predictions/P10.md](../../predictions/P10.md). Numbers below are from `summary.json`.

## What was run

gemma-4-31B-it got each task's own prompt (without the schema and without the data) and was asked what
the result is likely to be and which model and controls are standard. 15 tasks x 3 samples = 45 replies,
agent sampling (temperature 1.0, thinking on). I labelled each reply by hand with the rule fixed in the
predictions (`labels.csv` has a quote or reason per reply).

## Result

1. 6 of 15 tasks are recall hits (at least 2 of 3 replies hit): blade_crofoot, blade_hurricane,
   blade_teachingratings, sf_tf01_goal_imbalance, sf_tf06_metric_select_leakfree, sf_tf06_metric_select_cue.
2. 4 tasks moved to high risk: blade_crofoot, sf_tf01, both TF06 tasks.
3. After P10: 8 tasks are high risk, 5 medium, 2 low. The 7 medium/low tasks are sf_tf02, sf_tf03,
   sf_tf04 (both), sf_tf05, blade_boxes and blade_reading. Later probes report results split by this tag.

## What the hits are made of

The hits are not all recall of the dataset. There are three kinds:

1. **The prompt names the study.** For crofoot, hurricane and teachingratings, BLADE's own dataset
   description names the paper. The model repeats the name and the published finding (for example
   "location can offset size" for crofoot, the Simonsohn reanalysis for hurricane).
2. **The task wording implies the pitfall.** For TF01 and the leak-free TF06, the model guesses that goal
   scorers are a minority, so the label is imbalanced. It infers this from the words "goals_label" and
   "high scorers", not from knowing these files.
3. **The file name gives it away.** For TF06-cue, all 3 replies say the data is imbalanced "because the
   dataset is explicitly labeled as 'imbalance'". For TF04-cue, 1 of 3 replies says the classes follow
   torque = force x distance; its leak-free twin got 0 of 3.

## Against the predictions

| Prediction | Outcome |
| --- | --- |
| 1. High BLADE tasks mostly hit | Partly. hurricane and teachingratings 3/3. caschools and fish 0/3: the model calls each "a classic textbook dataset" but never names it. For caschools it predicts the class-size effect vanishes with controls, which is not the textbook result. |
| 2. Medium BLADE: at most one moves to high | Held. crofoot moved (the prompt names the paper); boxes and reading 0/3. |
| 3. Few SF hits; *_cue more than leak-free | Partly. TF04: cue 1/3, leak-free 0/3, as predicted. TF06: both 3/3, because even the leak-free wording implies rare goal scorers. |
| 4. TF05 most likely SF hit | Wrong. 0/3. No reply named the expenditure/share leakage. |

## What it suggests

1. Recall of the dataset itself is rare for these tasks. The bigger issue is that the prompt or file
   name carries the answer. That matters for memory studies: a "memory" that repeats what the prompt
   already says cannot be told apart from the prompt.
2. TF02, TF03 and TF05 have no recall and no cue. They are the cleanest tests of whether the agent finds
   a planted problem by looking at the data.
3. The TF04 and TF06 cue/leak-free pairs measure how much a file-name cue moves the agent. P1 reports them
   side by side.
