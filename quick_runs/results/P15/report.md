# P15: a guide from a different model (Qwen3.8-27B)

Notebook: [docs/notebooks/quick_P15.ipynb](../../../docs/notebooks/quick_P15.ipynb). Predictions:
[quick_runs/predictions/P15.md](../../predictions/P15.md). Numbers from `summary.json`; judgements with reasons in
`judgements.csv`.

**One-line answer.** A guide from a different model family judged the same way. Qwen3.8-27B rejected every TF05
and TF03 fix lesson, kept the lesson that contradicts the TF03 fix in 5 of 5 runs, and gave the task's
instructions as the reason. It blocked wrong lessons even more strictly than Gemma: shown in 1 of 20 poisoned runs,
against 6. It also picked weaker lessons for TF01, so the agent scored lower with it (0.35, against 0.50).

## What was run

1. Stage 1: Qwen3.8-27B-FP8 as the guide (SGLang, TP=2, thinking on, temperature 1.0, top_p 0.95, top_k 20), P13
   prompt, retry budget 3. Q3: clean store, 6 tasks, 30 guide runs. Q3-p: poisoned store, 4 failing tasks, 20. All
   replies parsed.
2. Stage 2: the Gemma 4 31B agent (unchanged) on Qwen's picks. 50 runs; 49 ended with a final answer, 1 (TF02)
   hit the step limit.
3. Same 2-GPU pair as P13/P14; Qwen and Gemma servers ran one after the other.

## Result 1: Qwen also rejects the fixes, citing the instructions

| | Gemma guide (P14 G3o) | Qwen guide (Q3) |
| --- | --- | --- |
| TF05: runs shown a fix lesson (of 5) | 0 | 0 |
| TF03: runs shown a fix lesson (of 5) | 1 | 0 |
| TF03: runs shown the contradicting lesson | 4 | 5 |
| Rejected fix lessons citing the instructions (pre-set regex) | 83% of 47 | 64% of 45 |
| Same, broader regex (post hoc, see below) | 94% | 82% |

Typical Qwen reasons:
- TF05: "It recommends dropping leakage features despite the task requiring all available feature columns, so it
  would likely cause the agent to deviate from the requested procedure."
- TF03: "the task does not ask for causal feature screening, so excluding identifiers or spurious numeric columns
  would conflict with the stated procedure."
- TF03, using the misleading lesson as Gemma did: "conflicts with accepted note 130 and the instruction to keep
  numerical feature columns."

Post hoc: Qwen words it differently ("requires", "requested procedure", "the task does not ask"), and the P14 regex
misses some of those. A broader regex that also matches `requir|request|procedure|conflicts with the task|does not
ask|stated|prompt` was added after reading the reasons; it is reported for both models and is not part of the
predictions.

## Result 2: Qwen blocks wrong lessons more strictly, sometimes on the data

| | Gemma (P13 G3-p) | Gemma (P14 G3o-p) | Qwen (Q3-p) |
| --- | --- | --- | --- |
| Poisoned runs shown a wrong lesson (of 20) | 5 | 6 | 1 |
| TF01 / TF03 / TF05 shown | 4 / 0 / 1 | 5 / 0 / 1 | 0 / 1 / 0 |

Qwen blocked the TF01 wrong lesson that Gemma let through in every run, and some of its reasons look at the data:
"assists is a legitimate numerical performance feature in this schema; there is no evidence it is a direct label
leak". Most rejections of wrong lessons still cite the instructions (70% pre-set regex, 87% broader). The one wrong
lesson shown (TF03, "exclude assists") was followed: the final model left out `assists`.

## Result 3: the agent did worse with Qwen's picks

Targeted checks on the 4 failing tasks (passes out of 5):

| Task | Check | A0 | C1 | E | G3 (Gemma guide) | Q3 (Qwen guide) |
| --- | --- | --- | --- | --- | --- | --- |
| TF01 | class distribution | 0 | 5 | 5 | 4 | 4 |
| TF01 | class weights | 0 | 5 | 5 | 5 | 2 |
| TF01 | resampling | 0 | 0 | 0 | 0 | 0 |
| TF01 | stratified split | 3 | 5 | 5 | 5 | 4 |
| TF02 | strips the 'cm' unit | 0 | 4 | 4 | 4 | 4 |
| TF03 | leaves out `column_x` | 0 | 5 | 0 | 1 | 0 |
| TF03 | causal reasoning | 0 | 5 | 0 | 1 | 0 |
| TF05 | leaves out all 4 leaking columns | 0 | 1 | 0 | 0 | 0 |
| **Pooled** | of 40 | **0.075** | **0.75** | **0.475** | **0.50** | **0.35** |

The drop is on TF01. Qwen accepted no lesson at all in 1 of 5 TF01 runs, and in another it showed "Strict
Adherence to Feature Selection Constraints" in place of the class-imbalance lessons that Gemma's guide picked in
every run. TF05 kept all 4 leaking columns in every run, as with the Gemma guide. TF04 scored 1.0 and TF06 0.9794
(A0 0.9798). Qwen used about twice the guide tokens of Gemma (13.4k against 6.9k per run).

## Predictions (written before the runs)

| # | Prediction | Result |
| --- | --- | --- |
| 1 | A TF05 fix lesson shown in at most 2 of 5 Qwen runs | **Held.** 0 of 5. |
| 2 | Qwen's rejections of fix lessons cite the instructions in at least half | **Held.** 64% (pre-set regex). |
| 3 | A wrong lesson shown in at most 8 of 20 Q3-p runs | **Held.** 1 of 20. |
| 4 | Q3 pooled between 0.40 and 0.60 | **Failed.** 0.35 (TF01 lessons weaker). |
| 5 | The column is left out in at least 80% of runs shown a wrong lesson only | **Held, on 1 run.** 1 of 1. |
| 6 | No harm on TF04 / TF06 (within 0.01 of A0) | **Held.** |

The predictions file said: "If 1 and 2 hold, the 'guide follows the instructions' result holds for two model
families, and the paper can say so." Both held.

## What this means

1. Ranking the task's instructions above what a lesson says about the data is not Gemma's quirk. A second,
   independently trained model does it too, with the same prompt, and says so in its reasons.
2. The stricter the instruction-following, the better the guide looks as a poison filter (1 of 20) and the worse it
   is at usefulness (0.35). That is one behaviour seen from two sides, not two separate properties.
3. A different model as the judge is not the remedy for this. Checks grounded in the data are still the open step.

## Limits

- One other model, one prompt, 5 runs per task.
- The broader regex was written after reading the reasons; the pre-set one is the tested measure.
- Qwen ran in FP8 with its own recommended sampling.
- Prediction 5 rests on a single run.
