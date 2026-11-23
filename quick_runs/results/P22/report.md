# P22: review round 6 controls, bias or prompt obedience?

Notebook: [docs/notebooks/quick_P22.ipynb](../../../docs/notebooks/quick_P22.ipynb). Predictions (committed before the
runs): [quick_runs/predictions/P22.md](../../predictions/P22.md). Numbers come from `summary.json`; the fix-lesson
reasons are in `fix_reasons.csv`.

**One-line answer.** The guide is not loosened by cutting any instruction, only the feature one: a heavy placebo that
removes a different, contentful instruction leaves TF03 fix exposure at 0 of 5, against 15 of 15 for the feature cut.
And when the guide is given both permission to deviate (the open prompt) and the data to judge against (the profile),
it accepts the TF03 fix in 5 of 5 runs. So it is better described as deferring to the instructions over a claim it
cannot check than as a blanket bias: given evidence, it acts on the evidence.

## Terms

- **G3s-pl2, heavy placebo.** The guide's task text with a contentful non-feature instruction cut: on TF03 the model
  family ("build a linear regression model" to "build a regression model"); on TF05 "fill missing values with 0".
  Retrieval and the agent keep the full task. Compare with the feature cut (G3s) and the light placebo (G3s-pl).
- **G3o-d, open prompt + profile.** The P14 open-prompt criterion ("a note can be useful even when it goes against
  the wording, if the data suggests…") plus the G3d data profile. The guide may deviate and has data to judge against.
- Fix lessons, keep-`column_x` lessons, pooled: as in P16-P18.

## What was run

40 guide runs (G3s-pl2 on TF03/TF05, 5 each; G3o-d on the 6 tasks, 5 each) and 30 G3o-d agent runs, scored as in P13.
29 of 30 agent runs ended with a final answer. Two TP=2 Gemma servers on GPUs 4-5 and 2-3; sandbox temp on `/data`.

## Result 1: only the feature instruction loosens the guide

Runs (of 5) shown a fix lesson:

| Task | G3 (no cut) | G3s (feature cut) | G3s-pl (light placebo) | G3s-pl2 (heavy placebo) |
| --- | --- | --- | --- | --- |
| TF03 | 1 | 15 of 15 | 1 | **0** |
| TF05 | 0 | 14 of 15 | 4 | **2** |

- TF03: the heavy placebo shows the fix in 0 of 5, the same as no cut; the feature cut shows it in 15 of 15 (Fisher
  p = 0.0001). Removing the model-family instruction does not loosen the guide.
- 74% of the heavy placebo's fix rejections still cite the instructions, for example "the task provides a specific
  instruction to 'Keep only numerical feature columns'". The feature sentence is still present, so it still governs.
- The 2 TF05 runs that showed a fix accepted lesson 168, which itself tells the agent to ignore the "use all"
  instruction, as in the light placebo.

## Result 2: given data, the open-prompt guide accepts the fix

Runs (of 5):

| | G3n (open, no data) | G3d (profile, original prompt) | G3o-d (open + profile) |
| --- | --- | --- | --- |
| TF03 fix shown | 0 | 5 | **5** |
| TF03 keep-`column_x` shown | 5 | 0 | **0** |
| TF05 fix shown | 2 | 0 | 0 |
| Pooled targeted (of 40) | 0.45 | 0.60 | **0.675** |

- The open prompt alone (G3n) showed no TF03 fix and the keep-`column_x` lesson in all 5 runs: with permission to
  deviate but no data, the guide could not tell the fix from the misleading lesson. Adding the profile flips both:
  it accepted the fix in 5 of 5 ("the data contains a blatant spurious correlation ('column_x'); this note
  encourages the agent to reason about the features") and the agent left `column_x` out (TF03 RMSE 4.77e6).
- G3o-d pooled 0.675, the best of any guide arm. Against G3d (profile, original prompt) p = 0.64 (the same); against
  G3n (open, no data) p = 0.07. The gain over the open prompt alone is TF03.
- TF05 is still missed (0 of 5): the leak does not show in the profile, so neither permission nor data helps.

## Predictions (written before the runs)

| # | Prediction | Result |
| --- | --- | --- |
| 1 | Heavy placebo: fix on TF03 + TF05 at most 3 of 10; at least 60% of rejections cite instructions | **Held.** 2 of 10; 74%. |
| 2 | G3o-d: TF03 fix at least 4 of 5, keep-`column_x` at most 1 of 5 | **Held.** 5 of 5, 0 of 5. |
| 3 | G3o-d: TF05 fix at most 1 of 5 | **Held.** 0 of 5. |
| 4 | G3o-d pooled at least 0.55 | **Held.** 0.675. |
| 5 | TF04 / TF06 within 0.01 of A0 in G3o-d | **Held.** |

5 held, 0 failed. Running total across P1-P22: 53 held, 20 failed, 1 not testable (of 74).

## What this means

1. The one-sentence result is specific to the feature instruction. A placebo that removes a different instruction of
   real weight does not reproduce it, so the guide is not simply loosened by any missing instruction.
2. The guide's behaviour is defer-to-instructions-over-unverifiable-claims, not a blanket bias. With the open prompt
   and no data it kept the misleading lesson; with the open prompt and the profile it took the fix in every run. It
   obeys the instructions when it cannot check a lesson's claim, and acts on the data when it can.
3. This refines the mechanism rather than overturning it: a usefulness judgment by an instruction-tuned model, with
   no data, ranks the instructions above a lesson's claim about the data. Where the leak does not show in the data
   (TF05), neither permission nor evidence helps.

## Limits

- 5 runs per cell; the heavy placebo rests on 2 tasks; the cut sentences are my choice.
- The profile includes test-set numbers (an upper bound), as in P16.
- One guide model (Gemma); the frontier-model guide is P20.
