# P18: review round 3 controls

Notebook: [docs/notebooks/quick_P18.ipynb](../../../docs/notebooks/quick_P18.ipynb). Predictions (committed before the
runs): [quick_runs/predictions/P18.md](../../predictions/P18.md). Numbers come from `summary.json`; every guide judgement
and its reason is in `judgements.csv`.

**One-line answer.** It is one sentence. Hiding only the feature sentence from the guide makes it show the fix lessons
in 10 of 10 runs, and a placebo cut does not (2 of 10). Random choice among the guide's own candidates, and a
cross-encoder with no instructions, both show the fixes far more often than the guide. The data profile's TF03 gain
needs the test-set numbers: with training numbers only, the guide reads `column_x`'s +1.00 correlation as a reason to
keep it.

## Terms

- **Fix lessons, keep-`column_x` lessons, wrong lessons, shown, pooled:** as in P16.
- **R3 / R3-p:** 3 lessons drawn at random from the candidates the P13 G3 / G3-p guide saw in the same run.
- **G3s:** the guide's task text without the feature sentence only. **G3s-pl:** placebo, an unrelated sentence cut.
- **G3d-tr:** the P16 data profile with every test-set number removed.
- **G3h:** the guide sees the suite's first hint-level wording; retrieval and candidates as for G3. Guide only.
- **X3:** embedding top-20 re-scored by a cross-encoder (ms-marco-MiniLM-L-6-v2, not instruction-tuned), top 3.

## What was run

210 guide-stage runs (110 with the guide; 100 without a model), then 170 agent runs, scored as in P13. 168 ended with
a final answer; 1 R3 run hit the step limit, and no other arm did. Two TP=2 Gemma servers on GPUs 4-5 and 2-3 (idle at
launch). Wall time 1 h 21 min.

## Result 1: one sentence flips the guide; a placebo cut does not

| Runs (of 5) shown a fix lesson | G3 | G3t (all wording hidden) | G3s (feature sentence hidden) | G3s-pl (placebo) |
| --- | --- | --- | --- | --- |
| TF03 | 1 | 5 | **5** | 0 |
| TF05 | 0 | 5 | **5** | 2 |
| Rejected fix lessons whose reason cites the instructions | 83% (P14) | 16% | **8% of 36** | 79% of 48 |

- G3 against G3s: 1 against 10 of 10 (Fisher p = 0.0001). G3s against G3s-pl: 10 against 2 (p = 0.0007).
- With the sentence hidden, the guide rejected fix lessons only as redundant with one it had accepted. With the
  placebo cut, it still wrote "the task prompt explicitly instructs to 'Keep only numerical feature columns', which is
  a structural constraint that overrides…".
- The 2 placebo runs that showed a TF05 fix both accepted lesson 168, which itself says to ignore the "use all"
  instruction.

## Result 2: the guide's judgement adds no usefulness; its only effect is to block contradicting lessons

| | E (top-3) | G3 (guide) | R3 (random of G3's candidates) | X3 (cross-encoder) |
| --- | --- | --- | --- | --- |
| Fix lesson shown, TF03 + TF05 (of 10) | 10 | 1 | **9** | **10** |
| Keep-`column_x` shown, TF03 (of 5) | 5 | 4 | 2 | **0** |
| Pooled targeted checks (of 40) | 0.475 | 0.50 | 0.425 | 0.575 |
| P13 wrong lessons shown (of 20) | 10 (E-p) | 5 (G3-p) | 10 (R3-p) | |
| Agreeing wrong lessons shown (of 20) | 20 (E-pc) | 20 (G3-pc) | | 15 (X3-pc) |

- G3 against R3, fix lessons shown: 1 against 9 of 10 (p = 0.001). Pooled: 0.50 against 0.425 (p = 0.65).
- G3-p against R3-p, P13 wrong lessons: 5 against 10 of 20 (p = 0.19). The direction matches the guide filtering
  lessons that contradict the wording, but it is not significant at 20 runs.
- X3 kept the fix and dropped the keep-`column_x` lesson in every TF03 run, and the agent then left `column_x` out
  in 5 of 5 (RMSE 4.77e6, the same as all own lessons). It lost TF02, where the cross-encoder put no unit lesson in
  the top 3 (0 of 5). Pooled 0.575 against 0.475 for E (p = 0.50).

## Result 3: the agent follows a fix only when no lesson contradicts it

All 50 TF03 agent runs from 10 arms (E, G3, G3t, G3d, R3, R3-p, G3s, G3s-pl, G3d-tr, X3), by what was shown:

| Shown | Runs | `column_x` left out |
| --- | --- | --- |
| A fix lesson, no keep-`column_x` lesson | 19 | **19** |
| A fix lesson and a keep-`column_x` lesson | 16 | 0 |
| A keep-`column_x` lesson, no fix lesson | 15 | 0 |

This is why G3s showed fixes in 10 of 10 runs but passed TF03 in only 2 of 5: in 3 runs it also showed lesson 130.

## Result 4: the TF03 profile gain needs test-set numbers

| | G3d (full profile) | G3d-tr (train only) |
| --- | --- | --- |
| TF03 fix lesson shown (of 5) | 5 | **0** |
| TF03 keep-`column_x` shown (of 5) | 0 | **5** |
| TF03 `column_x` left out (of 5) | 5 | 0 |
| TF01 / TF02 checks (class weights, stratified split, unit) | 5 / 5 / 3 | 5 / 5 / 5 |
| Agreeing wrong lessons shown, TF01 / TF02 / TF03 / TF05 | 0 / 0 / 3 / 5 | 0 / **5** / 5 / 5 |
| Pooled, clean store / agreeing store | 0.60 / 0.475 | 0.40 / 0.325 |

- With training numbers only, the guide accepted lesson 130 because "the data profile shows 'column_x' has a 1.0
  correlation with the label". The train/test contrast (+1.00 against −0.04) was what it used in P16.
- TF01 still works from training numbers: "the data profile shows the label is highly imbalanced (94.9% vs 5.1%)".
- TF02 did not: the guide accepted "coerce height and impute" in 5 of 5 runs, and the agent never stripped the unit.
  The training file shows the '183cm' values in both profiles, so the full profile's test-set numbers must have
  mattered. Which numbers did so is not tested here.
- Pooled G3d against G3d-tr: 0.60 against 0.40 (p = 0.12). Agreeing store: 0.0 for G3-pc against 0.325 for
  G3d-tr-pc (p = 0.0001), so the training-only profile still helps against agreeing wrong lessons, through TF01.

## Result 5: hint-level wording flips the guide where it supports a lesson

| | G3 (no hint) | G3h (hint wording) |
| --- | --- | --- |
| TF03 fix shown (of 5) / keep-`column_x` shown | 1 / 4 | **5 / 0** |
| TF05 fix shown (of 5) | 0 | 0 |
| Agreeing wrong lessons shown (of 20): TF01 / TF02 / TF03 / TF05 | 20: 5 / 5 / 5 / 5 | **5: 5 / 0 / 0 / 0** |

- TF05's hint wording names the leaking columns. The guide then rejected the TF05 fix lessons in all 15 judgements, every time as
  redundant with the task ("The task prompt already explicitly identifies the leakage columns"). The fix is in the task itself, so "fix shown" does not measure acceptance on this task.
- On TF02, TF03 and TF05, the hint wording turned the agreeing wrong lessons into contradicting ones, and the guide
  blocked all of them (G3-pc against G3h-pc: 20 against 5 of 20, p < 0.0001). 81% of the 48 rejections of wrong lessons
  cite the instructions.
- TF01's first hint only says "consider whether [the label distribution] affects…". The guide still accepted "the
  classes are balanced" in 5 of 5 runs.

## No harm

TF04 scored 1.0 in R3, G3d-tr and X3. TF06 stayed within 0.0004 of no memory (0.9798) in all three.

## Predictions (written before the runs)

| # | Prediction | Result |
| --- | --- | --- |
| 1 | G3s: fix shown on TF03 + TF05 in ≥ 6 of 10 | **Held.** 10 of 10. |
| 2 | G3s-pl: ≤ 2 of 10 | **Held.** 2 of 10 (at the threshold). |
| 3 | R3 pooled in [0.35, 0.65] | **Held.** 0.425. |
| 4 | R3-p wrong shown ≥ 9 of 20 | **Held.** 10 of 20. |
| 5 | R3 fix shown on TF03 + TF05 ≥ 3 of 10 | **Held.** 9 of 10. |
| 6 | G3d-tr: TF03 fix ≥ 3 of 5, keep-`column_x` ≤ 1 of 5, pooled ≥ 0.55 | **Failed.** 0, 5, 0.40. |
| 7 | G3d-tr-pc: ≤ 2 of 10 on TF01 + TF02 | **Failed.** 5 of 10 (TF01 0, TF02 5). |
| 8 | G3h: fix shown on TF03 + TF05 ≥ 7 of 10 | **Failed.** 5 of 10 (TF03 5, TF05 0, rejected as redundant with the hint). |
| 9 | G3h-pc: ≤ 8 of 20 | **Held.** 5 of 20. |
| 10 | X3 pooled within 0.15 of 0.475; X3-pc ≥ 12 of 20 | **Held.** 0.575; 15 of 20. |
| 11 | TF04 / TF06 within 0.01 of A0 in R3, G3d-tr, X3 | **Held.** |

8 held, 3 failed. Running total across P1-P18: 36 held, 16 failed, 1 not testable (of 53).

## What this means

1. The causal claim can name the sentence. Cutting the feature sentence alone reproduces the whole effect of hiding
   the task wording, and cutting an unrelated sentence does not.
2. A judge that reads the instructions is what loses the fixes. Random choice among the same candidates, and a
   reranker without instructions, both pass the fixes.
3. The guide's one benefit, blocking lessons that contradict the wording, flips with the wording. At the hint level the
   same agreeing wrong lessons now contradict it and are blocked.
4. Showing a fix is not enough when a conflicting lesson is also shown: 0 of 16 such runs.
5. The data profile is partly an oracle. Its TF03 result and its TF02 block depend on test-set numbers; its TF01 block
   does not. The paper should call it an upper bound that uses test data.

## Limits

- 5 runs per cell; G3s and G3s-pl rest on 2 tasks.
- The placebo sentences and the hint level are my choice; TF05's hint makes "fix shown" uninformative.
- G3h was measured at the guide only.
- One small cross-encoder.
- Result 3 pools arms that differ in other ways; it is a cross-tabulation, not a controlled contrast.
