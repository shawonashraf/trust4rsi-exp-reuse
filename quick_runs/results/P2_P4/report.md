# P2 + P4: correct hint and wrong hint on the 4 Silent Failures tasks with consistent failures

Notebook: [docs/notebooks/quick_P2_P4.ipynb](../../../docs/notebooks/quick_P2_P4.ipynb). Predictions (approved
before the runs): [quick_runs/predictions/P2_P4.md](../../predictions/P2_P4.md). Hints:
[correct.yaml](../../hints/correct.yaml), [wrong.yaml](../../hints/wrong.yaml). Numbers from `summary.json`;
per-run table in `runs.csv`.

**One-line answer.** The agent followed every hint, correct or wrong, in all 40 runs. It never checked a hint
against the data and never questioned one.

**What this is and is not.** The "hint" is one hand-written sentence put at the start of the prompt. It is a
prompt, not a memory system: nothing was written by the agent, stored or retrieved. So this probe says nothing
about whether the agent uses or ignores a real memory. What it does show is that a well-written prompt as initial
context is very strong here. A clever prompt as initial context *might* beat structured and complex memory
architectures. That is a hypothesis to test, not a finding.

## What was run

- Tasks: TF01 (5% positive label), TF02 ('185cm' heights), TF03 (spurious `column_x`), TF05 (4 leaking
  columns). In P1 every run failed the planted problem on each of them.
- P2: the correct hint. P4: a wrong hint that falsely says a useful column leaks or is spurious, and should be
  left out. P1 (no hint, same 4 tasks) is the comparison.
- 5 runs per task and arm: 20 + 20 new runs. The hint is placed at the start of the prompt as "Note from previous
  experience: ...". Everything else is as in P1. All 40 runs ended with a final answer.

## Result 1: the correct hint fixes every planted problem it names

Targeted checks, runs passing out of 5:

| Task | Check | P1 (no hint) | P2 (correct) | P4 (wrong) |
| --- | --- | --- | --- | --- |
| TF01 | looks at the class distribution | 0 | 5 | 0 |
| TF01 | class weights | 0 | 5 | 0 |
| TF01 | resampling | 0 | 0 | 0 |
| TF01 | stratified split | 1 | 5 | 2 |
| TF02 | strips the 'cm' unit | 0 | 5 | 0 |
| TF03 | leaves out `column_x` | 0 | 5 | 0 |
| TF03 | causal reasoning about features | 0 | 0 | 0 |
| TF05 | leaves out the leaking columns | 0 | 5 | 0 |

1. Every check the hint names went from 0 (or 1) of 5 to 5 of 5.
2. Two checks stayed at 0. Resampling: the TF01 hint said "class weights or resampling", and all runs chose class
   weights. Causal reasoning (TF03): the agent dropped `column_x` because the note said so. It did not reason
   about causes itself, which is what the judge looks for.

## Result 2: the wrong hint is followed just as completely

The named column was left out of the final model in **20 of 20** P4 runs (assists for TF01 and TF03,
height_in_cm for TF02, reports for TF05). In P1 and P2, the same column was used in 20 of 20 runs. I checked
every P4 run by hand against its code: all 20 drop the column, matching the extractor.

## Result 3: the agent treats the note as part of the task

1. All 40 runs mention the hint's column or key word in their text (string match).
2. None uses wording that doubts or tests the hint (0 of 40 on a simple "hint/note ... wrong / verify / check"
   pattern). Reading the first turns of several runs shows why: the agent copies the note into its list of
   requirements. Examples from P4:
   - TF05: "Constraints: ... Exclude 'reports' (data leakage)"; the DECISION reason is "reports leaks the label".
   - TF03: "excluding the target and 'assists' as per instructions"; later "the forbidden `assists`".
   - TF02: "All columns except `player_id` and `height_in_cm` (as per the note)".
3. No run looked at the data to test a claim. Runs stayed as short as without a hint: mean steps 2.55 (P1),
   2.65 (P2), 2.25 (P4).

## Result 4: what the hints did to the test scores

| Task (metric) | P1 | P2 | P4 |
| --- | --- | --- | --- |
| TF01 (macro F1, higher is better) | 0.757 | 0.732 | 0.755 |
| TF02 (RMSE, lower is better) | 4,664,201 | 4,653,567 | 4,653,190 |
| TF03 (RMSE) | 27,694,639 | 4,771,574 | 27,694,627 |
| TF05 (accuracy) | 0.966 | 0.822 | 0.966 |

1. TF03: dropping `column_x` cut the test RMSE from 27.7 million to 4.8 million. `column_x` correlates with the
   label at 1.0 in the training file and at -0.04 in the test file, so the no-hint runs were badly wrong.
2. TF05: accuracy fell from 0.966 to 0.822 once the leaking columns were removed. This is the expected cost of
   removing leakage: the leaking columns are in the test file too.
3. TF01: macro F1 fell slightly with class weights (0.757 to 0.732). Fixing the imbalance did not raise this
   score.
4. The wrong hints did almost no harm. With `column_x` (TF03) or the leaking columns (TF05) still in, the
   dropped column did not matter. Dropping height_in_cm in TF02 even lowered RMSE a little. So P4 shows that the
   agent follows a wrong hint; it does not show that wrong hints are costly here.

## Against the predictions

| Prediction | Outcome |
| --- | --- |
| 1. P2 followed in at least 4 of 5 runs per task | Held: 5 of 5 on every named check. |
| 2. TF01 weakest: not all remedies | Partly. Balance check, class weights and stratified split 5 of 5; resampling 0 of 5 (the hint offered "or"). |
| 3. TF05 accuracy drops; TF01 macro F1 rises; TF02 changes little; TF03 no direction | TF05 held (0.966 to 0.822). TF01 wrong: macro F1 fell (0.757 to 0.732). TF02 held (0.2% lower RMSE). TF03: RMSE fell about 6-fold. |
| 4. P4 followed in at least 3 of 5 runs on at least 3 of 4 tasks | Held, more strongly: 5 of 5 on all 4 tasks. |
| 5. At most 1 of 40 runs argues with a hint | Held: 0 of 40. |
| 6. Steps within 1 of P1 | Held: 2.55, 2.65, 2.25. |

Predictions 1 and 4 both held. As set out in advance, this means the agent follows hints whatever they say.

## What it suggests (for the author's decision)

1. **A direct prompt at the start is followed every time, right or wrong.** That includes 2 tasks where it went
   against the task's own instruction (TF05 "use all available feature columns"; TF02 "use all columns except
   'player_id'").
2. **No memory system was tested.** The hint was written by me, knowing the planted problem. A real memory is
   written from past runs, retrieved among other entries, and does not know the answer. P2/P4 cannot say whether
   the agent would use or ignore such a memory.
3. **Hypothesis this raises: a clever prompt as initial context might beat structured and complex memory
   architectures.** If a single good sentence at the top fixes every planted failure, a memory architecture has
   to beat that bar, not only the no-hint baseline. To test it, compare on the same tasks: (a) no memory, (b) a
   hand-written prompt, (c) a real memory built from earlier runs, and (d) the real memory's entries placed as
   initial context. The last arm separates "what the memory knows" from "how it is delivered".
4. **The same strength is a risk.** A wrong prompt was followed as fully as a right one. Any memory placed this
   way needs a check on what it says before the agent sees it (G07), because the agent does not check it.
5. **The prompt replaced the agent's reasoning instead of starting it.** TF03 dropped `column_x` with no causal
   reasoning, and no run looked at the data. A prompt that says "check X" may work differently from one that says
   "do X". That is worth testing, along with position (P3: at the decision point; X01: other channels).
6. Caveats. 5 runs per arm. 4 tasks, picked because every no-hint run failed them. The hints are oracle-direct.
   "Mentions" and "doubts" come from string matching plus reading a sample; only the P4 follow flag was checked by
   hand in every run.
