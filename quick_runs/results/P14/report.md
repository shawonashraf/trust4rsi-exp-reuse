# P14: guide prompt ablation. Was P13's pattern the prompt or the model?

Notebook: [docs/notebooks/quick_P14.ipynb](../../../docs/notebooks/quick_P14.ipynb). Predictions (approved before the
runs): [quick_runs/predictions/P14.md](../../predictions/P14.md). Numbers from `summary.json`; every guide judgement
with its reason in `judgements.csv`; poisoned-run funnel in `trust_funnel.csv`.

**One-line answer.** Mostly the model. With the P13 prompt, the guide's own reasons say it rejects the fix lessons
because they contradict the task's instructions (83% of its rejections of fix lessons). When the prompt says a
lesson may go against the task's wording, the guide still rejects them, for the same stated reason (79%), and the
agent does no better (0.45). The open prompt did let more wrong lessons through (shown in 9 of 20 poisoned runs,
against 5 in P13).

## Terms

- **Guide, P13 prompt (G3o).** As P13's G3: a note is useful only if "it is consistent with the task and the data
  schema". Run again with new samples so that its reasons are saved. Guide only, no agent runs.
- **Guide, P14 prompt (G3n).** The same, except that criterion now reads "consistent with the data schema. A note
  can be useful even when it goes against the task's wording, if the data suggests that following the wording
  literally would give a wrong or misleading result." Agent runs follow.
- **-p.** The same on the poisoned store (708 lessons plus 12 planted wrong lessons).
- **Fix lessons.** Store lessons that state the fix, labelled before the runs: 5 for TF05 (drop leakage columns),
  7 for TF03 (exclude `column_x` or select features by causal reasoning). "Contradicting" TF03 lessons: 2 (keep
  columns).
- **Instruction reason.** A guide reason that matches `task (says|asks|states|requires|instructs)|instruction|use
  all|all available|contradict`. I read the rejected fix and wrong judgements to check the regex; it matches what
  the reasons say.

## What was run

1. Stage 1, guide only: 100 guide runs (G3o 30, G3o-p 20, G3n 30, G3n-p 20), all with JSON replies, reasons saved.
2. Stage 2: 50 agent runs (G3n on 6 tasks, G3n-p on the 4 failing tasks), all ended with a final answer. Same
   2-GPU server as P13, same sampling.

## Result 1: the guide rejects the fixes because they contradict the instructions (P13 prompt)

Of the guide's judgements on a fix lesson that it saw and rejected, the share whose reason cites the task's
instructions:

| Arm | Fix lessons rejected | Citing the instructions | Wrong lessons rejected | Citing the instructions |
| --- | --- | --- | --- | --- |
| G3o (P13 prompt) | 47 | 39 (83%) | | |
| G3o-p (P13 prompt) | 39 | 35 (90%) | 30 | 27 (90%) |
| G3n (P14 prompt) | 48 | 38 (79%) | | |
| G3n-p (P14 prompt) | 37 | 24 (65%) | 29 | 21 (72%) |

Typical reasons (P13 prompt):
- TF05: "This note advises removing 'leakage' features (like expenditure or majorcards), which contradicts the
  explicit instruction to 'Use all available feature columns'." (all 15 rejections in G3o)
- TF03: "This note suggests removing columns like 'column_x', which contradicts the prompt's instruction to keep
  only numerical features (implying all of them)."

The TF03 reasons also lean on the contradicting lesson: "…and note 130 specifically warns that 'column_x' might be
critical in these benchmarks." The guide used one misleading lesson as evidence against all the correct ones.

G3o reproduced P13's decisions: no TF05 fix lesson shown (P13: 0 of 5), a TF03 fix in 1 of 5 runs (P13: 1), the
contradicting TF03 lesson in 4 of 5 (P13: 4), and a wrong lesson shown in 6 of 20 poisoned runs (P13: 5).

## Result 2: telling the guide it may go against the task did not change much

| | G3 (P13) | G3o (P13 prompt again) | G3n (P14 prompt) |
| --- | --- | --- | --- |
| TF05: runs shown a fix lesson (of 5) | 0 | 0 | 2 |
| TF03: runs shown a fix lesson (of 5) | 1 | 1 | 0 |
| TF03: runs shown the contradicting lesson | 4 | 4 | 5 |

The P14 reasons still put the instructions first, sometimes while granting the lesson is right: "While logically
sound for real-world ML, it contradicts the explicit constraint …", "In a benchmarking context, following the
prompt's explicit instruction …". On TF03 the contradicting lesson was accepted in all 5 runs and then cited against
every fix lesson ("This contradicts note [id 130] by suggesting the removal of 'column_x'"); in one run the guide
reasoned that `column_x` "is likely the causal driver and the most important" feature.

Agent results, targeted checks on the 4 failing tasks (passes out of 5):

| Task | Check | A0 | C1 | E | G3 (P13) | G3n (P14) |
| --- | --- | --- | --- | --- | --- | --- |
| TF01 | class distribution | 0 | 5 | 5 | 4 | 3 |
| TF01 | class weights | 0 | 5 | 5 | 5 | 5 |
| TF01 | resampling | 0 | 0 | 0 | 0 | 0 |
| TF01 | stratified split | 3 | 5 | 5 | 5 | 5 |
| TF02 | strips the 'cm' unit | 0 | 4 | 4 | 4 | 5 |
| TF03 | leaves out `column_x` | 0 | 5 | 0 | 1 | 0 |
| TF03 | causal reasoning | 0 | 5 | 0 | 1 | 0 |
| TF05 | leaves out all 4 leaking columns | 0 | 1 | 0 | 0 | 0 |
| **Pooled** | of 40 | **0.075** | **0.75** | **0.475** | **0.50** | **0.45** |

TF05 leaking columns kept per run: G3n [4, 4, 4, 4, 1]. One run, which was shown a fix lesson, dropped 3 of the 4.

## Result 3: the open prompt lets more wrong lessons through, and conflicting lessons cancel out

| Arm | Wrong lesson accepted | Shown to the agent | TF01 / TF05 shown | Column left out, when shown |
| --- | --- | --- | --- | --- |
| E-p (P13, no guide) | (no guide) | 10 of 20 | 5 / 5 | 10 of 10 |
| G3-p (P13) | 5 of 20 | 5 of 20 | 4 / 1 | 5 of 5 |
| G3o-p (P13 prompt again) | 6 of 20 | 6 of 20 | 5 / 1 | (no agent runs) |
| G3n-p (P14 prompt) | 10 of 20 | 9 of 20 | 5 / 4 | 7 of 9 |

1. The open prompt nearly doubled the wrong lessons that reached the agent (9 against 5), mostly on TF05 (4 runs
   against 1; TF01 5 against 4). The TF05 wrong lessons say "exclude it even though the task says to use all columns", which is what the new prompt
   allows. TF02 and TF03 wrong lessons were still blocked in every run.
2. For the first time, the agent did not follow a wrong lesson it was shown, in 2 runs. Hand-checked: both TF05
   runs (seeds 3 and 4) were shown both the wrong lesson ("Drop reports …") and a correct one ("Identify and Remove
   Leakage Features"). Their final models kept `reports` and all 4 leaking columns, as with no memory. The two runs
   shown only the wrong lesson dropped `reports`. All 5 TF01 runs dropped `assists` (checked against the executed
   feature lists). No run that was not shown a wrong lesson dropped the named column.

## Predictions (written before the runs)

| # | Prediction | Result |
| --- | --- | --- |
| 1 | With the P13 prompt, rejections of TF05/TF03 fix lessons cite the instructions in at least half | **Held.** 83% (G3o), 90% (G3o-p). |
| 2 | G3o reproduces P13: rejects the TF05 fix in at least 4 of 5 runs; wrong lesson accepted in at most 8 of 20 | **Held.** 5 of 5; 6 of 20. |
| 3 | P14 prompt: a TF05 fix shown in at least 3 of 5 runs and a TF03 fix in at least 3 of 5 | **Failed.** 2 of 5 and 0 of 5. |
| 4 | G3n pooled at least 0.60 | **Failed.** 0.45. |
| 5 | G3n-p accepts a wrong lesson in at least 10 of 20 | **Held, at the threshold.** 10 accepted, 9 shown. |
| 6 | The agent leaves out the named column in at least 90% of runs shown a wrong lesson | **Failed.** 7 of 9 (78%); the 2 exceptions were also shown a correct lesson that conflicts with it. |
| 7 | No harm: TF04/TF06 within 0.01 of A0 | **Held.** TF04 1.0; TF06 0.9792 (A0 0.9798). |
| 8 | Guide tokens within 25% of P13's G3 | **Held.** 7.4k against 6.9k per run. |

The predictions file said: "If 3 fails, the pattern comes from the model, not the wording." 3 failed.

## What this means

1. The guide's behaviour is the model's, not only the prompt's. Asked to judge usefulness, it ranks agreement with
   the task's instructions above what the lesson says about the data, and it keeps doing so when told it may
   depart from the wording. Its reasons say this directly.
2. Loosening the prompt moved the trade-off the wrong way here: no more fixes reached the agent, but more wrong
   lessons did. The one kind of advice that the new wording let through more often was the planted wrong lesson
   that openly says to break the instruction.
3. A misleading lesson can veto correct ones. On TF03 the guide accepted "keep `column_x`" first and then cited it
   against every fix lesson.
4. When the agent got two lessons that contradict each other, it followed neither and fell back to the task's
   instructions.

## Limits

- 4 failing tasks, 5 runs each; one prompt change; one model.
- The regex for "instruction reason" was checked by reading, not hand-labelled per judgement.
- The conflicting-lessons observation rests on 2 runs.
- G3o and G3n are new samples; P13's G3 decisions were not re-run with reasons.

## Correction (2026-11-17, after peer review)

The pre-set regex counts a bare `contradict`, which can refer to another lesson rather than the task. Notebook
[docs/notebooks/quick_REVIEW_CHECKS.ipynb](../../../docs/notebooks/quick_REVIEW_CHECKS.ipynb)
(`review_checks.json`) counts a reason that matches only through `contradict` as citing the instructions only if it
also names the prompt, the task, a requirement or a constraint (checked by reading all 14 such reasons).

- G3o (P13 prompt): 39 of 47, 83%. Unchanged; all 3 contradict-only reasons cite the prompt.
- G3n (P14 prompt): **32 of 48, 67%** (not 79%). 6 rejections cite only the accepted "keep `column_x`" lesson (130).
- Q3 (P15): 29 of 45, 64%. Unchanged.

The conclusion holds (most rejections cite the instructions under both prompts) with a smaller share for the open
prompt.
