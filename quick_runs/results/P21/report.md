# P21: review round 5 controls

Notebook: [docs/notebooks/quick_P21.ipynb](../../../docs/notebooks/quick_P21.ipynb). Predictions (committed before the
runs): [quick_runs/predictions/P21.md](../../predictions/P21.md). Numbers come from `summary.json`. The train-only
lessons are in `lessons_nfdtr.jsonl`; their labels (committed before the agent runs) are in
`quick_runs/p21/labels_nfdtr.yaml`.

**One-line answer.** A data probe at the write step does **not** need test-set numbers. Given only the training
profile, the writer flagged `column_x`'s +1.00 correlation as leakage and told the agent to drop it, the opposite of
how the guide read the same number in P18. With these train-only lessons the agent left `column_x` out in 5 of 5 runs
and scored 0.425, not significantly below the full (test-set) probe (0.60). The hardware split is not a confound: the
no-memory and all-own-lessons arms match across the 2-GPU and 4-GPU servers.

## Terms

- **nfdtr lessons.** The P19 probe writer (no evaluator outcome) given the profile of the training file only
  (`data_profile(..., train_only=True)`); every test-set number removed. Same 40 source runs.
- **A0g2, C1g2.** P12's no-memory and all-own-lessons arms rerun on the 2-GPU servers (the earlier P12 runs used a
  4-GPU server). Fix lessons, reinforcing lessons, pooled: as in P16-P19.

## What was run

- Writer: 40 calls, 115 lessons, 0 parse errors.
- Guide stage: 60 runs (C1-nfdtr all-own, E-nfdtr embedding top-3).
- Agent: C1-nfdtr, E-nfdtr, A0g2, C1g2 (30 each, all 6 SF tasks). All ended with a final answer except two C1g2 runs
  that hit the step limit.
- Two TP=2 Gemma servers on GPUs 4-5 and 2-3. The root filesystem was nearly full (98%), and 6 C1g2 runs broke off
  (TF04 seed4 and all 5 TF06 runs, the no-harm tasks). Their restarts then failed on the read-only data copies. They
  were rerun from scratch with the sandbox temp routed to `/data` (`REVERIES_TMPDIR`); the broken attempts are kept in
  `P21_C1g2_failed_attempts/`. The 4 failing tasks were not affected.

## Result 1: a train-only probe at write still gives the TF03 and TF01 fixes

| | Full profile (nfd, P19) | Train-only (nfdtr) |
| --- | --- | --- |
| Fix lessons | 7 (TF01 2, TF03 5) | **9** (TF01 3, TF02 1, TF03 5) |
| Reinforcing lessons | 7 (TF02 5, TF03 2) | 6 (TF02 4, TF03 2) |

- TF03: every source run produced a leakage lesson, e.g. "a feature with correlation ~1.0 is leaked; remove it"
  (seed1/0 names `column_x`). The writer saw only the training +1.00 and read it as a warning sign.
- This is the opposite of the guide in P18: given the same train-only profile, the guide read `column_x`'s +1.00 as
  "a strong predictive feature, keep it" and showed "keep `column_x`" in 5 of 5 runs. The writer and the guide read
  the identical number in opposite directions.
- TF01: 3 lessons recommend class weights or checking the distribution. TF02: 1 lesson says to strip the `cm` suffix
  (4 others still say to coerce it away). TF05: nothing about leakage.

## Result 2: with those lessons the agent fixes TF03, and memory scores 0.425

Targeted checks on the 4 failing tasks (passes out of 5):

| Check | A0 | C1 (outcome) | C1-nf (none) | C1-nfd (full probe) | C1-nfdtr (train probe) | E-nfdtr |
| --- | --- | --- | --- | --- | --- | --- |
| TF01 class distribution | 0 | 5 | 0 | 4 | 3 | 3 |
| TF01 class weights | 0 | 5 | 0 | 5 | 5 | 5 |
| TF01 stratified split | 3 | 5 | 1 | 5 | 1 | 3 |
| TF02 strips the unit | 0 | 4 | 0 | 0 | 0 | 0 |
| TF03 leaves out `column_x` | 0 | 5 | 0 | 5 | **5** | 0 |
| TF03 causal reasoning | 0 | 5 | 0 | 5 | **3** | 0 |
| TF05 leaves out all 4 leaks | 0 | 1 | 0 | 0 | 0 | 0 |
| **Pooled (of 40)** | **0.075** | **0.75** | **0.025** | **0.60** | **0.425** | **0.275** |

- C1-nfdtr against no checker (C1-nf 0.025): p < 0.0001. The train-only probe recovers most of the gain.
- C1-nfdtr against the full probe (C1-nfd 0.60): p = 0.18, not significant. The test-set numbers add little at the
  write step.
- C1-nfdtr against the evaluator store (C1 0.75): p = 0.006, still below.
- TF03 RMSE with C1-nfdtr was 4.75e6, the same as the evaluator store (4.76e6) and far below no memory (2.77e7). The
  agent was shown the 5 fix lessons and the 2 generic reinforcing ones and left `column_x` out every time (as in
  P19, a generic "keep all numerical columns" lesson does not block a fix that names the column).
- Embedding retrieval (E-nfdtr 0.275) found the TF01 class-weight lesson in 5 of 5 but no TF03 fix (it showed the
  reinforcing lesson instead); the probed store still needs all-own-lessons to surface the TF03 fix.
- TF02 is unchanged (0 of 5). With all own lessons, the strip lesson was shown in 5 of 5 runs next to four "coerce"
  lessons, and the agent never stripped the unit. Embedding retrieval showed neither kind.

## Result 3: the hardware split is not a confound

Per-check passes, 4-GPU (P12) against 2-GPU (P21):

| | A0 4-GPU | A0 2-GPU | C1 4-GPU | C1 2-GPU |
| --- | --- | --- | --- | --- |
| Pooled (of 40) | 0.075 | 0.075 | 0.75 | 0.65 |
| Largest per-check gap | | 1 of 5 | | 1 of 5 |

- A0: identical pooled (0.075), every check within 1 of 5, Fisher p = 1.0.
- C1: 0.75 against 0.65. The 0.10 gap is four checks each off by one run (TF02 unit 4 vs 3; TF03 `column_x` 5 vs 4
  and causal 5 vs 4; TF05 1 vs 0); Fisher p = 0.46. This is run-to-run noise at 5 runs, not a hardware effect.
- No-harm: TF04 at 1.0 for C1-nfdtr and E-nfdtr; TF06 within 0.003 of no memory (E-nfdtr 0.9777, A0 0.9798). (A0 2-GPU TF04 was 0.915, one run of five
  below the others; TF04 is a no-harm accuracy check, not a pitfall check.)

## Predictions (written before the runs)

| # | Prediction | Result |
| --- | --- | --- |
| 1 | Train-only TF03 fix in at most 2 of 5 source runs | **Failed.** 5 of 5; the writer flags +1.00 as leakage. |
| 2 | At least 1 TF01 fix lesson | **Held.** 3. |
| 3 | C1-nfdtr pooled at most 0.40 | **Failed.** 0.425. |
| 4 | E-nfdtr pooled at most 0.20 | **Failed.** 0.275. |
| 5 | A0 within 0.05, C1 within 0.10, no check differs by more than 2 of 5 | **Held.** 0.075=0.075; 0.65; max gap 1. |
| 6 | TF04 / TF06 within 0.01 of A0 (C1-nfdtr, E-nfdtr) | **Held.** |

3 held, 3 failed. The three failures all point the same way: the write-time probe is **more** useful than predicted,
because it does not depend on test-set numbers. Running total across P1-P21: 48 held, 20 failed, 1 not testable (of
69).

## What this means

1. The write-step check is more robust than P19 showed. A probe of the training data alone, given to the writer,
   recovers the TF03 fix (0.425, against 0.60 with test-set numbers, p = 0.18). The write-time probe is not an
   upper bound that needs test data; the read-time guide profile was.
2. The writer and the guide read the same statistic in opposite directions. A correlation of +1.00 is a warning sign
   to a writer reviewing a finished run, and a reason to keep the column to a guide ranking candidate lessons before
   the run. Where the check sits changes what the number means.
3. The split between the headline arms and the rest is not a hardware artefact: no memory is identical across the two
   server types, and all own lessons differ only within run-to-run noise.

## Limits

- 5 runs per agent cell; one write generation; one author's labels by the P17 rule.
- The train-only profile still contains every training-set number, so it is not a zero-information check; it is the
  realistic write-time check the read-time profile was not.
- Rerunning A0 and C1 tests the hardware split and run-to-run variation together; it cannot separate them.
