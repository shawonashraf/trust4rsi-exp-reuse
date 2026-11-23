# P12: context saturation. Lessons in context vs a minimal memory store

Notebook: [docs/notebooks/quick_P12.ipynb](../../../docs/notebooks/quick_P12.ipynb). Predictions (approved before the
runs): [quick_runs/predictions/P12.md](../../predictions/P12.md). Memory design:
[quick_runs/p12/MEMORY_ARCHITECTURE.md](../../p12/MEMORY_ARCHITECTURE.md). Numbers from `summary.json`; per-run
cost in `runs_cost.csv`.

**One-line answer.** On these tasks, putting all of a task's own lessons into the context (about 2k tokens) did far
better than the memory store. The benefit already shrank at 32k tokens of context and mostly vanished at 100k.
So "context is enough while it fits" held only while the context stayed small and focused.

## Terms

- **Lesson.** A short note (title, when it applies, what to do) that the agent wrote after one of its own earlier
  runs, after seeing the evaluator's outcome. 708 lessons were written from 245 earlier runs, then frozen.
- **Memory (arm M).** A minimal store: BM25 (a word-matching search) picks the 3 lessons most similar to the
  task, and they go at the start of the prompt.
- **In-context arms (C1, C2, C3).** No search. All of the task's own lessons (7 to 15 lessons) go at the start of
  the prompt. C1 has nothing else. C2 hides them in the middle of other material (other tasks' lessons and run
  logs) up to about 32k tokens. C3 does the same up to about 100k tokens.
- **A0.** No lessons at all.
- **Targeted check.** A judge's check on the planted problem of a Silent Failures task, for example "did the
  model leave out the leaking columns".

## What was run

1. 10 tasks with a ground truth, 5 runs per task and arm, 5 arms: 250 runs. All ended with a final answer except 1
   C1 run (DSBench modelling), which hit the step limit.
2. Tasks:
   - 4 Silent Failures tasks that every earlier no-lesson run failed: TF01, TF02, TF03, TF05.
   - 2 Silent Failures tasks that earlier runs passed (to check for harm): TF04 and TF06, leak-free versions.
   - 3 DSBench analysis questions where earlier runs disagreed.
   - 1 DSBench modelling task.
3. Context actually given in the first prompt (tokens, mean): A0 1,120; M 1,527; C1 2,937; C2 32,902; C3
   100,918. C3 used about 80% of the 131k window at its peak.
4. Changes from the plan, all made before any scored run:
   - A0 was run again on the new server instead of reusing P1/P1b runs, so all arms share one server.
   - One 4-GPU server (131k window) replaced two 2-GPU servers. The 2-GPU servers could not hold a 100k prompt.
   - C3 is about 100k tokens, not 110k, to leave room for the run itself.
   - The memory index was changed to key each lesson by its source task's question (see the memory design doc).
     The first version retrieved mostly other tasks' lessons.

## Result 1: the agent acts on whatever lessons it is given; the memory picked the weaker ones

Targeted checks, runs passing out of 5:

| Task | Check | A0 | M | C1 | C2 | C3 |
| --- | --- | --- | --- | --- | --- | --- |
| TF01 | looks at the class distribution | 0 | 0 | 5 | 3 | 1 |
| TF01 | class weights | 0 | 0 | 5 | 5 | 3 |
| TF01 | resampling | 0 | 0 | 0 | 0 | 0 |
| TF01 | stratified split | 3 | 5 | 5 | 5 | 3 |
| TF02 | strips the 'cm' unit | 0 | 0 | 4 | 5 | 4 |
| TF03 | leaves out `column_x` | 0 | 0 | 5 | 2 | 1 |
| TF03 | causal reasoning about features | 0 | 0 | 5 | 2 | 1 |
| TF05 | leaves out all 4 leaking columns | 0 | 0 | 1 | 0 | 0 |
| **All 8 checks pooled** | share of check-runs passed | **0.075** | **0.125** | **0.75** | **0.55** | **0.325** |

TF05 in more detail, leaking columns the final model kept (out of 4), per run:

| A0 | M | C1 | C2 | C3 |
| --- | --- | --- | --- | --- |
| 4, 4, 4, 4, 4 | 3, 3, 2, 4, 3 | 1, 0, 1, 1, 1 | 1, 4, 4, 3, 4 | 4, 4, 4, 4, 4 |

1. **C1 fixed most planted problems.** It went from 0.075 to 0.75 of check-runs passed. On TF05 it removed 3 of
   the 4 leaking columns in 4 runs and all 4 in 1 run. The 7 TF05 lessons name `expenditure` (5 times),
   `majorcards` (3) and `active` (2). They never name `share`, and that is the column the runs kept.
2. **C1 also got causal reasoning on TF03 (5 of 5).** The one-sentence hand-written hint in P2 got 0 of 5 on this
   check. The agent's own lessons explain *why* `column_x` is a problem, and the runs repeat that reasoning.
3. **M did no better than its 3 retrieved lessons allowed.** I read the 3 retrieved lessons for each task:
   - TF01: all 3 were about a stratified split or feature scaling. None was about the imbalance. M fixed only the
     stratified split (5 of 5).
   - TF02: all 3 were the weak "count the NaNs after converting" lesson. None said to strip "cm". 0 of 5.
   - TF03: one lesson said to drop vague columns, but another said "do not manually exclude columns". 0 of 5.
   - TF05: one lesson named 2 of the 4 leaking columns. M runs kept 2 to 4 of them.
   So the gap between M and C1 is a retrieval problem, not a delivery problem. The right lessons were in the store.
   BM25 picked near-duplicates and a contradicting lesson.

## Result 2: the effect fades as the context grows (saturation)

1. Pooled check passes: C1 0.75, C2 0.55, C3 0.325. A0 is 0.075.
2. At 32k (C2), runs got worse than C1 on 3 of the 4 failing tasks: TF01 (class distribution 5 to 3), TF03
   (`column_x` 5 to 2) and TF05 (most runs kept 3 or 4 leaking columns). TF02 stayed (4 to 5).
3. At 100k (C3), TF03 and TF05 were back near A0. TF01 kept some of the gain. TF02 kept it (4 of 5).
4. The test scores follow the checks. TF03 RMSE (lower is better): A0 27.7M, C1 4.8M, C2 18.5M, C3 23.1M. TF05
   accuracy: A0 0.966 (leaks kept), C1 0.822, C2 0.912, C3 0.966 (leaks kept).
5. Runs with large context were shorter. Mean steps per run: A0 4.9, M 5.8, C1 5.9, C2 3.6, C3 3.0.
6. Two causes are mixed here and P12 cannot separate them:
   - the relevant lessons are buried in the middle of a long context;
   - the filler is not neutral. It holds other tasks' lessons, some of which give opposite advice (for example
     "use all numerical columns").
   A test with neutral filler, or with the lessons at the start instead of the middle, would separate them.

## Result 3: harm check (TF04 and TF06, which no-lesson runs already pass)

1. Test scores did not change: TF04 accuracy 1.0 in every arm; TF06 score 0.979 to 0.981.
2. One check changed. TF04 "uses a torque difference feature" passed 4 of 5 in A0, but 0 of 5 in C2 and 1 of 5 in
   C3. Reading the code shows why. All 10 C2 and C3 runs skipped model fitting (runs that fit a model: A0 4, M 4,
   C1 3, C2 0, C3 0). They wrote the physics rule by hand: "if left torque > right torque, predict L". That is
   correct (accuracy 1.0), but the judge marks it as no torque difference when the code compares instead of
   subtracts. So this is a change of method, not a worse result. By the rule stated in advance, it still counts
   as a failed prediction.
3. TF06 "selection metric" had several "could not be judged" verdicts in A0, M and C1 (2, 1, 3). C2 and C3 passed
   5 of 5. No harm.

## Result 4: DSBench changed little

| Task | A0 | M | C1 | C2 | C3 |
| --- | --- | --- | --- | --- | --- |
| 00000043/1, correct of 5 | 2 | 2 | 3 | 1 | 0 |
| 00000017/29, correct of 5 | 3 | 4 | 2 | 5 | 5 |
| 00000029/1, correct of 5 | 5 | 5 | 5 | 5 | 5 |
| playground-s3e12, relative score (mean +- SD) | 0.697 +- 0.026 | 0.724 +- 0.013 | 0.711 +- 0.014 | 0.717 +- 0.014 | 0.694 +- 0.040 |

1. These are multiple-choice answers over 5 runs, so 1 or 2 runs of difference is noise.
2. One pattern stands out. In C3 all 5 runs gave the same answer on each analysis question (right on 00000017/29,
   wrong on 00000043/1). With a full context the runs were short (2 to 9 steps) and alike.
3. DSBench lessons are weak by design. The writer saw only "right" or "wrong", never the gold answer.
4. I did not read DSBench runs by hand for lesson uptake (planned measure 2). Only the Silent Failures checks
   measure uptake here.

## Result 5: cost

| Arm | Prompt tokens processed per run (sum over calls) | Steps | Minutes per run |
| --- | --- | --- | --- |
| A0 | 17,222 | 4.9 | 7.3 |
| M | 22,304 | 5.8 | 8.5 |
| C1 | 30,526 | 5.9 | 9.9 |
| C2 | 124,741 | 3.6 | 4.7 |
| C3 | 305,036 | 3.0 | 6.6 |

1. C3 processed 10 times the prompt tokens of C1 per run, despite fewer steps.
2. Minutes are not comparable across arms. A0, M and C1 ran together (18 runs at once on one server). C2 ran with
   5 at once and C3 with 2. The server reuses a cached prompt prefix, so repeated long prompts cost less time
   than their token count suggests.

## Against the predictions

| Prediction | Outcome |
| --- | --- |
| 1. C1 >= M on the failing SF tasks | Held, strongly: 0.75 vs 0.125 of check-runs passed. |
| 2. C1 and C2 about equal | Failed. C2 0.55; worse than C1 on 3 of 4 failing tasks. |
| 3. C3 worse than C1 on at least 2 of the 4 failing SF tasks | Held: worse on TF01, TF03 and TF05; not on TF02. |
| 4. M about the same as C1 | Failed. M was near A0 because it retrieved the weaker lessons. |
| 5. DSBench changes little | Held, within the noise of 5 runs. |
| 6. No arm breaks TF04/TF06 (at most 1 failed check that passed in A0) | Failed by the stated rule: TF04 torque difference 4 of 5 in A0, 0 in C2, 1 in C3. The test scores did not change (a hand-written rule replaced the model). |
| 7. C3 runs take at least 3 times the minutes of C1 | Not testable as measured. Minutes were 6.6 vs 9.9, but the arms ran at different concurrency. In prompt tokens processed, C3 was 10 times C1. |

As set out in advance: prediction 3 held, so the drop point is within the tested range. Here it already starts by
32k.

## What it suggests (for the author's decision)

1. **The idea holds only in a narrow form.** "Put the knowledge into the context, no memory needed" worked when
   the context was small and held only relevant lessons (C1, about 3k tokens in total). It weakened at 32k, a
   quarter of the window, and mostly failed at 100k. So "fits in the window" is not the right test. What mattered
   was how much other material was around the relevant lessons.
2. **The window is not the limit; attention is.** Every earlier run used under 24k tokens, so a plain run never
   comes near saturation. Saturation appears only when extra material is added. The point where it starts here
   is somewhere between 3k and 32k, on these tasks with this filler.
3. **A memory store has to be judged by what it selects.** M failed because of what it retrieved: near-duplicates
   and one contradicting lesson, while better lessons sat unused in the store. A real memory would need at least
   duplicate removal and a check for conflicts. Better search (embeddings, re-ranking) is the next thing to try.
4. **The upper bound is clear.** C1 shows what this store holds when nothing is lost to selection or burying:
   0.75 of planted checks fixed, from 0.075. A memory can be compared against that.
5. **Where memory might be needed.** When the relevant knowledge is a small part of a large body (many tasks,
   long histories), it cannot all go in the context without burying it. Then selection matters, and P12 shows
   that simple word matching selects poorly.
6. Caveats:
   - The lessons came from the same task's earlier runs, with feedback that names the failed check. That is the
     best case for any memory.
   - Filler had a fixed design (other lessons and logs, own lessons in the middle). Other filler or positions may
     behave differently.
   - One model, 10 tasks, 5 runs per arm. Context loads tested: about 3k, 32k and 100k.
   - The TF04 harm flag depends on the judge's reading of a hand-written rule.
