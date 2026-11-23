# Quick runs: baseline of how the no-memory agent behaves

Branch `quick/baseline-behaviour`. Plan: [plans/quick_runs.md](../plans/quick_runs.md). Date: 2026-11-11.

Interactive version: [docs/report/quick_runs.html](../docs/report/quick_runs.html) (open in a browser; built by
`quick_runs/build_html.py` from the probes' `summary.json` files).

**One-line answer.** Without memory, gemma-4-31B mostly gives the same answer across runs. Where it is wrong,
it is usually wrong the same way every time. On Silent Failures, every run misses the same planted problems,
because the agent writes the whole pipeline before looking at the data. On DSBench it is mostly right. So a
memory would need to change the agent's choice. Making runs agree with each other would add little.

Setup (all probes):
- gemma-4-31B-it on SGLang (two servers, 2 x A6000 each).
- Temperature 1.0, thinking on.
- A ReAct agent: each turn, the model states its DECISION lines and runs one code block.
- A post-hoc extractor reads the executed code and records what was actually decided. A hand check of 5 runs
  found it right on 52 of 55 decisions ([setup/hand_check.md](results/setup/hand_check.md)).

Each probe below links its report and its executed notebook. Every number comes from the notebook's
`summary.json`.

Terms used throughout:
- *Answer consistency*: the share of run pairs that gave the same final answer.
- *Decision agreement*: the same share, but for the value executed at one decision type (for example, the
  feature list).
- *Consistent failure*: all 5 runs fail the same check.

## Done

### P10: recall probe (no agent runs)

[report](results/P10/report.md), [notebook](../docs/notebooks/quick_P10.ipynb)

- Question: does the model already know the expected answer without the data?
- Runs: 15 tasks x 3 prompts, labelled by hand.
- Numbers: 6 of 15 tasks are recall hits. Most hits come from the prompt itself. Either it names the study
  (BLADE), or the task wording or file name gives the pitfall away (TF01, TF06-cue).
- After P10: 8 tasks are high risk, 5 medium, 2 low.
- What it suggests: results on high-risk tasks may be recall, not reasoning. Later probes report them split by
  risk.

### P9: serving noise floor (12 runs)

[report](results/P9/report.md), [notebook](../docs/notebooks/quick_P9.ipynb)

- Question: how different are runs when only the server varies?
- Runs: greedy decoding, 3 tasks, 3 batched reruns plus 1 run alone.
- Numbers:
  - Reruns differ in 8 of 9 pairs, often within the first 100 characters.
  - 4 greedy crofoot runs reached 3 different verdicts.
  - TF01 and TF03 failed the same checks in every run.
- What it suggests: there is no "same seed, same output" baseline here. Small early differences can flip a
  BLADE verdict. Planted failures survive the noise.

### P1: divergence census, Silent Failures + BLADE (75 runs)

[report](results/P1/report.md), [notebook](../docs/notebooks/quick_P1.ipynb)

- Question: how much do runs disagree, where does the disagreement start, and how often do all runs fail the
  same way?
- Runs: 15 tasks x 5 runs, no hint.
- Numbers:
  - Answer consistency: 0.81 on Silent Failures, 0.86 on BLADE.
  - 8 checks fail in all 5 runs. TF01: no look at the 5% class balance. TF02: '185cm' turned into missing
    values. TF03: the spurious column kept. TF05: leaking columns used.
  - Decision agreement is lowest at transformation (0.27), controls (0.29) and variables (0.52). It is high at
    validation (0.97) and metric (0.95).
  - The agent writes the whole SF pipeline in 1 to 2 cells (3.2 steps per run) and rarely looks at the data
    first.
- What it suggests: this is the plan's "little disagreement, many consistent failures" case. Reframe the
  memory around consistent failures, not forks.

### P1b: divergence census, DSBench (170 runs)

[report](results/P1b/report.md), [notebook](../docs/notebooks/quick_P1b.ipynb)

- Question: the same as P1, on tasks with known answers (added because hypothesis search has no ground truth).
- Runs: 26 workbook questions and 8 Kaggle modelling tasks, 5 runs each.
- Numbers:
  - Analysis accuracy 0.896 on 25 scorable questions. One DSBench gold answer is shifted (00000020/35).
  - All 18 questions with 5 identical answers were right. No consistent failures.
  - Modelling: 40 of 40 valid submissions. Relative-gap spread above 0.02 on 1 of 8 tasks.
- What it suggests: these DSBench tasks are too easy (and likely too well known) to show the memory problem.
  One useful pattern: on 00000029/1, 3 runs found the right number, then kept hunting for an error the question
  hinted at, and ran out of steps. Prompt text overrode the agent's own correct result.

### P2 + P4: correct hint and wrong hint on the 4 failing SF tasks (40 runs)

[report](results/P2_P4/report.md), [notebook](../docs/notebooks/quick_P2_P4.ipynb)

- Question: does a correct hint fix the consistent failures, and is a plausible wrong hint followed too?
- Runs: TF01, TF02, TF03, TF05, 5 runs per arm. Hint at the start of the prompt.
- Numbers:
  - Correct hint: every named check went from 0 (or 1) of 5 in P1 to 5 of 5.
  - Wrong hint (leave out a useful column): followed in 20 of 20 runs (hand-checked against the code).
  - 0 of 40 runs questioned a hint or tested it on the data. The agent writes the note into its requirements
    ("as per instructions").
  - TF03 test RMSE fell from 27.7 million to 4.8 million with the correct hint. The wrong hints did almost no
    harm on these tasks.
- What it is not: the hint is a hand-written prompt, not a memory system (nothing written by the agent, stored or
  retrieved). It says nothing yet about whether the agent uses or ignores a real memory.
- What it suggests: a clever prompt as initial context *might* beat structured and complex memory
  architectures. Any memory architecture has to beat that bar, not only the no-hint baseline. A wrong prompt is
  followed just as fully, so whatever is placed there needs a check first (G07).

### P12: context saturation, lessons in context vs a memory store (250 runs)

[report](results/P12/report.md), [notebook](../docs/notebooks/quick_P12.ipynb),
[memory design](p12/MEMORY_ARCHITECTURE.md)

- Question: while everything fits in the context window, is putting knowledge into the context as good as a
  memory store? Where does that stop?
- Knowledge: 708 lessons the agent wrote from its own P1/P1b runs, after seeing the evaluator's outcome.
- Arms: A0 no lessons; M a minimal memory (top 3 lessons by word-matching search); C1 all the task's own lessons
  in context (about 3k tokens in total); C2 the same inside about 32k tokens of other material; C3 inside about
  100k. 10 tasks, 5 runs each. One 4-GPU server, 131k window.
- Numbers (share of planted-problem check-runs passed, 4 failing SF tasks):
  - A0 0.075, M 0.125, C1 0.75, C2 0.55, C3 0.325.
  - M retrieved the weaker lessons (near-duplicates, one that contradicts the fix). The right lessons were in
    the store but were not picked.
  - TF03 test RMSE: A0 27.7M, C1 4.8M, C2 18.5M, C3 23.1M.
  - Test scores on the no-harm tasks and DSBench barely moved. With large context, runs were shorter (3.0 steps
    in C3 vs 5.9 in C1). They processed 10 times the prompt tokens.
- What it suggests: knowledge in context works when the context is small and focused. It already weakened at
  32k (a quarter of the window) and mostly failed at 100k. So the useful limit is how much other material sits
  around the relevant knowledge, not the window size. A memory store is judged by what it selects; simple word
  matching selected poorly here.

### P0 and P7: old runs (background only)

[P0 report](results/P0/report.md), [P7 report](results/P7/report.md)

These re-mine old DiscoveryBench hypothesis-search runs. That task type is out of scope for now (no ground
truth), so they are context only.
- P0: old `try_next` suggestions were followed in 34% of judgeable cases.
- P7: a dead-end ledger would block 21 of 21 retries on task 5 and 25 of 30 on task 16. But many of the
  blocked attempts had ended "supported".

## Not run yet

| Probe | What it needs before running |
| --- | --- |
| P3 hint delivery | Next. P2 moved every task fully, so the plan's task rule ("5 tasks where P2 moved least") needs a new rule |
| P5 check coverage | Nothing (no model runs); uses P1's fork list |
| P6 transfer | Task pairs with the same data shape |
| P8 size (gemma-4-12B) | A 12B server |
| P11 MaTTS | Fewer reasons to run now: runs already agree, so MaTTS has little to fix |
| X01 memory channel ([plan](../plans/X01_memory_channel_exploration.md)) | P2, P3, P4 done |

## Next steps (for the author to pick)

| Option | Why | Cost |
| --- | --- | --- |
| A. ~~P2 + P4 on the 4 SF tasks with consistent failures~~ Done 2026-11-15: hints followed in 40 of 40 runs | Does a correct hint change the choice, and does a wrong one? (A prompt test, not a memory test.) | 4 tasks x 5 runs x 2 = 40 runs (SF runs averaged 3.8 minutes each in P1) |
| B. P3 mode 3 (a data check in `before_execute`) on the same 4 tasks | P1 suggests the failure is that the agent never looks at the data. A check may matter more than any text | 20 runs + check code |
| C. Harder ground-truth tasks | P1b was too easy. Take DSBench questions with low reported accuracy, or more Silent Failures variants | Selection work + about 100 runs |
| D. Look closer at the 00000029/1 pattern | A case of prompt text overriding the agent's own evidence | Reading transcripts, no runs |
| E. P8 on gemma-4-12B | Does a smaller model fail in more varied ways? | 75 runs and a new server |

After P2/P4: the agent follows a direct prompt at the start, right or wrong. The open question is whether a
real memory does any better than that. A fair test on the same tasks has 4 arms: no memory, a hand-written
prompt, a real memory built from earlier runs, and that memory's entries placed as initial context. Varying the
prompt's wording ("check X" vs "do X") and position (P3) shows how strong the prompt bar is. B (a data check)
stays useful as the guard against wrong entries.

After P12: the agent's own lessons, placed alone in the context, fixed most planted problems (C1). The memory
store lost that gain through poor selection. The gain also faded as unrelated material was added. Open next
questions:
1. Does better selection (duplicate removal, a conflict check, embedding search) bring M up to C1?
2. Is the fade at 32k caused by burying, or by the conflicting advice in the filler? Neutral filler, or the
   lessons at the start, would tell.
3. Does it transfer? Lessons from other tasks only (no same-task runs) is the realistic memory case.

## Limits

- 5 runs per task. Effects smaller than 1 run in 5 cannot be seen.
- Silent Failures checks are judged by the same model that ran the agent. Its 3 draws agreed with each other
  on 79 of 80 run-checks, but it was not checked against a second judge.
- The server is not reproducible even at temperature 0 (P9).
- Several tasks are high contamination risk (P10). DSBench recall was not tested.
