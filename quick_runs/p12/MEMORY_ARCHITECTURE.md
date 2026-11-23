# P12 memory architecture

Date: 2026-11-15. Used by probe P12 (context saturation). Design and predictions:
[quick_runs/predictions/P12.md](../predictions/P12.md).

**In one line.** A minimal ReasoningBank-style memory. After each earlier run, the agent writes 1 to 3 lessons
using the evaluator's outcome. The lessons are stored as records. At the start of a new run, the 3 most similar
lessons are retrieved with BM25 and put at the top of the first prompt.

It is deliberately minimal: no embeddings, no merging of duplicate lessons, no updates after a run, no check on
whether a lesson is true. P12 compares it with putting lessons straight into the context, so the memory only
needs to be a fair, simple version of "store + retrieve + inject".

## Overview

```
WRITE (once, offline)                 STORE                           READ (at the start of each new run)
---------------------                 -----                           -----------------------------------
earlier run (P1 / P1b)                lessons.jsonl                   new task question
  + evaluator outcome      ──►        708 records         ──►         BM25 over all records, top 3
  lesson writer (gemma)               (frozen)                        format as notes
  1-3 lessons per run                                                 put before the task in the first user message
```

## 1. Write path

Code: `quick_runs/p12/lessons.py`. Output: `quick_runs/results/P12/lessons.jsonl`.

**Input per run.** For each of the 245 runs in P1 (15 tasks x 5) and P1b (34 tasks x 5):
1. the task question,
2. what the agent did: code cells, their outputs and the final answer (the `transcript()` used by the judge,
   cut to 24,000 characters: the first third and last two thirds are kept),
3. the evaluator's outcome for that run:

| Source | Outcome shown to the writer |
| --- | --- |
| Silent Failures | Each check by name, PASSED / FAILED / could not be judged, with the check's question; the test-set score |
| BLADE | Whether the outcome variable, main predictor and model family match an expert's choice |
| DSBench analysis | The final answer was CORRECT / INCORRECT / missing. The gold answer is never shown |
| DSBench modelling | Whether the submission was valid, and the relative score (0 = baseline, 1 = best possible) |

**Writer.** gemma-4-31B-it, thinking off, temperature 0, at most 1,500 output tokens, one call per run. The
prompt (`WRITER` in `lessons.py`) asks for 1 to 3 lessons that would help on this task or a similar one,
learning from what went wrong and right. It says not to state the final answer or numeric result. The reply must
be JSON: `{"lessons": [{"title", "when", "content"}]}`. At most 3 lessons per run are kept.

**Result.** 708 lessons from 245 runs, 0 parse errors. Written once on 2026-11-15, then frozen. The memory does not
change during P12.

## 2. Store

One JSON line per lesson:

```json
{"source_task": "blade_boxes",        // task the lesson came from
 "source_probe": "P1",                // which probe's run
 "source_run": "seed0",               // which run
 "k": 0,                              // position of the lesson within that run's reply (0-2)
 "title": "Handle Outliers and Low-Sample Groups Early",
 "when": "when performing group-based analysis or regression with age/categorical variables",
 "content": "Check the distribution of your independent variables ... before fitting the primary model ..."}
```

Nothing else is stored: no scores per lesson, no use counts, no embeddings.

## 3. Read path

Code: `quick_runs/p12/contexts.py`. Output: `/data/anon/reveries_runs/P12_inputs/M.yaml` ({task_id: text}).

**Index.** Each lesson is indexed by the question of the task it came from, plus its title, "when" and content.
This follows ReasoningBank, which keys a memory by the query it was built from. A first version indexed the
lesson text alone; it retrieved lessons from other tasks (0 own-task lessons for 4 of the 6 Silent Failures
tasks), because lessons are written in general terms. It was changed before any P12 run.

**Search.** Okapi BM25 (k1 = 1.5, b = 0.75). Tokens are lower-case runs of letters, digits and underscores. The
query is the new task's question (without the file list, formatting instructions or schema). The pool is all 708
lessons, including those from the same task's earlier runs. The top 3 are returned. Ties keep file order.

**What it returns for the 10 P12 tasks.** 3 lessons from the task's own earlier runs for 9 tasks. For TF06
leak-free, 1 own lesson and 2 from its twin task (TF06 with the original file names).

## 4. Injection

The retrieved lessons are put at the very start of the first user message, before the task question. They are
given once, at step 0. Nothing is added later in the run. Format (same in every arm):

```
Notes from previous experience (written after earlier runs of data science tasks; each note says which task it came from):

[Note from a run on: <first 160 characters of the source task's question>]
<title>
When: <when>
What to do: <content>

[Note from a run on: ...]
...

<task question, data files, formatting instructions, data schema>
```

The agent's system prompt is unchanged. It does not mention notes or memory.

Mechanism in the agent: `Hint(text, where="start", prefix="")` in `reveries/agent.py`;
`initial_user_message()` builds the message. The runner passes it with
`--hint-file <arm>.yaml --hint-prefix ""`. Tested in `tests/test_agent_hint.py`; retrieval in
`tests/test_p12_bm25.py`.

## 5. The comparison arms (same lessons, no memory architecture)

| Arm | What goes at the start of the prompt | Size (tokens, 10 tasks) |
| --- | --- | --- |
| A0 | Nothing | 0 |
| M | The memory: top 3 lessons from the whole store (above) | 362 to 457 |
| C1 | All lessons from this task's earlier runs (7 to 15), no retrieval | 1,132 to 2,144 |
| C2 | C1's lessons in the middle of other material, about 32k in total | 31,704 to 31,953 |
| C3 | The same, about 100k in total | 99,730 to 99,934 |

The filler for C2 and C3 is other tasks' lessons plus logs of other tasks' earlier runs (code, outputs, final
answer; up to 12,000 characters each), in a fixed shuffled order (seed = task id). The own-task lessons sit in the
middle. Material from the same task is never used as filler; material from its twin task can be. Token counts
use the gemma-4-31B tokenizer. Each arm's exact text has a SHA-256 in `quick_runs/results/P12/contexts.json`.

## 6. Known limits

1. **Feedback is close to an oracle on Silent Failures.** The outcome names each failed check and its question,
   so most lessons state the fix (see `quick_runs/results/P12/lesson_audit.md`: TF01 and TF02 5 of 5 runs, TF03
   and TF05 3 of 5). This is the best case for any memory. DSBench analysis lessons get only right/wrong.
2. **Lessons can contradict each other.** TF03 has two lessons saying not to drop `column_x` or other columns,
   next to three saying to drop it. Nothing removes or flags this.
3. **No deduplication.** Retrieval can return near-copies. For TF02 all 3 retrieved lessons are the weaker "check
   how many NaNs coercion created" lesson; the sharper "strip the 'cm' suffix" lessons are in C1 but not in M.
4. **Same-task memory.** The lessons come from earlier runs of the same task. Transfer to a new task is not tested.
5. **No verification.** A wrong lesson is shown as readily as a right one (P4 showed the agent follows a wrong
   note in the prompt).
6. **Lexical retrieval.** BM25 matches words. Tasks with similar wording (the player-data tasks) can pull each
   other's lessons.

## 7. How to rebuild

```
uv run --no-active python quick_runs/p12/lessons.py     # needs the server; overwrites lessons.jsonl
uv run --no-active python quick_runs/p12/contexts.py    # deterministic given lessons.jsonl
```

Server for P12: one SGLang server, gemma-4-31B-it, 4 x A6000 (TP = 4), 131,072-token window, KV pool 128,352
tokens. Two 2-GPU servers could not hold a 100k prompt (their pool was about 15.5k tokens each).
