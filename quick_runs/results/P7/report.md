# P7. Loop-guard replay

> **Scope note (2026-11-11, the author): background only.** This probe re-mines old DiscoveryBench
> hypothesis-search runs. Hypothesis search is out of scope for now; the quick runs focus on
> task-completion problems. Do not build later steps on these numbers.

Executed notebook: [docs/notebooks/quick_P7.ipynb](../../../docs/notebooks/quick_P7.ipynb).

**Question.** Would a dead-end ledger have stopped the repeated dead ends in the old experience-memory runs?

## What was run

- Input: the four old runs under `/data/anon/debug/e01_experience_memory/results/` (tasks 5 and 16; arm E0 = experience memory off, arm E1 = on). One run per task and arm. No model calls.
- An *attempt* is one record in the run's `data/trajectories.jsonl`: one hypothesis tested by one executed experiment and then analysed. The run logs one `trajectory_recorded` event per record, in the same order (checked), so file order is time order. Code crashes that the agent fixed inside the same attempt (`experiment_failed` events) are not separate attempts.
- *Outcome class* of an attempt, first match wins: *error* (experiment did not finish), *rejected* (the analyst's outcome), *null* (the corrector flagged a null result), *inconclusive*, *supported*. A *dead* attempt is error, rejected or null.
- The *ledger key* is the variables tested plus the design fingerprint the old run stored (old `src/memory/design_fingerprint.py`: model family, controls, moderators, transformations, scope). These tasks have no validation step, so validation is not part of the key. An attempt with no recorded design (fingerprint `none`) has no key and is never blocked.
- A *retry* is an attempt whose key appeared earlier in the run.
- The ledger *blocks* an attempt when its key already had a dead attempt earlier. A blocked attempt would not have run, so it does not change the ledger.
- The old notebook's *dead-end retry* counts an attempt whose latest earlier attempt in the same group was rejected or null. Recomputed here as a check.
- A second, coarser key (variables, model family with aliases merged, controls; scope, moderators and transformations ignored) is shown for comparison.

## Numbers

| task | arm | attempts | retries | old dead-end retries | of those blocked | blocked (all) | blocked that ended dead again | blocked that ended supported | no fingerprint |
|---|---|---|---|---|---|---|---|---|---|
| 5 | e0 | 150 | 109 | 4 | 4 | 14/109 (13%) of retries | 1 | 12 | 0 |
| 5 | e1 | 150 | 92 | 21 | 21 | 44/92 (48%) of retries | 14 | 29 | 2 |
| 16 | e0 | 150 | 130 | 4 | 4 | 32/130 (25%) of retries | 0 | 32 | 0 |
| 16 | e1 | 150 | 79 | 30 | 25 | 47/79 (59%) of retries | 17 | 30 | 18 |

Old-notebook check: the recomputed old dead-end retry counts are task 5 e0 = 4, task 5 e1 = 21, task 16 e0 = 4, task 16 e1 = 30.

Coarser key (variables, family, controls):

| task | arm | retries | blocked | blocked that ended dead again | blocked that ended supported |
|---|---|---|---|---|---|
| 5 | e0 | 127 | 56 | 3 | 52 |
| 5 | e1 | 127 | 101 | 28 | 67 |
| 16 | e0 | 138 | 90 | 2 | 88 |
| 16 | e1 | 136 | 115 | 35 | 80 |

## Retries the ledger would miss

A *missed near-duplicate* is an attempt the fingerprint ledger lets through although an earlier dead attempt tested the same variables with almost the same design. Three kinds, closest first:

- *same design, other wording*: same family, controls, moderators and transformations after lower-casing and removing spaces; only the scope text or spelling changed the fingerprint.
- *no design recorded*: the attempt has fingerprint `none`, so no key at all.
- *one element off*: same family; controls, moderators and transformations differ by one item. This is a loose match: adding one control can be a real change.

| task | arm | same design, other wording | no design recorded | one element off | of these, ended dead again |
|---|---|---|---|---|---|
| 5 | e0 | 0 | 0 | 27 | 1 |
| 5 | e1 | 0 | 2 | 61 | 13 |
| 16 | e0 | 0 | 0 | 27 | 1 |
| 16 | e1 | 24 | 18 | 14 | 14 |

Examples (earlier dead attempt, then the later attempt the ledger lets through):

- Task 5 e1, no design recorded: attempt 0 (rejected, fp eb74b17d44c9) `logit; controls=['INCOME', 'GENDER', 'BLACK', 'HISPANIC']; moderators=[]; transforms=['AGE_2']; scope='all respondents'` -> attempt 54 (inconclusive, fp none) `-; controls=[]; moderators=[]; transforms=[]; scope='-'`
- Task 5 e1, no design recorded: attempt 55 (rejected, fp 8c3b0ef33928) `logit; controls=['LOG_INCOME', 'GENDER']; moderators=[]; transforms=[]; scope='all respondents'` -> attempt 69 (inconclusive, fp none) `-; controls=[]; moderators=[]; transforms=[]; scope='-'`
- Task 5 e1, one element off: attempt 1 (rejected, fp 1fd4ee2bc711) `logit; controls=['GENDER', 'BLACK', 'HISPANIC']; moderators=['INCOME']; transforms=['log(INCOME)']; scope='all respondents'` -> attempt 6 (rejected, fp 1f4fca324fc8) `logit; controls=['BLACK', 'HISPANIC']; moderators=['INCOME']; transforms=['log(INCOME)']; scope='all respondents'`
- Task 5 e1, one element off: attempt 1 (rejected, fp 1fd4ee2bc711) `logit; controls=['GENDER', 'BLACK', 'HISPANIC']; moderators=['INCOME']; transforms=['log(INCOME)']; scope='all respondents'` -> attempt 11 (supported, fp cbae50ef2be9) `logit; controls=['GENDER', 'BLACK', 'HISPANIC']; moderators=['SAMESAVE', 'INCOME']; transforms=['log(INCOME)']; scope='all respondents'`
- Task 16 e1, same design other wording: attempt 30 (rejected, fp 53d7475d15b1) `ols; controls=[]; moderators=[]; transforms=[]; scope='all species in the dataset'` -> attempt 32 (supported, fp 1ae9f13e5e71) `ols; controls=[]; moderators=[]; transforms=[]; scope='all rows in the dataset'`
- Task 16 e1, same design other wording: attempt 30 (rejected, fp 53d7475d15b1) `ols; controls=[]; moderators=[]; transforms=[]; scope='all species in the dataset'` -> attempt 36 (supported, fp 1ae9f13e5e71) `ols; controls=[]; moderators=[]; transforms=[]; scope='all rows in the dataset'`

## What it suggests

1. In E1 the ledger blocks the repeated dead ends the old notebook counted: task 5 21 of 21; task 16 25 of 30 (5 of the 5 it misses have no recorded design, so no key).
2. It blocks more attempts than those, because a key stays dead once it has failed: 44/92 (48%) of retries on task 5 E1 and 47/79 (59%) on task 16 E1.
3. The cost: many blocked attempts in fact ended supported (29 of 44 on task 5 E1, 30 of 47 on task 16 E1). The key holds the variables and the design, not the claim. The same design on the same variables can end rejected for one hypothesis and supported for the next. A hard block would remove those later successes.
4. The exact fingerprint misses near-duplicates of dead attempts. Task 5 E1: 0 same design with other wording, 2 with no recorded design, 61 one element off. Task 16 E1: 24 same design with other wording, 18 with no recorded design, 14 one element off.
5. Outcome classes seen: {'rejected': 109, 'supported': 481, 'inconclusive': 10}. 41 of 41 corrector null flags sit on attempts that were also rejected, so they fall in the rejected class.

## Caveats

- Two tasks, one run per arm. These are counts, not estimates.
- The replay is counterfactual only in a simple sense: we count what would be blocked. We cannot know what the agent would have done instead.
- The fingerprints are the ones the old runs stored. They were not recomputed.
