# P0. Re-mine old runs

> **Scope note (2026-11-11, the author): background only.** This probe re-mines old DiscoveryBench
> hypothesis-search runs. Hypothesis search is out of scope for now; the quick runs focus on
> task-completion problems. Do not build later steps on these numbers.

Executed notebook: [docs/notebooks/quick_P0.ipynb](../../../docs/notebooks/quick_P0.ipynb). Run list: [run_folders.csv](run_folders.csv).

**Question.** When the old multi-agent system ran the same DiscoveryBench task more than once, did the runs split at analysis decisions, and at which decision types?

## What was run

1. Run folders. All 387 folders under the four old result trees were listed. 263 are scoring twins (they only link to another run's event log), so they were dropped. That leaves 124 generation runs over 25 tasks. Every task was run at least twice and every run executed code, so all 124 are used. Runs per task: 3 runs for 14 tasks, 5 runs for 1 tasks, 6 runs for 5 tasks, 9 runs for 4 tasks, 11 runs for 1 tasks.
2. What each run gave the extractor. Each old run is a 10-iteration search over many hypotheses (about 150 experiments). The old synthesis step ranked the hypotheses (supported first, then by confidence) and built the final answer from the top 4. We used the same ranking. For each of the top 4 hypotheses we took the last experiment that ran without error, in run order. The final answer is the old p15 synthesis text. The full code was recovered from the Engineer's reply, because the event log keeps only the first 500 characters.
3. The post-hoc extractor (`reveries.decisions.extract_executed`, gemma-4-31b-it, temperature 0, thinking off, one draw) read the code and the final answer and returned one value per decision type. Values were put in canonical form (`reveries.decisions.canonical`).
4. Extracted runs: 124 of 124. Failures: none.

Terms:

- *Agreement* at a decision type, for one task: the share of run pairs that chose the same canonical value. A run with no value for the type is left out. `quick_runs/metrics.py` computes it.
- *Pooled agreement*: the mean of the per-task agreements, each task counted once. The pair-weighted version counts every run pair once, so tasks with many runs weigh more.
- *First divergence*: the first decision type, in the fixed type order, where two runs of a task differ. The old runs have no DECISION lines, so the fixed order stands in for the run's own order.

## Numbers

All runs (124 runs, 25 tasks). The v8 columns use only the three v8 arms (75 runs, 25 tasks, same code version, 3 runs per task).

| decision type | pooled agreement | pair-weighted | tasks where all runs agree | tasks where no two runs agree | v8 only: pooled | v8 only: all agree |
|---|---|---|---|---|---|---|
| framing.target | 0.74 | 0.81 | 14 / 25 | 1 / 25 | 0.80 | 18 / 25 |
| framing.population | 0.07 | 0.12 | 0 / 25 | 18 / 25 | 0.05 | 0 / 25 |
| framing.unit | 0.67 | 0.60 | 12 / 25 | 0 / 25 | 0.69 | 14 / 25 |
| variables | 0.22 | 0.18 | 2 / 25 | 8 / 25 | 0.20 | 2 / 25 |
| transformation | 0.10 | 0.09 | 0 / 25 | 14 / 25 | 0.12 | 1 / 25 |
| model.family | 0.79 | 0.87 | 17 / 25 | 0 / 25 | 0.84 | 19 / 25 |
| model.controls | 0.27 | 0.11 | 5 / 25 | 10 / 25 | 0.36 | 7 / 25 |
| validation | 1.00 | 1.00 | 25 / 25 | 0 / 25 | 1.00 | 25 / 25 |
| metric | 0.78 | 0.90 | 16 / 25 | 2 / 25 | 0.73 | 16 / 25 |
| verdict | 0.82 | 0.86 | 16 / 25 | 0 / 25 | 0.76 | 17 / 25 |
| final_selection | 0.03 | 0.06 | 0 / 25 | 18 / 25 | 0.05 | 1 / 25 |
| source.target (column view) | 0.81 | 0.92 | 18 / 25 | 2 / 25 | 0.81 | 19 / 25 |
| source.variables (column view) | 0.30 | 0.24 | 2 / 25 | 4 / 25 | 0.24 | 2 / 25 |
| source.controls (column view) | 0.30 | 0.14 | 5 / 25 | 9 / 25 | 0.37 | 7 / 25 |
| model.all_families (extra view) | 0.42 | 0.56 | 3 / 25 | 4 / 25 | 0.39 | 5 / 25 |

The rows marked *column view* are not decision types. They compare the original dataset columns behind the target, the main variables and the controls, so a renamed derived column does not count as a split. `model.all_families` is every model family the run fitted, not only the final one.

**The 3 types where runs split most** (lowest pooled agreement, all runs): `final_selection` (0.03); `framing.population` (0.07); `transformation` (0.10). On v8 only: `final_selection` (0.05); `framing.population` (0.05); `transformation` (0.12).

**First divergence** (number of tasks, all runs): `framing.population` 14, `framing.target` 11.

## What it suggests

1. High agreement (pooled 0.80 or more): `validation` 1.00, `verdict` 0.82.
2. Middle (0.30 to 0.80): `framing.target` 0.74, `framing.unit` 0.67, `model.family` 0.79, `metric` 0.78.
3. Low agreement (below 0.30): `framing.population` 0.07, `variables` 0.22, `transformation` 0.10, `model.controls` 0.27, `final_selection` 0.03.
4. In the column view, main variables agree at 0.30 and controls at 0.30. Compare with `variables` 0.22 and `model.controls` 0.27: the gap is the part due to naming of derived columns.
5. The low-agreement types are where repeated runs fork. They are the candidates for fork memory and for the checks in P5. `transformation` is a list of many small steps, so its agreement is low almost by construction; read it with that in mind.

## Caveats

- **These runs differ by arm, not only by seed.** The arms change memory, retrieval settings, the corrector and design discipline, and the v8 and v10 runs come from different code versions. Part of the split may come from the arms. The v8-only columns hold the code version fixed but still mix three memory arms.
- Each old run explores many hypotheses. The top-ranked hypotheses of two runs can be different questions, so a split here can mean "answered a different sub-question", not only "chose a different analysis for the same question".
- Free-text types (`framing.population`, `framing.unit`, `final_selection`) are compared as lower-cased strings, so wording differences count as splits. Treat their agreement as a lower bound.
- The extractor ran once per run at temperature 0. It was not hand-checked against the code for these runs.

## What the old memory showed the agent (descriptive)

99 of the included runs had memory on. We looked at the prompt in which the Engineer (the agent that writes the experiment code) designs each experiment. Two kinds of memory item appear there, and both are shown verbatim in that prompt:

- *Code memories* (52 runs, the v8 memory arms and one e01 arm): past experiments' code under "Relevant memories".
- *Design-digest items* (47 runs, the v10 and e01 arms): one line per past design with its outcome tally, plus a `try_next` suggestion from the analyst's reflection.

How we judged "followed" (a fixed rule, no model and no hand reading):

- Code memory: we reduce the shown code and the code the Engineer then wrote to a signature (model families called, plus the names in model formulas and feature lists). Same signature = *followed*. Both readable but different = *ignored*. Either unreadable (for example a t-test with no formula) = *cannot tell*.
- Design-digest item: we take the dataset columns that the `try_next` text names and the item's own design does not already use. If the design in the same prompt names at least one of them = *followed*. If it names none = *ignored*. If `try_next` names no new column = *cannot tell*. Only items about the same variables as the hypothesis being designed are judged.

Each item is judged every time it is shown, so one item shown in ten prompts counts ten times.

Examples (verbatim, shortened to one line):

- Task 0, v8_transfer:mem_all_graphon_refsum_v8, code memory. Shown: `formula = "BMI ~ DISSAVED + AGE + AGE_2 + INCOME + GENDER_MALE + BLACK + HISPANIC"`. Next design: `formula = "BMI ~ DISSAVED + AGE + AGE_2 + INCOME + GENDER_MALE + BLACK + HISPANIC"`. Judged: **followed**.
- Task 0, v8_transfer:mem_all_graphon_refsum_v8, code memory. Shown: `formula = "BMI ~ BEHAVIOR + INCOME_GROUP_High + BEHAVIOR:INCOME_GROUP_High + AGE + BLACK + HISPANIC"`. Next design: `formula = "BMI ~ BEHAVIOR * INCOME_GROUP + AGE + BLACK + HISPANIC"`. Judged: **ignored**.
- Task 1, v8_transfer:mem_all_graphon_refsum_v8, code memory. Shown: `t_stat, p_val = stats.ttest_ind(dissaved_group, not_dissaved_group, equal_var=False)`. Next design: `t_stat, p_val_ttest = stats.ttest_ind(dissaved_group, not_dissaved_group, equal_var=False)`. Judged: **cannot tell**.
- Task 5, v10_stage2:c_v10_v10, design digest. Shown: `logit; controls: GENDER, INCOME, BMI; moderators: BLACK, HISPANIC; transforms: -; scope: all respondents [8 supported, 1 inconclusive] try_next: Explore the non-linear relationship of age by including the AGE_2 term in the logit model.`. Next design: `logit; controls: log(INCOME), BMI, AGE_2; moderators: BLACK, HISPANIC; transforms: -; scope: all respondents`. Judged: **followed**.
- Task 5, v10_stage1:r_v10_v10, design digest. Shown: `logit; controls: GENDER, BMI; moderators: LOG_INCOME, BLACK, HISPANIC; transforms: -; scope: all respondents [1 supported] try_next: Include AGE_2 to determine if the relationship between age and dissaving is quadratic rather than linear.`. Next design: `logit; controls: -; moderators: INCOME; transforms: -; scope: all respondents`. Judged: **ignored**.

Counts:

- Code memories: 12004 shown. Followed 7882, ignored 484, cannot tell 3638. Where it can be told, followed 7882/8366 (94%).
- But 100% of the code memories shown were earlier experiments for the *same* hypothesis the Engineer was designing (retrieval was scoped to it). So "followed" here mostly means the Engineer re-ran its own earlier code. For comparison, the new code matched an earlier experiment that was *not* shown 2.4% of the time.
- Design-digest items about the same variables: 29973 shown. Followed 2421, ignored 4732, cannot tell 22820. Where it can be told, followed 2421/7153 (34%).
- Per prompt (one prompt shows several items): of 3491 prompts with at least one item that can be told, 1678 followed at least one suggestion and 1813 followed none.

Limits of this judgment: a signature match can happen without the memory (the hypothesis fixes the variables). A design can follow advice without naming a column (for example "try a nonparametric test"), which the rule calls *cannot tell*. The Researcher and Analyst also got memory sections (facts, past results); those are not judged here.
