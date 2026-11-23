# P12 lesson audit (read by hand before any P12 run, 2026-11-15)

Lessons: `lessons.jsonl` (708 lessons from 245 P1/P1b runs, 0 parse errors). Written by gemma-4-31B from each
run plus the evaluator's outcome. The Silent Failures outcome lists each check by name with its question, so
these lessons are close to oracle level for those tasks. Counted per earlier run (5 per task):

| Task | Runs whose lessons state the planted fix | Notes |
| --- | --- | --- |
| TF01 | 5 of 5 | value_counts / class_weight='balanced' / stratify=y; every run's lessons |
| TF02 | 5 of 5 | strip the 'cm' suffix before pd.to_numeric |
| TF03 | 3 of 5 | seeds 0, 2, 3 say exclude 'column_x'. Two lessons contradict: seed1.1 "ensure key indicators (like 'column_x') are not accidentally dropped"; seed4.0 "do not manually exclude columns unless told" |
| TF05 | 3 of 5 | seeds 1, 2, 4 name expenditure / active / majorcards; seeds 0 and 3 say "find and drop leakage columns" without naming them |

DSBench analysis lessons get only right/wrong feedback; they are about method (read the sheet layout, check
formulas), not the answer. Retrieval (arm M) returns 3 lessons from the task's own runs for 9 of 10 tasks
(TF06 leak-free: 1 own, 2 from its cue twin), after keying each lesson by its source-task question.
