# Hand check: executed-decision extractor vs the code (before P1)

Date: 2026-11-11. Runs: 5 development runs (probe `dev`, seed label 100, not used in any result),
one per task: sf_tf01_goal_imbalance, sf_tf02_player_rep_norm, sf_tf05_credit_card_leakage,
blade_caschools, blade_crofoot. Raw files: /data/anon/reveries_runs/dev/<task>/seed100/.

Method: I read every executed code cell and the final answer, and compared each of the 11 decision
types in the extractor's output (executed.json) with what the code did.

## Result: 52 of 55 decisions match (0.95)

| Run | Match | What did not match |
| --- | --- | --- |
| sf_tf01_goal_imbalance | 11/11 | - |
| sf_tf02_player_rep_norm | 11/11 | - |
| sf_tf05_credit_card_leakage | 11/11 | (transformation lists the yes/no label mapping as "derive:card"; counted as correct) |
| blade_caschools | 11/11 | - |
| blade_crofoot | 8/11 | variables: gives only rel_size, but the final answer rests on rel_size and rel_dist (fitted in separate models); transformation: misses that rel_size is a log ratio; final_selection: names one of the two final models |

## What this means for the metrics

1. On the prediction tasks (Silent Failures) and on a run with one final model, the extractor is
   reliable.
2. When a BLADE run fits several small models and answers from more than one, the extractor picks
   one. Agreement on variables / final_selection for such runs is a lower bound.
3. Transformation values are fine-grained ("derive:rel_size" vs "derive:log_rel_size"), so agreement
   on transformation will be low partly by construction. Report it, but do not lean on it.

## Things seen along the way (not metrics)

- sf_tf02: the agent ran `pd.to_numeric(errors='coerce')` on height_in_cm and international_caps. That
  turns '185cm' into a missing value (then mean-imputed) instead of stripping the suffix. This is the
  planted failure, done silently.
- sf_tf05: the agent used all columns, including the four leakage columns (expenditure, share, active,
  majorcards); test accuracy 0.966.
- Stated vs executed (smoke run on blade_fish, 2026-11-11): the agent stated "Negative Binomial" twice
  and then fitted a Poisson GLM with robust errors; the extractor reported poisson.
