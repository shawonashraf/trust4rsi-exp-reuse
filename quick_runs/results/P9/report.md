# P9 serving noise floor

Notebook: [docs/notebooks/quick_P9.ipynb](../../../docs/notebooks/quick_P9.ipynb). Predictions:
[quick_runs/predictions/P9.md](../../predictions/P9.md). Numbers below are from `summary.json`.

## What was run

Greedy decoding (temperature 0). With greedy decoding the model always picks its most likely next
token, so in theory every rerun is identical. Any difference comes from the server (how requests are
batched together, and the order of floating-point sums), not from sampling.

- 3 tasks: sf_tf01_goal_imbalance, sf_tf03_correlation_causation, blade_crofoot.
- Per task, 3 reruns sent at the same time to one server ("batched", the normal setting), plus 1 run
  alone on an idle server, one request at a time ("bs1", batch size 1).
- 12 runs. Everything else as in P1 (same prompt, agent, step budget).

## Result

1. **Greedy reruns are almost never identical.** Of 9 comparisons against the first batched run, 1 was
   identical (TF03, batched_c). In the other 8, the text already differs in the first message for both
   Silent Failures tasks (after 90 to 540 characters). For crofoot it differs in the first, third or
   fourth message.
2. **Once the text differs, the runs drift.** The 4 crofoot runs took 15, 10, 8 and 10 steps.
3. **Answers differ on the BLADE task.** The 4 greedy crofoot runs reached 3 different conclusions:
   - "both positive, supported" (1 run),
   - "mixed, supported" (2 runs),
   - "no significant effect, not supported" (1 run).
   Answer consistency (share of run pairs with the same direction and verdict) is 0.17.
4. **Answers are more stable on the prediction tasks.** TF03: all 4 runs report the same RMSE
   (answer consistency 1.0). TF01: 3 of 4 runs report macro F1 0.792 and one reports 0.777 (it picked
   a different set of feature columns); answer consistency 0.5.
5. **What agrees and what does not (mean over the 3 tasks).** Full agreement (1.0) on framing.target,
   framing.population, framing.unit, model.family, model.controls and validation. Lower on variables
   (0.33), final_selection (0.44), transformation (0.50), verdict (0.50) and metric (0.72).
6. **The batch-size-1 run is no closer to the batched runs than they are to each other.** It differs
   from batched_a on all 3 tasks. So batching is not the only cause; the server is not deterministic
   even for single requests in this setup.
7. **Same failures every time.** Every TF01 run failed the same 4 of 5 imbalance checks (no look at the
   class distribution, no resampling, no stratified split, no class weights). Every TF03 run kept the
   spurious column_x and did no causal reasoning. These are failures that repeat even when the text
   differs.

## Against the predictions

| Prediction | Outcome |
| --- | --- |
| 1. Reruns not identical; differ after a few hundred tokens, then drift | Held, and earlier than predicted: often within the first 100 characters. |
| 2. Agreement higher than in P1 | Mixed. Answer consistency, greedy (P9) vs sampled (P1): crofoot 0.17 vs 1.0, TF01 0.5 vs 0.1, TF03 1.0 vs 1.0. Greedy decoding did not make crofoot more stable. |
| 3. bs1 differs about as much as batched reruns | Held. |
| 4. framing.target full agreement; variables/transformation/controls first to differ | Mostly held. Target agreed fully; variables was the first divergence on all 3 tasks. Controls agreed fully. |

## What it suggests

1. There is no "same seed, same output" baseline on this server. The seed field is ignored, and even
   greedy decoding is not reproducible. So run-to-run differences in P1 are a mix of sampling and
   serving noise. P9 is the floor: differences of this size in P1 are not the agent's choice.
2. A small difference early (a different feature list in the first code cell) is enough to change a
   BLADE verdict. On crofoot the verdict itself is fragile, even at temperature 0.
3. Consistent failures survive noise. The planted pitfalls in TF01 and TF03 were missed in every run.
   For memory, this means the target is not only "make runs agree" but "make them agree on the right
   choice".
4. Caveat: 3 tasks and 4 runs each. If we need exact reproducibility later, SGLang's deterministic
   inference mode (`--enable-deterministic-inference`) would be the thing to test; it was not used here.
