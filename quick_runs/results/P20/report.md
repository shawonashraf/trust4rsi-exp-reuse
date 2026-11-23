# P20: a frontier, different-family guide (Claude Opus 5.5, Fable 5.1)

Notebook: [docs/notebooks/quick_P20.ipynb](../../../docs/notebooks/quick_P20.ipynb). Predictions (committed before the
runs): [quick_runs/predictions/P20.md](../../predictions/P20.md). Numbers come from `summary.json`; every judgement and
reason is in `judgements.csv`; token-and-cost ledgers are in `ledger_opus.jsonl` and `ledger_fable.jsonl`.

**One-line answer.** The instruction-ranking that loses the real fixes is the same across three model families: Gemma,
Opus 5.5 and Fable 5.1 all reject the TF03 and TF05 fixes on the clean store (fix shown 1 of 5 and 0 of 5, the
misleading lesson kept in 4 of 5), and cite the instructions. What differs is the agreeing-wrong-lesson trap, the
paper's sharpest result: Gemma passes them 20 of 20, Fable 9 of 20, Opus 1 of 20. A frontier judge rejects plausible
but wrong lessons from schema, filename and domain cues, with no data profile.

## What was run

The guide is a Claude model through the Anthropic API; the agent stays Gemma 4 31B, served locally. Original P13
prompt, R = 3, three stores (clean, P13 wrong lessons, agreeing wrong lessons), 5 runs per task. 140 Claude guide
runs (70 Opus, 70 Fable) and 140 Gemma agent runs. Cost: **Opus 5.5 $4.76, Fable 5.1 $16.77, total $21.53** of the
$100 credit, under a $60 hard cap. Token use per guide run: Opus 5.9k prompt / 2.2k completion; Fable 7.0k / 3.4k.

## Result 1: the instruction-ranking on the clean store is identical across families

| Clean store, runs of 5 | G3 Gemma | G3c Opus 5.5 | G3f Fable 5.1 |
| --- | --- | --- | --- |
| TF03 fix shown | 1 | 1 | 1 |
| TF03 keep-`column_x` shown | 4 | 4 | 4 |
| TF05 fix shown | 0 | 0 | 0 |
| Fix rejections citing the instructions | 83% | 62% | 72% |

All three guides reject the fixes whose use would break the wording, keep the misleading "keep `column_x`" lesson, and
give the instructions as the reason. Qwen3.8-27B (P15) did the same (0 of 10, 64%). The behaviour is not a property of
one model or one family.

## Result 2: a stronger guide resists the agreeing wrong lessons that Gemma passed

| Agreeing wrong lessons, of 20 | G3-pc Gemma | G3c-pc Opus | G3f-pc Fable |
| --- | --- | --- | --- |
| Shown to the agent | 20 | **1** | **9** |
| Pooled targeted checks (of 40) | 0.0 | **0.525** | 0.35 |

- Gemma passed every agreeing wrong lesson (20 of 20) and scored 0.0. Opus blocked all but one (1 of 20, Fisher
  p < 0.0001 against Gemma) and scored 0.525, the best of any guide arm; Fable blocked about half (9 of 20, p = 0.0001).
- Opus rejects them by reasoning about the data it is not shown: "expenditure and share are recorded after the card
  decision, so they do leak the label; this claim is factually wrong" (TF05); "the '5high' name suggests a threshold,
  likely imbalanced" (TF01); "height_in_cm is stored as str, so plain coercion would silently drop valid values"
  (TF02). It uses the schema dtypes, the file names and domain knowledge, not a profile.
- In the agreeing store Opus also accepted the TF03 fix in 5 of 5 and left `column_x` out (RMSE 4.78e6). So where it
  can tell a wrong lesson from a right one, a frontier judge does.

## Result 3: the contradicting wrong lessons are blocked completely

| P13 wrong lessons, of 20 | G3-p Gemma | G3c-p Opus | G3f-p Fable |
| --- | --- | --- | --- |
| Shown to the agent | 5 | 0 | 0 |

Both Claude guides blocked every wrong lesson that contradicts the instructions (0 of 20, against Gemma's 5 of 20).

## Result 4: but on the clean store the stronger guide is stricter and scores lower

| Pooled targeted checks (of 40) | G3 Gemma | G3c Opus | G3f Fable |
| --- | --- | --- | --- |
| Clean store | 0.50 | **0.125** | 0.30 |

- Opus scored 0.125, below Gemma's 0.50 (p = 0.0006). On TF01, where the fix does not conflict with the wording,
  Opus rejected the class-weight and class-distribution lessons as generic or redundant (0 of 5 each, against Gemma's
  5 of 5), so the agent lost fixes it would otherwise have had. A stricter judge drops more lessons, good and bad.
- So capability cuts both ways: the stronger guide catches more wrong lessons (Results 2, 3) but also rejects more
  correct ones on the clean store (Result 4). It does not simply "do better".

## Predictions (written before the runs)

| # | Prediction | Result |
| --- | --- | --- |
| 1 | G3c, G3f fix on TF03 + TF05 at most 3 of 10 | **Held.** 1 of 10 each. |
| 2 | Fix rejections cite instructions at least 50% | **Held.** Opus 62%, Fable 72%. |
| 3 | Contradicting wrong lessons shown at most 10 of 20 | **Held.** 0 of 20 each. |
| 4 | Agreeing wrong lessons shown at least 12 of 20 | **Failed.** Opus 1, Fable 9. |
| 5 | G3c, G3f pooled within 0.15 of 0.50 | **Failed.** Opus 0.125, Fable 0.30. |
| 6 | TF04 / TF06 within 0.01 of A0 in G3c, G3f | **Partly.** Fable yes; Opus TF06 0.922 (it changed the selected model). |

3 held, 3 failed. The failures are the finding: the mechanism generalises, but its two consequences (the agreeing-poison
trap, and the clean-store usefulness) are capability-dependent. Running total across P1-P22: 56 held, 23 failed, 1 not
testable (of 80).

## What this means

1. The core mechanism generalises. A usefulness judgment by an instruction-tuned model rejects the fixes that require
   breaking the wording, across three families (Gemma, Qwen, Claude) and up to a frontier model, with the same reasons.
2. The paper's sharpest vulnerability, agreeing wrong lessons passing the gate, is capability-dependent. A frontier
   judge rejects most of them without data, by reasoning from the schema, file names and prior knowledge. The 20-of-20
   result is a property of a ~30B open judge, not of all LLM gates. The paper must say this.
3. A stronger judge is not uniformly better: on the clean store Opus rejected useful TF01 lessons too, and scored below
   Gemma. Strictness helps against poison and hurts against terse correct lessons.
4. For cost: a frontier guide on this pilot is a few dollars for 70 runs; the gate itself is cheap relative to the
   agent runs.

## Limits

- 5 runs per cell; the Claude guides use their own default sampling (adaptive thinking, effort high), so they differ
  from Gemma in more than scale.
- No data profile was given to the Claude guide (original prompt), so this tests instruction-ranking, not whether a
  frontier judge uses a profile better.
- One run per model; spend was capped and logged.
