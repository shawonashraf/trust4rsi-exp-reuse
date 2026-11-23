# Probe: DeepSeek agent + Claude guide (NOT for the paper)

> **Superseded by `report.md` (2026-11-19).** This note treats the "G3cds" arm as guided for all 10 seeds, but the
> P20 Opus picks cover seeds 0-4 only, so seeds 5-9 ran without lessons; and its pooled shares are of 80 check-runs,
> not 40. The valid comparison (seeds 0-4) is 0.425 without and 0.45 with the Opus guide.

Branch `probe/deepseek-agent-claude-guide`. Agent = DeepSeek-V41-Flash-1017-FP8 (SPIKE, OpenAI-compatible);
guide = Claude Opus 5.5 (reused P20 `G3c.yaml`, clean store, original prompt). Judge = Claude Haiku 5.5 (GPU-free;
every local GPU was taken by other users). 4 failing tasks, 10 seeds. All 80 runs ended with a final answer.

## Claude-judged targeted checks (passes / 10)

| check | A0ds (DeepSeek, no memory) | G3cds (DeepSeek + Opus guide) | Gemma no-mem (paper) |
| --- | --- | --- | --- |
| TF01 class weights | 7 | 7 | 0 |
| TF02 strip the unit | 9 | 10 | 0 |
| TF03 drop column_x | 1 | 2 | 0 |
| TF03 causal reasoning | 1 | 1 | 0 |
| TF05 drop leakage | 0 | 0 | 0 |
| **Pooled (of 40)** | **0.425** | **0.475** | **0.075** |

## Read
1. A capable agent clears most silent-failure pitfalls unprompted: DeepSeek no-memory 0.425 vs Gemma no-memory 0.075.
   It strips the '185cm' unit 9/10 and weights classes 7/10 on its own.
2. Opus clean-store guidance adds almost nothing (0.425 -> 0.475, within noise): that guide rejects the fixes
   (instruction-ranking), so it has no useful lesson to pass. Consistent with G3c Gemma 0.125.
3. The two hardest pitfalls stay unsolved even for a strong agent: TF03 column_x 1-2/10, TF05 leakage 0/10. These need
   the right lesson, which an instruction-ranking guide drops.

So the paper's premise holds from the other side: memory matters most for a weak agent; a strong agent needs it far
less, and an instruction-ranking guide still fails to deliver the hard fixes.

## Cost
DeepSeek agent (SPIKE): ~0.4M prompt / 0.3M completion per 40 runs (trivial vs the 110M/11M weekly budget).
Claude Haiku judge: $0.10 total (240 calls). No new Anthropic guide spend (reused P20).
