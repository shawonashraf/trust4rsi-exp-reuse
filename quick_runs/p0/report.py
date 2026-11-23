"""Render quick_runs/results/P0/report.md from summary.json (every number comes from there).

Usage: uv run --no-active python quick_runs/p0/report.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "quick_runs/results/P0"


def f2(x):
    return "-" if x is None else f"{x:.2f}"


def main():
    s = json.loads((OUT / "summary.json").read_text())
    runs = pd.read_csv(OUT / "run_folders.csv")
    inc = runs[runs.included]
    a, v8 = s["results"]["all"], s["results"]["v8_only"]
    v8p = {p["type"]: p for p in v8["pooled"]}
    rpt = Counter(a["runs_per_task"].values())
    L = ["# P0. Re-mine old runs", "",
         "Executed notebook: [docs/notebooks/quick_P0.ipynb](../../../docs/notebooks/quick_P0.ipynb). "
         "Run list: [run_folders.csv](run_folders.csv).", "",
         "**Question.** When the old multi-agent system ran the same DiscoveryBench task more than once, did the "
         "runs split at analysis decisions, and at which decision types?", "",
         "## What was run", "",
         f"1. Run folders. All {len(runs)} folders under the four old result trees were listed. "
         f"{int((runs.why.str.startswith('excluded: scoring twin')).sum())} are scoring twins (they only link to "
         f"another run's event log), so they were dropped. That leaves {len(inc)} generation runs over "
         f"{inc.task.nunique()} tasks. Every task was run at least twice and every run executed code, so all "
         f"{len(inc)} are used. Runs per task: "
         + ", ".join(f"{n} runs for {k} tasks" for n, k in sorted(rpt.items())) + ".",
         "2. What each run gave the extractor. Each old run is a 10-iteration search over many hypotheses "
         "(about 150 experiments). The old synthesis step ranked the hypotheses (supported first, then by "
         "confidence) and built the final answer from the top 4. We used the same ranking. For each of the top "
         "4 hypotheses we took the last experiment that ran without error, in run order. The final answer is the "
         "old p15 synthesis text. The full code was recovered from the Engineer's reply, because the event log "
         "keeps only the first 500 characters.",
         "3. The post-hoc extractor (`reveries.decisions.extract_executed`, gemma-4-31b-it, temperature 0, "
         "thinking off, one draw) read the code and the final answer and returned one value per decision type. "
         "Values were put in canonical form (`reveries.decisions.canonical`).",
         f"4. Extracted runs: {s['n_extracted']} of {s['n_included_runs']}. Failures: "
         f"{'; '.join(s['extraction_failures']) or 'none'}.", "",
         "Terms:", "",
         "- *Agreement* at a decision type, for one task: the share of run pairs that chose the same canonical "
         "value. A run with no value for the type is left out. `quick_runs/metrics.py` computes it.",
         "- *Pooled agreement*: the mean of the per-task agreements, each task counted once. The pair-weighted "
         "version counts every run pair once, so tasks with many runs weigh more.",
         "- *First divergence*: the first decision type, in the fixed type order, where two runs of a task "
         "differ. The old runs have no DECISION lines, so the fixed order stands in for the run's own order.", "",
         "## Numbers", "",
         f"All runs ({a['n_runs']} runs, {a['n_tasks']} tasks). The v8 columns use only the three v8 arms "
         f"({v8['n_runs']} runs, {v8['n_tasks']} tasks, same code version, 3 runs per task).", "",
         "| decision type | pooled agreement | pair-weighted | tasks where all runs agree | tasks where no two runs agree | v8 only: pooled | v8 only: all agree |",
         "|---|---|---|---|---|---|---|"]
    for p in a["pooled"]:
        q = v8p[p["type"]]
        extra = (" (extra view)" if p["type"] == "model.all_families" else " (column view)") \
            if p["type"] in s["extra_types"] else ""
        L.append(f"| {p['type']}{extra} | {f2(p['mean_agreement_over_tasks'])} | {f2(p['pair_weighted_agreement'])} | "
                 f"{p['tasks_all_runs_same']} / {p['tasks_with_2plus_values']} | {p['tasks_all_runs_differ']} / "
                 f"{p['tasks_with_2plus_values']} | {f2(q['mean_agreement_over_tasks'])} | "
                 f"{q['tasks_all_runs_same']} / {q['tasks_with_2plus_values']} |")
    main_types = {p["type"]: p for p in a["pooled"]}

    def names(cond):
        xs = [f"`{p['type']}` {f2(p['mean_agreement_over_tasks'])}" for p in a["pooled"]
              if p["type"] in s["decision_types"] and p["mean_agreement_over_tasks"] is not None
              and cond(p["mean_agreement_over_tasks"])]
        return ", ".join(xs) or "none"
    L += ["", "The rows marked *column view* are not decision types. They compare the original dataset columns "
          "behind the target, the main variables and the controls, so a renamed derived column does not count "
          "as a split. `model.all_families` is every model family the run fitted, not only the final one.", "",
          "**The 3 types where runs split most** (lowest pooled agreement, all runs): "
          + "; ".join(f"`{t}` ({f2(main_types[t]['mean_agreement_over_tasks'])})" for t in a["split_most"])
          + f". On v8 only: " + "; ".join(f"`{t}` ({f2(v8p[t]['mean_agreement_over_tasks'])})" for t in v8["split_most"]) + ".",
          "", "**First divergence** (number of tasks, all runs): "
          + ", ".join(f"`{k}` {v}" for k, v in sorted(a["first_divergence_counts"].items(), key=lambda kv: -kv[1])) + ".",
          "", "## What it suggests", "",
          f"1. High agreement (pooled 0.80 or more): {names(lambda x: x >= 0.8)}.",
          f"2. Middle (0.30 to 0.80): {names(lambda x: 0.3 <= x < 0.8)}.",
          f"3. Low agreement (below 0.30): {names(lambda x: x < 0.3)}.",
          f"4. In the column view, main variables agree at "
          f"{f2(main_types['source.variables']['mean_agreement_over_tasks'])} and controls at "
          f"{f2(main_types['source.controls']['mean_agreement_over_tasks'])}. Compare with "
          f"`variables` {f2(main_types['variables']['mean_agreement_over_tasks'])} and `model.controls` "
          f"{f2(main_types['model.controls']['mean_agreement_over_tasks'])}: the gap is the part due to naming "
          "of derived columns.",
          "5. The low-agreement types are where repeated runs fork. They are the candidates for fork memory and "
          "for the checks in P5. `transformation` is a list of many small steps, so its agreement is low almost "
          "by construction; read it with that in mind.",
          "", "## Caveats", "",
          "- **These runs differ by arm, not only by seed.** The arms change memory, retrieval settings, the "
          "corrector and design discipline, and the v8 and v10 runs come from different code versions. Part of "
          "the split may come from the arms. The v8-only columns hold the code version fixed but still mix three "
          "memory arms.",
          "- Each old run explores many hypotheses. The top-ranked hypotheses of two runs can be different "
          "questions, so a split here can mean \"answered a different sub-question\", not only \"chose a different "
          "analysis for the same question\".",
          "- Free-text types (`framing.population`, `framing.unit`, `final_selection`) are compared as lower-cased "
          "strings, so wording differences count as splits. Treat their agreement as a lower bound.",
          "- The extractor ran once per run at temperature 0. It was not hand-checked against the code for these runs.",
          ""]
    m = s["memory_use"]
    c, d = m.get("code_items", {}), m.get("digest_items", {})
    cj, dj = c.get("judgments", {}), d.get("judgments_same_variables", {})
    L += ["## What the old memory showed the agent (descriptive)", "",
          f"{m['memory_arm_runs']} of the included runs had memory on. We looked at the prompt in which the "
          "Engineer (the agent that writes the experiment code) designs each experiment. Two kinds of memory item "
          "appear there, and both are shown verbatim in that prompt:", "",
          f"- *Code memories* ({m['runs_with_code_memories']} runs, the v8 memory arms and one e01 arm): past "
          "experiments' code under \"Relevant memories\".",
          f"- *Design-digest items* ({m['runs_with_design_digest']} runs, the v10 and e01 arms): one line per "
          "past design with its outcome tally, plus a `try_next` suggestion from the analyst's reflection.", "",
          "How we judged \"followed\" (a fixed rule, no model and no hand reading):", "",
          "- Code memory: we reduce the shown code and the code the Engineer then wrote to a signature (model "
          "families called, plus the names in model formulas and feature lists). Same signature = *followed*. "
          "Both readable but different = *ignored*. Either unreadable (for example a t-test with no formula) = "
          "*cannot tell*.",
          "- Design-digest item: we take the dataset columns that the `try_next` text names and the item's own "
          "design does not already use. If the design in the same prompt names at least one of them = "
          "*followed*. If it names none = *ignored*. If `try_next` names no new column = *cannot tell*. Only "
          "items about the same variables as the hypothesis being designed are judged.", "",
          "Each item is judged every time it is shown, so one item shown in ten prompts counts ten times.", "",
          "Examples (verbatim, shortened to one line):", ""]
    for e in m["examples"]:
        L.append(f"- Task {e['task']}, {e['arm']}, {e['kind']}. Shown: `{e['shown']}`. Next design: "
                 f"`{e['next_design']}`. Judged: **{e['judgment'].replace('_', ' ')}**.")
    tot_c = cj.get("followed", 0) + cj.get("ignored", 0)
    tot_d = dj.get("followed", 0) + dj.get("ignored", 0)
    L += ["", "Counts:", "",
          f"- Code memories: {c.get('shown', 0)} shown. Followed {cj.get('followed', 0)}, ignored "
          f"{cj.get('ignored', 0)}, cannot tell {cj.get('cannot_tell', 0)}. Where it can be told, followed "
          f"{cj.get('followed', 0)}/{tot_c} ({100 * cj.get('followed', 0) / max(tot_c, 1):.0f}%).",
          f"- But {100 * c.get('same_hypothesis_share', 0):.0f}% of the code memories shown were earlier "
          "experiments for the *same* hypothesis the Engineer was designing (retrieval was scoped to it). So "
          "\"followed\" here mostly means the Engineer re-ran its own earlier code. For comparison, the new code "
          "matched an earlier experiment that was *not* shown "
          f"{100 * (c.get('base_rate_match_unshown_earlier') or 0):.1f}% of the time.",
          f"- Design-digest items about the same variables: {d.get('shown_same_variables', 0)} shown. Followed "
          f"{dj.get('followed', 0)}, ignored {dj.get('ignored', 0)}, cannot tell {dj.get('cannot_tell', 0)}. "
          f"Where it can be told, followed {dj.get('followed', 0)}/{tot_d} "
          f"({100 * dj.get('followed', 0) / max(tot_d, 1):.0f}%).",
          f"- Per prompt (one prompt shows several items): of {d.get('prompts_judged', 0)} prompts with at least "
          f"one item that can be told, {d.get('prompts_followed_at_least_one', 0)} followed at least one "
          f"suggestion and {d.get('prompts_ignored_all', 0)} followed none.", "",
          "Limits of this judgment: a signature match can happen without the memory (the hypothesis fixes the "
          "variables). A design can follow advice without naming a column (for example \"try a nonparametric "
          "test\"), which the rule calls *cannot tell*. The Researcher and Analyst also got memory sections "
          "(facts, past results); those are not judged here.", ""]
    (OUT / "report.md").write_text("\n".join(L))
    print("wrote", OUT / "report.md")


if __name__ == "__main__":
    main()
