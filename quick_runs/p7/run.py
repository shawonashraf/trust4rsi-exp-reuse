"""P7: loop-guard replay over the old experience-memory runs (read-only over /data/anon).

Usage: uv run --no-active python quick_runs/p7/run.py
Writes quick_runs/results/P7/summary.json and report.md.
"""
from __future__ import annotations

import os
import json
import sys
from collections import Counter
from pathlib import Path

LEGACY = Path(os.environ.get("REVERIES_LEGACY", "/data/anon"))  # pre-paper runs, read-only

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ledger import (DEAD, coarse_key, missed_near_duplicates, old_dead_end_retries,  # noqa: E402
                    outcome_class, replay)

REPO = Path(__file__).resolve().parents[2]
ROOT = LEGACY / "debug/e01_experience_memory/results"
RUN = "lambda01_gemma4_mem_all_graphon_corr_{arm}_dbg"
TASKS, ARMS = (5, 16), ("e0", "e1")
OUT = REPO / "quick_runs/results/P7"


def run_dir(task, arm):
    return ROOT / f"discoverybench_train_{task}" / RUN.format(arm=arm)


def load(task, arm):
    rd = run_dir(task, arm)
    recs = [json.loads(l) for l in (rd / "data/trajectories.jsonl").read_text().splitlines() if l.strip()]
    order = []
    with open(rd / "provenance/provenance.jsonl") as f:
        for line in f:
            if '"trajectory_recorded"' in line:
                order.append(json.loads(line)["metadata"]["record_id"])
    assert order == [r["record_id"] for r in recs], "trajectory file order != event order"
    return recs


def design_str(r):
    d = r.get("design_used") or {}
    return (f"{d.get('model_family') or '-'}; controls={d.get('controls') or []}; moderators={d.get('moderators') or []}; "
            f"transforms={d.get('transformations') or []}; scope={d.get('scope') or '-'!r}")


def per_run(task, arm):
    recs = load(task, arm)
    rep = replay(recs)
    rep_c = replay(recs, coarse_key)
    missed = missed_near_duplicates(recs)
    retries = [x for x in rep if x["retry"]]
    blocked = [x for x in rep if x["blocked"]]
    old = [x for x in rep if x["old_dead_end_retry"]]
    row = {
        "task": task, "arm": arm, "attempts": len(recs),
        "outcome_classes": dict(Counter(x["cls"] for x in rep)),
        "attempts_without_fingerprint": sum(x["key"] is None for x in rep),
        "null_flags": sum(bool(r["corrector"].get("null_result")) for r in recs),
        "null_flags_also_rejected": sum(bool(r["corrector"].get("null_result")) and r["outcome"] == "rejected" for r in recs),
        "retries": len(retries),
        "old_dead_end_retries": len(old),
        "old_dead_end_retries_check_old_notebook_count": old_dead_end_retries(recs),
        "old_dead_end_retries_blocked": sum(x["blocked"] for x in old),
        "old_dead_end_retries_unblocked_no_key": sum((not x["blocked"]) and x["key"] is None for x in old),
        "blocked": len(blocked),
        "blocked_actual_outcome": dict(Counter(x["cls"] for x in blocked)),
        "blocked_that_were_dead_again": sum(x["cls"] in DEAD for x in blocked),
        "blocked_that_were_supported": sum(x["cls"] == "supported" for x in blocked),
        "coarse_key": {"retries": sum(x["retry"] for x in rep_c), "blocked": sum(x["blocked"] for x in rep_c),
                       "blocked_that_were_dead_again": sum(x["blocked"] and x["cls"] in DEAD for x in rep_c),
                       "blocked_that_were_supported": sum(x["blocked"] and x["cls"] == "supported" for x in rep_c)},
        "missed_near_duplicates": dict(Counter(m["kind"] for m in missed)),
        "missed_near_duplicates_dead_again": dict(Counter(m["kind"] for m in missed if rep[m["j"]]["cls"] in DEAD)),
    }
    examples = []
    for kind in ("same_design_other_wording", "no_design_recorded", "one_element_off"):
        for m in [m for m in missed if m["kind"] == kind][:2]:
            a, b = recs[m["i"]], recs[m["j"]]
            examples.append({"task": task, "arm": arm, "kind": kind,
                             "earlier": {"attempt": m["i"], "iteration": a["iteration"], "outcome": outcome_class(a),
                                         "fingerprint": a["design_fingerprint"], "variables": a["pair_columns"],
                                         "design": design_str(a)},
                             "later": {"attempt": m["j"], "iteration": b["iteration"], "outcome": outcome_class(b),
                                       "fingerprint": b["design_fingerprint"], "variables": b["pair_columns"],
                                       "design": design_str(b)}})
    return row, examples


def pct(a, b):
    return f"{a}/{b} ({100 * a / b:.0f}%)" if b else f"{a}/0"


def report(rows, examples):
    L = ["# P7. Loop-guard replay", "",
         "Executed notebook: [docs/notebooks/quick_P7.ipynb](../../../docs/notebooks/quick_P7.ipynb).", "",
         "**Question.** Would a dead-end ledger have stopped the repeated dead ends in the old experience-memory runs?",
         "", "## What was run", "",
         "- Input: the four old runs under `/data/anon/debug/e01_experience_memory/results/` (tasks 5 and 16; "
         "arm E0 = experience memory off, arm E1 = on). One run per task and arm. No model calls.",
         "- An *attempt* is one record in the run's `data/trajectories.jsonl`: one hypothesis tested by one executed "
         "experiment and then analysed. The run logs one `trajectory_recorded` event per record, in the same order "
         "(checked), so file order is time order. Code crashes that the agent fixed inside the same attempt "
         "(`experiment_failed` events) are not separate attempts.",
         "- *Outcome class* of an attempt, first match wins: *error* (experiment did not finish), *rejected* "
         "(the analyst's outcome), *null* (the corrector flagged a null result), *inconclusive*, *supported*. "
         "A *dead* attempt is error, rejected or null.",
         "- The *ledger key* is the variables tested plus the design fingerprint the old run stored "
         "(old `src/memory/design_fingerprint.py`: model family, controls, moderators, transformations, scope). "
         "These tasks have no validation step, so validation is not part of the key. An attempt with no recorded "
         "design (fingerprint `none`) has no key and is never blocked.",
         "- A *retry* is an attempt whose key appeared earlier in the run.",
         "- The ledger *blocks* an attempt when its key already had a dead attempt earlier. A blocked attempt would "
         "not have run, so it does not change the ledger.",
         "- The old notebook's *dead-end retry* counts an attempt whose latest earlier attempt in the same group "
         "was rejected or null. Recomputed here as a check.",
         "- A second, coarser key (variables, model family with aliases merged, controls; scope, moderators and "
         "transformations ignored) is shown for comparison.", "",
         "## Numbers", "",
         "| task | arm | attempts | retries | old dead-end retries | of those blocked | blocked (all) | blocked that ended dead again | blocked that ended supported | no fingerprint |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['task']} | {r['arm']} | {r['attempts']} | {r['retries']} | {r['old_dead_end_retries']} | "
                 f"{r['old_dead_end_retries_blocked']} | {pct(r['blocked'], r['retries'])} of retries | "
                 f"{r['blocked_that_were_dead_again']} | {r['blocked_that_were_supported']} | "
                 f"{r['attempts_without_fingerprint']} |")
    L += ["", "Old-notebook check: the recomputed old dead-end retry counts are "
          + ", ".join(f"task {r['task']} {r['arm']} = {r['old_dead_end_retries_check_old_notebook_count']}" for r in rows)
          + ".", "",
          "Coarser key (variables, family, controls):", "",
          "| task | arm | retries | blocked | blocked that ended dead again | blocked that ended supported |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        c = r["coarse_key"]
        L.append(f"| {r['task']} | {r['arm']} | {c['retries']} | {c['blocked']} | {c['blocked_that_were_dead_again']} | "
                 f"{c['blocked_that_were_supported']} |")
    L += ["", "## Retries the ledger would miss", "",
          "A *missed near-duplicate* is an attempt the fingerprint ledger lets through although an earlier dead "
          "attempt tested the same variables with almost the same design. Three kinds, closest first:", "",
          "- *same design, other wording*: same family, controls, moderators and transformations after lower-casing "
          "and removing spaces; only the scope text or spelling changed the fingerprint.",
          "- *no design recorded*: the attempt has fingerprint `none`, so no key at all.",
          "- *one element off*: same family; controls, moderators and transformations differ by one item. This is "
          "a loose match: adding one control can be a real change.", "",
          "| task | arm | same design, other wording | no design recorded | one element off | of these, ended dead again |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        m, md = r["missed_near_duplicates"], r["missed_near_duplicates_dead_again"]
        L.append(f"| {r['task']} | {r['arm']} | {m.get('same_design_other_wording', 0)} | {m.get('no_design_recorded', 0)} | "
                 f"{m.get('one_element_off', 0)} | {sum(md.values())} |")
    L += ["", "Examples (earlier dead attempt, then the later attempt the ledger lets through):", ""]
    for e in examples:
        L.append(f"- Task {e['task']} {e['arm']}, {e['kind'].replace('_', ' ')}: attempt {e['earlier']['attempt']} "
                 f"({e['earlier']['outcome']}, fp {e['earlier']['fingerprint']}) `{e['earlier']['design']}` -> "
                 f"attempt {e['later']['attempt']} ({e['later']['outcome']}, fp {e['later']['fingerprint']}) "
                 f"`{e['later']['design']}`")
    e1 = {r["task"]: r for r in rows if r["arm"] == "e1"}

    def caught(r):
        n, b = r["old_dead_end_retries"], r["old_dead_end_retries_blocked"]
        nk = r["old_dead_end_retries_unblocked_no_key"]
        miss = "" if n == b else f" ({nk} of the {n - b} it misses have no recorded design, so no key)"
        return f"task {r['task']} {b} of {n}{miss}"

    def miss_mix(r):
        m = r["missed_near_duplicates"]
        return (f"{m.get('same_design_other_wording', 0)} same design with other wording, "
                f"{m.get('no_design_recorded', 0)} with no recorded design, {m.get('one_element_off', 0)} one element off")
    L += ["", "## What it suggests", "",
          f"1. In E1 the ledger blocks the repeated dead ends the old notebook counted: {caught(e1[5])}; {caught(e1[16])}.",
          f"2. It blocks more attempts than those, because a key stays dead once it has failed: "
          f"{pct(e1[5]['blocked'], e1[5]['retries'])} of retries on task 5 E1 and "
          f"{pct(e1[16]['blocked'], e1[16]['retries'])} on task 16 E1.",
          f"3. The cost: many blocked attempts in fact ended supported "
          f"({e1[5]['blocked_that_were_supported']} of {e1[5]['blocked']} on task 5 E1, "
          f"{e1[16]['blocked_that_were_supported']} of {e1[16]['blocked']} on task 16 E1). "
          "The key holds the variables and the design, not the claim. The same design on the same variables can end "
          "rejected for one hypothesis and supported for the next. A hard block would remove those later successes.",
          f"4. The exact fingerprint misses near-duplicates of dead attempts. Task 5 E1: {miss_mix(e1[5])}. "
          f"Task 16 E1: {miss_mix(e1[16])}.",
          f"5. Outcome classes seen: {dict(sum((Counter(r['outcome_classes']) for r in rows), Counter()))}. "
          f"{sum(r['null_flags_also_rejected'] for r in rows)} of {sum(r['null_flags'] for r in rows)} corrector null "
          "flags sit on attempts that were also rejected, so they fall in the rejected class.",
          "", "## Caveats", "",
          "- Two tasks, one run per arm. These are counts, not estimates.",
          "- The replay is counterfactual only in a simple sense: we count what would be blocked. We cannot know "
          "what the agent would have done instead.",
          "- The fingerprints are the ones the old runs stored. They were not recomputed."]
    return "\n".join(L) + "\n"


def main():
    rows, examples = [], []
    for t in TASKS:
        for a in ARMS:
            r, ex = per_run(t, a)
            rows.append(r)
            examples += ex
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"input": str(ROOT), "runs": rows, "examples": examples}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    (OUT / "report.md").write_text(report(rows, [e for e in examples if e["arm"] == "e1"][:6]))
    for r in rows:
        print(r["task"], r["arm"], r["retries"], r["old_dead_end_retries"], r["old_dead_end_retries_blocked"],
              r["blocked"], r["blocked_actual_outcome"], r["missed_near_duplicates"])


if __name__ == "__main__":
    main()
