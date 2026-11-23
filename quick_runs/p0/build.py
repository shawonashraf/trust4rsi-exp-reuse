"""P0 step 1-2a: list run folders, choose the ones to use, cache per-run facts.

Writes quick_runs/results/P0/run_folders.csv and /data/anon/reveries_runs/P0/facts/<run_key>.json.gz.
Usage: uv run --no-active python quick_runs/p0/build.py
"""
from __future__ import annotations

import os
import csv
import gzip
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runs import arm_flags, columns_for, discover, final_answer, final_code_cells, load_run  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT_CSV = REPO / "quick_runs/results/P0/run_folders.csv"
CACHE = Path(os.environ.get("REVERIES_RUNS", "/data/anon/reveries_runs")) / "P0"


def run_key(r: dict) -> str:
    return f"t{r['task']:02d}__{r['source']}__{r['run_name']}"


def section(q: str, start: str, stop: str = "\n\n") -> str | None:
    i = q.find(start)
    if i < 0:
        return None
    j = q.find(stop, i + len(start) + 2)
    return q[i: j if j > 0 else len(q)]


def engineer_memory_view(calls: list[dict]) -> list[dict]:
    """Per Engineer design call: the memory sections it was shown and the code it wrote."""
    out = []
    for c in calls:
        q = c["query"]
        rel = None
        i = q.find("Relevant memories:")
        if i >= 0:
            # The section runs to the next prompt heading after the memory list.
            rest = q[i:]
            ends = [rest.find(m) for m in ("\nDesign a Python experiment", "\nDataset Description", "\nDataset:", "\nHypothesis:", "\nDesign an experiment",
                                           "\nWrite ", "\nRequirements", "\nInstructions") if rest.find(m) > 0]
            rel = rest[: min(ends)] if ends else rest[:20000]
        out.append({"order": c["order"], "relevant_memories": rel,
                    "designs_already_run": section(q, "Designs already run on this task", "\n\nDataset"),
                    "design_to_implement": section(q, "Design to implement", "\n\n"),
                    "is_design_call": "Design an experiment" in q, "code": c["code"]})
    return out


def build_one(r: dict) -> dict:
    facts = load_run(r["path"])
    cells = final_code_cells(facts)
    final, final_src = final_answer(r["path"])
    cols = columns_for(r["task"], r["path"], facts["dataset_path"])
    rec = {**r, "key": run_key(r), "goal": facts["goal"], "memory_enabled": facts["config"].get("memory_enabled"),
           "config": facts["config"], "columns": cols, "final_answer": final, "final_source": final_src,
           "code_cells": cells, "n_experiments": len(facts["experiments"]),
           "n_experiments_ok": sum(x["success"] for x in facts["experiments"]),
           "n_code_full": sum(x["code_full"] for x in facts["experiments"]),
           "n_hypotheses": len(facts["hypotheses"]),
           "experiments": [{k: x[k] for k in ("order", "experiment_id", "hypothesis_id", "success", "code", "design_used")}
                           for x in facts["experiments"]],
           "engineer": engineer_memory_view(facts["engineer_calls"]),
           "retrievals": facts["retrievals"]}
    (CACHE / "facts").mkdir(parents=True, exist_ok=True)
    with gzip.open(CACHE / "facts" / f"{rec['key']}.json.gz", "wt") as f:
        json.dump(rec, f)
    return {k: rec[k] for k in ("key", "task", "source", "run_name", "path", "arm", "goal", "memory_enabled",
                                "final_source", "n_experiments", "n_experiments_ok", "n_code_full")} | \
        {"n_code_cells": len(cells), "n_columns": len(cols)}


def main():
    found = discover()
    gen = [r for r in found if not r["twin"] and r["has_provenance"]]
    with ProcessPoolExecutor(8) as ex:
        built = list(ex.map(build_one, gen))
    by_key = {b["key"]: b for b in built}
    usable = [b for b in built if b["n_code_cells"] > 0]
    per_task = Counter(b["task"] for b in usable)
    goals = {}
    for b in usable:
        goals.setdefault(b["task"], set()).add(b["goal"])
    rows = []
    for r in found:
        k = run_key(r)
        b = by_key.get(k)
        if r["twin"]:
            why, inc = "excluded: scoring twin (symlinks the generation run's provenance; same run)", False
        elif not r["has_provenance"]:
            why, inc = "excluded: no provenance/provenance.jsonl", False
        elif b["n_code_cells"] == 0:
            why, inc = "excluded: no executed code", False
        elif per_task[r["task"]] < 2:
            why, inc = "excluded: task run only once", False
        elif len(goals[r["task"]]) > 1:
            why, inc = "excluded: task goal differs between runs", False
        else:
            why, inc = f"included: task run {per_task[r['task']]} times with executed code", True
        rows.append({"task": r["task"], "run_name": r["run_name"], "path": r["path"], "arm": r["arm"],
                     "flags": arm_flags(r["run_name"]) if not r["twin"] else "",
                     "memory_enabled": b["memory_enabled"] if b else "", "included": inc, "why": why,
                     "n_experiments": b["n_experiments"] if b else "", "n_code_cells_used": b["n_code_cells"] if b else "",
                     "final_answer_source": b["final_source"] if b else "", "key": k if b else ""})
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda x: (x["task"], x["arm"], x["run_name"])))
    print(f"found {len(found)} folders, {len(gen)} generation runs, included {sum(r['included'] for r in rows)}")


if __name__ == "__main__":
    main()
