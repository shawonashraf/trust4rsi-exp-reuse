"""P0 steps 3-5: decision agreement per task and type, first divergence, memory use; writes summary.json.

Usage: uv run --no-active python quick_runs/p0/agree.py
Reads quick_runs/results/P0/run_folders.csv and the caches in /data/anon/reveries_runs/P0/.
"""
from __future__ import annotations

import os
import gzip
import itertools
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "quick_runs"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import decision_agreement, first_divergence  # noqa: E402
from reveries.decisions import DECISION_TYPES, canonical  # noqa: E402

CACHE = Path(os.environ.get("REVERIES_RUNS", "/data/anon/reveries_runs")) / "P0"
OUT = REPO / "quick_runs/results/P0"
# Column-level views, from the extractor's "source_columns" (original dataset columns behind each role):
# less sensitive to how a derived column was named.
EXTRA_TYPES = ["source.target", "source.variables", "source.controls", "model.all_families"]
SUBSETS = {"all": lambda arm: True, "v8_only": lambda arm: arm.startswith("v8_transfer:"),
           "v10_only": lambda arm: arm.startswith("v10_")}


def canon(dtype: str, value) -> str | None:
    """Canonical value; None (extractor said null / no value) is left out of agreement."""
    if value is None:
        return None
    c = canonical(dtype, value)
    return "(none)" if c == "" else c


def run_record(executed: dict) -> dict:
    dec = {t: canon(t, executed.get(t)) for t in DECISION_TYPES + ["model.all_families"]}
    src = executed.get("source_columns") or {}
    if isinstance(src, dict):
        for role in ("target", "variables", "controls"):
            dec[f"source.{role}"] = canon(f"source.{role}", src.get(role))
    return {"decisions": dec, "order": []}


def task_table(runs_by_task: dict[int, list[dict]], types: list[str]) -> pd.DataFrame:
    """One row per task: n_runs, agreement per type, first divergence over DECISION_TYPES."""
    rows = []
    for task, runs in sorted(runs_by_task.items()):
        row = {"task": task, "n_runs": len(runs)}
        for t in types:
            row[t] = decision_agreement(runs, t)
        row["first_divergence"] = first_divergence(runs, DECISION_TYPES)
        rows.append(row)
    return pd.DataFrame(rows)


def pair_weighted(runs_by_task: dict[int, list[dict]], dtype: str) -> float | None:
    same = total = 0
    for runs in runs_by_task.values():
        vals = [r["decisions"].get(dtype) for r in runs if r["decisions"].get(dtype) is not None]
        for a, b in itertools.combinations(vals, 2):
            same += a == b
            total += 1
    return same / total if total else None


def pooled(table: pd.DataFrame, runs_by_task, types: list[str]) -> list[dict]:
    out = []
    for t in types:
        col = table[t].dropna()
        out.append({"type": t, "tasks_with_2plus_values": int(col.size),
                    "mean_agreement_over_tasks": float(col.mean()) if col.size else None,
                    "pair_weighted_agreement": pair_weighted(runs_by_task, t),
                    "tasks_all_runs_same": int((col == 1.0).sum()),
                    "tasks_all_runs_differ": int((col == 0.0).sum())})
    return out


def load_records(runs: pd.DataFrame) -> tuple[dict, list[str]]:
    recs, failed = {}, []
    for r in runs.itertuples():
        p = CACHE / "extract" / f"{r.key}.json"
        if not p.is_file():
            failed.append(f"{r.key}: not extracted")
            continue
        ex = json.loads(p.read_text())["executed"]
        if not isinstance(ex, dict):
            failed.append(f"{r.key}: extractor reply not parseable")
            continue
        recs[r.key] = {**run_record(ex), "task": r.task, "arm": r.arm, "key": r.key, "executed": ex}
    return recs, failed


def analyse(recs: dict, subset: str) -> dict:
    by_task: dict[int, list[dict]] = {}
    for rec in recs.values():
        if SUBSETS[subset](rec["arm"]):
            by_task.setdefault(rec["task"], []).append(rec)
    by_task = {t: rs for t, rs in by_task.items() if len(rs) >= 2}
    types = DECISION_TYPES + EXTRA_TYPES
    table = task_table(by_task, types)
    pool = pooled(table, by_task, types)
    main = [p for p in pool if p["type"] in DECISION_TYPES and p["mean_agreement_over_tasks"] is not None]
    split_most = sorted(main, key=lambda p: (p["mean_agreement_over_tasks"], p["type"]))[:3]
    return {"subset": subset, "n_tasks": len(by_task), "n_runs": sum(len(v) for v in by_task.values()),
            "runs_per_task": {int(k): len(v) for k, v in sorted(by_task.items())},
            "pooled": pool, "split_most": [p["type"] for p in split_most],
            "first_divergence_counts": dict(Counter(table["first_divergence"].fillna("(never)"))),
            "per_task": json.loads(table.to_json(orient="records"))}


def memory_summary() -> dict:
    from memory_use import analyse as mem_analyse
    from runs import load_run
    runs = pd.read_csv(OUT / "run_folders.csv")
    runs = runs[runs.included & (runs.memory_enabled == True)]  # noqa: E712
    facts, queries = [], {}
    for r in runs.itertuples():
        with gzip.open(CACHE / "facts" / f"{r.key}.json.gz", "rt") as f:
            fa = json.load(f)
        facts.append(fa)
        if any(e.get("designs_already_run") for e in fa["engineer"]):
            queries[r.key] = load_run(fa["path"])["engineer_calls"]
    res = mem_analyse(facts, queries)
    code, dig = pd.DataFrame(res["code_rows"]), pd.DataFrame(res["digest_rows"])
    out = {"memory_arm_runs": len(facts),
           "runs_with_code_memories": int(code.key.nunique()) if len(code) else 0,
           "runs_with_design_digest": int(dig.key.nunique()) if len(dig) else 0}
    if len(code):
        out["code_items"] = {
            "shown": len(code),
            "judgments": dict(Counter(code.judgment)),
            "same_hypothesis_share": float(code.same_hypothesis.mean()),
            "judgments_same_hypothesis": dict(Counter(code[code.same_hypothesis].judgment)),
            "judgments_other_hypothesis": dict(Counter(code[~code.same_hypothesis].judgment)),
            "base_rate_match_unshown_earlier": float(code.base_rate.dropna().mean()) if code.base_rate.notna().any() else None,
        }
    if len(dig):
        same = dig[dig.same_variables]
        prompts = same.groupby(["key", "order"]).judgment.agg(lambda s: "followed" if (s == "followed").any()
                                                             else ("ignored" if (s == "ignored").any() else "cannot_tell"))
        out["digest_items"] = {"shown": len(dig), "shown_same_variables": len(same),
                               "judgments_same_variables": dict(Counter(same.judgment)),
                               "prompts_judged": int((prompts != "cannot_tell").sum()),
                               "prompts_followed_at_least_one": int((prompts == "followed").sum()),
                               "prompts_ignored_all": int((prompts == "ignored").sum()),
                               "prompts_cannot_tell": int((prompts == "cannot_tell").sum())}
    ex = []
    if len(code):
        for j in ("followed", "ignored", "cannot_tell"):
            sub = code[(code.judgment == j) & (code.item_model_line != "")]
            if len(sub):
                x = sub.iloc[0]
                ex.append({"kind": "code memory", "task": int(x.task), "arm": x.arm, "judgment": j,
                           "same_hypothesis": bool(x.same_hypothesis),
                           "shown": x.item_model_line, "next_design": x.new_model_line})
    if len(dig):
        for j in ("followed", "ignored"):
            sub = dig[(dig.judgment == j) & dig.same_variables]
            if len(sub):
                x = sub.iloc[len(sub) // 2]
                ex.append({"kind": "design digest", "task": int(x.task), "arm": x.arm, "judgment": j,
                           "shown": f"{x.item_design} [{x.tally}] try_next: {x.try_next}",
                           "new_columns_suggested": list(x.new_columns), "next_design": x.design_line})
    out["examples"] = ex
    return out


def main():
    runs = pd.read_csv(OUT / "run_folders.csv")
    runs = runs[runs.included]
    recs, failed = load_records(runs)
    summary = {"n_included_runs": int(len(runs)), "n_extracted": len(recs), "extraction_failures": failed,
               "decision_types": DECISION_TYPES, "extra_types": EXTRA_TYPES,
               "results": {s: analyse(recs, s) for s in SUBSETS}}
    summary["memory_use"] = memory_summary()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    a = summary["results"]["all"]
    print(f"{len(recs)} runs extracted, {a['n_tasks']} tasks; split most: {a['split_most']}")
    for p in a["pooled"]:
        print(f"  {p['type']:22s} mean={p['mean_agreement_over_tasks']} same={p['tasks_all_runs_same']}/{p['tasks_with_2plus_values']}")


if __name__ == "__main__":
    main()
