"""P17 step 1: the P12 writer, without the evaluator's outcome ("nf" lessons). Same 40 Silent Failures source runs (P1),
same model and settings. Output (frozen once written): quick_runs/results/P17/lessons_nf.jsonl. Design: quick_runs/predictions/P17.md.

  uv run --no-active python quick_runs/p17/write_nf.py --servers http://localhost:8100
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "quick_runs" / "p12"))
sys.path.insert(0, str(REPO / "quick_runs"))
sys.path.insert(0, str(REPO))
import lessons as P12  # noqa: E402

OUT = REPO / "quick_runs" / "results" / "P17" / "lessons_nf.jsonl"
EVAL_SECTION = "## Evaluation of this run\n{outcome}\n\n"
assert EVAL_SECTION in P12.WRITER
WRITER_NF = P12.WRITER.replace(EVAL_SECTION, "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--servers", nargs="+", default=["http://localhost:8100"])
    args = ap.parse_args()
    P12.SERVERS = args.servers
    P12.WRITER = WRITER_NF  # write_one formats WRITER with outcome=...; the placeholder is gone, so it is unused
    from reveries.tasks import load_tasks
    tasks = {t.task_id: t for f in ("tasks.yaml", "tasks_p1b.yaml") for t in load_tasks(REPO / "quick_runs" / f)[0]}
    jobs = []
    for line in open(P12.RUNS / "P1" / "records.jsonl"):
        rec = json.loads(line)
        if rec["task_id"].startswith("sf_"):
            t = tasks[rec["task_id"]]
            jobs.append((len(jobs), "P1", rec, t.question, t.gold.get("criteria")))
    assert len(jobs) == 40, len(jobs)
    with ThreadPoolExecutor(16) as pool:
        out = [L for ls in pool.map(P12.write_one, jobs) for L in ls]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for L in out:
            f.write(json.dumps({**L, "nf": True}) + "\n")
    print(len(jobs), "runs;", sum(L["k"] >= 0 for L in out), "lessons;", sum(L["k"] < 0 for L in out), "parse errors")


if __name__ == "__main__":
    main()
