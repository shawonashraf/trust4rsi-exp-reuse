"""P19 step 1: a write-time check without the evaluator. The P17 writer (no outcome) is given the P16 data profile of the
task's files in place of the evaluation section ("nfd" lessons). Same 40 Silent Failures source runs (P1), same model
and settings. Output (frozen once written): quick_runs/results/P19/lessons_nfd.jsonl. Design: quick_runs/predictions/P19.md.

  uv run --no-active python quick_runs/p19/write_nfd.py --servers http://localhost:8100
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for p in ("p12", "p13", ""):
    sys.path.insert(0, str(REPO / "quick_runs" / p))
sys.path.insert(0, str(REPO))
import lessons as P12  # noqa: E402
import guide  # noqa: E402

OUT = REPO / "quick_runs" / "results" / "P19" / "lessons_nfd.jsonl"
EVAL_SECTION = "## Evaluation of this run\n{outcome}\n\n"
assert EVAL_SECTION in P12.WRITER
PROBE_SECTION = ("## Data profile (computed from the task's files)\n{outcome}\n\n"
                 "Use the profile to check what the agent assumed about the data.\n\n")
WRITER_NFD = P12.WRITER.replace(EVAL_SECTION, PROBE_SECTION)
# the 8 Silent Failures source tasks: the two cue variants share the leak-free tasks' files and labels
LABEL = {**guide.LABEL, "sf_tf04_balance_torque_cue": "Class Name", "sf_tf06_metric_select_cue": "goals_label"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--servers", nargs="+", default=["http://localhost:8100"])
    ap.add_argument("--train-only", action="store_true", help="P21: the profile without any test-set number")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    P12.SERVERS = args.servers
    from reveries.tasks import load_tasks
    tasks = {t.task_id: t for f in ("tasks.yaml", "tasks_p1b.yaml") for t in load_tasks(REPO / "quick_runs" / f)[0]}
    profiles = {tid: guide.data_profile(tasks[tid].data_files, LABEL[tid], train_only=args.train_only) for tid in LABEL}
    P12.WRITER = WRITER_NFD
    P12.outcome_text = lambda spec, rec, criteria: profiles[rec["task_id"]]  # the profile replaces the outcome
    jobs = []
    for line in open(P12.RUNS / "P1" / "records.jsonl"):
        rec = json.loads(line)
        if rec["task_id"].startswith("sf_"):
            t = tasks[rec["task_id"]]
            jobs.append((len(jobs), "P1", rec, t.question, t.gold.get("criteria")))
    assert len(jobs) == 40, len(jobs)
    with ThreadPoolExecutor(16) as pool:
        out = [L for ls in pool.map(P12.write_one, jobs) for L in ls]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for L in out:
            f.write(json.dumps({**L, "nf": True, "probe": "train" if args.train_only else True}) + "\n")
    print(len(jobs), "runs;", sum(L["k"] >= 0 for L in out), "lessons;", sum(L["k"] < 0 for L in out), "parse errors")


if __name__ == "__main__":
    main()
