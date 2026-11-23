"""P12 step 1: the agent writes lessons from its own earlier runs (P1 and P1b), with the evaluator's outcome.

ReasoningBank-style: one call per run, gemma-4-31B, thinking off, temperature 0, 1 to 3 lessons per run.
Outcome shown to the writer: Silent Failures, each check passed/failed with its question; BLADE, the expert-match
result; DSBench analysis, right or wrong (never the gold answer); DSBench modelling, the relative gap.
Lessons are frozen once written: quick_runs/results/P12/lessons.jsonl.

  uv run --no-active python quick_runs/p12/lessons.py
"""
from __future__ import annotations

import os
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "quick_runs"))
from reveries.llm import LLM  # noqa: E402
from score import SF_CHECKS, load_events, transcript  # noqa: E402

RUNS = Path(os.environ.get("REVERIES_RUNS", "/data/anon/reveries_runs"))
OUT = REPO / "quick_runs" / "results" / "P12" / "lessons.jsonl"
SERVERS = ["http://localhost:8100"]  # one 4-GPU server (a 100k prompt needs the larger KV pool)
SPECS = {s["id"]: s for f in ("tasks.yaml", "tasks_p1b.yaml")
         for s in yaml.safe_load(open(REPO / "quick_runs" / f))["tasks"]}

WRITER = """You are reviewing one run of a data science agent, to write lessons for future runs.

## Task given to the agent
{question}

## What the agent did (code cells, outputs, final answer)
{transcript}

## Evaluation of this run
{outcome}

Write 1 to 3 lessons that would help an agent doing this task, or a similar one, next time. Learn from what
went wrong as well as what went right. Each lesson must be general enough to act on (what to check or do, and
when). Do not state the final answer or numeric result of this task.

Reply with only a JSON object: {{"lessons": [{{"title": "...", "when": "when this applies", "content": "what to do"}}]}}"""


def outcome_text(spec: dict, rec: dict, criteria: dict | None) -> str:
    src = spec["source"]
    if src == "silent_failures":
        q = {c["key"]: c["question"] for c in criteria["criteria"]}
        lines = []
        for c, ok in rec["checks"].items():
            verdict = {True: "PASSED", False: "FAILED", None: "could not be judged"}[ok]
            lines.append(f"- {c}: {verdict}. (Check: {q.get(c, '')[:400]})")
        return "Checks on how the analysis was done:\n" + "\n".join(lines) + f"\nTest-set score: {rec['score']}"
    if src == "blade":
        c = rec["checks"]
        return ("Comparison with analyses by expert analysts of the same data:\n"
                f"- outcome variable matches an expert choice: {c.get('dv_matches_expert')}\n"
                f"- main predictor matches an expert choice: {c.get('iv_matches_expert')}\n"
                f"- model family matches an expert choice: {c.get('family_matches_expert')}")
    if src == "dsbench_analysis":
        ok = rec["checks"].get("correct")
        return "The final answer was " + {True: "CORRECT.", False: "INCORRECT.", None: "missing (no answer given)."}[ok]
    if src == "dsbench_modeling":
        return (f"Submission valid: {rec['checks'].get('valid_submission')}. Relative score: {rec['score']} "
                "(0 = a simple baseline, 1 = the best possible score).")
    raise ValueError(src)


def write_one(job):
    i, probe, rec, task_question, criteria = job
    spec = SPECS[rec["task_id"]]
    events = load_events(RUNS / probe / rec["task_id"] / rec["run"])
    prompt = WRITER.format(question=task_question, transcript=transcript(events, max_chars=24000),
                           outcome=outcome_text(spec, rec, criteria))
    llm = LLM(base_url=SERVERS[i % len(SERVERS)], thinking=False, sampling={"temperature": 0.0})
    r = llm.chat([{"role": "user", "content": prompt}], max_tokens=1500)
    m = re.search(r"\{.*\}", r.content, re.S)
    try:
        lessons = json.loads(m.group(0))["lessons"] if m else []
    except (json.JSONDecodeError, KeyError, TypeError):
        lessons = []
    return [{"source_task": rec["task_id"], "source_probe": probe, "source_run": rec["run"], "k": k,
             "title": str(L.get("title", "")), "when": str(L.get("when", "")), "content": str(L.get("content", ""))}
            for k, L in enumerate(lessons[:3]) if isinstance(L, dict)] or [
        {"source_task": rec["task_id"], "source_probe": probe, "source_run": rec["run"], "k": -1,
         "title": "", "when": "", "content": "", "parse_error": r.content[:500]}]


def main():
    from reveries.tasks import load_tasks
    tasks = {t.task_id: t for f in ("tasks.yaml", "tasks_p1b.yaml") for t in load_tasks(REPO / "quick_runs" / f)[0]}
    jobs = []
    for probe in ("P1", "P1b"):
        for line in open(RUNS / probe / "records.jsonl"):
            rec = json.loads(line)
            t = tasks[rec["task_id"]]
            jobs.append((len(jobs), probe, rec, t.question, t.gold.get("criteria")))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(16) as pool:
        out = [L for ls in pool.map(write_one, jobs) for L in ls]
    with open(OUT, "w") as f:
        for L in out:
            f.write(json.dumps(L) + "\n")
    print(len(jobs), "runs;", sum(L["k"] >= 0 for L in out), "lessons;", sum(L["k"] < 0 for L in out), "parse errors")


if __name__ == "__main__":
    main()
