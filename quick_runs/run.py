"""Run a probe: (task x seed) agent runs spread over the SGLang servers, then extract
the executed decisions of each run.

  uv run python quick_runs/run.py --probe P1 --seeds 0 1 2 3 4
  uv run python quick_runs/run.py --probe P9 --tasks blade_fish --seeds 0 0 0 --rep-tags a b c

Runs land in $REVERIES_RUNS/<probe>/<task_id>/<run_name>/ (default /data/anon/reveries_runs).
A run with run.json and executed.json is skipped (resume-safe).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd  # noqa: E402

from reveries.agent import Agent, Hint  # noqa: E402
from reveries.decisions import extract_executed  # noqa: E402
from reveries.llm import LLM, server_info  # noqa: E402
from reveries.tasks import load_tasks  # noqa: E402

RUNS = Path(os.environ.get("REVERIES_RUNS", "/data/anon/reveries_runs"))


def code_cells(run_dir: Path) -> tuple[list[str], list[dict], str | None]:
    cells, stated, final = [], [], None
    for line in open(run_dir / "events.jsonl"):
        e = json.loads(line)
        if e["event"] == "code_executed":
            cells.append(e["code"])
        elif e["event"] == "decision_stated":
            stated.append(e)
        elif e["event"] == "final_answer":
            final = e["text"]
    return cells, stated, final


def data_columns(f: Path) -> list[str]:
    if f.suffix.lower() == ".xlsx":
        return [f"sheet:{n}" for n in pd.ExcelFile(f).sheet_names]
    return list(pd.read_csv(f, nrows=1).columns)


# P-probe: an OpenAI-compatible agent backend (e.g. DeepSeek on SPIKE). Set by main() from CLI args.
AGENT_OPENAI = None  # dict(base_url, model, key_env, ledger) or None


def _agent_llm(base_url):
    if AGENT_OPENAI:
        import os
        from reveries.llm import OpenAILLM, ClaudeSpend
        return OpenAILLM(base_url=AGENT_OPENAI["base_url"], model=AGENT_OPENAI["model"],
                         api_key=os.environ[AGENT_OPENAI["key_env"]],
                         spend=ClaudeSpend(float("inf"), AGENT_OPENAI["ledger"]))
    return None


def one_run(task, seed, run_dir: Path, base_url: str, hint: Hint | None, meta: dict, sampling: dict | None = None) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    if not (run_dir / "run.json").exists():
        if (run_dir / "events.jsonl").exists():
            (run_dir / "events.jsonl").rename(run_dir / f"events.incomplete.{int(time.time())}.jsonl")
        llm = _agent_llm(base_url) or (LLM(base_url=base_url) if sampling is None else LLM(base_url=base_url, sampling=sampling))
        agent = Agent(llm, hint=hint)
        agent.run(task, run_dir, seed=seed, meta=meta)
    if not (run_dir / "executed.json").exists():
        cells, stated, final = code_cells(run_dir)
        cols = sorted({c for f in task.data_files for c in data_columns(f)})
        # Extractor: an evaluation tool, not the agent -> temperature 0, thinking off.
        ex_llm = _agent_llm(base_url) or LLM(base_url=base_url, sampling={"temperature": 0.0}, thinking=False)
        ex = extract_executed(ex_llm, cols, cells, stated, final)
        (run_dir / "executed.json").write_text(json.dumps(ex, indent=1))
    return json.loads((run_dir / "run.json").read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--tasks", nargs="*", default=None, help="task ids (default: all 15)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--rep-tags", nargs="*", default=None,
                    help="one tag per seed entry; lets the same seed run several times (P9)")
    ap.add_argument("--servers", nargs="+", default=["http://localhost:8100", "http://localhost:8101"])
    ap.add_argument("--agent-openai", action="store_true", help="probe: run the agent on an OpenAI-compatible endpoint")
    ap.add_argument("--agent-model", default=None)
    ap.add_argument("--agent-base-url", default=None)
    ap.add_argument("--agent-key-env", default="SPIKE_API_KEY")
    ap.add_argument("--agent-ledger", default=None)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--hint-file", default=None, help="YAML {task_id: hint text} (P2/P4)")
    ap.add_argument("--hint-where", default="start", choices=["start", "decision"])
    ap.add_argument("--tasks-file", default=None, help="task YAML (default quick_runs/tasks.yaml)")
    ap.add_argument("--temperature", type=float, default=None,
                    help="override the fixed sampling with greedy/other temperature (P9 uses 0)")
    ap.add_argument("--hint-type-file", default=None, help="YAML {task_id: decision type} for --hint-where decision")
    ap.add_argument("--hint-prefix", default="Note from previous experience: ",
                    help="text put before a start hint (P12 passes its own heading inside the hint text, so '')")
    args = ap.parse_args()
    if args.agent_openai:
        global AGENT_OPENAI
        AGENT_OPENAI = {"base_url": args.agent_base_url, "model": args.agent_model,
                        "key_env": args.agent_key_env, "ledger": args.agent_ledger}

    tasks, skipped = load_tasks(Path(args.tasks_file)) if args.tasks_file else load_tasks()
    if skipped:
        print("SKIPPED tasks:", skipped)
    if args.tasks:
        tasks = [t for t in tasks if t.task_id in args.tasks]
    hints = {}
    if args.hint_file:
        import yaml
        hints = yaml.safe_load(open(args.hint_file))
        types = yaml.safe_load(open(args.hint_type_file)) if args.hint_type_file else {}
    tags = args.rep_tags or [f"seed{s}" for s in args.seeds]
    assert len(tags) == len(args.seeds)

    probe_dir = RUNS / args.probe
    probe_dir.mkdir(parents=True, exist_ok=True)
    servers = {u: server_info(u) for u in args.servers}
    (probe_dir / f"servers.{int(time.time())}.json").write_text(json.dumps(servers, indent=1, default=str))

    sampling = None if args.temperature is None else {"temperature": args.temperature}
    jobs = [(t, s, tag) for t in tasks for s, tag in zip(args.seeds, tags)]
    rr = itertools.cycle(args.servers)
    lock = threading.Lock()
    print(f"{args.probe}: {len(jobs)} runs on {len(args.servers)} servers, concurrency {args.concurrency}", flush=True)
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(args.concurrency) as pool:
        futs = {}
        for t, s, tag in jobs:
            with lock:
                url = next(rr)
            hint = None
            text = hints.get(t.task_id)
            if isinstance(text, dict):  # per-run hints (P13): {task_id: {run tag: text}}
                text = text.get(tag)
            if text:
                hint = Hint(text, args.hint_where, types.get(t.task_id) if args.hint_where == "decision" else None,
                            prefix=args.hint_prefix)
            meta = {"probe": args.probe, "run_name": tag, "server": url, "concurrency": args.concurrency}
            futs[pool.submit(one_run, t, s, probe_dir / t.task_id / tag, url, hint, meta, sampling)] = (t.task_id, tag)
        for f in as_completed(futs):
            tid, tag = futs[f]
            done += 1
            try:
                r = f.result()
                print(f"[{done}/{len(jobs)} {time.time() - t0:.0f}s] {tid}/{tag}: {r['stop']} steps={r['steps']}", flush=True)
            except Exception as e:
                print(f"[{done}/{len(jobs)}] {tid}/{tag}: FAILED {e!r}", flush=True)


if __name__ == "__main__":
    main()
