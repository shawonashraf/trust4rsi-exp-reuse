"""P0 step 2b: run the post-hoc decision extractor over each included run (cached).

Uses reveries.decisions.extract_executed with temperature 0 and thinking off, at most 3 requests per
server on the two shared servers (6 in total). Cache: /data/anon/reveries_runs/P0/extract/<key>.json.
Usage: uv run --no-active python quick_runs/p0/extract.py
"""
from __future__ import annotations

import os
import gzip
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from reveries.decisions import extract_executed  # noqa: E402
from reveries.llm import LLM, server_info  # noqa: E402

CACHE = Path(os.environ.get("REVERIES_RUNS", "/data/anon/reveries_runs")) / "P0"
SERVERS = ["http://localhost:8100", "http://localhost:8101"]
PER_SERVER = 3


def load_facts(key: str) -> dict:
    with gzip.open(CACHE / "facts" / f"{key}.json.gz", "rt") as f:
        return json.load(f)


def main():
    runs = pd.read_csv(REPO / "quick_runs/results/P0/run_folders.csv")
    keys = list(runs[runs.included].key)
    out_dir = CACHE / "extract"
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [k for k in keys if not (out_dir / f"{k}.json").is_file()]
    print(f"{len(keys)} runs, {len(todo)} to extract", flush=True)
    sems = {s: threading.Semaphore(PER_SERVER) for s in SERVERS}
    info = {s: server_info(s) for s in SERVERS}

    def work(i_key):
        i, key = i_key
        srv = SERVERS[i % len(SERVERS)]
        f = load_facts(key)
        with sems[srv]:
            llm = LLM(base_url=srv, sampling={"temperature": 0.0}, thinking=False)
            res = extract_executed(llm, f["columns"], f["code_cells"], [], f["final_answer"])
        rec = {"key": key, "server": srv, "server_info": {k: info[srv].get(k) for k in ("model_path", "version")
                                                          if isinstance(info[srv], dict)},
               **res}
        (out_dir / f"{key}.json").write_text(json.dumps(rec, indent=1))
        print(f"done {key} parsed={res['executed'] is not None}", flush=True)

    with ThreadPoolExecutor(PER_SERVER * len(SERVERS)) as ex:
        list(ex.map(work, enumerate(todo)))


if __name__ == "__main__":
    main()
