"""P10 recall probe: does the model 'know' the expected analysis without seeing the data?

15 tasks x 3 samples, agent sampling (temperature 1.0, thinking on). Replies are saved to
quick_runs/results/P10/replies.jsonl; labels are written by hand (see predictions/P10.md).

  uv run python quick_runs/p10_recall.py --server http://localhost:8100
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reveries.llm import LLM  # noqa: E402
from reveries.tasks import load_tasks  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "P10"
SAMPLES = 3

PROMPT = """You do not have access to the data. Here is a data-analysis task:

---
{task}
---

What is the result of this analysis likely to be? Which model and controls are standard for it?
Answer from what you know about this dataset and question."""


def strip_format(q: str) -> str:
    for marker in ("Formatting instructions:", "When you are done, give your conclusion"):
        q = q.split(marker)[0]
    return q.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://localhost:8100")
    a = ap.parse_args()
    tasks, _ = load_tasks()
    llm = LLM(base_url=a.server)
    jobs = [(t, k) for t in tasks for k in range(SAMPLES)]

    def ask(job):
        t, k = job
        r = llm.chat([{"role": "user", "content": PROMPT.format(task=strip_format(t.question))}], max_tokens=6000)
        return {"task_id": t.task_id, "sample": k, "contamination_risk_before": t.contamination_risk,
                "reply": r.content, "reasoning": r.reasoning, "usage": r.usage, "request": r.request}

    with ThreadPoolExecutor(16) as pool:
        rows = list(pool.map(ask, jobs))
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "replies.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} replies to {OUT / 'replies.jsonl'}")


if __name__ == "__main__":
    main()
