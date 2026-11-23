"""P12 step 2: build what each arm puts at the start of the prompt, per test task.

Arms (predictions/P12.md):
  M   memory: top-3 lessons from the whole pool, BM25 of the task question against each lesson's source-task
      question + lesson text.
  C1  context: all lessons from this task's earlier runs, nothing else.
  C2  the same lessons in the middle of other material (other tasks' lessons and run logs), about 32k tokens.
  C3  the same, about 100k tokens (window 131k; leaves room for the run itself).
Every note has the same format in every arm. Deterministic (fixed shuffle seed).

Writes /data/anon/reveries_runs/P12_inputs/<arm>.yaml ({task_id: text}) and
quick_runs/results/P12/contexts.json (token counts, sha256, which lessons each arm holds).

  uv run --no-active python quick_runs/p12/contexts.py
"""
from __future__ import annotations

import os
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

import yaml
from tokenizers import Tokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "quick_runs"))
from score import load_events, transcript  # noqa: E402

RUNS = Path(os.environ.get("REVERIES_RUNS", "/data/anon/reveries_runs"))
LESSONS = REPO / "quick_runs" / "results" / "P12" / "lessons.jsonl"
OUT_DIR = RUNS / "P12_inputs"
META = REPO / "quick_runs" / "results" / "P12" / "contexts.json"
TOK = Tokenizer.from_file(str(next(Path("/data/cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots").glob("*/tokenizer.json"))))
TEST_TASKS = [s["id"] for s in yaml.safe_load(open(REPO / "quick_runs" / "tasks_p12.yaml"))["tasks"]]
SPECS = {s["id"]: s for f in ("tasks.yaml", "tasks_p1b.yaml") for s in yaml.safe_load(open(REPO / "quick_runs" / f))["tasks"]}
TARGET = {"C2": 32_000, "C3": 100_000}
HEADER = ("Notes from previous experience (written after earlier runs of data science tasks; each note says "
          "which task it came from):\n\n")


def ntok(text: str) -> int:
    return len(TOK.encode(text).ids)


QUESTIONS: dict[str, str] = {}


def task_label(task_id: str) -> str:
    return QUESTIONS[task_id][:160].replace("\n", " ")


def fmt_lesson(L: dict) -> str:
    return (f"[Note from a run on: {task_label(L['source_task'])}]\n{L['title']}\nWhen: {L['when']}\n"
            f"What to do: {L['content']}\n")


def fmt_log(probe: str, task_id: str, run: str) -> str:
    t = transcript(load_events(RUNS / probe / task_id / run), max_chars=12000)
    return f"[Run log from an earlier run on: {task_label(task_id)}]\n{t}\n"


# --- BM25 (Okapi, k1 = 1.5, b = 0.75) over lesson text, query = task question ---
def words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", s.lower())


def bm25_top(query: str, docs: list[str], k: int = 3) -> list[int]:
    toks = [words(d) for d in docs]
    n, avg = len(docs), sum(map(len, toks)) / len(docs)
    df = Counter(w for t in toks for w in set(t))
    q = words(query)
    scores = []
    for i, t in enumerate(toks):
        tf = Counter(t)
        s = sum(math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5)) * tf[w] * 2.5 / (tf[w] + 1.5 * (0.25 + 0.75 * len(t) / avg))
                for w in q if w in tf)
        scores.append(s)
    return sorted(range(n), key=lambda i: -scores[i])[:k]


def main():
    from reveries.tasks import load_tasks
    tasks = {t.task_id: t for f in ("tasks.yaml", "tasks_p1b.yaml") for t in load_tasks(REPO / "quick_runs" / f)[0]}
    QUESTIONS.update({t: v.question.split("\n\nData files")[0] for t, v in tasks.items()})
    lessons = [json.loads(l) for l in open(LESSONS)]
    lessons = [L for L in lessons if L["k"] >= 0]
    # Each lesson is keyed by the question of the task it came from, plus its own text (ReasoningBank keys a
    # memory by the query it was built from). Keying on the lesson text alone drifted to other tasks.
    docs = [f"{QUESTIONS[L['source_task']]} {L['title']} {L['when']} {L['content']}" for L in lessons]
    logs = [(p, json.loads(l)["task_id"], json.loads(l)["run"]) for p in ("P1", "P1b") for l in open(RUNS / p / "records.jsonl")]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    arms = {a: {} for a in ("M", "C1", "C2", "C3")}
    meta = {}
    for tid in TEST_TASKS:
        own = [L for L in lessons if L["source_task"] == tid]
        top = bm25_top(QUESTIONS[tid], docs, 3)
        arms["M"][tid] = HEADER + "\n".join(fmt_lesson(lessons[i]) for i in top)
        own_block = "\n".join(fmt_lesson(L) for L in own)
        arms["C1"][tid] = HEADER + own_block
        rng = random.Random(tid)
        filler = [fmt_lesson(L) for L in lessons if L["source_task"] != tid]
        filler += [fmt_log(*x) for x in logs if x[1] != tid]
        rng.shuffle(filler)
        info = {"own_lessons": len(own), "M_sources": [lessons[i]["source_task"] for i in top],
                "M_own": sum(lessons[i]["source_task"] == tid for i in top)}
        for arm, target in TARGET.items():
            budget = target - ntok(HEADER + own_block)
            picked, used = [], 0
            for x in filler:
                c = ntok(x)
                if used + c > budget:
                    continue
                picked.append(x)
                used += c
                if used > budget - 300:
                    break
            half = len(picked) // 2  # own lessons in the middle
            arms[arm][tid] = HEADER + "\n".join(picked[:half] + [own_block] + picked[half:])
            info[f"{arm}_filler_items"] = len(picked)
        for arm in arms:
            info[f"{arm}_tokens"] = ntok(arms[arm][tid])
            info[f"{arm}_sha256"] = hashlib.sha256(arms[arm][tid].encode()).hexdigest()[:16]
        meta[tid] = info
        print(tid, {k: v for k, v in info.items() if k.endswith("tokens") or k in ("own_lessons", "M_own")})
    for arm, d in arms.items():
        (OUT_DIR / f"{arm}.yaml").write_text(yaml.safe_dump(d, width=10**9, allow_unicode=True))
    META.write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
