"""Pick the DSBench data-analysis questions for P1b with a fixed rule (written before any run).

Rule:
1. Keep competitions whose data is exactly one .xlsx workbook, optionally plus .csv files.
   Drop competitions with images (.jpg), .xlsb/.xlsm workbooks or notebooks (.ipynb), and the
   theory quizzes that have no data file.
2. From each kept competition, take the first question (in data.json order) whose gold answer is an
   option letter, and also the first question whose gold answer is a whole number, if there is one.
   (Whole-number answers cannot be hit by guessing among 4-9 options.) Text answers ('$5753961',
   '31 Dec 2029') are skipped because they need a judge to match.
Writes quick_runs/tasks_p1b_analysis.yaml.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

DATA = Path(os.environ.get("REVERIES_DATA", "/data/anon/reveries_data"))
META = DATA / "dsbench" / "data_analysis" / "data.json"
ROOT = DATA / "dsbench_hf" / "data_analysis" / "data"
OUT = Path(__file__).resolve().parent / "tasks_p1b_analysis.yaml"


def main():
    comps = [json.loads(line) for line in open(META)]
    tasks, dropped = [], []
    for c in comps:
        d = ROOT / c["id"]
        files = sorted(f for f in os.listdir(d) if not f.startswith((".", "question")) and f != "introduction.txt")
        exts = [f.rsplit(".", 1)[-1].lower() for f in files]
        if exts.count("xlsx") != 1 or any(e not in ("xlsx", "csv") for e in exts):
            dropped.append({"id": c["id"], "name": c["name"], "files": files})
            continue
        taken = set()
        for q, a in zip(c["questions"], c["answers"]):
            kind = "letter" if (isinstance(a, str) and len(a) == 1 and a.isalpha()) else (
                "integer" if isinstance(a, int) else None)
            if kind and kind not in taken:
                taken.add(kind)
                tasks.append({"id": f"dsa_{c['id']}_{q}", "source": "dsbench_analysis", "competition": c["id"],
                              "name": c["name"], "question_file": f"{q}.txt", "files": files, "gold": a,
                              "answer_type": kind,
                              "contamination_risk": "medium",
                              "risk_reason": "Public ModelOff competition question with published answers (eloquens.com)."})
    header = ("# DSBench data-analysis questions for P1b (selected by quick_runs/select_dsbench_analysis.py;\n"
              "# rule in that file's docstring). Data: /data/anon/reveries_data/dsbench_hf/data_analysis/data/<id>/\n")
    OUT.write_text(header + yaml.safe_dump({"tasks": tasks, "dropped_competitions": dropped}, sort_keys=False, width=120))
    print(len(tasks), "questions;", sum(t["answer_type"] == "integer" for t in tasks), "integer answers;",
          len(dropped), "competitions dropped")


if __name__ == "__main__":
    main()
