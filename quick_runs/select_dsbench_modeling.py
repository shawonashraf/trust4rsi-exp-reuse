"""Pick the DSBench data-modelling tasks for P1b with a fixed rule (written before any run).

Rule:
1. Keep competitions with an evaluation script, a GT (perfect) score and a baseline score, and whose
   train/test data (data_resplit/) is under 20 MB in total.
2. Drop text tasks and time-series forecasting tasks (listed below by name, from their descriptions):
   our question is about tabular analysis decisions, and these need different tooling.
3. Sort the rest by name and take 8 at evenly spaced positions (index round(k * n / 8), k = 0..7).
Score: DSBench's relative performance gap, (score - baseline) / (GT - baseline), clipped to [0, 1]
in the summary (raw value kept).
Writes quick_runs/tasks_p1b_modeling.yaml.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

DATA = Path(os.environ.get("REVERIES_DATA", "/data/anon/reveries_data"))
ROOT = DATA / "dsbench_hf" / "data_modeling" / "data"
EVAL = DATA / "dsbench" / "data_modeling" / "evaluation"
PERF = DATA / "dsbench_perf" / "save_performance"
OUT = Path(__file__).resolve().parent / "tasks_p1b_modeling.yaml"

TEXT = {"commonlitreadabilityprize", "feedback-prize-english-language-learning", "google-quest-challenge",
        "nlp-getting-started", "tweet-sentiment-extraction", "us-patent-phrase-to-phrase-matching",
        "learning-agency-lab-automated-essay-scoring-2", "lmsys-chatbot-arena"}
TIME_SERIES = {"covid19-global-forecasting-week-1", "covid19-global-forecasting-week-2",
               "covid19-global-forecasting-week-3", "covid19-global-forecasting-week-4",
               "demand-forecasting-kernels-only", "tabular-playground-series-jan-2022",
               "tabular-playground-series-sep-2022", "playground-series-s3e19", "ventilator-pressure-prediction"}
MAX_MB = 20
N = 8


def main():
    eligible, dropped = [], []
    for name in sorted(os.listdir(ROOT / "data_resplit")):
        d = ROOT / "data_resplit" / name
        mb = sum(f.stat().st_size for f in d.iterdir()) / 1e6
        ok_files = (EVAL / f"{name}_eval.py").exists() and (PERF / "GT" / name / "result.txt").exists() \
            and (PERF / "baseline" / name / "result.txt").exists() and (ROOT / "answers" / name).exists()
        reason = ("missing eval/GT/baseline/answers" if not ok_files else f"{mb:.0f} MB > {MAX_MB} MB" if mb >= MAX_MB
                  else "text task" if name in TEXT else "time series" if name in TIME_SERIES else None)
        (dropped if reason else eligible).append({"name": name, "reason": reason, "mb": round(mb, 1)})
    picks = [eligible[round(k * len(eligible) / N)] for k in range(N)]
    tasks = []
    for p in picks:
        n = p["name"]
        tasks.append({"id": f"dsm_{n}", "source": "dsbench_modeling", "competition": n,
                      "gt": float((PERF / "GT" / n / "result.txt").read_text().strip()),
                      "baseline": float((PERF / "baseline" / n / "result.txt").read_text().strip()),
                      "contamination_risk": "high" if n in ("titanic", "bike-sharing-demand", "spaceship-titanic")
                      else "medium",
                      "risk_reason": "Public Kaggle competition with public notebooks."})
    header = ("# DSBench data-modelling tasks for P1b (selected by quick_runs/select_dsbench_modeling.py;\n"
              "# rule in that file's docstring). Metric per task: see `metric`, read from the DSBench eval script.\n")
    OUT.write_text(header + yaml.safe_dump({"tasks": tasks, "eligible": [e["name"] for e in eligible],
                                            "dropped": dropped}, sort_keys=False, width=120))
    print(len(eligible), "eligible;", [t["competition"] for t in tasks])


if __name__ == "__main__":
    main()
