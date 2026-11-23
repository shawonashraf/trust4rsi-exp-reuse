"""Build docs/report/quick_runs.html: an interactive page explaining quick_runs/REPORT.md.

Every number on the page is read from quick_runs/results/<probe>/summary.json (rule: numbers come from the
executed notebooks, never typed by hand).

  uv run python quick_runs/build_html.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "quick_runs" / "results"
OUT = REPO / "docs" / "report" / "quick_runs.html"
TEMPLATE = Path(__file__).resolve().parent / "report_template.html"


def load(p):
    return json.loads((RES / p / "summary.json").read_text())


def main():
    p10, p9, p1, p1b = load("P10"), load("P9"), load("P1"), load("P1b")
    data = {
        "p10": {"per_task": p10["per_task"], "hits": p10["tasks_with_recall_hit"], "risk_after": p10["risk_after_counts"]},
        "p9": {"first_diff": p9["first_diff"], "identical": p9["identical_to_batched_a"], "pairs": p9["compared_pairs"],
               "per_task": {t: {"answer_consistency": v["answer_consistency"]} for t, v in p9["per_task"].items()}},
        "p1": {"runs": p1["runs"], "agree_SF": p1["agreement_SF"], "agree_BLADE": p1["agreement_BLADE"],
               "ac": p1["answer_consistency_by_source"], "checks": p1["check_table"],
               "first_div": p1["first_divergence_counts"], "behaviour": p1["behaviour_by_source"],
               "silent": p1["silent_type_counts"], "judge": p1["judge_draws_agree"],
               "per_task": {t: {k: v[k] for k in ("risk", "answer_consistency", "score_mean", "first_divergence")}
                            for t, v in p1["per_task"].items()}},
        "p1b": {"runs": p1b["runs"], "stops": p1b["stops"], "pooled": p1b["dsa_pooled_clean"],
                "by_type": p1b["dsa_by_type_clean"], "label_errors": p1b["dsa_label_errors"],
                "all_agree": p1b["dsa_all_agree"], "dsa": [{k: r[k] for k in ("task", "type", "gold", "answers", "n_correct",
                                                                             "answer_consistency")} for r in p1b["dsa_table"]],
                "dsm": p1b["dsm_per_task"], "dsm_valid": p1b["dsm_valid"],
                "agree_DSA": p1b["agreement_rank"]["DSA"], "agree_DSM": p1b["agreement_rank"]["DSM"],
                "behaviour": p1b["behaviour_by_source"]},
    }
    blob = json.dumps(data, default=str).replace("</", "<\\/")
    blob = blob.replace("NaN", "null")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(TEMPLATE.read_text().replace("/*__DATA__*/null", blob))
    print(OUT)


if __name__ == "__main__":
    main()
