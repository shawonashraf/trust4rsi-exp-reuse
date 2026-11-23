"""Build and execute the probe notebooks (rule 11: every cited number comes from an executed notebook).

  uv run python quick_runs/notebooks.py P10
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parent.parent
NB_DIR = REPO / "docs" / "notebooks"

SETUP = """import json, os, sys, statistics
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd, yaml
REPO = Path.cwd().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "quick_runs"))
RUNS = Path(os.environ.get("REVERIES_RUNS", "/data/anon/reveries_runs"))
TASKS = {t["id"]: t for t in yaml.safe_load(open(REPO / "quick_runs/tasks.yaml"))["tasks"]}
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 80)"""

P10 = [
    "# P10 recall probe\nDoes gemma-4-31B-it name the planted pitfall (Silent Failures) or the source study / "
    "published finding (BLADE) without seeing the data? Labels are hand-written (`quick_runs/results/P10/labels.csv`), "
    "rule in `quick_runs/predictions/P10.md`. A task with >= 2 of 3 hits moves to contamination_risk = high.",
    SETUP,
    """lab = pd.read_csv(REPO / "quick_runs/results/P10/labels.csv")
notes = json.load(open(REPO / "quick_runs/results/P10/notes.json"))
replies = [json.loads(l) for l in open(REPO / "quick_runs/results/P10/replies.jsonl")]
assert len(replies) == 45 and len(lab) == 45
per = lab.groupby("task_id")["recall_hit"].sum().rename("hits").reset_index()
per["risk_before"] = per.task_id.map(lambda t: TASKS[t]["contamination_risk"])
per["recall_hit"] = per.hits >= 2
per["risk_after"] = [ "high" if h else r for h, r in zip(per.recall_hit, per.risk_before)]
per["note"] = per.task_id.map(lambda t: notes.get(t, ""))
per["source"] = per.task_id.str.split("_").str[0]
per""",
    """summary = {
  "replies": len(replies),
  "tasks_with_recall_hit": int(per.recall_hit.sum()),
  "recall_hit_tasks": sorted(per[per.recall_hit].task_id),
  "moved_to_high": sorted(per[per.recall_hit & (per.risk_before != "high")].task_id),
  "hits_by_source": per.groupby("source").hits.sum().to_dict(),
  "risk_after_counts": per.risk_after.value_counts().to_dict(),
  "per_task": per.set_index("task_id")[["hits", "risk_before", "risk_after", "note"]].to_dict("index"),
}
json.dump(summary, open(REPO / "quick_runs/results/P10/summary.json", "w"), indent=1, default=int)
summary""",
]

COMMON = """from metrics import decision_agreement, first_divergence, answer_consistency, consistent_failures, score_spread, task_score
from reveries.decisions import DECISION_TYPES
from score import family_of, SF_CHECKS
DTYPES = [t for t in DECISION_TYPES if t != "final_selection"] + ["final_selection"]
def load_records(probe):
    recs = [json.loads(l) for l in open(RUNS / probe / "records.jsonl")]
    for r in recs:
        r["answer"] = tuple(r["answer"]) if isinstance(r["answer"], list) else r["answer"]
        r["risk"] = TASKS[r["task_id"]]["contamination_risk"]
        r["source"] = "SF" if r["task_id"].startswith("sf_") else "BLADE"
    return recs
def by_task(recs):
    d = defaultdict(list)
    for r in recs: d[r["task_id"]].append(r)
    return dict(sorted(d.items()))
def task_table(groups):
    rows = []
    for t, rs in groups.items():
        row = {"task": t, "risk": TASKS[t]["contamination_risk"], "n": len(rs),
               "answer_consistency": answer_consistency(rs), "score_mean": task_score(rs), "score_sd": score_spread(rs),
               "first_divergence": first_divergence(rs, DTYPES),
               "consistent_failures": ",".join(consistent_failures(rs))}
        for dt in DTYPES:
            row[dt] = decision_agreement(rs, dt)
        rows.append(row)
    return pd.DataFrame(rows)"""

P9 = [
    "# P9 serving noise floor\\nGreedy decoding (temperature 0). Per task: 3 reruns sent together (batched) and 1 run "
    "alone on an idle server (bs1). Any difference is serving noise. Predictions: `quick_runs/predictions/P9.md`.",
    SETUP, COMMON,
    """recs = load_records("P9"); groups = by_task(recs)
print({t: sorted(r["run"] for r in rs) for t, rs in groups.items()})
tt = task_table(groups); tt""",
    """# Where do the greedy runs first differ? Compare the model's text step by step against batched_a.
def contents(task, run):
    return [json.loads(l)["content"] for l in open(RUNS / "P9" / task / run / "events.jsonl") if json.loads(l)["event"] == "llm_call"]
div = []
for t in groups:
    ref = contents(t, "batched_a")
    for run in ["batched_b", "batched_c", "bs1"]:
        other = contents(t, run)
        step, char = None, None
        for i, (a, b) in enumerate(zip(ref, other)):
            if a != b:
                step = i; char = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b))); break
        if step is None and len(ref) != len(other): step, char = min(len(ref), len(other)), 0
        div.append({"task": t, "vs_batched_a": run, "identical": step is None, "first_diff_step": step, "first_diff_char": char,
                    "steps_a": len(ref), "steps_other": len(other)})
div = pd.DataFrame(div); div""",
    """# Answers per run
pd.DataFrame([{"task": r["task_id"], "run": r["run"], "answer": r["answer"], "score": r["score"],
               "checks": r["checks"]} for r in sorted(recs, key=lambda r: (r["task_id"], r["run"]))])""",
    """summary = {"runs": len(recs), "tasks": list(groups),
  "identical_to_batched_a": int(div.identical.sum()), "compared_pairs": len(div),
  "first_diff": div.to_dict("records"),
  "per_task": tt.set_index("task").to_dict("index"),
  "mean_agreement_by_type": tt[DTYPES].mean().round(3).to_dict(),
  "mean_answer_consistency": round(tt.answer_consistency.mean(), 3)}
json.dump(summary, open(REPO / "quick_runs/results/P9/summary.json", "w"), indent=1, default=str)
summary["identical_to_batched_a"], summary["mean_answer_consistency"], summary["mean_agreement_by_type"]""",
]

P1 = [
    "# P1 divergence census\\n15 tasks x 5 runs, no memory, no hint, temperature 1.0. Executed decisions from the "
    "post-hoc extractor. Predictions: `quick_runs/predictions/P1.md`. Also: how the agent works (behaviour section).",
    SETUP, COMMON,
    """recs = load_records("P1"); groups = by_task(recs)
print(len(recs), "runs;", Counter(r["stop"] for r in recs))
tt = task_table(groups)
tt[["task", "risk", "n", "answer_consistency", "score_mean", "score_sd", "first_divergence", "consistent_failures"]]""",
    """# Decision agreement per type, per task (share of run pairs with the same executed value)
tt.set_index("task")[DTYPES].round(2)""",
    """# Decision types ranked by disagreement (pooled = mean over tasks), split by source
rank = pd.DataFrame({"all": tt[DTYPES].mean(), "SF": tt[tt.task.str.startswith("sf_")][DTYPES].mean(),
                     "BLADE": tt[tt.task.str.startswith("blade_")][DTYPES].mean(),
                     "tasks_all_same": (tt[DTYPES] == 1.0).sum()}).sort_values("all")
rank.round(2)""",
    """# First divergence: how often is it one of the first two decision types the runs state?
def first_two(rs):
    c = Counter(t for r in rs for t in r["order"][:2]); return {t for t, _ in c.most_common(2)}
fd = pd.DataFrame([{"task": t, "first_divergence": first_divergence(rs, DTYPES), "first_two_stated": sorted(first_two(rs))} for t, rs in groups.items()])
fd["in_first_two"] = [f in s for f, s in zip(fd.first_divergence, fd.first_two_stated)]
print(Counter(fd.first_divergence)); fd""",
    """# Fork list: task, decision type, the options, runs per option (types with > 1 executed value)
forks = []
for t, rs in groups.items():
    for dt in DTYPES:
        c = Counter(r["decisions"].get(dt) for r in rs if r["decisions"].get(dt) is not None)
        if len(c) > 1:
            forks.append({"task": t, "type": dt, "n_options": len(c), "options": dict(c.most_common())})
forks = pd.DataFrame(forks)
print(len(forks), "forks over", forks.task.nunique(), "tasks")
forks[forks.type.isin(["variables", "model.family", "model.controls", "validation", "metric", "verdict", "framing.population"])]""",
    """# Consistent failures and per-check pass rates (Silent Failures: judge majority of 3 draws; BLADE: expert match)
cf = []
for t, rs in groups.items():
    for c in sorted(set().union(*(r["checks"] for r in rs))):
        v = [r["checks"].get(c) for r in rs]
        cf.append({"task": t, "check": c, "pass": v.count(True), "fail": v.count(False), "unknown": v.count(None),
                   "consistent_failure": all(x is False for x in v)})
cf = pd.DataFrame(cf); cf""",
    """# Judge stability (Silent Failures): share of (run, check) where the 3 judge draws agree
ja = [a for r in recs if r["source"] == "SF" for a in r["judge_agreement"].values()]
print("judge draws agree on", sum(ja), "of", len(ja), "run-checks")""",
    """# Behaviour: how the agent works
def run_events(r):
    return [json.loads(l) for l in open(RUNS / "P1" / r["task_id"] / r["run"] / "events.jsonl")]
beh = []
for r in recs:
    ev = run_events(r)
    cells = [e for e in ev if e["event"] == "code_executed"]
    stated_fam = [family_of(e["value"]) for e in ev if e["event"] == "decision_stated" and e["type"] == "model.family"]
    ex_fam = r["executed"].get("model.family")
    beh.append({"task": r["task_id"], "source": r["source"], "run": r["run"], "stop": r["stop"], "steps": r["steps"],
        "code_cells": len(cells), "cell_errors": sum(not c["ok"] for c in cells),
        "no_code_turns": sum(e["event"] == "no_code" for e in ev),
        "stated_decisions": len(r["stated"]), "stated_types": len(set(d["type"] for d in r["stated"])),
        "silent_types": len(r["executed"].get("silent") or []),
        "families_fitted": len(r["executed"].get("model.all_families") or []),
        "stated_family_last": stated_fam[-1] if stated_fam else None, "executed_family": ex_fam,
        "family_mismatch": (bool(stated_fam) and ex_fam not in (None, "other") and stated_fam[-1] not in ("other", "several")
                            and {"linear_regression": "ols"}.get(ex_fam, ex_fam) != stated_fam[-1]),
        "completion_tokens": r["tokens"]["completion"], "reasoning_tokens": r["tokens"]["reasoning"],
        "minutes": (ev[-1]["t"] - ev[0]["t"]) / 60})
beh = pd.DataFrame(beh)
beh.groupby("source")[["steps", "code_cells", "cell_errors", "no_code_turns", "stated_decisions", "silent_types",
                      "families_fitted", "completion_tokens", "reasoning_tokens", "minutes"]].describe().T.round(1)""",
    """# Stated vs executed model family (keyword map of the last stated model.family vs the extractor)
beh[beh.family_mismatch][["task", "run", "stated_family_last", "executed_family"]]""",
    """# Which decision types are made silently (made in code, never announced)?
sil = Counter(t for r in recs for t in (r["executed"].get("silent") or []))
pd.Series(sil).sort_values(ascending=False)""",
    """# Results split by contamination risk (after P10)
tt.groupby("risk")[["answer_consistency", "score_sd"] + DTYPES].mean().round(2).T""",
    """summary = {"runs": len(recs), "stops": dict(Counter(r["stop"] for r in recs)),
  "agreement_rank_all": rank["all"].round(3).to_dict(), "agreement_SF": rank["SF"].round(3).to_dict(),
  "agreement_BLADE": rank["BLADE"].round(3).to_dict(), "tasks_all_same_by_type": rank["tasks_all_same"].to_dict(),
  "first_divergence_counts": dict(Counter(fd.first_divergence)), "first_divergence_in_first_two": int(fd.in_first_two.sum()),
  "n_forks": len(forks), "forks": forks.to_dict("records"),
  "consistent_failures": cf[cf.consistent_failure][["task", "check"]].to_dict("records"),
  "check_table": cf.to_dict("records"),
  "judge_draws_agree": [int(sum(ja)), len(ja)],
  "answer_consistency_by_source": tt.assign(src=tt.task.str.split("_").str[0]).groupby("src").answer_consistency.mean().round(3).to_dict(),
  "per_task": tt.set_index("task").to_dict("index"),
  "behaviour_by_source": beh.groupby("source")[["steps", "code_cells", "cell_errors", "stated_decisions", "silent_types",
       "families_fitted", "completion_tokens", "minutes"]].mean().round(2).to_dict("index"),
  "family_mismatch_runs": int(beh.family_mismatch.sum()), "silent_type_counts": dict(sil),
  "by_risk": tt.groupby("risk")[["answer_consistency", "score_sd"]].mean().round(3).to_dict("index")}
(REPO / "quick_runs/results/P1").mkdir(parents=True, exist_ok=True)
json.dump(summary, open(REPO / "quick_runs/results/P1/summary.json", "w"), indent=1, default=str)
forks.to_csv(REPO / "quick_runs/results/P1/forks.csv", index=False); cf.to_csv(REPO / "quick_runs/results/P1/checks.csv", index=False)
beh.to_csv(REPO / "quick_runs/results/P1/behaviour.csv", index=False)
"done" """,
]

P1B_SETUP = SETUP.replace('quick_runs/tasks.yaml', 'quick_runs/tasks_p1b.yaml')
P1B_LOAD = """def load_records(probe):
    recs = [json.loads(l) for l in open(RUNS / probe / "records.jsonl")]
    for r in recs:
        t = TASKS[r["task_id"]]
        r["source"] = "DSA" if t["source"] == "dsbench_analysis" else "DSM"
        r["answer_type"] = t.get("answer_type")
        r["risk"] = t["contamination_risk"]
    return recs"""

P1b = [
    "# P1b divergence census on DSBench (ground-truth tasks)\\n34 tasks x 5 runs (26 data-analysis questions, 8 "
    "data-modelling tasks), no memory, no hint, temperature 1.0. Predictions: `quick_runs/predictions/P1b.md`. "
    "Analysis: exact match. Modelling: DSBench relative gap, (score - baseline) / (GT - baseline).",
    P1B_SETUP, COMMON, P1B_LOAD,
    """recs = load_records("P1b"); groups = by_task(recs)
print(len(recs), "runs;", Counter(r["stop"] for r in recs))
print({t: [r["run"] for r in rs if r["stop"] != "final_answer"] for t, rs in groups.items() if any(r["stop"] != "final_answer" for r in rs)})
tt = task_table(groups); tt["src"] = tt.task.map(lambda t: "DSA" if t.startswith("dsa_") else "DSM")
tt[["task", "src", "n", "answer_consistency", "score_mean", "score_sd", "first_divergence", "consistent_failures"]]""",
    """# Analysis questions: per-question answers, accuracy, consistency, consistent wrong answers
dsa = []
for t, rs in groups.items():
    if not t.startswith("dsa_"): continue
    ans = [r["answer"] for r in rs]; c = Counter(a for a in ans if a is not None)
    top, top_n = c.most_common(1)[0] if c else (None, 0)
    dsa.append({"task": t, "type": TASKS[t]["answer_type"], "gold": TASKS[t]["gold"], "answers": ans,
                "n_correct": sum(r["checks"]["correct"] is True for r in rs), "no_answer": sum(a is None for a in ans),
                "answer_consistency": answer_consistency(rs), "modal": top, "modal_n": top_n,
                "all_wrong_same": top_n == len(rs) and all(r["checks"]["correct"] is False for r in rs),
                "all_wrong": all(r["checks"]["correct"] is not True for r in rs)})
dsa = pd.DataFrame(dsa); dsa""",
    """acc = dsa.assign(acc=dsa.n_correct / 5).groupby("type").agg(questions=("task", "size"), accuracy=("acc", "mean"),
    answer_consistency=("answer_consistency", "mean"), all5_correct=("n_correct", lambda x: int((x == 5).sum())),
    all5_wrong=("all_wrong", "sum"), all5_same_wrong=("all_wrong_same", "sum"))
pooled = {"accuracy": round(dsa.n_correct.sum() / (5 * len(dsa)), 3), "answer_consistency": round(dsa.answer_consistency.mean(), 3)}
print(pooled); acc.round(3)""",
    """# DSBench label error: competition 00000020 lists 10 answers for questions 28-37, but question 35 (a letter
# question, options A-I) has the number 14978914492, which belongs to a later question. Its gold is unusable.
LABEL_ERRORS = ["dsa_00000020_question35"]
clean = dsa[~dsa.task.isin(LABEL_ERRORS)]
acc_clean = clean.assign(acc=clean.n_correct / 5).groupby("type").agg(questions=("task", "size"), accuracy=("acc", "mean"),
    answer_consistency=("answer_consistency", "mean"), all5_correct=("n_correct", lambda x: int((x == 5).sum())), all5_wrong=("all_wrong", "sum"))
pooled_clean = {"questions": len(clean), "accuracy": round(clean.n_correct.sum() / (5 * len(clean)), 3),
                "answer_consistency": round(clean.answer_consistency.mean(), 3)}
print(pooled_clean); acc_clean.round(3)""",
    """# Is consistency tied to correctness? (questions where all 5 runs agree)
agree = dsa[dsa.answer_consistency == 1.0]
print(len(agree), "questions with all 5 answers identical;", int((agree.n_correct == 5).sum()), "all right,",
      int((agree.n_correct == 0).sum()), "all wrong")""",
    """# Modelling tasks: relative gap per run, spread, valid submissions
dsm = []
for r in recs:
    if r["source"] != "DSM": continue
    dsm.append({"task": r["task_id"], "run": r["run"], "stop": r["stop"], "valid": r["checks"]["valid_submission"],
                "raw_metric": r.get("raw_metric"), "rel_gap": r["score"], "score_error": (r.get("score_error") or "")[:80],
                "family": r["executed"].get("model.family"), "all_families": r["executed"].get("model.all_families"),
                "validation": r["executed"].get("validation"), "steps": r["steps"]})
dsm = pd.DataFrame(dsm)
dsm_task = dsm.groupby("task").agg(valid=("valid", "sum"), rel_gap_mean=("rel_gap", "mean"), rel_gap_sd=("rel_gap", "std"),
    rel_gap_min=("rel_gap", "min"), rel_gap_max=("rel_gap", "max"), families=("family", lambda x: dict(Counter(x))),
    validation=("validation", lambda x: dict(Counter(x))))
print("valid submissions:", int(dsm.valid.sum()), "of", len(dsm)); dsm_task.round(3)""",
    """# Decision agreement by type, split by part (mean over tasks)
rank = pd.DataFrame({"all": tt[DTYPES].mean(), "DSA": tt[tt.src == "DSA"][DTYPES].mean(), "DSM": tt[tt.src == "DSM"][DTYPES].mean(),
                     "tasks_scored_DSA": tt[tt.src == "DSA"][DTYPES].notna().sum(), "tasks_scored_DSM": tt[tt.src == "DSM"][DTYPES].notna().sum()}).sort_values("all")
print(Counter(tt[tt.src == "DSA"].first_divergence), Counter(tt[tt.src == "DSM"].first_divergence)); rank.round(2)""",
    """# Analysis questions: on questions where runs disagree, which decision types differ?
dis = tt[(tt.src == "DSA") & (tt.answer_consistency < 1.0)]
pd.DataFrame({"disagreeing_q": dis[DTYPES].mean(), "agreeing_q": tt[(tt.src == "DSA") & (tt.answer_consistency == 1.0)][DTYPES].mean()}).round(2)""",
    """# bike-sharing-demand: do runs use casual / registered (in train only; they sum to the target)?
def code_cells(r):
    return [json.loads(l) for l in open(RUNS / "P1b" / r["task_id"] / r["run"] / "events.jsonl") if json.loads(l)["event"] == "code_executed"]
bike = []
for r in groups["dsm_bike-sharing-demand"]:
    cells = code_cells(r)
    feat = [c for c in cells if ("casual" in c["code"] or "registered" in c["code"])]
    bike.append({"run": r["run"], "cells_mentioning": len(feat),
                 "dropped": any(("drop" in c["code"] and ("casual" in c["code"] or "registered" in c["code"])) for c in cells),
                 "rel_gap": r["score"], "extractor_variables_has_them": bool({"casual", "registered"} & set(r["executed"].get("variables") or []))})
pd.DataFrame(bike)""",
    """# Behaviour: how the agent works
def run_events(r):
    return [json.loads(l) for l in open(RUNS / "P1b" / r["task_id"] / r["run"] / "events.jsonl")]
beh = []
for r in recs:
    ev = run_events(r); cells = [e for e in ev if e["event"] == "code_executed"]
    beh.append({"task": r["task_id"], "source": r["source"], "run": r["run"], "stop": r["stop"], "steps": r["steps"],
        "code_cells": len(cells), "cell_errors": sum(not c["ok"] for c in cells),
        "stated_types": len(set(d["type"] for d in r["stated"])), "silent_types": len(r["executed"].get("silent") or []),
        "completion_tokens": r["tokens"]["completion"], "minutes": (ev[-1]["t"] - ev[0]["t"]) / 60})
beh = pd.DataFrame(beh)
beh.groupby("source")[["steps", "code_cells", "cell_errors", "stated_types", "silent_types", "completion_tokens", "minutes"]].describe().T.round(1)""",
    """summary = {"runs": len(recs), "stops": dict(Counter(r["stop"] for r in recs)),
  "dsa_pooled": pooled, "dsa_by_type": acc.round(3).to_dict("index"), "dsa_label_errors": LABEL_ERRORS,
  "dsa_pooled_clean": pooled_clean, "dsa_by_type_clean": acc_clean.round(3).to_dict("index"),
  "dsa_all_agree": [len(agree), int((agree.n_correct == 5).sum()), int((agree.n_correct == 0).sum())],
  "dsa_consistent_wrong_same": sorted(dsa[dsa.all_wrong_same].task), "dsa_all_wrong": sorted(dsa[dsa.all_wrong].task),
  "dsa_table": dsa.to_dict("records"),
  "dsm_valid": [int(dsm.valid.sum()), len(dsm)], "dsm_tasks_sd_above_0.02": int((dsm_task.rel_gap_sd > 0.02).sum()),
  "dsm_per_task": dsm_task.round(4).to_dict("index"),
  "agreement_rank": rank[["all", "DSA", "DSM"]].round(3).to_dict(),
  "first_divergence_DSA": dict(Counter(tt[tt.src == "DSA"].first_divergence)), "first_divergence_DSM": dict(Counter(tt[tt.src == "DSM"].first_divergence)),
  "bike_casual_registered": bike,
  "behaviour_by_source": beh.groupby("source")[["steps", "code_cells", "cell_errors", "silent_types", "minutes"]].mean().round(2).to_dict("index"),
  "per_task": tt.set_index("task").to_dict("index")}
(REPO / "quick_runs/results/P1b").mkdir(parents=True, exist_ok=True)
json.dump(summary, open(REPO / "quick_runs/results/P1b/summary.json", "w"), indent=1, default=str)
dsa.to_csv(REPO / "quick_runs/results/P1b/analysis.csv", index=False); dsm.to_csv(REPO / "quick_runs/results/P1b/modeling.csv", index=False)
beh.to_csv(REPO / "quick_runs/results/P1b/behaviour.csv", index=False)
"done" """,
]

P2P4 = [
    "# P2 + P4: correct hint and wrong hint on the 4 Silent Failures tasks with consistent failures\\n"
    "4 tasks x 5 runs per arm. P1 = no hint (same tasks). P2 = correct hint, P4 = wrong hint, both at the start of the "
    "prompt. Predictions: `quick_runs/predictions/P2_P4.md`. Hints: `quick_runs/hints/`.",
    SETUP, COMMON,
    """TASKS4 = ["sf_tf01_goal_imbalance", "sf_tf02_player_rep_norm", "sf_tf03_correlation_causation", "sf_tf05_credit_card_leakage"]
HINTS = {"P2": yaml.safe_load(open(REPO / "quick_runs/hints/correct.yaml")), "P4": yaml.safe_load(open(REPO / "quick_runs/hints/wrong.yaml"))}
WRONG_COL = {"sf_tf01_goal_imbalance": "assists", "sf_tf02_player_rep_norm": "height_in_cm",
             "sf_tf03_correlation_causation": "assists", "sf_tf05_credit_card_leakage": "reports"}
TARGET = {"sf_tf01_goal_imbalance": ["checks_class_distribution", "uses_class_weight", "uses_resampling", "uses_stratified_split"],
          "sf_tf02_player_rep_norm": ["handles_unit_suffix_in_height"], "sf_tf03_correlation_causation": ["column_x_included", "applies_causal_reasoning"],
          "sf_tf05_credit_card_leakage": ["used_leakage_features"]}
recs = {a: [r for r in load_records(a) if r["task_id"] in TASKS4] for a in ["P1", "P2", "P4"]}
print({a: (len(v), dict(Counter(r["stop"] for r in v))) for a, v in recs.items()})""",
    """# Targeted checks: passes out of 5, per arm (column_x_included and used_leakage_features are 'pass' when the column was NOT used)
rows = []
for t in TASKS4:
    for c in TARGET[t]:
        row = {"task": t, "check": c}
        for a in recs:
            v = [r["checks"].get(c) for r in recs[a] if r["task_id"] == t]
            row[a] = f"{v.count(True)}/{len(v)}" + (f" ({v.count(None)} unk)" if None in v else "")
        rows.append(row)
targeted = pd.DataFrame(rows); targeted""",
    """# Wrong hint (P4): is the named column a feature of the final model? Extractor's executed variables, per arm.
def uses(r, col):
    v = r["executed"].get("variables") or []
    return any(col == x or x.startswith(col + "_") for x in v)
wr = []
for a in recs:
    for r in recs[a]:
        wr.append({"arm": a, "task": r["task_id"], "run": r["run"], "col": WRONG_COL[r["task_id"]], "uses_col": uses(r, WRONG_COL[r["task_id"]])})
wr = pd.DataFrame(wr)
wr.pivot_table(index="task", columns="arm", values="uses_col", aggfunc="sum")""",
    """# Does the agent's text mention the hint? (assistant messages: column name or key word from the hint)
KEY = {"P2": {"sf_tf01_goal_imbalance": r"imbalanc|class balance|class_weight|stratif", "sf_tf02_player_rep_norm": r"'cm'|cm suffix|strip",
              "sf_tf03_correlation_causation": r"column_x", "sf_tf05_credit_card_leakage": r"leak"},
       "P4": {t: c for t, c in WRONG_COL.items()}}
import re
def texts(a, r):
    return [json.loads(l)["content"] for l in open(RUNS / a / r["task_id"] / r["run"] / "events.jsonl") if json.loads(l)["event"] == "llm_call"]
DOUBT = r"(hint|note|previous experience)[^.]{0,120}(wrong|incorrect|not (seem|appear)|doubt|verify|check whether|actually)"
men = []
for a in ["P2", "P4"]:
    for r in recs[a]:
        tx = "\\n".join(texts(a, r))
        men.append({"arm": a, "task": r["task_id"], "run": r["run"], "mentions": bool(re.search(KEY[a][r["task_id"]], tx, re.I)),
                    "doubt_phrase": bool(re.search(DOUBT, tx, re.I)), "steps": r["steps"]})
men = pd.DataFrame(men)
men.groupby(["arm", "task"])[["mentions", "doubt_phrase"]].sum()""",
    """# Test scores per arm (SF metric from the final JSON) and steps
sc = pd.DataFrame([{"arm": a, "task": r["task_id"], "run": r["run"], "score": r["score"], "steps": r["steps"]} for a in recs for r in recs[a]])
sc.pivot_table(index="task", columns="arm", values="score", aggfunc=["mean", "std", "count"]).round(4)""",
    """sc.groupby("arm").steps.mean().round(2)""",
    """summary = {"runs": {a: len(v) for a, v in recs.items()}, "stops": {a: dict(Counter(r["stop"] for r in v)) for a, v in recs.items()},
  "targeted_checks": targeted.to_dict("records"),
  "wrong_col_used": wr.pivot_table(index="task", columns="arm", values="uses_col", aggfunc="sum").to_dict("index"),
  "mentions": men.groupby(["arm", "task"])[["mentions", "doubt_phrase"]].sum().reset_index().to_dict("records"),
  "score": sc.groupby(["task", "arm"]).score.agg(["mean", "std", "count"]).round(5).reset_index().to_dict("records"),
  "steps_mean": sc.groupby("arm").steps.mean().round(2).to_dict()}
(REPO / "quick_runs/results/P2_P4").mkdir(parents=True, exist_ok=True)
json.dump(summary, open(REPO / "quick_runs/results/P2_P4/summary.json", "w"), indent=1, default=str)
wr.merge(men, on=["arm", "task", "run"], how="left").merge(sc, on=["arm", "task", "run"]).to_csv(REPO / "quick_runs/results/P2_P4/runs.csv", index=False)
"done" """,
]

P12_SETUP = SETUP.replace('quick_runs/tasks.yaml', 'quick_runs/tasks_p12.yaml')
P12 = [
    "# P12 context saturation: in-context lessons vs a memory store\\n10 ground-truth tasks x 5 runs per arm. "
    "A0 no memory; M top-3 retrieved lessons; C1 the task's own lessons in context (~2k tokens); C2 the same lessons "
    "inside ~32k tokens of other material; C3 inside ~100k. Lessons written by the agent from its P1/P1b runs with "
    "the evaluator's outcome. All arms on one 4-GPU server (131k window). Predictions: `quick_runs/predictions/P12.md`.",
    P12_SETUP, COMMON,
    """ARMS = ["P12_A0", "P12_M", "P12_C1", "P12_C2", "P12_C3"]
NAME = {"P12_A0": "A0 none", "P12_M": "M memory", "P12_C1": "C1 ctx 2k", "P12_C2": "C2 ctx 32k", "P12_C3": "C3 ctx 100k"}
TARGET = {"sf_tf01_goal_imbalance": ["checks_class_distribution", "uses_class_weight", "uses_resampling", "uses_stratified_split"],
          "sf_tf02_player_rep_norm": ["handles_unit_suffix_in_height"], "sf_tf03_correlation_causation": ["column_x_included", "applies_causal_reasoning"],
          "sf_tf05_credit_card_leakage": ["used_leakage_features"]}
CTX = json.load(open(REPO / "quick_runs/results/P12/contexts.json"))
def load(arm):
    recs = [json.loads(l) for l in open(RUNS / arm / "records.jsonl")]
    for r in recs:
        r["arm"] = arm; r["answer"] = tuple(r["answer"]) if isinstance(r["answer"], list) else r["answer"]
        ev = [json.loads(l) for l in open(RUNS / arm / r["task_id"] / r["run"] / "events.jsonl")]
        calls = [e for e in ev if e["event"] == "llm_call" and e.get("usage")]
        r["first_prompt_tokens"] = calls[0]["usage"]["prompt_tokens"] if calls else None
        r["peak_tokens"] = max((c["usage"]["prompt_tokens"] + c["usage"]["completion_tokens"] for c in calls), default=None)
        r["sum_prompt_tokens"] = sum(c["usage"]["prompt_tokens"] for c in calls)
        r["minutes"] = (ev[-1]["t"] - ev[0]["t"]) / 60
    return recs
recs = {a: load(a) for a in ARMS}
print({a: (len(v), dict(Counter(r["stop"] for r in v))) for a, v in recs.items()})""",
    """# Context actually given (tokens of the first prompt), per arm, mean over tasks
pd.DataFrame({NAME[a]: pd.Series([r["first_prompt_tokens"] for r in v]).describe()[["mean", "min", "max"]] for a, v in recs.items()}).round(0)""",
    """# Silent Failures with consistent failures in P1: targeted checks passed (out of 5 runs)
rows = []
for t, checks in TARGET.items():
    for c in checks:
        row = {"task": t, "check": c}
        for a in ARMS:
            v = [r["checks"].get(c) for r in recs[a] if r["task_id"] == t]
            row[NAME[a]] = v.count(True)
        rows.append(row)
targeted = pd.DataFrame(rows)
pooled = targeted[[NAME[a] for a in ARMS]].sum() / (5 * len(targeted))
print("share of targeted check-runs passed:", pooled.round(3).to_dict()); targeted""",
    """# Harm check: TF04 / TF06 leak-free (all checks; passes out of 5)
rows = []
for t in ["sf_tf04_balance_torque_leakfree", "sf_tf06_metric_select_leakfree"]:
    for c in sorted({c for r in recs["P12_A0"] if r["task_id"] == t for c in r["checks"]}):
        row = {"task": t, "check": c}
        for a in ARMS:
            v = [r["checks"].get(c) for r in recs[a] if r["task_id"] == t]
            row[NAME[a]] = f"{v.count(True)}/{len(v)}" + (f" ({v.count(None)} unk)" if None in v else "")
        rows.append(row)
harm = pd.DataFrame(rows); harm""",
    """# TF05 in detail: how many of the 4 leaking columns the final model kept (judge's list, most common over 3 draws)
def leak_kept(a, r):
    j = json.load(open(RUNS / a / r["task_id"] / r["run"] / "judge.json"))
    return len(Counter(tuple(sorted(d.get("leakage_features_used") or [])) for d in j["draws"]).most_common(1)[0][0])
tf05_kept = {NAME[a]: [leak_kept(a, r) for r in sorted(recs[a], key=lambda r: r["run"]) if r["task_id"] == "sf_tf05_credit_card_leakage"] for a in ARMS}
# TF04 in detail: runs whose executed code fits a model (.fit() call) vs apply a hand-written torque rule
def fits(a, r):
    ev = [json.loads(l) for l in open(RUNS / a / r["task_id"] / r["run"] / "events.jsonl")]
    return any(".fit(" in e["code"] for e in ev if e["event"] == "code_executed")
tf04_fit = {NAME[a]: sum(fits(a, r) for r in recs[a] if r["task_id"] == "sf_tf04_balance_torque_leakfree") for a in ARMS}
print("TF05 leaking columns kept per run:", tf05_kept); print("TF04 runs that fit a model (of 5):", tf04_fit)""",
    """# DSBench: analysis questions correct (of 5) and answer consistency; modelling relative gap
rows = []
for t in [t for t in TASKS if t.startswith("ds")]:
    row = {"task": t}
    for a in ARMS:
        rs = [r for r in recs[a] if r["task_id"] == t]
        if t.startswith("dsa_"):
            row[NAME[a]] = f"{sum(r['checks'].get('correct') is True for r in rs)}/{len(rs)} (ac {answer_consistency(rs):.1f})"
        else:
            sc = [r["score"] for r in rs if r["score"] is not None]
            row[NAME[a]] = f"{statistics.mean(sc):.3f} +- {statistics.stdev(sc):.3f}" if len(sc) > 1 else str(sc)
    rows.append(row)
ds = pd.DataFrame(rows); ds""",
    """# Test-set scores, all tasks (mean over runs)
sc = pd.DataFrame([{"arm": NAME[a], "task": r["task_id"], "score": r["score"]} for a in ARMS for r in recs[a]])
sc.pivot_table(index="task", columns="arm", values="score", aggfunc="mean")[[NAME[a] for a in ARMS]].round(4)""",
    """# Cost and process per arm
cost = pd.DataFrame([{"arm": NAME[a], "steps": r["steps"], "minutes": r["minutes"], "first_prompt_tokens": r["first_prompt_tokens"],
                      "sum_prompt_tokens": r["sum_prompt_tokens"], "peak_fill": (r["peak_tokens"] or 0) / 131072,
                      "src": r["task_id"].split("_")[0]} for a in ARMS for r in recs[a]])
cost.groupby("arm")[["steps", "minutes", "first_prompt_tokens", "sum_prompt_tokens", "peak_fill"]].mean().loc[[NAME[a] for a in ARMS]].round(2)""",
    """cost.groupby(["src", "arm"])[["steps", "minutes"]].mean().unstack(0).loc[[NAME[a] for a in ARMS]].round(1)""",
    """summary = {"runs": {a: len(v) for a, v in recs.items()}, "stops": {a: dict(Counter(r["stop"] for r in v)) for a, v in recs.items()},
  "targeted": targeted.to_dict("records"), "targeted_pooled": pooled.round(3).to_dict(), "harm": harm.to_dict("records"),
  "dsbench": ds.to_dict("records"), "tf05_leak_cols_kept": tf05_kept, "tf04_runs_fitting_model": tf04_fit,
  "score_mean": sc.pivot_table(index="task", columns="arm", values="score", aggfunc="mean").round(5).to_dict("index"),
  "cost": cost.groupby("arm")[["steps", "minutes", "first_prompt_tokens", "sum_prompt_tokens", "peak_fill"]].mean().round(3).to_dict("index"),
  "cost_by_src": cost.groupby(["src", "arm"])[["steps", "minutes"]].mean().round(2).reset_index().to_dict("records"),
  "contexts": CTX}
json.dump(summary, open(REPO / "quick_runs/results/P12/summary.json", "w"), indent=1, default=str)
cost.to_csv(REPO / "quick_runs/results/P12/runs_cost.csv", index=False)
"done" """,
]


P13 = [
    "# P13 guided retrieval: a guide (same model) judges usefulness after retrieval and asks for retries\\n"
    "6 Silent Failures tasks x 5 runs per arm. E embedding top-3; G0/G1/G3 guide with 0/1/3 retries; E-p and G3-p the "
    "same on a store with 12 planted wrong lessons (4 failing tasks only). A0, M, C1 reused from P12. "
    "P13 ran on a 2-GPU server (TP=2, 64k window) with the same model, image and sampling; P12 used a 4-GPU server. "
    "Predictions: `quick_runs/predictions/P13.md`. Guide trace: `quick_runs/results/P13/guide_trace.jsonl`.",
    P12_SETUP, COMMON,
    """NEW = ["P13_E", "P13_G0", "P13_G1", "P13_G3", "P13_Ep", "P13_G3p"]
OLD = ["P12_A0", "P12_M", "P12_C1"]
ARMS = OLD + NEW
NAME = {"P12_A0": "A0 none", "P12_M": "M bm25", "P12_C1": "C1 own", "P13_E": "E emb", "P13_G0": "G0", "P13_G1": "G1",
        "P13_G3": "G3", "P13_Ep": "E-p", "P13_G3p": "G3-p"}
TRACE_ARM = {"P13_E": "E", "P13_G0": "G0", "P13_G1": "G1", "P13_G3": "G3", "P13_Ep": "E-p", "P13_G3p": "G3-p"}
TASKS4 = ["sf_tf01_goal_imbalance", "sf_tf02_player_rep_norm", "sf_tf03_correlation_causation", "sf_tf05_credit_card_leakage"]
SF6 = TASKS4 + ["sf_tf04_balance_torque_leakfree", "sf_tf06_metric_select_leakfree"]
TARGET = {"sf_tf01_goal_imbalance": ["checks_class_distribution", "uses_class_weight", "uses_resampling", "uses_stratified_split"],
          "sf_tf02_player_rep_norm": ["handles_unit_suffix_in_height"], "sf_tf03_correlation_causation": ["column_x_included", "applies_causal_reasoning"],
          "sf_tf05_credit_card_leakage": ["used_leakage_features"]}
WRONG_COL = {"sf_tf01_goal_imbalance": "assists", "sf_tf02_player_rep_norm": "height_in_cm",
             "sf_tf03_correlation_causation": "assists", "sf_tf05_credit_card_leakage": "reports"}
def load(arm):
    recs = [json.loads(l) for l in open(RUNS / arm / "records.jsonl")]
    recs = [r for r in recs if r["task_id"] in SF6]
    for r in recs:
        r["arm"] = arm
        ev = [json.loads(l) for l in open(RUNS / arm / r["task_id"] / r["run"] / "events.jsonl")]
        calls = [e for e in ev if e["event"] == "llm_call" and e.get("usage")]
        r["first_prompt_tokens"] = calls[0]["usage"]["prompt_tokens"] if calls else None
        r["sum_prompt_tokens"] = sum(c["usage"]["prompt_tokens"] for c in calls)
        r["sum_completion_tokens"] = sum(c["usage"]["completion_tokens"] for c in calls)
        r["minutes"] = (ev[-1]["t"] - ev[0]["t"]) / 60
    return recs
recs = {a: load(a) for a in ARMS}
TRACE = {(t["arm"], t["task"], t["run"]): t for t in map(json.loads, open(REPO / "quick_runs/results/P13/guide_trace.jsonl"))}
print({a: (len(v), dict(Counter(r["stop"] for r in v))) for a, v in recs.items()})""",
    """# Targeted checks on the 4 failing tasks: passes out of 5 runs (column_x_included / used_leakage_features pass = column NOT used)
rows = []
for t, checks in TARGET.items():
    for c in checks:
        row = {"task": t, "check": c}
        for a in ARMS:
            v = [r["checks"].get(c) for r in recs[a] if r["task_id"] == t]
            row[NAME[a]] = v.count(True)
        rows.append(row)
targeted = pd.DataFrame(rows)
pooled = targeted[[NAME[a] for a in ARMS]].sum() / (5 * len(targeted))
unknown = {NAME[a]: sum(r["checks"].get(c) is None for t, cs in TARGET.items() for c in cs for r in recs[a] if r["task_id"] == t) for a in ARMS}
print("share of targeted check-runs passed:", pooled.round(3).to_dict()); print("unknown check results:", unknown); targeted""",
    """# Pooled share per task (how much of each arm's total comes from which task)
pt = pd.DataFrame({NAME[a]: {t[:7]: sum(r["checks"].get(c) is True for c in TARGET[t] for r in recs[a] if r["task_id"] == t) / (5 * len(TARGET[t]))
                            for t in TASKS4} for a in ARMS}).round(2); pt""",
    """# TF05: leaking columns the final model kept (of 4), per run; judge's list, most common over its draws
def leak_kept(a, r):
    j = json.load(open(RUNS / a / r["task_id"] / r["run"] / "judge.json"))
    return len(Counter(tuple(sorted(d.get("leakage_features_used") or [])) for d in j["draws"]).most_common(1)[0][0])
tf05_kept = {NAME[a]: [leak_kept(a, r) for r in sorted(recs[a], key=lambda r: r["run"]) if r["task_id"] == "sf_tf05_credit_card_leakage"] for a in ARMS}
tf05_kept""",
    """# What the agent was shown (P13 arms): own-task lessons, wrong lessons, and lessons naming the planted fix (keyword proxy, to hand-check)
FIX = {"sf_tf01_goal_imbalance": r"imbalanc|class_weight|class weight|stratif", "sf_tf02_player_rep_norm": r"'cm'|suffix|unit",
       "sf_tf03_correlation_causation": r"column_x", "sf_tf05_credit_card_leakage": r"expenditure|majorcards|\\bactive\\b|leak",
       "sf_tf04_balance_torque_leakfree": r"$^", "sf_tf06_metric_select_leakfree": r"$^"}
import re
sh = []
for a in NEW:
    for r in recs[a]:
        tr = TRACE[(TRACE_ARM[a], r["task_id"], r["run"])]
        L = tr["shown_lessons"]
        sh.append({"arm": NAME[a], "task": r["task_id"][:7], "run": r["run"], "n_shown": len(L),
                   "own": sum(x["source_task"] == r["task_id"] and not x["wrong"] for x in L),
                   "wrong": sum(bool(x["wrong"]) for x in L),
                   "fix_kw": sum(bool(re.search(FIX[r["task_id"]], x["title"], re.I)) and not x["wrong"] for x in L),
                   "rounds": len(tr["rounds"])})
sh = pd.DataFrame(sh)
sh.groupby(["task", "arm"])[["n_shown", "own", "fix_kw", "wrong", "rounds"]].sum().unstack(1)""",
    """# Trust funnel on the poisoned store, per run: wrong lesson retrieved (seen) -> accepted -> shown -> named column left out of the final model
WRONG_IDS = set(range(708, 720))
def uses(r, col):
    v = r["executed"].get("variables") or []
    return any(col == x or x.startswith(col + "_") for x in v)
fun = []
for a in ["P13_Ep", "P13_G3p"]:
    for r in recs[a]:
        tr = TRACE[(TRACE_ARM[a], r["task_id"], r["run"])]
        seen = set(tr.get("seen", tr["shown"])); acc = set(tr.get("accepted", tr["shown"])); shown = set(tr["shown"])
        fun.append({"arm": NAME[a], "task": r["task_id"][:7], "run": r["run"], "retrieved": bool(seen & WRONG_IDS),
                    "accepted": bool(acc & WRONG_IDS), "shown": bool(shown & WRONG_IDS),
                    "col_dropped": not uses(r, WRONG_COL[r["task_id"]]), "variables": r["executed"].get("variables")})
fun = pd.DataFrame(fun)
fun.groupby(["arm", "task"])[["retrieved", "accepted", "shown", "col_dropped"]].sum()""",
    """# Named column dropped, split by whether a wrong lesson was shown (all poisoned runs, plus clean arms as baseline)
base = pd.DataFrame([{"arm": NAME[a], "task": r["task_id"][:7], "col_dropped": not uses(r, WRONG_COL[r["task_id"]])}
                     for a in ["P12_A0", "P13_E", "P13_G3"] for r in recs[a] if r["task_id"] in TASKS4])
print(base.groupby(["arm", "task"]).col_dropped.sum().unstack(0))
fun.groupby(["arm", "shown"]).col_dropped.agg(["sum", "count"])""",
    """# Harm: TF04 / TF06 test scores (mean over runs) and their checks, vs A0
sc = pd.DataFrame([{"arm": NAME[a], "task": r["task_id"], "score": r["score"]} for a in ARMS for r in recs[a]])
scores = sc.pivot_table(index="task", columns="arm", values="score", aggfunc="mean")[[NAME[a] for a in ARMS if a not in ("P13_Ep", "P13_G3p")] + ["E-p", "G3-p"]].round(4)
rows = []
for t in ["sf_tf04_balance_torque_leakfree", "sf_tf06_metric_select_leakfree"]:
    for c in sorted({c for r in recs["P12_A0"] if r["task_id"] == t for c in r["checks"]}):
        rows.append({"task": t[:7], "check": c, **{NAME[a]: sum(r["checks"].get(c) is True for r in recs[a] if r["task_id"] == t) for a in ARMS[:7]}})
harm = pd.DataFrame(rows); print(harm.to_string()); scores""",
    """# Cost: guide calls and tokens per run vs the agent's own prompt tokens
co = []
for a in NEW:
    for r in recs[a]:
        tr = TRACE[(TRACE_ARM[a], r["task_id"], r["run"])]
        gp = sum(x["usage"]["prompt_tokens"] for x in tr["rounds"] if x.get("usage"))
        gc = sum(x["usage"]["completion_tokens"] for x in tr["rounds"] if x.get("usage"))
        co.append({"arm": NAME[a], "guide_calls": len(tr["rounds"]), "guide_prompt": gp, "guide_completion": gc,
                   "guide_parse_fail": sum(not x.get("parsed", True) for x in tr["rounds"]),
                   "agent_prompt": r["sum_prompt_tokens"], "agent_completion": r["sum_completion_tokens"], "steps": r["steps"], "minutes": r["minutes"]})
co = pd.DataFrame(co)
cs = co.groupby("arm").mean().loc[[NAME[a] for a in NEW]]
cs["guide_share_of_prompt"] = cs.guide_prompt / cs.agent_prompt
cs["guide_share_of_all_tokens"] = (cs.guide_prompt + cs.guide_completion) / (cs.agent_prompt + cs.agent_completion)
first = {NAME[a]: round(statistics.mean(r["first_prompt_tokens"] for r in recs[a] if r["task_id"] in TASKS4)) for a in ARMS}
print("first prompt tokens, failing tasks, mean:", first)
cs.round(3)""",
    """# Runs that hit the step limit
pd.DataFrame([{"arm": NAME[a], "task": r["task_id"], "run": r["run"], "steps": r["steps"]} for a in ARMS for r in recs[a] if r["stop"] != "final_answer"])""",
    """summary = {"runs": {a: len(v) for a, v in recs.items()}, "stops": {a: dict(Counter(r["stop"] for r in v)) for a, v in recs.items()},
  "targeted": targeted.to_dict("records"), "targeted_pooled": pooled.round(3).to_dict(), "unknown_checks": unknown,
  "pooled_by_task": pt.to_dict(), "tf05_leak_cols_kept": tf05_kept,
  "shown": sh.groupby(["task", "arm"])[["n_shown", "own", "fix_kw", "wrong", "rounds"]].sum().reset_index().to_dict("records"),
  "trust_funnel": fun.groupby(["arm", "task"])[["retrieved", "accepted", "shown", "col_dropped"]].sum().reset_index().to_dict("records"),
  "col_dropped_by_shown": fun.groupby(["arm", "shown"]).col_dropped.agg(["sum", "count"]).reset_index().to_dict("records"),
  "col_dropped_clean": base.groupby(["arm", "task"]).col_dropped.sum().reset_index().to_dict("records"),
  "harm_checks": harm.to_dict("records"), "score_mean": scores.to_dict("index"), "cost": cs.round(4).to_dict("index"),
  "first_prompt_tokens_failing": first}
(REPO / "quick_runs/results/P13").mkdir(parents=True, exist_ok=True)
json.dump(summary, open(REPO / "quick_runs/results/P13/summary.json", "w"), indent=1, default=str)
fun.to_csv(REPO / "quick_runs/results/P13/trust_funnel.csv", index=False); co.to_csv(REPO / "quick_runs/results/P13/runs_cost.csv", index=False)
"done" """,
]

P14 = [
    "# P14 guide prompt ablation\\n"
    "P13's guide prompt asked for notes 'consistent with the task'. P14 replaces that with 'consistent with the data "
    "schema. A note can be useful even when it goes against the task's wording, if the data suggests ...'. "
    "Stage 1 (guide only, reasons saved): G3o / G3o-p = P13 prompt again; G3n / G3n-p = P14 prompt. Stage 2: agent "
    "runs for G3n (6 tasks) and G3n-p (4 failing tasks). Predictions: `quick_runs/predictions/P14.md`.",
    P12_SETUP, COMMON,
    """import re
TASKS4 = ["sf_tf01_goal_imbalance", "sf_tf02_player_rep_norm", "sf_tf03_correlation_causation", "sf_tf05_credit_card_leakage"]
SF6 = TASKS4 + ["sf_tf04_balance_torque_leakfree", "sf_tf06_metric_select_leakfree"]
TARGET = {"sf_tf01_goal_imbalance": ["checks_class_distribution", "uses_class_weight", "uses_resampling", "uses_stratified_split"],
          "sf_tf02_player_rep_norm": ["handles_unit_suffix_in_height"], "sf_tf03_correlation_causation": ["column_x_included", "applies_causal_reasoning"],
          "sf_tf05_credit_card_leakage": ["used_leakage_features"]}
WRONG_COL = {"sf_tf01_goal_imbalance": "assists", "sf_tf02_player_rep_norm": "height_in_cm",
             "sf_tf03_correlation_causation": "assists", "sf_tf05_credit_card_leakage": "reports"}
FIX = {"sf_tf05_credit_card_leakage": {164, 166, 167, 168, 169}, "sf_tf03_correlation_causation": {127, 129, 131, 132, 134, 135, 137}}
CONTRA3 = {130, 136}; WRONG_IDS = set(range(708, 720))
INSTR = r"task (says|asks|states|requires|instructs)|instruction|use all|all available|contradict"
T14 = [json.loads(l) for l in open(REPO / "quick_runs/results/P14/guide_trace.jsonl")]
T13 = [json.loads(l) for l in open(REPO / "quick_runs/results/P13/guide_trace.jsonl")]
print(Counter(t["arm"] for t in T14))""",
    """# Stage 1: fix lessons and wrong lessons, per guide arm (runs out of 5 or 20). P13 G3 / G3-p for reference.
def stage1(trs, arm):
    row = {"arm": arm}
    for t, ids in FIX.items():
        rs = [x for x in trs if x["arm"] == arm.split(" ")[0] and x["task"] == t]
        if not rs: continue
        row[t[:7] + " fix seen"] = sum(bool(set(x["seen"]) & ids) for x in rs)
        row[t[:7] + " fix shown"] = sum(bool(set(x["shown"]) & ids) for x in rs)
        row[t[:7] + " n"] = len(rs)
    if t := [x for x in trs if x["task"] == "sf_tf03_correlation_causation" and x["arm"] == arm.split(" ")[0]]:
        row["tf03 contra shown"] = sum(bool(set(x["shown"]) & CONTRA3) for x in t)
    return row
def wrong(trs, arm):
    rs = [x for x in trs if x["arm"] == arm]
    return {"arm": arm, "runs": len(rs), "wrong seen": sum(bool(set(x["seen"]) & WRONG_IDS) for x in rs),
            "wrong accepted": sum(bool(set(x["accepted"]) & WRONG_IDS) for x in rs), "wrong shown": sum(bool(set(x["shown"]) & WRONG_IDS) for x in rs),
            **{f"{t[:7]} shown": sum(bool(set(x["shown"]) & WRONG_IDS) for x in rs if x["task"] == t) for t in TASKS4}}
s1 = pd.DataFrame([stage1(T13, "G3"), stage1(T14, "G3o"), stage1(T14, "G3n")]).set_index("arm")
wr = pd.DataFrame([wrong(T13, "G3-p"), wrong(T14, "G3o-p"), wrong(T14, "G3n-p")]).set_index("arm")
print(s1.to_string()); wr""",
    """# Reasons: for every judgement on a fix lesson (TF03/TF05) or a wrong lesson, is it rejected, and does the reason cite the task's instructions?
rows = []
for t in T14:
    for r in t["rounds"]:
        for j in r.get("judgements") or []:
            try: i = int(j.get("id"))
            except (TypeError, ValueError): continue
            kind = "fix" if i in FIX.get(t["task"], set()) else "wrong" if i in WRONG_IDS else None
            if kind is None or i not in r["candidates"]: continue
            reason = str(j.get("reason", ""))
            rows.append({"arm": t["arm"], "task": t["task"][:7], "run": t["run"], "id": i, "kind": kind, "useful": j.get("useful") is True,
                         "instr": bool(re.search(INSTR, reason, re.I)), "reason": reason})
jr = pd.DataFrame(rows)
rej = jr[~jr.useful]
print("rejected judgements citing instructions (share):")
print(rej.groupby(["arm", "kind"]).instr.agg(["sum", "count", "mean"]).round(2).to_string())
print("\\naccepted judgements:"); print(jr[jr.useful].groupby(["arm", "kind", "task"]).size().to_string())
jr.groupby(["arm", "kind", "useful"]).size().unstack()""",
    """# Examples of reasons (first rejected fix and wrong judgement per arm and task)
ex = jr[~jr.useful].groupby(["arm", "kind", "task"]).head(1)[["arm", "kind", "task", "id", "reason"]]
for _, r in ex.iterrows(): print(f"{r.arm} {r.kind} {r.task} [{r.id}]: {r.reason[:300]}")""",
    """# Stage 2: agent runs. Targeted checks on the 4 failing tasks (passes out of 5), P14 G3n next to P12/P13 arms
def load(arm):
    recs = [json.loads(l) for l in open(RUNS / arm / "records.jsonl")]
    recs = [r for r in recs if r["task_id"] in SF6]
    for r in recs:
        ev = [json.loads(l) for l in open(RUNS / arm / r["task_id"] / r["run"] / "events.jsonl")]
        calls = [e for e in ev if e["event"] == "llm_call" and e.get("usage")]
        r["sum_tokens"] = sum(c["usage"]["prompt_tokens"] + c["usage"]["completion_tokens"] for c in calls)
    return recs
ARMS = ["P12_A0", "P12_C1", "P13_E", "P13_G3", "P14_G3n", "P13_Ep", "P13_G3p", "P14_G3np"]
NAME = dict(zip(ARMS, ["A0 none", "C1 own", "E emb", "G3 (P13)", "G3n (P14)", "E-p", "G3-p (P13)", "G3n-p (P14)"]))
recs = {a: load(a) for a in ARMS}
print({a: (len(v), dict(Counter(r["stop"] for r in v))) for a, v in recs.items()})
rows = []
for t, checks in TARGET.items():
    for c in checks:
        rows.append({"task": t[:7], "check": c, **{NAME[a]: sum(r["checks"].get(c) is True for r in recs[a] if r["task_id"] == t) for a in ARMS}})
targeted = pd.DataFrame(rows)
pooled = targeted[[NAME[a] for a in ARMS]].sum() / 40
print("pooled:", pooled.round(3).to_dict()); targeted""",
    """# TF05 leaking columns kept (of 4) per run
def leak_kept(a, r):
    j = json.load(open(RUNS / a / r["task_id"] / r["run"] / "judge.json"))
    return len(Counter(tuple(sorted(d.get("leakage_features_used") or [])) for d in j["draws"]).most_common(1)[0][0])
tf05_kept = {NAME[a]: [leak_kept(a, r) for r in sorted(recs[a], key=lambda r: r["run"]) if r["task_id"] == "sf_tf05_credit_card_leakage"] for a in ARMS}
tf05_kept""",
    """# Trust funnel for G3n-p: wrong lesson shown -> named column left out of the final model (executed variables; hand-check)
T14i = {(t["arm"], t["task"], t["run"]): t for t in T14}
def uses(r, col):
    v = r["executed"].get("variables") or []
    return any(col == x or x.startswith(col + "_") for x in v)
fun = pd.DataFrame([{"task": r["task_id"][:7], "run": r["run"], "shown": bool(set(T14i[("G3n-p", r["task_id"], r["run"])]["shown"]) & WRONG_IDS),
                     "col_dropped": not uses(r, WRONG_COL[r["task_id"]]), "variables": r["executed"].get("variables")} for r in recs["P14_G3np"]])
print(fun.groupby("shown").col_dropped.agg(["sum", "count"])); fun.groupby("task")[["shown", "col_dropped"]].sum()""",
    """# Harm and cost
sc = pd.DataFrame([{"arm": NAME[a], "task": r["task_id"], "score": r["score"], "tokens": r["sum_tokens"]} for a in ARMS for r in recs[a]])
scores = sc.pivot_table(index="task", columns="arm", values="score", aggfunc="mean").round(4)
gcost = {}
for arm, trs in [("G3 (P13)", [t for t in T13 if t["arm"] == "G3"]), ("G3n (P14)", [t for t in T14 if t["arm"] == "G3n"])]:
    gcost[arm] = statistics.mean(sum(x["usage"]["prompt_tokens"] + x["usage"]["completion_tokens"] for x in t["rounds"] if x.get("usage")) for t in trs)
print("guide tokens per run:", {k: round(v) for k, v in gcost.items()}); scores""",
    """summary = {"stage1_fix": s1.reset_index().to_dict("records"), "stage1_wrong": wr.reset_index().to_dict("records"),
  "reasons_rejected_instr": rej.groupby(["arm", "kind"]).instr.agg(["sum", "count"]).reset_index().to_dict("records"),
  "judgements": jr.groupby(["arm", "kind", "useful"]).size().reset_index(name="n").to_dict("records"),
  "runs": {a: len(v) for a, v in recs.items()}, "stops": {a: dict(Counter(r["stop"] for r in v)) for a, v in recs.items()},
  "targeted": targeted.to_dict("records"), "targeted_pooled": pooled.round(3).to_dict(), "tf05_leak_cols_kept": tf05_kept,
  "trust_g3np": fun.groupby("shown").col_dropped.agg(["sum", "count"]).reset_index().to_dict("records"),
  "score_mean": scores.to_dict("index"), "guide_tokens_per_run": gcost}
json.dump(summary, open(REPO / "quick_runs/results/P14/summary.json", "w"), indent=1, default=str)
jr.to_csv(REPO / "quick_runs/results/P14/judgements.csv", index=False); fun.to_csv(REPO / "quick_runs/results/P14/trust_funnel.csv", index=False)
"done" """,
]


P15 = [
    "# P15 cross-model guide: Qwen3.8-27B as the guide, Gemma 4 31B as the agent\\n"
    "P13 guide prompt, retry budget 3, reasons saved. Q3 clean store (6 tasks), Q3-p poisoned (4 failing tasks). "
    "Compared with P14 G3o / G3o-p (Gemma guide, same prompt) and P13 agent arms. Predictions: `quick_runs/predictions/P15.md`.",
    P12_SETUP, COMMON,
    P14[3],
    """T15 = [json.loads(l) for l in open(REPO / "quick_runs/results/P15/guide_trace.jsonl")]
print(Counter(t["arm"] for t in T15))
TALL = T14 + T15""",
    P14[4].replace('s1 = pd.DataFrame([stage1(T13, "G3"), stage1(T14, "G3o"), stage1(T14, "G3n")]).set_index("arm")',
                   's1 = pd.DataFrame([stage1(T13, "G3"), stage1(T14, "G3o"), stage1(T15, "Q3")]).set_index("arm")')
          .replace('wr = pd.DataFrame([wrong(T13, "G3-p"), wrong(T14, "G3o-p"), wrong(T14, "G3n-p")]).set_index("arm")',
                   'wr = pd.DataFrame([wrong(T13, "G3-p"), wrong(T14, "G3o-p"), wrong(T15, "Q3-p")]).set_index("arm")'),
    P14[5].replace("for t in T14:", "for t in TALL:\n    if t['arm'] not in ('G3o', 'G3o-p', 'Q3', 'Q3-p'): continue"),
    P14[6],
    P14[7].replace('ARMS = ["P12_A0", "P12_C1", "P13_E", "P13_G3", "P14_G3n", "P13_Ep", "P13_G3p", "P14_G3np"]',
                   'ARMS = ["P12_A0", "P12_C1", "P13_E", "P13_G3", "P15_Q3", "P13_Ep", "P13_G3p", "P15_Q3p"]')
          .replace('["A0 none", "C1 own", "E emb", "G3 (P13)", "G3n (P14)", "E-p", "G3-p (P13)", "G3n-p (P14)"]',
                   '["A0 none", "C1 own", "E emb", "G3 gemma", "Q3 qwen", "E-p", "G3-p gemma", "Q3-p qwen"]'),
    P14[8],
    P14[9].replace('T14i = {(t["arm"], t["task"], t["run"]): t for t in T14}', 'T14i = {(t["arm"], t["task"], t["run"]): t for t in T15}')
          .replace('T14i[("G3n-p", r["task_id"], r["run"])]', 'T14i[("Q3-p", r["task_id"], r["run"])]').replace('recs["P14_G3np"]', 'recs["P15_Q3p"]'),
    P14[10].replace('("G3n (P14)", [t for t in T14 if t["arm"] == "G3n"])', '("Q3 qwen", [t for t in T15 if t["arm"] == "Q3"])'),
    P14[11].replace("quick_runs/results/P14/", "quick_runs/results/P15/").replace('"trust_g3np"', '"trust_q3p"'),
]


P16 = [
    "# P16 mechanism controls\n"
    "One change each from G3 (P13 prompt, retry budget 3). G3t: the guide sees a goal-only task (no feature or "
    "preprocessing instructions). G3d: the guide also gets a fixed data profile. '-p': P13 wrong lessons (drop a useful "
    "column, against the task). '-pc': wrong lessons that agree with the task's wording. E-pc: embedding top-3 on the "
    "'pc' store. Predictions: `quick_runs/predictions/P16.md`.",
    P12_SETUP, COMMON,
    P14[3], P14[4],
    """T16 = [json.loads(l) for l in open(REPO / "quick_runs/results/P16/guide_trace.jsonl")]
for t in T16:  # embedding-only arms have no guide: seen = accepted = shown
    t.setdefault("seen", t["shown"]); t.setdefault("accepted", t["shown"])
print(Counter(t["arm"] for t in T16))
# Fix lessons and the contradicting TF03 lesson (runs out of 5); P13 G3 and P14 G3o (same prompt) for reference
s1 = pd.DataFrame([stage1(T13, "G3"), stage1(T14, "G3o"), stage1(T16, "G3t"), stage1(T16, "G3d")]).set_index("arm")
# Wrong lessons shown (runs out of 20). -p: P13 wrong lessons; -pc: instruction-agreeing wrong lessons
wr = pd.DataFrame([wrong(T13, "G3-p"), wrong(T14, "G3o-p"), wrong(T16, "G3t-p"), wrong(T16, "G3d-p"),
                   wrong(T16, "E-pc"), wrong(T16, "G3-pc"), wrong(T16, "G3d-pc")]).set_index("arm")
print(s1.to_string()); wr""",
    """# Reasons for rejecting fix lessons: share citing the task's instructions (pre-set regex + contradict-only rule from REVIEW_CHECKS)
NOC = r"task (says|asks|states|requires|instructs)|instruction|use all|all available"
TASKWORD = r"prompt|task|requirement|constraint"
def cites_instr(reason):
    if not re.search(INSTR, reason, re.I): return False
    return bool(re.search(NOC, reason, re.I) or re.search(TASKWORD, reason, re.I))
rows = []
for t in T14 + T16:
    if t["arm"] not in ("G3o", "G3o-p", "G3t", "G3t-p", "G3d", "G3d-p", "G3-pc", "G3d-pc"): continue
    for r in t["rounds"]:
        for j in r.get("judgements") or []:
            try: i = int(j.get("id"))
            except (TypeError, ValueError): continue
            kind = "fix" if i in FIX.get(t["task"], set()) else "wrong" if i in WRONG_IDS else None
            if kind is None or i not in r["candidates"]: continue
            reason = str(j.get("reason", ""))
            rows.append({"arm": t["arm"], "task": t["task"][:7], "run": t["run"], "id": i, "kind": kind,
                         "useful": j.get("useful") is True, "instr": cites_instr(reason), "reason": reason})
jr = pd.DataFrame(rows)
rej = jr[~jr.useful]
print(rej.groupby(["arm", "kind"]).instr.agg(["sum", "count", "mean"]).round(2).to_string())
ex = rej.groupby(["arm", "kind", "task"]).head(1)
for _, r in ex.iterrows(): print(f"{r.arm} {r.kind} {r.task} [{r.id}]: {r.reason[:250]}")
print("accepted wrong lessons, examples:")
for _, r in jr[jr.useful & (jr.kind == "wrong")].groupby(["arm", "task"]).head(1).iterrows(): print(f"{r.arm} {r.task} [{r.id}]: {r.reason[:250]}")""",
    P14[7].replace('ARMS = ["P12_A0", "P12_C1", "P13_E", "P13_G3", "P14_G3n", "P13_Ep", "P13_G3p", "P14_G3np"]',
                   'ARMS = ["P12_A0", "P12_C1", "P13_E", "P13_G3", "P16_G3t", "P16_G3d", "P13_Ep", "P13_G3p", "P16_G3tp", "P16_G3dp", "P16_Epc", "P16_G3pc", "P16_G3dpc"]')
          .replace('["A0 none", "C1 own", "E emb", "G3 (P13)", "G3n (P14)", "E-p", "G3-p (P13)", "G3n-p (P14)"]',
                   '["A0 none", "C1 own", "E emb", "G3", "G3t blind", "G3d profile", "E-p", "G3-p", "G3t-p", "G3d-p", "E-pc", "G3-pc", "G3d-pc"]'),
    P14[8],
    """# Trust funnel. -p arms: named column left out of the final model (executed variables). -pc arms: the lesson's advice
# followed = the targeted check fails (checks are stored as pass/fail: TF01 no class weights; TF02 unit not stripped;
# TF03 column_x used; TF05 a leaking column used)
TR = {(t["arm"], t["task"], t["run"]): t for t in T13 + T16}
def uses(r, col):
    v = r["executed"].get("variables") or []
    return any(col == x or x.startswith(col + "_") for x in v)
FOLLOW_C = {"sf_tf01_goal_imbalance": ("uses_class_weight", False), "sf_tf02_player_rep_norm": ("handles_unit_suffix_in_height", False),
            "sf_tf03_correlation_causation": ("column_x_included", False), "sf_tf05_credit_card_leakage": ("used_leakage_features", False)}
GARM = {"P13_G3p": "G3-p", "P16_G3tp": "G3t-p", "P16_G3dp": "G3d-p", "P16_Epc": "E-pc", "P16_G3pc": "G3-pc", "P16_G3dpc": "G3d-pc", "P13_Ep": "E-p"}
rows = []
for a, g in GARM.items():
    for r in recs[a]:
        t = TR.get((g, r["task_id"], r["run"]))
        if t is None: continue
        shown = bool(set(t["shown"]) & WRONG_IDS)
        if g.endswith("-pc"):
            key, val = FOLLOW_C[r["task_id"]]; followed = r["checks"].get(key) is val
        else:
            followed = not uses(r, WRONG_COL[r["task_id"]])
        rows.append({"arm": g, "task": r["task_id"][:7], "run": r["run"], "shown": shown, "followed": followed})
fun = pd.DataFrame(rows)
print(fun.groupby(["arm", "shown"]).followed.agg(["sum", "count"]).to_string())
fun.groupby(["arm", "task"])[["shown", "followed"]].sum()""",
    """# Harm and cost
sc = pd.DataFrame([{"arm": NAME[a], "task": r["task_id"], "score": r["score"]} for a in ARMS for r in recs[a]])
scores = sc.pivot_table(index="task", columns="arm", values="score", aggfunc="mean").round(4)
gcost = {}
for arm, trs in [("G3 (P13)", [t for t in T13 if t["arm"] == "G3"]), ("G3t", [t for t in T16 if t["arm"] == "G3t"]), ("G3d", [t for t in T16 if t["arm"] == "G3d"])]:
    gcost[arm] = statistics.mean(sum(x["usage"]["prompt_tokens"] + x["usage"]["completion_tokens"] for x in t["rounds"] if x.get("usage")) for t in trs)
print("guide tokens per run:", {k: round(v) for k, v in gcost.items()}); scores""",
    """# Exact tests (two-sided Fisher), runs or check-runs as the unit
from scipy.stats import fisher_exact
passed = (pooled * 40).round().astype(int)
def fp(a, b, n): return round(fisher_exact([[a, n - a], [b, n - b]])[1], 4)
fisher = {"passed G3 vs G3d profile": fp(passed["G3"], passed["G3d profile"], 40),
          "passed G3-p vs G3d-p": fp(passed["G3-p"], passed["G3d-p"], 40),
          "passed G3-pc vs G3d-pc": fp(passed["G3-pc"], passed["G3d-pc"], 40),
          "passed E emb vs E-pc": fp(passed["E emb"], passed["E-pc"], 40),
          "wrong shown G3-p vs G3t-p": fp(int(wr.loc["G3-p", "wrong shown"]), int(wr.loc["G3t-p", "wrong shown"]), 20),
          "wrong shown G3-p vs G3-pc": fp(int(wr.loc["G3-p", "wrong shown"]), int(wr.loc["G3-pc", "wrong shown"]), 20),
          "wrong shown G3-pc vs G3d-pc": fp(int(wr.loc["G3-pc", "wrong shown"]), int(wr.loc["G3d-pc", "wrong shown"]), 20),
          "fix shown TF03+TF05 G3 vs G3t (of 10)": fp(1, 10, 10)}
fisher""",
    """summary = {"stage1_fix": s1.reset_index().to_dict("records"), "stage1_wrong": wr.reset_index().to_dict("records"),
  "reasons_rejected_instr": rej.groupby(["arm", "kind"]).instr.agg(["sum", "count"]).reset_index().to_dict("records"),
  "runs": {a: len(v) for a, v in recs.items()}, "stops": {a: dict(Counter(r["stop"] for r in v)) for a, v in recs.items()},
  "targeted": targeted.to_dict("records"), "targeted_pooled": pooled.round(3).to_dict(), "tf05_leak_cols_kept": tf05_kept,
  "fisher_p": fisher, "trust": fun.groupby(["arm", "shown"]).followed.agg(["sum", "count"]).reset_index().to_dict("records"),
  "trust_by_task": fun.groupby(["arm", "task"])[["shown", "followed"]].sum().reset_index().to_dict("records"),
  "score_mean": scores.to_dict("index"), "guide_tokens_per_run": gcost}
json.dump(summary, open(REPO / "quick_runs/results/P16/summary.json", "w"), indent=1, default=str)
jr.to_csv(REPO / "quick_runs/results/P16/judgements.csv", index=False); fun.to_csv(REPO / "quick_runs/results/P16/trust_funnel.csv", index=False)
"done" """,
]


P17 = [
    "# P17 store written without a checker\n"
    "Silent Failures lessons rewritten from the same 40 runs without the evaluator's outcome ('nf'), labelled before "
    "any guide or agent run. Arms: C1-nf (all own nf lessons), E-nf (embedding top-3), G3-nf (P13 guide), G3d-nf "
    "(P16 data-profile guide). Predictions: `quick_runs/predictions/P17.md`.",
    P12_SETUP, COMMON,
    P14[3], P14[4],
    """import sys, yaml
sys.path.insert(0, str(REPO / "quick_runs" / "p13")); sys.path.insert(0, str(REPO / "quick_runs" / "p12"))
import guide as GD
NF = GD.load_store("nf")
LAB = yaml.safe_load(open(REPO / "quick_runs/p17/labels_nf.yaml"))
def ids(label):
    return {i for i, L in enumerate(NF) if L.get("nf") and L["source_task"] in LAB
            and f"{L['source_run']}/{L['k']}" in (LAB[L["source_task"]][label] or [])}
REINF, FIXNF = ids("reinforcing"), ids("fix")
nf_own = Counter(L["source_task"] for L in NF if L.get("nf"))
print("nf lessons:", sum(nf_own.values()), dict(nf_own)); print("reinforcing:", len(REINF), "fix:", len(FIXNF))
T17 = [json.loads(l) for l in open(REPO / "quick_runs/results/P17/guide_trace.jsonl")]
for t in T17: t.setdefault("seen", t["shown"]); t.setdefault("accepted", t["shown"])
print(Counter(t["arm"] for t in T17))
rows = []
for t in T17:
    if t["task"] not in TASKS4: continue
    rows.append({"arm": t["arm"], "task": t["task"][:7], "run": t["run"], "n_shown": len(t["shown"]),
                 "reinf_seen": bool(set(t["seen"]) & REINF), "reinf_shown": bool(set(t["shown"]) & REINF),
                 "n_reinf_shown": len(set(t["shown"]) & REINF), "fix_shown": bool(set(t["shown"]) & FIXNF)})
ex = pd.DataFrame(rows)
print(ex.groupby("arm")[["reinf_seen", "reinf_shown", "fix_shown", "n_shown"]].sum().to_string())
ex.groupby(["arm", "task"])[["reinf_seen", "reinf_shown"]].sum().unstack()""",
    """# Guide reasons on reinforcing lessons (accepted and rejected)
rows = []
for t in T17:
    for r in t.get("rounds", []):
        for j in r.get("judgements") or []:
            try: i = int(j.get("id"))
            except (TypeError, ValueError): continue
            if i in REINF and i in r["candidates"]:
                rows.append({"arm": t["arm"], "task": t["task"][:7], "useful": j.get("useful") is True, "reason": str(j.get("reason", ""))})
jr = pd.DataFrame(rows)
print(jr.groupby(["arm", "useful"]).size().to_string())
for _, r in jr.groupby(["arm", "task", "useful"]).head(1).iterrows(): print(f"{r.arm} {r.task} useful={r.useful}: {r.reason[:240]}")""",
    P14[7].replace('ARMS = ["P12_A0", "P12_C1", "P13_E", "P13_G3", "P14_G3n", "P13_Ep", "P13_G3p", "P14_G3np"]',
                   'ARMS = ["P12_A0", "P12_C1", "P13_E", "P13_G3", "P16_G3d", "P17_C1nf", "P17_Enf", "P17_G3nf", "P17_G3dnf"]')
          .replace('["A0 none", "C1 own", "E emb", "G3 (P13)", "G3n (P14)", "E-p", "G3-p (P13)", "G3n-p (P14)"]',
                   '["A0 none", "C1 own", "E emb", "G3", "G3d", "C1-nf", "E-nf", "G3-nf", "G3d-nf"]'),
    """# Harm, cost and exact tests
from scipy.stats import fisher_exact
sc = pd.DataFrame([{"arm": NAME[a], "task": r["task_id"], "score": r["score"]} for a in ARMS for r in recs[a]])
scores = sc.pivot_table(index="task", columns="arm", values="score", aggfunc="mean").round(4)
passed = (pooled * 40).round().astype(int)
fp = lambda a, b: round(fisher_exact([[a, 40 - a], [b, 40 - b]])[1], 4)
fisher = {"C1 vs C1-nf": fp(passed["C1 own"], passed["C1-nf"]), "E vs E-nf": fp(passed["E emb"], passed["E-nf"]),
          "G3 vs G3-nf": fp(passed["G3"], passed["G3-nf"]), "A0 vs C1-nf": fp(passed["A0 none"], passed["C1-nf"])}
print(fisher); scores""",
    """summary = {"nf_lessons": dict(nf_own), "n_reinforcing": len(REINF), "n_fix": len(FIXNF),
  "exposure": ex.groupby("arm")[["reinf_seen", "reinf_shown", "fix_shown"]].sum().reset_index().to_dict("records"),
  "exposure_by_task": ex.groupby(["arm", "task"])[["reinf_seen", "reinf_shown"]].sum().reset_index().to_dict("records"),
  "reinf_judgements": jr.groupby(["arm", "useful"]).size().reset_index(name="n").to_dict("records") if len(jr) else [],
  "runs": {a: len(v) for a, v in recs.items()}, "stops": {a: dict(Counter(r["stop"] for r in v)) for a, v in recs.items()},
  "targeted": targeted.to_dict("records"), "targeted_pooled": pooled.round(3).to_dict(), "fisher_p": fisher,
  "score_mean": scores.to_dict("index")}
json.dump(summary, open(REPO / "quick_runs/results/P17/summary.json", "w"), indent=1, default=str)
"done" """,
]


REVIEW_CHECKS = [
    "# Review checks (2026-11-17): regex sensitivity of the instruction-reason share, and exact tests\n"
    "Asked for by the peer review (Open ScholarPeer) of the paper. Reads saved P14/P15 judgements and summaries; no model runs.\n"
    "1. The pre-set regex counts a bare `contradict`. Which rejections of fix lessons match only through it, and do those cite the "
    "task or only another lesson? Rule: a contradict-only reason counts as citing the instructions only if it also names the "
    "prompt, the task, a requirement or a constraint. 2. Two-sided Fisher exact tests for the paper's headline comparisons.",
    """import csv, re, json
from pathlib import Path
from scipy.stats import fisher_exact
REPO = Path.cwd()
while not (REPO / "quick_runs").exists(): REPO = REPO.parent
PRE = re.compile(r"task (says|asks|states|requires|instructs)|instruction|use all|all available|contradict", re.I)
NOC = re.compile(r"task (says|asks|states|requires|instructs)|instruction|use all|all available", re.I)
TASKWORD = re.compile(r"prompt|task|requirement|constraint", re.I)
out = {}
for f, arm in [("P14", "G3o"), ("P14", "G3n"), ("P15", "Q3")]:
    R = [r for r in csv.DictReader(open(REPO / f"quick_runs/results/{f}/judgements.csv"))
         if r["arm"] == arm and r["kind"] == "fix" and r["useful"].lower() in ("false", "0")]
    pre = [r for r in R if PRE.search(r["reason"])]
    only_c = [r for r in pre if not NOC.search(r["reason"])]
    lesson_only = [r for r in only_c if not TASKWORD.search(r["reason"])]
    n_instr = len(pre) - len(lesson_only)
    out[arm] = {"rejected_fix": len(R), "pre_set": len(pre), "contradict_only": len(only_c),
                "contradict_only_lesson_not_task": len(lesson_only), "citing_instructions": n_instr,
                "share": round(n_instr / len(R), 2)}
    print(arm, out[arm])
    for r in only_c: print("   ", "LESSON-ONLY" if r in lesson_only else "task      ", r["reason"][:160])""",
    """# Exact tests (two-sided Fisher), runs as the unit
tests = {"wrong shown: embedding 10/20 vs guide 5/20": (10, 5, 20), "wrong shown: Gemma guide 5/20 vs Qwen guide 1/20": (5, 1, 20),
         "passed: guide R=3 20/40 vs Qwen guide 14/40": (20, 14, 40), "passed: embedding 19/40 vs guide R=3 20/40": (19, 20, 40),
         "passed: embedding 19/40 vs all own lessons 30/40": (19, 30, 40), "passed: no memory 3/40 vs embedding 19/40": (3, 19, 40)}
pv = {k: round(fisher_exact([[a, n - a], [b, n - b]])[1], 4) for k, (a, b, n) in tests.items()}
for k, v in pv.items(): print(f"{v:.4f}  {k}")
json.dump({"instr_share": out, "fisher_p": pv}, open(REPO / "quick_runs/results/P14/review_checks.json", "w"), indent=1)""",
]


REVIEW2_JUDGE = [
    "# Review round 2 check: does the Gemma judge agree with simple patterns in the executed code?\n"
    "The review asked whether the judge (Gemma 4 31B, the agent's model) was validated. For 5 of the 8 targeted checks a "
    "pattern in the successfully executed code cells gives an independent verdict; it abstains when the pattern is "
    "ambiguous. Agreement is counted on runs where both give a verdict. No model calls.",
    """import json, os, re
from collections import Counter
from pathlib import Path
import pandas as pd
REPO = Path.cwd()
while not (REPO / "quick_runs").exists(): REPO = REPO.parent
RUNS = Path(os.environ.get("REVERIES_RUNS", "/data/anon/reveries_runs"))
PROBES = sorted(p.name for p in RUNS.iterdir() if p.is_dir() and re.match(r"P1[2-7]_", p.name) and (p / "records.jsonl").exists())
LEAK = ["expenditure", "share", "active", "majorcards"]
DROP = r"(drop|exclude|remove|!=|not in|\\[\\s*~)"
def code_of(probe, r):
    ev = [json.loads(l) for l in open(RUNS / probe / r["task_id"] / r["run"] / "events.jsonl")]
    return "\\n".join(e["code"] for e in ev if e["event"] == "code_executed" and e.get("ok"))
def rule(check, code):
    if check == "uses_class_weight":
        return bool(re.search(r"class_weight\\s*=\\s*['\\"{]|scale_pos_weight\\s*=", code)
                    or (re.search(r"class_weight\\s*=\\s*(?!None)\\w", code) and "balanced" in code))  # tuned via a variable
    if check == "uses_stratified_split":
        return bool(re.search(r"stratify\\s*=\\s*[^N\\s)]|Stratified(K|Shuffle)", code))
    if check == "handles_unit_suffix_in_height":
        return bool(re.search(r"(replace|extract|rstrip|strip|sub)\\(.{0,40}cm|\\\\d\\+|replace\\(\\s*r?['\\"]\\[\\^?[a-zA-Z0-9.\\\\-]*\\]", code)) if "height" in code else None
    if check == "column_x_included":  # pass = left out
        if "column_x" not in code: return False  # never named: kept by a dtype selection
        return True if re.search(DROP + r".{0,80}column_x|column_x.{0,40}" + DROP, code) else None
    if check == "used_leakage_features":  # pass = none used
        named = [c for c in LEAK if re.search(r"['\\"]" + c + r"['\\"]", code)]
        if not named: return False
        return True if all(re.search(DROP + r".{0,120}['\\"]" + c + r"['\\"]", code, re.S) for c in LEAK) else None
CHECKS = {"sf_tf01_goal_imbalance": ["uses_class_weight", "uses_stratified_split"], "sf_tf02_player_rep_norm": ["handles_unit_suffix_in_height"],
          "sf_tf03_correlation_causation": ["column_x_included"], "sf_tf05_credit_card_leakage": ["used_leakage_features"]}
rows = []
for probe in PROBES:
    for l in open(RUNS / probe / "records.jsonl"):
        r = json.loads(l)
        if r["task_id"] not in CHECKS: continue
        code = code_of(probe, r)
        for c in CHECKS[r["task_id"]]:
            rows.append({"probe": probe, "task": r["task_id"][:7], "check": c, "judge": r["checks"].get(c), "rule": rule(c, code)})
ag = pd.DataFrame(rows)
both = ag[ag.judge.notna() & ag.rule.notna()]
tab = both.assign(agree=both.judge == both.rule).groupby("check").agree.agg(["sum", "count", "mean"]).round(3)
print("probes:", len(PROBES), "| check-runs:", len(ag), "| both decided:", len(both), "| overall agreement:", round((both.judge == both.rule).mean(), 3))
tab""",
    """# Disagreements, by check and direction
dis = both[both.judge != both.rule]
print(dis.groupby(["check", "judge", "rule"]).size().to_string())
json.dump({"n_check_runs": len(ag), "n_both": len(both), "agreement": round(float((both.judge == both.rule).mean()), 4),
           "by_check": tab.reset_index().to_dict("records"), "probes": PROBES},
          open(REPO / "quick_runs/results/P14/review2_judge_check.json", "w"), indent=1)
dis.head(20)""",
]


def build(name: str, cells: list[str]) -> Path:
    nb = nbformat.v4.new_notebook()
    # The first cell is the markdown title; every other cell is code (code cells may start with a # comment).
    nb.cells = [nbformat.v4.new_markdown_cell(c) if i == 0 else nbformat.v4.new_code_cell(c) for i, c in enumerate(cells)]
    NB_DIR.mkdir(parents=True, exist_ok=True)
    path = NB_DIR / f"quick_{name}.ipynb"
    NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(NB_DIR)}}).execute()
    nbformat.write(nb, path)
    return path



P18 = [
    "# P18 review round 3 controls\n"
    "R3 / R3-p: 3 random lessons from the P13 G3 / G3-p guide's candidates (no judgement). G3s: the guide's task text "
    "without the feature sentence only; G3s-pl: placebo, an unrelated sentence removed (TF03, TF05). G3d-tr / G3d-tr-pc: "
    "the data profile without test-set numbers. G3h / G3h-pc: the guide sees the suite's first hint-level wording (guide "
    "only). X3 / X3-pc: embedding top-20 re-scored by a cross-encoder, top 3. Predictions: `quick_runs/predictions/P18.md`.",
    P12_SETUP, COMMON,
    P14[3], P14[4],
    """T16 = [json.loads(l) for l in open(REPO / "quick_runs/results/P16/guide_trace.jsonl")]
T18 = [json.loads(l) for l in open(REPO / "quick_runs/results/P18/guide_trace.jsonl")]
for t in T13 + T16 + T18:  # arms without a guide: seen = accepted = shown
    t.setdefault("seen", t["shown"]); t.setdefault("accepted", t["shown"])
print(Counter(t["arm"] for t in T18))
s1 = pd.DataFrame([stage1(T13, "G3"), stage1(T16, "G3t"), stage1(T16, "G3d"), stage1(T18, "R3"), stage1(T18, "G3s"),
                   stage1(T18, "G3s-pl"), stage1(T18, "G3d-tr"), stage1(T18, "G3h"), stage1(T13, "E"), stage1(T18, "X3")]).set_index("arm")
wr = pd.DataFrame([wrong(T13, "E-p"), wrong(T13, "G3-p"), wrong(T18, "R3-p"), wrong(T16, "E-pc"), wrong(T16, "G3-pc"),
                   wrong(T16, "G3d-pc"), wrong(T18, "G3d-tr-pc"), wrong(T18, "G3h-pc"), wrong(T18, "X3-pc")]).set_index("arm")
print(s1.to_string()); wr""",
    """# Reasons for rejecting fix lessons and accepting/rejecting wrong lessons (pre-set regex + contradict-only rule)
NOC = r"task (says|asks|states|requires|instructs)|instruction|use all|all available"
TASKWORD = r"prompt|task|requirement|constraint"
def cites_instr(reason):
    if not re.search(INSTR, reason, re.I): return False
    return bool(re.search(NOC, reason, re.I) or re.search(TASKWORD, reason, re.I))
rows = []
for t in T18:
    for r in t.get("rounds", []):
        for j in r.get("judgements") or []:
            try: i = int(j.get("id"))
            except (TypeError, ValueError): continue
            kind = "fix" if i in FIX.get(t["task"], set()) else "wrong" if i in WRONG_IDS else "contra" if (i in CONTRA3 and t["task"] == "sf_tf03_correlation_causation") else None
            if kind is None or i not in r["candidates"]: continue
            reason = str(j.get("reason", ""))
            rows.append({"arm": t["arm"], "task": t["task"][:7], "run": t["run"], "id": i, "kind": kind,
                         "useful": j.get("useful") is True, "instr": cites_instr(reason), "reason": reason})
jr = pd.DataFrame(rows)
rej = jr[~jr.useful]
print(rej.groupby(["arm", "kind"]).instr.agg(["sum", "count", "mean"]).round(2).to_string())
for _, r in jr.groupby(["arm", "kind", "task", "useful"]).head(1).iterrows(): print(f"{r.arm} {r.kind} {r.task} useful={r.useful} [{r.id}]: {r.reason[:240]}")""",
    P14[7].replace('ARMS = ["P12_A0", "P12_C1", "P13_E", "P13_G3", "P14_G3n", "P13_Ep", "P13_G3p", "P14_G3np"]',
                   'ARMS = ["P12_A0", "P12_C1", "P13_E", "P13_G3", "P16_G3t", "P16_G3d", "P18_R3", "P18_G3s", "P18_G3spl", "P18_G3dtr", "P18_X3", "P13_Ep", "P13_G3p", "P18_R3p", "P16_Epc", "P16_G3pc", "P16_G3dpc", "P18_G3dtrpc", "P18_X3pc"]')
          .replace('["A0 none", "C1 own", "E emb", "G3 (P13)", "G3n (P14)", "E-p", "G3-p (P13)", "G3n-p (P14)"]',
                   '["A0 none", "C1 own", "E emb", "G3", "G3t", "G3d", "R3", "G3s", "G3s-pl", "G3d-tr", "X3", "E-p", "G3-p", "R3-p", "E-pc", "G3-pc", "G3d-pc", "G3d-tr-pc", "X3-pc"]'),
    P14[8],
    """# TF03 and TF05 runs (of 5) for the two-task arms: checks passed, and runs that left column_x out / kept no leaking column
tf35 = targeted[targeted.task.isin(["sf_tf03", "sf_tf05"])].set_index(["task", "check"])[["A0 none", "C1 own", "G3", "G3t", "G3d", "G3s", "G3s-pl", "G3d-tr"]]
tf35""",
    """# Trust funnel: wrong lesson shown -> followed. -p: named column left out (executed variables). -pc: targeted check failed.
TR = {(t["arm"], t["task"], t["run"]): t for t in T13 + T16 + T18}
def uses(r, col):
    v = r["executed"].get("variables") or []
    return any(col == x or x.startswith(col + "_") for x in v)
FOLLOW_C = {"sf_tf01_goal_imbalance": ("uses_class_weight", False), "sf_tf02_player_rep_norm": ("handles_unit_suffix_in_height", False),
            "sf_tf03_correlation_causation": ("column_x_included", False), "sf_tf05_credit_card_leakage": ("used_leakage_features", False)}
GARM = {"P13_Ep": "E-p", "P13_G3p": "G3-p", "P18_R3p": "R3-p", "P16_G3dpc": "G3d-pc", "P18_G3dtrpc": "G3d-tr-pc", "P18_X3pc": "X3-pc", "P16_Epc": "E-pc"}
rows = []
for a, g in GARM.items():
    for r in recs[a]:
        t = TR.get((g, r["task_id"], r["run"]))
        if t is None: continue
        shown = bool(set(t["shown"]) & WRONG_IDS)
        if g.endswith("-pc"):
            key, val = FOLLOW_C[r["task_id"]]; followed = r["checks"].get(key) is val
        else:
            followed = not uses(r, WRONG_COL[r["task_id"]])
        rows.append({"arm": g, "task": r["task_id"][:7], "run": r["run"], "shown": shown, "followed": followed})
fun = pd.DataFrame(rows)
print(fun.groupby(["arm", "shown"]).followed.agg(["sum", "count"]).to_string())
fun.groupby(["arm", "task"])[["shown", "followed"]].sum().unstack()""",
    """# Harm, cost and exact tests (two-sided Fisher; runs or check-runs as the unit)
from scipy.stats import fisher_exact
sc = pd.DataFrame([{"arm": NAME[a], "task": r["task_id"], "score": r["score"]} for a in ARMS for r in recs[a]])
scores = sc.pivot_table(index="task", columns="arm", values="score", aggfunc="mean").round(4)
gcost = {}
for arm in ("G3s", "G3s-pl", "G3d-tr", "G3h", "G3h-pc"):
    gcost[arm] = statistics.mean(sum(x["usage"]["prompt_tokens"] + x["usage"]["completion_tokens"] for x in t["rounds"] if x.get("usage")) for t in T18 if t["arm"] == arm)
passed = (pooled * 40).round().astype(int)
def fp(a, b, n): return round(fisher_exact([[a, n - a], [b, n - b]])[1], 4)
fix35 = lambda arm: int(s1.loc[arm, ["sf_tf03 fix shown", "sf_tf05 fix shown"]].sum())
fisher = {"fix shown TF03+TF05 G3 vs G3s (of 10)": fp(fix35("G3"), fix35("G3s"), 10),
          "fix shown TF03+TF05 G3s vs G3s-pl (of 10)": fp(fix35("G3s"), fix35("G3s-pl"), 10),
          "fix shown TF03+TF05 G3 vs G3h (of 10)": fp(fix35("G3"), fix35("G3h"), 10),
          "fix shown TF03+TF05 G3 vs R3 (of 10)": fp(fix35("G3"), fix35("R3"), 10),
          "wrong shown G3-p vs R3-p": fp(int(wr.loc["G3-p", "wrong shown"]), int(wr.loc["R3-p", "wrong shown"]), 20),
          "wrong shown G3-pc vs G3h-pc": fp(int(wr.loc["G3-pc", "wrong shown"]), int(wr.loc["G3h-pc", "wrong shown"]), 20),
          "wrong shown G3d-pc vs G3d-tr-pc": fp(int(wr.loc["G3d-pc", "wrong shown"]), int(wr.loc["G3d-tr-pc", "wrong shown"]), 20),
          "passed G3 vs R3": fp(passed["G3"], passed["R3"], 40), "passed E vs X3": fp(passed["E emb"], passed["X3"], 40),
          "passed G3d vs G3d-tr": fp(passed["G3d"], passed["G3d-tr"], 40), "passed G3-pc vs G3d-tr-pc": fp(passed["G3-pc"], passed["G3d-tr-pc"], 40)}
print("guide tokens per run:", {k: round(v) for k, v in gcost.items()}); print(fisher); scores""",
    """summary = {"stage1_fix": s1.reset_index().to_dict("records"), "stage1_wrong": wr.reset_index().to_dict("records"),
  "reasons_rejected_instr": rej.groupby(["arm", "kind"]).instr.agg(["sum", "count"]).reset_index().to_dict("records"),
  "judgements": jr.groupby(["arm", "kind", "useful"]).size().reset_index(name="n").to_dict("records"),
  "runs": {a: len(v) for a, v in recs.items()}, "stops": {a: dict(Counter(r["stop"] for r in v)) for a, v in recs.items()},
  "targeted": targeted.to_dict("records"), "targeted_pooled": pooled.round(3).to_dict(), "tf05_leak_cols_kept": tf05_kept,
  "fisher_p": fisher, "trust": fun.groupby(["arm", "shown"]).followed.agg(["sum", "count"]).reset_index().to_dict("records"),
  "trust_by_task": fun.groupby(["arm", "task"])[["shown", "followed"]].sum().reset_index().to_dict("records"),
  "score_mean": scores.to_dict("index"), "guide_tokens_per_run": gcost}
json.dump(summary, open(REPO / "quick_runs/results/P18/summary.json", "w"), indent=1, default=str)
jr.to_csv(REPO / "quick_runs/results/P18/judgements.csv", index=False); fun.to_csv(REPO / "quick_runs/results/P18/trust_funnel.csv", index=False)
"done" """,
]


P19 = [
    "# P19 review round 4 controls\n"
    "G3s / G3s-pl: P18's one-sentence cut and placebo, 10 more guide runs each (15 per task in all). G3x / G3x-p / G3x-pc: "
    "escalating guide that flags notes whose use would depart from the instructions (guide only). G3n-nf: the open-prompt "
    "guide on the no-checker store. nfd: lessons written with the data profile and without the outcome; C1-nfd / E-nfd. "
    "A0h: no memory, hint-level task. Predictions: `quick_runs/predictions/P19.md`.",
    P12_SETUP, COMMON,
    P14[3], P14[4],
    """import sys, yaml
sys.path.insert(0, str(REPO / "quick_runs" / "p13")); sys.path.insert(0, str(REPO / "quick_runs" / "p12"))
import guide as GD
from scipy.stats import fisher_exact
fp = lambda a, na, b, nb: round(fisher_exact([[a, na - a], [b, nb - b]])[1], 4)
T18 = [json.loads(l) for l in open(REPO / "quick_runs/results/P18/guide_trace.jsonl")]
T19 = [json.loads(l) for l in open(REPO / "quick_runs/results/P19/guide_trace.jsonl")]
print(Counter(t["arm"] for t in T19))
# 1. One-sentence cut against placebo, per task, P18 + P19 runs
cut = []
for task, ids in FIX.items():
    row = {"task": task[:7]}
    for arm in ("G3s", "G3s-pl"):
        rs = [t for t in T18 + T19 if t["arm"] == arm and t["task"] == task]
        row[arm + " fix shown"] = sum(bool(set(t["shown"]) & ids) for t in rs); row[arm + " n"] = len(rs)
    row["p"] = fp(row["G3s fix shown"], row["G3s n"], row["G3s-pl fix shown"], row["G3s-pl n"])
    cut.append(row)
cut = pd.DataFrame(cut); cut""",
    """# 2. Escalating guide: runs (of 5 per task) in which a fix / wrong lesson was flagged for the user or shown to the agent
rows = []
for t in T19:
    if not t["arm"].startswith("G3x"): continue
    f, s = set(t["flagged"]), set(t["shown"]); fx = FIX.get(t["task"], set())
    rows.append({"arm": t["arm"], "task": t["task"][:7], "fix flagged": bool(f & fx), "fix shown": bool(s & fx),
                 "wrong flagged": bool(f & WRONG_IDS), "wrong shown": bool(s & WRONG_IDS), "n flagged": len(f)})
esc = pd.DataFrame(rows)
esc_t = esc.groupby(["arm", "task"])[["fix flagged", "fix shown", "wrong flagged", "wrong shown"]].sum()
print(esc.groupby("arm")[["fix flagged", "fix shown", "wrong flagged", "wrong shown", "n flagged"]].sum().to_string())
esc_t""",
    """# Reasons on flagged lessons (examples)
for t in T19:
    if not t["arm"].startswith("G3x") or t["run"] != "seed0": continue
    for r in t["rounds"]:
        for j in r.get("judgements") or []:
            if j.get("conflict") is True:
                try: i = int(j["id"])
                except (TypeError, ValueError, KeyError): continue
                kind = "fix" if i in FIX.get(t["task"], set()) else "wrong" if i in WRONG_IDS else "other"
                print(f"{t['arm']} {t['task'][:7]} {kind} [{i}]: {str(j.get('reason'))[:200]}")""",
    """# 3. Open prompt on the no-checker store; 4. labels of the write-time-probe (nfd) lessons
NF, NFD = GD.load_store("nf"), GD.load_store("nfd")
def ids(store, labfile, label):
    LAB = yaml.safe_load(open(REPO / labfile))
    return {i for i, L in enumerate(store) if L.get("nf") and L["source_task"] in LAB
            and f"{L['source_run']}/{L['k']}" in (LAB[L["source_task"]][label] or [])}
REINF = ids(NF, "quick_runs/p17/labels_nf.yaml", "reinforcing")
g3nnf = pd.DataFrame([{"task": t["task"][:7], "reinf shown": bool(set(t["shown"]) & REINF)} for t in T19 if t["arm"] == "G3n-nf"]).groupby("task").sum()
LABD = yaml.safe_load(open(REPO / "quick_runs/p19/labels_nfd.yaml"))
nfd_labels = {t[:7]: {k: len(v or []) for k, v in d.items()} for t, d in LABD.items()}
nfd_n = Counter(L["source_task"][:7] for L in NFD if L.get("probe"))
print("nfd lessons:", sum(nfd_n.values()), dict(nfd_n)); print("labels:", nfd_labels)
g3nnf""",
    P14[7].replace('ARMS = ["P12_A0", "P12_C1", "P13_E", "P13_G3", "P14_G3n", "P13_Ep", "P13_G3p", "P14_G3np"]',
                   'ARMS = ["P12_A0", "P12_C1", "P13_E", "P17_C1nf", "P17_Enf", "P19_C1nfd", "P19_Enfd", "P19_A0h"]')
          .replace('["A0 none", "C1 own", "E emb", "G3 (P13)", "G3n (P14)", "E-p", "G3-p (P13)", "G3n-p (P14)"]',
                   '["A0 none", "C1 own", "E emb", "C1-nf", "E-nf", "C1-nfd", "E-nfd", "A0h hint"]'),
    """# Exposure of the nfd store: runs (of 5) shown a fix / reinforcing nfd lesson
FIXD, REINFD = ids(NFD, "quick_runs/p19/labels_nfd.yaml", "fix"), ids(NFD, "quick_runs/p19/labels_nfd.yaml", "reinforcing")
TN = [json.loads(l) for l in open(REPO / "quick_runs/results/P19/trace_nfd.jsonl")]
expo = pd.DataFrame([{"arm": t["arm"], "task": t["task"][:7], "fix shown": bool(set(t["shown"]) & FIXD),
                      "reinf shown": bool(set(t["shown"]) & REINFD)} for t in TN if t["task"] in TASKS4]).groupby(["arm", "task"]).sum()
expo""",
    """# Harm and exact tests
sc = pd.DataFrame([{"arm": NAME[a], "task": r["task_id"], "score": r["score"]} for a in ARMS for r in recs[a]])
scores = sc.pivot_table(index="task", columns="arm", values="score", aggfunc="mean").round(4)
passed = (pooled * 40).round().astype(int)
fisher = {"C1-nf vs C1-nfd": fp(passed["C1-nf"], 40, passed["C1-nfd"], 40), "C1-nfd vs C1": fp(passed["C1-nfd"], 40, passed["C1 own"], 40),
          "E-nf vs E-nfd": fp(passed["E-nf"], 40, passed["E-nfd"], 40), "A0 vs A0h": fp(passed["A0 none"], 40, passed["A0h hint"], 40),
          **{f"cut vs placebo {r.task}": r.p for r in cut.itertuples()}}
print(fisher); scores""",
    """summary = {"cut_per_task": cut.to_dict("records"), "escalation": esc_t.reset_index().to_dict("records"),
  "escalation_totals": esc.groupby("arm")[["fix flagged", "fix shown", "wrong flagged", "wrong shown"]].sum().reset_index().to_dict("records"),
  "g3n_nf_reinf_shown": g3nnf.reset_index().to_dict("records"), "nfd_lessons": dict(nfd_n), "nfd_labels": nfd_labels,
  "nfd_exposure": expo.reset_index().to_dict("records"),
  "runs": {a: len(v) for a, v in recs.items()}, "stops": {a: dict(Counter(r["stop"] for r in v)) for a, v in recs.items()},
  "targeted": targeted.to_dict("records"), "targeted_pooled": pooled.round(3).to_dict(), "fisher_p": fisher,
  "score_mean": scores.to_dict("index")}
json.dump(summary, open(REPO / "quick_runs/results/P19/summary.json", "w"), indent=1, default=str)
"done" """,
]


P21 = [
    "# P21 review round 5 controls\n"
    "nfdtr: lessons written with the TRAIN-ONLY data profile and no evaluator outcome (P21 writer --train-only). "
    "C1-nfdtr / E-nfdtr on that store. A0g2 / C1g2: P12's no-memory and all-own-lessons arms rerun on the 2-GPU "
    "servers, to check the hardware split. Predictions: `quick_runs/predictions/P21.md`.",
    P12_SETUP, COMMON,
    P14[3], P14[4],
    """import sys, yaml
sys.path.insert(0, str(REPO / "quick_runs" / "p13")); sys.path.insert(0, str(REPO / "quick_runs" / "p12"))
import guide as GD
from scipy.stats import fisher_exact
fp = lambda a, na, b, nb: round(fisher_exact([[a, na - a], [b, nb - b]])[1], 4)
def labids(store, labfile, label):
    LAB = yaml.safe_load(open(REPO / labfile))
    return {i for i, L in enumerate(store) if L.get("nf") and L["source_task"] in LAB
            and f"{L['source_run']}/{L['k']}" in (LAB[L["source_task"]][label] or [])}
NFDTR = GD.load_store("nfdtr")
FIXD = labids(NFDTR, "quick_runs/p21/labels_nfdtr.yaml", "fix")
REINFD = labids(NFDTR, "quick_runs/p21/labels_nfdtr.yaml", "reinforcing")
nfd_n = Counter(L["source_task"][:7] for L in NFDTR if L.get("probe") == "train")
labd = {t[:7]: {k: len(v or []) for k, v in d.items()} for t, d in yaml.safe_load(open(REPO / "quick_runs/p21/labels_nfdtr.yaml")).items()}
print("nfdtr lessons:", sum(nfd_n.values()), dict(nfd_n)); print("labels:", labd)
# exposure: which fix / reinforcing lessons the C1-nfdtr and E-nfdtr guide stages showed
TN = [json.loads(l) for l in open(REPO / "quick_runs/results/P21/trace_nfdtr.jsonl")]
expo = pd.DataFrame([{"arm": t["arm"], "task": t["task"][:7], "fix shown": bool(set(t["shown"]) & FIXD),
                      "reinf shown": bool(set(t["shown"]) & REINFD)} for t in TN if t["task"] in TASKS4]).groupby(["arm", "task"]).sum()
expo""",
    P14[7].replace('ARMS = ["P12_A0", "P12_C1", "P13_E", "P13_G3", "P14_G3n", "P13_Ep", "P13_G3p", "P14_G3np"]',
                   'ARMS = ["P12_A0", "P12_C1", "P21_A0g2", "P21_C1g2", "P13_E", "P17_C1nf", "P19_C1nfd", "P21_C1nfdtr", "P21_Enfdtr"]')
          .replace('["A0 none", "C1 own", "E emb", "G3 (P13)", "G3n (P14)", "E-p", "G3-p (P13)", "G3n-p (P14)"]',
                   '["A0 4gpu", "C1 4gpu", "A0 2gpu", "C1 2gpu", "E emb", "C1-nf", "C1-nfd", "C1-nfdtr", "E-nfdtr"]'),
    """# Hardware split: per-task targeted checks, 4-GPU (P12) vs 2-GPU (P21) for A0 and C1
hw = targeted.set_index(["task", "check"])[["A0 4gpu", "A0 2gpu", "C1 4gpu", "C1 2gpu"]]
hw["A0 diff"] = (hw["A0 4gpu"] - hw["A0 2gpu"]).abs(); hw["C1 diff"] = (hw["C1 4gpu"] - hw["C1 2gpu"]).abs()
print("max per-check diff: A0", int(hw["A0 diff"].max()), "C1", int(hw["C1 diff"].max()))
hw""",
    """# Harm and exact tests
sc = pd.DataFrame([{"arm": NAME[a], "task": r["task_id"], "score": r["score"]} for a in ARMS for r in recs[a]])
scores = sc.pivot_table(index="task", columns="arm", values="score", aggfunc="mean").round(4)
passed = (pooled * 40).round().astype(int)
fisher = {"C1-nf vs C1-nfdtr": fp(passed["C1-nf"], 40, passed["C1-nfdtr"], 40),
          "C1-nfd vs C1-nfdtr": fp(passed["C1-nfd"], 40, passed["C1-nfdtr"], 40),
          "C1-nfdtr vs C1 4gpu": fp(passed["C1-nfdtr"], 40, passed["C1 4gpu"], 40),
          "A0 4gpu vs A0 2gpu": fp(passed["A0 4gpu"], 40, passed["A0 2gpu"], 40),
          "C1 4gpu vs C1 2gpu": fp(passed["C1 4gpu"], 40, passed["C1 2gpu"], 40)}
print(fisher); scores""",
    """summary = {"nfdtr_lessons": dict(nfd_n), "nfdtr_labels": labd,
  "nfdtr_exposure": expo.reset_index().to_dict("records"),
  "runs": {a: len(v) for a, v in recs.items()}, "stops": {a: dict(Counter(r["stop"] for r in v)) for a, v in recs.items()},
  "targeted": targeted.to_dict("records"), "targeted_pooled": pooled.round(3).to_dict(),
  "hardware": hw.reset_index().to_dict("records"), "fisher_p": fisher, "score_mean": scores.to_dict("index")}
json.dump(summary, open(REPO / "quick_runs/results/P21/summary.json", "w"), indent=1, default=str)
"done" """,
]


P22 = [
    "# P22 review round 6 controls: bias or prompt obedience?\n"
    "G3s-pl2: heavy placebo, a contentful non-feature instruction cut (TF03 model family, TF05 'fill missing with 0'). "
    "G3o-d: the P14 open prompt plus the data profile, so the guide may depart from the wording and has data to judge. "
    "Predictions: `quick_runs/predictions/P22.md`.",
    P12_SETUP, COMMON,
    P14[3], P14[4],
    """T18 = [json.loads(l) for l in open(REPO / "quick_runs/results/P18/guide_trace.jsonl")]
T19 = [json.loads(l) for l in open(REPO / "quick_runs/results/P19/guide_trace.jsonl")]
T22 = [json.loads(l) for l in open(REPO / "quick_runs/results/P22/guide_trace.jsonl")]
print(Counter(t["arm"] for t in T22))
# Heavy placebo vs feature cut vs light placebo, fix lessons shown per task (of 5; cut/placebo also have P19's 10 more)
rows = []
for task, ids in {k: FIX[k] for k in ("sf_tf03_correlation_causation", "sf_tf05_credit_card_leakage")}.items():
    row = {"task": task[:7]}
    row["G3 (none cut)"] = sum(bool(set(t["shown"]) & ids) for t in T13 if t["arm"] == "G3" and t["task"] == task)
    row["G3s feature cut"] = sum(bool(set(t["shown"]) & ids) for t in T18 + T19 if t["arm"] == "G3s" and t["task"] == task)
    row["G3s-pl light"] = sum(bool(set(t["shown"]) & ids) for t in T18 + T19 if t["arm"] == "G3s-pl" and t["task"] == task)
    row["G3s-pl2 heavy"] = sum(bool(set(t["shown"]) & ids) for t in T22 if t["arm"] == "G3s-pl2" and t["task"] == task)
    rows.append(row)
heavy = pd.DataFrame(rows); print(heavy.to_string()); heavy""",
    """# G3o-d (open prompt + profile): TF03/TF05 fix and keep-column_x shown, vs G3n (open, no data) and G3d (profile)
def fx(trs, arm, task, ids): return sum(bool(set(t["shown"]) & ids) for t in trs if t["arm"] == arm and t["task"] == task)
T14 = [json.loads(l) for l in open(REPO / "quick_runs/results/P14/guide_trace.jsonl")]
T16 = [json.loads(l) for l in open(REPO / "quick_runs/results/P16/guide_trace.jsonl")]
rows = []
for task in ("sf_tf03_correlation_causation", "sf_tf05_credit_card_leakage"):
    rows.append({"task": task[:7], "arm": "G3n open", "fix": fx(T14, "G3n", task, FIX[task]), "keepx": fx(T14, "G3n", task, CONTRA3) if task.endswith("causation") else 0})
    rows.append({"task": task[:7], "arm": "G3d profile", "fix": fx(T16, "G3d", task, FIX[task]), "keepx": fx(T16, "G3d", task, CONTRA3) if task.endswith("causation") else 0})
    rows.append({"task": task[:7], "arm": "G3o-d open+prof", "fix": fx(T22, "G3o-d", task, FIX[task]), "keepx": fx(T22, "G3o-d", task, CONTRA3) if task.endswith("causation") else 0})
od = pd.DataFrame(rows); print(od.to_string()); od""",
    """# Reasons: do G3s-pl2 and G3o-d cite the instructions when rejecting fix lessons? (pre-set regex + contradict rule)
NOC = r"task (says|asks|states|requires|instructs)|instruction|use all|all available"
TASKWORD = r"prompt|task|requirement|constraint"
def cites_instr(reason):
    if not re.search(INSTR, reason, re.I): return False
    return bool(re.search(NOC, reason, re.I) or re.search(TASKWORD, reason, re.I))
rows = []
for t in T22:
    for r in t.get("rounds", []):
        for j in r.get("judgements") or []:
            try: i = int(j.get("id"))
            except (TypeError, ValueError): continue
            if i in FIX.get(t["task"], set()) and i in r["candidates"]:
                rows.append({"arm": t["arm"], "task": t["task"][:7], "useful": j.get("useful") is True,
                             "instr": cites_instr(str(j.get("reason", ""))), "reason": str(j.get("reason", ""))})
jr = pd.DataFrame(rows)
print(jr[~jr.useful].groupby("arm").instr.agg(["sum", "count", "mean"]).round(2).to_string())
for _, r in jr.groupby(["arm", "task", "useful"]).head(1).iterrows(): print(f"{r.arm} {r.task} useful={r.useful}: {r.reason[:200]}")""",
    P14[7].replace('ARMS = ["P12_A0", "P12_C1", "P13_E", "P13_G3", "P14_G3n", "P13_Ep", "P13_G3p", "P14_G3np"]',
                   'ARMS = ["P12_A0", "P12_C1", "P13_E", "P13_G3", "P14_G3n", "P16_G3d", "P22_G3od"]')
          .replace('["A0 none", "C1 own", "E emb", "G3 (P13)", "G3n (P14)", "E-p", "G3-p (P13)", "G3n-p (P14)"]',
                   '["A0 none", "C1 own", "E emb", "G3", "G3n open", "G3d profile", "G3o-d open+prof"]'),
    """# Harm and exact tests
from scipy.stats import fisher_exact
sc = pd.DataFrame([{"arm": NAME[a], "task": r["task_id"], "score": r["score"]} for a in ARMS for r in recs[a]])
scores = sc.pivot_table(index="task", columns="arm", values="score", aggfunc="mean").round(4)
passed = (pooled * 40).round().astype(int)
fp = lambda a, b: round(fisher_exact([[a, 40 - a], [b, 40 - b]])[1], 4)
h = heavy.set_index("task")
# feature cut has 15 runs/task (P18+P19); heavy placebo has 5 (P22)
f03, hv03 = int(h.loc["sf_tf03","G3s feature cut"]), int(h.loc["sf_tf03","G3s-pl2 heavy"])
fisher = {"TF03 fix G3s(15) vs G3s-pl2(5)": round(fisher_exact([[f03, 15-f03],[hv03, 5-hv03]])[1],4),
          "G3o-d vs G3d pooled": fp(passed["G3o-d open+prof"], passed["G3d profile"]),
          "G3o-d vs G3n pooled": fp(passed["G3o-d open+prof"], passed["G3n open"])}
print(fisher); scores""",
    """summary = {"heavy_placebo": heavy.to_dict("records"), "open_profile": od.to_dict("records"),
  "reasons_instr": jr[~jr.useful].groupby("arm").instr.agg(["sum", "count"]).reset_index().to_dict("records"),
  "runs": {a: len(v) for a, v in recs.items()}, "stops": {a: dict(Counter(r["stop"] for r in v)) for a, v in recs.items()},
  "targeted": targeted.to_dict("records"), "targeted_pooled": pooled.round(3).to_dict(), "fisher_p": fisher,
  "score_mean": scores.to_dict("index")}
json.dump(summary, open(REPO / "quick_runs/results/P22/summary.json", "w"), indent=1, default=str)
jr.to_csv(REPO / "quick_runs/results/P22/fix_reasons.csv", index=False)
"done" """,
]


P20 = [
    "# P20 a frontier, different-family guide (Claude Opus 5.5, Fable 5.1)\n"
    "The guide is a Claude model via the Anthropic API; the agent stays Gemma 4 31B. Original P13 prompt, R=3, three "
    "stores (clean, P13 wrong lessons, agreeing wrong lessons). Tests whether the instruction-agreement behaviour "
    "generalises beyond the two open ~30B judges (Gemma, Qwen). Predictions: `quick_runs/predictions/P20.md`.",
    P12_SETUP, COMMON,
    P14[3], P14[4],
    """TC = [json.loads(l) for l in open(REPO / "quick_runs/results/P20/trace_opus.jsonl")]
TF = [json.loads(l) for l in open(REPO / "quick_runs/results/P20/trace_fable.jsonl")]
T16 = [json.loads(l) for l in open(REPO / "quick_runs/results/P16/guide_trace.jsonl")]
for t in TC + TF + T16: t.setdefault("seen", t["shown"]); t.setdefault("accepted", t["shown"])
print(Counter(t["arm"] for t in TC), Counter(t["arm"] for t in TF))
# Stage 1: fix lessons and wrong lessons shown, Claude vs Gemma (G3/G3-p/G3-pc) and Qwen (Q3)
s1 = pd.DataFrame([stage1(T13, "G3"), stage1(TC, "G3c"), stage1(TF, "G3f")]).set_index("arm")
wr = pd.DataFrame([wrong(T13, "G3-p"), wrong(TC, "G3c-p"), wrong(TF, "G3f-p"),
                   wrong(T16, "G3-pc"), wrong(TC, "G3c-pc"), wrong(TF, "G3f-pc")]).set_index("arm")
print(s1.to_string()); wr""",
    """# Reasons: share of fix-lesson rejections citing the instructions (pre-set regex)
rows = []
for t in TC + TF:
    for r in t.get("rounds", []):
        for j in r.get("judgements") or []:
            try: i = int(j.get("id"))
            except (TypeError, ValueError): continue
            kind = "fix" if i in FIX.get(t["task"], set()) else "wrong" if i in WRONG_IDS else None
            if kind is None or i not in r["candidates"]: continue
            rows.append({"model": "Opus" if t in TC else "Fable", "arm": t["arm"], "kind": kind,
                         "useful": j.get("useful") is True, "instr": bool(re.search(INSTR, str(j.get("reason","")), re.I)),
                         "reason": str(j.get("reason", ""))})
jr = pd.DataFrame(rows)
print(jr[(~jr.useful) & (jr.kind == "fix")].groupby("model").instr.agg(["sum", "count", "mean"]).round(2).to_string())
print("\\naccepted/rejected agreeing wrong lessons, examples:")
for _, r in jr[jr.kind == "wrong"].groupby(["model", "useful"]).head(1).iterrows(): print(f"{r.model} useful={r.useful}: {r.reason[:170]}")""",
    P14[7].replace('ARMS = ["P12_A0", "P12_C1", "P13_E", "P13_G3", "P14_G3n", "P13_Ep", "P13_G3p", "P14_G3np"]',
                   'ARMS = ["P12_A0", "P13_E", "P13_G3", "P20_G3c", "P20_G3f", "P13_G3p", "P20_G3cp", "P20_G3fp", "P16_G3pc", "P20_G3cpc", "P20_G3fpc"]')
          .replace('["A0 none", "C1 own", "E emb", "G3 (P13)", "G3n (P14)", "E-p", "G3-p (P13)", "G3n-p (P14)"]',
                   '["A0 none", "E emb", "G3 Gemma", "G3c Opus", "G3f Fable", "G3-p", "G3c-p", "G3f-p", "G3-pc", "G3c-pc", "G3f-pc"]'),
    """# Harm, cost (Claude USD from the ledgers), and exact tests
from scipy.stats import fisher_exact
sc = pd.DataFrame([{"arm": NAME[a], "task": r["task_id"], "score": r["score"]} for a in ARMS for r in recs[a]])
scores = sc.pivot_table(index="task", columns="arm", values="score", aggfunc="mean").round(4)
def usd(f):
    try: return round(sum(json.loads(l)["usd"] for l in open(REPO / f)), 2)
    except FileNotFoundError: return None
cost = {"Opus 5.5 guide": usd("quick_runs/results/P20/ledger_opus.jsonl"), "Fable 5.1 guide": usd("quick_runs/results/P20/ledger_fable.jsonl")}
passed = (pooled * 40).round().astype(int)
fp = lambda a, b: round(fisher_exact([[a, 40 - a], [b, 40 - b]])[1], 4)
fisher = {"pooled G3 vs G3c": fp(passed["G3 Gemma"], passed["G3c Opus"]), "pooled G3 vs G3f": fp(passed["G3 Gemma"], passed["G3f Fable"]),
          "agreeing shown G3-pc vs G3c-pc": round(fisher_exact([[int(wr.loc["G3-pc","wrong shown"]),20-int(wr.loc["G3-pc","wrong shown"])],[int(wr.loc["G3c-pc","wrong shown"]),20-int(wr.loc["G3c-pc","wrong shown"])]])[1],4),
          "agreeing shown G3-pc vs G3f-pc": round(fisher_exact([[int(wr.loc["G3-pc","wrong shown"]),20-int(wr.loc["G3-pc","wrong shown"])],[int(wr.loc["G3f-pc","wrong shown"]),20-int(wr.loc["G3f-pc","wrong shown"])]])[1],4)}
print("Claude guide cost (USD):", cost); print(fisher); scores""",
    """summary = {"stage1_fix": s1.reset_index().to_dict("records"), "stage1_wrong": wr.reset_index().to_dict("records"),
  "fix_reject_instr": jr[(~jr.useful)&(jr.kind=="fix")].groupby("model").instr.agg(["sum","count"]).reset_index().to_dict("records"),
  "agreeing_judgements": jr[jr.kind=="wrong"].groupby(["model","useful"]).size().reset_index(name="n").to_dict("records"),
  "runs": {a: len(v) for a, v in recs.items()}, "stops": {a: dict(Counter(r["stop"] for r in v)) for a, v in recs.items()},
  "targeted": targeted.to_dict("records"), "targeted_pooled": pooled.round(3).to_dict(), "fisher_p": fisher, "cost_usd": cost,
  "score_mean": scores.to_dict("index")}
json.dump(summary, open(REPO / "quick_runs/results/P20/summary.json", "w"), indent=1, default=str)
jr.to_csv(REPO / "quick_runs/results/P20/judgements.csv", index=False)
"done" """,
]

NOTEBOOKS = {"P10": P10, "P9": P9, "P1": P1, "P1b": P1b, "P2_P4": P2P4, "P12": P12, "P13": P13, "P14": P14, "P15": P15, "REVIEW_CHECKS": REVIEW_CHECKS, "P16": P16, "P17": P17, "REVIEW2_JUDGE": REVIEW2_JUDGE, "P18": P18, "P19": P19, "P21": P21, "P22": P22, "P20": P20}

if __name__ == "__main__":
    for n in sys.argv[1:]:
        print(build(n, NOTEBOOKS[n]))
