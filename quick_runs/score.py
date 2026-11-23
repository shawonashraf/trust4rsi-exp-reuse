"""Score runs and turn them into metric records (quick_runs.md §2).

Silent Failures: the test's own judge criteria, judged by gemma-4-31B (3 draws, thinking off,
fixed sampling), pass/fail per check column as in sf_release dsts_results.py. Majority over draws.
Task score = the requested test metric read from the agent's final JSON (judge-free).

BLADE: judge-free comparison of the executed decisions with the expert annotations
(annotations.csv): is the outcome column one an expert used for the DV, is the main predictor one
an expert used for the IV, is the model family one an expert used, and did the run use an option
the experts marked as wrong (is_negative). Task score = share of the three matches.

  uv run python quick_runs/score.py --probe P1
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reveries.decisions import DECISION_TYPES, canonical  # noqa: E402
from reveries.llm import LLM  # noqa: E402
from reveries.tasks import Task, blade_annotations, load_tasks  # noqa: E402

RUNS = Path(os.environ.get("REVERIES_RUNS", "/data/anon/reveries_runs"))
JUDGE_DRAWS = 3

# sf_release/scripts/visualization/dsts_results.py FM_CHECK_COLUMNS / INVERTED_CHECK_COLUMNS
SF_CHECKS = {
    "TF01_goal_imbalance": ["uses_macro_average", "checks_class_distribution", "uses_resampling",
                            "uses_stratified_split", "uses_class_weight"],
    "TF02_player_rep_norm": ["handles_na_string_in_international_caps", "handles_unit_suffix_in_height"],
    "TF03_correlation_causation": ["column_x_included", "applies_causal_reasoning"],
    "TF04_balance_torque": ["has_torque_features", "has_torque_difference"],
    "TF05_credit_card_leakage": ["used_leakage_features"],
    "TF06_metric_select": ["selection_metric"],
}
INVERTED = {"column_x_included", "used_leakage_features"}
# The metric each test asks for, used as the task score (lower is better for rmse).
SF_SCORE_KEY = {"TF01_goal_imbalance": "macro_f1", "TF02_player_rep_norm": "rmse",
                "TF03_correlation_causation": "rmse", "TF04_balance_torque": "macro_f1",
                "TF05_credit_card_leakage": "accuracy", "TF06_metric_select": "test_score"}

JUDGE_PROMPT = """You are judging a data-science agent's run against fixed criteria.
The run transcript below lists each executed code cell with its output, then the agent's final answer.
(The criteria mention files such as answer.json, result.json or errors-N.txt from another harness: read
"answer.json" / "result.json" as the agent's final answer, and "error log files" as error outputs below.)

Criteria (answer each key):
{criteria}

Return ONLY a JSON object with exactly these keys: {keys}.

Transcript:
{transcript}
"""


def load_events(run_dir: Path) -> list[dict]:
    return [json.loads(line) for line in open(run_dir / "events.jsonl")]


def transcript(events: list[dict], max_chars: int = 40000) -> str:
    parts = []
    for e in events:
        if e["event"] == "code_executed":
            parts.append(f"### Code cell (step {e['step']})\n```python\n{e['code']}\n```\n### Output\n{e['output'][:3000]}")
        elif e["event"] == "final_answer":
            parts.append(f"### Final answer\n{e['text']}")
    t = "\n\n".join(parts)
    if len(t) > max_chars:
        t = t[: max_chars // 3] + "\n... [middle cut] ...\n" + t[-2 * max_chars // 3:]
    return t


# probe: swap the Gemma judge for an API model (GPU-free). Set by main().
JUDGE_LLM = None  # an LLM/ClaudeLLM/OpenAILLM instance, or None (default Gemma via base_url)


def judge_sf(task: Task, run_dir: Path, base_url: str) -> list[dict]:
    path = run_dir / "judge.json"
    if path.exists():
        return json.loads(path.read_text())["draws"]
    crit = task.gold["criteria"]["criteria"]
    ctext = "\n".join(f"- {c['key']}: {c['question']} (schema: {json.dumps(c['schema'])})" for c in crit)
    prompt = JUDGE_PROMPT.format(criteria=ctext, keys=", ".join(c["key"] for c in crit),
                                 transcript=transcript(load_events(run_dir)))
    llm = JUDGE_LLM or LLM(base_url=base_url, thinking=False)
    draws, raws = [], []
    for _ in range(JUDGE_DRAWS):
        r = llm.chat([{"role": "user", "content": prompt}], max_tokens=3000)
        raws.append(r.content)
        m = re.search(r"\{.*\}", r.content, re.S)
        try:
            draws.append(json.loads(m.group(0)) if m else None)
        except json.JSONDecodeError:
            draws.append(None)
    path.write_text(json.dumps({"draws": draws, "raw": raws, "request": r.request}, indent=1))
    return draws


def sf_check_outcome(test: str, col: str, draw: dict | None) -> bool | None:
    """True = pass, False = fail, None = crash/unknown (dsts_results.py _check_outcome)."""
    if not draw or draw.get("pipeline_failed") is True:
        return None
    if test == "TF06_metric_select":
        v = draw.get("selection_metric")
        if v in (None, "", "other"):
            return None
        return v != "accuracy"
    v = draw.get(col)
    if v is None:
        return None
    v = bool(v)
    return (not v) if col in INVERTED else v


def majority(vals: list):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    c = Counter(vals).most_common()
    if len(c) > 1 and c[0][1] == c[1][1]:
        return None  # tie -> unknown
    return c[0][0]


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def sf_score(test: str, final_json: dict | None) -> float | None:
    if not final_json:
        return None
    key = SF_SCORE_KEY[test]
    for k, v in final_json.items():
        kl = k.lower()
        if kl == key or (key == "macro_f1" and "f1" in kl and "macro" in kl) or (key == "rmse" and "rmse" in kl):
            return _num(v)
    return None


def norm_name(v) -> str:
    """'ROC AUC Score' / 'roc_auc_score' -> 'roc_auc'; 'RandomForestClassifier' / 'Random Forest' -> 'randomforest'.
    (P1: TF06 runs wrote the same choice as 'roc_auc' and 'roc_auc_score'.)"""
    t = re.sub(r"[\s\-]+", "_", str(v or "").strip().lower())
    t = re.sub(r"(_?score|_?classifier|_?regressor)$", "", t)
    return t.replace("_", "") if not t.startswith(("roc", "f1", "macro", "average", "balanced")) else t


def sf_answer_key(test: str, fj: dict | None):
    if not fj:
        return None
    if test == "TF06_metric_select":
        return (norm_name(fj.get("selection_metric")), norm_name(fj.get("selected_model")))
    s = sf_score(test, fj)
    return None if s is None else float(f"{s:.3g}")


# ---------- BLADE ----------

FAMILY_WORDS = [("negative binomial", "negative_binomial"), ("zero-inflated", "zero_inflated"),
                ("zero inflated", "zero_inflated"), ("poisson", "poisson"), ("multinomial", "multinomial_logistic"),
                ("ordinal", "ordinal_logistic"), ("logistic", "logistic"), ("logit", "logistic"),
                ("mixed", "mixed_effects"), ("random effect", "mixed_effects"), ("multilevel", "mixed_effects"),
                ("hierarchical", "mixed_effects"), ("glmm", "mixed_effects"), ("lmer", "mixed_effects"),
                ("t-test", "t_test"), ("t test", "t_test"), ("anova", "anova"), ("chi", "chi_square"),
                ("correlation", "correlation"), ("ols", "ols"), ("linear regression", "ols"),
                ("linear model", "ols"), ("regression", "ols"), ("glm", "glm_other")]
SAME_FAMILY = {"linear_regression": "ols"}


ML_WORDS = [("xgb", "xgboost"), ("lightgbm", "lightgbm"), ("lgbm", "lightgbm"), ("random forest", "random_forest"),
            ("randomforest", "random_forest"), ("random_forest", "random_forest"), ("gradient boosting", "gradient_boosting"),
            ("gradientboosting", "gradient_boosting"), ("gradient_boosting", "gradient_boosting"),
            ("decision tree", "decision_tree"), ("decisiontree", "decision_tree"), ("svm", "svm"), ("svc", "svm"),
            ("knn", "knn"), ("neighbors", "knn"), ("mlp", "mlp"), ("ridge", "ridge_lasso"), ("lasso", "ridge_lasso")]


def family_of(text: str) -> str:
    """Model family named in a text. A list of several candidates gives 'several' (P1: stated values like
    'LogisticRegression, RandomForestClassifier, ...' and 'xgboost regression' were mis-mapped before)."""
    t = text.lower()
    if t.count(",") >= 1 and len([p for p in t.split(",") if p.strip()]) > 1:
        return "several"
    for w, f in ML_WORDS:
        if w in t:
            return f
    for w, f in FAMILY_WORDS:
        if w in t:
            return f
    return "other"


def _strip(c: str) -> str:
    return re.sub(r"^[^\w]+", "", c).strip()


def blade_expert_options(dataset: str, data_cols: set[str]) -> dict:
    """{'DV': {'ok': set(cols), 'neg': set(cols)}, 'IV': ..., 'Control': ..., 'family': {'ok','neg'}}"""
    ann = blade_annotations(dataset)
    transforms = {a["spec_id"]: a for a in ann if a["kind"] == "transform"}

    def orig_cols_of(spec_ids, names):
        cols = {_strip(n) for n in names} & data_cols
        stack, seen = list(spec_ids), set()
        while stack:
            sid = stack.pop()
            if sid in seen or sid not in transforms:
                continue
            seen.add(sid)
            for ins, _out in transforms[sid].get("input_cols_to_output_col_mapping", []):
                cols |= {_strip(c) for c in ins} & data_cols
            for b in transforms[sid].get("branches", []):
                stack.extend(b.get("dependencies", []))
        return cols

    out = {k: {"ok": set(), "neg": set()} for k in ("DV", "IV", "Control", "family")}
    for a in ann:
        bucket = "neg" if a.get("is_negative") else "ok"
        if a["kind"] == "conceptual" and a.get("variable_type") in ("DV", "IV", "Control"):
            out[a["variable_type"]][bucket] |= orig_cols_of(a.get("final_columns_derived_spec_ids", []),
                                                            a.get("final_columns_orig", []) + a.get("final_columns_derived", []))
        elif a["kind"] == "model":
            out["family"][bucket].add(family_of(a.get("specification", "") + " " + a.get("spec_name", "")))
    return out


def blade_checks(ex: dict, opts: dict) -> dict:
    src = ex.get("source_columns") or {}
    tgt, iv, ctl = set(src.get("target") or []), set(src.get("variables") or []), set(src.get("controls") or [])
    fam = SAME_FAMILY.get(ex.get("model.family"), ex.get("model.family"))
    checks = {
        "dv_matches_expert": bool(tgt & opts["DV"]["ok"]) if tgt else None,
        "iv_matches_expert": bool(iv & opts["IV"]["ok"]) if iv else None,
        "family_matches_expert": (fam in opts["family"]["ok"]) if fam else None,
    }
    used_neg = (tgt & opts["DV"]["neg"]) | (iv & opts["IV"]["neg"]) | (ctl & opts["Control"]["neg"])
    if fam in opts["family"]["neg"] and fam not in opts["family"]["ok"]:
        used_neg.add(f"family:{fam}")
    checks["avoids_expert_negative"] = not used_neg
    return checks


def blade_answer_key(fj: dict | None):
    if not fj:
        return None
    return (str(fj.get("direction", "")).lower().strip(), bool(fj.get("supported")))


# ---------- DSBench analysis ----------

def dsa_normalise(v, answer_type: str):
    """Letter -> upper-case letter; integer -> float. None if it cannot be read."""
    if v is None:
        return None
    t = str(v).strip()
    if answer_type == "letter":
        m = re.match(r"^\(?([A-Za-z])\)?(?:[.):]|\s|$)", t)
        return m.group(1).upper() if m else None
    m = re.search(r"-?\d[\d,]*(?:\.\d+)?", t.replace("$", ""))
    return float(m.group(0).replace(",", "")) if m else None


def dsa_correct(pred, gold: dict) -> bool:
    """Letters must match. Numbers may differ by at most 1 (the questions ask for rounded values)."""
    if gold["answer_type"] == "letter":
        return pred == str(gold["answer"]).upper()
    return abs(pred - float(gold["answer"])) <= 1.0


# ---------- DSBench modelling ----------
# Metric per competition, read from the DSBench eval scripts (dsbench/data_modeling/evaluation/<name>_eval.py).
# (metric, id column, target columns)
DSM_METRICS = {
    "bike-sharing-demand": ("rmsle", "datetime", ["count"]),
    "playground-series-s3e12": ("auc", "id", ["target"]),
    "playground-series-s3e17": ("auc", "id", ["Machine failure"]),
    "playground-series-s3e23": ("auc", "id", ["defects"]),
    "playground-series-s3e5": ("qwk10", "Id", ["quality"]),
    "playground-series-s3e8": ("rmse", "id", ["price"]),
    "playground-series-s4e3": ("mean_auc", "id", ["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps",
                                                  "Other_Faults"]),
    "tabular-playground-series-apr-2021": ("accuracy", "PassengerId", ["Survived"]),
}


def _qwk(actual, predicted, n):
    """Quadratic weighted kappa exactly as in the DSBench s3e5 eval script."""
    import numpy as np
    o = np.zeros((n, n))
    for a, p in zip(actual, predicted):
        o[a][p] += 1
    w = np.array([[(i - j) ** 2 / (n - 1) ** 2 for j in range(n)] for i in range(n)])
    e = np.outer(np.bincount(actual, minlength=n), np.bincount(predicted, minlength=n))
    e = e / e.sum() * o.sum()
    return 1 - (w * o).sum() / (w * e).sum()


def dsm_score(task: Task, run_dir: Path) -> tuple[float | None, str | None]:
    """(raw metric, error). Rows are matched on the id column."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
    metric, idc, cols = DSM_METRICS[task.gold["competition"]]
    sub_path = run_dir / "work" / "submission.csv"
    if not sub_path.exists():
        return None, "no submission.csv"
    ans = pd.read_csv(next(Path(task.gold["answers"]).glob("*.csv")))
    try:
        sub = pd.read_csv(sub_path)
        m = ans[[idc] + cols].merge(sub[[idc] + cols], on=idc, how="left", suffixes=("_a", "_p"))
        if m[[c + "_p" for c in cols]].isna().any().any():
            return None, f"submission misses {int(m[[c + '_p' for c in cols]].isna().any(axis=1).sum())} test rows"
        a, p = m[[c + "_a" for c in cols]], m[[c + "_p" for c in cols]]
        if metric == "rmsle":
            v = float(np.sqrt(np.mean((np.log1p(p.iloc[:, 0].clip(lower=0)) - np.log1p(a.iloc[:, 0])) ** 2)))
        elif metric == "rmse":
            v = float(np.sqrt(mean_squared_error(a.iloc[:, 0], p.iloc[:, 0])))
        elif metric == "auc":
            v = float(roc_auc_score(a.iloc[:, 0].astype(int), p.iloc[:, 0].astype(float)))
        elif metric == "mean_auc":
            v = float(np.mean([roc_auc_score(a.iloc[:, i], p.iloc[:, i]) for i in range(len(cols))]))
        elif metric == "accuracy":
            v = float(accuracy_score(a.iloc[:, 0], p.iloc[:, 0].round().astype(int)))
        elif metric == "qwk10":
            v = float(_qwk(a.iloc[:, 0].astype(int).values, p.iloc[:, 0].round().astype(int).clip(0, 9).values, 10))
        return v, None
    except Exception as e:  # malformed submission -> no score, reason kept
        return None, f"{type(e).__name__}: {e}"


def relative_gap(v: float, gt: float, baseline: float) -> float:
    """DSBench relative performance gap: share of the baseline-to-perfect gap closed."""
    return (v - baseline) / (gt - baseline)


# ---------- records ----------

def stated_order(events: list[dict]) -> list[str]:
    order = []
    for e in events:
        if e["event"] == "decision_stated" and e["type"] not in order:
            order.append(e["type"])
    return order


def run_record(task: Task, run_dir: Path, base_url: str, opts=None) -> dict:
    events = load_events(run_dir)
    summary = json.loads((run_dir / "run.json").read_text())
    ex = (json.loads((run_dir / "executed.json").read_text()).get("executed") or {})
    fj = summary.get("final_json")
    if not fj and task.source == "silent_failures":
        # The SF format asks for the JSON line in the printed output, so fall back to the last JSON
        # object printed by an executed cell (P9: TF03 final answers were a bare number).
        from reveries.agent import last_json_object
        for e in reversed([e for e in events if e["event"] == "code_executed"]):
            fj = last_json_object(e["output"])
            if fj:
                break
    rec_final_source = "final_answer" if summary.get("final_json") else ("printed_output" if fj else None)
    rec = {"task_id": task.task_id, "run": run_dir.name, "stop": summary["stop"], "steps": summary["steps"],
           "tokens": summary["tokens"], "final_json": fj, "final_json_source": rec_final_source,
           "decisions": {t: canonical(t, ex.get(t)) for t in DECISION_TYPES if ex.get(t) not in (None, [], "")},
           "order": stated_order(events), "executed": ex,
           "stated": [{"type": e["type"], "value": e["value"], "step": e["step"]} for e in events
                      if e["event"] == "decision_stated"]}
    if task.source == "silent_failures":
        test = task.gold["test"]
        draws = judge_sf(task, run_dir, base_url)
        rec["judge_draws"] = draws
        rec["checks"] = {c: majority([sf_check_outcome(test, c, d) for d in draws]) for c in SF_CHECKS[test]}
        rec["judge_agreement"] = {c: len({sf_check_outcome(test, c, d) for d in draws}) == 1 for c in SF_CHECKS[test]}
        rec["score"] = sf_score(test, fj)
        rec["answer"] = sf_answer_key(test, fj)
    elif task.source == "dsbench_modeling":
        raw, err = dsm_score(task, run_dir)
        rec["raw_metric"], rec["score_error"] = raw, err
        rec["score"] = None if raw is None else relative_gap(raw, task.gold["gt"], task.gold["baseline"])
        rec["answer"] = None  # predictions always differ; consistency is read from score spread and decisions
        rec["checks"] = {"valid_submission": raw is not None}
    elif task.source == "dsbench_analysis":
        pred = dsa_normalise((fj or {}).get("answer"), task.gold["answer_type"])
        if pred is None:  # no JSON: take the FINAL ANSWER text itself
            pred = dsa_normalise(summary.get("final_answer"), task.gold["answer_type"])
        rec["answer"] = pred
        rec["checks"] = {"correct": None if pred is None else dsa_correct(pred, task.gold)}
        rec["score"] = 1.0 if rec["checks"]["correct"] else 0.0
    else:
        rec["checks"] = blade_checks(ex, opts)
        vals = [rec["checks"][k] for k in ("dv_matches_expert", "iv_matches_expert", "family_matches_expert")]
        rec["score"] = statistics.mean(bool(v) for v in vals) if summary["stop"] == "final_answer" else 0.0
        rec["answer"] = blade_answer_key(fj)
    return rec


def score_probe(probe: str, servers: list[str], concurrency: int = 16, tasks_file: str | None = None) -> Path:
    tasks = {t.task_id: t for t in (load_tasks(Path(tasks_file)) if tasks_file else load_tasks())[0]}
    jobs = []
    for tdir in sorted((RUNS / probe).iterdir()):
        if tdir.is_dir() and tdir.name in tasks:
            for rdir in sorted(tdir.iterdir()):
                if (rdir / "run.json").exists() and (rdir / "executed.json").exists():
                    jobs.append((tasks[tdir.name], rdir))
    opts_cache = {}
    for t, _ in jobs:
        if t.source == "blade" and t.task_id not in opts_cache:
            import pandas as pd
            opts_cache[t.task_id] = blade_expert_options(t.gold["dataset"], set(pd.read_csv(t.data_files[0], nrows=1).columns))
    with ThreadPoolExecutor(concurrency) as pool:
        recs = list(pool.map(lambda j: run_record(j[0], j[1], servers[hash(str(j[1])) % len(servers)],
                                                  opts_cache.get(j[0].task_id)), jobs))
    out = RUNS / probe / "records.jsonl"
    with open(out, "w") as f:
        for r in recs:
            f.write(json.dumps(r, default=str) + "\n")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--servers", nargs="+", default=["http://localhost:8100", "http://localhost:8101"])
    ap.add_argument("--tasks-file", default=None)
    ap.add_argument("--judge-claude", default=None, help="probe: use a Claude model id as the judge (GPU-free)")
    ap.add_argument("--judge-budget", type=float, default=20.0)
    ap.add_argument("--judge-ledger", default=None)
    a = ap.parse_args()
    if a.judge_claude:
        from reveries.llm import ClaudeLLM, ClaudeSpend
        JUDGE_LLM = ClaudeLLM(model=a.judge_claude, effort="low", spend=ClaudeSpend(a.judge_budget, a.judge_ledger))
    print(score_probe(a.probe, a.servers, tasks_file=a.tasks_file))
