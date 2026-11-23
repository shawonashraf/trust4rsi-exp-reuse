"""Decision log: parse stated DECISION lines, and extract executed decisions from code.

Stated decisions are what the agent says it does. Executed decisions come from a
post-hoc LLM pass over the code that actually ran (catches silent decisions). The
executed values are put into a canonical form so runs can be compared.
"""
from __future__ import annotations

import json
import re

DECISION_TYPES = [
    "framing.target", "framing.population", "framing.unit", "variables", "transformation",
    "model.family", "model.controls", "validation", "metric", "verdict", "final_selection",
]

DECISION_RE = re.compile(r"^\s*\**DECISION\**\s+type\s*=\s*([\w.]+)\s+value\s*=\s*(.*?)(?:\s+reason\s*=\s*(.*))?\s*$",
                         re.M | re.I)


def parse_decisions(text: str) -> list[dict]:
    out = []
    for m in DECISION_RE.finditer(text):
        t = m.group(1).strip().lower()
        out.append({"type": t, "value": m.group(2).strip().strip("`"), "reason": (m.group(3) or "").strip(),
                    "known_type": t in DECISION_TYPES})
    return out


# Canonical vocabularies for the executed-decision extractor.
MODEL_FAMILIES = ["ols", "logistic", "multinomial_logistic", "ordinal_logistic", "poisson", "negative_binomial",
                  "zero_inflated", "glm_other", "mixed_effects", "t_test", "chi_square", "correlation", "anova",
                  "nonparametric_test", "linear_regression", "ridge_lasso", "random_forest", "gradient_boosting",
                  "xgboost", "lightgbm", "svm", "knn", "decision_tree", "naive_bayes", "mlp", "other"]
VALIDATIONS = ["none", "given_test_set_only", "holdout_split", "kfold_cv", "stratified_kfold_cv", "grid_search_cv",
               "holdout_and_cv", "other"]
METRICS = ["accuracy", "macro_f1", "f1_binary", "f1_weighted", "roc_auc", "average_precision", "precision", "recall",
           "balanced_accuracy", "rmse", "mae", "r2", "coefficient_and_p", "odds_ratio", "incidence_rate_ratio",
           "test_statistic_and_p", "correlation", "other"]

EXTRACTOR_PROMPT = """You are auditing what a data-analysis run actually did. Below are the dataset columns,
the Python code cells that were executed (in order) and the final answer. Report the decisions as they were
EXECUTED in the code that produced the final answer (ignore abandoned attempts unless they changed the data).

Return ONLY a JSON object with exactly these keys:
- "framing.target": the exact column name used as outcome / label (string), or null.
- "framing.population": rows kept for the final analysis, as a short canonical phrase, e.g. "all rows",
  "drop rows with missing X", "dyslexia_bin == 1". Use "all rows" if no filter.
- "framing.unit": the unit of analysis, one or two words (e.g. "player", "course", "contest", "page view").
- "variables": sorted list of exact column names (or derived feature names) used as the main predictor(s) /
  features of the final model, excluding the target. For ML tasks with many features list them all.
- "transformation": sorted list of short canonical items for each data transformation, using the form
  "<operation>:<column or 'all'>", operations from: dropna, fillna_zero, fillna_mean, fillna_median,
  strip_suffix, to_numeric, replace_na_string, one_hot, label_encode, standardize, log, derive, interaction,
  drop_column, resample_oversample, resample_undersample, smote, class_weight, threshold_tuning, aggregate, other.
- "model.family": the family of the FINAL model behind the reported answer, one of: {families}.
- "model.all_families": sorted list of all model families fitted (same vocabulary).
- "model.controls": sorted list of exact column names included as covariates/controls besides the main
  predictor (for ML tasks with many features, use []).
- "validation": one of: {validations}.
- "metric": the metric(s) used to choose or report the final result, sorted list from: {metrics}.
- "verdict": for a research question, "supported", "rejected" or "inconclusive"; for a prediction task null.
- "final_selection": name of the final model or analysis reported (short string).
- "source_columns": object with keys "target", "variables", "controls"; each a sorted list of the ORIGINAL
  dataset columns (from the column list below) that the target / main variables / controls were built from.
- "silent": list of decision types above that the code made but the agent never announced in a DECISION line
  (compare with the stated decisions listed below).

Dataset columns: {columns}

Stated DECISION lines:
{stated}

Executed code cells:
{code}

Final answer:
{final}
"""


def extract_executed(llm, columns: list[str], code_cells: list[str], stated: list[dict], final: str | None,
                     max_code_chars: int = 30000) -> dict:
    code = "\n\n# ---- next cell ----\n".join(code_cells)
    if len(code) > max_code_chars:
        code = code[: max_code_chars // 3] + "\n# ... [middle cut] ...\n" + code[-2 * max_code_chars // 3:]
    stated_txt = "\n".join(f"- {d['type']}: {d['value']}" for d in stated) or "(none)"
    prompt = EXTRACTOR_PROMPT.format(families=", ".join(MODEL_FAMILIES), validations=", ".join(VALIDATIONS),
                                     metrics=", ".join(METRICS), columns=", ".join(columns), stated=stated_txt,
                                     code=code, final=final or "(no final answer)")
    reply = llm.chat([{"role": "user", "content": prompt}], max_tokens=4000, thinking=False)
    parsed = None
    m = re.search(r"\{.*\}", reply.content, re.S)
    if m:
        try:
            parsed = json.loads(m.group(0))
        except json.JSONDecodeError:
            parsed = None
    return {"executed": parsed, "raw": reply.content, "request": reply.request}


def canonical(dtype: str, value) -> str:
    """Hashable canonical form of one executed decision value, for agreement."""
    if value is None:
        return "null"
    if isinstance(value, list):
        return "|".join(sorted(str(v).strip().lower() for v in value))
    return str(value).strip().lower()
