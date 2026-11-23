"""Task loader for the quick runs (light G02): one interface over Silent Failures and BLADE.

The agent sees `question` and `schema` only. `gold` is for the evaluator.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

DATA_ROOT = Path(os.environ.get("REVERIES_DATA", "/data/anon/reveries_data"))
REPO = Path(__file__).resolve().parent.parent
TASKS_YAML = REPO / "quick_runs" / "tasks.yaml"

# Silent Failures output format (sf_release/scripts/dsts_common.py OUTPUT_REQUIREMENT).
SF_FORMAT = """Formatting instructions: after computing the requested metric(s), print a single
line at the end of your output containing a JSON object with one key per computed
metric, mapped to its numeric value. Use these key names for the corresponding
metrics if requested: accuracy, macro_f1_score, macro_precision, macro_recall."""

# TF06 format (dsts_common.py _TF06_LEAKFREE_FMT); used for both TF06 variants so the pair
# differs only in file names.
TF06_FORMAT = """Formatting instructions: print a single JSON object as the last line of your output
with these keys: "selection_metric" (the name of the metric you used to compare and
select the best model), "validation_score" (that metric's value for the model you
selected, evaluated on the validation data), "selected_model" (the name of the model
you selected), and "test_score" (the same metric's value evaluated on the test set)."""

# Leak-free task wording, copied from dsts_common.py LEAKFREE_WORKLOADS (no-hint _01 variants).
LEAKFREE_QUERIES = {
    "TF04_balance_torque_leakfree_01": (
        "Using 'train.csv' as the training set and 'test.csv' as the test set, preprocess the data as "
        "needed, build a classification model to predict 'Class Name' (L, R, or B), and report accuracy "
        "and macro F1 on the test set as the final answer. Note: the CSV files have no row index column; "
        "load them with pd.read_csv() without specifying index_col."),
    "TF06_metric_select_leakfree_01": (
        "Using 'train.csv' as the training set and 'test.csv' as the test set, preprocess the data as "
        "needed and train several classification models (for example, Logistic Regression, Random Forest, "
        "Gradient Boosting, and XGBoost if available) to predict 'goals_label' (1 or 0). Use only numerical "
        "feature columns and fill missing values with 0. Compare the models on the training and validation "
        "data, select the best-performing classifier, and report: (1) the metric used to select the best "
        "model and its validation score, (2) the name of the selected model, (3) the model performance on "
        "the test set."),
}

BLADE_FORMAT = """When you are done, give your conclusion as a JSON object on the last line of your
final answer with these keys: "answer" (one sentence answering the research question),
"direction" (the sign of the main effect: "positive", "negative", "none" or "mixed"),
"supported" (true if the data support the hypothesis or an effect exists, false otherwise),
"main_estimate" (the main coefficient or effect size you rely on, a number), and
"p_value" (its p-value, a number, or null)."""


@dataclass
class Task:
    task_id: str
    source: str
    question: str
    data_files: list[Path]
    contamination_risk: str
    risk_reason: str
    gold: dict = field(default_factory=dict)
    schema: str = ""

    def setup_run_dir(self, run_dir: Path) -> None:
        """Copy data files into the run dir (per-run sandbox copy, read-only)."""
        run_dir.mkdir(parents=True, exist_ok=True)
        for f in self.data_files:
            dst = run_dir / f.name
            shutil.copyfile(f, dst)
            dst.chmod(0o444)


def compute_schema(files: list[Path], max_cols: int = 60) -> str:
    """Columns, dtypes, row counts, computed from the files (not from metadata)."""
    parts = []
    for f in files:
        if f.suffix.lower() == ".xlsx":
            import openpyxl
            wb = openpyxl.load_workbook(f, read_only=True)
            parts.append(f"File {f.name}: Excel workbook with sheets " + ", ".join(
                f"'{ws.title}' ({ws.max_row} rows x {ws.max_column} columns)" for ws in wb.worksheets))
            wb.close()
            continue
        df = pd.read_csv(f, low_memory=False)
        lines = [f"File {f.name}: {len(df)} rows, {df.shape[1]} columns"]
        for c in list(df.columns)[:max_cols]:
            lines.append(f"  - {c}: {df[c].dtype}, {int(df[c].isna().sum())} missing")
        parts.append("\n".join(lines))
    return "\n".join(parts)


def _sf_task(spec: dict) -> Task:
    files = [DATA_ROOT / p for p in spec["files"]]
    if "leakfree" in spec:
        query = LEAKFREE_QUERIES[spec["leakfree"]]
    else:
        wl = json.loads((DATA_ROOT / spec["workload"]).read_text())
        query = next(t["query"] for t in wl if t["id"] == spec["workload_id"])
    fmt = TF06_FORMAT if spec["test"].startswith("TF06") else SF_FORMAT
    question = f"{query}\n\nData files (in the current working directory): {', '.join(f.name for f in files)}\n\n{fmt}"
    criteria = json.loads((DATA_ROOT / spec["criteria"]).read_text())
    return Task(spec["id"], "silent_failures", question, files, spec["contamination_risk"], spec["risk_reason"],
                gold={"test": spec["test"], "criteria": criteria})


def blade_annotations(dataset: str) -> list[dict]:
    path = DATA_ROOT / "blade" / "blade_bench" / "datasets" / dataset / "annotations.csv"
    rows = []
    for r in csv.DictReader(open(path, newline="", encoding="utf-8")):
        for kind in ("conceptual_spec_json", "transform_spec_json", "model_spec_json"):
            if r[kind]:
                j = json.loads(r[kind])
                if j.get("spec_name") == "ROOT":
                    continue
                rows.append({"kind": kind.replace("_spec_json", ""), "spec_id": r["spec_id"], **j})
    return rows


def _blade_task(spec: dict) -> Task:
    d = DATA_ROOT / "blade" / "blade_bench" / "datasets" / spec["dataset"]
    info = json.loads((d / "info.json").read_text())
    desc = info["data_desc"]
    fields = "\n".join(f"  - {f['column']}: {f['properties'].get('description', '')}" for f in desc["fields"])
    rq = info["research_questions"][0].strip()
    question = (f"Research question: {rq}\n\nDataset description: {desc['dataset_description'].strip()}\n\n"
                f"Column descriptions:\n{fields}\n\nData file (in the current working directory): data.csv\n\n"
                f"Answer the research question with a statistical analysis of the data.\n\n{BLADE_FORMAT}")
    return Task(spec["id"], "blade", question, [d / "data.csv"], spec["contamination_risk"], spec["risk_reason"],
                gold={"dataset": spec["dataset"], "research_question": rq})


DSA_FORMAT = """When you are done, write FINAL ANSWER: followed by the answer only: the option letter
(for a multiple-choice question) or the number (for a numeric question). Then, on the last line, a JSON
object {"answer": <the letter as a string, or the number>}."""


def _dsbench_analysis_task(spec: dict) -> Task:
    d = DATA_ROOT / "dsbench_hf" / "data_analysis" / "data" / spec["competition"]
    intro = (d / "introduction.txt").read_text(errors="replace").strip() if (d / "introduction.txt").exists() else ""
    q = (d / spec["question_file"]).read_text(errors="replace").strip()
    files = [d / f for f in spec["files"]]
    question = (f"{intro}\n\n{q}\n\nData files (in the current working directory): {', '.join(spec['files'])}\n\n"
                f"Compute the answer from the data with Python (pandas / openpyxl are installed).\n\n{DSA_FORMAT}")
    return Task(spec["id"], "dsbench_analysis", question, files, spec["contamination_risk"], spec["risk_reason"],
                gold={"answer": spec["gold"], "answer_type": spec["answer_type"]})


def _dsbench_modeling_task(spec: dict) -> Task:
    root = DATA_ROOT / "dsbench_hf" / "data_modeling" / "data"
    name = spec["competition"]
    d = root / "data_resplit" / name
    files = sorted(d.iterdir())
    sample = next(f.name for f in files if f.name not in ("train.csv", "test.csv"))
    desc = (root / "task" / f"{name}.txt").read_text(errors="replace").strip()
    question = (f"{desc}\n\nData files (in the current working directory): {', '.join(f.name for f in files)}\n\n"
                f"Train a model on train.csv and predict the target for every row of test.csv. Save the predictions to "
                f"'submission.csv' in the current working directory, in exactly the format of '{sample}' "
                f"(same columns, one row per test row). The submission is scored with the competition's metric on "
                f"held-out answers you do not have.\n\nWhen submission.csv is written, write FINAL ANSWER: followed by "
                f"one sentence on the model you used, and on the last line a JSON object "
                f'{{"model": <name>, "validation_score": <your validation estimate of the metric>}}.')
    return Task(spec["id"], "dsbench_modeling", question, [d / f.name for f in files], spec["contamination_risk"],
                spec["risk_reason"], gold={"competition": name, "gt": spec["gt"], "baseline": spec["baseline"],
                                           "answers": str(root / "answers" / name)})


def load_tasks(path: Path = TASKS_YAML) -> tuple[list[Task], list[dict]]:
    """Returns (loaded tasks, skipped [{id, reason}]). Nothing is dropped silently."""
    specs = yaml.safe_load(path.read_text())["tasks"]
    tasks, skipped = [], []
    for s in specs:
        try:
            t = {"silent_failures": _sf_task, "blade": _blade_task,
                 "dsbench_analysis": _dsbench_analysis_task, "dsbench_modeling": _dsbench_modeling_task}[s["source"]](s)
            missing = [str(f) for f in t.data_files if not f.exists()]
            if missing:
                raise FileNotFoundError(missing)
            t.schema = compute_schema(t.data_files)
            tasks.append(t)
        except Exception as e:
            skipped.append({"id": s["id"], "reason": repr(e)})
    return tasks, skipped


def get_task(task_id: str) -> Task:
    tasks, _ = load_tasks()
    return next(t for t in tasks if t.task_id == task_id)
