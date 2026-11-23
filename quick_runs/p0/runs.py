"""P0: find old run folders and read what each run executed (read-only over /data/anon).

A "run" is one generation folder: it holds provenance/provenance.jsonl as a real file.
Scoring twins (…_p15, …_p15b, …_p16, …) only symlink the generation run's provenance, so
they are the same run and are skipped; the p15 twin's synthesized_prediction.json is used
as the run's final answer (the old project's standard synthesis prompt).
"""
from __future__ import annotations

import os
import csv
import json
import re
from pathlib import Path

LEGACY = Path(os.environ.get("REVERIES_LEGACY", "/data/anon"))  # pre-paper runs, read-only

SOURCES = {
    "v8_transfer": LEGACY / "lms/transfer",
    "v10_stage1": LEGACY / "debug/v10_stage1_20261013/results",
    "v10_stage2": LEGACY / "debug/v10_stage2_20261017/results",
    "e01": LEGACY / "debug/e01_experience_memory/results",
}
ARM_FLAGS_DIR = LEGACY / "lms_exp/runs"
HF_DB = Path.home() / ".cache/huggingface/hub/datasets--allenai--discoverybench/snapshots"
TWIN_RE = re.compile(r"_p1\d+b?$")

# Same ranking the old synthesis used (src/evaluation/discoverybench_eval.py,
# _rank_hypotheses_for_synthesis): rejected out unless all rejected; status rank, then confidence.
STATUS_RANK = {"supported": 0, "validated": 0, "weakened": 1, "tested": 1, "inconclusive": 2, "pending": 3,
               "proposed": 3}
TOP_HYPOTHESES = 4  # MAX_SYNTHESIS_FINDINGS in the old synthesis


def arm_of(source: str, run_name: str) -> str:
    for pre in ("hpc_gemma4_", "lambda01_gemma4_"):
        if run_name.startswith(pre):
            return f"{source}:{run_name[len(pre):]}"
    return f"{source}:{run_name}"


def discover() -> list[dict]:
    """Every generation run folder under the known sources (twins skipped)."""
    out = []
    for source, root in SOURCES.items():
        for task_dir in sorted(root.glob("discoverybench_train_*")):
            task = int(task_dir.name.rsplit("_", 1)[1])
            for rd in sorted(p for p in task_dir.iterdir() if p.is_dir()):
                prov = rd / "provenance" / "provenance.jsonl"
                twin = bool(TWIN_RE.search(rd.name)) or (rd / "provenance").is_symlink()
                out.append({"task": task, "source": source, "run_name": rd.name, "path": str(rd),
                            "arm": arm_of(source, rd.name), "twin": twin, "has_provenance": prov.is_file()})
    return out


def read_events(path: Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_run(run_dir: str) -> dict:
    """Facts of one run from its event log: goal, config, hypotheses, experiments (in order)."""
    rd = Path(run_dir)
    goal, config, dataset_path = "", {}, None
    hyps: dict[str, dict] = {}
    exps: list[dict] = []
    engineer_calls = []
    retrievals = []
    last_code = None
    for i, e in enumerate(read_events(rd / "provenance" / "provenance.jsonl")):
        et, md = e.get("event_type"), e.get("metadata") or {}
        if et == "orchestrator_initialized":
            config = md
            goal = goal or md.get("goal", "")
        elif et == "workflow_start":
            goal = goal or md.get("goal", "")
        elif et == "dataset_profiled":
            dataset_path = md.get("dataset_path")
        elif et == "memory_create" and md.get("memory_type") == "hypothesis":
            hyps[e["memory_id"]] = {"id": e["memory_id"], "content": md.get("content_summary", ""),
                                    "status": md.get("status", "pending"), "confidence": md.get("confidence", 0),
                                    "order": i}
        elif et == "memory_update" and e.get("memory_id") in hyps:
            up = md.get("updates", {})
            nested = up.get("metadata", {}) if isinstance(up.get("metadata"), dict) else {}
            st = up.get("status", nested.get("status"))
            cf = up.get("confidence", nested.get("confidence"))
            if up.get("content"):
                hyps[e["memory_id"]]["content"] = up["content"]
            if st is not None:
                hyps[e["memory_id"]]["status"] = st
            if cf is not None:
                hyps[e["memory_id"]]["confidence"] = cf
        elif et in ("experiment_complete", "experiment_failed"):
            stub = e.get("query") or ""
            full = last_code if last_code and _norm(last_code)[:300] == _norm(stub)[:300] else None
            exps.append({"order": i, "experiment_id": md.get("experiment_id"), "hypothesis_id": md.get("hypothesis_id"),
                         "success": et == "experiment_complete", "returncode": md.get("returncode"),
                         "code": full or stub, "code_full": full is not None, "output": e.get("response") or "",
                         "design_used": md.get("design_used")})
        elif et == "llm_call" and e.get("agent_name") == "Engineer":
            resp = e.get("response") or ""
            code = code_block(resp)
            if code:
                last_code = code
            engineer_calls.append({"order": i, "query": e.get("query") or "", "response": resp, "code": code})
        elif et == "retrieval":
            retrievals.append({"order": i, "agent": e.get("agent_name"), "ids": md.get("memory_ids") or [],
                               "type": md.get("memory_type_requested")})
    return {"goal": goal, "config": config, "dataset_path": dataset_path, "hypotheses": list(hyps.values()),
            "experiments": exps, "engineer_calls": engineer_calls, "retrievals": retrievals}


FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def code_block(text: str) -> str | None:
    """The longest fenced code block in an LLM reply (the script it wrote), or None."""
    blocks = FENCE_RE.findall(text or "")
    return max(blocks, key=len).strip() if blocks else None


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def rank_hypotheses(hyps: list[dict]) -> list[dict]:
    kept = [h for h in hyps if h.get("status") != "rejected"] or list(hyps)
    return sorted(kept, key=lambda h: (STATUS_RANK.get(h.get("status"), 9), -float(h.get("confidence") or 0)))


def final_code_cells(facts: dict, top: int = TOP_HYPOTHESES) -> list[str]:
    """Executed code behind the final answer: for each of the top-ranked hypotheses (the ones the
    old synthesis put first), the last experiment of that hypothesis that ran without error.
    Returned in execution order."""
    ok = [x for x in facts["experiments"] if x["success"] and x["code"]]
    last_by_h = {}
    for x in ok:
        last_by_h[x["hypothesis_id"]] = x
    chosen = []
    for h in rank_hypotheses(facts["hypotheses"]):
        if h["id"] in last_by_h:
            chosen.append(last_by_h[h["id"]])
        if len(chosen) == top:
            break
    if not chosen and ok:
        chosen = ok[-top:]
    return [x["code"] for x in sorted(chosen, key=lambda x: x["order"])]


def final_answer(run_dir: str) -> tuple[str | None, str]:
    """Final hypothesis text: the p15 twin's synthesis if present, else the run's own."""
    rd = Path(run_dir)
    for cand, label in ((rd.parent / f"{rd.name}_p15" / "synthesized_prediction.json", "p15_twin"),
                        (rd / "synthesized_prediction.json", "own")):
        if cand.is_file():
            d = json.loads(cand.read_text())
            return d.get("hypothesis"), label
    return None, "none"


def columns_for(task: int, run_dir: str, dataset_path: str | None) -> list[str]:
    gold = Path(run_dir).parent / "gold_task.json"
    if gold.is_file():
        meta = json.loads(gold.read_text())["dataset_meta"]["datasets"]
        cols = []
        for d in meta:
            c = d["columns"]
            for item in (c.get("raw", []) if isinstance(c, dict) else c):
                cols.append(item["name"])
        if cols:
            return cols
    if dataset_path and "discoverybench/" in dataset_path:
        rel = dataset_path.split("/discoverybench/", 1)[1]
        for snap in HF_DB.iterdir():
            p = snap / "discoverybench" / rel
            if p.is_file():
                with open(p) as f:
                    return next(csv.reader(f))
    return []


def arm_flags(run_name: str) -> str:
    p = ARM_FLAGS_DIR / f"{run_name}.arm.json"
    if p.is_file():
        return json.loads(p.read_text()).get("flags", "")
    return ""
