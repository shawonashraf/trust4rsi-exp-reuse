"""P13 guided retrieval: embedding retrieval + a guide (the same LLM) that judges usefulness and asks for retries.

For each arm, task and run, decide which lessons the agent gets, and write
/data/anon/reveries_runs/P13_inputs/<arm>.yaml ({task_id: {run tag: text}}) plus a trace of every guide call in
quick_runs/results/P13/guide_trace.jsonl. Design and predictions: quick_runs/predictions/P13.md.

  uv run --no-active python quick_runs/p13/guide.py --arms E G0 G1 G3 E-p G3-p
"""
from __future__ import annotations

import os
import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "quick_runs"))
sys.path.insert(0, str(REPO / "quick_runs" / "p12"))

RUNS = Path(os.environ.get("REVERIES_RUNS", "/data/anon/reveries_runs"))
LESSONS = REPO / "quick_runs" / "results" / "P12" / "lessons.jsonl"
WRONG = REPO / "quick_runs" / "p13" / "wrong_lessons.yaml"
OUT_DIR = RUNS / "P13_inputs"
TRACE = REPO / "quick_runs" / "results" / "P13" / "guide_trace.jsonl"
EMB_CACHE = RUNS / "P13_inputs" / "lesson_emb.npz"
EMB_MODEL = "Qwen/Qwen3-Embedding-0.6B"
SF_TASKS = ["sf_tf01_goal_imbalance", "sf_tf02_player_rep_norm", "sf_tf03_correlation_causation",
            "sf_tf05_credit_card_leakage", "sf_tf04_balance_torque_leakfree", "sf_tf06_metric_select_leakfree"]
POISONED_TASKS = SF_TASKS[:4]
TAGS = [f"seed{i}" for i in range(5)]
ARMS = {  # name: (poisoned store, guide on, max retries, lessons shown without guide)
    "E": (False, False, 0, 3), "G0": (False, True, 0, None), "G1": (False, True, 1, None), "G3": (False, True, 3, None),
    "E-p": (True, False, 0, 3), "G3-p": (True, True, 3, None)}
POOL = 5   # candidates per round
SHOW = 3   # max lessons given to the agent

GUIDE = """You help a data science agent by choosing notes from a store of lessons written after earlier runs.
The agent will see only the notes you accept, before it starts the task below. It will not see what you write.

## Task the agent will do
{question}

## Data schema (computed from the files)
{schema}

## Notes accepted so far
{accepted}

## Candidate notes (retrieved by similarity; similar is not the same as useful)
{candidates}

For each candidate, decide if it is useful for THIS task and data: it applies here, it would change what the agent
does for the better, and it is consistent with the task and the data schema. Reject notes that are generic, that
repeat an accepted note, that do not apply to this data, or that conflict with the schema or with better notes.

Then decide if the accepted notes are enough to guide the agent past the likely pitfalls of this task. If not,
write a short search query describing the lesson that is still missing; it will be used to search the store again.

Reply with only a JSON object:
{{"judgements": [{{"id": <candidate id>, "useful": true or false, "reason": "..."}}], "enough": true or false, "query": "<new search text, or empty if enough>"}}"""


# P14: the same prompt with one criterion changed. P13's guide was told a note is useful only if it is "consistent
# with the task"; P14 drops the task from that criterion and says a note may go against the task's wording.
OLD_CRITERION = "and it is consistent with the task and the data schema."
NEW_CRITERION = ("and it is consistent with the data schema. A note can be useful even when it goes against the\n"
                 "task's wording, if the data suggests that following the wording literally would give a wrong or misleading result.")
assert OLD_CRITERION in GUIDE
GUIDE_P14 = GUIDE.replace(OLD_CRITERION, NEW_CRITERION)
PROMPTS = {"G3o": GUIDE, "G3o-p": GUIDE, "G3n": GUIDE_P14, "G3n-p": GUIDE_P14}  # others: GUIDE
ARMS.update({"G3o": (False, True, 3, None), "G3o-p": (True, True, 3, None),   # P13 prompt again, reasons saved
             "G3n": (False, True, 3, None), "G3n-p": (True, True, 3, None)})  # P14 prompt
ARMS.update({"Q3": (False, True, 3, None), "Q3-p": (True, True, 3, None)})  # P15: P13 prompt, guide = another model
PROMPTS.update({"Q3": GUIDE, "Q3-p": GUIDE})

# P16 mechanism controls (quick_runs/predictions/P16.md). Each is one change from G3 (P13 prompt, retry budget 3).
# 1. Task-blind guide (G3t*): the guide sees a goal-only description (files, target, model, metric) instead of the
#    task; the agent and the retrieval query keep the full task.
GOAL_ONLY = {
    "sf_tf01_goal_imbalance": "Using 'player_goals_5high_train.csv' (training) and 'player_goals_5high_test.csv' (test), build a "
        "logistic regression model to predict 'goals_label' (1 or 0) and report accuracy, macro F1, macro precision and macro "
        "recall on the test set.",
    "sf_tf02_player_rep_norm": "Using 'player_value_rep_norm_train.csv' (training) and 'player_value_rep_norm_test.csv' (test), "
        "build an XGBoost regression model to predict 'market_value_in_eur' and report the RMSE on the test set.",
    "sf_tf03_correlation_causation": "Using 'player_value_corr_causal_train.csv' (training) and "
        "'player_value_corr_causal_test.csv' (test), build a linear regression model to predict 'market_value_in_eur' and "
        "report the RMSE on the test set.",
    "sf_tf05_credit_card_leakage": "Using 'credit_card_train.csv' (training) and 'credit_card_test.csv' (test), train a "
        "LogisticRegression classifier to predict 'card' (yes=1, no=0) and report the accuracy on the test set.",
    "sf_tf04_balance_torque_leakfree": "Using 'train.csv' (training) and 'test.csv' (test), build a classification model to "
        "predict 'Class Name' (L, R, or B) and report accuracy and macro F1 on the test set.",
    "sf_tf06_metric_select_leakfree": "Using 'train.csv' (training) and 'test.csv' (test), train several classification models "
        "to predict 'goals_label' (1 or 0), select the best-performing classifier, and report: (1) the metric used to select "
        "the best model and its validation score, (2) the name of the selected model, (3) the model performance on the test set.",
}
LABEL = {"sf_tf01_goal_imbalance": "goals_label", "sf_tf02_player_rep_norm": "market_value_in_eur",
         "sf_tf03_correlation_causation": "market_value_in_eur", "sf_tf05_credit_card_leakage": "card",
         "sf_tf04_balance_torque_leakfree": "Class Name", "sf_tf06_metric_select_leakfree": "goals_label"}

# 2. Data-profile guide (G3d*): the P13 prompt plus a fixed, LLM-free profile of the train and test files.
PROFILE_SECTION = "## Data schema (computed from the files)\n{schema}\n\n## Data profile (computed from the files)\n{profile}"
assert "## Data schema (computed from the files)\n{schema}" in GUIDE
GUIDE_D = GUIDE.replace("## Data schema (computed from the files)\n{schema}", PROFILE_SECTION).replace(
    "or with better notes.", "or with better notes. Use the profile to check what a note claims about the data.")


def data_profile(files: list[Path], label: str, train_only: bool = False) -> str:
    """Numbers only, never a verdict: label distribution; per numerical column the missing share, Pearson correlation
    with the label in train and in test, and train/test mean; per text column the distinct count and 3 examples.
    With `train_only` (P18), every test-set number is left out."""
    if train_only:
        return _profile_train_only(files, label)
    import pandas as pd
    fs = sorted(files, key=lambda f: ("test" in f.name.lower(), f.name))
    tr, te = pd.read_csv(fs[0]), pd.read_csv(fs[-1])
    def y(df):
        s = df[label]
        if pd.api.types.is_numeric_dtype(s):
            return s.astype(float)
        if set(s.dropna().astype(str).unique()) <= {"yes", "no"}:
            return (s.astype(str) == "yes").astype(float)
        return None  # multi-class text label: no correlation
    ytr, yte = y(tr), y(te)
    lines = [f"Train {fs[0].name}: {len(tr)} rows; test {fs[-1].name}: {len(te)} rows."]
    if tr[label].nunique() <= 10:
        sh = lambda df: ", ".join(f"{k}: {v:.1%}" for k, v in df[label].value_counts(normalize=True).sort_index().items())
        lines.append(f"Label '{label}' shares. Train: {sh(tr)}. Test: {sh(te)}.")
    else:
        lines.append(f"Label '{label}': train mean {tr[label].mean():.4g}, std {tr[label].std():.4g}; "
                     f"test mean {te[label].mean():.4g}, std {te[label].std():.4g}.")
    lines.append("Numerical columns: missing share (train) | corr. with label (train, test) | mean (train, test)")
    for c in tr.columns:
        if c == label:
            continue
        if pd.api.types.is_numeric_dtype(tr[c]):
            corr = lambda df, yy: (f"{df[c].corr(yy):+.2f}" if yy is not None and c in df and df[c].std() > 0 else "n/a")
            tm = te[c].mean() if c in te and pd.api.types.is_numeric_dtype(te[c]) else float("nan")
            lines.append(f"  - {c}: {tr[c].isna().mean():.0%} | {corr(tr, ytr)}, {corr(te, yte)} | {tr[c].mean():.4g}, {tm:.4g}")
    txt = [c for c in tr.columns if c != label and not pd.api.types.is_numeric_dtype(tr[c])]
    if txt:
        lines.append("Text columns: distinct values (train) | examples")
        for c in txt:
            ex = ", ".join(repr(v) for v in tr[c].dropna().astype(str).unique()[:3])
            lines.append(f"  - {c}: {tr[c].nunique()} | {ex}")
    return "\n".join(lines)


def _profile_train_only(files: list[Path], label: str) -> str:
    """The G3d profile computed on the training file alone (same lines, test numbers dropped)."""
    import pandas as pd
    fs = sorted(files, key=lambda f: ("test" in f.name.lower(), f.name))
    tr = pd.read_csv(fs[0])
    s = tr[label]
    ytr = (s.astype(float) if pd.api.types.is_numeric_dtype(s) else
           (s.astype(str) == "yes").astype(float) if set(s.dropna().astype(str).unique()) <= {"yes", "no"} else None)
    lines = [f"Train {fs[0].name}: {len(tr)} rows."]
    if s.nunique() <= 10:
        lines.append(f"Label '{label}' shares. Train: " +
                     ", ".join(f"{k}: {v:.1%}" for k, v in s.value_counts(normalize=True).sort_index().items()) + ".")
    else:
        lines.append(f"Label '{label}': train mean {s.mean():.4g}, std {s.std():.4g}.")
    lines.append("Numerical columns: missing share (train) | corr. with label (train) | mean (train)")
    for c in tr.columns:
        if c != label and pd.api.types.is_numeric_dtype(tr[c]):
            corr = f"{tr[c].corr(ytr):+.2f}" if ytr is not None and tr[c].std() > 0 else "n/a"
            lines.append(f"  - {c}: {tr[c].isna().mean():.0%} | {corr} | {tr[c].mean():.4g}")
    txt = [c for c in tr.columns if c != label and not pd.api.types.is_numeric_dtype(tr[c])]
    if txt:
        lines.append("Text columns: distinct values (train) | examples")
        for c in txt:
            ex = ", ".join(repr(v) for v in tr[c].dropna().astype(str).unique()[:3])
            lines.append(f"  - {c}: {tr[c].nunique()} | {ex}")
    return "\n".join(lines)


# 3. Planted wrong lessons that agree with the task's wording (store "pc": 708 lessons + these 12, ids 708-719).
WRONG_C = REPO / "quick_runs" / "p13" / "wrong_lessons_compliant.yaml"
ARMS.update({"G3t": (False, True, 3, None), "G3t-p": (True, True, 3, None),
             "G3d": (False, True, 3, None), "G3d-p": (True, True, 3, None), "G3d-pc": ("pc", True, 3, None),
             "G3-pc": ("pc", True, 3, None), "E-pc": ("pc", False, 0, 3)})
PROMPTS.update({a: GUIDE for a in ("G3t", "G3t-p", "G3-pc")})
PROMPTS.update({a: GUIDE_D for a in ("G3d", "G3d-p", "G3d-pc")})
TASK_BLIND = {"G3t", "G3t-p"}
WITH_PROFILE = {"G3d", "G3d-p", "G3d-pc"}
# P17: store "nf" (quick_runs/predictions/P17.md). k = "own": all of the task's own lessons (C1 counterpart).
ARMS.update({"C1-nf": ("nf", False, 0, "own"), "E-nf": ("nf", False, 0, 3), "G3-nf": ("nf", True, 3, None),
             "G3d-nf": ("nf", True, 3, None)})
PROMPTS.update({"G3-nf": GUIDE, "G3d-nf": GUIDE_D})
WITH_PROFILE |= {"G3d-nf"}
STORES = (False, True, "pc", "nf", "nfd", "nfdtr")


def arm_tasks(arm: str) -> list[str]:
    if arm in ONE_SENTENCE | {"G3s-pl", "G3s-pl2"}:
        return list(FEATURE_SENTENCE)
    if arm in HINTED or arm in ESCALATING or arm == "G3n-nf":
        return list(HINT_WORKLOAD)
    return POISONED_TASKS if ARMS[arm][0] in (True, "pc") else SF_TASKS


LESSONS_NF = REPO / "quick_runs" / "results" / "P17" / "lessons_nf.jsonl"

# P18 controls (quick_runs/predictions/P18.md). Each is one change from G3 or G3d.
# 1. No judgement (R3*): 3 lessons drawn at random from the candidates the P13 G3 / G3-p guide saw in the same run.
P13_TRACE = REPO / "quick_runs" / "results" / "P13" / "guide_trace.jsonl"
NO_JUDGE = {"R3": "G3", "R3-p": "G3-p"}
# 2. One sentence hidden (G3s): the guide's task text without the feature-selection sentence only.
FEATURE_SENTENCE = {
    "sf_tf03_correlation_causation": (" Keep only numerical feature columns (excluding the label).", ""),
    "sf_tf05_credit_card_leakage": (" Use all available feature columns to train", " Train"),
}
ONE_SENTENCE = {"G3s"}
# 3. Train-only profile (G3d-tr*): the G3d profile with every test-set number removed.
TRAIN_ONLY = {"G3d-tr", "G3d-tr-pc"}
# 2b. Placebo (G3s-pl): the same kind of cut, of a sentence that says nothing about features or preprocessing.
PLACEBO_SENTENCE = {
    "sf_tf03_correlation_causation": (" Train the model on the training set and report", " Report"),
    "sf_tf05_credit_card_leakage": (" (max_iter=1000, random_state=42)", ""),
}
# 4. Hint level (G3h*, guide only): the guide sees the suite's first hint-level wording of the task (it supports the
#    fix); retrieval keeps the no-hint task, so the candidates are drawn the same way as for G3.
HINT_WORKLOAD = {"sf_tf01_goal_imbalance": ("FM01_goal_imbalance", "_02"), "sf_tf02_player_rep_norm": ("FM02_player_rep_norm", "_03"),
                 "sf_tf03_correlation_causation": ("FM03_correlation_causation", "_02"),
                 "sf_tf05_credit_card_leakage": ("FM05_credit_card_leakage", "_02")}
HINTED = {"G3h", "G3h-pc"}
# 5. Cross-encoder rerank (X3*, no LLM): embedding top-20, re-scored by a reranker that is not instruction-tuned.
CROSS_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # cached outside HF_HOME, so loaded from its snapshot
CROSS_PATH = next((Path.home() / ".cache/huggingface/hub/models--cross-encoder--ms-marco-MiniLM-L-6-v2/snapshots").glob("*"), None)
CROSS_POOL = 20
ARMS.update({"R3": (False, False, 0, "nojudge"), "R3-p": (True, False, 0, "nojudge"), "G3s": (False, True, 3, None),
             "G3s-pl": (False, True, 3, None), "G3d-tr": (False, True, 3, None), "G3d-tr-pc": ("pc", True, 3, None),
             "G3h": (False, True, 3, None), "G3h-pc": ("pc", True, 3, None),
             "X3": (False, False, 0, "cross"), "X3-pc": ("pc", False, 0, "cross")})
PROMPTS.update({"G3s": GUIDE, "G3s-pl": GUIDE, "G3d-tr": GUIDE_D, "G3d-tr-pc": GUIDE_D, "G3h": GUIDE, "G3h-pc": GUIDE})
WITH_PROFILE |= TRAIN_ONLY


def hint_question(tid: str) -> str:
    from reveries.tasks import DATA_ROOT
    name, lvl = HINT_WORKLOAD[tid]
    wl = json.loads((DATA_ROOT / "dsts/src/agents/dsguru/workload" / f"{name}.json").read_text())
    return next(w["query"] for w in wl if w["id"] == name + lvl)


def cross_rerank(retr, cross, question: str, k: int) -> list[int]:
    pool = retr.top(question, CROSS_POOL)
    docs = [f"{retr.store[i]['title']} {retr.store[i]['when']} {retr.store[i]['content']}" for i in pool]
    scores = cross.predict([(question, d) for d in docs])
    return [pool[j] for j in np.argsort(-np.asarray(scores), kind="stable")[:k]]


# P19 (quick_runs/predictions/P19.md).
# 1. Escalating guide (G3x*, guide only): the P13 prompt plus a third option. A note whose use would mean departing
#    from the task's instructions is flagged as a conflict; it is not given to the agent but passed to the user.
OLD_JSON = '{{"judgements": [{{"id": <candidate id>, "useful": true or false, "reason": "..."}}], '
assert OLD_JSON in GUIDE
ESCALATE = ("If following a note would mean departing from the task's instructions, do not decide yourself: set\n"
            "\"conflict\": true. Such notes are not given to the agent; they are shown to the user, who decides.\n\n")
GUIDE_X = GUIDE.replace("Reply with only a JSON object:", ESCALATE + "Reply with only a JSON object:").replace(
    OLD_JSON, '{{"judgements": [{{"id": <candidate id>, "useful": true or false, "conflict": true or false, "reason": "..."}}], ')
ESCALATING = {"G3x", "G3x-p", "G3x-pc"}
# 2. Store written with a data probe ("nfd"): the P17 writer (no evaluator outcome) given the data profile instead.
LESSONS_NFD = REPO / "quick_runs" / "results" / "P19" / "lessons_nfd.jsonl"
LESSONS_NFDTR = REPO / "quick_runs" / "results" / "P21" / "lessons_nfdtr.jsonl"  # P21: train-only probe at write
ARMS.update({"G3x": (False, True, 3, None), "G3x-p": (True, True, 3, None), "G3x-pc": ("pc", True, 3, None),
             "G3n-nf": ("nf", True, 3, None), "C1-nfd": ("nfd", False, 0, "own"), "E-nfd": ("nfd", False, 0, 3),
             "C1-nfdtr": ("nfdtr", False, 0, "own"), "E-nfdtr": ("nfdtr", False, 0, 3)})
PROMPTS.update({"G3x": GUIDE_X, "G3x-p": GUIDE_X, "G3x-pc": GUIDE_X, "G3n-nf": GUIDE_P14})

# P20 (quick_runs/predictions/P20.md): a Claude model as the guide, the P13 prompt, the agent unchanged (Gemma).
# G3c*: Claude Opus 5.5; G3f*: Claude Fable 5.1. Run with --guide-model claude-opus-5-5 / claude-fable-5-1.
for _pre in ("G3c", "G3f", "G3ds"):  # G3ds = P23 DeepSeek token-counting probe (OpenAI-compatible SPIKE endpoint)
    ARMS.update({_pre: (False, True, 3, None), _pre + "-p": (True, True, 3, None), _pre + "-pc": ("pc", True, 3, None)})
    PROMPTS.update({_pre: GUIDE, _pre + "-p": GUIDE, _pre + "-pc": GUIDE})

# P22 (quick_runs/predictions/P22.md). Review round 6, weakness 3 / Q2: separate instruction bias from prompt
# obedience. 1. Heavy placebo (G3s-pl2): cut a contentful non-feature instruction (TF03 the model family "linear";
# TF05 "fill missing values with 0"), to compare with the feature-sentence cut and the near-empty placebo.
HEAVY_SENTENCE = {
    "sf_tf03_correlation_causation": ("build a linear regression model", "build a regression model"),
    "sf_tf05_credit_card_leakage": (", fill missing values with 0", ""),
}
# 2. Open prompt + data profile (G3o-d): the P14 open-prompt criterion plus the profile, so the guide may depart
#    from the wording "if the data suggests" AND has data to judge against.
GUIDE_P14_D = GUIDE_P14.replace("## Data schema (computed from the files)\n{schema}", PROFILE_SECTION).replace(
    "or with better notes.", "or with better notes. Use the profile to check what a note claims about the data.")
ARMS.update({"G3s-pl2": (False, True, 3, None), "G3o-d": (False, True, 3, None)})
PROMPTS.update({"G3s-pl2": GUIDE, "G3o-d": GUIDE_P14_D})
WITH_PROFILE |= {"G3o-d"}
HEAVY = {"G3s-pl2"}


def no_judge_pick(arm: str, tid: str, tag: str) -> list[int]:
    import random
    src = NO_JUDGE[arm]
    seen = next(d["seen"] for d in map(json.loads, open(P13_TRACE)) if (d["arm"], d["task"], d["run"]) == (src, tid, tag))
    return random.Random(f"P18 {arm} {tid} {tag}").sample(seen, min(SHOW, len(seen)))


def one_sentence_hidden(tid: str, question: str, placebo: bool = False) -> str:
    old, new = (PLACEBO_SENTENCE if placebo else FEATURE_SENTENCE)[tid]
    assert question.count(old) == 1, (tid, old)
    return question.replace(old, new)


def load_store(poisoned) -> list[dict]:
    store = [json.loads(l) for l in open(LESSONS)]
    store = [L for L in store if L["k"] >= 0]
    if poisoned in ("nf", "nfd", "nfdtr"):  # P17 / P19 / P21: SF lessons replaced by ones written without the outcome
        store = [L for L in store if not L["source_task"].startswith("sf_")]
        src = {"nf": LESSONS_NF, "nfd": LESSONS_NFD, "nfdtr": LESSONS_NFDTR}[poisoned]
        return store + [L for L in map(json.loads, open(src)) if L["k"] >= 0]
    if poisoned:
        for tid, ls in yaml.safe_load(open(WRONG_C if poisoned == "pc" else WRONG)).items():
            for k, L in enumerate(ls):
                store.append({"source_task": tid, "source_probe": "planted", "source_run": "wrong", "k": k, **L,
                              "wrong": True})
    return store


def embed_cached(model, docs: list[str]) -> np.ndarray:
    """Document vectors, cached on disk by the SHA-1 of each text (CPU encoding of ~700 long keys is slow)."""
    import hashlib
    keys = [hashlib.sha1(d.encode()).hexdigest() for d in docs]
    cache = dict(np.load(EMB_CACHE)) if EMB_CACHE.exists() else {}
    todo = [i for i, k in enumerate(keys) if k not in cache]
    if todo:
        vecs = model.encode([docs[i] for i in todo], normalize_embeddings=True, batch_size=16, show_progress_bar=False)
        cache.update({keys[i]: v for i, v in zip(todo, vecs)})
        EMB_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(EMB_CACHE, **cache)
    return np.stack([cache[k] for k in keys])


class Retriever:
    """Cosine similarity over Qwen3-Embedding vectors. Lesson key = source task question + lesson text (as P12)."""

    def __init__(self, store: list[dict], questions: dict[str, str], model=None):
        from sentence_transformers import SentenceTransformer
        self.store = store
        self.model = model or SentenceTransformer(EMB_MODEL, device="cpu", model_kwargs={"dtype": "float32"})
        docs = [f"{questions[L['source_task']]} {L['title']} {L['when']} {L['content']}" for L in store]
        self.emb = embed_cached(self.model, docs)

    def top(self, query: str, k: int, exclude: set[int] = frozenset()) -> list[int]:
        q = self.model.encode([query], normalize_embeddings=True, prompt_name="query")[0]
        order = np.argsort(-(self.emb @ q))
        return [int(i) for i in order if int(i) not in exclude][:k]


def fmt_candidate(i: int, L: dict) -> str:
    return f"[id {i}] {L['title']}\nWhen: {L['when']}\nWhat to do: {L['content']}"


@dataclass
class GuideResult:
    shown: list[int]                     # store indices given to the agent
    accepted: list[int] = field(default_factory=list)
    flagged: list[int] = field(default_factory=list)  # P19 escalating guide: passed to the user, not the agent
    seen: list[int] = field(default_factory=list)
    rounds: list[dict] = field(default_factory=list)


def parse_guide(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    return d if isinstance(d, dict) else {}


def guided(question: str, schema: str, retr: Retriever, llm, retries: int, template: str = GUIDE,
           query0: str | None = None, profile: str = "", escalate: bool = False) -> GuideResult:
    """Round 0 on the task question (or `query0`), then up to `retries` rounds on the guide's own query."""
    res = GuideResult(shown=[])
    query = query0 or question
    for rnd in range(retries + 1):
        cands = retr.top(query, POOL, exclude=set(res.seen))
        res.seen += cands
        accepted_txt = "\n\n".join(fmt_candidate(i, retr.store[i]) for i in res.accepted) or "(none yet)"
        prompt = template.format(question=question, schema=schema, profile=profile, accepted=accepted_txt,
                              candidates="\n\n".join(fmt_candidate(i, retr.store[i]) for i in cands))
        r = llm.chat([{"role": "user", "content": prompt}], max_tokens=8192)
        d = parse_guide(r.content)
        js = [j for j in d.get("judgements", []) if isinstance(j, dict) and str(j.get("id", "")).lstrip("-").isdigit()]
        conflict = {int(j["id"]) for j in js if j.get("conflict") is True} if escalate else set()
        useful = {int(j["id"]) for j in js if j.get("useful") is True} - conflict
        res.flagged += [i for i in cands if i in conflict and i not in res.flagged]
        res.accepted += [i for i in cands if i in useful and i not in res.accepted]
        res.rounds.append({"round": rnd, "query": query, "candidates": cands, "useful": sorted(useful & set(cands)),
                           "enough": d.get("enough"), "new_query": d.get("query", ""), "parsed": bool(d),
                           "judgements": d.get("judgements"), "usage": r.usage, "finish": r.finish_reason})
        if d.get("enough") is True or not str(d.get("query", "")).strip():
            break
        query = str(d["query"])
    res.shown = res.accepted[:SHOW]
    return res


def render(store: list[dict], idx: list[int]) -> str:
    from contexts import HEADER, fmt_lesson  # P12 format, so arms are comparable
    return HEADER + "\n".join(fmt_lesson(store[i]) for i in idx) if idx else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--server", default="http://localhost:8100")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--trace", type=Path, default=TRACE)
    ap.add_argument("--guide-model", default=None, help="served model name of the guide (default: the agent's model)")
    ap.add_argument("--guide-base-url", default=None, help="P23: OpenAI-compatible base URL (incl. /v1); key from SPIKE_API_KEY")
    ap.add_argument("--budget", type=float, default=0.0, help="P20: hard USD cap for a Claude guide (required for claude-*)")
    ap.add_argument("--ledger", default=None, help="P20: JSONL file recording every Claude call's tokens and cost")
    ap.add_argument("--tags", nargs="+", default=TAGS, help="run tags (P19: seed5..seed14 for more guide runs)")
    ap.add_argument("--guide-sampling", default=None, help='JSON, e.g. {"temperature": 1.0, "top_p": 0.95, "top_k": 20}')
    args = ap.parse_args()
    import contexts
    from reveries.llm import LLM
    from reveries.tasks import load_tasks
    tasks = {t.task_id: t for f in ("tasks.yaml", "tasks_p1b.yaml") for t in load_tasks(REPO / "quick_runs" / f)[0]}
    questions = {t: v.question.split("\n\nData files")[0] for t, v in tasks.items()}
    contexts.QUESTIONS.update(questions)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMB_MODEL, device="cpu", model_kwargs={"dtype": "float32"})
    have = {"nf": LESSONS_NF.exists(), "nfd": LESSONS_NFD.exists(), "nfdtr": LESSONS_NFDTR.exists()}
    retrievers = {p: Retriever(load_store(p), questions, model) for p in STORES if have.get(p, True)}
    xenc = None
    if any(ARMS[a][3] == "cross" for a in args.arms):
        from sentence_transformers import CrossEncoder
        xenc = CrossEncoder(str(CROSS_PATH), device="cpu")
    cross = lambda: xenc
    llm = LLM(base_url=args.server)  # agent's sampling, thinking on
    if args.guide_base_url:  # P23: OpenAI-compatible endpoint (e.g. DeepSeek on SPIKE); token-only ledger
        import os
        from reveries.llm import OpenAILLM, ClaudeSpend
        llm = OpenAILLM(base_url=args.guide_base_url, model=args.guide_model, api_key=os.environ["SPIKE_API_KEY"],
                        spend=ClaudeSpend(float("inf"), args.ledger))
    elif args.guide_model and args.guide_model.startswith("claude-"):  # P20: Anthropic API, hard spend cap
        from reveries.llm import ClaudeLLM, ClaudeSpend
        assert args.budget > 0, "--budget is required for a Claude guide"
        llm = ClaudeLLM(model=args.guide_model, spend=ClaudeSpend(args.budget, args.ledger))
    elif args.guide_model:  # P15: a different model as the guide, with its own recommended sampling
        llm = LLM(base_url=args.server, model=args.guide_model, sampling=json.loads(args.guide_sampling))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.trace.parent.mkdir(parents=True, exist_ok=True)

    def job(arm, tid, tag):
        poisoned, guide_on, retries, k = ARMS[arm]
        retr = retrievers[poisoned]
        if not guide_on:
            idx = ([i for i, L in enumerate(retr.store) if L["source_task"] == tid] if k == "own"
                   else no_judge_pick(arm, tid, tag) if k == "nojudge"
                   else cross_rerank(retr, cross(), questions[tid], SHOW) if k == "cross" else retr.top(questions[tid], k))
            return arm, tid, tag, idx, {"arm": arm, "task": tid, "run": tag, "shown": idx, "rounds": []}
        if arm in HEAVY:
            old, new = HEAVY_SENTENCE[tid]; assert questions[tid].count(old) == 1, (tid, old)
            guide_q = questions[tid].replace(old, new)
        else:
            guide_q = (GOAL_ONLY[tid] if arm in TASK_BLIND else one_sentence_hidden(tid, questions[tid])
                       if arm in ONE_SENTENCE else one_sentence_hidden(tid, questions[tid], placebo=True)
                       if arm == "G3s-pl" else hint_question(tid) if arm in HINTED else questions[tid])
        profile = (data_profile(tasks[tid].data_files, LABEL[tid], train_only=arm in TRAIN_ONLY)
                   if arm in WITH_PROFILE else "")
        res = guided(guide_q, tasks[tid].schema, retr, llm, retries, PROMPTS.get(arm, GUIDE), query0=questions[tid],
                     profile=profile, escalate=arm in ESCALATING)
        return arm, tid, tag, res.shown, {"arm": arm, "task": tid, "run": tag, "shown": res.shown, "flagged": res.flagged,
                                          "accepted": res.accepted, "seen": res.seen, "rounds": res.rounds}

    jobs = [(a, t, g) for a in args.arms for t in arm_tasks(a) for g in args.tags]
    out = {a: {} for a in args.arms}
    with ThreadPoolExecutor(12) as pool, open(args.trace, "a") as tr:
        for arm, tid, tag, idx, trace in pool.map(lambda j: job(*j), jobs):
            store = retrievers[ARMS[arm][0]].store
            trace["shown_lessons"] = [{k: store[i].get(k) for k in ("source_task", "source_run", "k", "title", "wrong")}
                                      for i in idx]
            tr.write(json.dumps(trace) + "\n")
            out[arm].setdefault(tid, {})[tag] = render(store, idx)
            print(arm, tid, tag, "shown", len(idx), "rounds", len(trace["rounds"]), flush=True)
    for arm, d in out.items():
        (args.out_dir / f"{arm}.yaml").write_text(yaml.safe_dump(d, width=10**9, allow_unicode=True))


if __name__ == "__main__":
    main()
