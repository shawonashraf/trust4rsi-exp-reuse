"""P13 guide loop with a fake retriever and a fake LLM (no server, no embedding model)."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "quick_runs" / "p13"))
import guide  # noqa: E402


class FakeRetr:
    def __init__(self, n):
        self.store = [{"title": f"t{i}", "when": "w", "content": f"c{i}"} for i in range(n)]
        self.queries = []

    def top(self, query, k, exclude=frozenset()):
        self.queries.append(query)
        return [i for i in range(len(self.store)) if i not in exclude][:k]


class FakeLLM:
    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def chat(self, messages, max_tokens=None):
        self.prompts.append(messages[0]["content"])
        return SimpleNamespace(content=self.replies.pop(0), usage={}, finish_reason="stop")


def reply(useful, enough, query=""):
    return json.dumps({"judgements": [{"id": i, "useful": True, "reason": ""} for i in useful],
                       "enough": enough, "query": query})


def test_no_retry_budget_stops_after_round0():
    r, llm = FakeRetr(20), FakeLLM([reply([1, 3], False, "more")])
    res = guided = guide.guided("q", "s", r, llm, retries=0)
    assert res.shown == [1, 3] and len(res.rounds) == 1 and r.queries == ["q"]


def test_retry_uses_guide_query_and_skips_seen():
    r, llm = FakeRetr(20), FakeLLM([reply([0], False, "units"), reply([7], True)])
    res = guide.guided("q", "s", r, llm, retries=3)
    assert r.queries == ["q", "units"]
    assert res.seen == list(range(10)) and res.shown == [0, 7] and len(res.rounds) == 2
    assert "t0" in llm.prompts[1].split("## Candidate notes")[0]  # accepted note shown back to the guide


def test_ids_outside_candidates_ignored_and_cap_three():
    r = FakeRetr(30)
    llm = FakeLLM([reply([0, 1, 99], False, "a"), reply([5, 6], False, "b"), reply([10], True)])
    res = guide.guided("q", "s", r, llm, retries=3)
    assert res.accepted == [0, 1, 5, 6, 10] and res.shown == [0, 1, 5]


def test_unparseable_reply_accepts_nothing_and_stops():
    r, llm = FakeRetr(10), FakeLLM(["no json here"])
    res = guide.guided("q", "s", r, llm, retries=3)
    assert res.shown == [] and len(res.rounds) == 1 and res.rounds[0]["parsed"] is False


def test_poisoned_store_adds_12_marked_lessons():
    clean, pois = guide.load_store(False), guide.load_store(True)
    assert len(pois) - len(clean) == 12 and sum(L.get("wrong", False) for L in pois) == 12


def test_p14_prompt_changes_only_the_task_criterion():
    assert guide.GUIDE_P14 != guide.GUIDE
    assert guide.GUIDE_P14.replace(guide.NEW_CRITERION, guide.OLD_CRITERION) == guide.GUIDE
    assert "consistent with the task" not in guide.GUIDE_P14
    assert guide.PROMPTS["G3n"] is guide.GUIDE_P14 and guide.PROMPTS["G3o"] is guide.GUIDE


def test_p14_guide_uses_given_template_and_saves_reasons():
    retr = FakeRetr(10)
    llm = FakeLLM([json.dumps({"judgements": [{"id": 0, "useful": True, "reason": "fits"}], "enough": True, "query": ""})])
    res = guide.guided("Q", "S", retr, llm, retries=3, template=guide.GUIDE_P14)
    assert guide.NEW_CRITERION.split("\n")[0] in llm.prompts[0]
    assert res.rounds[0]["judgements"][0]["reason"] == "fits"


def test_p16_compliant_store_replaces_the_wrong_lessons():
    clean, pc = guide.load_store(False), guide.load_store("pc")
    assert len(pc) - len(clean) == 12 and all(L.get("wrong") for L in pc[len(clean):])
    assert {L["title"] for L in pc[len(clean):]}.isdisjoint({L["title"] for L in guide.load_store(True)[len(clean):]})


def test_p16_task_blind_guide_sees_goal_only_but_retrieves_on_full_task():
    retr = FakeRetr(10)
    llm = FakeLLM([json.dumps({"judgements": [], "enough": True, "query": ""})])
    res = guide.guided("GOAL ONLY", "S", retr, llm, retries=3, query0="FULL TASK use all available features")
    assert "GOAL ONLY" in llm.prompts[0] and "use all available" not in llm.prompts[0]
    assert res.rounds[0]["query"] == "FULL TASK use all available features"
    assert set(guide.GOAL_ONLY) == set(guide.SF_TASKS) and set(guide.LABEL) == set(guide.SF_TASKS)
    for t in guide.GOAL_ONLY.values():
        assert not any(w in t.lower() for w in ("use all", "numerical feature", "fill missing", "impute", "except"))


def test_p16_profile_prompt_and_profile_numbers(tmp_path):
    import pandas as pd
    pd.DataFrame({"y": ["yes", "no", "yes", "no"], "a": [1.0, 0.0, 1.0, 0.0], "t": ["1cm", "2", "3", "4"]}).to_csv(tmp_path / "d_train.csv", index=False)
    pd.DataFrame({"y": ["yes", "no", "no", "yes"], "a": [0.0, 1.0, 1.0, 0.0], "t": ["1", "2", "3", "4"]}).to_csv(tmp_path / "d_test.csv", index=False)
    prof = guide.data_profile([tmp_path / "d_test.csv", tmp_path / "d_train.csv"], "y")
    assert "a: 0% | +1.00, -1.00" in prof and "'1cm'" in prof and "no: 50.0%, yes: 50.0%" in prof
    assert "{profile}" in guide.GUIDE_D and guide.GUIDE_D.replace(
        "\n\n## Data profile (computed from the files)\n{profile}", "").replace(
        " Use the profile to check what a note claims about the data.", "") == guide.GUIDE


def test_p18_no_judge_picks_from_the_same_candidates_and_indices_still_match():
    trace = {(d["arm"], d["task"], d["run"]): d for d in map(json.loads, open(guide.P13_TRACE))}
    for arm, src, store in (("R3", "G3", guide.load_store(False)), ("R3-p", "G3-p", guide.load_store(True))):
        for (a, tid, tag), d in trace.items():
            if a != src:
                continue
            pick = guide.no_judge_pick(arm, tid, tag)
            assert len(pick) == 3 and set(pick) <= set(d["seen"]) and pick == guide.no_judge_pick(arm, tid, tag)
            for i, L in zip(d["shown"], d["shown_lessons"]):  # P13 store indices point to the same lessons today
                assert store[i]["title"] == L["title"]


def test_p18_one_sentence_hidden_removes_only_the_feature_sentence():
    q3 = "Build a model. Keep only numerical feature columns (excluding the label). Train the model."
    assert guide.one_sentence_hidden("sf_tf03_correlation_causation", q3) == "Build a model. Train the model."
    q5 = "Fill missing values with 0). Use all available feature columns to train a LogisticRegression."
    assert guide.one_sentence_hidden("sf_tf05_credit_card_leakage", q5) == "Fill missing values with 0). Train a LogisticRegression."
    assert guide.arm_tasks("G3s") == ["sf_tf03_correlation_causation", "sf_tf05_credit_card_leakage"]


def test_p18_train_only_profile_has_no_test_numbers(tmp_path):
    import pandas as pd
    pd.DataFrame({"y": ["yes", "no", "yes", "no"], "a": [1.0, 0.0, 1.0, 0.0], "t": ["1cm", "2", "3", "4"]}).to_csv(tmp_path / "d_train.csv", index=False)
    pd.DataFrame({"y": ["yes", "no", "no", "yes"], "a": [0.0, 1.0, 1.0, 0.0], "t": ["1", "2", "3", "4"]}).to_csv(tmp_path / "d_test.csv", index=False)
    prof = guide.data_profile([tmp_path / "d_test.csv", tmp_path / "d_train.csv"], "y", train_only=True)
    assert "a: 0% | +1.00 | 0.5" in prof and "-1.00" not in prof and "test" not in prof.lower() and "'1cm'" in prof
    full = guide.data_profile([tmp_path / "d_test.csv", tmp_path / "d_train.csv"], "y")
    assert prof != full and guide.PROMPTS["G3d-tr"] is guide.GUIDE_D and guide.TRAIN_ONLY <= guide.WITH_PROFILE


def test_p18_placebo_and_hint_texts():
    real = {"sf_tf03_correlation_causation": "build a model. Keep only numerical feature columns (excluding the label). Train the model on the training set and report the RMSE.",
            "sf_tf05_credit_card_leakage": "fill missing values with 0). Use all available feature columns to train a LogisticRegression classifier (max_iter=1000, random_state=42) to predict 'card'."}
    for tid, q in real.items():
        pl = guide.one_sentence_hidden(tid, q, placebo=True)
        assert pl != q and guide.FEATURE_SENTENCE[tid][0] in pl
    assert guide.arm_tasks("G3s-pl") == guide.arm_tasks("G3s") and guide.arm_tasks("G3h") == guide.POISONED_TASKS
    h = {t: guide.hint_question(t) for t in guide.HINT_WORKLOAD}
    assert "exclude them from features" in h["sf_tf05_credit_card_leakage"]
    assert "unit suffix" in h["sf_tf02_player_rep_norm"] and "causal relationship" in h["sf_tf03_correlation_causation"]
    assert "label distribution" in h["sf_tf01_goal_imbalance"]


def test_p18_cross_rerank_orders_by_cross_score():
    class R:
        store = [{"title": f"t{i}", "when": "", "content": ""} for i in range(30)]
        def top(self, q, k, exclude=frozenset()):
            return list(range(k))
    class X:
        def predict(self, pairs):
            return [float(d.split()[0][1:]) for _, d in pairs]  # higher id, higher score
    assert guide.cross_rerank(R(), X(), "q", 3) == [19, 18, 17]


def test_p19_escalating_guide_flags_conflicts_and_withholds_them():
    retr = FakeRetr(10)
    r = json.dumps({"judgements": [{"id": 0, "useful": True, "conflict": True, "reason": "breaks the rule"},
                                   {"id": 1, "useful": True, "conflict": False, "reason": "fits"},
                                   {"id": 2, "useful": False, "conflict": True, "reason": "x"}], "enough": True, "query": ""})
    res = guide.guided("Q", "S", retr, FakeLLM([r]), retries=3, template=guide.GUIDE_X, escalate=True)
    assert res.shown == [1] and res.flagged == [0, 2]
    res2 = guide.guided("Q", "S", retr, FakeLLM([r]), retries=3, template=guide.GUIDE)  # conflict ignored without escalate
    assert res2.shown == [0, 1] and res2.flagged == []
    assert '"conflict": true or false' in guide.GUIDE_X and guide.GUIDE_X.replace(guide.ESCALATE, "").replace(
        '"conflict": true or false, ', "") == guide.GUIDE
    assert guide.arm_tasks("G3x") == guide.POISONED_TASKS and guide.PROMPTS["G3n-nf"] is guide.GUIDE_P14


def test_p19_writer_prompt_swaps_outcome_for_profile():
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("write_nfd", Path(guide.REPO) / "quick_runs/p19/write_nfd.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    assert "Evaluation of this run" not in m.WRITER_NFD and "## Data profile" in m.WRITER_NFD
    assert m.WRITER_NFD.replace(m.PROBE_SECTION, m.EVAL_SECTION) == m.P12.WRITER
    assert set(m.LABEL) >= {"sf_tf04_balance_torque_cue", "sf_tf06_metric_select_cue"} | set(guide.SF_TASKS)


def test_p20_claude_spend_cap_and_arms():
    from reveries.llm import ClaudeSpend, BudgetExceeded, CLAUDE_PRICE
    sp = ClaudeSpend(0.05)
    c = sp.add("claude-opus-5-5", {"prompt_tokens": 5000, "completion_tokens": 1500})
    assert abs(c - (5000 * 4 + 1500 * 20) / 1e6) < 1e-9
    sp.add("claude-opus-5-5", {"prompt_tokens": 0, "completion_tokens": 2000})
    try:
        sp.check(); assert False
    except BudgetExceeded:
        pass
    assert {"claude-opus-5-5", "claude-fable-5-1"} <= set(CLAUDE_PRICE)
    for a in ("G3c", "G3c-p", "G3c-pc", "G3f", "G3f-p", "G3f-pc"):
        assert guide.PROMPTS[a] is guide.GUIDE
    assert guide.arm_tasks("G3c-pc") == guide.POISONED_TASKS and guide.arm_tasks("G3f") == guide.SF_TASKS


def test_p22_heavy_placebo_and_open_profile():
    assert guide.arm_tasks("G3s-pl2") == list(guide.FEATURE_SENTENCE)
    for tid, (old, new) in guide.HEAVY_SENTENCE.items():
        assert old != new and tid in guide.FEATURE_SENTENCE
    # G3o-d: open-prompt criterion AND the data profile section, and it is on the profile path
    assert "{profile}" in guide.GUIDE_P14_D and guide.NEW_CRITERION.split("\n")[0] in guide.GUIDE_P14_D
    assert guide.PROMPTS["G3o-d"] is guide.GUIDE_P14_D and "G3o-d" in guide.WITH_PROFILE
    assert guide.PROMPTS["G3s-pl2"] is guide.GUIDE  # heavy placebo keeps the original prompt
