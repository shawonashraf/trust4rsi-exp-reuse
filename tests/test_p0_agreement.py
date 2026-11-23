import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "quick_runs"))
sys.path.insert(0, str(ROOT / "quick_runs" / "p0"))
from agree import pair_weighted, pooled, run_record, task_table  # noqa: E402
from memory_use import (code_items, design_signature, digest_items, judge_code_item,  # noqa: E402
                        judge_digest_item)
from runs import final_code_cells, rank_hypotheses  # noqa: E402


def ex(target, controls, family, verdict=None):
    return {"framing.target": target, "model.controls": controls, "model.family": family, "verdict": verdict,
            "source_columns": {"target": [target], "variables": ["X"], "controls": controls}}


def test_run_record_canonical_and_missing():
    r = run_record(ex("BMI", ["b", "A"], "OLS"))
    assert r["decisions"]["model.controls"] == "a|b"
    assert r["decisions"]["model.family"] == "ols"
    assert r["decisions"]["verdict"] is None          # null is left out, not a value
    assert run_record(ex("BMI", [], "ols"))["decisions"]["model.controls"] == "(none)"


def test_task_table_and_pooled():
    # task 1: 3 runs, family 2/1 split -> 1 of 3 pairs agree; target all same.
    # task 2: 2 runs, both differ on target.
    t1 = [run_record(ex("y", [], "ols")), run_record(ex("y", [], "ols")), run_record(ex("y", [], "logistic"))]
    t2 = [run_record(ex("y", ["a"], "ols")), run_record(ex("z", ["a"], "ols"))]
    by = {1: t1, 2: t2}
    types = ["framing.target", "model.family", "model.controls"]
    tab = task_table(by, types).set_index("task")
    assert tab.loc[1, "model.family"] == 1 / 3
    assert tab.loc[1, "framing.target"] == 1.0
    assert tab.loc[2, "framing.target"] == 0.0
    assert tab.loc[2, "first_divergence"] == "framing.target"   # first in the fixed type order
    assert tab.loc[1, "first_divergence"] == "model.family"
    p = {x["type"]: x for x in pooled(tab.reset_index(), by, types)}
    assert p["framing.target"]["mean_agreement_over_tasks"] == 0.5
    assert p["framing.target"]["tasks_all_runs_same"] == 1
    assert p["framing.target"]["tasks_all_runs_differ"] == 1
    # pair-weighted: task 1 has 3 agreeing pairs, task 2 has 1 disagreeing pair -> 3/4
    assert pair_weighted(by, "framing.target") == 0.75


def test_final_code_cells_uses_top_ranked_hypotheses():
    facts = {"hypotheses": [{"id": "h1", "status": "rejected", "confidence": 1.0},
                            {"id": "h2", "status": "supported", "confidence": 0.5},
                            {"id": "h3", "status": "supported", "confidence": 0.9}],
             "experiments": [{"order": 1, "hypothesis_id": "h2", "success": True, "code": "c2a"},
                             {"order": 2, "hypothesis_id": "h3", "success": True, "code": "c3"},
                             {"order": 3, "hypothesis_id": "h2", "success": True, "code": "c2b"},
                             {"order": 4, "hypothesis_id": "h1", "success": True, "code": "c1"},
                             {"order": 5, "hypothesis_id": "h3", "success": False, "code": "c3bad"}]}
    assert [h["id"] for h in rank_hypotheses(facts["hypotheses"])] == ["h3", "h2"]
    assert final_code_cells(facts, top=2) == ["c3", "c2b"]   # last successful per hypothesis, run order


def test_code_memory_judgment():
    a = 'model = smf.logit("Y ~ AGE + INCOME", data=df).fit()'
    b = 'm = smf.logit(formula="Y ~ INCOME + AGE", data=d).fit()'
    c = 'm = smf.ols("Y ~ AGE", data=d).fit()'
    assert design_signature(a) == design_signature(b)
    assert judge_code_item(a, b) == "followed"
    assert judge_code_item(a, c) == "ignored"
    assert judge_code_item("print(1)", a) == "cannot_tell"
    sec = ("Relevant memories:\n\n- Experiment for Hypothesis 0185b71d-34f4-42d8-9edb-64a82c630376:\n" + a +
           "\n- Results for Experiment ea795ae7-da21-41ce-94cd-a39573b75165:\nok\n")
    items = code_items(sec)
    assert len(items) == 1 and items[0]["hypothesis_id"].startswith("0185b71d") and "smf.logit" in items[0]["text"]


def test_digest_judgment():
    sec = ("Designs already run on this task and how they ended.\n"
           "(age, dissaved) · logit; controls: INCOME; moderators: -; transforms: -; scope: all: 2 attempts: 2 null\n"
           "  latest reflection: rules_out x | try_next Include AGE_2 and GENDER in the model.\n"
           "(age, dissaved) · ols; controls: -; moderators: -; transforms: -; scope: all: 1 attempts: 1 supported\n"
           "  latest reflection: rules_out y | try_next Try another model.\n")
    items = digest_items(sec)
    assert [i["tally"] for i in items] == ["2 null", "1 supported"]
    cols = ["AGE", "AGE_2", "DISSAVED", "INCOME", "GENDER"]
    assert judge_digest_item(items[0], "logit; controls: INCOME, AGE_2", cols) == ("followed", ["AGE_2", "GENDER"])
    assert judge_digest_item(items[0], "logit; controls: INCOME", cols) == ("ignored", ["AGE_2", "GENDER"])
    assert judge_digest_item(items[1], "logit; controls: INCOME", cols) == ("cannot_tell", [])
