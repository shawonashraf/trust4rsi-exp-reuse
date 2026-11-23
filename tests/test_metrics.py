import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "quick_runs"))
from metrics import (answer_consistency, consistent_failures, decision_agreement,  # noqa: E402
                     first_divergence, score_spread, task_score)

TYPES = ["framing.target", "variables", "model.family", "validation"]


def run(target, variables, family, validation, order, answer, score, checks):
    return {"decisions": {"framing.target": target, "variables": variables, "model.family": family,
                          "validation": validation},
            "order": order, "answer": answer, "score": score, "checks": checks}


# Hand-made 5-run example: all agree on target; variables split 3/2; family split 2/2/1;
# validation missing in one run.
RUNS = [
    run("y", "a|b", "ols", "none", ["framing.target", "variables", "model.family"], "pos", 1.0, {"leak": True, "cw": False}),
    run("y", "a|b", "ols", "none", ["framing.target", "variables", "model.family"], "pos", 2.0, {"leak": True, "cw": False}),
    run("y", "a|b", "logistic", "none", ["framing.target", "model.family", "variables"], "pos", 3.0, {"leak": False, "cw": False}),
    run("y", "a", "logistic", "none", ["framing.target", "variables", "model.family"], "neg", 4.0, {"leak": True, "cw": False}),
    run("y", "a", "poisson", None, ["framing.target", "variables", "model.family"], None, None, {"leak": True, "cw": None}),
]


def test_decision_agreement():
    assert decision_agreement(RUNS, "framing.target") == 1.0
    # variables: 3x "a|b", 2x "a" -> (3 + 1) / 10 agreeing pairs
    assert decision_agreement(RUNS, "variables") == pytest.approx(0.4)
    # family: 2 ols, 2 logistic, 1 poisson -> 2 / 10
    assert decision_agreement(RUNS, "model.family") == pytest.approx(0.2)
    # validation: 4 runs, all "none" -> 1.0 (missing run left out)
    assert decision_agreement(RUNS, "validation") == 1.0
    assert decision_agreement(RUNS[:1], "variables") is None


def test_first_divergence():
    # variables mean position (1+1+2+1+1)/5 = 1.2 < model.family (2+2+1+2+2)/5 = 1.8
    assert first_divergence(RUNS, TYPES) == "variables"
    same = [RUNS[0], RUNS[1]]
    assert first_divergence(same, TYPES) is None


def test_answer_consistency():
    # answers pos,pos,pos,neg,None -> matching pairs: 3 of 10
    assert answer_consistency(RUNS) == pytest.approx(0.3)


def test_consistent_failures():
    # cw: False in 4 runs, None in one -> not counted; leak: one False only -> not counted
    assert consistent_failures(RUNS) == []
    assert consistent_failures(RUNS[:4]) == ["cw"]


def test_score():
    assert task_score(RUNS) == pytest.approx(2.5)
    assert score_spread(RUNS) == pytest.approx(1.2909944)


def test_dsbench_answer_matching():
    from score import dsa_correct, dsa_normalise
    assert dsa_normalise("D", "letter") == "D"
    assert dsa_normalise("(b) 73 days", "letter") == "B"
    assert dsa_normalise("C. 74", "letter") == "C"
    assert dsa_normalise("about 75", "letter") is None
    assert dsa_normalise("$1,661,626", "integer") == 1661626.0
    assert dsa_correct("D", {"answer": "D", "answer_type": "letter"})
    assert not dsa_correct("A", {"answer": "D", "answer_type": "letter"})
    assert dsa_correct(1661627.0, {"answer": 1661626, "answer_type": "integer"})
    assert not dsa_correct(1661630.0, {"answer": 1661626, "answer_type": "integer"})


def test_norm_name():
    from score import norm_name
    assert norm_name("roc_auc_score") == norm_name("roc_auc") == norm_name("ROC AUC") == "roc_auc"
    assert norm_name("RandomForestClassifier") == norm_name("Random Forest") == norm_name("randomforest")
    assert norm_name("macro_f1") != norm_name("f1_binary")


def test_family_of():
    from score import family_of
    assert family_of("xgboost regression") == "xgboost"
    assert family_of("LogisticRegression, RandomForestClassifier, XGBClassifier") == "several"
    assert family_of("Negative Binomial") == "negative_binomial"
    assert family_of("logistic regression") == "logistic"
    assert family_of("OLS linear regression") == "ols"
