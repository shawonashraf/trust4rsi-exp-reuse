import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "quick_runs" / "p7"))
from ledger import (coarse_key, missed_near_duplicates, old_dead_end_retries, outcome_class,  # noqa: E402
                    replay)


def rec(fp, outcome="supported", null=False, status="success", pair=("a", "b"), family="logit", controls=(),
        scope="all"):
    return {"pair_columns": list(pair), "design_fingerprint": fp, "outcome": outcome,
            "corrector": {"null_result": null}, "experiment": {"execution_status": status},
            "design_used": {"model_family": family, "controls": list(controls), "moderators": [],
                            "transformations": [], "scope": scope}}


def test_outcome_class_order():
    assert outcome_class(rec("x", status="error")) == "error"
    assert outcome_class(rec("x", outcome="rejected", null=True)) == "rejected"
    assert outcome_class(rec("x", null=True)) == "null"
    assert outcome_class(rec("x", outcome="inconclusive")) == "inconclusive"
    assert outcome_class(rec("x")) == "supported"


def test_replay_counts():
    # A: supported, A again (retry, not blocked), B rejected, B again (blocked), B again (blocked),
    # C null, C (blocked), none (no key), A with other variables (new key)
    recs = [rec("A"), rec("A"), rec("B", "rejected"), rec("B"), rec("B", "rejected"),
            rec("C", null=True), rec("C"), rec("none", "rejected"), rec("A", pair=("a", "c"))]
    r = replay(recs)
    assert [x["retry"] for x in r] == [False, True, False, True, True, False, True, False, False]
    assert [x["blocked"] for x in r] == [False, False, False, True, True, False, True, False, False]
    assert r[7]["key"] is None
    # Old definition looks at the real previous record in the group: B(rej)->B counts, B(sup)->B(rej) does not.
    assert [x["old_dead_end_retry"] for x in r] == [False, False, False, True, False, False, True, False, False]
    assert old_dead_end_retries(recs) == 2


def test_error_blocks_too():
    r = replay([rec("E", status="error"), rec("E")])
    assert r[1]["blocked"]


def test_coarse_key_ignores_scope_and_merges_family_aliases():
    a = rec("X", family="Logit", controls=["AGE"], scope="all rows")
    b = rec("Y", family="logistic regression", controls=["AGE"], scope="all respondents")
    assert coarse_key(a) == coarse_key(b)
    r = replay([{**a, "outcome": "rejected"}, b], coarse_key)
    assert r[1]["blocked"]


def test_missed_near_duplicates_kinds():
    recs = [rec("X", "rejected", controls=["AGE"], scope="all rows"),
            rec("Y", controls=["AGE"], scope="all respondents"),      # same design, other wording
            rec("Z", controls=["AGE", "SEX"]),                         # one element off
            rec("none"),                                               # no design recorded
            rec("W", family="ols", controls=["AGE"])]                  # different family: not near
    kinds = {m["j"]: m["kind"] for m in missed_near_duplicates(recs)}
    assert kinds == {1: "same_design_other_wording", 2: "one_element_off", 3: "no_design_recorded"}
