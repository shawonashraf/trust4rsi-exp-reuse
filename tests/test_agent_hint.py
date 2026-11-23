from types import SimpleNamespace

from reveries.agent import Hint, initial_user_message

TASK = SimpleNamespace(question="Q?", schema="cols")


def test_no_hint():
    assert initial_user_message(TASK, None) == "Q?\n\nData schema (computed from the files):\ncols"


def test_default_prefix_unchanged_from_p2():
    m = initial_user_message(TASK, Hint("drop x"))
    assert m.startswith("Note from previous experience: drop x\n\nQ?")


def test_empty_prefix_passes_text_verbatim():
    m = initial_user_message(TASK, Hint("## Notes\n- a", prefix=""))
    assert m.startswith("## Notes\n- a\n\nQ?")


def test_decision_hint_not_at_start():
    assert initial_user_message(TASK, Hint("x", where="decision", decision_type="variables")).startswith("Q?")
