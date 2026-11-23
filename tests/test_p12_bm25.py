import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "quick_runs"))
from p12.contexts import bm25_top  # noqa: E402


def test_bm25_ranks_matching_doc_first():
    docs = ["class imbalance in goals label", "strip cm suffix from height", "leakage columns credit card"]
    assert bm25_top("height values like 185cm need the cm suffix removed", docs, 1) == [1]
    assert bm25_top("credit card approval leakage", docs, 2)[0] == 2
