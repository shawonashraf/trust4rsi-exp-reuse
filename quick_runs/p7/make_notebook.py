"""Build and execute docs/notebooks/quick_P7.ipynb (recomputes the P7 numbers from the run files).

Usage: uv run --no-active python quick_runs/p7/make_notebook.py
"""
from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parents[2]
NB = REPO / "docs/notebooks/quick_P7.ipynb"

cells = [
    nbformat.v4.new_markdown_cell(
        "# P7. Loop-guard replay\n\n"
        "Would a dead-end ledger have stopped the repeated dead ends in the old experience-memory runs "
        "(tasks 5 and 16, arms E0 and E1)? This notebook recomputes every number in "
        "`quick_runs/results/P7/report.md` from the old run files (read only) and checks it against "
        "`summary.json`. Definitions are in `quick_runs/p7/ledger.py` and in the report."),
    nbformat.v4.new_code_cell(
        "import json, sys\nfrom pathlib import Path\nimport pandas as pd\n"
        "REPO = Path.cwd().resolve()\nwhile not (REPO / 'quick_runs').is_dir():\n    REPO = REPO.parent\n"
        "sys.path.insert(0, str(REPO / 'quick_runs' / 'p7'))\n"
        "import run as p7\nfrom ledger import replay, coarse_key, outcome_class\n"
        "rows, examples = [], []\nfor t in p7.TASKS:\n    for a in p7.ARMS:\n"
        "        r, ex = p7.per_run(t, a)\n        rows.append(r); examples += ex\n"
        "saved = json.loads((REPO / 'quick_runs/results/P7/summary.json').read_text())\n"
        "assert rows == saved['runs'], 'recomputed numbers differ from summary.json'\n"
        "print('recomputed numbers match summary.json')"),
    nbformat.v4.new_markdown_cell("## Blocked attempts per run (fingerprint key = variables + old design fingerprint)"),
    nbformat.v4.new_code_cell(
        "tab = pd.DataFrame([{k: r[k] for k in ['task', 'arm', 'attempts', 'retries', 'old_dead_end_retries',\n"
        "      'old_dead_end_retries_check_old_notebook_count', 'old_dead_end_retries_blocked', 'blocked',\n"
        "      'blocked_that_were_dead_again', 'blocked_that_were_supported', 'attempts_without_fingerprint']} for r in rows])\n"
        "tab['blocked_share_of_retries'] = (tab.blocked / tab.retries).round(2)\ntab"),
    nbformat.v4.new_markdown_cell("## Outcome classes per run"),
    nbformat.v4.new_code_cell("pd.DataFrame([{'task': r['task'], 'arm': r['arm'], **r['outcome_classes'], "
                              "'null_flags': r['null_flags'], 'null_flags_also_rejected': r['null_flags_also_rejected']} "
                              "for r in rows]).fillna(0)"),
    nbformat.v4.new_markdown_cell("## Coarser key (variables, model family, controls)"),
    nbformat.v4.new_code_cell("pd.DataFrame([{'task': r['task'], 'arm': r['arm'], **r['coarse_key']} for r in rows])"),
    nbformat.v4.new_markdown_cell("## Near-duplicates of dead attempts that the fingerprint ledger lets through"),
    nbformat.v4.new_code_cell(
        "pd.DataFrame([{'task': r['task'], 'arm': r['arm'], **r['missed_near_duplicates'],\n"
        "               'ended_dead_again': sum(r['missed_near_duplicates_dead_again'].values())} for r in rows]).fillna(0)"),
    nbformat.v4.new_code_cell(
        "pd.set_option('display.max_colwidth', 200)\n"
        "pd.DataFrame([{'task': e['task'], 'arm': e['arm'], 'kind': e['kind'],\n"
        "               'earlier': f\"#{e['earlier']['attempt']} {e['earlier']['outcome']}: {e['earlier']['design']}\",\n"
        "               'later': f\"#{e['later']['attempt']} {e['later']['outcome']}: {e['later']['design']}\"} for e in examples])"),
    nbformat.v4.new_markdown_cell("## When do blocked attempts happen? (cumulative count over the run, E1)"),
    nbformat.v4.new_code_cell(
        "import matplotlib.pyplot as plt\nfig, ax = plt.subplots(figsize=(7, 3.5))\n"
        "for t in p7.TASKS:\n    rep = replay(p7.load(t, 'e1'))\n"
        "    ax.plot(pd.Series([x['blocked'] for x in rep]).cumsum().values, label=f'task {t}: blocked')\n"
        "    ax.plot(pd.Series([x['old_dead_end_retry'] for x in rep]).cumsum().values, '--', label=f'task {t}: old dead-end retries')\n"
        "ax.set_xlabel('attempt'); ax.set_ylabel('cumulative count'); ax.legend(); plt.tight_layout()"),
]
nb = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3",
                                                                    "language": "python"}})
NB.parent.mkdir(parents=True, exist_ok=True)
NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(NB.parent)}}).execute()
nbformat.write(nb, NB)
print("wrote", NB)
