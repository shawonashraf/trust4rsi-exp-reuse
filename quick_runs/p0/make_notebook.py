"""Build and execute docs/notebooks/quick_P0.ipynb (recomputes the P0 numbers from the cached files).

Usage: uv run --no-active python quick_runs/p0/make_notebook.py
"""
from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parents[2]
NB = REPO / "docs/notebooks/quick_P0.ipynb"

cells = [
    nbformat.v4.new_markdown_cell(
        "# P0. Re-mine old runs\n\n"
        "Do repeated runs of the same DiscoveryBench task split at analysis decisions, and at which types? "
        "This notebook recomputes the numbers in `quick_runs/results/P0/report.md` from the run list "
        "(`run_folders.csv`), the per-run facts and the cached extractor replies under "
        "`/data/anon/reveries_runs/P0/`, and checks them against `summary.json`.\n\n"
        "Caveat: the runs of one task differ by arm (memory, retrieval, corrector, design discipline) and by "
        "code version, not only by seed."),
    nbformat.v4.new_code_cell(
        "import json, sys\nfrom pathlib import Path\nimport pandas as pd\n"
        "REPO = Path.cwd().resolve()\nwhile not (REPO / 'quick_runs').is_dir():\n    REPO = REPO.parent\n"
        "for p in (REPO, REPO / 'quick_runs', REPO / 'quick_runs' / 'p0'):\n    sys.path.insert(0, str(p))\n"
        "import agree\n"
        "runs = pd.read_csv(REPO / 'quick_runs/results/P0/run_folders.csv')\n"
        "print(runs.why.value_counts().to_string())\n"
        "inc = runs[runs.included]\nprint('included runs per task:', inc.groupby('task').size().to_dict())\n"
        "print(inc.arm.value_counts().to_string())"),
    nbformat.v4.new_markdown_cell("## Recompute agreement from the cached extractor replies"),
    nbformat.v4.new_code_cell(
        "recs, failed = agree.load_records(inc)\nprint(len(recs), 'runs extracted; failures:', failed)\n"
        "results = {s: agree.analyse(recs, s) for s in agree.SUBSETS}\n"
        "saved = json.loads((REPO / 'quick_runs/results/P0/summary.json').read_text())\n"
        "assert json.loads(json.dumps(results)) == saved['results'], 'recomputed agreement differs from summary.json'\n"
        "assert len(recs) == saved['n_extracted']\nprint('recomputed agreement matches summary.json')"),
    nbformat.v4.new_code_cell(
        "def pooled_table(subset):\n"
        "    df = pd.DataFrame(results[subset]['pooled']).set_index('type')\n"
        "    return df.round(3)\n"
        "print('all runs:', results['all']['n_runs'], 'runs,', results['all']['n_tasks'], 'tasks')\npooled_table('all')"),
    nbformat.v4.new_code_cell("print('v8 only:', results['v8_only']['n_runs'], 'runs')\npooled_table('v8_only')"),
    nbformat.v4.new_code_cell(
        "print('split most (all):', results['all']['split_most'])\n"
        "print('split most (v8 only):', results['v8_only']['split_most'])\n"
        "print('split most (v10 only):', results['v10_only']['split_most'])\n"
        "print('first divergence (all):', results['all']['first_divergence_counts'])"),
    nbformat.v4.new_code_cell(
        "import matplotlib.pyplot as plt\n"
        "df = pd.DataFrame(results['all']['pooled']).set_index('type')['mean_agreement_over_tasks']\n"
        "df8 = pd.DataFrame(results['v8_only']['pooled']).set_index('type')['mean_agreement_over_tasks']\n"
        "order = df.sort_values().index\nfig, ax = plt.subplots(figsize=(7, 5))\n"
        "y = range(len(order))\nax.barh([i + 0.2 for i in y], df[order], height=0.4, label='all runs')\n"
        "ax.barh([i - 0.2 for i in y], df8[order], height=0.4, label='v8 only')\n"
        "ax.set_yticks(list(y)); ax.set_yticklabels(order); ax.set_xlim(0, 1)\n"
        "ax.set_xlabel('pooled agreement (mean over tasks)'); ax.legend(); plt.tight_layout()"),
    nbformat.v4.new_markdown_cell("## Per-task agreement (all runs)"),
    nbformat.v4.new_code_cell("pd.DataFrame(results['all']['per_task']).set_index('task').round(2)"),
    nbformat.v4.new_markdown_cell("## Example extracted decisions for one task (to eyeball the extractor)"),
    nbformat.v4.new_code_cell(
        "t = 5\npd.set_option('display.max_colwidth', 80)\n"
        "pd.DataFrame({r['arm']: r['decisions'] for r in recs.values() if r['task'] == t}).T"),
    nbformat.v4.new_markdown_cell(
        "## What the old memory showed the agent (descriptive)\n\n"
        "Rules (see `quick_runs/p0/memory_use.py`): a code memory is *followed* when the next code has the same "
        "design signature; a design-digest item is *followed* when the next design names a column its "
        "`try_next` suggested. This cell re-reads the event logs of the digest runs (about a minute)."),
    nbformat.v4.new_code_cell(
        "mem = agree.memory_summary()\n"
        "assert json.loads(json.dumps(mem)) == saved['memory_use'], 'memory numbers differ from summary.json'\n"
        "print('memory numbers match summary.json')\n"
        "print(json.dumps({k: v for k, v in mem.items() if k != 'examples'}, indent=1))"),
    nbformat.v4.new_code_cell("pd.set_option('display.max_colwidth', 220)\npd.DataFrame(mem['examples'])"),
]
nb = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3",
                                                                    "language": "python"}})
NB.parent.mkdir(parents=True, exist_ok=True)
NotebookClient(nb, timeout=1200, kernel_name="python3", resources={"metadata": {"path": str(NB.parent)}}).execute()
nbformat.write(nb, NB)
print("wrote", NB)
