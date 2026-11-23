# Can LLMs Guide Trustworthy Experience Reuse in Agents?

How to run the code and the experiments of the paper.

## Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/). Dependencies are pinned in `uv.lock`.
- Re-analysis of the reported results needs no GPU and no API key.
- New agent runs need Gemma 4 31B (`google/gemma-4-31B-it`) served with SGLang 0.5.19 behind an OpenAI-compatible
  endpoint, by default at `http://localhost:8100` and `http://localhost:8101` (paper: tensor parallelism over 2 or 4
  NVIDIA RTX A6000 48 GB GPUs). The Qwen guide is served the same way (`Qwen/Qwen3.8-27B-FP8`). The Claude guides need
  an Anthropic API key; the DeepSeek guide needs an OpenAI-compatible endpoint.

```bash
uv sync
```

## Configuration

| Variable | Contents |
| --- | --- |
| `REVERIES_DATA` | Benchmark data (see Data). |
| `REVERIES_RUNS` | Agent run records: events, executed code, judge verdicts, guide inputs. |
| `REVERIES_LEGACY` | Older runs, read only by the pre-study probes P0 and P7. |
| `REVERIES_TMPDIR` | Optional temp directory for the agent's sandbox. |

The defaults are placeholder paths (`/data/anon/...`), so set these variables. API keys go in `.env` (template:
`.env.example`): `ANTHROPIC_API_KEY` for the Claude guides, `SPIKE_API_KEY` and `SPIKE_BASE_URL` for the
OpenAI-compatible DeepSeek endpoint.

## Data

Place the benchmark data under `$REVERIES_DATA`:

| Folder | Source |
| --- | --- |
| `dsts/` | Silent Failures test-suite data and workloads (`hongqianx/tabular-agent-failures`, commit `a1957ea`) |
| `sf_release/` | Silent Failures release: leak-free TF04/TF06 data and judge criteria (`trl-lab/silent-failures-agentic-ds`) |
| `blade/` | BLADE datasets and expert annotations (`behavioral-data/BLADE`) |
| `dsbench_hf/` | DSBench data-analysis and data-modelling tasks |

The header of `quick_runs/tasks.yaml` lists the variant of each task that is used.

The agent run records behind the reported numbers (about 1,700 runs) are a separate dataset; its link is withheld
during review and will be added on publication.

## Experiment sets

Each set has a prediction file in `quick_runs/predictions/`, results in `quick_runs/results/<set>/` and a notebook
in `docs/notebooks/`.

| Set | Experiment |
| --- | --- |
| P12 | Baselines: no memory, BM25, all own lessons, padded contexts |
| P13 | Embedding retrieval, guided retrieval (retry budget 0/1/3), planted wrong lessons |
| P14 | Open guide prompt, guide reasons; judge check (`REVIEW_CHECKS`) |
| P15 | Qwen as guide |
| P16 | Controls: wording hidden, data profile, random of candidates; judge check (`REVIEW2_JUDGE`) |
| P17 | Store written without a checker |
| P18 | One sentence cut, cross-encoder, hint wording |
| P19 | Escalating guide, probe at write |
| P20 | Claude Opus 5.5 and Fable 5.1 as guide |
| P21 | Train-only probe at write, hardware split |
| P22 | Bias vs. obedience controls |
| P23 | DeepSeek as guide (report only) |

Arm names: `A0` no memory, `M` BM25 top-3, `E` embedding top-3, `G0`/`G1`/`G3` guided retrieval with retry budget
0/1/3, `C1` all own lessons; suffix `-p` is the store with the planted wrong lessons. All guide arms are defined in
`ARMS` in `quick_runs/p13/guide.py`; each set's prediction file gives its arms, stores and run counts.

## Re-run the analysis (no GPU)

```bash
export REVERIES_RUNS=/path/to/run_records/runs
export REVERIES_DATA=/path/to/reveries_data
uv run python quick_runs/notebooks.py P13     # executes docs/notebooks/quick_P13.ipynb, writes results/P13/summary.json
uv run python quick_runs/notebooks.py P12 P13 P14 P15 P16 P17 P18 P19 P20 P21 P22
```

Valid names are the keys of `NOTEBOOKS` in `quick_runs/notebooks.py`.

## Run an experiment (GPU)

Example for P13, guided retrieval on the evaluated store:

```bash
# 1. Write the lesson store from earlier runs. Overwrites the committed store used in the paper.
uv run --no-active python quick_runs/p12/lessons.py

# 2. Retrieve and run the guide per arm; writes $REVERIES_RUNS/P13_inputs/<arm>.yaml and the guide trace.
uv run --no-active python quick_runs/p13/guide.py --arms E G0 G1 G3 E-p G3-p

# 3. Run the agent with the selected lessons, 5 runs per task.
uv run python quick_runs/run.py --probe P13_G3 --tasks-file quick_runs/tasks.yaml \
    --hint-file $REVERIES_RUNS/P13_inputs/G3.yaml --hint-prefix "" --seeds 0 1 2 3 4

# 4. Judge the runs, then build the notebook.
uv run python quick_runs/score.py --probe P13_G3
uv run python quick_runs/notebooks.py P13
```

Options:

- `run.py`: `--tasks` (subset of task ids), `--servers` (model endpoints), `--concurrency`; runs with `run.json` and
  `executed.json` are skipped, so an interrupted set can be resumed.
- `guide.py`: `--guide-model` (served model name), `--guide-base-url` (OpenAI-compatible endpoint), or a `claude-*`
  model with `--budget` (hard USD cap) and `--ledger` (token and cost log); `--tags` sets the run tags.
- `score.py`: `--judge-claude <model>` judges with a Claude model instead of Gemma.
- Stores for P17 and P19 are written with `quick_runs/p17/write_nf.py` and `quick_runs/p19/write_nfd.py`; the P21
  store with `quick_runs/p19/write_nfd.py --train-only --out quick_runs/results/P21/lessons_nfdtr.jsonl`. Each takes
  `--servers http://localhost:8100`.

Sampling seeds are not set, so a re-run reproduces the distribution of outcomes, not identical runs.

## Tests

```bash
PYTHONPATH=. uv run pytest
```

`tests/test_p12_bm25.py` and `tests/test_p13_guide.py::test_p18_placebo_and_hint_texts` need the Gemma tokenizer in the
local Hugging Face cache and the data under `$REVERIES_DATA`; the other tests run without them.
