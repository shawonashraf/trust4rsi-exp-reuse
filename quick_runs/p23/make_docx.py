"""Build the SPIKE token-budget docx with charts from the measured DeepSeek ledgers.
  uv run --no-active --with python-docx python quick_runs/p23/make_docx.py
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

OUT = Path("quick_runs/results/P23"); OUT.mkdir(parents=True, exist_ok=True)
# measured per-run means (prompt, completion)
GUIDE = dict(prompt=3434, completion=7146)       # DeepSeek guide, 70 runs
AGENT = dict(prompt=9368, completion=7050)       # DeepSeek agent, 81 runs
CLAUDE = {"Opus 5.5": 2217, "Fable 5.1": 3383, "DeepSeek": 7146}  # completion per guide run
WEEKLY_P, WEEKLY_C = 110_000_000, 11_000_000
BLUE, ORANGE, GREY, GREEN, RED = "#2E5E9C", "#E08214", "#8C8C8C", "#2E8B57", "#C0392B"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "figure.dpi": 150})


def grouped(ax, cats, series, colors):
    x = np.arange(len(cats)); w = 0.8 / len(series)
    for i, (name, vals) in enumerate(series.items()):
        b = ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=name, color=colors[i])
        for r in b:
            ax.annotate(f"{r.get_height():,.0f}", (r.get_x() + r.get_width() / 2, r.get_height()),
                        ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(cats)


# Chart 1: per-run tokens, guide vs agent
fig, ax = plt.subplots(figsize=(6.4, 3.6))
grouped(ax, ["Guide run", "Agent run"],
        {"Prompt": [GUIDE["prompt"], AGENT["prompt"]], "Completion": [GUIDE["completion"], AGENT["completion"]]},
        [BLUE, ORANGE])
ax.set_ylabel("tokens per run"); ax.set_title("DeepSeek tokens per run: guide vs agent"); ax.legend()
fig.tight_layout(); fig.savefig(OUT / "c1_per_run.png"); plt.close(fig)

# Chart 2: weekly capacity (completion-bound)
fig, ax = plt.subplots(figsize=(6.4, 3.6))
caps = {"Guide only": WEEKLY_C // GUIDE["completion"], "Agent only": WEEKLY_C // AGENT["completion"],
        "Guide + agent": WEEKLY_C // (GUIDE["completion"] + AGENT["completion"])}
b = ax.barh(list(caps), list(caps.values()), color=[BLUE, ORANGE, GREEN])
for r in b:
    ax.annotate(f"{int(r.get_width()):,} runs/wk", (r.get_width(), r.get_y() + r.get_height() / 2),
                ha="left", va="center", fontsize=10, xytext=(4, 0), textcoords="offset points")
ax.set_xlabel("runs per week within the 11M completion budget"); ax.set_xlim(0, 1800)
ax.set_title("Weekly capacity (completion is the binding budget)"); ax.invert_yaxis()
fig.tight_layout(); fig.savefig(OUT / "c2_capacity.png"); plt.close(fig)

# Chart 3: budget utilization by example experiment (% of weekly completion)
fig, ax = plt.subplots(figsize=(6.4, 3.6))
exp = {"Guide\n70 runs": 70 * GUIDE["completion"], "Agent\n40 runs": 40 * AGENT["completion"],
       "Guide\n140 runs": 140 * GUIDE["completion"], "Agent\n200 runs": 200 * AGENT["completion"]}
pct = [v / WEEKLY_C * 100 for v in exp.values()]
b = ax.bar(list(exp), pct, color=GREY)
for r, p in zip(b, pct):
    ax.annotate(f"{p:.1f}%", (r.get_x() + r.get_width() / 2, r.get_height()), ha="center", va="bottom", fontsize=9)
ax.axhline(100, color=RED, lw=1, ls="--")
ax.annotate("weekly budget (100%)", (3.4, 100), color=RED, va="bottom", ha="right", fontsize=9)
ax.set_ylabel("% of weekly completion budget (11M)"); ax.set_ylim(0, 110)
ax.set_title("Each experiment is a small slice of the weekly budget")
fig.tight_layout(); fig.savefig(OUT / "c3_utilization.png"); plt.close(fig)

# Chart 4: completion per guide run, DeepSeek vs Claude
fig, ax = plt.subplots(figsize=(6.4, 3.3))
b = ax.bar(list(CLAUDE), list(CLAUDE.values()), color=[GREY, GREY, BLUE])
for r in b:
    ax.annotate(f"{int(r.get_height()):,}", (r.get_x() + r.get_width() / 2, r.get_height()),
                ha="center", va="bottom", fontsize=9)
ax.set_ylabel("completion tokens per guide run")
ax.set_title("DeepSeek reasons more than Claude (per guide run)")
fig.tight_layout(); fig.savefig(OUT / "c4_vs_claude.png"); plt.close(fig)

# assemble the docx
d = Document()
d.styles["Normal"].font.name = "Calibri"; d.styles["Normal"].font.size = Pt(11)
d.add_heading("SPIKE Token Budget — DeepSeek Usage Probe", 0)
p = d.add_paragraph()
r = p.add_run("Measured token footprint of DeepSeek-V41-Flash as guide and as agent, against the light-tier weekly "
              "budget (110M prompt / 11M completion). Figures come from on-cluster token ledgers; completion tokens are "
              "the binding constraint.")
r.italic = True
d.add_heading("Summary", 1)
for t in [f"Guide run: {GUIDE['prompt']:,} prompt / {GUIDE['completion']:,} completion tokens (mean).",
          f"Agent run: {AGENT['prompt']:,} prompt / {AGENT['completion']:,} completion tokens (mean).",
          "Completion is the binding budget. The 11M/week ceiling allows about 1,500 guide or agent runs, or ~775 "
          "full guide+agent task-runs.",
          "A full experiment (70-140 runs) uses 4.5-9% of the weekly completion budget; the prompt budget is barely "
          "touched (under 1%).",
          "DeepSeek's completion per run is about 2-3x Claude's, because it reasons at length."]:
    d.add_paragraph(t, style="List Bullet")
for png, cap in [("c1_per_run.png", "Figure 1. Tokens per run, guide vs agent. The agent sends more prompt (the "
                                    "transcript grows each step); completion is similar."),
                 ("c2_capacity.png", "Figure 2. Weekly run capacity, set by the 11M completion budget."),
                 ("c3_utilization.png", "Figure 3. Each experiment as a share of the weekly completion budget; the "
                                        "dashed line marks 100%."),
                 ("c4_vs_claude.png", "Figure 4. Completion per guide run: DeepSeek vs Claude Opus 5.5 and Fable 5.1.")]:
    d.add_picture(str(OUT / png), width=Inches(5.6))
    d.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    c = d.add_paragraph(); cr = c.add_run(cap); cr.italic = True; cr.font.size = Pt(9.5)
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
d.add_heading("Takeaway for the budget request", 1)
d.add_paragraph("The current weekly budget is more than sufficient for a full DeepSeek guide or agent experiment, with "
                "headroom to repeat it many times. A larger completion budget is warranted only to run several such "
                "studies per week, add more seeds or arms, or use longer-context tasks. Size any request on completion "
                "tokens, not prompt.")
d.save(str(OUT / "spike_token_budget.docx"))
print("wrote", OUT / "spike_token_budget.docx")
