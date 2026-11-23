"""P0 step 5 (descriptive): what the old memory showed the agent, and whether the next design followed it.

Two kinds of memory item reached the Engineer (the agent that writes the experiment code) in the prompt
where it designs the next experiment:

1. Code memories (v8 memory arms): the "Relevant memories:" section lists past experiments verbatim,
   as "- Experiment for Hypothesis <id>:" followed by that experiment's code.
   Rule: reduce the shown code and the code the Engineer then wrote to a design signature
   (model families called + the identifiers in model formulas / feature lists).
     followed      = same signature
     ignored       = both signatures readable and different
     cannot tell   = either signature unreadable (no model call or no formula / feature list found)
2. Design-digest items (v10 and e01 arms): the "Designs already run on this task" section lists past
   designs with their outcome tally and a "try_next" suggestion from the analyst's reflection.
   Rule: take the dataset columns the try_next text names that the item's own design does not already
   use ("new columns").
     followed      = the design in the same prompt ("Design to implement") names at least one of them
     ignored       = it names none of them
     cannot tell   = the try_next text names no new column
   Only items about the same variables as the hypothesis being designed are judged.

Counts are per (prompt, item): an item shown in many prompts is judged each time it is shown.
"""
from __future__ import annotations

import re

FAMILY_PATTERNS = [
    (r"\bsmf\.logit\(|\bsm\.Logit\(|\bLogit\(|LogisticRegression\(", "logit"),
    (r"\bsmf\.probit\(|\bsm\.Probit\(|\bProbit\(", "probit"),
    (r"\bsmf\.ols\(|\bsm\.OLS\(|\bOLS\(|LinearRegression\(", "ols"),
    (r"mixedlm\(|MixedLM\(", "mixed"),
    (r"\bsmf\.glm\(|\bsm\.GLM\(|\bGLM\(", "glm"),
    (r"smf\.poisson\(|sm\.Poisson\(", "poisson"),
    (r"negativebinomial\(|NegativeBinomial\(", "negbin"),
    (r"ttest_ind\(|ttest_rel\(|ttest_1samp\(", "t_test"),
    (r"pearsonr\(|spearmanr\(|kendalltau\(", "correlation"),
    (r"chi2_contingency\(", "chi_square"),
    (r"f_oneway\(|anova_lm\(", "anova"),
    (r"mannwhitneyu\(|kruskal\(|wilcoxon\(", "nonparametric"),
    (r"quantreg\(|QuantReg\(", "quantile"),
    (r"PanelOLS\(|RandomEffects\(", "panel"),
    (r"RandomForest\w*\(|GradientBoosting\w*\(|XGB\w*\(", "ml"),
]
FORMULA_RE = re.compile(r"""["']([^"'\n]*~[^"'\n]*)["']""")
DOUBLE_LIST_RE = re.compile(r"\[\s*\[([^\[\]]+)\]\s*\]")
QUOTED_RE = re.compile(r"""["']([^"'\n]+)["']""")
IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
FORMULA_STOP = {"c", "np", "i", "q", "log", "log1p", "exp", "sqrt", "center", "standardize", "scale", "treatment",
                "sum", "poly", "bs", "cr", "reference", "levels"}


def design_signature(code: str):
    """(sorted families, sorted terms) or None when no model call or no terms can be read."""
    if not code:
        return None
    fams = sorted({name for pat, name in FAMILY_PATTERNS if re.search(pat, code)})
    terms = set()
    for f in FORMULA_RE.findall(code):
        terms |= {t.lower() for t in IDENT_RE.findall(f) if t.lower() not in FORMULA_STOP}
    for inner in DOUBLE_LIST_RE.findall(code):
        terms |= {q.lower() for q in QUOTED_RE.findall(inner)}
    if not fams or not terms:
        return None
    return (tuple(fams), tuple(sorted(terms)))


def judge_code_item(item_code: str, new_code: str) -> str:
    a, b = design_signature(item_code), design_signature(new_code)
    if a is None or b is None:
        return "cannot_tell"
    return "followed" if a == b else "ignored"


ITEM_RE = re.compile(r"^- (Experiment for Hypothesis|Results for Experiment) ([0-9a-f-]{36}):\s*$", re.M)


def code_items(section: str) -> list[dict]:
    """Split a "Relevant memories:" section into items; keeps the past-experiment code items."""
    if not section:
        return []
    ms = list(ITEM_RE.finditer(section))
    out = []
    for k, m in enumerate(ms):
        end = ms[k + 1].start() if k + 1 < len(ms) else len(section)
        if m.group(1) == "Experiment for Hypothesis":
            out.append({"hypothesis_id": m.group(2), "text": section[m.end():end].strip("\n")})
    return out


GROUP_RE = re.compile(r"^\((?P<pair>[^)]*)\) · (?P<design>.*?): (?P<n>\d+) attempts: (?P<tally>.*)$")


def digest_items(section: str) -> list[dict]:
    """Groups in a "Designs already run on this task" section, each with its try_next text (or None)."""
    if not section:
        return []
    out = []
    for line in section.split("\n"):
        m = GROUP_RE.match(line.strip())
        if m:
            out.append({"pair": [p.strip().lower() for p in m.group("pair").split(",")], "design": m.group("design"),
                        "tally": m.group("tally"), "try_next": None, "line": line.strip()})
        elif out and "try_next" in line:
            out[-1]["try_next"] = line.split("try_next", 1)[1].strip()
            out[-1]["reflection"] = line.strip()
    return out


def columns_named(text: str, columns: list[str]) -> set[str]:
    t = text or ""
    return {c for c in columns if re.search(rf"(?<![A-Za-z0-9_]){re.escape(c)}(?![A-Za-z0-9_])", t, re.I)}


def _alnum(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def judge_digest_item(item: dict, design_line: str, columns: list[str]) -> tuple[str, list[str]]:
    if not item.get("try_next"):
        return "cannot_tell", []
    pair = {_alnum(p) for p in item["pair"]}
    new = columns_named(item["try_next"], columns) - columns_named(item["design"], columns) \
        - {c for c in columns if _alnum(c) in pair}
    if not new:
        return "cannot_tell", []
    hit = new & columns_named(design_line, columns)
    return ("followed" if hit else "ignored"), sorted(new)


# ---------------------------------------------------------------- driver
VARS_RE = re.compile(r"^\s*Variables:\s*(.+)$", re.M)


def _short(line: str, n: int = 160) -> str:
    line = " ".join(line.split())
    return line if len(line) <= n else line[: n - 3] + "..."


def model_line(code: str) -> str:
    for ln in (code or "").split("\n"):
        if "~" in ln and ("'" in ln or '"' in ln):
            return _short(ln.strip())
    for ln in (code or "").split("\n"):
        if any(re.search(p, ln) for p, _ in FAMILY_PATTERNS):
            return _short(ln.strip())
    return ""


def analyse(facts_list: list[dict], queries: dict[str, list[dict]]) -> dict:
    """facts_list: cached P0 facts of the memory-arm runs. queries: key -> load_run(...)["engineer_calls"]
    (full prompts, needed for the 'Variables:' line of digest prompts)."""
    code_rows, digest_rows = [], []
    for f in facts_list:
        exps = f["experiments"]
        sig_by_exp = [(x["order"], x["hypothesis_id"], design_signature(x["code"])) for x in exps if x["success"]]
        full_q = {c["order"]: c["query"] for c in queries.get(f["key"], [])}
        for e in f["engineer"]:
            if not e["is_design_call"] or not e["code"]:
                continue
            nxt = next((x for x in exps if x["order"] > e["order"]), None)
            cur_h = nxt["hypothesis_id"] if nxt else None
            for it in code_items(e["relevant_memories"]):
                j = judge_code_item(it["text"], e["code"])
                new_sig = design_signature(e["code"])
                others = [s for o, h, s in sig_by_exp if o < e["order"] and s is not None and h != it["hypothesis_id"]]
                base = (sum(s == new_sig for s in others) / len(others)) if (others and new_sig) else None
                code_rows.append({"key": f["key"], "task": f["task"], "arm": f["arm"], "order": e["order"],
                                  "judgment": j, "same_hypothesis": it["hypothesis_id"] == cur_h, "base_rate": base,
                                  "item_model_line": model_line(it["text"]), "new_model_line": model_line(e["code"])})
            dsec = e.get("designs_already_run")
            if dsec:
                q = full_q.get(e["order"], "")
                m = VARS_RE.search(q)
                cur_vars = {_alnum(v) for v in m.group(1).split(",")} if m else set()
                dl = (e.get("design_to_implement") or "").split("\n")
                design_line = dl[1].strip() if len(dl) > 1 else ""
                for it in digest_items(dsec):
                    same = {_alnum(p) for p in it["pair"]} == cur_vars
                    j, new = judge_digest_item(it, design_line, f["columns"])
                    digest_rows.append({"key": f["key"], "task": f["task"], "arm": f["arm"], "order": e["order"],
                                        "same_variables": same, "judgment": j, "new_columns": new,
                                        "try_next": _short(it.get("try_next") or "", 200),
                                        "item_design": _short(it["design"], 160), "tally": it["tally"],
                                        "design_line": _short(design_line, 160)})
    return {"code_rows": code_rows, "digest_rows": digest_rows}
