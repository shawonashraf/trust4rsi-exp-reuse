"""P7 core: replay a run's attempts through a dead-end ledger.

Definitions (from the old run files):
- Attempt: one trajectory record in data/trajectories.jsonl, i.e. one hypothesis tested by one
  executed experiment and then analysed. The old code writes one `trajectory_recorded` event per
  record, in the same order, so file order is time order.
- Outcome class of an attempt, first match wins:
    error        experiment.execution_status != "success"
    rejected     outcome == "rejected"
    null         corrector.null_result is true (the corrector found no effect)
    inconclusive outcome == "inconclusive"
    supported    otherwise
  "Dead" = error, rejected or null.
- Ledger key: the variables tested (pair_columns) plus the design fingerprint the old run wrote
  (src/memory/design_fingerprint.py: model family, controls, moderators, transformations, scope).
  An attempt whose fingerprint is "none" (no design recorded) has no key: the ledger cannot see it.
- Retry: an attempt whose key already appeared earlier in the run.
- Blocked: an attempt whose key already had a dead attempt earlier in the run. A blocked attempt
  would not run, so it adds nothing to the ledger (its key is already marked dead).
- Old "dead-end retry" (the 21 / 30 in the old notebook): the latest earlier attempt with the
  same key was rejected or null. Recomputed here to check we read the files the same way.
"""
from __future__ import annotations

import re

DEAD = ("error", "rejected", "null")


def outcome_class(rec: dict) -> str:
    if (rec.get("experiment") or {}).get("execution_status", "success") != "success":
        return "error"
    if rec.get("outcome") == "rejected":
        return "rejected"
    if (rec.get("corrector") or {}).get("null_result"):
        return "null"
    if rec.get("outcome") == "inconclusive":
        return "inconclusive"
    return "supported"


def fingerprint_key(rec: dict):
    fp = rec.get("design_fingerprint") or "none"
    if fp == "none":
        return None
    return (",".join(rec.get("pair_columns") or []), fp)


def norm_family(f: str | None) -> str:
    s = re.sub(r"[^a-z0-9]+", " ", str(f or "").lower()).strip()
    if "logit" in s or "logistic" in s:
        return "logit"
    if "probit" in s:
        return "probit"
    if "pgls" in s or "phylogenetic" in s:
        return "pgls"
    if s in ("ols", "linear regression", "linear", "lm") or s.startswith("ols") or "least squares" in s:
        return "ols"
    return s


def _norm_items(xs) -> set[str]:
    out = set()
    for x in xs or []:
        s = re.sub(r"[\s_]+", "", str(x).lower())
        if s and s not in ("none", "-", "n/a", "na", "null", "nil"):
            out.add(s)
    return out


def design_parts(rec: dict):
    d = rec.get("design_used") or rec.get("design_spec") or {}
    return (norm_family(d.get("model_family")), _norm_items(d.get("controls")), _norm_items(d.get("moderators")),
            _norm_items(d.get("transformations")))


def coarse_key(rec: dict):
    """Plan's coarser key: variables, model family (aliases merged), controls. Scope, moderators and
    transformations are ignored. None when no model family is recorded."""
    fam, controls, _, _ = design_parts(rec)
    if not fam:
        return None
    return (",".join(rec.get("pair_columns") or []), fam, ",".join(sorted(controls)))


def replay(recs: list[dict], key_fn=fingerprint_key) -> list[dict]:
    seen: dict = {}       # key -> outcome classes of earlier attempts the ledger let run
    real: dict = {}       # old group (pair, fingerprint incl. "none") -> real earlier records
    out = []
    for i, r in enumerate(recs):
        cls = outcome_class(r)
        k = key_fn(r)
        prior = seen.get(k, []) if k is not None else []
        blocked = any(c in DEAD for c in prior)
        g = (",".join(r.get("pair_columns") or []), r.get("design_fingerprint") or "none")
        last = real.get(g)
        old = last is not None and (last.get("outcome") in ("null", "rejected")
                                    or bool((last.get("corrector") or {}).get("null_result")))
        out.append({"i": i, "key": k, "cls": cls, "retry": bool(prior), "blocked": blocked,
                    "old_dead_end_retry": old})
        real[g] = r
        if k is not None and not blocked:
            seen.setdefault(k, []).append(cls)
    return out


def old_dead_end_retries(recs: list[dict]) -> int:
    """The old notebook's count (cell 60): per (pair_columns, design_fingerprint) group, including
    'none', each record whose previous record in the group was rejected or null."""
    groups: dict = {}
    for r in recs:
        groups.setdefault((",".join(r["pair_columns"]), r["design_fingerprint"]), []).append(r)
    n = 0
    for rs in groups.values():
        for prev, _ in zip(rs, rs[1:]):
            if prev["outcome"] in ("null", "rejected") or prev["corrector"]["null_result"]:
                n += 1
    return n


def near_duplicate(a: dict, b: dict) -> str | None:
    """How attempt b nearly repeats earlier attempt a with a different fingerprint, or None.
    'same_design_other_wording': same variables, family, controls, moderators, transformations once
       normalised (so only scope wording or spelling made the fingerprint differ).
    'one_element_off': same variables and family, controls+moderators+transformations differ by one item.
    'no_design_recorded': b has no fingerprint ('none') and tests the same variables."""
    if a.get("pair_columns") != b.get("pair_columns"):
        return None
    if (b.get("design_fingerprint") or "none") == "none":
        return "no_design_recorded"
    if a.get("design_fingerprint") == b.get("design_fingerprint"):
        return None
    fa, ca, ma, ta = design_parts(a)
    fb, cb, mb, tb = design_parts(b)
    if not fa or fa != fb:
        return None
    if (ca, ma, ta) == (cb, mb, tb):
        return "same_design_other_wording"
    diff = (ca ^ cb) | (ma ^ mb) | (ta ^ tb)
    if len(diff) == 1:
        return "one_element_off"
    return None


def missed_near_duplicates(recs: list[dict]) -> list[dict]:
    """Attempts the fingerprint ledger lets through although an earlier DEAD attempt is a near-duplicate.
    Each missed attempt is counted once, with the closest kind found (wording > one element > none)."""
    rank = {"same_design_other_wording": 0, "one_element_off": 1, "no_design_recorded": 2}
    rep = replay(recs)
    out = []
    for j, b in enumerate(recs):
        if rep[j]["blocked"]:
            continue
        best = None
        for i in range(j):
            if rep[i]["blocked"] or rep[i]["cls"] not in DEAD:
                continue
            kind = near_duplicate(recs[i], b)
            if kind and (best is None or rank[kind] < rank[best[0]]):
                best = (kind, i)
        if best:
            out.append({"j": j, "i": best[1], "kind": best[0]})
    return out
