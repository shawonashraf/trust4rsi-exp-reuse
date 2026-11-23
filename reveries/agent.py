"""Single ReAct data-science agent with a typed decision log (light G03).

Loop: the model writes DECISION lines and one python block; we run the block and
return the output; repeat until the model writes FINAL ANSWER or the step budget
runs out. Every step is written to events.jsonl in the run dir.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .decisions import DECISION_TYPES, parse_decisions
from .llm import LLM
from .sandbox import Sandbox
from .tasks import Task

SYSTEM_PROMPT = """You are a careful data scientist. You work in a stateful Python environment
(variables persist between steps). The data files are in the current working directory.
pandas, numpy, scipy, statsmodels, scikit-learn, xgboost and imbalanced-learn are installed.

How to work, at every step:
1. Before the code that carries out an analytic decision, state the decision on its own line:
   DECISION type=<type> value=<value> reason=<short reason>
   <type> is one of: {types}.
   State every decision you make, including ones that seem obvious (e.g. which rows you drop,
   which columns you use, which model, how you validate, which metric you report).
2. Then write exactly one ```python code block and stop. You will see its output next.
3. When you are finished, write a line starting with FINAL ANSWER: followed by your answer.
   Do not write code in the same message as the final answer.
""".format(types=", ".join(DECISION_TYPES))

CODE_RE = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.S)
# Generation stops at the end of the first code block, so the model cannot write
# several blocks and invent their outputs in one message (seen in the first smoke run).
STOP = ["```\n", "```\r\n"]


def close_open_block(text: str) -> str:
    """The stop string is not returned; put the closing fence back."""
    if text.count("```") % 2 == 1:
        text = text.rstrip() + "\n```"
    return text
FINAL_RE = re.compile(r"FINAL ANSWER:\s*(.*)", re.S)


@dataclass
class Hint:
    text: str
    where: str = "start"           # "start" or "decision"
    decision_type: str | None = None  # for where="decision"
    prefix: str = "Note from previous experience: "  # put before the text at the start of the prompt


def initial_user_message(task, hint: "Hint | None") -> str:
    """The first user message: task question + schema, with a start hint (if any) in front."""
    user = f"{task.question}\n\nData schema (computed from the files):\n{task.schema}"
    if hint and hint.where == "start":
        user = f"{hint.prefix}{hint.text}\n\n{user}"
    return user


BeforeExecute = Callable[[str, list[dict]], "str | None"]  # returns a block reason, or None to allow


def noop_before_execute(code: str, decisions: list[dict]) -> str | None:
    return None


def last_json_object(text: str) -> dict | None:
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", text))):
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
    return None


class Agent:
    def __init__(self, llm: LLM, max_steps: int = 25, cell_timeout: float = 300.0,
                 before_execute: BeforeExecute = noop_before_execute, hint: Hint | None = None):
        self.llm = llm
        self.max_steps = max_steps
        self.cell_timeout = cell_timeout
        self.before_execute = before_execute
        self.hint = hint

    def run(self, task: Task, run_dir: Path, seed: int, meta: dict | None = None) -> dict:
        run_dir = Path(run_dir)
        task.setup_run_dir(run_dir / "work")
        log = open(run_dir / "events.jsonl", "a")

        def event(kind: str, **kw):
            log.write(json.dumps({"t": time.time(), "event": kind, **kw}, default=str) + "\n")
            log.flush()

        user = initial_user_message(task, self.hint)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
        event("run_start", task_id=task.task_id, seed=seed, llm=self.llm.base_url, model=self.llm.model,
              sampling=self.llm.sampling, thinking=self.llm.thinking, max_steps=self.max_steps,
              hint=None if self.hint is None else vars(self.hint), prompt=messages, meta=meta or {})

        sandbox = Sandbox(run_dir / "work", timeout=self.cell_timeout)
        final, stop, hint_given = None, "max_steps", False
        tokens = {"prompt": 0, "completion": 0, "reasoning": 0}
        try:
            for step in range(self.max_steps):
                reply = self.llm.chat(messages, seed=seed * 1000 + step, stop=STOP)
                for k, uk in (("prompt", "prompt_tokens"), ("completion", "completion_tokens"), ("reasoning", "reasoning_tokens")):
                    tokens[k] += reply.usage.get(uk) or 0
                text = close_open_block(reply.content)
                decisions = parse_decisions(text)
                event("llm_call", step=step, content=text, reasoning=reply.reasoning, usage=reply.usage,
                      finish_reason=reply.finish_reason, seconds=reply.seconds, request=reply.request)
                for d in decisions:
                    event("decision_stated", step=step, **d)
                messages.append({"role": "assistant", "content": text})
                codes = CODE_RE.findall(text)
                fm = FINAL_RE.search(text)
                if fm and not codes:
                    final = fm.group(1).strip()
                    event("final_answer", step=step, text=final, parsed=last_json_object(final))
                    stop = "final_answer"
                    break
                if not codes:
                    note = ("No python code block found. Write one ```python block, or write "
                            "'FINAL ANSWER:' if you are done.")
                    if reply.finish_reason == "length":
                        note = "Your message was cut off (too long). Write a shorter step with one ```python block."
                    event("no_code", step=step, finish_reason=reply.finish_reason)
                    messages.append({"role": "user", "content": note})
                    continue
                if (self.hint and self.hint.where == "decision" and not hint_given
                        and any(d["type"] == self.hint.decision_type for d in decisions)):
                    hint_given = True
                    event("hint_inserted", step=step, decision_type=self.hint.decision_type)
                    messages.append({"role": "user", "content": f"Before you carry out this {self.hint.decision_type} "
                                     f"decision, note from previous experience: {self.hint.text}\n"
                                     "The code above was NOT run. Restate your decision and code."})
                    continue
                code = codes[0]
                block = self.before_execute(code, decisions)
                event("hook_called", step=step, hook="before_execute", blocked=block)
                if block:
                    messages.append({"role": "user", "content": f"The code above was NOT run. Reason: {block}"})
                    continue
                res = sandbox.run(code)
                event("code_executed", step=step, code=code, **res)
                messages.append({"role": "user", "content": f"Output:\n{res['output'] or '(no output)'}"})
        except Exception as e:
            stop = f"error: {e!r}"
            event("run_error", error=repr(e))
        finally:
            sandbox.close()
        summary = {"task_id": task.task_id, "seed": seed, "stop": stop, "final_answer": final,
                   "final_json": last_json_object(final or ""), "steps": step + 1, "tokens": tokens}
        event("run_end", **summary)
        log.close()
        (run_dir / "run.json").write_text(json.dumps(summary, indent=1, default=str))
        return summary
