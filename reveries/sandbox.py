"""Stateful Python executor for the agent (one worker process per run).

The worker runs inside the run directory, which holds read-only copies of the task's
data files. Variables persist between cells, like a notebook. A cell that runs past
the timeout kills the worker; a fresh one is started and the agent is told that its
state was lost.
"""
from __future__ import annotations

import json
import os
import select
import subprocess
import sys
from pathlib import Path

WORKER = r'''
import sys, json, io, traceback, contextlib, os
import warnings
warnings.filterwarnings("ignore")
ns = {"__name__": "__main__"}
for line in sys.stdin:
    code = json.loads(line)["code"]
    buf = io.StringIO()
    ok = True
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            exec(compile(code, "<cell>", "exec"), ns)
        except BaseException:
            ok = False
            traceback.print_exc()
    sys.__stdout__.write(json.dumps({"ok": ok, "output": buf.getvalue()}) + "\n")
    sys.__stdout__.flush()
'''

MAX_OUTPUT_CHARS = 6000


class Sandbox:
    def __init__(self, workdir: Path, timeout: float = 300.0):
        self.workdir = Path(workdir)
        self.timeout = timeout
        self.proc: subprocess.Popen | None = None

    def _start(self):
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg",
               "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}
        tmp = os.environ.get("REVERIES_TMPDIR")  # route sandbox temp off a full root fs (e.g. onto /data)
        if tmp:
            os.makedirs(tmp, exist_ok=True)
            env.update({"TMPDIR": tmp, "TMP": tmp, "TEMP": tmp,
                        "MPLCONFIGDIR": f"{tmp}/mpl", "XDG_CACHE_HOME": f"{tmp}/cache"})
        self.proc = subprocess.Popen(
            [sys.executable, "-I", "-c", WORKER], cwd=self.workdir, env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )

    def run(self, code: str) -> dict:
        """Returns {"ok": bool, "output": str, "timed_out": bool, "restarted": bool}."""
        restarted = False
        if self.proc is None or self.proc.poll() is not None:
            restarted = self.proc is not None
            self._start()
        self.proc.stdin.write(json.dumps({"code": code}) + "\n")
        self.proc.stdin.flush()
        ready, _, _ = select.select([self.proc.stdout], [], [], self.timeout)
        if not ready:
            self.close()
            return {"ok": False, "output": f"TIMEOUT after {self.timeout:.0f}s; the Python state was reset.",
                    "timed_out": True, "restarted": True}
        line = self.proc.stdout.readline()
        if not line:
            self.close()
            return {"ok": False, "output": "The Python process crashed; the state was reset.",
                    "timed_out": False, "restarted": True}
        res = json.loads(line)
        out = res["output"]
        if len(out) > MAX_OUTPUT_CHARS:
            out = out[: MAX_OUTPUT_CHARS // 2] + f"\n... [{len(out) - MAX_OUTPUT_CHARS} chars cut] ...\n" + out[-MAX_OUTPUT_CHARS // 2:]
        return {"ok": res["ok"], "output": out, "timed_out": False, "restarted": restarted}

    def close(self):
        if self.proc is not None:
            self.proc.kill()
            self.proc.wait()
            self.proc = None
