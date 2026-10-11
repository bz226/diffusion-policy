"""Small Amendment G-1 process fixtures; no environments or scientific sampling."""
import ast
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

from agent.finetune import grad_mean as gm

ROOT = Path(__file__).resolve().parent
SOURCE = Path(gm.__file__).resolve()
CONTEXT = mp.get_context("fork")


def waiting(event):
    event.wait()


class Env:
    def __init__(self, number=1, behavior="normal"):
        self.behavior = behavior
        self.event = CONTEXT.Event()
        self.release_close = threading.Event()
        self.processes = [CONTEXT.Process(target=waiting, args=(self.event,)) for _ in range(number)]
        for process in self.processes:
            process.start()

    def close(self):
        if self.behavior == "raise":
            raise ValueError("intentional synthetic close failure")
        if self.behavior == "hang":
            self.release_close.wait()
            return
        self.event.set()
        for process in self.processes:
            process.join()


def agent(name, env):
    path = ROOT / name
    path.mkdir()
    return SimpleNamespace(logdir=str(path), venv=env)


def record(a):
    return json.loads((Path(a.logdir) / "cleanup.json").read_text())


def check_reaped(result):
    assert result["all_children_exited"] is True
    assert not result["remaining_direct_children"] and not result["remaining_active_children"]
    assert all(not row["alive"] for row in result["workers"])


def main():
    assert not (ROOT / "unit_report.json").exists()
    source = SOURCE.read_text()
    ast.parse(source, feature_version=(3, 8))
    (ROOT / "grad_mean_source.py").write_text(source)
    checks = []
    a = agent("normal", Env(2))
    result = gm.cleanup_environment(a, close_timeout=.2, join_timeout=.2)
    check_reaped(result)
    assert result["status"] == "passed" and len(result["workers"]) == 2
    assert all(row["exitcode"] == 0 for row in result["workers"])
    checks.append("graceful close and zero-exit joins")

    original = {name: getattr(gm, name) for name in ("settings", "_validate", "seed_environments", "collect", "cleanup_environment")}
    class PrimaryFailure(Exception):
        pass
    def primary(a, cfg):
        raise PrimaryFailure("intentional synthetic collection failure")
    try:
        gm.settings = lambda a: {"gm_step": "collect"}
        gm._validate = lambda a, cfg: None
        gm.seed_environments = lambda a: None
        gm.collect = primary
        gm.cleanup_environment = lambda a, primary_error=None: original["cleanup_environment"](
            a, primary_error=primary_error, close_timeout=.2, join_timeout=.2)
        a = agent("primary_failure", Env())
        try:
            gm.run_grad_mean(a)
        except PrimaryFailure:
            pass
        else:
            raise AssertionError("primary error was suppressed")
        result = record(a)
        check_reaped(result)
        assert result["status"] == "passed" and result["primary_error"]["type"] == "PrimaryFailure"
        checks.append("primary traceback retained after successful cleanup")
        a = agent("primary_and_close_failure", Env(behavior="raise"))
        try:
            gm.run_grad_mean(a)
        except RuntimeError as error:
            assert isinstance(error.__cause__, PrimaryFailure)
        else:
            raise AssertionError("cleanup error was suppressed")
        result = record(a)
        check_reaped(result)
        assert result["status"] == "failed" and result["errors"] and result["actions"]
        assert "PrimaryFailure" in result["primary_error"]["traceback"]
        checks.append("close failure terminates own workers and chains primary traceback")
    finally:
        for name, value in original.items():
            setattr(gm, name, value)

    env = Env(behavior="hang")
    a = agent("blocked_close", env)
    try:
        gm.cleanup_environment(a, close_timeout=.1, join_timeout=.1)
    except RuntimeError:
        pass
    else:
        raise AssertionError("blocked close incorrectly passed")
    finally:
        env.release_close.set()
    result = record(a)
    check_reaped(result)
    assert result["status"] == "failed" and any(row["type"] == "TimeoutError" for row in result["errors"])
    checks.append("blocked close is bounded and workers are reaped")

    process = subprocess.Popen([sys.executable, "-B", "-c", "import time; time.sleep(60)"])
    try:
        a = agent("other_owned_child", Env(0))
        try:
            gm.cleanup_environment(a, close_timeout=.1, join_timeout=.2)
        except RuntimeError:
            pass
        else:
            raise AssertionError("unexpected owned child incorrectly passed")
        result = record(a)
        check_reaped(result)
        assert any(row["pid"] == process.pid for row in result["actions"])
        checks.append("non-multiprocessing direct child ownership checked and reaped")
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()

    class NotOwned:
        pid = 99999999
        _parent_pid = os.getpid() + 1
        def terminate(self):
            raise AssertionError("non-owned process signalled")
        kill = terminate
    env = Env(0)
    env.processes = [NotOwned()]
    env.close = lambda: None
    a = agent("non_owned_guard", env)
    try:
        gm.cleanup_environment(a, close_timeout=.1, join_timeout=.1)
    except RuntimeError:
        pass
    else:
        raise AssertionError("non-owned process was accepted")
    result = record(a)
    check_reaped(result)
    assert result["status"] == "failed" and not result["actions"]
    checks.append("non-owned process is never signalled")
    assert not mp.active_children() and not gm._direct_children()
    report = {"status": "passed", "checks": checks, "source": str(SOURCE),
              "command": [sys.executable] + sys.argv, "cwd": str(Path.cwd()),
              "PYTHONPATH": os.environ.get("PYTHONPATH"), "python": sys.version,
              "scope": "synthetic child-process cleanup fixtures only; no environments, gradients, training, or scientific gates",
              "expected_failure_cases": ["primary_failure", "primary_and_close_failure", "blocked_close", "other_owned_child", "non_owned_guard"],
              "no_remaining_children": True}
    (ROOT / "unit_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS: six synthetic cleanup checks; no children remain; no scientific execution")


if __name__ == "__main__":
    main()
