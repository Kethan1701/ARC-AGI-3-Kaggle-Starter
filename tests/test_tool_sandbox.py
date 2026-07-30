"""Tests for `run_python_tool`, which executes model-written code.

This is the surface the model acts through, so its failure modes matter more
than its happy path: code that silently does nothing, output that buries the
conversation, a loop that hangs the game thread, or -- worst, because a
Kaggle log would never reveal it -- one game's output appearing in another
game's transcript.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "ARC-AGI-3-Agents"
if not VENDOR.exists():  # pragma: no cover - depends on `make setup`
    pytest.skip("framework not vendored; run `make setup`", allow_module_level=True)
sys.path.insert(0, str(VENDOR))


def _load():
    spec = importlib.util.spec_from_file_location(
        "my_agent_sandbox_test", ROOT / "agent" / "my_agent.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


agent = _load()
run_python_tool = agent.run_python_tool


def test_print_is_captured_and_returned():
    out = run_python_tool("print('hello', 1 + 1)", {})
    assert out["stdout"] == "hello 2\n"
    assert out["error"] == ""


def test_result_assignment_comes_back():
    """TAAF's convention: assign a compact object to `result` rather than
    printing a wall of text."""
    out = run_python_tool("result = {'moves': ['LEFT', 'UP']}", {})
    assert out["result"] == {"moves": ["LEFT", "UP"]}
    assert out["error"] == ""


def test_the_allowed_modules_import():
    """The model is told it can write BFS, so the modules that takes have to
    actually be there."""
    code = (
        "import collections, heapq, itertools, math\n"
        "q = collections.deque([1])\n"
        "result = (len(q), heapq.nsmallest(1, [3, 1, 2]), math.isqrt(9))\n"
    )
    out = run_python_tool(code, {})
    assert out["error"] == ""
    assert out["result"] == (1, [1], 3)


@pytest.mark.parametrize("module", ["numpy", "os", "sys", "subprocess", "pathlib"])
def test_modules_outside_the_list_are_refused(module):
    """`numpy` and `os` for different reasons: numpy would not exist on the
    Kaggle notebook image the agent ships into, and os is simply not the
    model's business. Both should fail loudly, not half-work."""
    out = run_python_tool(f"import {module}", {})
    assert "not available in the tool sandbox" in out["error"]
    assert module in out["error"]


def test_file_and_eval_builtins_are_gone():
    for snippet in ("open('/etc/passwd')", "eval('1+1')", "exec('x=1')"):
        out = run_python_tool(snippet, {})
        assert out["error"], f"{snippet!r} should have failed"


def test_a_runaway_loop_is_stopped():
    """A `while True` from the model would otherwise hold the game thread
    until the whole run times out."""
    started = time.monotonic()
    out = run_python_tool("while True:\n    pass\n", {}, timeout=0.5)
    elapsed = time.monotonic() - started
    assert "exceeded" in out["error"]
    # Generous bound: the point is that it returns at all, not that the
    # deadline is precise. Line-granular tracing cannot be exact.
    assert elapsed < 10.0


def test_output_is_capped():
    """`segment_frame` on a noisy board is 2184 nodes. Without a cap one tool
    call evicts the model's own reasoning from its context."""
    out = run_python_tool("for i in range(10000):\n    print('x' * 50)\n", {}, output_limit=200)
    assert out["truncated"] is True
    assert "output truncated" in out["stdout"]
    assert len(out["stdout"]) < 400


def test_stdout_is_never_touched(capsys):
    """The concurrency claim in the docstring, pinned. 25 agents share one
    process; if this ever redirected `sys.stdout`, one game's tool output
    would surface in another game's transcript."""
    run_python_tool("print('should not escape')", {})
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_the_namespace_is_visible_and_writable():
    """The model reads frame views out of the namespace and the caller reads
    its conclusions back, so the traffic has to work in both directions."""
    namespace = {"valid_actions": ["LEFT", "RIGHT"]}
    out = run_python_tool(
        "chosen = valid_actions[0]\nresult = chosen\n", namespace
    )
    assert out["result"] == "LEFT"
    assert namespace["chosen"] == "LEFT"


def test_action_is_callable_from_inside_the_code():
    """The whole point of the port: the model executes actions from within its
    own search rather than returning one action per turn."""
    executed: list[list[str]] = []

    def action(actions):
        executed.append(list(actions))
        return {"board_changed": True}

    code = (
        "for step in ['LEFT', 'LEFT', 'UP']:\n"
        "    r = action([step])\n"
        "result = r['board_changed']\n"
    )
    out = run_python_tool(code, {"action": action})
    assert out["error"] == ""
    assert executed == [["LEFT"], ["LEFT"], ["UP"]]
    assert out["result"] is True


def test_a_syntax_error_is_reported_not_raised():
    # Note "this is not python" would NOT do: it parses as `this is (not
    # python)` and fails at run time with NameError instead.
    out = run_python_tool("def broken(:\n    pass\n", {})
    assert "SyntaxError" in out["error"]


def test_an_undefined_name_is_reported_too():
    out = run_python_tool("this is not python", {})
    assert "NameError" in out["error"]


def test_a_runtime_error_comes_back_as_a_traceback():
    """The model has to be able to fix its own code from what it gets."""
    out = run_python_tool("x = 1 / 0", {})
    assert "ZeroDivisionError" in out["error"]


def test_the_tracer_is_restored_afterwards():
    """`sys.settrace` is per-thread and global to it. Leaking a trace
    function would quietly slow every later action in that game."""
    before = sys.gettrace()
    run_python_tool("result = 1", {})
    assert sys.gettrace() is before
    run_python_tool("raise ValueError('boom')", {})
    assert sys.gettrace() is before
