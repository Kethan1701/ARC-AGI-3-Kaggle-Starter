"""Unit tests for the per-game memory now living in `agent/my_agent.py`.

Run with:  .venv/bin/python -m pytest tests/ -q

These replace the `if __name__ == "__main__":` smoke test that used to sit
at the bottom of a standalone `game_memory.py`. The runtime classes moved
into `agent/my_agent.py` because `scripts/build_notebook.py` splices that
one hardcoded path into the submission notebook -- a sibling module under
`agent/` would import fine locally and raise ModuleNotFoundError on the
Kaggle kernel, i.e. only after a submission slot had been spent.

The three things asserted here are the three claims the memory makes:

  1. after a reset, a solved level replays from cache with ZERO exploration
     (this is the documented score leak, and the reason the module exists);
  2. a solved path survives a MemoryStore round-trip to disk;
  3. `predict_frame` hands back a cached frame, so the planner can look
     ahead without spending a squared-cost action.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "ARC-AGI-3-Agents"
if not VENDOR.exists():  # pragma: no cover - depends on `make setup`
    pytest.skip("framework not vendored; run `make setup`", allow_module_level=True)
sys.path.insert(0, str(VENDOR))


def _load():
    """Import agent/my_agent.py by path, as tests/test_my_agent.py does.

    `agent/` is not a package (no __init__.py) and the vendored framework
    has to be on sys.path before the module body runs, so a plain
    `from agent.my_agent import ...` is not available here. This loads the
    same file and yields the same classes.
    """
    spec = importlib.util.spec_from_file_location(
        "my_agent_memory_under_test", ROOT / "agent" / "my_agent.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


agent = _load()

GameMemory = agent.GameMemory
MemoryStore = agent.MemoryStore
fingerprint = agent.fingerprint


# --------------------------------------------------------------------- #
# A toy deterministic env standing in for the SDK: a 1-D corridor. The
# frame is the agent's position rendered as a small tuple grid, which is
# fingerprintable exactly like a real 64x64 board.
# --------------------------------------------------------------------- #
GOAL = 4
ACTIONS = ["LEFT", "RIGHT"]


def render(pos: int) -> tuple[int, ...]:
    return tuple(1 if i == pos else 0 for i in range(5))


def step(pos: int, action: str) -> int:
    return min(pos + 1, GOAL) if action == "RIGHT" else max(pos - 1, 0)


def _solve_by_exploring(mem: GameMemory, level: int = 1) -> list[str]:
    """Walk the corridor the expensive way, recording every transition."""
    pos = 0
    start_frame = render(pos)
    path: list[str] = []
    while pos != GOAL and len(path) <= 20:
        frame = render(pos)
        untried = mem.untried_actions(frame, ACTIONS)
        action = untried[0] if untried else "RIGHT"
        nxt = step(pos, action)
        mem.record(frame, action, render(nxt))
        path.append(action)
        pos = nxt
    assert pos == GOAL, "the toy explorer failed to reach the goal"
    mem.mark_solved(level, start_frame, path)
    return path


@pytest.fixture()
def solved_memory() -> GameMemory:
    mem = GameMemory("corridor")
    _solve_by_exploring(mem)
    return mem


# --------------------------------------------------------------------- #
# 1. the leak: a reset must replay, not re-explore
# --------------------------------------------------------------------- #
def test_reset_replays_cached_path_with_zero_exploration(solved_memory):
    mem = solved_memory
    edges_before = dict(mem.transitions)
    tried_before = {fp: set(t) for fp, t in mem.tried.items()}

    # A reset drops us back onto level 1 at the start state.
    replay = mem.solved_path(1, render(0))
    assert replay is not None, "cache miss -- the agent would have re-explored"

    pos = 0
    for action in replay:
        pos = step(pos, action)
    assert pos == GOAL, "replay did not reproduce the solve"

    # Zero exploration: replaying observed nothing new. No transition was
    # added and no state gained a newly-tried action, which is what
    # distinguishes a replay from a second search.
    assert mem.transitions == edges_before
    assert {fp: set(t) for fp, t in mem.tried.items()} == tried_before

    # And it costs exactly the banked number of committed actions.
    assert len(replay) == mem.solved[1]["count"]


def test_solved_path_refuses_a_mismatched_start_state(solved_memory):
    # The determinism guard: replaying from the wrong board would desync,
    # so a fingerprint mismatch must fall back to exploration.
    assert solved_memory.solved_path(1, render(2)) is None
    assert solved_memory.solved_path(99, render(0)) is None


def test_mark_solved_keeps_the_shortest_path(solved_memory):
    short = ["RIGHT"] * 4
    solved_memory.mark_solved(1, render(0), short)
    assert solved_memory.solved[1]["count"] == len(short)
    # A longer route for the same level must not overwrite it.
    solved_memory.mark_solved(1, render(0), ["RIGHT"] * 9)
    assert solved_memory.solved[1]["count"] == len(short)


# --------------------------------------------------------------------- #
# 2. persistence: knowledge survives a fresh process
# --------------------------------------------------------------------- #
def test_memory_store_persists_solved_path_across_reload(tmp_path):
    store = MemoryStore(path=str(tmp_path / "arc_memory"))
    mem = store.game("corridor")
    path = _solve_by_exploring(mem)
    store.save("corridor")

    # A brand-new store, as a later process would build it.
    reloaded = MemoryStore(path=str(tmp_path / "arc_memory")).game("corridor")
    assert reloaded.has_solved(1)
    assert reloaded.solved[1]["count"] == len(path)

    # The reloaded memory must still replay, not merely remember a count:
    # the start fingerprint and the action tokens both have to round-trip.
    replay = reloaded.solved_path(1, render(0))
    assert replay == path

    pos = 0
    for action in replay:
        pos = step(pos, action)
    assert pos == GOAL

    # The transition graph survives too, so exploration is not repeated.
    assert reloaded.transitions == mem.transitions
    assert reloaded.visited == mem.visited


def test_memory_store_starts_empty_for_an_unknown_game(tmp_path):
    fresh = MemoryStore(path=str(tmp_path / "arc_memory")).game("never-seen")
    assert not fresh.has_solved(1)
    assert fresh.solved_path(1, render(0)) is None


# --------------------------------------------------------------------- #
# 3. free look-ahead: predict_frame returns a cached frame
# --------------------------------------------------------------------- #
def test_predict_frame_returns_a_cached_frame(solved_memory):
    predicted = solved_memory.predict_frame(render(0), "RIGHT")
    assert predicted == render(1), "look-ahead did not reproduce the transition"
    # It is the real cached frame, not a re-derivation.
    assert fingerprint(predicted) == solved_memory.predict_fp(render(0), "RIGHT")


def test_predict_frame_is_none_for_an_unobserved_transition(solved_memory):
    assert solved_memory.predict_frame(render(0), "JUMP") is None
    assert solved_memory.predict_frame(("never", "seen"), "RIGHT") is None


def test_predict_frame_is_none_when_frames_are_not_stored():
    mem = GameMemory("corridor", store_frames=False)
    _solve_by_exploring(mem)
    # The fingerprint is still known -- only the frame itself was dropped.
    assert mem.predict_fp(render(0), "RIGHT") is not None
    assert mem.predict_frame(render(0), "RIGHT") is None


# --------------------------------------------------------------------- #
# the agent wires these in, and must be inert unless ARC_GAME_MEMORY is set
# --------------------------------------------------------------------- #
def test_game_memory_is_off_by_default():
    # v47 graded 0.29 with none of this running. The default must stay
    # byte-identical, so the knob defaults to 0 and every call site is
    # guarded by `self._memory is not None`.
    assert agent.GAME_MEMORY == 0
