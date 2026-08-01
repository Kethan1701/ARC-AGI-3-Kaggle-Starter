"""Tests for planning in a simulator the model wrote.

The bet these cover: exploration is paid for once, in a handful of real
actions, and everything after that happens in a learned forward model where
being wrong is free. So the parts that must hold are (a) a correct
simulator yields the SHORTEST route, because the score is the square of the
route's length, and (b) an incorrect simulator degrades into a reported
disagreement rather than a crash or a silent wrong answer.

None of this needs a GPU or a model. `predict` is just a function, so it
can be written by hand here -- correct, wrong, or hostile.
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
    spec = importlib.util.spec_from_file_location(
        "my_agent_planner_test", ROOT / "agent" / "my_agent.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


agent = _load()


# --- a toy game, standing in for a real one -----------------------------
#
# One row, one token. ACTION1 moves it right, ACTION2 left, and the walls
# hold. Small enough to reason about by hand, and structurally the same
# problem as a real board: a state, some actions, a target configuration.

WIDTH = 12


def line(pos: int):
    return tuple([tuple(1 if x == pos else 0 for x in range(WIDTH))])


def where(grid) -> int:
    return grid[0].index(1)


def truth(grid, act):
    pos = where(grid)
    if act == "ACTION1":
        return line(min(pos + 1, WIDTH - 1))
    if act == "ACTION2":
        return line(max(pos - 1, 0))
    return grid


MOVES = ["ACTION1", "ACTION2"]


# --- normalise_grid -----------------------------------------------------


def test_lists_of_lists_are_accepted():
    """The model writes `predict`, and it will return lists, not tuples."""
    assert agent.normalise_grid([[1, 2], [3, 4]]) == ((1, 2), (3, 4))


def test_a_ragged_grid_is_rejected():
    """Rows of different lengths would index out of range mid-search."""
    assert agent.normalise_grid([[1, 2], [3]]) is None


def test_junk_is_rejected_rather_than_raising():
    for bad in (None, 5, "grid", [[None]], [[]], []):
        assert agent.normalise_grid(bad) is None


def test_the_result_is_hashable():
    """It goes straight into the visited set; a list would raise there."""
    assert hash(agent.normalise_grid([[1, 2]])) is not None


# --- verify_predictions -------------------------------------------------


def _log(start: int, acts: list[str]):
    """Real transitions, as the agent records them while playing."""
    grid = line(start)
    out = []
    for act in acts:
        nxt = truth(grid, act)
        out.append((grid, act, nxt))
        grid = nxt
    return out


def test_a_correct_model_scores_one():
    report = agent.verify_predictions(truth, _log(5, ["ACTION1", "ACTION2"] * 3))
    assert report["accuracy"] == 1.0
    assert report["wrong"] == []


def test_a_wrong_model_reports_the_cells_it_got_wrong():
    """A percentage does not correct a forward model; a cell does."""
    def backwards(grid, act):
        return truth(grid, "ACTION2" if act == "ACTION1" else "ACTION1")

    report = agent.verify_predictions(backwards, _log(5, ["ACTION1"]))
    assert report["accuracy"] == 0.0
    assert report["wrong"][0]["action"] == "ACTION1"
    # It predicted the token at 4, it really went to 6.
    assert (4, 0, 1, 0) in report["wrong"][0]["sample"]
    assert (6, 0, 0, 1) in report["wrong"][0]["sample"]


def test_a_model_that_raises_is_counted_not_propagated():
    def hostile(grid, act):
        raise ValueError("nope")

    report = agent.verify_predictions(hostile, _log(5, ["ACTION1", "ACTION1"]))
    assert report["raised"] == 2
    assert report["correct"] == 0
    assert "ValueError: nope" in report["wrong"][0]["raised"]


def test_a_model_returning_junk_says_so():
    report = agent.verify_predictions(lambda g, a: "banana", _log(5, ["ACTION1"]))
    assert report["wrong"][0]["note"] == "predict did not return a grid"


def test_verification_costs_no_actions():
    """The whole reason this exists: the transitions were already paid for,
    so checking a hypothesis against them again is free and can be done as
    many times as the model likes."""
    log = _log(5, ["ACTION1"] * 4)
    before = list(log)
    for _ in range(50):
        agent.verify_predictions(truth, log)
    assert log == before


# --- plan_actions -------------------------------------------------------


def test_it_finds_the_shortest_route():
    """Shortest is not a nicety. The score is (baseline/actions) squared, so
    a route two actions longer than necessary is scored measurably worse."""
    out = agent.plan_actions(
        truth, lambda g: where(g) == 9, line(5), MOVES
    )
    assert out["found"]
    assert out["actions"] == ["ACTION1"] * 4


def test_it_walks_the_other_way_when_that_is_shorter():
    out = agent.plan_actions(truth, lambda g: where(g) == 2, line(5), MOVES)
    assert out["actions"] == ["ACTION2"] * 3


def test_an_already_satisfied_goal_costs_no_actions():
    out = agent.plan_actions(truth, lambda g: where(g) == 5, line(5), MOVES)
    assert out["found"] and out["actions"] == []


def test_an_unreachable_goal_reports_why_rather_than_hanging():
    out = agent.plan_actions(truth, lambda g: False, line(5), MOVES)
    assert not out["found"]
    assert "without reaching is_goal" in out["reason"]


def test_the_depth_limit_is_honoured():
    out = agent.plan_actions(
        truth, lambda g: where(g) == 11, line(0), MOVES, max_depth=3
    )
    assert not out["found"]
    assert "3 actions" in out["reason"]


def test_the_node_budget_is_honoured():
    out = agent.plan_actions(
        truth, lambda g: where(g) == 11, line(0), MOVES, max_nodes=4
    )
    assert not out["found"]
    assert out["nodes"] <= 8
    assert "gave up" in out["reason"]


def test_a_predict_that_raises_prunes_that_branch_only():
    """A partly-broken simulator should still yield a route through the
    parts that work, because the alternative is discarding a model that is
    right about most of the board."""
    def flaky(grid, act):
        if act == "ACTION2":
            raise RuntimeError("boom")
        return truth(grid, act)

    out = agent.plan_actions(flaky, lambda g: where(g) == 7, line(5), MOVES)
    assert out["actions"] == ["ACTION1", "ACTION1"]


def test_an_is_goal_that_raises_on_the_start_board_is_reported():
    def bad(grid):
        raise TypeError("bad goal")

    out = agent.plan_actions(truth, bad, line(5), MOVES)
    assert not out["found"]
    assert "is_goal raised" in out["reason"]


def test_planning_is_bounded_in_time():
    """A model can write a `predict` that is merely very slow. Planning has
    to come back either way, because the turn holds a live game open."""
    def slow(grid, act):
        # ~25ms a call, so the clock trips within a couple of nodes. The
        # toy game only has twelve states, and an earlier version of this
        # test used a fast `predict` and passed for the wrong reason: the
        # search exhausted the space before the deadline could matter.
        for _ in range(1_000_000):
            pass
        return truth(grid, act)

    out = agent.plan_actions(
        slow, lambda g: False, line(5), MOVES, seconds=0.05, max_nodes=10**9
    )
    assert not out["found"]
    assert "ran out of time" in out["reason"]


def test_no_actions_to_search_is_not_a_crash():
    out = agent.plan_actions(truth, lambda g: True, line(5), [])
    assert not out["found"] and "no actions" in out["reason"]


def test_a_wrong_simulator_yields_a_route_that_is_simply_wrong():
    """The failure mode the design accepts, pinned so it stays legible: a
    plan from a bad model is confidently wrong, not detectably wrong. That
    is why execution re-checks every step against `predict` and reports
    `mispredicted` -- the planner cannot catch this and does not try."""
    def mirrored(grid, act):
        return truth(grid, "ACTION2" if act == "ACTION1" else "ACTION1")

    out = agent.plan_actions(mirrored, lambda g: where(g) == 8, line(5), MOVES)
    assert out["found"]
    # Confident, short, and takes the token to 2 in the real game.
    assert out["actions"] == ["ACTION2"] * 3
    real = line(5)
    for act in out["actions"]:
        real = truth(real, act)
    assert where(real) == 2


# --- grid_diff ----------------------------------------------------------


def test_the_diff_is_capped_so_a_wrong_model_cannot_flood_the_prompt():
    left = tuple(tuple(0 for _ in range(64)) for _ in range(64))
    right = tuple(tuple(7 for _ in range(64)) for _ in range(64))
    assert len(agent.grid_diff(left, right)) == 4


def test_the_diff_reads_predicted_then_actual():
    assert agent.grid_diff(((1, 2),), ((1, 9),)) == [(1, 0, 2, 9)]
