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


# --- name_goals ---------------------------------------------------------


def _at(target: int):
    def goal(grid):
        return where(grid) == target
    goal.__name__ = f"at{target}"
    return goal


def test_a_single_function_is_named_after_itself():
    assert [n for n, _ in agent.name_goals(_at(3))] == ["at3"]


def test_a_dict_keeps_the_models_own_names():
    named = agent.name_goals({"left": _at(0), "right": _at(11)})
    assert [n for n, _ in named] == ["left", "right"]


def test_repeated_names_are_disambiguated():
    """A row of lambdas is all `<lambda>`, and the names are the answer."""
    named = agent.name_goals([lambda g: False, lambda g: False])
    assert len({n for n, _ in named}) == 2


def test_junk_goals_are_rejected_rather_than_raising():
    for bad in (None, 5, "goal", object()):
        assert agent.name_goals(bad) == []


def test_non_callable_entries_are_dropped():
    assert len(agent.name_goals([_at(3), None, 7])) == 1


# --- planning to several goals at once ----------------------------------
#
# The v17 failure this addresses: the model built a perfect simulator, then
# guessed a single target coordinate, executed a route to it, and learned
# only that that one guess was wrong -- at full price. Testing candidates
# is what costs; the search itself is free.


def test_every_candidate_gets_its_own_route():
    out = agent.plan_actions(
        truth, {"a": _at(2), "b": _at(8)}, line(5), MOVES
    )
    assert out["found"]
    assert out["routes"]["a"] == ["ACTION2"] * 3
    assert out["routes"]["b"] == ["ACTION1"] * 3


def test_the_headline_route_is_the_cheapest_candidate():
    """`actions` is what gets billed, so it must be the shortest found."""
    out = agent.plan_actions(
        truth, {"far": _at(11), "near": _at(6)}, line(5), MOVES
    )
    assert out["goal"] == "near"
    assert out["actions"] == ["ACTION1"]


def test_candidates_that_cannot_be_reached_are_named():
    out = agent.plan_actions(
        truth, {"here": _at(6), "nowhere": lambda g: False}, line(5), MOVES
    )
    assert out["found"] is True
    assert out["unreachable"] == ["nowhere"]


def test_one_broken_candidate_does_not_sink_the_others():
    """With several goals a raising one is just a bad hypothesis."""
    def explodes(grid):
        raise ValueError("the model wrote this")

    out = agent.plan_actions(
        truth, {"bad": explodes, "good": _at(7)}, line(5), MOVES
    )
    assert out["found"]
    assert out["routes"]["good"] == ["ACTION1"] * 2
    assert out["unreachable"] == ["bad"]


def test_a_single_broken_goal_is_still_reported_loudly():
    """Alone, it is a typo worth naming rather than an idea worth dropping."""
    def explodes(grid):
        raise ValueError("boom")

    out = agent.plan_actions(truth, explodes, line(5), MOVES)
    assert out["found"] is False
    assert "ValueError" in out["reason"]


def test_running_out_of_budget_still_returns_what_was_found():
    """Six hypotheses tested for free is six tested, not a failure."""
    out = agent.plan_actions(
        truth, {"near": _at(6), "far": _at(0)}, line(5), MOVES, max_nodes=4
    )
    assert out["found"] is True
    assert out["routes"] == {"near": ["ACTION1"]}
    assert out["unreachable"] == ["far"]


def test_first_only_stops_at_the_nearest():
    out = agent.plan_actions(
        truth, {"far": _at(9), "near": _at(6)}, line(5), MOVES, first_only=True
    )
    assert list(out["routes"]) == ["near"]


def test_the_single_goal_shape_is_unchanged():
    """Existing callers pass one function and read `actions`."""
    out = agent.plan_actions(truth, _at(8), line(5), MOVES)
    assert out["found"] and out["actions"] == ["ACTION1"] * 3
    assert "goal" not in out


# --- touring the candidates ---------------------------------------------


def _walk(grid, route):
    for act in route:
        grid = truth(grid, act)
    return grid


def test_the_tour_visits_every_candidate():
    out = agent.plan_tour(
        truth, {"a": _at(2), "b": _at(6), "c": _at(8)}, line(5), MOVES
    )
    assert set(out["reached"]) == {"a", "b", "c"}
    assert out["unreachable"] == []


def test_each_checkpoint_is_where_it_says_it_is():
    """The invariant the whole idea rests on: play up to checkpoint k in
    the real game and you are standing on candidate k. If that does not
    hold, the tour tests nothing and bills for the privilege."""
    goals = {"a": _at(2), "b": _at(6), "c": _at(8)}
    out = agent.plan_tour(truth, goals, line(5), MOVES)
    for mark in out["checkpoints"]:
        board = _walk(line(5), out["actions"][: mark["after"]])
        assert goals[mark["goal"]](board), mark


def test_the_nearest_candidate_is_tested_first():
    """Greedy on purpose: the guesses least likely to be paid for are the
    expensive ones, so they go last."""
    out = agent.plan_tour(
        truth, {"a": _at(2), "b": _at(6), "c": _at(8)}, line(5), MOVES
    )
    assert out["reached"] == ["b", "c", "a"]
    assert [m["after"] for m in out["checkpoints"]] == [1, 3, 9]


def test_the_tour_is_cheaper_than_testing_one_at_a_time():
    """The claim being made to the model, checked. Reaching all three
    separately from the start costs 3 + 1 + 3 and strands you at the last
    one each time; the tour reaches all three for 9 and, stopping at
    whichever is real, usually far less."""
    goals = {"a": _at(2), "b": _at(6), "c": _at(8)}
    out = agent.plan_tour(truth, goals, line(5), MOVES)
    assert out["checkpoints"][0]["after"] == 1
    assert len(out["actions"]) == 9


def test_an_unreachable_candidate_does_not_strand_the_tour():
    out = agent.plan_tour(
        truth, {"real": _at(6), "never": lambda g: False}, line(5), MOVES
    )
    assert out["reached"] == ["real"]
    assert out["unreachable"] == ["never"]
    assert out["actions"] == ["ACTION1"]


def test_a_goal_already_true_costs_nothing():
    out = agent.plan_tour(truth, {"here": _at(5)}, line(5), MOVES)
    assert out["actions"] == []
    assert out["checkpoints"] == [{"after": 0, "goal": "here"}]


def test_a_predict_that_will_not_replay_is_reported_not_raised():
    """Found the route, then refuses to walk it. The model wrote both."""
    calls = {"n": 0}

    def flaky(grid, act):
        calls["n"] += 1
        if calls["n"] > 200:
            raise RuntimeError("no")
        return truth(grid, act)

    out = agent.plan_tour(flaky, {"a": _at(6), "b": _at(2)}, line(5), MOVES)
    assert isinstance(out["actions"], list)
    assert out["nodes"] >= 0


def test_junk_goals_give_an_empty_tour_rather_than_a_crash():
    out = agent.plan_tour(truth, None, line(5), MOVES)
    assert out["actions"] == [] and out["reached"] == []
    assert "function" in out["reason"]


def test_the_tour_is_bounded_in_time():
    def slow(grid, act):
        for _ in range(200_000):
            pass
        return truth(grid, act)

    out = agent.plan_tour(
        slow, {"a": _at(0), "b": _at(11)}, line(5), MOVES, seconds=0.05
    )
    assert isinstance(out["actions"], list)


# --- running out of world before running out of goals -------------------
#
# v18's real finding. The model's `predict` moved a coloured block and
# copied everything else, so the only states it had were positions of that
# block -- about eight hundred. It visited all of them, satisfied none of
# its goals, and read that as evidence about the goals. It is evidence
# about the simulator: one wrong goal is one hypothesis failing, every
# reachable state failing is the state space being too small.


def test_exhausting_the_state_space_is_called_out():
    note = agent.exhaustion_note(
        "explored every reachable state without reaching is_goal", 807, 13
    )
    assert note is not None
    assert "807" in note and "13" in note
    assert "too small" in note


def test_running_out_of_time_is_not_exhaustion():
    """A search that was cut off proves nothing about the state space."""
    assert agent.exhaustion_note("planning ran out of time after 3596 states",
                                 3596, 5) is None


def test_giving_up_on_the_node_budget_is_not_exhaustion():
    assert agent.exhaustion_note("gave up after 200000 simulated states",
                                 200000, 5) is None


def test_reaching_every_goal_is_not_exhaustion():
    """Exhausting the space having answered every question is a success."""
    assert agent.exhaustion_note(
        "explored every reachable state without reaching is_goal", 807, 0
    ) is None


def test_the_diagnosis_points_at_predict_not_at_the_goals():
    note = agent.exhaustion_note(
        "explored every reachable state without reaching is_goal", 500, 4
    )
    assert "not a wrong goal" in note
    assert "`predict`" in note
    assert "Do not write more goals" in note


def test_a_real_exhausted_tour_carries_the_reason_it_is_built_on():
    """The wording keys off `reason`, so the search must really produce it."""
    out = agent.plan_tour(
        truth, {"a": _at(6), "never": lambda g: False}, line(5), MOVES
    )
    assert "explored every reachable state" in out["reason"]
    assert agent.exhaustion_note(
        out["reason"], out["nodes"], len(out["unreachable"])
    ) is not None
