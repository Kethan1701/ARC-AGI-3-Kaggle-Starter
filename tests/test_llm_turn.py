"""Tests for the turn: what the model is shown, and how its code acts.

The seam these cover is the one the leaderboard actually sees. The
framework's loop is one action per `choose_action`, and the model's turn
is not -- it acts from inside its own code, as many times as its plan
needs. Getting that wrong costs billed actions on a quadratic metric, and
it cannot be observed locally because the real model only exists on a
Kaggle GPU. So it is pinned here instead.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import deque
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "ARC-AGI-3-Agents"
if not VENDOR.exists():  # pragma: no cover - depends on `make setup`
    pytest.skip("framework not vendored; run `make setup`", allow_module_level=True)
sys.path.insert(0, str(VENDOR))

from arcengine import FrameData, GameState  # noqa: E402


def _load():
    spec = importlib.util.spec_from_file_location(
        "my_agent_turn_test", ROOT / "agent" / "my_agent.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


agent_mod = _load()


# --- fixtures -----------------------------------------------------------


def _grid(fill: int = 0, mark: tuple[int, int, int] | None = None):
    rows = [[fill] * 8 for _ in range(8)]
    if mark is not None:
        y, x, colour = mark
        rows[y][x] = colour
    return rows


def _flat(rows):
    return tuple(agent_mod.chain.from_iterable(tuple(tuple(r) for r in rows)))


def _frame(rows, *, levels=0, state=GameState.NOT_FINISHED, available=(1, 2, 6)):
    return FrameData(
        game_id="tst",
        frame=[rows],
        state=state,
        levels_completed=levels,
        win_levels=3,
        guid="g",
        full_reset=False,
        available_actions=list(available),
    )


class _FakeEnv:
    """Records what was sent and hands back scripted frames."""

    def __init__(self, start, results=()):
        self.current = start
        self.results = list(results)
        self.sent: list[tuple] = []

    def step(self, action, data=None, reasoning=None):
        payload = data or {}
        self.sent.append((action.value, payload.get("x"), payload.get("y")))
        if self.results:
            self.current = self.results.pop(0)
        return self.current

    @property
    def observation_space(self):
        return self.current


def _turn_agent(env):
    """A MyAgent with only what the turn path touches.

    `__new__` skips `__init__`, and `_convert_raw_frame_data` is stubbed to
    identity because the fake env already yields `FrameData` -- that
    conversion is a numpy/pydantic detail covered elsewhere, and is not
    what these tests are about.
    """
    a = agent_mod.MyAgent.__new__(agent_mod.MyAgent)
    a.game_id = "tst"
    a.guid = ""
    a.frames = []
    a.action_counter = 0
    a.arc_env = env
    a._convert_raw_frame_data = lambda raw: raw
    a._llm_dead = False
    a._llm_failures = 0
    a._llm_timeouts = 0
    a._llm_ns = {}
    a._llm_messages = []
    a._llm_history = []
    a._llm_turns = 0
    a._llm_note = ""
    a._transitions = deque(maxlen=agent_mod.TRANSITION_LOG)
    a._verify_calls = 0
    a._verify_best = 0.0
    a._plan_calls = 0
    a._plans_found = 0
    a._tour_calls = 0
    a._goal_misses = 0
    a._exhaustions = 0
    a._verify_live = 0.0
    a._verify_live_best = 0.0
    a._verify_peak = 0.0
    a._turn_actions = 0
    a._turn_frame = None
    a._turn_level = 0
    a._turn_labels = []
    a._last_turn_labels = []
    a._repeat_turns = 0
    a._last_view = None
    a._noop = False
    a._volatile = frozenset()
    a._steps = 0
    a._recorded = 0
    a._pending = None
    return a


def _reply(content=None, code=None, call_id="c1"):
    tool_calls = None
    if code is not None:
        tool_calls = [{
            "id": call_id,
            "type": "function",
            "function": {"name": "python", "arguments": json.dumps({"code": code})},
        }]
    return {"choices": [{"message": {
        "role": "assistant", "content": content, "tool_calls": tool_calls,
    }}]}


def _script(monkeypatch, replies):
    queue = list(replies)

    def fake_post(_payload):
        if not queue:
            raise AssertionError("the turn asked for more replies than scripted")
        return queue.pop(0)

    monkeypatch.setattr(agent_mod, "chat_completion", fake_post)


# --- reading what the model wrote --------------------------------------


def test_a_click_tuple_is_one_action_but_a_list_of_names_is_many():
    """The whole ambiguity of `action(...)`, in one assertion."""
    assert agent_mod.split_action_specs(("ACTION6", 12, 30)) == [("ACTION6", 12, 30)]
    assert agent_mod.split_action_specs(["A", "B", "C"]) == ["A", "B", "C"]
    assert agent_mod.split_action_specs("ACTION1") == ["ACTION1"]
    assert agent_mod.split_action_specs(1) == [1]


def test_action_names_and_ids_both_parse():
    for spec in ("ACTION1", "action1", " ACTION 1 ", 1):
        assert agent_mod.parse_action_spec(spec, [1, 2]) == (1, None, None)
    assert agent_mod.parse_action_spec("RESET", [1]) == (0, None, None)
    assert agent_mod.parse_action_spec(("ACTION6", 12, 30), [6]) == (6, 12, 30)


def test_an_action_this_game_does_not_accept_is_refused():
    with pytest.raises(ValueError, match="not accepted by this game"):
        agent_mod.parse_action_spec("ACTION3", [1, 2])


def test_a_click_without_coordinates_is_refused_not_guessed():
    with pytest.raises(ValueError, match="needs coordinates"):
        agent_mod.parse_action_spec("ACTION6", [6])


def test_a_click_off_the_board_is_refused():
    with pytest.raises(ValueError, match="off the board"):
        agent_mod.parse_action_spec(("ACTION6", 64, 0), [6])


def test_coordinates_on_a_button_are_refused():
    """`[1, 2, 3]` reads as a click; this is where that surfaces to the
    model rather than silently playing something it did not mean."""
    with pytest.raises(ValueError, match="takes no coordinates"):
        agent_mod.parse_action_spec([1, 2, 3], [1, 6])


def test_nonsense_names_itself():
    with pytest.raises(ValueError, match="not an action name"):
        agent_mod.parse_action_spec("go left", [1])


# --- what the model is shown -------------------------------------------


def test_the_frame_description_leads_with_the_biggest_objects():
    view = {"nodes": [
        {"id": 0, "color": 1, "pixels": 2, "bbox": (0, 0, 0, 1), "children": []},
        {"id": 1, "color": 3, "pixels": 40, "bbox": (1, 0, 8, 4), "children": [0]},
    ], "adjacency_list": []}
    text = agent_mod.describe_frame(view)
    assert text.index("#1") < text.index("#0")
    assert "encloses [0]" in text


def test_the_description_caps_and_says_it_capped():
    nodes = [
        {"id": i, "color": 1, "pixels": 1, "bbox": (0, 0, 0, 0), "children": []}
        for i in range(50)
    ]
    text = agent_mod.describe_frame({"nodes": nodes, "adjacency_list": []}, limit=5)
    assert text.count("\n") == 5  # five objects plus the tail line
    assert "45 smaller objects" in text


def test_the_system_prompt_lists_this_games_actions_and_the_scoring():
    prompt = agent_mod.build_system_prompt([1, 2, 6])
    assert "ACTION1, ACTION2, ACTION6" in prompt
    assert "quadratic" in prompt


def test_the_system_prompt_never_claims_what_an_action_does():
    """The mapping differs per game and nothing declares it. Naming
    ACTION1 'up' would be a guess the model then acts on."""
    prompt = agent_mod.build_system_prompt([1, 2, 3, 4]).lower()
    for guess in ("move up", "move down", "move left", "move right"):
        assert guess not in prompt


# --- the turn ----------------------------------------------------------


def test_the_model_acts_from_inside_its_own_code(monkeypatch):
    env = _FakeEnv(_frame(_grid()), [_frame(_grid(mark=(1, 1, 5)))] * 3)
    a = _turn_agent(env)
    _script(monkeypatch, [
        _reply(code="action(['ACTION1', 'ACTION2', ('ACTION6', 3, 4)])"),
        _reply(content="that told me what I needed"),
    ])

    acted = a._llm_turn(agent_mod.grid_of(env.current), env.current)

    assert acted is True
    assert env.sent == [(1, None, None), (2, None, None), (6, 3, 4)]
    assert a._steps == 3


def test_a_turn_that_acted_does_not_spend_one_more_action(monkeypatch):
    """The framework bills one action per `choose_action`. The model has
    already played its moves, so that one must never reach the server."""
    env = _FakeEnv(_frame(_grid()), [_frame(_grid(mark=(1, 1, 5)))])
    a = _turn_agent(env)
    _script(monkeypatch, [_reply(code="action('ACTION1')"), _reply(content="done")])

    a._llm_turn(agent_mod.grid_of(env.current), env.current)
    a._noop = True
    before = len(env.sent)

    returned = a.do_action_request(agent_mod.GameAction.from_id(1))

    assert len(env.sent) == before, "the formality action was sent to the server"
    assert returned is env.current
    assert a._noop is False, "the suppression must not persist to the next action"


def test_a_dead_server_retires_the_model_after_a_bounded_number_of_tries(
    monkeypatch,
):
    env = _FakeEnv(_frame(_grid()))
    a = _turn_agent(env)

    def dead(_payload):
        raise OSError("connection refused")

    monkeypatch.setattr(agent_mod, "chat_completion", dead)
    grid = agent_mod.grid_of(env.current)

    for i in range(1, agent_mod.LLM_MAX_FAILURES):
        assert a._llm_turn(grid, env.current) is False
        assert a._llm_dead is False, f"retired after only {i} failure(s)"

    assert a._llm_turn(grid, env.current) is False
    assert a._llm_dead is True
    assert env.sent == []


def test_one_slow_generation_does_not_write_off_a_working_server(monkeypatch):
    """Measured on Kaggle, not imagined: the model played 136 actions, one
    generation ran past the deadline, and the old code retired a perfectly
    healthy server for the rest of the game."""
    env = _FakeEnv(_frame(_grid()), [_frame(_grid(mark=(1, 1, 5)))] * 4)
    a = _turn_agent(env)
    grid = agent_mod.grid_of(env.current)

    calls = {"n": 0}

    def flaky(_payload):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("timed out")
        if calls["n"] == 2:
            return _reply(code="action('ACTION1')")
        return _reply(content="ok")

    monkeypatch.setattr(agent_mod, "chat_completion", flaky)

    assert a._llm_turn(grid, env.current) is False   # the slow one
    assert a._llm_dead is False, "a timeout is not a dead server"
    assert a._llm_turn(grid, env.current) is True    # and it recovers
    assert env.sent == [(1, None, None)]


def test_a_busy_server_is_not_a_dead_one(monkeypatch):
    """v25: 25 games share one vLLM server, 23 requests in flight on
    average. A timeout there means the queue is deep, not that anything
    broke -- but `cn04` hit three in a row, tripped the dead-server rule,
    and played the rest of its run on the search alone: 28 actions, last
    of 25 games."""
    env = _FakeEnv(_frame(_grid()))
    a = _turn_agent(env)

    def slow(_payload):
        raise TimeoutError("timed out")

    monkeypatch.setattr(agent_mod, "chat_completion", slow)
    grid = agent_mod.grid_of(env.current)

    assert agent_mod.LLM_MAX_TIMEOUTS > agent_mod.LLM_MAX_FAILURES, (
        "a timeout has to be cheaper than a refused connection or this "
        "test is measuring nothing"
    )
    for i in range(1, agent_mod.LLM_MAX_TIMEOUTS):
        assert a._llm_turn(grid, env.current) is False
        assert a._llm_dead is False, f"retired after {i} timeout(s)"

    assert a._llm_turn(grid, env.current) is False
    assert a._llm_dead is True, "a server that only ever times out is gone"


def test_timeouts_and_transport_failures_are_counted_apart(monkeypatch):
    """Otherwise alternating one of each retires the model on a server
    that is merely busy and occasionally unlucky."""
    env = _FakeEnv(_frame(_grid()))
    a = _turn_agent(env)
    a._llm_timeouts = agent_mod.LLM_MAX_TIMEOUTS - 1

    monkeypatch.setattr(
        agent_mod, "chat_completion",
        lambda _p: (_ for _ in ()).throw(OSError("connection refused")),
    )
    a._llm_turn(agent_mod.grid_of(env.current), env.current)

    assert a._llm_failures == 1, "a refused connection is not a timeout"
    assert a._llm_timeouts == agent_mod.LLM_MAX_TIMEOUTS - 1
    assert a._llm_dead is False


def test_a_timeout_wrapped_in_a_urlerror_is_still_a_timeout():
    """`urlopen` raises a bare TimeoutError once connected and a URLError
    wrapping one if it timed out connecting. Reading the second spelling
    as a dead server is exactly the bug being fixed."""
    import urllib.error

    assert agent_mod._is_timeout(TimeoutError("timed out"))
    assert agent_mod._is_timeout(urllib.error.URLError(TimeoutError("x")))
    assert not agent_mod._is_timeout(OSError("connection refused"))
    assert not agent_mod._is_timeout(
        urllib.error.HTTPError("u", 400, "Bad Request", {}, None)
    )


def test_a_recovered_turn_clears_the_failure_count(monkeypatch):
    """Otherwise three slow generations spread across a whole game would
    retire a server that worked fine in between."""
    env = _FakeEnv(_frame(_grid()))
    a = _turn_agent(env)
    a._llm_failures = agent_mod.LLM_MAX_FAILURES - 1
    _script(monkeypatch, [_reply(content="thinking")])

    a._llm_turn(agent_mod.grid_of(env.current), env.current)
    assert a._llm_failures == 0


def test_a_failed_turn_does_not_leave_a_broken_transcript(monkeypatch):
    """A half-finished turn can end on an assistant tool_call whose result
    never arrived, which the server rejects on the next request."""
    env = _FakeEnv(_frame(_grid()))
    a = _turn_agent(env)
    grid = agent_mod.grid_of(env.current)

    calls = {"n": 0}

    def flaky(_payload):
        calls["n"] += 1
        if calls["n"] == 1:
            return _reply(code="action('ACTION1')")
        raise TimeoutError("timed out")

    monkeypatch.setattr(agent_mod, "chat_completion", flaky)
    a._llm_turn(grid, env.current)

    assert [m["role"] for m in a._llm_messages] == ["system"]


def test_a_turn_that_only_reasons_falls_back_rather_than_stalling(monkeypatch):
    """A move has to be played every step or the game never advances."""
    env = _FakeEnv(_frame(_grid()))
    a = _turn_agent(env)
    _script(monkeypatch, [_reply(content="I will think about it")])

    assert a._llm_turn(agent_mod.grid_of(env.current), env.current) is False
    assert a._llm_dead is False, "a quiet turn is not a broken server"
    assert "action()" in a._llm_note


def test_acting_stops_at_a_level_boundary(monkeypatch):
    """Past the boundary the model is planning against a level that no
    longer exists, and every action is billed against the new one."""
    env = _FakeEnv(
        _frame(_grid()),
        [_frame(_grid()), _frame(_grid(), levels=1), _frame(_grid())],
    )
    a = _turn_agent(env)
    _script(monkeypatch, [_reply(code="out = action(['ACTION1'] * 3)")])

    a._llm_turn(agent_mod.grid_of(env.current), env.current)

    assert len(env.sent) == 2, "kept playing into the next level"
    assert a._turn_actions == 2


def test_acting_stops_when_the_game_ends(monkeypatch):
    env = _FakeEnv(
        _frame(_grid()),
        [_frame(_grid(), state=GameState.GAME_OVER), _frame(_grid())],
    )
    a = _turn_agent(env)
    _script(monkeypatch, [_reply(code="action(['ACTION1'] * 4)")])

    a._llm_turn(agent_mod.grid_of(env.current), env.current)

    assert len(env.sent) == 1


def test_a_bad_step_in_a_batch_sends_nothing(monkeypatch):
    """Parsing happens up front so a typo in the fourth move does not
    leave the first three played on a metric that bills every one."""
    env = _FakeEnv(_frame(_grid()))
    a = _turn_agent(env)
    a._turn_frame = env.current

    with pytest.raises(ValueError):
        a._llm_act(["ACTION1", "ACTION2", "ACTION9"])

    assert env.sent == []


def test_the_turn_action_limit_holds(monkeypatch):
    env = _FakeEnv(_frame(_grid()))
    a = _turn_agent(env)
    monkeypatch.setattr(agent_mod, "LLM_TURN_ACTIONS", 3)
    _script(monkeypatch, [_reply(code="out = action(['ACTION1'] * 20)")])

    a._llm_turn(agent_mod.grid_of(env.current), env.current)

    assert len(env.sent) == 3
    assert a._turn_actions == 3


# --- the lp85 loop -----------------------------------------------------
#
# v25, game lp85: nineteen consecutive turns, each one tool call emitting
# the identical sixty-action batch, level frozen at 1. It had cleared
# level 1 at action 13 and then spent 1238 more going nowhere. On a
# (baseline/actions)^2 metric that is not idling, it is the most
# expensive thing the agent did all run -- and no note ever fired,
# because every existing nudge is about `predict` and none of them
# tripped.


def _play(a, env, monkeypatch, code):
    _script(monkeypatch, [_reply(code=code), _reply(content="ok")])
    return a._llm_turn(agent_mod.grid_of(env.current), env.current)


def test_replaying_the_previous_turn_move_for_move_is_called_out(monkeypatch):
    env = _FakeEnv(_frame(_grid()))
    a = _turn_agent(env)

    _play(a, env, monkeypatch, "action(['ACTION1', 'ACTION2'])")
    assert a._repeat_turns == 0, "the first turn has nothing to repeat"
    assert "STOP" not in a._llm_note

    _play(a, env, monkeypatch, "action(['ACTION1', 'ACTION2'])")
    assert a._repeat_turns == 1
    assert "STOP" in a._llm_note
    assert "SQUARED" in a._llm_note, "say why it is expensive, not just that"


def test_a_persistent_repeat_throws_the_transcript_away(monkeypatch):
    """The note is tried first. If the model repeats anyway, what is
    generating the repeat is the transcript of it agreeing with itself,
    and only dropping that breaks the cycle."""
    env = _FakeEnv(_frame(_grid()))
    a = _turn_agent(env)
    code = "action(['ACTION1', 'ACTION2'])"

    _play(a, env, monkeypatch, code)
    _play(a, env, monkeypatch, code)
    assert len(a._llm_messages) > 1, "one repeat is a note, not a reset"

    _play(a, env, monkeypatch, code)
    assert len(a._llm_messages) == 1, "back to the system prompt alone"
    assert "STOP" in a._llm_note, "the note survives the reset"
    assert a._repeat_turns == 0, "the cycle is broken, start counting again"


def test_a_different_batch_is_not_a_repeat(monkeypatch):
    """Exact match on purpose. A model executing a long plan it actually
    found must not be told it is looping."""
    env = _FakeEnv(_frame(_grid()))
    a = _turn_agent(env)

    _play(a, env, monkeypatch, "action(['ACTION1', 'ACTION2'])")
    _play(a, env, monkeypatch, "action(['ACTION2', 'ACTION1'])")

    assert a._repeat_turns == 0, "same actions, different order, is progress"
    assert "STOP" not in a._llm_note


def test_the_same_batch_that_cleared_a_level_is_not_a_repeat(monkeypatch):
    """Replaying a sequence that just advanced the game is the one case
    where doing the same thing twice is exactly right."""
    env = _FakeEnv(_frame(_grid()), [_frame(_grid(), levels=1)] * 8)
    a = _turn_agent(env)
    code = "action(['ACTION1', 'ACTION2'])"

    _play(a, env, monkeypatch, code)
    _play(a, env, monkeypatch, code)

    assert a._repeat_turns == 0
    assert "STOP" not in a._llm_note


def test_a_repeat_does_not_swallow_the_diagnostic_that_explains_it(
    monkeypatch,
):
    """The first cut made the repeat warning the top branch of the note
    chain, and it silently suppressed the drift note -- which is the half
    that says WHY replaying the batch is not working. Both, or the model
    is told to stop without being told what to do instead."""
    a, _ = _drifting_agent(
        monkeypatch, "result = action(['ACTION1', 'ACTION1'])\n"
    )
    assert a._repeat_turns == 1, "turn 2 replayed turn 1 move for move"
    assert a._llm_note.startswith("STOP."), "the repeat leads"
    assert "Checked for you" in a._llm_note, "and the drift note survives"


def test_a_turn_that_played_nothing_is_not_a_repeat(monkeypatch):
    """Two turns that both only reasoned have identical (empty) action
    lists, and that is what the no-action nudge is for, not this one."""
    env = _FakeEnv(_frame(_grid()))
    a = _turn_agent(env)

    _script(monkeypatch, [_reply(content="thinking")])
    a._llm_turn(agent_mod.grid_of(env.current), env.current)
    _script(monkeypatch, [_reply(content="thinking")])
    a._llm_turn(agent_mod.grid_of(env.current), env.current)

    assert a._repeat_turns == 0
    assert "action()" in a._llm_note, "the no-action nudge still owns this"


def test_the_action_counter_tracks_every_step_the_model_took(monkeypatch):
    """`Agent.main()` bills one per loop iteration and the turn takes
    many, so without this MAX_ACTIONS would stop meaning anything."""
    env = _FakeEnv(_frame(_grid()))
    a = _turn_agent(env)
    _script(monkeypatch, [_reply(code="action(['ACTION1'] * 5)"), _reply(content="ok")])

    a._llm_turn(agent_mod.grid_of(env.current), env.current)

    assert a.action_counter == 5


def test_history_records_whether_the_board_moved(monkeypatch):
    env = _FakeEnv(_frame(_grid()), [_frame(_grid()), _frame(_grid(mark=(2, 2, 7)))])
    a = _turn_agent(env)
    _script(monkeypatch, [_reply(code="action(['ACTION1', 'ACTION2'])"), _reply("ok")])

    a._llm_turn(agent_mod.grid_of(env.current), env.current)

    assert a._llm_history == ["ACTION1 -> no change", "ACTION2 -> board changed"]


def test_history_stays_bounded(monkeypatch):
    env = _FakeEnv(_frame(_grid()))
    a = _turn_agent(env)
    a._turn_frame = env.current
    a._llm_act(["ACTION1"] * (agent_mod.LLM_HISTORY + 10))
    assert len(a._llm_history) == agent_mod.LLM_HISTORY


def test_the_system_prompt_is_built_once_and_kept(monkeypatch):
    """It is the framing eviction must never drop -- rebuilding it every
    turn would mean the transcript carried several copies."""
    env = _FakeEnv(_frame(_grid()))
    a = _turn_agent(env)
    _script(monkeypatch, [_reply(content="a"), _reply(content="b")])

    grid = agent_mod.grid_of(env.current)
    a._llm_turn(grid, env.current)
    a._llm_turn(grid, env.current)

    systems = [m for m in a._llm_messages if m.get("role") == "system"]
    assert len(systems) == 1
    assert a._llm_messages[0]["role"] == "system"


def test_choose_action_suppresses_its_own_action_after_the_model_acted(
    monkeypatch,
):
    """The end-to-end shape of the design, through a real agent.

    `Agent.main()` does `choose_action` then `take_action`. When the model
    has already played its moves, the action `choose_action` hands back is
    a formality the loop requires -- and if it reaches the server it is an
    extra billed step, chosen by nobody, on a board the model has moved
    past. This is the test that would have caught that.
    """
    env = _FakeEnv(_frame(_grid()), [_frame(_grid(mark=(1, 1, 5)))] * 4)
    a = agent_mod.MyAgent(
        card_id="c", game_id="tst", agent_name="a", ROOT_URL="",
        record=False, arc_env=env,
    )
    a._convert_raw_frame_data = lambda raw: raw
    monkeypatch.setattr(agent_mod, "LLM_ENABLED", 1)
    a._llm_dead = False
    _script(monkeypatch, [
        _reply(code="action(['ACTION1', 'ACTION2'])"),
        _reply(content="done"),
    ])

    returned = a.choose_action(a.frames, env.current)
    assert env.sent == [(1, None, None), (2, None, None)]
    assert a._noop is True

    # ...and now the framework takes the action it was handed.
    a.take_action(returned)
    assert env.sent == [(1, None, None), (2, None, None)], "billed an extra action"


def test_choose_action_still_plays_the_search_when_the_model_is_off():
    """The fallback is the shipping configuration whenever ARC_LLM is
    unset, so it has to reach the server exactly once per call."""
    env = _FakeEnv(_frame(_grid()))
    a = agent_mod.MyAgent(
        card_id="c", game_id="tst", agent_name="a", ROOT_URL="",
        record=False, arc_env=env,
    )
    a._convert_raw_frame_data = lambda raw: raw
    a._llm_dead = True

    returned = a.choose_action(a.frames, env.current)
    a.take_action(returned)

    assert len(env.sent) == 1
    assert a._noop is False


def test_the_turn_sends_the_board_as_an_image(monkeypatch):
    """The served model has a vision encoder. The first real run fed it
    prose about a visual puzzle and it never converged."""
    env = _FakeEnv(_frame(_grid(mark=(3, 3, 9))))
    a = _turn_agent(env)
    _script(monkeypatch, [_reply(content="ok")])

    a._llm_turn(agent_mod.grid_of(env.current), env.current)

    user = [m for m in a._llm_messages if m.get("role") == "user"][-1]
    assert isinstance(user["content"], list)
    kinds = [p["type"] for p in user["content"]]
    assert kinds == ["text", "image_url"]
    assert user["content"][1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )


def test_only_the_current_board_is_ever_in_flight(monkeypatch):
    env = _FakeEnv(_frame(_grid()))
    a = _turn_agent(env)
    _script(monkeypatch, [_reply(content="a"), _reply(content="b")])
    grid = agent_mod.grid_of(env.current)

    a._llm_turn(grid, env.current)
    a._llm_turn(grid, env.current)

    images = sum(
        1
        for m in a._llm_messages
        if isinstance(m.get("content"), list)
        for p in m["content"]
        if p.get("type") == "image_url"
    )
    assert images == 1


def test_the_system_prompt_tells_the_model_it_can_see(monkeypatch):
    """It used to say the opposite -- 'you are not shown the raw numbers as
    a picture' -- which was true then and is a lie now."""
    prompt = agent_mod.build_system_prompt([1, 2, 6]).lower()
    assert "image" in prompt
    assert "not shown" not in prompt


def test_a_forward_model_written_one_turn_survives_to_the_next(monkeypatch):
    """The change the whole strategy rests on.

    The namespace used to be rebuilt each turn, so a simulator written on
    turn 3 was destroyed before turn 4 could search it -- which made
    "plan in the model, not the environment" impossible to follow.
    """
    env = _FakeEnv(_frame(_grid()))
    a = _turn_agent(env)
    grid = agent_mod.grid_of(env.current)
    _script(monkeypatch, [
        _reply(code="def predict(g, act):\n    return 'moved'\nnotes.append('A1 moves')"),
        _reply(content="built it"),
        _reply(code="result = predict(None, 'ACTION1')"),
        _reply(content="reused it"),
    ])

    a._llm_turn(grid, env.current)
    a._llm_turn(grid, env.current)

    tools = [m["content"] for m in a._llm_messages if m.get("role") == "tool"]
    assert "moved" in tools[-1], "the simulator did not survive the turn"


def test_the_grid_the_model_sees_updates_when_it_acts(monkeypatch):
    """Straight from the v15 logs. The model wrote:

        result2 = action('ACTION2')
        for y in range(64):
            if grid[y][x] != state_after_action1[y][x]: ...

    and later commented "The grid variable should have been updated". It
    was not -- `grid` was bound once per turn -- so every diff compared a
    board against itself and reported no change whatever the action did.
    """
    env = _FakeEnv(_frame(_grid()), [_frame(_grid(mark=(2, 2, 7)))])
    a = _turn_agent(env)
    _script(monkeypatch, [
        _reply(code="before = grid\naction('ACTION1')\nresult = (grid is before, grid[2][2])"),
        _reply(content="ok"),
    ])

    a._llm_turn(agent_mod.grid_of(env.current), env.current)

    tool = [m["content"] for m in a._llm_messages if m.get("role") == "tool"][0]
    assert "(False, 7)" in tool, f"grid did not refresh after action(): {tool[:200]}"


def test_current_frame_refreshes_alongside_the_grid(monkeypatch):
    env = _FakeEnv(_frame(_grid()), [_frame(_grid(mark=(2, 2, 7)))])
    a = _turn_agent(env)
    _script(monkeypatch, [
        _reply(code="action('ACTION1')\nresult = sorted(n['color'] for n in current_frame['nodes'])"),
        _reply(content="ok"),
    ])

    a._llm_turn(agent_mod.grid_of(env.current), env.current)

    tool = [m["content"] for m in a._llm_messages if m.get("role") == "tool"][0]
    assert "7" in tool, "current_frame still describes the old board"


def test_notes_survive_and_are_read_back(monkeypatch):
    """Eviction trims the transcript, so this is the only memory that
    outlives it."""
    env = _FakeEnv(_frame(_grid()))
    a = _turn_agent(env)
    grid = agent_mod.grid_of(env.current)
    _script(monkeypatch, [
        _reply(code="notes.append('ACTION1 moves the player up')"),
        _reply(content="noted"),
        _reply(content="ok"),
    ])

    a._llm_turn(grid, env.current)
    a._llm_turn(grid, env.current)

    user = [m for m in a._llm_messages if m.get("role") == "user"][-1]
    text = user["content"][0]["text"] if isinstance(user["content"], list) else user["content"]
    assert "ACTION1 moves the player up" in text


def test_notes_exist_before_the_model_touches_them(monkeypatch):
    """`notes.append(...)` has to work on the first turn without the model
    having to guess whether it needs creating."""
    env = _FakeEnv(_frame(_grid()))
    a = _turn_agent(env)
    _script(monkeypatch, [_reply(code="notes.append('first')"), _reply("ok")])

    a._llm_turn(agent_mod.grid_of(env.current), env.current)
    assert a._llm_ns["notes"] == ["first"]


def test_the_refreshed_five_do_not_go_stale(monkeypatch):
    """Persisting the namespace must not persist a stale board."""
    env = _FakeEnv(_frame(_grid()), [_frame(_grid(mark=(5, 5, 3)))])
    a = _turn_agent(env)
    _script(monkeypatch, [_reply(content="a"), _reply(content="b")])

    a._llm_turn(agent_mod.grid_of(env.current), env.current)
    first = a._llm_ns["grid"]
    env.current = _frame(_grid(mark=(5, 5, 3)))
    a._llm_turn(agent_mod.grid_of(env.current), env.current)

    assert a._llm_ns["grid"] != first
    assert a._llm_ns["grid"][5][5] == 3


def test_notes_render_trims_rather_than_flooding_the_prompt():
    long = [f"finding number {i} " + "x" * 200 for i in range(50)]
    text = agent_mod.render_notes(long, limit=500)
    assert len(text) < 1200
    assert "trimmed" in text


def test_notes_render_handles_whatever_the_model_puts_there():
    assert agent_mod.render_notes(None) == ""
    assert agent_mod.render_notes([]) == ""
    assert "a: 1" in agent_mod.render_notes({"a": 1})
    assert "just a string" in agent_mod.render_notes("just a string")


def test_the_prompt_tells_it_to_simulate_rather_than_probe():
    prompt = agent_mod.build_system_prompt([1, 2, 6]).lower()
    assert "predict(" in prompt
    assert "persist" in prompt
    assert "notes" in prompt


def test_the_namespace_offers_both_views(monkeypatch):
    """TAAF withholds the raw grid and substitutes an image. We have no
    image to substitute, so withholding it would leave the model blind."""
    env = _FakeEnv(_frame(_grid(mark=(3, 3, 9))))
    a = _turn_agent(env)
    _script(monkeypatch, [
        _reply(code="result = (len(current_frame['nodes']), grid[3][3], valid_actions)"),
        _reply(content="ok"),
    ])

    a._llm_turn(agent_mod.grid_of(env.current), env.current)

    tool = [m for m in a._llm_messages if m.get("role") == "tool"][0]
    assert "9" in tool["content"]
    assert "[1, 2, 6]" in tool["content"]


# --- the forward-model contract ----------------------------------------
#
# The agent now does the mechanical half of the strategy the prompt asks
# for: it keeps every real transition, replays them against the model's
# `predict` on demand, searches that simulator for a route, and reports
# where reality diverged from it. The model supplies only `predict` and
# `is_goal`. These pin that seam, because it decides whether actions get
# spent on a checked route or on exploration.

# A forward model for the fake game below: the mark of colour 5 slides one
# column right. Written as source, because the model writes it as source.
SLIDE_RIGHT = (
    "def predict(g, act):\n"
    "    rows = [list(r) for r in g]\n"
    "    for y, row in enumerate(rows):\n"
    "        for x, v in enumerate(row):\n"
    "            if v == 5:\n"
    "                rows[y][x] = 0\n"
    "                rows[y][x + 1] = 5\n"
    "                return rows\n"
    "    return rows\n"
)


def test_real_transitions_are_recorded_as_they_are_played(monkeypatch):
    """These are the evidence `verify` runs on. Every one cost a billed
    action, so they are kept rather than recomputed."""
    env = _FakeEnv(
        _frame(_grid(mark=(1, 1, 5))),
        [_frame(_grid(mark=(1, 2, 5))), _frame(_grid(mark=(1, 3, 5)))],
    )
    a = _turn_agent(env)
    _script(monkeypatch, [
        _reply(code="action(['ACTION1', 'ACTION1'])"),
        _reply(content="ok"),
    ])

    a._llm_turn(agent_mod.grid_of(env.current), env.current)

    assert len(a._transitions) == 2
    before, label, after = a._transitions[0]
    assert label == "ACTION1"
    assert before[1][1] == 5 and after[1][2] == 5


def test_verify_scores_a_forward_model_without_spending_actions(monkeypatch):
    env = _FakeEnv(_frame(_grid(mark=(1, 1, 5))), [_frame(_grid(mark=(1, 2, 5)))])
    a = _turn_agent(env)
    _script(monkeypatch, [
        _reply(code="action('ACTION1')"),
        _reply(code=SLIDE_RIGHT + "result = verify(predict)"),
        _reply(content="ok"),
    ])

    a._llm_turn(agent_mod.grid_of(env.current), env.current)

    assert a._steps == 1, "verify must not play anything"
    report = [m for m in a._llm_messages if m.get("role") == "tool"][-1]["content"]
    assert "'accuracy': 1.0" in report


def test_verify_before_anything_has_been_played_says_so(monkeypatch):
    env = _FakeEnv(_frame(_grid(mark=(1, 1, 5))))
    a = _turn_agent(env)
    _script(monkeypatch, [
        _reply(code=SLIDE_RIGHT + "result = verify(predict)"),
        _reply(content="ok"),
    ])

    a._llm_turn(agent_mod.grid_of(env.current), env.current)

    tool = [m for m in a._llm_messages if m.get("role") == "tool"][-1]["content"]
    assert "nothing to check against yet" in tool


def test_plan_searches_the_simulator_and_returns_a_route(monkeypatch):
    """The point of the design: the search that finds the route runs over
    `predict`, so it costs GPU seconds instead of score."""
    env = _FakeEnv(_frame(_grid(mark=(1, 1, 5))))
    a = _turn_agent(env)
    code = (
        SLIDE_RIGHT
        + "def is_goal(g):\n"
        "    return any(row[5] == 5 for row in g)\n"
        "result = plan(predict, is_goal)\n"
    )
    _script(monkeypatch, [_reply(code=code), _reply(content="ok")])

    a._llm_turn(agent_mod.grid_of(env.current), env.current)

    assert a._steps == 0, "planning must not touch the environment"
    tool = [m for m in a._llm_messages if m.get("role") == "tool"][-1]["content"]
    assert "'found': True" in tool
    # The mark starts at column 1 and the goal is column 5: four moves, and
    # the planner must not come back with five. Counted off the route
    # itself rather than off the rendered dict, which also carries the
    # per-candidate `routes` and would double every name in it.
    assert "'actions': [" + ", ".join(["'ACTION1'"] * 4) + "]" in tool


def test_plan_excludes_clicks_unless_the_model_supplies_targets():
    """ACTION6 has 4,096 targets. Branching on all of them makes the first
    level of the search bigger than the whole node budget, so a click game
    has to shortlist its own."""
    env = _FakeEnv(_frame(_grid(mark=(1, 1, 5)), available=(1, 2, 6)))
    a = _turn_agent(env)
    a._turn_frame = env.current
    a._llm_ns["grid"] = agent_mod.grid_of(env.current)
    seen: list = []

    a._llm_plan(
        lambda grid, act: seen.append(act) or grid, lambda g: False, max_depth=1
    )

    assert set(seen) == {"ACTION1", "ACTION2"}


def test_a_wrong_forward_model_comes_back_as_a_counterexample(monkeypatch):
    """A plan is only as good as the simulator it was found in, so the
    first divergence is the most useful thing a turn can report."""
    env = _FakeEnv(_frame(_grid(mark=(1, 1, 5))), [_frame(_grid(mark=(1, 2, 5)))])
    a = _turn_agent(env)
    wrong = SLIDE_RIGHT.replace("rows[y][x + 1]", "rows[y][x - 1]")
    _script(monkeypatch, [
        _reply(code=wrong + "result = action('ACTION1')"),
        _reply(content="ok"),
    ])

    a._llm_turn(agent_mod.grid_of(env.current), env.current)

    tool = [m for m in a._llm_messages if m.get("role") == "tool"][-1]["content"]
    assert "mispredicted" in tool
    # It predicted column 0; the mark really went to column 2.
    assert "(0, 1, 5, 0)" in tool and "(2, 1, 0, 5)" in tool


def test_a_correct_forward_model_reports_an_empty_disagreement(monkeypatch):
    """Empty rather than absent: "I checked and it held" is a different
    message from "nobody checked", and the model acts on the difference."""
    env = _FakeEnv(_frame(_grid(mark=(1, 1, 5))), [_frame(_grid(mark=(1, 2, 5)))])
    a = _turn_agent(env)
    _script(monkeypatch, [
        _reply(code=SLIDE_RIGHT + "result = action('ACTION1')"),
        _reply(content="ok"),
    ])

    a._llm_turn(agent_mod.grid_of(env.current), env.current)

    tool = [m for m in a._llm_messages if m.get("role") == "tool"][-1]["content"]
    assert "'mispredicted': []" in tool


def test_a_predict_that_raises_never_stops_the_game_being_played(monkeypatch):
    env = _FakeEnv(_frame(_grid(mark=(1, 1, 5))), [_frame(_grid(mark=(1, 2, 5)))])
    a = _turn_agent(env)
    _script(monkeypatch, [
        _reply(code="def predict(g, act):\n    raise RuntimeError('boom')\n"
                    "result = action('ACTION1')"),
        _reply(content="ok"),
    ])

    assert a._llm_turn(agent_mod.grid_of(env.current), env.current) is True
    assert a._steps == 1


def test_the_forward_model_and_goal_survive_into_the_next_turn(monkeypatch):
    """Rebuilding them each turn would make the strategy impossible: the
    model would re-derive its simulator at full price every turn."""
    env = _FakeEnv(_frame(_grid(mark=(1, 1, 5))))
    a = _turn_agent(env)
    _script(monkeypatch, [
        _reply(code=SLIDE_RIGHT + "def is_goal(g):\n    return False\n"),
        _reply(content="built it"),
        _reply(code="result = (callable(predict), callable(is_goal))"),
        _reply(content="still here"),
    ])

    grid = agent_mod.grid_of(env.current)
    a._llm_turn(grid, env.current)
    a._llm_turn(grid, env.current)

    tool = [m for m in a._llm_messages if m.get("role") == "tool"][-1]["content"]
    assert "(True, True)" in tool


def test_the_transition_log_is_bounded():
    """It rides in memory for a whole game, across 25 concurrent threads."""
    assert agent_mod.TRANSITION_LOG <= 1000
    a = _turn_agent(_FakeEnv(_frame(_grid())))
    assert a._transitions.maxlen == agent_mod.TRANSITION_LOG


def test_whether_the_forward_model_improved_is_recorded(monkeypatch):
    """The question a GPU run exists to answer. Three previous runs could
    not answer theirs because nothing recorded it, so the best accuracy
    seen is tracked and printed on every turn line."""
    env = _FakeEnv(_frame(_grid(mark=(1, 1, 5))), [_frame(_grid(mark=(1, 2, 5)))])
    a = _turn_agent(env)
    wrong = SLIDE_RIGHT.replace("rows[y][x + 1]", "rows[y][x - 1]")
    _script(monkeypatch, [
        _reply(code="action('ACTION1')"),
        _reply(code=wrong + "result = verify(predict)"),
        _reply(code=SLIDE_RIGHT + "result = verify(predict)"),
        _reply(content="fixed it"),
    ])

    a._llm_turn(agent_mod.grid_of(env.current), env.current)

    assert a._verify_calls == 2
    assert a._verify_best == 1.0, "the improvement must be visible"


def test_planning_attempts_are_counted_separately_from_successes(monkeypatch):
    """A model that plans constantly and never finds a route looks, in the
    action counts alone, exactly like one that never tried."""
    env = _FakeEnv(_frame(_grid(mark=(1, 1, 5))))
    a = _turn_agent(env)
    _script(monkeypatch, [
        _reply(code=SLIDE_RIGHT
               + "plan(predict, lambda g: False, max_depth=2)\n"
                 "result = plan(predict, lambda g: any(r[5] == 5 for r in g))"),
        _reply(content="ok"),
    ])

    a._llm_turn(agent_mod.grid_of(env.current), env.current)

    assert a._plan_calls == 2
    assert a._plans_found == 1


def test_a_usable_model_that_never_planned_is_called_out(monkeypatch):
    """The exact v16 failure. It drove verify accuracy from 0.04 to 0.58,
    called plan() zero times in the whole run, and went back to playing
    actions one at a time. Reaching a usable model and not planning with it
    is the most expensive state the agent can sit in, so it is named."""
    env = _FakeEnv(_frame(_grid(mark=(1, 1, 5))), [_frame(_grid(mark=(1, 2, 5)))])
    a = _turn_agent(env)
    _script(monkeypatch, [
        _reply(code=SLIDE_RIGHT + "action('ACTION1')\nresult = verify(predict)"),
        _reply(content="ok"),
    ])

    a._llm_turn(agent_mod.grid_of(env.current), env.current)

    assert a._verify_best >= agent_mod.PLAN_NUDGE_ACCURACY
    assert a._plan_calls == 0
    assert "not called plan() once" in a._llm_note


def test_the_nudge_stops_once_it_has_planned(monkeypatch):
    """Otherwise it becomes noise the model learns to skip."""
    env = _FakeEnv(_frame(_grid(mark=(1, 1, 5))), [_frame(_grid(mark=(1, 2, 5)))])
    a = _turn_agent(env)
    _script(monkeypatch, [
        _reply(code=SLIDE_RIGHT + "action('ACTION1')\nverify(predict)\n"
                                  "result = plan(predict, lambda g: False)"),
        _reply(content="ok"),
    ])

    a._llm_turn(agent_mod.grid_of(env.current), env.current)

    assert a._plan_calls == 1
    assert "not called plan() once" not in a._llm_note


def test_a_weak_model_is_not_nudged_to_plan(monkeypatch):
    """Planning on a model that is wrong about everything wastes a turn."""
    env = _FakeEnv(_frame(_grid(mark=(1, 1, 5))), [_frame(_grid(mark=(1, 2, 5)))])
    a = _turn_agent(env)
    wrong = SLIDE_RIGHT.replace("rows[y][x + 1]", "rows[y][x - 1]")
    _script(monkeypatch, [
        _reply(code=wrong + "action('ACTION1')\nresult = verify(predict)"),
        _reply(content="ok"),
    ])

    a._llm_turn(agent_mod.grid_of(env.current), env.current)

    assert a._verify_best == 0.0
    assert "not called plan() once" not in a._llm_note


def test_the_prompt_no_longer_gates_planning_on_perfection():
    """The bug that cost v16 its whole run: 'verify again until accuracy is
    1.0' is a gate that never opens on a board with any noise in it, and
    the model obeyed it literally.

    The gate is gone in the stronger sense now: searching does not wait on
    a simulator because it no longer needs one. A model that cannot build
    a forward model is told to drop it and search the real game, which is
    the case v19 hit and had no answer for -- `predict` raised on branch
    after branch and the plan came back empty with nothing saying why."""
    prompt = agent_mod.build_system_prompt([1, 2, 6])
    assert "until accuracy is 1.0" not in prompt
    assert "abandon it -- do not repair it" in prompt
    assert "never calling `plan`" in prompt


def test_the_prompt_forbids_the_repeat_loop_by_name():
    """v16 turns 6 and 7 ran `for i in range(10): action('ACTION4')` and
    spent 104 actions between them."""
    prompt = agent_mod.build_system_prompt([1, 2, 6])
    assert "for i in range(10): action" in prompt
    assert "no more than twenty" in prompt


def test_the_prompt_teaches_the_free_tools():
    prompt = agent_mod.build_system_prompt([1, 2, 6])
    assert "verify(predict)" in prompt
    assert "plan(predict, is_goal" in prompt
    assert "mispredicted" in prompt
    # The two scoring facts that redirected the whole approach.
    assert "capped" in prompt
    assert "tutorial" in prompt


# --- knowing WHICH of the two things was wrong --------------------------
#
# v17 built a forward model that verified at 1.00 accuracy, planned with
# it three times, executed all three routes, and cleared nothing -- because
# `is_goal` was a coordinate it had guessed. It read each failure as
# evidence against the simulator, went back to probing, and spent the rest
# of the run grinding sixty actions a turn. The simulator was never the
# problem. These cover saying so.


def test_tour_is_available_and_costs_no_actions(monkeypatch):
    env = _FakeEnv(_frame(_grid(mark=(1, 1, 5))))
    a = _turn_agent(env)
    code = (
        SLIDE_RIGHT
        + "result = tour(predict, {'a': lambda g: any(r[3] == 5 for r in g),\n"
          "                        'b': lambda g: any(r[5] == 5 for r in g)})\n"
    )
    _script(monkeypatch, [_reply(code=code), _reply(content="ok")])

    a._llm_turn(agent_mod.grid_of(env.current), env.current)

    assert a._steps == 0, "touring must not touch the environment"
    assert a._tour_calls == 1
    tool = [m for m in a._llm_messages if m.get("role") == "tool"][-1]["content"]
    assert "'reached': ['a', 'b']" in tool


def test_the_tour_orders_the_candidates_by_what_they_cost(monkeypatch):
    """Cheapest hypothesis first, so the dear ones are the likeliest never
    to be played at all."""
    env = _FakeEnv(_frame(_grid(mark=(1, 1, 5))))
    a = _turn_agent(env)
    code = (
        SLIDE_RIGHT
        + "r = tour(predict, {'far': lambda g: any(x[6] == 5 for x in g),\n"
          "                   'near': lambda g: any(x[2] == 5 for x in g)})\n"
        "result = r['reached']\n"
    )
    _script(monkeypatch, [_reply(code=code), _reply(content="ok")])

    a._llm_turn(agent_mod.grid_of(env.current), env.current)

    tool = [m for m in a._llm_messages if m.get("role") == "tool"][-1]["content"]
    assert "['near', 'far']" in tool


def _goal_miss_agent(monkeypatch, extra=""):
    """A turn that plans, then plays a route the simulator gets exactly
    right, and does not clear the level."""
    env = _FakeEnv(
        _frame(_grid(mark=(1, 1, 5))),
        [_frame(_grid(mark=(1, 2, 5))), _frame(_grid(mark=(1, 3, 5)))],
    )
    a = _turn_agent(env)
    code = (
        SLIDE_RIGHT
        + "plan(predict, lambda g: any(r[3] == 5 for r in g))\n"
        + extra
        + "result = action(['ACTION1', 'ACTION1'])\n"
    )
    _script(monkeypatch, [_reply(code=code), _reply(content="ok")])
    a._llm_turn(agent_mod.grid_of(env.current), env.current)
    return a


def test_a_route_that_ran_clean_and_cleared_nothing_blames_the_goal(monkeypatch):
    a = _goal_miss_agent(monkeypatch)

    assert a._goal_misses == 1
    tool = [m for m in a._llm_messages if m.get("role") == "tool"][-1]["content"]
    assert "goal_miss" in tool
    assert "is not the goal" in tool


def test_the_goal_miss_says_not_to_touch_the_simulator(monkeypatch):
    """The inference v17 got backwards, stated in the words that stop it."""
    a = _goal_miss_agent(monkeypatch)
    tool = [m for m in a._llm_messages if m.get("role") == "tool"][-1]["content"]
    assert "do NOT rewrite `predict`" in tool
    assert "rewrite `is_goal`" in tool
    assert "tour(predict" in tool


def test_probing_before_any_plan_is_not_a_goal_miss(monkeypatch):
    """Ordinary exploration does not advance the level either, and calling
    that a wrong goal would fire the diagnosis on every early turn."""
    env = _FakeEnv(
        _frame(_grid(mark=(1, 1, 5))),
        [_frame(_grid(mark=(1, 2, 5))), _frame(_grid(mark=(1, 3, 5)))],
    )
    a = _turn_agent(env)
    code = SLIDE_RIGHT + "result = action(['ACTION1', 'ACTION1'])\n"
    _script(monkeypatch, [_reply(code=code), _reply(content="ok")])

    a._llm_turn(agent_mod.grid_of(env.current), env.current)

    assert a._goal_misses == 0
    tool = [m for m in a._llm_messages if m.get("role") == "tool"][-1]["content"]
    assert "goal_miss" not in tool


def test_a_route_the_simulator_got_wrong_is_not_a_goal_miss(monkeypatch):
    """`mispredicted` is non-empty, so the simulator IS the suspect and the
    two diagnoses must not both fire."""
    env = _FakeEnv(
        _frame(_grid(mark=(1, 1, 5))),
        [_frame(_grid(mark=(5, 5, 5))), _frame(_grid(mark=(6, 6, 5)))],
    )
    a = _turn_agent(env)
    code = (
        SLIDE_RIGHT
        + "plan(predict, lambda g: any(r[3] == 5 for r in g))\n"
        "result = action(['ACTION1', 'ACTION1'])\n"
    )
    _script(monkeypatch, [_reply(code=code), _reply(content="ok")])

    a._llm_turn(agent_mod.grid_of(env.current), env.current)

    assert a._goal_misses == 0
    tool = [m for m in a._llm_messages if m.get("role") == "tool"][-1]["content"]
    assert "mispredicted" in tool and "goal_miss" not in tool


def test_a_route_that_cleared_the_level_is_not_a_goal_miss(monkeypatch):
    env = _FakeEnv(
        _frame(_grid(mark=(1, 1, 5))),
        [_frame(_grid(mark=(1, 2, 5))), _frame(_grid(mark=(1, 3, 5)), levels=1)],
    )
    a = _turn_agent(env)
    code = (
        SLIDE_RIGHT
        + "plan(predict, lambda g: any(r[3] == 5 for r in g))\n"
        "result = action(['ACTION1', 'ACTION1'])\n"
    )
    _script(monkeypatch, [_reply(code=code), _reply(content="ok")])

    a._llm_turn(agent_mod.grid_of(env.current), env.current)

    assert a._goal_misses == 0


def test_guessing_targets_one_at_a_time_gets_nudged_towards_tour(monkeypatch):
    a = _goal_miss_agent(monkeypatch)
    assert "tour(" in a._llm_note
    assert "`is_goal` is wrong" in a._llm_note


def test_the_goal_nudge_stops_once_it_has_toured(monkeypatch):
    """Belt and braces, but only until the braces are on -- a nudge that
    keeps firing after it has been acted on is noise in the prompt."""
    a = _goal_miss_agent(
        monkeypatch, extra="tour(predict, {'a': lambda g: False})\n"
    )
    assert a._goal_misses == 1
    assert a._tour_calls == 1
    assert "tour(" not in a._llm_note


# --- checking the simulator without being asked -------------------------
#
# v18 called verify() four times, all inside turn 2, scored 1.00 over
# thirteen transitions, and then played a hundred and forty more actions
# across three turns without ever checking again. The only number anyone
# saw was the model's own high-water mark, which is not a check on
# anything. The turn now measures it.


def _drifting_agent(monkeypatch, second_code):
    """Turn 1 fits a simulator to the moves so far; turn 2 plays moves it
    does not fit."""
    env = _FakeEnv(
        _frame(_grid(mark=(1, 1, 5))),
        [_frame(_grid(mark=(1, 2, 5))), _frame(_grid(mark=(1, 3, 5))),
         _frame(_grid(mark=(7, 7, 5))), _frame(_grid(mark=(6, 6, 5)))],
    )
    a = _turn_agent(env)
    _script(monkeypatch, [
        _reply(code=SLIDE_RIGHT + "result = action(['ACTION1', 'ACTION1'])\n"),
        _reply(content="ok"),
        _reply(code=second_code),
        _reply(content="ok"),
    ])
    a._llm_turn(agent_mod.grid_of(env.current), env.current)
    first = a._verify_live
    a._llm_turn(agent_mod.grid_of(env.current), env.current)
    return a, first


def test_the_turn_scores_the_live_simulator_without_being_asked(monkeypatch):
    env = _FakeEnv(
        _frame(_grid(mark=(1, 1, 5))),
        [_frame(_grid(mark=(1, 2, 5))), _frame(_grid(mark=(1, 3, 5)))],
    )
    a = _turn_agent(env)
    code = SLIDE_RIGHT + "result = action(['ACTION1', 'ACTION1'])\n"
    _script(monkeypatch, [_reply(code=code), _reply(content="ok")])

    a._llm_turn(agent_mod.grid_of(env.current), env.current)

    assert a._verify_live == 1.0
    assert a._verify_calls == 0, "the model never called verify itself"


def test_a_simulator_that_has_gone_stale_is_reported(monkeypatch):
    """The v18 mechanism: a 1.00 measured once and trusted long after."""
    a, first = _drifting_agent(
        monkeypatch, "result = action(['ACTION1', 'ACTION1'])\n"
    )
    assert first == 1.0
    assert a._verify_live < 1.0
    assert "Checked for you" in a._llm_note
    assert "%" in a._llm_note


def test_a_simulator_that_still_fits_is_not_nagged(monkeypatch):
    env = _FakeEnv(
        _frame(_grid(mark=(1, 1, 5))),
        [_frame(_grid(mark=(1, 2, 5))), _frame(_grid(mark=(1, 3, 5)))],
    )
    a = _turn_agent(env)
    code = SLIDE_RIGHT + "result = action(['ACTION1', 'ACTION1'])\n"
    _script(monkeypatch, [_reply(code=code), _reply(content="ok")])

    a._llm_turn(agent_mod.grid_of(env.current), env.current)

    assert "Checked for you" not in a._llm_note


def test_the_check_survives_a_predict_that_raises(monkeypatch):
    """It runs outside the model's own try/except, so it must not be able
    to end the game."""
    env = _FakeEnv(
        _frame(_grid(mark=(1, 1, 5))),
        [_frame(_grid(mark=(1, 2, 5))), _frame(_grid(mark=(1, 3, 5)))],
    )
    a = _turn_agent(env)
    code = (
        "def predict(g, act):\n    raise RuntimeError('boom')\n"
        "result = action(['ACTION1', 'ACTION1'])\n"
    )
    _script(monkeypatch, [_reply(code=code), _reply(content="ok")])

    a._llm_turn(agent_mod.grid_of(env.current), env.current)

    assert a._verify_live == 0.0


def test_no_simulator_yet_means_nothing_to_check(monkeypatch):
    env = _FakeEnv(_frame(_grid(mark=(1, 1, 5))),
                   [_frame(_grid(mark=(1, 2, 5)))])
    a = _turn_agent(env)
    _script(monkeypatch, [_reply(code="result = action('ACTION1')\n"),
                          _reply(content="ok")])

    a._llm_turn(agent_mod.grid_of(env.current), env.current)

    assert a._verify_live == 0.0
    assert "Checked for you" not in a._llm_note


def test_exhausting_the_state_space_reaches_the_model_and_outranks_the_rest(
    monkeypatch,
):
    """Three diagnoses now compete for one note. This is the deepest of
    them -- while the state space is wrong, nothing else is worth saying."""
    env = _FakeEnv(_frame(_grid(mark=(1, 1, 5))))
    a = _turn_agent(env)
    code = (
        SLIDE_RIGHT
        + "result = tour(predict, {'a': lambda g: any(r[3] == 5 for r in g),\n"
          "                        'never': lambda g: False})\n"
    )
    _script(monkeypatch, [_reply(code=code), _reply(content="ok")])

    a._llm_turn(agent_mod.grid_of(env.current), env.current)

    assert a._exhaustions == 1
    tool = [m for m in a._llm_messages if m.get("role") == "tool"][-1]["content"]
    assert "exhausted" in tool
    assert "Stop writing goals" in a._llm_note


def test_a_tour_that_reached_everything_is_not_an_exhaustion(monkeypatch):
    env = _FakeEnv(_frame(_grid(mark=(1, 1, 5))))
    a = _turn_agent(env)
    code = (
        SLIDE_RIGHT
        + "result = tour(predict, {'a': lambda g: any(r[3] == 5 for r in g)})\n"
    )
    _script(monkeypatch, [_reply(code=code), _reply(content="ok")])

    a._llm_turn(agent_mod.grid_of(env.current), env.current)

    assert a._exhaustions == 0


def test_drift_is_measured_against_the_harness_own_readings(monkeypatch):
    """A model that never calls verify() still gets caught.

    The first cut compared the live score against `_verify_best`, which is
    only ever set by the model calling verify() itself -- so a model that
    never called it had that sitting at 0.0 and could never be found to
    have drifted, which is precisely the model most likely to.
    """
    a, first = _drifting_agent(
        monkeypatch, "result = action(['ACTION1', 'ACTION1'])\n"
    )
    assert a._verify_calls == 0, "the model never self-reported anything"
    assert a._verify_best == 0.0
    assert first == 1.0 and a._verify_live < 1.0
    assert "Checked for you" in a._llm_note


def test_the_prompt_makes_writing_the_search_the_job():
    """Ported from the solver the 1.21-1.86 leaderboard cluster runs
    (Tufa Labs' TAAF, public and MIT licensed). The division of labour is
    the whole difference: they hand the model structured state and one
    `action()` and let it program the search; we wrote the search and
    asked the model to feed it a simulator. Writing a correct forward
    model of an unknown game is strictly harder than writing a search
    that calls the real game, and v19 paid for that -- its blocker was
    `predict raised on N branches`."""
    prompt = agent_mod.build_system_prompt([1, 2, 6])
    assert "WRITE THE SEARCH" in prompt
    assert "This is the job" in prompt
    # The algorithms their prompt names, because a model told only to
    # "search" writes a loop over actions instead.
    for algorithm in ("breadth-first search", "flood fill", "beam search"):
        assert algorithm in prompt, algorithm
    # Writing a simulator is now the exception, not the route.
    assert "OPTIONAL SHORTCUT" in prompt


def test_the_prompt_names_the_importable_modules():
    """The sandbox allows a fixed subset. A model that assumes numpy
    spends a tool call finding out."""
    prompt = agent_mod.build_system_prompt([1, 2, 6])
    for module in ("heapq", "collections", "itertools", "functools"):
        assert module in prompt, module


def test_the_prompt_warns_about_the_timer_bar():
    """`hud_mask` fixes the search's state key; it does nothing for the
    model, which still receives the strip as a row of tiny objects."""
    prompt = agent_mod.build_system_prompt([1, 2, 6])
    assert "THE BAR AT THE EDGE IS NOT THE GAME" in prompt
    assert "did not change" in prompt


def test_objects_are_the_primary_view_not_the_raw_grid():
    """TAAF withholds the numeric grid outright. `plan()` and `tour()`
    still read it here, so it stays available -- but the prompt no longer
    presents it as an equal way to look at the board."""
    prompt = agent_mod.build_system_prompt([1, 2, 6])
    assert "work from the OBJECTS" in prompt
    assert "never to scan the whole board" in prompt


# --- LLM_PER_ACTION_MASK -------------------------------------------------


def test_learn_mask_runs_once_per_real_action_not_once_per_turn(monkeypatch):
    """The whole point (see LLM_PER_ACTION_MASK's comment): without this
    flag `_learn_mask` only ever sees a turn's worth of change at once,
    because it is normally called once from `choose_action`, which the
    model's own turn calls into exactly once no matter how many real
    actions it plays. Pin that turning the flag on calls it once per real
    action instead."""
    monkeypatch.setattr(agent_mod, "LLM_PER_ACTION_MASK", 1)
    env = _FakeEnv(
        _frame(_grid()), [_frame(_grid()), _frame(_grid()), _frame(_grid())]
    )
    a = _turn_agent(env)
    a._turn_frame = env.current
    calls: list[tuple] = []
    monkeypatch.setattr(
        a, "_learn_mask", lambda flat, w, h: calls.append((w, h)) or False
    )

    a._llm_act(["ACTION1", "ACTION1", "ACTION1"])

    assert len(calls) == 3, "not called once per real action"
    assert all(wh == (8, 8) for wh in calls)


def test_learn_mask_is_not_called_per_action_when_the_flag_is_off(monkeypatch):
    env = _FakeEnv(_frame(_grid()), [_frame(_grid()), _frame(_grid())])
    a = _turn_agent(env)
    a._turn_frame = env.current
    calls: list[tuple] = []
    monkeypatch.setattr(
        a, "_learn_mask", lambda flat, w, h: calls.append((w, h)) or False
    )

    a._llm_act(["ACTION1", "ACTION1"])

    assert calls == [], "LLM_PER_ACTION_MASK defaults off"


def test_mid_turn_mask_adoption_resets_the_graph_immediately(monkeypatch):
    """The same reset `choose_action` performs on adoption must happen
    right here, mid-turn -- nothing about the model's own turn logic
    depends on the search graph (see the "model's steps are not recorded
    as graph edges" note near `_pending`), so there is nothing to
    interrupt; the turn keeps playing against a freshly rekeyed graph."""
    monkeypatch.setattr(agent_mod, "LLM_PER_ACTION_MASK", 1)
    env = _FakeEnv(_frame(_grid()), [_frame(_grid()), _frame(_grid())])
    a = _turn_agent(env)
    a._turn_frame = env.current
    a._blocked = {("look", 1, 1)}
    a._at = (1, 1)
    a._at_confirmed = 3
    a._total = {1: 1}
    a._cursor = {1: 1}
    a._edges = {1: {2: 3}}
    a._fatal = {(1, 2)}
    a._deaths = {(1, 2): 1}
    a._cache = {1: [2]}
    a._plan = deque([1, 2])
    a._level_start_node = 1
    a._level_start_board = 1
    a._board_idx = 0
    a._status_boards = {1: 2}
    a._status = 2
    a._last_respawn = 5
    a._pending = (1, 2)
    a._sol_start = 1
    a._sol_edges = {1: {2: 3}}
    verdicts = [False, True]
    monkeypatch.setattr(a, "_learn_mask", lambda flat, w, h: verdicts.pop(0))

    a._llm_act(["ACTION1", "ACTION1"])

    assert a._total == {}
    assert a._edges == {}
    assert a._cache == {}
    assert a._blocked == set()
    assert a._at is None
    assert a._pending is None
    assert a._sol_start is None
    assert a._sol_edges == {}
    # The turn was not interrupted: both scripted actions still played.
    assert len(env.sent) == 2


# --- LLM_SEED_EDGES -------------------------------------------------------


def test_llm_seed_edges_populates_the_graph_from_real_transitions(monkeypatch):
    """The whole point (see LLM_SEED_EDGES's comment): `_edges` -- normally
    untouched while the model drives -- gets exactly what `_observe` would
    have recorded had a search-driven action taken this step."""
    monkeypatch.setattr(agent_mod, "LLM_SEED_EDGES", 1)
    before_rows = _grid()
    after_rows = _grid(mark=(0, 0, 5))
    env = _FakeEnv(_frame(before_rows), [_frame(after_rows)])
    a = _turn_agent(env)
    a._turn_frame = env.current
    a._kept = None
    a._edges = {}
    a._fatal = set()
    a._deaths = {}

    a._llm_act(["ACTION1"])

    before_node = a._node_key(_flat(before_rows))
    after_node = a._node_key(_flat(after_rows))
    code = agent_mod.encode_action(1, None, None)
    assert a._edges == {before_node: {code: after_node}}


def test_llm_seed_edges_is_off_by_default(monkeypatch):
    env = _FakeEnv(_frame(_grid()), [_frame(_grid(mark=(0, 0, 5)))])
    a = _turn_agent(env)
    a._turn_frame = env.current
    a._kept = None
    a._edges = {}
    a._fatal = set()
    a._deaths = {}

    a._llm_act(["ACTION1"])

    assert a._edges == {}


def test_llm_seed_edges_marks_a_move_fatal_after_death_tolerance(monkeypatch):
    """Mirrors `_observe`'s own DEATH_TOLERANCE logic exactly, so a move
    that has already killed the model is remembered as lethal the moment
    the search fallback takes over -- not re-discovered by dying again."""
    monkeypatch.setattr(agent_mod, "LLM_SEED_EDGES", 1)
    before_rows = _grid()
    death_frame = _frame(before_rows, state=GameState.GAME_OVER)
    env = _FakeEnv(_frame(before_rows))
    a = _turn_agent(env)
    a._kept = None
    a._edges = {}
    a._fatal = set()
    a._deaths = {}
    code = agent_mod.encode_action(1, None, None)
    before_node = a._node_key(_flat(before_rows))
    # A live edge for this move exists before the fatal verdict lands --
    # the tolerance logic must be able to retract it, not just refuse to
    # add a new one.
    a._edges[before_node] = {code: before_node}

    for _ in range(agent_mod.DEATH_TOLERANCE):
        env.current = _frame(before_rows)
        env.results = [death_frame]
        a._turn_frame = env.current
        a._llm_act(["ACTION1"])

    assert (before_node, code) in a._fatal
    assert code not in a._edges.get(before_node, {})
