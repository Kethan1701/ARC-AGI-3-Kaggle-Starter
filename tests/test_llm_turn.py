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
    a._llm_messages = []
    a._llm_history = []
    a._llm_turns = 0
    a._llm_note = ""
    a._turn_actions = 0
    a._turn_frame = None
    a._turn_level = 0
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


def test_a_dead_server_retires_the_model_for_the_rest_of_the_game(monkeypatch):
    env = _FakeEnv(_frame(_grid()))
    a = _turn_agent(env)

    def dead(_payload):
        raise OSError("connection refused")

    monkeypatch.setattr(agent_mod, "chat_completion", dead)

    assert a._llm_turn(agent_mod.grid_of(env.current), env.current) is False
    assert a._llm_dead is True
    assert env.sent == []


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
