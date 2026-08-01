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
    # the planner must not come back with five.
    assert tool.count("ACTION1") == 4


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
    the model obeyed it literally."""
    prompt = agent_mod.build_system_prompt([1, 2, 6])
    assert "until accuracy is 1.0" not in prompt
    assert "Do NOT wait for 1.0" in prompt
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
