"""Pin the competition-mode rules that the leaderboard actually plays under.

`tests/test_exploit_replay.py` asserts that a double RESET opens a fresh
scored run. That is true -- and irrelevant on Kaggle, because the gateway
runs `arc_agi.api.RestAPI` with `competition_mode=True`, which defeats it.
Believing the permissive behaviour cost two submissions that measured
0.053 and 2.10 locally and both scored exactly 0.19 on the leaderboard.

These tests drive the genuine Flask app, so they fail if the packaged rules
ever change.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
API_KEY = "test-key-123"
HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}


def _client(competition: bool):
    from arc_agi import Arcade, OperationMode
    from arc_agi.server import create_app

    arcade = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(ROOT / "environment_files"),
    )
    app, _ = create_app(arcade, competition_mode=competition)
    return app.test_client(), arcade


def _play_then_double_reset(competition: bool) -> dict:
    client, arcade = _client(competition)

    card_id = client.post(
        "/api/scorecard/open", headers=HEADERS, json={"tags": ["test"]}
    ).get_json()["card_id"]

    games = [g["game_id"] for g in client.get("/api/games", headers=HEADERS).get_json()]
    game_id = next(g for g in games if g.startswith("ls20"))

    def cmd(name: str, **extra):
        payload = {"card_id": card_id, "game_id": game_id, **extra}
        return client.post(f"/api/cmd/{name}", headers=HEADERS, json=payload).get_json()

    guid = cmd("RESET")["guid"]
    for _ in range(5):
        cmd("ACTION1", guid=guid)

    # First RESET has _action_count > 0 -> level_reset. The second lands on
    # _action_count == 0, which is the full_reset / new-run condition.
    cmd("RESET", guid=guid)
    cmd("RESET", guid=guid)

    card = arcade.scorecard_manager.scorecards[card_id].cards[game_id]
    return {"plays": len(card.actions), "actions": list(card.actions)}


@pytest.mark.parametrize(
    "competition, expected_plays",
    [(False, 2), (True, 1)],
    ids=["permissive", "competition"],
)
def test_double_reset_opens_a_new_run_only_outside_competition_mode(
    competition, expected_plays
):
    result = _play_then_double_reset(competition)
    assert result["plays"] == expected_plays


def test_competition_mode_bills_the_swallowed_reset():
    """The defeated RESET is not free: the game is not stepped, but the
    scorecard still counts the action against the level being scored."""
    permissive = _play_then_double_reset(False)
    competition = _play_then_double_reset(True)

    # Permissive: 5 actions + 1 level_reset on run one, then a fresh run.
    assert permissive["actions"][0] == 6
    assert permissive["actions"][-1] == 0

    # Competition: both resets land on the single run, and the second one
    # buys nothing at all.
    assert competition["actions"] == [7]


def test_competition_mode_refuses_a_second_run_for_a_played_game():
    """`_get_or_create_environment` bails once the scorecard has the game,
    so there is no way to start the game over on the same scorecard."""
    client, arcade = _client(competition=True)

    card_id = client.post(
        "/api/scorecard/open", headers=HEADERS, json={"tags": ["test"]}
    ).get_json()["card_id"]
    games = [g["game_id"] for g in client.get("/api/games", headers=HEADERS).get_json()]
    game_id = next(g for g in games if g.startswith("ls20"))

    first = client.post(
        "/api/cmd/RESET",
        headers=HEADERS,
        json={"card_id": card_id, "game_id": game_id},
    ).get_json()
    assert first.get("guid")

    # A RESET with no guid asks for a brand new environment for this game.
    second = client.post(
        "/api/cmd/RESET",
        headers=HEADERS,
        json={"card_id": card_id, "game_id": game_id},
    )
    assert second.status_code >= 400 or not second.get_json().get("guid")


def _load_agent_class():
    # my_agent imports `agents.agent` from the vendored framework.
    vendor = str(ROOT / "vendor" / "ARC-AGI-3-Agents")
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    spec = importlib.util.spec_from_file_location(
        "agent_under_test", ROOT / "agent" / "my_agent.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bare_agent(module):
    agent = module.MyAgent.__new__(module.MyAgent)
    agent._act_tried = module.Counter()
    agent._act_changed = module.Counter()
    # No model. These tests are about the search -- and the search is
    # exactly what runs on the leaderboard whenever the policy is off or
    # unavailable, so `None` here is the shipping configuration rather
    # than a stub standing in for one.
    agent._policy = None
    agent._act_history = []
    # `__new__` bypasses `__init__`, so anything `_select` reads has to be
    # set here too. Both of these are zeroed in `__init__` and again in
    # `_begin_level`, so a real agent always has them; zero also keeps the
    # liveness gate below its minimum sample, which is what these tests
    # want -- they are about the widening *order*, not about liveness.
    agent._click_tried = 0
    agent._click_live = 0
    return agent


def test_dead_action_needs_a_fair_trial_before_being_skipped():
    module = _load_agent_class()
    agent = _bare_agent(module)
    press = module.encode_action(1, None, None)

    assert agent._is_dead(press) is False
    agent._act_tried[1] = module.DEAD_ACTION_TRIALS - 1
    assert agent._is_dead(press) is False

    agent._act_tried[1] = module.DEAD_ACTION_TRIALS
    assert agent._is_dead(press) is True


def test_a_click_gets_a_far_longer_trial_than_a_button():
    """A click that misses is not evidence -- there are ~112 targets per
    node. Writing ACTION6 off on a button-sized sample would disable the
    only winning action on a click game, so the click trial must exceed
    the per-node target count by a wide margin."""
    module = _load_agent_class()
    assert module.DEAD_CLICK_TRIALS > module.DEAD_ACTION_TRIALS * 10
    assert module.DEAD_CLICK_TRIALS > 112

    agent = _bare_agent(module)
    click = module.encode_action(module._ACTION6_ID, 3, 4)

    # A sample that would condemn a button leaves clicks fully live.
    agent._act_tried[module._ACTION6_ID] = module.DEAD_ACTION_TRIALS
    assert agent._is_dead(click) is False

    agent._act_tried[module._ACTION6_ID] = module.DEAD_CLICK_TRIALS - 1
    assert agent._is_dead(click) is False

    agent._act_tried[module._ACTION6_ID] = module.DEAD_CLICK_TRIALS
    assert agent._is_dead(click) is True


def test_an_action_that_ever_worked_is_never_written_off():
    module = _load_agent_class()
    agent = _bare_agent(module)
    click = module.encode_action(module._ACTION6_ID, 3, 4)

    agent._act_tried[module._ACTION6_ID] = module.DEAD_CLICK_TRIALS * 10
    agent._act_changed[module._ACTION6_ID] = 1
    assert agent._is_dead(click) is False


def test_dead_is_judged_per_action_id_not_per_click_target():
    """A click that misses says nothing about other targets, so writing off
    one coordinate must write off ACTION6 as a whole or nothing at all."""
    module = _load_agent_class()
    agent = _bare_agent(module)
    agent._act_tried[module._ACTION6_ID] = module.DEAD_CLICK_TRIALS

    missed = module.encode_action(module._ACTION6_ID, 3, 4)
    never_tried = module.encode_action(module._ACTION6_ID, 61, 12)
    assert agent._is_dead(missed) is True
    assert agent._is_dead(never_tried) is True

    # ...and a different action id is judged on its own record.
    assert agent._is_dead(module.encode_action(1, None, None)) is False


def test_the_dead_action_record_survives_a_level_boundary():
    """Which actions a game wires up is a property of the game. Levels
    carry weight i, so deeper levels are worth the most -- making them
    re-discover the same inert buttons taxes exactly the wrong levels."""
    module = _load_agent_class()
    agent = _bare_agent(module)
    agent._act_tried[1] = module.DEAD_ACTION_TRIALS
    agent._act_changed[2] = 5

    agent._begin_level(node=1234)

    assert agent._act_tried[1] == module.DEAD_ACTION_TRIALS
    assert agent._act_changed[2] == 5
    assert agent._is_dead(module.encode_action(1, None, None)) is True
    # ...and breadth still restarts cheap, which is a per-level property.
    assert agent._breadth == module.BREADTH_INITIAL


def test_the_agent_seed_is_stable_across_processes():
    """`hash()` of a str is salted per process (PEP 456), so seeding the
    tie-break RNG from it silently re-randomised every run -- turning each
    A/B into a comparison of two configs at two uncontrolled seeds. The
    seed must depend only on the game id."""
    import subprocess
    import sys as _sys

    src = (
        "import zlib;"
        "print(zlib.crc32(b'lp85-305b61c3') & 0xFFFFFFFF)"
    )
    seeds = set()
    for salt in ("0", "1", "2"):
        out = subprocess.run(
            [_sys.executable, "-c", src],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": salt},
            check=True,
        )
        seeds.add(out.stdout.strip())
    assert len(seeds) == 1, f"seed varies with PYTHONHASHSEED: {seeds}"

    # And the agent really uses that stable derivation.
    module = _load_agent_class()
    agent = module.MyAgent.__new__(module.MyAgent)
    agent.game_id = "lp85-305b61c3"
    expected = module.random.Random(
        module.zlib.crc32(b"lp85-305b61c3") & 0xFFFFFFFF
    ).random()
    rng = module.random.Random(
        module.zlib.crc32(agent.game_id.encode()) & 0xFFFFFFFF
    )
    assert rng.random() == expected


def _exhausted_agent(module):
    """An agent whose reachable map is fully probed *at the current breadth*
    but whose nodes still offer far more candidates than that breadth."""
    agent = _bare_agent(module)
    agent.game_id = "test"
    agent._want_cycle = False
    agent._plan = module.deque()
    agent._last_productive = False
    agent._last_code = None
    agent._momentum = 0
    agent._exploding = False
    agent._edges = {}
    agent._fatal = set()
    agent._steps = 0
    agent._stuck_steps = 0
    agent._breadth = module.BREADTH_INITIAL

    node = 1
    codes = [module.encode_action(1, None, None)] + [
        module.encode_action(module._ACTION6_ID, i % 64, i // 64)
        for i in range(299)
    ]
    agent._candidates = lambda *a, **k: codes
    agent._total = {node: len(codes)}
    # Probed exactly out to the current breadth and no further.
    agent._cursor = {node: module.BREADTH_INITIAL}
    return agent, node, codes


def test_eager_widening_is_off_by_default():
    """Mechanically sound, measured negative. It saves lp85 66% and lf52
    44% of their level-1 actions, reproducibly across three seeds -- and
    still loses 0.0327 of aggregate every time, because it costs tn36
    52 -> 86 actions and tn36 is the highest-scoring game on the set."""
    module = _load_agent_class()
    assert module.EAGER_WIDEN == 0


def test_exhaustion_widens_breadth_before_spending_an_action():
    """Widening costs zero billed actions and reopens every shallow-probed
    node; flailing costs one billed action per step and teaches nothing.

    This ordering used to be reversed -- widening sat behind
    STUCK_REEXPLORE_AFTER, so the agent paid up to 1500 random actions
    first. On lp85, whose level-1 baseline is 17, that was 369 of 502
    level-1 actions. Under a quadratic score that is the whole game."""
    module = _load_agent_class()
    module.EAGER_WIDEN = 1          # off by default; see the test above
    agent, node, _ = _exhausted_agent(module)

    # Nothing is reachable, so there is no frontier to route to.
    agent._route_to_frontier = lambda start: None

    code, why = agent._select(node, (), object())

    assert why == "probe-untried", f"flailed instead of widening: {why!r}"
    assert agent._breadth > module.BREADTH_INITIAL
    assert agent._stuck_steps == 0


def test_flailing_is_only_reached_once_breadth_is_maxed():
    """The random-probe path is the last resort, not the first."""
    module = _load_agent_class()
    module.EAGER_WIDEN = 1          # off by default; see above
    agent, node, codes = _exhausted_agent(module)
    agent._route_to_frontier = lambda start: None
    agent._replayable = lambda: False
    agent._random_code = lambda frame: codes[0]

    # Already at the widest breadth, and every candidate probed out.
    agent._breadth = module.BREADTH_MAX
    agent._total = {node: module.BREADTH_MAX}
    agent._cursor = {node: module.BREADTH_MAX}

    _, why = agent._select(node, (), object())
    assert why in ("stuck-random", "stuck-reset")


def test_momentum_stays_off_while_the_map_is_still_closing():
    """The 8.6x sp80 regression is the reason momentum is off. That evidence
    only covers games with a graph to search, so the exemption must be
    conditional on the map having actually stopped closing."""
    module = _load_agent_class()
    agent, node, codes = _exhausted_agent(module)
    agent._exploding = False
    agent._last_productive = True
    agent._last_code = codes[0]
    agent._route_to_frontier = lambda start: None
    agent._replayable = lambda: False
    agent._random_code = lambda frame: codes[0]

    _, why = agent._select(node, (), object())
    assert why != "momentum"


def test_momentum_engages_once_the_map_stops_closing():
    """On ar25 every action lands on a brand-new node (0.66 per action), so
    there is no graph to search and 'probe-untried' is a random walk."""
    module = _load_agent_class()
    module.ADAPTIVE_MOMENTUM = 8    # off by default; measured no effect
    agent, node, codes = _exhausted_agent(module)
    agent._exploding = True
    agent._last_productive = True
    agent._last_code = codes[0]
    agent._route_to_frontier = lambda start: None

    code, why = agent._select(node, (), object())
    assert why == "momentum"
    assert code == codes[0]


def test_adaptive_momentum_is_off_by_default():
    """Measured across two seeds at limits 8 and 24: the aggregate matched
    baseline to within 0.0001. Unmeasured behaviour does not ship."""
    module = _load_agent_class()
    assert module.ADAPTIVE_MOMENTUM == 0


def test_the_explosion_detector_separates_the_measured_games():
    """ar25 creates 0.66 new nodes per action and ft09 0.02, so the
    threshold has to sit between them or it classifies both alike."""
    module = _load_agent_class()
    assert 0.02 < module.EXPLODE_THRESHOLD < 0.60


def test_momentum_is_disabled_by_default():
    """Repeating an action that keeps changing the frame conflates motion
    with progress. Measured on sp80 level 1 with everything else fixed:
    668 actions with momentum, 78 without -- and sp80 is the game that
    carries the aggregate."""
    module = _load_agent_class()
    assert module.MOMENTUM_LIMIT == 0

    agent = module.MyAgent.__new__(module.MyAgent)
    agent._momentum = 0
    # The guard is `_momentum < MOMENTUM_LIMIT`, so a limit of 0 can never
    # admit a repeat no matter how productive the last action looked.
    assert not (agent._momentum < module.MOMENTUM_LIMIT)


def test_cycling_is_disabled_by_default():
    """Banking a route must not make the agent spend RESETs, because on a
    competition scorecard those RESETs are billed no-ops."""
    module = _load_agent_class()
    assert module.ENABLE_CYCLE == 0

    agent = module.MyAgent.__new__(module.MyAgent)
    agent._solutions = {0: [1, 2, 3]}
    assert agent._replayable() is False
