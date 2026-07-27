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
    return agent


def test_dead_action_needs_a_fair_trial_before_being_skipped():
    module = _load_agent_class()
    agent = _bare_agent(module)
    click = module.encode_action(module._ACTION6_ID, 3, 4)

    assert agent._is_dead(click) is False
    agent._act_tried[module._ACTION6_ID] = module.DEAD_ACTION_TRIALS - 1
    assert agent._is_dead(click) is False

    agent._act_tried[module._ACTION6_ID] = module.DEAD_ACTION_TRIALS
    assert agent._is_dead(click) is True


def test_an_action_that_ever_worked_is_never_written_off():
    module = _load_agent_class()
    agent = _bare_agent(module)
    click = module.encode_action(module._ACTION6_ID, 3, 4)

    agent._act_tried[module._ACTION6_ID] = module.DEAD_ACTION_TRIALS * 10
    agent._act_changed[module._ACTION6_ID] = 1
    assert agent._is_dead(click) is False


def test_dead_is_judged_per_action_id_not_per_click_target():
    """A click that misses says nothing about other targets, so writing off
    one coordinate must write off ACTION6 as a whole or nothing at all."""
    module = _load_agent_class()
    agent = _bare_agent(module)
    agent._act_tried[module._ACTION6_ID] = module.DEAD_ACTION_TRIALS

    missed = module.encode_action(module._ACTION6_ID, 3, 4)
    never_tried = module.encode_action(module._ACTION6_ID, 61, 12)
    assert agent._is_dead(missed) is True
    assert agent._is_dead(never_tried) is True

    # ...and a different action id is judged on its own record.
    assert agent._is_dead(module.encode_action(1, None, None)) is False


def test_cycling_is_disabled_by_default():
    """Banking a route must not make the agent spend RESETs, because on a
    competition scorecard those RESETs are billed no-ops."""
    module = _load_agent_class()
    assert module.ENABLE_CYCLE == 0

    agent = module.MyAgent.__new__(module.MyAgent)
    agent._solutions = {0: [1, 2, 3]}
    assert agent._replayable() is False
