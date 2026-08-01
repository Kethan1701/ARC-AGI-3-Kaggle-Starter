"""Tests for the restated scoring arithmetic.

`scripts/score_math.py` exists to stop us optimising the wrong number, so
it is only worth having if it agrees with `arc_agi/scorecard.py`. These pin
the two properties that are easy to get wrong by reading the formula
casually, and that between them explain why "it cleared a level" was a
misleading thing to celebrate.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "environment_files").exists():  # pragma: no cover
    pytest.skip("no environment_files; run `make setup`", allow_module_level=True)


def _load():
    spec = importlib.util.spec_from_file_location(
        "score_math_test", ROOT / "scripts" / "score_math.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sm = _load()


def test_the_quadratic_penalty_is_quadratic():
    assert sm.level_score(100, 100) == 100.0
    assert sm.level_score(100, 200) == 25.0
    assert sm.level_score(100, 1000) == pytest.approx(1.0)


def test_beating_the_baseline_earns_nothing():
    """The non-obvious one, and it changes what to aim at.

    `to_score` clamps a game's score to `max_weights / total_weights * 100`,
    so a level cleared in half the intended actions scores exactly what one
    cleared in the intended number scores. Aim at baseline; going under it
    is wasted effort that could have gone into reaching the next level.
    """
    baselines = [20, 40, 60]
    at_baseline = sm.env_score(baselines, {1: 20})
    twice_as_fast = sm.env_score(baselines, {1: 10})
    assert at_baseline == twice_as_fast


def test_levels_never_reached_still_sit_in_the_denominator():
    """Which is why one level of a nine-level game is worth so little."""
    short = sm.env_score([20, 40], {1: 20})
    long = sm.env_score([20, 40, 60, 80, 100], {1: 20})
    assert short > long
    # 100 * 1 / (1+2) versus 100 * 1 / (1+2+3+4+5)
    assert short == pytest.approx(100 / 3)
    assert long == pytest.approx(100 / 15)


def test_later_levels_are_weighted_more_heavily():
    baselines = [20, 40, 60]
    first = sm.env_score(baselines, {1: 20})
    through_third = sm.env_score(baselines, {1: 20, 2: 40, 3: 60})
    assert through_third == pytest.approx(100.0)
    assert through_third > 5 * first


def test_an_uncleared_game_scores_zero():
    assert sm.env_score([20, 40], {}) == 0.0


def test_clearing_a_level_slowly_is_worth_almost_nothing():
    """ls20 level 1, at the 526 actions our own search takes against a
    baseline of 22. This is the number that made three GPU runs look like
    progress when they were not."""
    assert sm.level_score(22, 526) < 0.2


def test_every_game_has_baselines():
    games = sm.load_baselines()
    assert len(games) == 25
    assert all(len(b) >= 6 for b in games.values())


def test_the_reference_table_agrees_with_the_scorecard_arithmetic():
    """Spot-check one cell end to end rather than trusting the table."""
    games = sm.load_baselines()
    by_hand = 0.0
    for baselines in games.values():
        n = len(baselines)
        by_hand += 100 * 1 / (n * (n + 1) // 2)
    assert sm._uniform(1, 1) == pytest.approx(by_hand / len(games))


def test_clearing_level_one_everywhere_cannot_reach_the_target():
    """The finding that redirected the whole approach: even a perfect
    level-1 sweep at three times baseline lands nowhere near 1.50, while
    reaching level 3 at the same efficiency clears it."""
    assert sm._uniform(1, 3) < 0.5
    assert sm._uniform(3, 3) > 1.5


def test_it_reproduces_real_scorer_output():
    """Measured, not derived.

    Restating a formula from source can go wrong in ways tests written from
    the same reading will not catch. These six rows are per-game scores and
    per-level action counts copied off a real
    `scripts/score_competition.py --max-steps 5000` run (aggregate 0.1299),
    so they check this module against the scorer's behaviour rather than
    against my reading of the scorer's code.
    """
    observed = {
        "tn36": ({1: 52}, 1.35),
        "sp80": ({1: 78, 2: 3253}, 1.19),
        "r11l": ({1: 112}, 0.18),
        "cd82": ({1: 191, 2: 819}, 0.40),
        "tu93": ({1: 396, 2: 187, 3: 769}, 0.05),
        "ls20": ({1: 526}, 0.01),
    }
    games = sm.load_baselines()
    for game, (actions, expected) in observed.items():
        assert sm.env_score(games[game], actions) == pytest.approx(
            expected, abs=0.005
        ), game


def test_one_efficient_level_beats_three_slow_ones():
    """The thesis, in two rows of that same real run.

    tn36 cleared ONE level, at 52 actions against a baseline of 32, and
    scored 1.35. tu93 cleared THREE, at 21x, 12x and 23x baseline, and
    scored 0.05 -- twenty-seven times less for three times the progress.
    Any measure that counts levels cleared ranks these backwards.
    """
    games = sm.load_baselines()
    one_fast = sm.env_score(games["tn36"], {1: 52})
    three_slow = sm.env_score(games["tu93"], {1: 396, 2: 187, 3: 769})
    assert one_fast > 20 * three_slow


def test_treating_level_one_as_a_tutorial_costs_little():
    """Sacrificing level 1 entirely gives up a small fraction of what
    reaching level 3 is worth, which is what makes it the right place to
    spend actions learning the rules."""
    tutorial = sm._uniform(3, 3, skip_first=True)
    tight = sm._uniform(3, 3)
    assert tutorial > 0.8 * tight
