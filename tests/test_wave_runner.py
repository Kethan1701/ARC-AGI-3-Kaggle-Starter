"""Unit tests for scripts/wave_runner.py's pure game-list-splitting logic.

The subprocess-dispatch half of wave_runner.py (run_waves/main) isn't
covered here -- it shells out to `main.py` against a live gateway, which
is exactly what scripts/kops/push.sh + a Phase A GPU run exercises. This
file is for the part that's cheap and deterministic to get right on its
own: how the game list gets cut into waves.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "wave_runner", ROOT / "scripts" / "wave_runner.py"
)
wave_runner = importlib.util.module_from_spec(spec)
sys.modules["wave_runner"] = wave_runner
spec.loader.exec_module(wave_runner)

split_into_waves = wave_runner.split_into_waves


def test_empty_games_gives_no_waves():
    assert split_into_waves([], 5) == []
    assert split_into_waves([], 0) == []


def test_wave_size_zero_means_one_wave_with_everything():
    games = [f"g{i}" for i in range(25)]
    waves = split_into_waves(games, 0)
    assert waves == [games]


def test_wave_size_negative_means_one_wave_with_everything():
    games = [f"g{i}" for i in range(5)]
    assert split_into_waves(games, -1) == [games]


def test_wave_size_at_least_len_games_means_one_wave():
    games = [f"g{i}" for i in range(10)]
    assert split_into_waves(games, 10) == [games]
    assert split_into_waves(games, 25) == [games]


def test_even_split():
    games = [f"g{i}" for i in range(10)]
    waves = split_into_waves(games, 5)
    assert waves == [games[0:5], games[5:10]]


def test_uneven_split_last_wave_is_the_remainder():
    games = [f"g{i}" for i in range(25)]
    waves = split_into_waves(games, 13)
    assert len(waves) == 2
    assert waves[0] == games[0:13]
    assert waves[1] == games[13:25]
    assert len(waves[1]) == 12


def test_waves_partition_the_input_with_no_overlap_and_no_drops():
    games = [f"g{i}" for i in range(25)]
    waves = split_into_waves(games, 7)
    flattened = [g for wave in waves for g in wave]
    assert flattened == games
    assert len(set(flattened)) == len(games)


def test_wave_size_one_gives_one_wave_per_game():
    games = ["a", "b", "c"]
    waves = split_into_waves(games, 1)
    assert waves == [["a"], ["b"], ["c"]]


def test_returned_waves_are_independent_lists_not_views():
    games = ["a", "b", "c", "d"]
    waves = split_into_waves(games, 2)
    waves[0].append("mutated")
    assert games == ["a", "b", "c", "d"]
