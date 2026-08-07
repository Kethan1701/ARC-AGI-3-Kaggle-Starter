"""Unit tests for scripts/kops/analyze.py's scorecard() merging.

Focused on the 2026-08-06 change: a run can now legitimately contain
several DISTINCT "--- FINAL SCORECARD REPORT ---" entries (one per wave,
see scripts/wave_runner.py), and scorecard() must merge all of them
instead of silently keeping only the first -- the exact wrong-number bug
this file's own docstring warns about.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "kops_analyze", ROOT / "scripts" / "kops" / "analyze.py"
)
analyze = importlib.util.module_from_spec(spec)
sys.modules["kops_analyze"] = analyze
spec.loader.exec_module(analyze)


def _env(gid: str, score: float, levels_completed: int = 1, actions: int = 10) -> dict:
    return {
        "id": gid,
        "score": score,
        "levels_completed": levels_completed,
        "level_count": 6,
        "actions": actions,
        "resets": 1,
    }


def _card(card_id: str, envs: list[dict]) -> dict:
    return {
        "card_id": card_id,
        "score": sum(e["score"] for e in envs) / len(envs),
        "environments": envs,
    }


def _write_log(tmp_path: Path, label: str, data_lines: list[str]) -> None:
    events = [{"stream_name": "stdout", "time": i, "data": line} for i, line in enumerate(data_lines)]
    out = tmp_path / label
    out.mkdir(parents=True, exist_ok=True)
    (out / analyze.GAME_LOG).write_text(json.dumps(events))


def test_single_scorecard_unaffected(tmp_path, monkeypatch):
    monkeypatch.setattr(analyze, "RUNS_DIR", tmp_path)
    card = _card("card-1", [_env("aaaa-1", 2.5), _env("bbbb-1", 0.0, levels_completed=0)])
    line = "--- FINAL SCORECARD REPORT ---\n" + json.dumps(card, indent=2)
    _write_log(tmp_path, "single", [line])

    result = analyze.scorecard("single")
    assert result["card_id"] == "card-1"
    assert len(result["environments"]) == 2
    assert result["score"] == 1.25


def test_triplicated_identical_scorecard_dedupes_to_one(tmp_path, monkeypatch):
    # Mirrors the real, pre-existing behaviour: every log line appears 3x
    # (25 interleaved games, non-adjacent repeats) -- the same scorecard,
    # not three different ones.
    monkeypatch.setattr(analyze, "RUNS_DIR", tmp_path)
    card = _card("card-1", [_env("aaaa-1", 4.0)])
    blob = "--- FINAL SCORECARD REPORT ---\n" + json.dumps(card, indent=2)
    _write_log(tmp_path, "triplicated", [blob, blob, blob])

    result = analyze.scorecard("triplicated")
    assert len(result["environments"]) == 1
    assert result["score"] == 4.0


def test_two_distinct_wave_scorecards_are_merged_not_dropped(tmp_path, monkeypatch):
    monkeypatch.setattr(analyze, "RUNS_DIR", tmp_path)
    wave1 = _card("card-wave1", [_env("aaaa-1", 4.0), _env("bbbb-1", 2.0)])
    wave2 = _card("card-wave2", [_env("cccc-1", 0.0, levels_completed=0), _env("dddd-1", 6.0)])
    lines = [
        "--- FINAL SCORECARD REPORT ---\n" + json.dumps(wave1, indent=2),
        "--- FINAL SCORECARD REPORT ---\n" + json.dumps(wave2, indent=2),
    ]
    _write_log(tmp_path, "two_waves", lines)

    result = analyze.scorecard("two_waves")
    assert len(result["environments"]) == 4
    ids = {e["id"] for e in result["environments"]}
    assert ids == {"aaaa-1", "bbbb-1", "cccc-1", "dddd-1"}
    # mean of 4.0, 2.0, 0.0, 6.0
    assert result["score"] == 3.0
    assert result["total_actions"] == 40
    assert result["total_environments"] == 4
    assert result["total_environments_completed"] == 3


def test_missing_log_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(analyze, "RUNS_DIR", tmp_path)
    assert analyze.scorecard("does-not-exist") is None


def test_no_marker_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(analyze, "RUNS_DIR", tmp_path)
    _write_log(tmp_path, "no_marker", ["nothing interesting here"])
    assert analyze.scorecard("no_marker") is None
