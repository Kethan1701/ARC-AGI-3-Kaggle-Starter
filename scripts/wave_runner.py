"""Splits a game list into groups of a given size.

2026-08-06: this used to be a full wave-based concurrency reducer -- split
25 games into N sequential batches, play each batch through its own
`main.py --agent myagent --game ...` subprocess, so every game still ran
under the real Swarm orchestration but with less than 25-way concurrency
(motivated by log analysis showing 73-82% of vLLM server samples sitting in
the 80-99% GPU-KV-cache-usage band at full concurrency; see
arc-agi-3-kaggle-gpu-infra memory).

It was built, wired into both the real run cell and the Phase A rehearsal,
and reverted the same day: `arc_agi.api`'s `ScorecardManager` rejects a
second `open_scorecard()` call with a 409 ("cannot open multiple scorecards
in competition mode"), and the flag it checks (`scorecard_openned`) is never
reset back to False anywhere in that file -- competition mode allows exactly
one scorecard per server process, for the life of that process, full stop.
Confirmed two ways: reading arc_agi/api.py directly, and reproducing the 409
locally (`score_competition.py --game ar25,bp35,cd82,cn04 --wave-size 2`
opened wave 1's scorecard fine, closed it fine, then wave 2's open failed).
No amount of client-side orchestration -- separate subprocesses, separate
Swarm() calls, whatever -- routes around a server-side flag that's checked
once and never cleared.

A real reduced-concurrency design would need ONE scorecard opened once and
kept open across every wave (`arc.make(game_id, scorecard_id=...)` per wave,
close only at the very end), which means bypassing the official `Swarm`
class entirely and hand-rolling its thread-per-game/join loop -- reusing
`main.py`/`Swarm` unmodified, which was the whole point of the subprocess
approach, is not compatible with that. Nobody has attempted that
reimplementation; it would be materially riskier (more custom code standing
in for tested framework behaviour) and deserves its own decision before
being tried.

`split_into_waves` itself has no opinion about scorecards -- it is pure,
tested (tests/test_wave_runner.py), and still correct. Kept here in case a
genuinely different design (the single-shared-scorecard reimplementation
above, or something else) ever needs it again.
"""
from __future__ import annotations


def split_into_waves(games: list[str], wave_size: int) -> list[list[str]]:
    """Split games into sequential waves of at most wave_size each.

    wave_size <= 0, or >= len(games), means "one wave, everything" -- the
    default, and behaviourally identical to no waving at all. Order is
    preserved so the first wave is always the first `wave_size` games in
    the input order, matching how a caller would read "wave 1" in a log.
    """
    if not games:
        return []
    if wave_size <= 0 or wave_size >= len(games):
        return [list(games)]
    return [games[i : i + wave_size] for i in range(0, len(games), wave_size)]
