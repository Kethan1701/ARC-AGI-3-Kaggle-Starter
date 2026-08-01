"""What a result is actually worth, before spending a run to get it.

`score_competition.py` tells you what happened. This tells you what it was
worth, which turned out to be a different question and the more important
one: three GPU runs were spent chasing "can it clear a level", and clearing
ls20 level 1 at our search's 526 actions is worth 0.002 of final score.

The arithmetic is `arc_agi/scorecard.py`, restated here so it can be asked
questions offline:

    level_score = min((baseline / actions) ** 2 * 100, 115)
    env_score   = sum(level_score[i] * i for i in levels) / T
    T           = N(N+1)/2 over EVERY level in the game, reached or not
    env_score   = min(env_score, sum(i for i cleared) / T * 100)
    final       = mean(env_score) over the 25 games

Two consequences the code makes easy to miss:

  * That last clamp means beating the baseline is worth nothing. A level
    cleared in half the intended actions scores exactly what one cleared in
    the intended number scores. Aim at baseline, not past it.
  * The denominator counts levels you never reach, and the numerator weights
    by level index. Level 1 of every game, cleared perfectly, is 3.52. The
    points are deeper in, where the baselines are also larger and there is
    more room to be imprecise.

Usage:

    python scripts/score_math.py                 # the reference table
    python scripts/score_math.py --game ls20     # one game, level by level
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENTS_DIR = Path(
    os.environ.get("ARC_ENVIRONMENTS_DIR", str(ROOT / "environment_files"))
)


def load_baselines() -> dict[str, list[int]]:
    """Per-game human baseline actions, from the competition's own files."""
    games: dict[str, list[int]] = {}
    for path in sorted(ENVIRONMENTS_DIR.glob("*/*/metadata.json")):
        game = path.relative_to(ENVIRONMENTS_DIR).parts[0]
        data = json.loads(path.read_text())
        baselines = data.get("baseline_actions") or []
        if baselines:
            games[game] = baselines
    return games


def level_score(baseline: int, actions: int) -> float:
    """One level, scored exactly as the scorecard scores it."""
    if actions <= 0:
        return 0.0
    return min((baseline / actions) ** 2 * 100, 115.0)


def env_score(baselines: list[int], actions: dict[int, int]) -> float:
    """One game. `actions` maps 1-based level index to actions taken.

    A level absent from `actions` was not cleared and scores zero, but it
    still sits in the denominator -- which is the part that makes clearing
    one level of a nine-level game worth so little.
    """
    n = len(baselines)
    total_weight = n * (n + 1) // 2
    if not total_weight:
        return 0.0
    earned = 0.0
    cleared_weight = 0
    for index in range(1, n + 1):
        taken = actions.get(index)
        if taken is None:
            continue
        earned += level_score(baselines[index - 1], taken) * index
        cleared_weight += index
    return min(earned / total_weight, cleared_weight / total_weight * 100)


def final_score(results: dict[str, dict[int, int]]) -> float:
    """Mean over all 25 games, which is what the leaderboard shows."""
    games = load_baselines()
    if not games:
        return 0.0
    return sum(
        env_score(baselines, results.get(game, {}))
        for game, baselines in games.items()
    ) / len(games)


def _uniform(depth: int, multiple: float, skip_first: bool = False) -> float:
    """Every game cleared to `depth`, each level at `multiple` x baseline."""
    games = load_baselines()
    results: dict[str, dict[int, int]] = {}
    for game, baselines in games.items():
        taken: dict[int, int] = {}
        for index in range(1, min(depth, len(baselines)) + 1):
            if skip_first and index == 1:
                # Cleared, but slowly enough to score ~0: the tutorial.
                taken[index] = baselines[0] * 100
            else:
                taken[index] = max(1, round(baselines[index - 1] * multiple))
        results[game] = taken
    return final_score(results)


def reference_table() -> str:
    lines = [
        "Final score if EVERY game is cleared to the given depth,",
        "with each level taking the given multiple of its baseline:",
        "",
        f"  {'depth':>7}" + "".join(f"{m:>8}x" for m in (1, 2, 3, 5, 10)),
    ]
    for depth in (1, 2, 3, 4):
        row = "".join(f"{_uniform(depth, m):>9.2f}" for m in (1, 2, 3, 5, 10))
        lines.append(f"  {depth:>7}{row}")
    lines += [
        "",
        "Same, but treating level 1 as a tutorial (cleared, scored ~0):",
        "",
        f"  {'depth':>7}" + "".join(f"{m:>8}x" for m in (1, 2, 3, 5)),
    ]
    for depth in (2, 3, 4, 5):
        row = "".join(
            f"{_uniform(depth, m, skip_first=True):>9.2f}" for m in (1, 2, 3, 5)
        )
        lines.append(f"  {depth:>7}{row}")
    lines += [
        "",
        "Target is 1.50. Reading these: clearing level 1 everywhere is not",
        "a route to it at any efficiency, and depth is more forgiving than",
        "precision -- later levels weigh more AND allow more actions.",
    ]
    return "\n".join(lines)


def per_game(game: str) -> str:
    games = load_baselines()
    if game not in games:
        return f"unknown game {game!r}; known: {', '.join(sorted(games))}"
    baselines = games[game]
    n = len(baselines)
    total_weight = n * (n + 1) // 2
    lines = [
        f"{game}: {n} levels, weight denominator {total_weight}",
        "",
        f"  {'level':>5} {'baseline':>9} {'weight':>7} "
        f"{'final if cleared at 1x':>23} {'at 3x':>8}",
    ]
    for index, baseline in enumerate(baselines, start=1):
        at_one = 100 * index / total_weight / len(games)
        lines.append(
            f"  {index:>5} {baseline:>9} {index:>7} "
            f"{at_one:>23.3f} {at_one / 9:>8.3f}"
        )
    lines += [
        "",
        "The 'final if cleared' columns are this level's contribution to the",
        "whole leaderboard score, assuming the levels before it were cleared.",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", help="show one game level by level")
    args = parser.parse_args()

    if not load_baselines():
        raise SystemExit(
            f"no baselines under {ENVIRONMENTS_DIR}; set ARC_ENVIRONMENTS_DIR"
        )
    print(per_game(args.game) if args.game else reference_table())


if __name__ == "__main__":
    main()
