"""What aggregate score is reachable, and by which lever.

Answers "is this optimisation worth doing" BEFORE running it, by
reimplementing the competition scorer over an existing scorecard and
asking what the aggregate would be under hypothetical play.

The formula is not ours; it is read off the vendored engine
(`arc_agi/scorecard.py`, `EnvironmentScoreCalculator.add_level` and
`.to_score`):

    per level    min(115, (baseline / actions) ** 2 * 100) if COMPLETED else 0
    per game     sum(level_score * i) / sum(i)   over ALL levels the game defines
    aggregate    plain mean over the environments

Two consequences that are easy to get wrong by reasoning from memory:

  * Levels the agent never reached still sit in the denominator, so a game
    with 9 levels can score at most 100 and a level-1-only clear on it is
    worth 1/45 of that.
  * Weight is the level index, so level 1 is the *cheapest* level to hold
    and the deep levels carry the score.

Usage:
    python scripts/score_ceiling.py [path/to/score.log]

Validation: the "recomputed current" line must match the log's own
printed aggregate. If it does not, the parse is wrong and every other
number here is meaningless -- the script says so rather than printing
plausible garbage.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

DEFAULT_LOG = Path("kaggle_runs/ab/inert3/score.log")

# gid  score  runs  levels  <actions/baseline pairs...>
ROW = re.compile(r"^([a-z0-9]{4})\s+([\d.]+)\s+(\d+)\s+(\d+)\s+(.*)$")


def load(path: Path):
    games = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = ROW.match(line.strip())
        if not m:
            continue
        gid, printed, _runs, levels, rest = m.groups()
        pairs = [
            tuple(int(x) for x in tok.split("/"))
            for tok in rest.split()
            if "/" in tok
        ]
        if pairs:
            # A non-zero action count does NOT mean the level cleared: the
            # last non-zero entry is normally the level the agent died on.
            # `levels_completed` is the only authority.
            games.append((gid, float(printed), pairs, int(levels)))
    return games


def aggregate(games, actions_for) -> float:
    """actions_for(level, actions, baseline, n_cleared) -> actions | None."""
    total = 0.0
    for _gid, _printed, pairs, ncleared in games:
        num = den = 0.0
        for i, (act, base) in enumerate(pairs, start=1):
            den += i
            a = actions_for(i, act, base, ncleared)
            if a is None or a <= 0:
                continue
            num += min(115.0, (base / a) ** 2 * 100) * i
        total += num / den if den else 0.0
    return total / len(games) if games else 0.0


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_LOG
    if not path.exists():
        print(f"no such scorecard log: {path}")
        return 2
    games = load(path)
    if not games:
        print(f"parsed no game rows out of {path}")
        return 2

    now = aggregate(games, lambda i, a, b, n: a if i <= n else None)
    printed = sum(p for _, p, _, _ in games) / len(games)
    print(f"{path}   ({len(games)} environments, "
          f"{sum(1 for _, _, _, n in games if n > 0)} with a clear)\n")
    print(f"recomputed current : {now:.4f}")
    print(f"printed  current   : {printed:.4f}")
    if abs(now - printed) > 0.005:
        print("\nMISMATCH -- the parse is wrong, ignore everything below.")
        return 1
    print()

    print("what each lever buys, played to its limit:")
    print(f"{aggregate(games, lambda i,a,b,n: a/2 if i<=n else None):8.3f}  "
          "EFFICIENCY: halve every clear we already get")
    print(f"{aggregate(games, lambda i,a,b,n: b if i<=n else None):8.3f}  "
          "EFFICIENCY, PERFECT: every clear at human baseline (unbeatable)")
    print(f"{aggregate(games, lambda i,a,b,n: b*14 if i<=3 else None):8.3f}  "
          "DEPTH ONLY: levels 1-3 everywhere at 14x baseline (slow-game pace)")
    print(f"{aggregate(games, lambda i,a,b,n: b):8.3f}  "
          "EVERY level of EVERY game at baseline (theoretical max)\n")

    mults = (1, 2, 3, 5, 10)
    print("depth x efficiency (x = multiple of human baseline, all games):\n")
    print(f"{'levels':>10} | " + " | ".join(f"{m:>6}x" for m in mults))
    print("-" * (13 + 9 * len(mults)))
    for depth in (1, 2, 3, 4, 6):
        cells = [
            f"{aggregate(games, lambda i,a,b,n,d=depth,k=m: b*k if i<=d else None):7.2f}"
            for m in mults
        ]
        print(f"{'1-' + str(depth):>10} | " + " | ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
