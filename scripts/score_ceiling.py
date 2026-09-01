"""Price an optimisation against OUR measured scorecard, before building it.

`score_math.py` is the canonical model of the competition scorer and
answers "what would uniform play at depth D and multiple M be worth".
This script answers the neighbouring question: given the run we actually
just measured, what would the aggregate be if a specific lever worked?

It therefore imports `score_math` rather than restating the formula --
notably `env_score`, which applies the final clamp
(`min(earned/T, cleared_weight/T * 100)`) that makes beating the baseline
worth exactly nothing. An earlier version of this file reimplemented the
arithmetic, omitted that clamp, and overstated the efficiency lever.

Usage:
    python scripts/score_ceiling.py [path/to/score.log]

Validation: it recomputes the log's own printed aggregate and refuses to
print anything else if the parse disagrees, because a mis-parsed
scorecard produces plausible numbers rather than obvious garbage.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from score_math import env_score  # noqa: E402

DEFAULT_LOG = Path("kaggle_runs/ab/inert3/score.log")

# gid  score  runs  levels  <actions/baseline pairs...>
ROW = re.compile(r"^([a-z0-9]{4})\s+([\d.]+)\s+(\d+)\s+(\d+)\s+(.*)$")


def load(path: Path):
    """-> [(gid, printed_score, baselines, actions_by_level, n_cleared)]"""
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
        if not pairs:
            continue
        # A non-zero action count does NOT mean the level cleared: the last
        # non-zero entry is normally the level the agent died on.
        # `levels_completed` is the only authority.
        games.append(
            (
                gid,
                float(printed),
                [b for _a, b in pairs],
                [a for a, _b in pairs],
                int(levels),
            )
        )
    return games


def aggregate(games, actions_for) -> float:
    """actions_for(level, actions, baseline, n_cleared) -> actions | None."""
    total = 0.0
    for _gid, _printed, baselines, acts, ncleared in games:
        taken = {}
        for i in range(1, len(baselines) + 1):
            a = actions_for(i, acts[i - 1], baselines[i - 1], ncleared)
            if a is not None and a > 0:
                taken[i] = a
        total += env_score(baselines, taken)
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
    printed = sum(p for _, p, _, _, _ in games) / len(games)
    print(f"{path}   ({len(games)} environments, "
          f"{sum(1 for *_x, n in games if n > 0)} with a clear)\n")
    print(f"recomputed current : {now:.4f}")
    print(f"printed  current   : {printed:.4f}")
    if abs(now - printed) > 0.005:
        print("\nMISMATCH -- the parse is wrong, ignore everything below.")
        return 1
    print()

    print("what each lever buys on THIS run, played to its limit:")
    for label, fn in (
        ("EFFICIENCY: halve every clear we already get",
         lambda i, a, b, n: a / 2 if i <= n else None),
        ("EFFICIENCY, PERFECT: every clear at human baseline (unbeatable)",
         lambda i, a, b, n: b if i <= n else None),
        ("DEPTH ONLY: levels 1-3 everywhere at 14x baseline (slow pace)",
         lambda i, a, b, n: b * 14 if i <= 3 else None),
        ("EVERY level of EVERY game at baseline (theoretical max)",
         lambda i, a, b, n: b),
    ):
        print(f"{aggregate(games, fn):8.3f}  {label}")

    mults = (1, 2, 3, 5, 10)
    print("\ndepth x efficiency on this game set "
          "(x = multiple of human baseline):\n")
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
