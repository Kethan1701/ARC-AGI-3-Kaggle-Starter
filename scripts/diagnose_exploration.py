"""Why does a game never clear its first level? Budget, or comprehension?

Run the games with periodic budget reports, then point this at the log:

    ARC_BUDGET_EVERY=100 .venv/bin/python scripts/score_competition.py \
        --game ar25,bp35,... --max-steps 1200 > run.log 2>&1
    python scripts/diagnose_exploration.py run.log [baseline_scorecard.log]

`_report_budget` logs `nodes=` (cumulative distinct states) and a
per-window mix of what the actions were spent on. The discovery CURVE is
the diagnostic, and it has to be read against the human baseline or it
misleads:

  * "still discovering new states at the cap" sounds like a budget
    problem and is usually the opposite. A human clears ar25 level 1 in
    32 actions. If the agent visits 967 distinct states across 1200
    actions and still loses, the solution is shallow and the agent is not
    short of budget -- it is short of any notion of which state is closer
    to winning. Linear node growth means novelty is abundant and
    therefore worthless as a search signal.

  * The mirror failure is a game where the agent sees FEWER states than
    the human's whole solution: it cannot even produce the variety the
    task needs, usually visible as near-zero liveness (hardly any action
    changes the frame).

So the readings are:

    WANDERING   novelty never runs out; more budget buys more states
    EXHAUSTED   reachable space explored, still cannot win
    TRAPPED     fewer states seen than a human's entire solution
    slow crawl  in between
"""

from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

BUDGET = re.compile(
    # the agent logs self.game_id, which carries a hash: "ar25-0c556536"
    r"([a-z0-9]{4})(?:-[0-9a-f]+)?: still on level (\d+) after (\d+) actions "
    r"\(breadth=(\d+), nodes=(\d+)\) \| (.*?) "
    r"\|\| clicks=(\d+) \([\d.]+% of spend, ([\d.]+)% moved\) "
    r"presses=(\d+) \(([\d.]+)% moved\)"
)
ROW = re.compile(r"^([a-z0-9]{4})\s+([\d.]+)\s+(\d+)\s+(\d+)\s+(.*)$")


def load_baselines(path: Path) -> dict[str, int]:
    """Level-1 human baseline per game, off any scorecard log."""
    out: dict[str, int] = {}
    if not path or not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = ROW.match(line.strip())
        if m:
            pairs = [t for t in m.group(5).split() if "/" in t]
            if pairs:
                out[m.group(1)] = int(pairs[0].split("/")[1])
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    run = Path(sys.argv[1])
    if not run.exists():
        print(f"no such run log: {run}")
        return 2
    baseline = load_baselines(
        Path(sys.argv[2]) if len(sys.argv) > 2
        else Path("kaggle_runs/ab/inert3/score.log")
    )

    curve: dict[str, list[tuple[int, int]]] = defaultdict(list)
    mix: dict[str, Counter] = defaultdict(Counter)
    live: dict[str, list[tuple[int, float, int, float]]] = defaultdict(list)

    for raw in run.read_text(encoding="utf-8", errors="replace").splitlines():
        m = BUDGET.search(raw)
        if not m:
            continue
        gid, _lv, acts, _br, nodes, why, clicks, cmv, presses, pmv = m.groups()
        curve[gid].append((int(acts), int(nodes)))
        for tok in why.split():
            if "=" in tok:
                k, v = tok.rsplit("=", 1)
                if v.isdigit():
                    mix[gid][k] += int(v)
        live[gid].append((int(clicks), float(cmv), int(presses), float(pmv)))

    if not curve:
        print(f"no budget lines in {run} -- was ARC_BUDGET_EVERY set?")
        return 1

    print("human clears level 1 in `base` actions; this run spent the cap "
          "and lost.\n")
    print(f"{'game':<6} {'base':>4} {'nodes':>6} {'/base':>6} {'last300':>8} "
          f"{'rate':>6} {'live%':>6} {'probe%':>7}  reading")
    print("-" * 92)

    tally: Counter = Counter()
    for gid in sorted(curve):
        pts = sorted(curve[gid])
        final = pts[-1][1]
        b = baseline.get(gid, 0)
        last300 = final - next(n for a, n in pts if a >= pts[-1][0] - 300)
        rate = final / pts[-1][0] if pts[-1][0] else 0.0

        cl = sum(c for c, _, _, _ in live[gid])
        pr = sum(p for _, _, p, _ in live[gid])
        moved = (sum(c * m for c, m, _, _ in live[gid])
                 + sum(p * m for _, _, p, m in live[gid])) / max(cl + pr, 1)
        probe = 100.0 * mix[gid]["probe-untried"] / max(sum(mix[gid].values()), 1)

        if last300 >= 100:
            reading = "WANDERING: novelty never runs out; no progress signal"
        elif b and final < b:
            reading = "TRAPPED: fewer states seen than a human's whole solution"
        elif last300 <= 10:
            reading = "EXHAUSTED: explored what it can reach, still cannot win"
        else:
            reading = "slow crawl"
        tally[reading.split(":")[0]] += 1

        print(f"{gid:<6} {b:>4} {final:>6} {(final/b if b else 0):>5.1f}x "
              f"{last300:>8} {rate:>6.2f} {moved:>5.1f}% {probe:>6.1f}%  "
              f"{reading}")

    print("\ntally:")
    for reading, n in tally.most_common():
        print(f"  {n:>2}  {reading}")

    print("\naction mix, summed over all games:")
    tot: Counter = Counter()
    for g in mix:
        tot.update(mix[g])
    grand = sum(tot.values()) or 1
    for why, n in tot.most_common():
        print(f"  {n:>7}  {100.0 * n / grand:5.1f}%  {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
