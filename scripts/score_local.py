"""Score `agent/my_agent.py` locally with the REAL competition metric.

`play_local.py` reports levels completed, which is not what the
leaderboard measures. The metric is efficiency and it is quadratic:

    level_score = min((baseline_actions / actions_taken) ** 2 * 100, 115)

and a game's score is the best of its runs, so this also prints the
per-run breakdown -- without it there is no way to tell whether the
exploit/replay path is actually banking cheap runs.

Usage:
    .venv/bin/python scripts/score_local.py --game ls20 --max-steps 20000
    .venv/bin/python scripts/score_local.py --max-steps 5000        # all
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VENDOR = ROOT / "vendor" / "ARC-AGI-3-Agents"
if not VENDOR.exists():
    raise SystemExit(f"Framework not found at {VENDOR}. Run `make setup` first.")
sys.path.insert(0, str(VENDOR))

import arc_agi
from arc_agi import OperationMode


def load_my_agent_class():
    spec = importlib.util.spec_from_file_location(
        "user_agent_module", ROOT / "agent" / "my_agent.py"
    )
    if spec is None or spec.loader is None:
        raise SystemExit("Could not load agent/my_agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MyAgent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--game", default=None, help="Comma-separated ids; default all.")
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--quiet", action="store_true", help="Only the summary table.")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.CRITICAL if args.quiet else logging.INFO, format="%(message)s"
    )

    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    all_envs = arc.get_environments()

    if args.game:
        wanted = {g.strip().split("-")[0] for g in args.game.split(",")}
        game_ids = [e.game_id.split("-")[0] for e in all_envs
                    if e.game_id.split("-")[0] in wanted]
    else:
        game_ids = [e.game_id.split("-")[0] for e in all_envs]

    MyAgentCls = load_my_agent_class()
    MyAgentCls.MAX_ACTIONS = args.max_steps

    for i, game_id in enumerate(game_ids, 1):
        print(f"=== [{i}/{len(game_ids)}] {game_id} ===", flush=True)
        env = arc.make(game_id)
        if env is None:
            print(f"  could not create env for {game_id!r}, skipping")
            continue
        agent = MyAgentCls(
            card_id="local-dev", game_id=game_id,
            agent_name=f"MyAgent.score.{game_id}", ROOT_URL="http://localhost",
            record=False, arc_env=env, tags=["local-dev"],
        )
        agent.main()

    sc = arc.get_scorecard()
    print("\n=================== SCORECARD ===================")
    print(f"{'game':7} {'score':>7} {'runs':>5} {'lvls':>5}   per-level actions vs baseline")
    total = 0.0
    for env_score in sc.environments:
        gid = env_score.id.split("-")[0]
        best = max(env_score.runs, key=lambda r: r.score)
        pairs = ""
        if best.level_actions and best.level_baseline_actions:
            pairs = " ".join(
                f"{a}/{b}" for a, b in zip(best.level_actions,
                                           best.level_baseline_actions)
            )
        print(f"{gid:7} {env_score.score:7.2f} {len(env_score.runs):5} "
              f"{best.levels_completed:5}   {pairs}")
        total += env_score.score

    n = len(sc.environments)
    print("-" * 49)
    print(f"AGGREGATE SCORE: {total / n if n else 0.0:.4f}   over {n} environment(s)")
    print("(this is the leaderboard number)")


if __name__ == "__main__":
    main()
