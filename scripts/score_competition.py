"""Score `agent/my_agent.py` the way *Kaggle* scores it.

`score_local.py` drives the environment wrapper directly. That bypasses
`arc_agi/api.py`, which is where the competition rules actually live, so it
measures a strictly more permissive game than the leaderboard plays. Two
submissions that measured 0.053 and 2.10 under `score_local.py` both scored
exactly 0.19 on Kaggle -- the difference never existed.

What `competition_mode=True` changes (both verified against the real API):

  1. One run per game, forever. `_get_or_create_environment` returns None when
     `scorecard.has_environment(game_id)`, so a game can never be re-played on
     the same scorecard.
  2. RESET at `_action_count == 0` is a no-op that still bills an action. That
     is exactly the condition `arcengine.base_game.handle_reset` uses to pick
     `full_reset()`, so the "double RESET opens a fresh scored run" trick is
     specifically defeated -- and costs you an action for nothing.

Net effect: exploration is not free. Every action spent searching is an action
charged against the level you are searching, and there is no second pass to
redeem it. Optimise first-pass efficiency, not replay.

This harness stands up the genuine Flask app from `arc_agi.server.create_app`
with `competition_mode=True`, serves it over real HTTP, and points the
framework's Swarm at it in ONLINE mode -- the same shape as the Kaggle gateway
sidecar.

Usage:
    .venv/bin/python scripts/score_competition.py --max-steps 5000
    .venv/bin/python scripts/score_competition.py --game ls20 --max-steps 20000
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import socket
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VENDOR = ROOT / "vendor" / "ARC-AGI-3-Agents"
if not VENDOR.exists():
    raise SystemExit(f"Framework not found at {VENDOR}. Run `make setup` first.")
sys.path.insert(0, str(VENDOR))

API_KEY = "local-competition-key"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def load_my_agent_class():
    spec = importlib.util.spec_from_file_location(
        "user_agent_module", ROOT / "agent" / "my_agent.py"
    )
    if spec is None or spec.loader is None:
        raise SystemExit("Could not load agent/my_agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MyAgent


def start_gateway(port: int):
    """Serve the real arc_agi API locally with competition rules enabled."""
    from werkzeug.serving import make_server

    from arc_agi import Arcade, OperationMode
    from arc_agi.server import create_app

    # The server owns the environments; it must NOT talk to the real API.
    arcade = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(ROOT / "environment_files"),
    )
    app, _api = create_app(arcade, competition_mode=True)

    server = make_server("127.0.0.1", port, app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, arcade


def wait_for_gateway(port: int, timeout: float = 30.0) -> None:
    import requests

    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/api/healthcheck"
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return
        except requests.exceptions.RequestException:
            time.sleep(0.2)
    raise SystemExit(f"Gateway did not come up on port {port}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--game", default=None, help="Comma-separated ids; default all.")
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--quiet", action="store_true", help="Only the summary table.")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.CRITICAL if args.quiet else logging.INFO, format="%(message)s"
    )
    # Werkzeug logs every single HTTP request; at ~10^5 actions that is noise
    # that drowns the report and slows the run down measurably.
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    port = free_port()
    server, _server_arcade = start_gateway(port)
    wait_for_gateway(port)

    root_url = f"http://127.0.0.1:{port}"

    # The framework builds its own Arcade from the environment, so the client
    # side has to be configured before `Swarm` is constructed.
    os.environ["OPERATION_MODE"] = "online"
    os.environ["ARC_BASE_URL"] = root_url + "/"
    os.environ["ARC_API_KEY"] = API_KEY
    os.environ["SCHEME"] = "http"
    os.environ["HOST"] = "127.0.0.1"
    os.environ["PORT"] = str(port)
    os.environ.setdefault("RECORDINGS_DIR", str(ROOT / "recordings"))

    import requests

    r = requests.get(
        f"{root_url}/api/games", headers={"X-API-Key": API_KEY}, timeout=10
    )
    r.raise_for_status()
    full_games = [g["game_id"] for g in r.json()]

    if args.game:
        wanted = tuple(g.strip() for g in args.game.split(","))
        games = [g for g in full_games if g.startswith(wanted)]
    else:
        games = full_games

    if not games:
        raise SystemExit(f"No games matched {args.game!r} (have {full_games})")

    import agents
    from agents.swarm import Swarm

    MyAgentCls = load_my_agent_class()
    MyAgentCls.MAX_ACTIONS = args.max_steps
    agents.AVAILABLE_AGENTS["myagent"] = MyAgentCls

    started = time.time()
    swarm = Swarm("myagent", root_url, games, tags=["local-competition"])
    scorecard = swarm.main()
    elapsed = time.time() - started

    server.shutdown()

    if scorecard is None:
        raise SystemExit("No scorecard returned; the gateway run failed.")

    total_actions = 0
    print("\n================== COMPETITION SCORECARD ==================")
    print(f"{'game':7} {'score':>7} {'runs':>5} {'lvls':>5}   per-level actions/baseline")
    total = 0.0
    for env_score in scorecard.environments:
        gid = env_score.id.split("-")[0]
        best = max(env_score.runs, key=lambda r: r.score)
        pairs = ""
        if best.level_actions and best.level_baseline_actions:
            pairs = " ".join(
                f"{a}/{b}"
                for a, b in zip(best.level_actions, best.level_baseline_actions)
            )
        total_actions += sum(run.actions for run in env_score.runs)
        print(
            f"{gid:7} {env_score.score:7.2f} {len(env_score.runs):5} "
            f"{best.levels_completed:5}   {pairs}"
        )
        # `_calculate_score` has silent zeroing paths -- "size mismatch"
        # fires when a game records more level transitions than it has
        # levels, which repeated deaths can cause, and it zeroes the whole
        # game no matter how well it was played. Without surfacing the
        # message that is indistinguishable from simply playing badly.
        messages = {run.message for run in env_score.runs if run.message}
        for message in sorted(messages):
            print(f"{'':7} {'':>7} !! {message}")
        total += env_score.score

    n = len(scorecard.environments)
    print("-" * 59)
    print(f"AGGREGATE SCORE: {total / n if n else 0.0:.4f}   over {n} environment(s)")
    print(f"actions: {total_actions}   wall clock: {elapsed:.1f}s "
          f"({total_actions / elapsed if elapsed else 0:.0f} actions/s)")
    print("(competition rules: one run per game, RESET-to-new-run defeated)")


if __name__ == "__main__":
    main()
