"""Sketch of the ARC-AGI-3 turn loop, driven by the per-game memory.

WHERE THIS LIVES AND WHY
------------------------
`scripts/`, not `agent/`. `scripts/build_notebook.py` splices exactly one
hardcoded path -- `agent/my_agent.py` -- into the submission notebook, so
this file never reaches the Kaggle kernel. That is deliberate: it is a
scaffold for driving the real ARC-AGI-3 client locally, not competition
runtime. Anything here that turns out to be worth submitting has to be
moved *into* `agent/my_agent.py` to deploy at all.

WHAT IT SKETCHES
----------------
The loop the memory is designed around:

    for each level:
        cached = mem.solved_path(level, frame)
        if cached is not None:      -> replay it, zero exploration
        else:                       -> explore with untried_actions()
                                       + predict_frame() look-ahead
        on a win:                   -> mem.mark_solved(level, start, path)

Scoring is `min(1, human_actions / ai_actions) ** 2` per level, so every
committed action is squared against you and internal reasoning is free.
That asymmetry is the whole design: never re-explore a solved level, and
prefer a free simulated step to a paid real one.

STATUS: the ArcClient adapters below are STUBS. Every one raises
NotImplementedError and is marked TODO. Drop the real SDK client in
behind that interface and the loop should run unchanged.

    python scripts/agent_loop.py --demo     # run the loop on a toy env
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, Hashable, Iterable, Optional

ROOT = Path(__file__).resolve().parents[1]

Frame = Any
Action = Hashable

logger = logging.getLogger("agent_loop")


# --------------------------------------------------------------------- #
# MemoryStore lives in agent/my_agent.py (single-file deploy target), and
# `agent/` is not a package, so it is loaded by path -- the same way
# tests/test_game_memory.py does it.
# --------------------------------------------------------------------- #
def _load_agent_module():
    vendor = ROOT / "vendor" / "ARC-AGI-3-Agents"
    if vendor.exists():
        sys.path.insert(0, str(vendor))
    spec = importlib.util.spec_from_file_location(
        "my_agent_for_loop", ROOT / "agent" / "my_agent.py"
    )
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError("could not load agent/my_agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_agent = _load_agent_module()
MemoryStore = _agent.MemoryStore
GameMemory = _agent.GameMemory
fingerprint = _agent.fingerprint
_tok = _agent._tok
_untok = _agent._untok

# No level may spend more than this exploring before the loop gives up on
# it. A sketch that can spin forever is worse than one that stops: without
# it, an agent that has tried everything reachable just walks between
# known states paying squared-cost actions for nothing.
MAX_ACTIONS_PER_LEVEL = 2000


# --------------------------------------------------------------------- #
# SDK adapter -- STUBS. This is the whole surface the loop needs.
# --------------------------------------------------------------------- #
class ArcClient:
    """Adapter over the real ARC-AGI-3 client.

    The loop below touches nothing else, so wiring the real SDK means
    filling in these six methods and nothing more. Keep them thin: no
    strategy in here, or the loop stops being the thing you can reason
    about.
    """

    def frame(self) -> Frame:
        """Current observation: the 64x64 grid.

        TODO: return the live frame. Any of numpy array / nested
        list-or-tuple / bytes works -- `fingerprint()` handles all three
        and the memory never stores the raw grid, only its hash.
        """
        raise NotImplementedError("TODO: return the live frame from the SDK")

    def action_set(self) -> list[Action]:
        """Legal actions from the current state.

        TODO: return the live action set. An action is any hashable token:
        "ACTION3", or a tuple like ("ACTION6", x, y) for clicks. It is
        persisted via repr(), so keep it repr-stable.
        """
        raise NotImplementedError("TODO: return the SDK's available actions")

    def level(self) -> int:
        """Index of the level currently being played.

        TODO: read the live level counter. It must go UP on a clear and
        DOWN (or back to 0) on a reset -- the loop detects both by
        comparing this across turns.
        """
        raise NotImplementedError("TODO: return the SDK's current level index")

    def step(self, action: Action) -> None:
        """Commit one action. This is the expensive call -- it is squared.

        TODO: submit the action and block until the new frame is available.
        """
        raise NotImplementedError("TODO: submit the action to the SDK")

    def done(self) -> bool:
        """True when the game is over (won, or out of budget).

        TODO: read the SDK's terminal state.
        """
        raise NotImplementedError("TODO: return the SDK's terminal flag")

    def died(self) -> bool:
        """True when the run just reset us backwards onto an earlier level.

        TODO: read death/reset from the SDK. This is the event the whole
        memory exists for: it is where a naive agent re-explores a level
        it has already solved and pays for it twice, squared.
        """
        raise NotImplementedError("TODO: return the SDK's death/reset flag")


# --------------------------------------------------------------------- #
# the loop
# --------------------------------------------------------------------- #
def _route_to_frontier(
    mem: GameMemory,
    frame: Frame,
    candidates: list[Action],
) -> Optional[Action]:
    """First step of the shortest known path to a state with something new.

    Breadth-first over the remembered transition graph. This is the step
    that keeps the loop from livelocking: once every action from the
    current state has been tried, walking to the *nearest* state that
    still has an untried action is the only move that buys information,
    and the whole walk is planned for free against memory.

    Returns None when no reachable state has anything left to try, which
    is the honest signal that exploration here is finished.
    """
    start = fingerprint(frame)
    width = len(candidates)

    # fp -> [(action_tok, next_fp)], built from the remembered edges.
    outgoing: dict[str, list[tuple[str, str]]] = {}
    for (fp, tok), nfp in mem.transitions.items():
        outgoing.setdefault(fp, []).append((tok, nfp))

    def has_untried(fp: str) -> bool:
        return len(mem.tried.get(fp, ())) < width

    seen = {start}
    # queue holds (fp, first action token taken out of `start`)
    queue: list[tuple[str, Optional[str]]] = [(start, None)]
    while queue:
        fp, first = queue.pop(0)
        if first is not None and has_untried(fp):
            return _untok(first)
        for tok, nfp in outgoing.get(fp, ()):
            if nfp not in seen:
                seen.add(nfp)
                queue.append((nfp, first if first is not None else tok))
    return None


def choose_action(
    mem: GameMemory,
    frame: Frame,
    candidates: Iterable[Action],
) -> Optional[Action]:
    """Pick the next action to commit, spending as little as possible.

    Order matters and encodes the cost model:

      1. an action never tried from this exact state -- the only kind that
         can teach us anything new;
      2. failing that, an action whose predicted outcome is a state we
         have never stood in -- free look-ahead, one paid step;
      3. failing that, walk to the nearest state that still has an untried
         action, routing over remembered transitions;
      4. failing that, None -- everything reachable is exhausted, and the
         caller must reset rather than keep paying.
    """
    candidates = list(candidates)
    if not candidates:
        return None

    untried = mem.untried_actions(frame, candidates)
    if untried:
        return untried[0]

    # Everything here has been tried. Use the free look-ahead to prefer a
    # step toward an unexplored state over one we know loops back.
    for action in candidates:
        predicted = mem.predict_frame(frame, action)
        if predicted is not None and not mem.is_known_state(predicted):
            return action

    return _route_to_frontier(mem, frame, candidates)


def play_level(client: ArcClient, mem: GameMemory) -> bool:
    """Play the current level to a clear. True if it was cleared.

    Replays from cache when the memory holds this level and the board
    matches; otherwise explores and banks the result on the way out.
    """
    level = client.level()
    start_frame = client.frame()

    # ---- cached: replay, zero exploration --------------------------- #
    cached = mem.solved_path(level, start_frame)
    if cached is not None:
        logger.info(
            "level %d: replaying %d cached actions (no exploration)",
            level, len(cached),
        )
        for action in cached:
            client.step(action)
            if client.done() or client.died():
                break
        cleared = client.level() > level
        if not cleared:
            # The cache said this board was the level start and it did not
            # pay out. Determinism should make that impossible, so it means
            # the recorded path is stale -- drop it rather than trust it again.
            logger.warning("level %d: cached replay did not clear; dropping it", level)
            mem.solved.pop(level, None)
        return cleared

    # ---- uncached: explore, then bank ------------------------------- #
    logger.info("level %d: no cached path, exploring", level)
    path: list[Action] = []
    while not client.done() and len(path) < MAX_ACTIONS_PER_LEVEL:
        frame = client.frame()
        action = choose_action(mem, frame, client.action_set())
        if action is None:
            # Nothing reachable has an untried action left. Continuing
            # would spend squared-cost actions to learn nothing.
            logger.warning(
                "level %d: exploration exhausted after %d actions", level, len(path)
            )
            return False

        client.step(action)
        mem.record(frame, action, client.frame())
        path.append(action)

        if client.died() or client.level() < level:
            # Reset threw us backwards; this attempt is over. What we
            # learned is kept -- the transition graph is per-state, not
            # per-attempt, so none of it has to be re-paid.
            logger.info("level %d: died after %d actions", level, len(path))
            return False

        if client.level() > level:
            mem.mark_solved(level, start_frame, path)
            logger.info("level %d: cleared in %d actions (banked)", level, len(path))
            return True

    return False


def play_game(
    client: ArcClient,
    game_id: str,
    store: MemoryStore,
    max_levels: int = 100,
) -> GameMemory:
    """Drive one game to completion, persisting memory as levels fall."""
    mem = store.game(game_id)
    attempts = 0
    # Deaths are expected and are exactly what the memory is for, so an
    # attempt that loses ground is not a failure -- but the budget is
    # counted in ATTEMPTS, not levels, so a game that dies forever still
    # terminates instead of spinning.
    budget = max_levels * 4
    while not client.done() and attempts < budget:
        before = client.level()
        cleared = play_level(client, mem)
        # Persist on every level boundary, not at the end: a crash or a
        # kernel timeout must not cost the levels already solved.
        store.save(game_id)
        attempts += 1

        if not cleared and client.level() == before:
            # No ground gained and no reset either: the level is stuck and
            # replaying this same state will stay stuck. Stop rather than
            # burn the action budget discovering that repeatedly.
            logger.warning("level %d: stuck, giving up on this game", before)
            break

    store.save(game_id)
    return mem


# --------------------------------------------------------------------- #
# demo: the same toy corridor the tests use, so the loop can be exercised
# end to end without the SDK.
# --------------------------------------------------------------------- #
class _CorridorClient(ArcClient):
    """A 1-D corridor standing in for the SDK, for `--demo` only."""

    GOAL = 4

    def __init__(self, die_after: Optional[int] = None) -> None:
        self.pos = 0
        self.level_idx = 0
        self.steps = 0
        self.die_after = die_after
        self._died = False

    def frame(self) -> Frame:
        # The level index is part of the observation. Without it every
        # level renders identically and the memory cannot tell "level 1
        # start" from "level 0 start" -- which is a real property of the
        # fingerprint scheme worth making visible in the demo.
        return (self.level_idx,) + tuple(1 if i == self.pos else 0 for i in range(5))

    def action_set(self) -> list[Action]:
        return ["LEFT", "RIGHT"]

    def level(self) -> int:
        return self.level_idx

    def step(self, action: Action) -> None:
        self.steps += 1
        self.pos = min(self.pos + 1, 4) if action == "RIGHT" else max(self.pos - 1, 0)
        if self.die_after is not None and self.steps == self.die_after:
            self._died = True
            self.pos = 0
            return
        if self.pos == self.GOAL:
            self.level_idx += 1
            self.pos = 0

    def done(self) -> bool:
        return self.level_idx >= 2

    def died(self) -> bool:
        was, self._died = self._died, False
        return was


def _demo() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "arc_memory")

        print("=== pass 1: cold memory, must explore ===")
        first = _CorridorClient()
        mem = play_game(first, "corridor", MemoryStore(path=path))
        explored = len(mem.transitions)
        print(f"    committed actions : {first.steps}")
        print(f"    levels banked     : {sorted(mem.solved)}")
        print(f"    transitions learnt: {explored}")

        # A fresh MemoryStore, as a later process would build it: this also
        # proves the JSON round-trip, not just the in-process cache.
        print("\n=== pass 2: same game, memory reloaded from disk ===")
        second = _CorridorClient()
        warm = play_game(second, "corridor", MemoryStore(path=path))
        print(f"    committed actions : {second.steps}")
        print(f"    transitions learnt: {len(warm.transitions) - explored} (new)")

        if len(warm.transitions) == explored:
            print("    -> replayed from cache with ZERO exploration. Leak plugged.")
        else:
            print("    -> WARNING: re-explored; the cache did not hold.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo", action="store_true",
        help="run the loop against the toy corridor env instead of the SDK",
    )
    args = parser.parse_args()
    if args.demo:
        _demo()
        return
    parser.error(
        "no SDK client wired yet -- fill in the ArcClient TODOs, then call "
        "play_game(YourClient(), game_id, MemoryStore(path=...)). "
        "Use --demo to exercise the loop meanwhile."
    )


if __name__ == "__main__":
    main()
