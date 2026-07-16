"""Your ARC-AGI-3 agent. This is the *only* file that gets shipped to Kaggle.

CONTRACT (verified against the official `arcprize/ARC-AGI-3-Kaggle-Starter`
and `arcprize/ARC-AGI-3-Agents` repos -- MIT licensed, cloned and inspected
directly rather than guessed at):
    * When you build your actual submission, this file's raw text gets
      spliced verbatim into the Kaggle notebook by the official starter
      kit's `scripts/build_notebook.py` -- sibling files in THIS repo
      (hybrid_lib/*) are NOT carried along automatically. `hybrid_lib`
      must be installed as a wheel (offline, from a Kaggle Dataset) in
      the notebook's install cell, the same way the starter kit already
      installs the `arc-agi` package itself. See README.md "Two-repo
      workflow" for the exact steps -- this file assumes `hybrid_lib` is
      already importable.
    * Must subclass `agents.agent.Agent` (from the vendored
      ARC-AGI-3-Agents framework, MIT licensed).
    * Class must be named `MyAgent`.
    * Must implement `is_done(frames, latest_frame) -> bool` and
      `choose_action(frames, latest_frame) -> GameAction`.
    * `frames`/`latest_frame` are `arcengine.FrameData` -- note `.frame`
      is `list[list[list[int]]]`, i.e. a *list of 2D grids* per frame,
      not a single grid. `frames` is the SAME growing list as
      `Agent.frames`, and `latest_frame is frames[-1]` -- so on each
      call, `frames[-2]` (if it exists) is the frame from BEFORE the
      previous action, and `frames[-1]` is the frame that action
      produced. We use that to record the previous step's transition
      before deciding the next one.
    * `GameAction` is an Enum: RESET, ACTION1..ACTION5, ACTION7 are
      "simple" (no payload); ACTION6 is "complex" and needs
      `.set_data({"x": int, "y": int})` with x, y in [0, 63].
    * CONCURRENCY: verified in `agents/swarm.py` -- the framework
      instantiates one `MyAgent` PER GAME and runs all of them
      concurrently in separate threads within one process. So
      `__init__`/`choose_action` for *different games* can run in
      parallel. All shared state (the model, resource budget, session
      stats) lives in one process-wide `HybridDecisionCore` singleton
      (`hybrid_lib.decision_core.get_shared_core`), which is
      thread-safe; only this file's per-instance attributes are per-game.
"""

from __future__ import annotations

import logging
from typing import Any

from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent

from hybrid_lib.decision_core import AgentConfig, get_shared_core
from hybrid_lib.state import MultiGrid

logger = logging.getLogger()

# ----------------------------------------------------------------------
# EDIT ME before submitting: point at your offline model dataset and flip
# use_vllm on. Leave use_vllm=False (the default) to submit a GPU-free
# baseline that always uses the deterministic FallbackSolver -- a good
# first submission to prove the end-to-end pipeline works before betting
# on a model.
# ----------------------------------------------------------------------
CONFIG = AgentConfig(
    model_dir=None,          # e.g. "/kaggle/input/your-model-dataset/model"
    use_vllm=False,
    confidence_threshold=0.55,
)

# arcengine.GameAction.ACTION6 is the only action needing (x, y) payload
# data as of the arcengine version this was verified against (0.9.3) --
# re-check `GameAction.all_complex()` if you're on a newer arc-agi release.
_COMPLEX_ACTION_NAMES = frozenset(a.name for a in GameAction.all_complex())


def _to_multigrid(frame: FrameData) -> MultiGrid:
    return frame.frame


class MyAgent(Agent):
    """Hybrid LLM + deterministic-fallback agent. See module docstring."""

    MAX_ACTIONS = CONFIG.max_steps_per_game

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # self.game_id is set by Agent.__init__ above. get_shared_core()
        # is a process-wide singleton -- see this module's docstring on
        # concurrency. Only the first call across all game threads
        # actually constructs it (and loads the model, if enabled).
        self._core = get_shared_core(CONFIG)
        self._core.start_game(self.game_id)
        self._last_action: tuple[str, dict | None] | None = None

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        # Stop once we win. Don't stop on GAME_OVER -- RESET and retry
        # (matches the official sample's behavior). Also stop early if
        # this game's own wall-clock budget is exhausted, so a stuck game
        # doesn't sit occupying a thread/GPU slot for the rest of the run.
        if latest_frame.state is GameState.WIN:
            self._core.end_game(self.game_id, solved=True)
            return True
        if self._core.resources.game_time_exhausted(self.game_id):
            logger.info("game=%s stopping: per-game time budget exhausted", self.game_id)
            self._core.end_game(self.game_id, solved=False)
            return True
        return False

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        # Record the outcome of the PREVIOUS action, now visible as the
        # (frames[-2] -> frames[-1]) transition, before deciding the next
        # one. On the very first call frames has only the framework's
        # initial placeholder entry, so there's nothing to record yet.
        if self._last_action is not None and len(frames) >= 2:
            action_name, action_args = self._last_action
            self._core.record(
                self.game_id,
                _to_multigrid(frames[-2]),
                action_name,
                action_args,
                _to_multigrid(latest_frame),
                game_state_after=latest_frame.state.value,
                levels_completed_after=latest_frame.levels_completed,
            )
        self._last_action = None

        # First call or after a death -> reset the level (matches the
        # official sample's baseline behavior).
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            action = GameAction.RESET
            action.reasoning = {"why": "reset on NOT_PLAYED/GAME_OVER"}
            self._last_action = (action.name, None)
            return action

        action_space = [GameAction.from_id(i).name for i in latest_frame.available_actions] or [
            GameAction.RESET.name
        ]

        decision = self._core.choose(self.game_id, action_space, _COMPLEX_ACTION_NAMES)

        action = GameAction.from_name(decision.action)
        if action.is_complex():
            action.set_data(decision.action_args or {"x": 32, "y": 32})
        action.reasoning = {"source": decision.reasoning}

        self._last_action = (decision.action, decision.action_args)
        return action
