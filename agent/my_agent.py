"""ARC-AGI-3 agent: object-aware, frontier-directed state-graph explorer.

This is the *only* file shipped to Kaggle. `scripts/build_notebook.py`
splices its raw text into the submission notebook, so it must be
SELF-CONTAINED: standard library + `arcengine` + the vendored
`agents.agent.Agent` base class, nothing else. Do not add imports of
sibling modules or of any package that isn't already installed by the
notebook's first cell -- they will not exist at run time.

--------------------------------------------------------------------
CONTRACT (verified by reading the installed packages, not guessed)
--------------------------------------------------------------------
* `MyAgent` subclasses `agents.agent.Agent` and implements
  `is_done(frames, latest_frame)` and `choose_action(frames, latest_frame)`.
* `FrameData.frame` is `list[list[list[int]]]` -- a *list of 2D grids*.
  In practice the list has exactly one entry: the rendered 64x64 view,
  with cell values 0..15 (colors). Verified against ls20 and vc33.
* `FrameData.available_actions` is a `list[int]` of action ids that are
  live *for this game*, and it differs per game: ls20 exposes [1,2,3,4]
  (movement), vc33 exposes only [6] (click). RESET(0) is always usable
  but is never listed.
* `GameAction.ACTION6` is the only complex action; its payload is
  `ComplexAction(x, y)` with both in [0, 63]. The frame is indexed
  `grid[y][x]` (see `arcengine.interfaces.draw_sprite`, which slices
  `frame[start_y:end_y, start_x:end_x]`), so x is the COLUMN and y is
  the ROW.
* `FrameData.levels_completed` (0..`win_levels`) is the score signal.
  `GameState.WIN` means every level was cleared.
* CONCURRENCY: `agents/swarm.py` builds one `MyAgent` PER GAME and runs
  them all concurrently as threads in a single process. Every attribute
  on this class is therefore per-game and needs no locking. The one
  piece of genuinely shared state is the module-level deadline, which
  is read-only after import.

--------------------------------------------------------------------
STRATEGY
--------------------------------------------------------------------
ARC-AGI-3 games are deterministic puzzles with an unknown rule set. The
winning move is not to guess the rules but to *map the reachable state
space systematically* and never waste a step re-testing something
already known:

1. Every distinct 64x64 grid is a node. Every (node, action) pair we've
   executed is a recorded edge. This is a directed graph we build online.
2. From the current node, take an action never yet tried from this exact
   node. That is pure information gain -- a step that cannot be wasted.
3. When the current node is exhausted, BFS the recorded graph for the
   nearest node that still has untried actions and walk there along
   known edges. This is what turns the agent from a random flailer into
   a systematic explorer: no step is spent re-deriving known dynamics.
4. Clicks (ACTION6) are aimed at *objects*, not pixels. A uniform random
   click picks 1 of 4096 coordinates and almost always hits background.
   Instead we segment the frame into connected same-color components and
   click their representative pixels, smallest/rarest first --
   interactive elements are typically small and oddly-colored. This is
   the single largest win on click-driven games like vc33.
5. Any (node, action) that produced GAME_OVER is marked fatal and never
   repeated from that node, so a death is learned from permanently
   instead of being rediscovered.

--------------------------------------------------------------------
MEMORY
--------------------------------------------------------------------
25 games run concurrently for hours, so per-node storage has to be tiny.
A node therefore costs two ints -- how many candidate actions it has
(`_total`) and how many we've worked through (`_cursor`) -- plus its
outgoing edges. The candidate *list* is never stored: it is a pure
function of the grid, so it is recomputed on demand and kept in a small
bounded cache. Because candidate order is deterministic, the cursor
alone is enough to know what has and hasn't been tried.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from collections import Counter, deque
from typing import Any, Optional

from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent

logger = logging.getLogger()


# ---------------------------------------------------------------------
# Tuning. Every knob is env-overridable so a Kaggle run can be retuned
# without editing (and re-uploading) code.
# ---------------------------------------------------------------------
def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


# The competition run is capped at 9h wall clock. We stop well before
# that: the gateway still has to close the scorecard and write
# submission.parquet after every agent thread returns, and a run killed
# by the hard limit produces NO submission at all. Time left on the
# clock is worth far less than a submission that actually lands.
RUN_BUDGET_SECONDS = _env_float("ARC_RUN_BUDGET_SECONDS", 7.5 * 3600)

# Per-game action cap. The base class defaults to 80, which is a demo
# value: locally the engine sustains >200 actions/sec, so 80 actions is
# ~0.4 seconds of a 9-hour budget. Real termination is the deadline in
# `is_done`; this is only a runaway guard.
MAX_ACTIONS_PER_GAME = _env_int("ARC_MAX_ACTIONS", 250_000)

# Object-derived click targets considered per distinct frame.
MAX_CLICK_CANDIDATES = _env_int("ARC_MAX_CLICKS", 48)

# Coarse fallback sweep, tried only after every object click has been
# tried from a node: stride 8 over 64x64 => 64 extra targets.
LATTICE_STRIDE = _env_int("ARC_LATTICE_STRIDE", 8)

# Hard ceiling on nodes tracked per game, so a game that animates every
# frame can't grow the graph without bound over several hours.
MAX_TRACKED_STATES = _env_int("ARC_MAX_STATES", 60_000)

# Cap on BFS expansion when routing to a frontier node.
PLAN_NODE_CAP = _env_int("ARC_PLAN_NODE_CAP", 50_000)

# Candidate lists are recomputed rather than stored; this bounds the
# recompute cache. Exploration is spatially local, so a small cache
# gets a high hit rate.
CANDIDATE_CACHE_LIMIT = _env_int("ARC_CANDIDATE_CACHE", 512)

# How often to fall back to a hard RESET once the reachable graph is
# fully explored and random probing isn't finding anything new.
STUCK_RESET_INTERVAL = _env_int("ARC_STUCK_RESET_INTERVAL", 250)

# Deaths needed at the same (node, action) before it is treated as
# genuinely lethal and the edge is removed.
#
# It must be more than one. Games like ls20 end the run when a per-level
# STEP BUDGET expires, and because that budget is masked out of the node
# key (it is HUD), the death lands on whatever node the agent happened to
# be standing on. Pruning on the first death therefore deletes perfectly
# good edges at effectively random places in the maze, and can make the
# level unsolvable. A genuine hazard kills every time it is touched, so
# requiring a repeat separates the two.
DEATH_TOLERANCE = _env_int("ARC_DEATH_TOLERANCE", 2)

# Once the reachable graph is fully mapped and no progress is coming, the
# agent would otherwise spend the rest of the run taking uniformly random
# actions. Instead, after this many stuck steps, throw the map away and
# explore again from scratch: tie-breaking is seeded per node, so a fresh
# map re-runs the systematic search rather than flailing.
STUCK_REEXPLORE_AFTER = _env_int("ARC_STUCK_REEXPLORE", 1_500)

# HUD masking. Some games render a status display straight into the
# frame -- ls20 draws a decrementing `StepCounter` as a bar along rows
# 61-62 -- so the raw grid changes on EVERY step and no state is ever
# revisited. That silently destroys the whole approach: the graph has no
# cycles, every node looks like an unexplored frontier, and the agent
# just walks forwards forever.
#
# Detecting it needs care. A shrinking bar advances by one segment per
# step, so each individual bar CELL flips only once per life and its
# per-cell change rate is ~2%, far below any sensible threshold. The
# giveaway is spatial: the bar's *rows* contain a change on essentially
# every transition. Measured on ls20 over 400 transitions, rows 61 and
# 62 changed in 98.3% of them while the next busiest row was at 26%, and
# no single cell reached 90%. So we track per-cell, per-row and
# per-column rates during warm-up and mask whichever cross the
# threshold.
VOLATILE_WARMUP = _env_int("ARC_VOLATILE_WARMUP", 400)
VOLATILE_THRESHOLD = _env_float("ARC_VOLATILE_THRESHOLD", 0.9)

# Refuse to mask more than this fraction of the frame. If that much of
# the board qualifies, the reading is not a HUD and dropping it would
# collapse unrelated states together.
MASK_CAP_FRACTION = _env_float("ARC_MASK_CAP", 0.35)

# The base class keeps every frame it has ever seen in `self.frames` and
# appends a full 64x64 frame to a JSONL recording on every action. At the
# stock 80-action cap that is nothing; at our cap it is fatal. 250k
# actions x 25 concurrent games is ~6M frames: roughly 75 GB of recording
# in /kaggle/working (which is also the notebook's output) and ~10 GB of
# retained frame objects per game, plus an open/write/close syscall on
# every single step. Both are bounded here.
#
# Nothing in the scoring path reads either: `submission.parquet` is
# produced by the gateway sidecar from its own server-side record, and
# RECORDINGS_DIR is consumed only by the framework's `Playback` agent.
FRAME_HISTORY = _env_int("ARC_FRAME_HISTORY", 8)
KEEP_RECORDINGS = _env_int("ARC_KEEP_RECORDINGS", 0)

_DEADLINE = time.monotonic() + RUN_BUDGET_SECONDS

_RESET_ID = GameAction.RESET.value
_ACTION6_ID = GameAction.ACTION6.value

# An action is packed into a single int so the graph can key on it
# cheaply: action_id in the high bits, y and x in 6 bits each.
_COORD_BITS = 6
_COORD_MAX = (1 << _COORD_BITS) - 1  # 63, matching ComplexAction's bounds


def encode_action(action_id: int, x: Optional[int], y: Optional[int]) -> int:
    if x is None or y is None:
        return action_id << (2 * _COORD_BITS)
    return (action_id << (2 * _COORD_BITS)) | (y << _COORD_BITS) | x


def decode_action(code: int) -> tuple[int, Optional[int], Optional[int]]:
    action_id = code >> (2 * _COORD_BITS)
    if action_id == _ACTION6_ID:
        return action_id, code & _COORD_MAX, (code >> _COORD_BITS) & _COORD_MAX
    return action_id, None, None


# ---------------------------------------------------------------------
# Frame analysis. Pure functions over plain nested sequences -- no
# arcengine types -- so they are cheap to reason about and to test.
# ---------------------------------------------------------------------
def grid_of(frame: FrameData) -> tuple[tuple[int, ...], ...]:
    """The rendered view as a hashable grid; empty tuple if unavailable.

    `frame.frame` is a *list* of 2D grids; the rendered view is the last
    one. The base class seeds `Agent.frames` with an empty placeholder
    `FrameData()`, so the empty case is real and must not raise.
    """
    grids = frame.frame or []
    if not grids:
        return ()
    return tuple(tuple(row) for row in grids[-1])


def click_targets(
    grid: tuple[tuple[int, ...], ...],
    limit: int = MAX_CLICK_CANDIDATES,
    ignore: frozenset[int] = frozenset(),
) -> list[tuple[int, int]]:
    """Salient (x, y) click targets, most promising first.

    Segments the frame into 4-connected same-color components, ignoring
    the single most common color (the background), and returns one
    representative pixel per component. Components are ranked by (size,
    frame-wide pixel count of that color) ascending, because interactive
    elements -- pieces, buttons, cursors -- tend to be small and to use
    colors that are rare frame-wide, whereas walls and floors are large
    and common.

    The representative is the component member nearest the component's
    centroid, so it is guaranteed to lie *on* the object even when it is
    L-shaped or hollow; a raw centroid can fall in a hole and land on
    background instead.

    `ignore` holds flat cell indices (y * width + x) that are treated as
    background: these are the volatile HUD cells identified by the agent,
    and segmenting them would otherwise turn a step counter's digits into
    the highest-priority click targets on the board.

    Deterministic for a given grid, which is what lets the agent store a
    single cursor per node instead of the whole candidate list.
    """
    height = len(grid)
    if height == 0:
        return []
    width = len(grid[0])
    if width == 0:
        return []

    if ignore:
        color_totals = Counter(
            value
            for y, row in enumerate(grid)
            for x, value in enumerate(row)
            if y * width + x not in ignore
        )
        if not color_totals:
            return []
    else:
        color_totals = Counter(value for row in grid for value in row)
    background, _ = color_totals.most_common(1)[0]

    seen = [[False] * width for _ in range(height)]
    components: list[tuple[int, int, int, int]] = []  # size, color_total, x, y

    for y0 in range(height):
        row0 = grid[y0]
        for x0 in range(width):
            if seen[y0][x0] or row0[x0] == background or (y0 * width + x0) in ignore:
                continue
            color = row0[x0]
            seen[y0][x0] = True
            stack = [(y0, x0)]
            cells: list[tuple[int, int]] = []
            sum_y = 0
            sum_x = 0
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                sum_y += cy
                sum_x += cx
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and not seen[ny][nx]
                        and grid[ny][nx] == color
                        and (ny * width + nx) not in ignore
                    ):
                        seen[ny][nx] = True
                        stack.append((ny, nx))

            count = len(cells)
            mean_y = sum_y / count
            mean_x = sum_x / count
            best = min(cells, key=lambda c: (c[0] - mean_y) ** 2 + (c[1] - mean_x) ** 2)
            # cells hold (row, col) == (y, x); the payload wants (x, y).
            components.append((count, color_totals[color], best[1], best[0]))

    components.sort(key=lambda c: (c[0], c[1], c[3], c[2]))

    targets: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set()
    for _, _, x, y in components:
        if (x, y) in used:
            continue
        used.add((x, y))
        targets.append((x, y))
        if len(targets) >= limit:
            break
    return targets


def lattice_targets(
    width: int = 64, height: int = 64, stride: int = LATTICE_STRIDE
) -> list[tuple[int, int]]:
    """Coarse sweep of the whole frame, used only as a last resort.

    Offset by half a stride so samples land in the middle of cells
    rather than on the boundaries between them.
    """
    half = stride // 2
    return [
        (x, y)
        for y in range(half, min(height, _COORD_MAX + 1), stride)
        for x in range(half, min(width, _COORD_MAX + 1), stride)
    ]


def hud_mask(
    cell_changes: list[int],
    row_changes: list[int],
    col_changes: list[int],
    samples: int,
    width: int,
    height: int,
    threshold: float = VOLATILE_THRESHOLD,
    cap_fraction: float = MASK_CAP_FRACTION,
) -> frozenset[int]:
    """Flat cell indices to exclude from the node key, or an empty set.

    A cell qualifies if it changed in `threshold` of transitions, and so
    does every cell of any row or column that contained *some* change in
    that fraction of transitions. The row/column rule is the one that
    matters: a shrinking progress bar advances one segment per step, so
    no individual bar cell is volatile, but the bar's rows change
    constantly.

    Returns empty if nothing qualifies or if more than `cap_fraction` of
    the frame does -- at that point the reading is not a HUD, and masking
    it would fuse genuinely different states into one node.
    """
    total = width * height
    if samples <= 0 or total <= 0:
        return frozenset()

    cutoff = threshold * samples
    masked = {index for index, count in enumerate(cell_changes) if count >= cutoff}
    for y, count in enumerate(row_changes):
        if count >= cutoff:
            masked.update(range(y * width, y * width + width))
    for x, count in enumerate(col_changes):
        if count >= cutoff:
            masked.update(range(x, total, width))

    if not masked or len(masked) > cap_fraction * total:
        return frozenset()
    return frozenset(masked)


def candidate_actions(
    grid: tuple[tuple[int, ...], ...],
    available: list[int],
    node: int = 0,
    ignore: frozenset[int] = frozenset(),
) -> list[int]:
    """Ordered, packed action codes worth trying from a given frame.

    Order *is* the exploration priority: simple actions first (there are
    at most six, and each is one cheap, highly informative probe), then
    object-aimed clicks, then the coarse lattice.

    The simple actions are shuffled using `node` as the seed. That
    matters more than it looks: the agent always takes candidate
    `cursor`, and on a game whose frames rarely repeat the cursor is
    almost always 0, so a fixed order degenerates into pressing the same
    button forever (measured on ls20: ACTION1 1771 times vs ACTION4 41).
    Seeding on the node keeps the order a deterministic function of the
    frame -- which is what makes a single cursor per node sufficient --
    while decorrelating the choice across different frames.

    Click order is deliberately *not* shuffled: smallest-and-rarest-first
    is a real prior about which pixels are interactive.
    """
    live = [a for a in available if a != _RESET_ID]
    if not live:
        # Nothing advertised: fall back to every non-RESET action so we
        # can at least discover which ones are wired up.
        live = [a.value for a in GameAction if a is not GameAction.RESET]

    simple = [encode_action(a, None, None) for a in live if a != _ACTION6_ID]
    random.Random(node & 0xFFFFFFFF).shuffle(simple)

    codes = simple
    if _ACTION6_ID in live:
        coords = click_targets(grid, ignore=ignore)
        seen = set(coords)
        height = len(grid) or 64
        width = len(grid[0]) if grid else 64
        for point in lattice_targets(width, height):
            if point not in seen:
                seen.add(point)
                coords.append(point)
        codes.extend(encode_action(_ACTION6_ID, x, y) for x, y in coords)
    return codes


# ---------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------
class MyAgent(Agent):
    """One instance per game; see the module docstring for the strategy."""

    MAX_ACTIONS = MAX_ACTIONS_PER_GAME

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # Deterministic per game, but different across games, so two
        # games don't tie-break identically for no reason.
        self._rng = random.Random(hash(self.game_id) & 0xFFFFFFFF)

        # --- the learned state graph, two ints + edges per node ------
        self._total: dict[int, int] = {}          # node -> candidate count
        self._cursor: dict[int, int] = {}         # node -> next index to try
        self._edges: dict[int, dict[int, int]] = {}  # node -> code -> node
        self._fatal: set[tuple[int, int]] = set()     # (node, code) proven lethal
        self._deaths: dict[tuple[int, int], int] = {}  # deaths seen per (node, code)

        # Recomputed-on-demand candidate lists, bounded.
        self._cache: dict[int, list[int]] = {}

        # --- volatile-cell masking (see VOLATILE_WARMUP) --------------
        self._prev_flat: Optional[tuple[int, ...]] = None
        self._cell_changes: list[int] = []
        self._row_changes: list[int] = []
        self._col_changes: list[int] = []
        self._samples = 0
        self._mask_frozen = False
        self._kept: Optional[tuple[int, ...]] = None   # flat indices in the key
        self._volatile: frozenset[int] = frozenset()   # flat indices excluded

        # --- execution bookkeeping ----------------------------------
        # (node, code) awaiting its outcome, folded in on the next call
        self._pending: Optional[tuple[int, int]] = None
        # payload handed to the env; see do_action_request()
        self._pending_data: dict[str, Any] = {"game_id": ""}
        self._pending_reasoning: Optional[dict[str, Any]] = None
        # queued route to a frontier node: [(node_we_must_be_in, code), ...]
        self._plan: deque[tuple[int, int]] = deque()

        self._best_levels = 0
        self._stuck_steps = 0
        self._steps = 0
        self._recorded = 0
        self._graph_full = False

    # -- framework hooks ---------------------------------------------
    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        if latest_frame.state is GameState.WIN:
            logger.info(
                "%s: WIN after %d actions (%d levels)",
                self.game_id, self._steps, latest_frame.levels_completed,
            )
            return True
        if time.monotonic() >= _DEADLINE:
            logger.info(
                "%s: stopping on global time budget after %d actions "
                "(%d levels, %d nodes mapped)",
                self.game_id, self._steps, self._best_levels, len(self._total),
            )
            return True
        return False

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        self._steps += 1
        grid = grid_of(latest_frame)
        flat = tuple(value for row in grid for value in row)

        # 0. Once the HUD mask is known, every node id computed before it
        #    was keyed on different cells, so the graph has to be dropped.
        height = len(grid)
        width = len(grid[0]) if height else 0
        if self._learn_mask(flat, width, height):
            logger.info(
                "%s: masking %d volatile cells out of the state key at "
                "action %d; restarting the map",
                self.game_id, len(self._volatile), self._steps,
            )
            self._reset_graph()
            self._pending = None

        node = self._node_key(flat)

        # 1. Learn from the action taken last call, whose outcome is
        #    exactly the frame we are being handed now.
        self._observe(node, latest_frame)

        # 2. A dead or unstarted game can only be RESET.
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._plan.clear()
            self._pending = None
            return self._emit(_RESET_ID, None, None, "reset")

        # 3. Decide.
        code, why = self._select(node, grid, latest_frame)
        self._pending = (node, code)
        action_id, x, y = decode_action(code)
        return self._emit(action_id, x, y, why)

    def append_frame(self, frame: FrameData) -> None:
        """Keep only a short window of frames, and cap recording on disk.

        See FRAME_HISTORY / KEEP_RECORDINGS: the base implementation grows
        without bound, which only worked because the stock action cap was
        80. This agent never looks further back than the latest frame --
        its history lives in the state graph, not in `self.frames`.
        """
        self.frames.append(frame)
        if len(self.frames) > FRAME_HISTORY:
            del self.frames[:-FRAME_HISTORY]
        if frame.guid:
            self.guid = frame.guid
        if (
            self._recorded < KEEP_RECORDINGS
            and hasattr(self, "recorder")
            and not self.is_playback
        ):
            self.recorder.record(json.loads(frame.model_dump_json()))
            self._recorded += 1

    def do_action_request(self, action: GameAction) -> FrameData:
        """Send the action using OUR payload, not the shared enum's.

        `GameAction` is an Enum, so `GameAction.ACTION6` is a single
        process-wide object and `set_data()` mutates it in place. With
        ~25 games running as concurrent threads, one game's click
        coordinates can be overwritten by another's in the window
        between `choose_action` returning and the base class reading
        `action.action_data` -- silently sending the wrong click AND
        poisoning the state graph with a mislabelled edge.

        `EnvironmentWrapper.step()` uses `action` only for its identity
        and takes the payload from `data`, so passing our own
        per-instance dict removes the race entirely. The dict is shaped
        exactly as `action_data.model_dump()` would be.
        """
        raw = self.arc_env.step(
            action, data=dict(self._pending_data), reasoning=self._pending_reasoning
        )
        return self._convert_raw_frame_data(raw)

    # -- internals ----------------------------------------------------
    def _emit(
        self, action_id: int, x: Optional[int], y: Optional[int], why: str
    ) -> GameAction:
        if x is None:
            self._pending_data = {"game_id": ""}
        else:
            self._pending_data = {"game_id": "", "x": x, "y": y}
        self._pending_reasoning = {"why": why}
        return GameAction.from_id(action_id)

    def _observe(self, node: int, latest_frame: FrameData) -> None:
        """Fold the outcome of the previous action into the graph."""
        levels = latest_frame.levels_completed
        if levels > self._best_levels:
            logger.info(
                "%s: reached level %d at action %d (%d nodes mapped)",
                self.game_id, levels, self._steps, len(self._total),
            )
            self._best_levels = levels
            self._stuck_steps = 0
            # A cleared level is unreachable from here on, so everything
            # learned about it is dead weight. Dropping it keeps memory
            # proportional to the level being played, not the whole run.
            self._reset_graph()

        pending, self._pending = self._pending, None
        if pending is None:
            return
        prev_node, code = pending

        if latest_frame.state is GameState.GAME_OVER:
            # Only prune once the same move has killed us repeatedly; see
            # DEATH_TOLERANCE for why a single death proves nothing.
            key = (prev_node, code)
            deaths = self._deaths.get(key, 0) + 1
            self._deaths[key] = deaths
            if deaths >= DEATH_TOLERANCE:
                self._fatal.add(key)
                outgoing = self._edges.get(prev_node)
                if outgoing is not None:
                    outgoing.pop(code, None)
            self._plan.clear()
            return

        if prev_node in self._total:
            self._edges.setdefault(prev_node, {})[code] = node

    def _learn_mask(self, flat: tuple[int, ...], width: int, height: int) -> bool:
        """Accumulate change rates; freeze the HUD mask when ready.

        Returns True exactly once, on the transition where the mask is
        adopted and every existing node id becomes invalid.
        """
        if self._mask_frozen:
            return False

        previous, self._prev_flat = self._prev_flat, flat
        if previous is None or len(previous) != len(flat) or width <= 0:
            # First frame, or the grid was resized: start counting fresh.
            self._cell_changes = [0] * len(flat)
            self._row_changes = [0] * height
            self._col_changes = [0] * width
            self._samples = 0
            return False

        counts = self._cell_changes
        rows: set[int] = set()
        cols: set[int] = set()
        for index, (before, after) in enumerate(zip(previous, flat)):
            if before != after:
                counts[index] += 1
                rows.add(index // width)
                cols.add(index % width)
        for y in rows:
            self._row_changes[y] += 1
        for x in cols:
            self._col_changes[x] += 1
        self._samples += 1

        if self._samples < VOLATILE_WARMUP:
            return False

        self._mask_frozen = True
        volatile = hud_mask(
            counts, self._row_changes, self._col_changes,
            self._samples, width, height,
        )

        # Free the counters; they are dead weight for the rest of the run.
        self._cell_changes = []
        self._row_changes = []
        self._col_changes = []
        self._prev_flat = None

        if not volatile:
            return False
        self._volatile = volatile
        self._kept = tuple(i for i in range(len(flat)) if i not in volatile)
        return True

    def _node_key(self, flat: tuple[int, ...]) -> int:
        kept = self._kept
        if kept is None:
            return hash(flat)
        return hash(tuple(map(flat.__getitem__, kept)))

    def _reset_graph(self) -> None:
        self._total.clear()
        self._cursor.clear()
        self._edges.clear()
        self._fatal.clear()
        self._deaths.clear()
        self._cache.clear()
        self._plan.clear()

    def _candidates(
        self, node: int, grid: tuple[tuple[int, ...], ...], latest_frame: FrameData
    ) -> list[int]:
        """Candidate codes for `node`, recomputed and cached on demand."""
        cached = self._cache.get(node)
        if cached is not None:
            return cached

        codes = candidate_actions(
            grid, latest_frame.available_actions, node=node, ignore=self._volatile
        )

        if len(self._cache) >= CANDIDATE_CACHE_LIMIT:
            self._cache.clear()
        self._cache[node] = codes

        if node not in self._total:
            if len(self._total) < MAX_TRACKED_STATES:
                self._total[node] = len(codes)
                self._cursor[node] = 0
            elif not self._graph_full:
                self._graph_full = True
                logger.info(
                    "%s: node budget of %d reached; exploring greedily from here",
                    self.game_id, MAX_TRACKED_STATES,
                )
        return codes

    def _has_untried(self, node: int) -> bool:
        return self._cursor.get(node, 0) < self._total.get(node, 0)

    def _select(
        self, node: int, grid: tuple[tuple[int, ...], ...], latest_frame: FrameData
    ) -> tuple[int, str]:
        codes = self._candidates(node, grid, latest_frame)

        # (a) Follow a route to a frontier node, as long as reality still
        #     matches the plan. A mismatch means the game was less
        #     deterministic than recorded, so throw the plan away.
        while self._plan:
            expected, code = self._plan[0]
            if expected != node:
                self._plan.clear()
                break
            self._plan.popleft()
            if (node, code) in self._fatal:
                self._plan.clear()
                break
            return code, "route-to-frontier"

        # (b) Anything untried right here is the cheapest information
        #     available, so take it. The cursor is why this works with
        #     only an int per node: `codes` is deterministic for a grid,
        #     so index i always means the same action.
        index = self._cursor.get(node, 0)
        if index < len(codes) and node in self._total:
            self._cursor[node] = index + 1
            self._stuck_steps = 0
            return codes[index], "probe-untried"

        # (c) This node is exhausted. Walk to the nearest node that
        #     isn't, using only edges already verified by experience.
        route = self._route_to_frontier(node)
        if route:
            self._plan = route
            _, code = self._plan.popleft()
            self._stuck_steps = 0
            return code, "route-to-frontier"

        # (d) Every reachable node is exhausted.
        self._stuck_steps += 1

        # Measured on ls20: the reachable map closes after ~1.6k actions
        # and without this the remaining 34k actions all went to
        # "stuck-random", i.e. 86% of the budget spent flailing. Dropping
        # the map re-runs the systematic search instead, and because the
        # fatal set is cleared with it, edges wrongly pruned by a
        # step-budget death get another chance.
        if self._stuck_steps >= STUCK_REEXPLORE_AFTER:
            logger.info(
                "%s: reachable map exhausted after %d actions; re-exploring",
                self.game_id, self._steps,
            )
            self._reset_graph()
            self._stuck_steps = 0
            return encode_action(_RESET_ID, None, None), "re-explore"

        # Periodically RESET to shake the game out of its configuration;
        # otherwise inject a random probe, which can land outside the
        # object and lattice candidate sets.
        if self._stuck_steps % STUCK_RESET_INTERVAL == 0:
            self._plan.clear()
            return encode_action(_RESET_ID, None, None), "stuck-reset"
        return self._random_code(latest_frame), "stuck-random"

    def _random_code(self, latest_frame: FrameData) -> int:
        live = [a for a in latest_frame.available_actions if a != _RESET_ID]
        if not live:
            live = [a.value for a in GameAction if a is not GameAction.RESET]
        action_id = self._rng.choice(live)
        if action_id == _ACTION6_ID:
            return encode_action(
                action_id,
                self._rng.randint(0, _COORD_MAX),
                self._rng.randint(0, _COORD_MAX),
            )
        return encode_action(action_id, None, None)

    def _route_to_frontier(self, start: int) -> Optional[deque[tuple[int, int]]]:
        """BFS the recorded graph for the closest node with untried work.

        Returns the route as [(node_it_must_be_in, code_to_take), ...] so
        every step can be validated against reality before it is used.
        """
        queue: deque[int] = deque([start])
        came_from: dict[int, Optional[tuple[int, int]]] = {start: None}
        expanded = 0

        while queue and expanded < PLAN_NODE_CAP:
            current = queue.popleft()
            expanded += 1

            if current != start and self._has_untried(current):
                route: deque[tuple[int, int]] = deque()
                cursor = current
                while True:
                    edge = came_from[cursor]
                    if edge is None:
                        break
                    prev_node, code = edge
                    route.appendleft((prev_node, code))
                    cursor = prev_node
                return route

            for code, nxt in self._edges.get(current, {}).items():
                if nxt in came_from or (current, code) in self._fatal:
                    continue
                came_from[nxt] = (current, code)
                queue.append(nxt)

        return None
