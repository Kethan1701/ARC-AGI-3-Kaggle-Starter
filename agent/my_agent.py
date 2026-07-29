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
SCORING -- why exploration alone scores ~0
--------------------------------------------------------------------
Read `arc_agi/scorecard.py` before changing any of this. The metric is
NOT "levels completed"; it is efficiency, and it is quadratic:

    level_score = min((baseline_actions / actions_taken) ** 2 * 100, 115)

`baseline_actions` ships in each game's `metadata.json` and is small --
ls20's seven levels are [22, 123, 73, 84, 96, 192, 186]. A run that
maps the state space to find level 1 spends thousands of actions on a
22-action level and scores (22/5000)^2*100 ~= 0.002. Completing a level
is worth almost nothing; completing it *quickly* is worth everything.

There is no way to buy that back with a second attempt. The Kaggle
gateway runs `arc_agi.api.RestAPI` with `competition_mode=True`, and
that flag turns off both halves of the obvious exploit (verified against
the real Flask app in `tests/test_competition_rules.py`):

* `_get_or_create_environment` returns None once the scorecard already
  has the game, so there is exactly ONE run per game and
  `max(run.score for run in runs)` has a single candidate. Extra
  attempts are not free, they are impossible.
* A RESET sent when `_action_count == 0` -- exactly the condition
  `BaseGame.handle_reset` uses to choose `full_reset()` -- is swallowed
  by the server, which returns `g.observation_space` without stepping
  the game, and still bills an action. So "RESET twice to open a fresh
  run" cannot open one, and costs an action each time it tries.

Note what this means for measurement: any harness that drives the
environment wrapper directly, `scripts/score_local.py` included, never
executes `api.py` and therefore never applies these rules. Two
submissions measured 0.053 and 2.10 that way and both scored exactly
0.19 on the leaderboard. Use `scripts/score_competition.py`, which
stands up the genuine gateway with competition rules on.

So the only thing that scores is finishing a level near baseline on the
first and only pass, and the search is tuned for billed actions rather
than coverage:

  a. Probe at most `BREADTH_INITIAL` candidates at a node before moving
     on, and widen only when every reachable node is exhausted -- a node
     offers ~117 codes and draining them is rarely where the answer is.
  b. Repeat an action that just reached a new state (`MOMENTUM_LIMIT`
     times at most). Corridors, pushes and falls all reward repetition,
     and a pure state-graph search throws that hint away every step.
  c. Keep the solution graph and the BFS route extraction. They cost no
     actions, and they are what a within-level retry would replay if the
     rules ever allow one again (`ARC_ENABLE_CYCLE=1`).

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
import queue
import random
import re
import threading
import time
import zlib
from collections import Counter, deque
from itertools import chain
from typing import Any, Optional

from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent

logger = logging.getLogger()


# `Agent.main()` emits one INFO line per action on the ROOT logger, and
# `arc_agi` installs a root handler, so those lines really are written.
# At the stock 80-action cap that is 80 lines; with this agent's budget it
# is millions per run across 25 games -- enough I/O to eat into the action
# budget it is meant to report on, and enough output to risk tripping the
# notebook's log limit. Only that one message is dropped; everything else,
# including this agent's own progress logs, still comes through.
#
# The message is matched by content because the framework logs it through
# the root logger, which this agent also uses -- there is no separate
# logger to silence.
class _DropPerActionLine(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "avg fps" not in record.getMessage()


# Set ARC_KEEP_ACTION_LOG=1 to get the per-action line back for debugging.
if not os.environ.get("ARC_KEEP_ACTION_LOG"):
    try:
        logger.addFilter(_DropPerActionLine())
    except Exception:  # pragma: no cover - logging must never break the run
        pass


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
#
# `Agent.main()` loops on `action_counter <= MAX_ACTIONS`, so this is a
# HARD stop that silently ends the game early if it binds first. Left high
# because a low cap can only lose levels; on Kaggle the binding constraint
# is the 9h wall clock against ~55 actions/s of gateway round-trips, not
# this number.
MAX_ACTIONS_PER_GAME = _env_int("ARC_MAX_ACTIONS", 5_000_000)

# Replaying a banked route into a *fresh scored run* is worthless on Kaggle
# and actively harmful, so it is off by default. Two facts, both verified
# against the real `arc_agi.api` with `competition_mode=True`:
#
#   * `_get_or_create_environment` returns None once the scorecard already
#     has the game, so there is exactly one run per game and
#     `max(run.score for run in runs)` has a single candidate.
#   * a RESET sent when `_action_count == 0` -- precisely the condition
#     `arcengine.base_game.handle_reset` uses to choose `full_reset()` --
#     is swallowed by the server (`response = g.observation_space`, the
#     game is never stepped) yet still bills an action.
#
# So the cycle can never open the fresh run it exists to open, and every
# attempt costs an action against the level being scored. Set
# ARC_ENABLE_CYCLE=1 only when playing a non-competition scorecard.
ENABLE_CYCLE = _env_int("ARC_ENABLE_CYCLE", 0)

# Object-derived click targets considered per distinct frame.
MAX_CLICK_CANDIDATES = _env_int("ARC_MAX_CLICKS", 48)

# How many candidates a node may consume before we move on, and the ceiling
# it deepens to. A node offers ~117 codes (a few simple actions, up to 48
# object clicks, then a 64-point lattice sweep), and the old search drained
# all of them before routing anywhere else. Under competition rules that is
# the single largest source of waste: every probe is billed against the
# level being scored, and the metric is quadratic, so ls20 finishing level 1
# in 983 actions against a 22-action baseline scores 0.05 out of 100.
#
# Breadth-first over the *first few* candidates at many nodes finds the
# productive action far cheaper than draining one node, and nothing is lost
# permanently: when every reachable node is exhausted the breadth doubles
# and the same nodes reopen, with cursors still valid because candidate
# order is a pure function of the grid.
BREADTH_INITIAL = _env_int("ARC_BREADTH", 12)
BREADTH_MAX = _env_int("ARC_BREADTH_MAX", 256)

# Consecutive repeats of an action that keeps producing new states.
#
# OFF by default: measured, this is the single most expensive thing the
# agent was doing. On sp80 level 1, isolated with everything else held
# fixed:
#
#     momentum on   668 actions  (momentum=376 of them)
#     momentum off   78 actions
#
# An 8.6x regression, and sp80 is the game that carries the score -- at 78
# actions against a baseline of 39 it scores (39/78)^2*100 = 25 points; at
# 668 it scores 0.34. One game, 1.2 of a 1.33 aggregate.
#
# The trap was that "keeps producing new states" is not the same as
# "making progress". A game with any animation or scrolling emits a fresh
# frame for every repeat, so momentum reads its own motion as success and
# walks away from the frontier, then pays route-to-frontier to walk back.
# It was adopted on an 8% gain on ls20 (983 -> 904) without checking
# whether that held anywhere else. It does not.
#
# Kept behind the env var rather than deleted so the behaviour is still
# reachable for experiments, but nothing enables it by default.
MOMENTUM_LIMIT = _env_int("ARC_MOMENTUM", 0)

# An action id that has been tried this many times within a level without
# ever changing the state is treated as not wired up in this game, and is
# skipped until the next widening.
#
# This is worth far more than it looks. A node offers ~117 codes and ~112
# of them are ACTION6 clicks, so a game that ignores clicks entirely wastes
# almost every probe on them; conversely a pure click game wastes its cheap
# simple-action slots. The threshold is per action *id*, not per code, so a
# click that misses tells us nothing -- only ACTION6 never doing anything
# anywhere does.
#
# The two thresholds are wildly different on purpose, because the evidence
# is not comparable:
#
#   ACTION1-5 are global -- they take no coordinate, so pressing one from
#   half a dozen unrelated states and seeing the frame never move is strong
#   evidence the game does not wire it up. A fair trial is cheap.
#
#   ACTION6 is a click, and a miss is not evidence. There are ~112 targets
#   per node; writing clicks off after a few dozen misses is not merely
#   wasteful, it is *unsafe* -- on a game whose only progress comes from
#   one specific cell it disables the sole winning action until the next
#   widening. The click trial is therefore set above the per-node target
#   count, so "clicks do nothing" only fires once every target class has
#   been sampled several times over.
#
# Scoring makes the cheap half urgent: a level scores (baseline/actions)^2,
# level-1 baselines run ~7-78 actions, so 200 actions spent merely learning
# which buttons are inert already puts a near-perfect solve out of reach.
DEAD_ACTION_TRIALS = _env_int("ARC_DEAD_TRIALS", 6)
DEAD_CLICK_TRIALS = _env_int("ARC_DEAD_CLICK_TRIALS", 240)

# Shared empty mapping for `_edges` lookups on unmapped nodes.
EMPTY_EDGES: dict[int, int] = {}

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
# The warm-up length is not a free parameter: freezing the mask calls
# `_reset_graph()`, so every node learned during warm-up is discarded.
# The old 400 therefore put a ~400-action floor under level 1 on any game
# needing a mask -- against level-1 baselines of 7 to 78. Since a level
# scores (baseline/actions)^2, that floor alone forecloses any score,
# which is why doubling the action budget from 5k to 10k per game moved
# the aggregate by nothing: the constraint was never how many actions
# were available, it was that the first 400 were spent and thrown away.
#
# The clear times gave it away -- every masked game cleared just after the
# floor, and the one game that scores cleared before it:
#
#     tn36 426   r11l 473   vc33 475   lp85 502      sp80 78
#
# Swept, level-1 actions (baseline in brackets):
#
#     warmup      25    50   100   400
#     tn36 [32]   53    87   697   426
#     r11l [22]  113   172   235   473
#     vc33  [7]  100   118   169   475
#     ls20 [22]  527   874   681   984
#     tu93 [19]  397   489   540   813
#     sp80 [39]   79   390    78    78
#
# 25 wins or ties nearly everywhere, and takes tn36 to 1.66x baseline --
# a 36-point level where it used to score 0.02. The row rates the mask
# keys on are extreme by design (measured on ls20: 98.3% for the bar's
# rows against 26% for the next busiest), so they separate on far fewer
# samples than 400. sp80 pays one extra action; ls20 and tu93 nearly
# halve. Note the sweep is not monotone -- sp80 costs 390 at warmup 50 --
# so this is a floor worth re-measuring rather than tuning blindly.
# MEASUREMENT ONLY -- 0 reproduces shipped behaviour exactly (n ^ 0 == n).
#
# On the games that never close their map the agent is effectively taking a
# node-hash-seeded random walk, so the aggregate is decided by which one or
# two games happen to stumble into a level quickly. Two configs can then
# differ by 40% for reasons that are pure trajectory luck rather than merit.
# Varying this re-rolls the walk without changing anything else, which is
# the only way to tell a real effect from a lucky draw.
AGENT_SEED = _env_int("ARC_SEED", 0)

# Widen breadth the moment the map is exhausted, rather than after
# STUCK_REEXPLORE_AFTER actions of random flailing. Widening costs zero
# billed actions; flailing costs one per step. 0 restores the old order for
# A/B measurement.
# MEASURED AND REJECTED, kept as a flag because the mechanism is sound and
# the verdict may flip once the movement games stop random-walking.
#
# Widening on exhaustion costs zero billed actions where flailing costs one
# per step, so this looked like a free win, and on the games that flail it
# is one: lp85 501 -> 169, lf52 ~515 -> ~290, s5i5 5001 -> 2630, identical
# across three seeds. But it costs tn36 52 -> 86, also identical across
# three seeds, and tn36 is the single highest-scoring game on the dev set.
# Under a quadratic score that one game outweighs all three gains:
#
#   seed      off      on     delta
#     0    0.1129  0.0802   -0.0327
#   101    0.1223  0.0895   -0.0328
#   202    0.1651  0.1324   -0.0327
#
# Sum of per-game deltas across the 25 games is -2.46, so it is negative in
# expectation on this sample, not merely unlucky. Preserving the
# dead-action record through the widen (ARC_WIDEN_CLEARS_DEAD=0) was tested
# as the suspected cause and changed nothing: tn36 stayed at 86 and the
# aggregate moved by 0.0002. tn36 is click-only, so what the widen actually
# does there is admit lower-ranked click targets, and those turn out to be
# worse than the random coordinates the old stuck path was probing.
EAGER_WIDEN = _env_int("ARC_EAGER_WIDEN", 0)
WIDEN_CLEARS_DEAD = _env_int("ARC_WIDEN_CLEARS_DEAD", 1)

# 15 of the 25 dev games never clear level 1 at all, and the diagnostic
# splits them cleanly: on ar25/sc25/sb26 the node count grows ~0.2-0.66 per
# action, so the reachable map never closes and the search degenerates into
# a random walk. These control the detector for that state and the momentum
# it switches on. Set ADAPTIVE_MOMENTUM to 0 to restore a single global
# MOMENTUM_LIMIT for A/B measurement.
# MEASURED: no effect. At limits 8 and 24, across seeds 101 and 202, the
# aggregate matched the baseline to within 0.0001 (0.1224 vs 0.1223, 0.1650
# vs 0.1651). Either the detector does not fire where it matters or momentum
# changes nothing where it does. Off by default; the detector is left in
# because `_exploding` is the cheapest available signal that the search has
# degenerated, and it costs one counter per action.
EXPLODE_WINDOW = _env_int("ARC_EXPLODE_WINDOW", 200)
EXPLODE_THRESHOLD = _env_float("ARC_EXPLODE_THRESHOLD", 0.4)
ADAPTIVE_MOMENTUM = _env_int("ARC_ADAPTIVE_MOMENTUM", 0)

VOLATILE_WARMUP = _env_int("ARC_VOLATILE_WARMUP", 25)
VOLATILE_THRESHOLD = _env_float("ARC_VOLATILE_THRESHOLD", 0.9)

# Everything sampled before the mask is adopted is charged to level 1 and
# then thrown away: adopting it re-keys every node, so the map built so far
# is discarded. Traced on r11l, whose level 1 costs 112 actions against a
# baseline of 22 -- 26 of them are spent before the mask fires, and the
# route that actually clears the level is 4 actions long.
#
# Lowering VOLATILE_WARMUP globally is not the answer; that sweep is known
# to be non-monotone (sp80: 79 actions at 25, 390 at 50, 78 at 100 and 400)
# and cn04 stopped clearing level 1 entirely when it was reduced. Fewer
# samples means a worse mask, and a wrong mask fuses distinct states.
#
# So fire early only where the evidence is not in doubt: cells that changed
# on EVERY transition so far, agreeing for several consecutive samples. A
# ticking timer qualifies immediately; a cell that merely changes often does
# not, and waits for the full statistical rule at VOLATILE_WARMUP. This can
# only ever move the decision earlier, never later, and never changes which
# mask the 25-sample rule would pick when the early rule declines to fire.
#
# MEASURED AND REJECTED. It does what it says -- r11l and vc33 adopt their
# masks at action 10 instead of 26 -- and the saved actions buy nothing:
#
#     game    off     on
#     vc33     99     66     better
#     ls20    526    842     worse
#     r11l    112    112     identical (nodes 42 -> 52)
#     cd82/m0r0/lp85         identical (no mask adopted either way)
#
# Both arms adopt the IDENTICAL mask; only the timing differs. So the ±60%
# swings are trajectory luck of the kind this file already documents, not
# merit, and r11l is the tell: 16 actions saved before the mask, 16 more
# spent exploring after it, same 112 total. What sets a level's cost is the
# number of probes needed to stumble on the exit, not when the map starts.
# Kept as a flag because the mechanism is correct and costs one comparison
# per sample; it just is not where the money is.
EARLY_MASK = _env_int("ARC_EARLY_MASK", 0)
EARLY_MASK_MIN = _env_int("ARC_EARLY_MASK_MIN", 6)
EARLY_MASK_STABLE = _env_int("ARC_EARLY_MASK_STABLE", 3)

# A HUD that updates every other step never reaches VOLATILE_THRESHOLD, and
# missing it is not a small loss: the ticking cells stay in the node key, so
# almost every action mints a new node and there is no map left to search.
# cd82 is the case in point -- no mask adopted, 1258 nodes in 2461 actions,
# on a level whose baseline is 55.
#
# The ladder is tried only when the strict rule yields nothing, and stops at
# the first threshold that finds a HUD. Games that already mask are
# unreachable by this code, which is the point: lowering the global
# threshold instead was measured and breaks them (sp80 78 -> 211 actions at
# 0.5, its mask going 64 cells -> 556).
#
# MEASURED AND SHIPPED, across all 25 dev games against the same baseline:
#
#   cd82   0.00 -> 0.19   2461 -> 274 actions on level 1, and it now clears
#                         two levels instead of one
#   tr87   0.00 -> 0.00   5001 -> 2538, reaching level 1 for the first time
#   23 games byte-identical
#   aggregate 0.1129 -> 0.1206  (+0.0076)
#
# The 23 identical games are the result that matters more than the +0.0076:
# they are the evidence that the guard holds, since every one of them adopts
# a mask under the strict rule and so cannot be reached from here. The only
# two games that moved are the two that had no mask at all.
MASK_FALLBACK = _env_int("ARC_MASK_FALLBACK", 1)
MASK_LADDER = tuple(
    float(x) for x in os.environ.get("ARC_MASK_LADDER", "0.7,0.5,0.3").split(",")
    if x.strip()
)

# Which rung to take when several find a HUD. "First" stops at the tightest
# threshold that sees anything; the alternative keeps descending and takes
# the loosest rung that still clears MASK_CAP_FRACTION -- a bigger mask, but
# only ever one the cap already approved.
#
# It matters more than it sounds. cd82 finds a 64-cell mask at 0.5 and a
# 1016-cell one at 0.3, worth 274 actions and 191 respectively -- 0.19
# against 0.40 -- so stopping at the first hit leaves half the gain behind.
#
# MEASURED across all 25 dev games, against the same baseline:
#
#   first hit (0.7/0.5/0.3)   aggregate 0.1206   cd82 274 actions, 0.19
#   loosest   (0.7/0.5/0.3)   aggregate 0.1287   cd82 191 actions, 0.40
#   single rung at 0.3        aggregate 0.1289   cd82 191 actions, 0.40
#
# So descending is worth +0.008 over stopping early, and 24 of 25 games are
# byte-identical to baseline either way -- cd82 is the only one that moves.
#
# The last two are the same score, and the ladder is preferred over the bare
# 0.3 because it degrades where a single rung cannot: re86's mask is over
# cap at 0.3 and it falls back to 0.50 (496 cells), which `0.3` alone would
# have missed entirely. Equal locally, more likely to find a usable mask on
# a game neither of us has seen.
#
# One prediction died here and is worth recording: tr87 was expected to keep
# the level-1 clear its 0.5 rung buys, on the theory that its 0.3 mask would
# be over cap. It is 1180 cells, 28.8%, comfortably inside the 35% cap -- so
# tr87 takes 0.3 and goes back to never finishing level 1. That costs no
# score (it is 0.00 either way) but it is a real regression against the
# first-hit rule, and the reason to keep MASK_CAP_FRACTION under review.
MASK_PICK_FIRST = _env_int("ARC_MASK_PICK_FIRST", 0)

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
# Diagnostics only. The budget report normally fires at level transitions,
# which means the games that never clear level 1 -- 15 of 25 on the dev set,
# and the whole of the remaining headroom -- are precisely the ones it never
# says anything about. Set this to e.g. 1000 to get a periodic breakdown.
# 0 keeps it silent, which is what ships.
BUDGET_REPORT_EVERY = _env_int("ARC_BUDGET_EVERY", 0)

FRAME_HISTORY = _env_int("ARC_FRAME_HISTORY", 8)
KEEP_RECORDINGS = _env_int("ARC_KEEP_RECORDINGS", 0)

# --- exploit/replay (see the SCORING section of the module docstring) --
#
# Transitions observed on the level currently being solved, kept so the
# shortest route through it can be recovered once it is beaten. This is
# separate from the exploration graph (`_edges`), which gets dropped
# whenever the search restarts; the solution graph must survive that.
# Bounded because a level that is never solved would otherwise grow it
# for the whole run. Counted in EDGES, not source nodes: one node can
# hold ~116 of them, so a node-based cap understates the real footprint
# by two orders of magnitude across 25 concurrent games.
SOLUTION_EDGE_CAP = _env_int("ARC_SOL_EDGE_CAP", 150_000)

# A replayed solution is re-verified as it executes. If the level does
# not fall exactly where the recorded route says it should, the route is
# stale (the state key is not fully Markovian for this game) and we drop
# back to exploring rather than burning the run on a dead plan.
REPLAY_STRICT = _env_int("ARC_REPLAY_STRICT", 1)

# The cursor is not the only thing that spends a node's candidates:
# replay, stuck-random, momentum and the model policy all take codes out
# of turn, and `_edges` records the outcome. Without this the cursor
# later walks back over those codes and pays a second billed action for
# an answer already in hand. Off by default only because it is a genuine
# behaviour change to the agent that scored 0.26 and every such change
# here gets measured before it ships; it is forced on when the policy is
# driving, where skipping is load-bearing rather than an optimisation.
SKIP_RECORDED = _env_int("ARC_SKIP_RECORDED", 0)

# --- the model policy ------------------------------------------------
#
# Everything above is a pure-Python search, and it scored 0.26. Its
# measured failure is not that it searches badly but that it searches
# *blind*: on the 15 dev games that never clear level 1, 950-980 of every
# 1000 actions are "probe-untried", drawn in an order that is arbitrary
# with respect to the game (simple actions shuffled on the node id, then
# clicks by size-and-rarity). That is a random walk with extra steps.
#
# So the model is wired in as a *ranker over the candidates the search
# already generated*, never as a free generator of actions. This is the
# whole safety argument:
#
#   - It cannot emit an illegal action. It returns an index into a list
#     we built, or nothing.
#   - Dead actions, the fatal set, the state graph, replay and the level
#     budget all keep working exactly as they do today.
#   - Every failure mode -- no GPU, no vLLM, OOM, a timeout, a garbage
#     reply -- lands on `None`, and `None` means "run the search that
#     already scored 0.26". The floor is structural, not a promise.
#
# Off by default: `ARC_POLICY=1` opts in. With it off, not one line of
# model code executes and the agent is byte-for-byte the 0.26 submission.
POLICY_ENABLED = _env_int("ARC_POLICY", 0)

# Verified present on the competition image; see the Kaggle mount layout
# note -- ARC-AGI-3 nests inputs under /kaggle/input/models/... rather
# than the flat /kaggle/input/<slug> used elsewhere on Kaggle.
POLICY_MODEL = os.environ.get(
    "ARC_POLICY_MODEL",
    "/kaggle/input/models/michaelpoluektov/qwen3-6-27b-fp8/transformers/default/1",
)

# 25 games run as 25 threads against one engine. Generating for them one
# at a time measured 0.4 actions/s; batching the 25 together measured
# 47.8. So requests are queued and served in batches, and these two
# numbers are the whole tuning surface: wait long enough to actually
# collect a batch, never long enough to stall a game thread.
POLICY_BATCH_WAIT = _env_float("ARC_POLICY_BATCH_WAIT", 0.05)
POLICY_BATCH_MAX = _env_int("ARC_POLICY_BATCH_MAX", 32)

# A game thread blocks on the model. If the engine wedges, this is what
# stops it taking the whole run down with it: the wait expires, the
# caller gets None, and the search carries on unaided.
POLICY_TIMEOUT = _env_float("ARC_POLICY_TIMEOUT", 45.0)

# We ask for an integer, so the reply is tiny. The allowance is for the
# <think> block this model opens unprompted -- see `parse_choice`, which
# strips it. Thinking is disabled via the chat template where supported,
# and this is the belt to that braces.
POLICY_MAX_TOKENS = _env_int("ARC_POLICY_MAX_TOKENS", 24)
POLICY_TEMPERATURE = _env_float("ARC_POLICY_TEMPERATURE", 0.0)

# How many of the search's candidates to show. The first few are the
# simple actions; the rest are clicks, already sorted smallest-and-rarest
# first, which is a real prior worth keeping. Showing all ~117 would bury
# the useful ones and cost tokens for no signal.
POLICY_CANDIDATES = _env_int("ARC_POLICY_CANDIDATES", 24)

# Collapse runs of identical rows to one labelled line. Lossless: the
# label carries the row range. Measured on a representative board, the
# rendered grid is 70% of the prompt and the prompt is ~1900 tokens, so
# 25 games prefill ~46k tokens per batch; these boards are mostly
# background, and collapsing them is the cheapest large saving available.
POLICY_COLLAPSE_ROWS = _env_int("ARC_POLICY_COLLAPSE", 1)

# Recent (action, did-the-board-move) pairs shown to the model. This is
# the only history it gets: the state graph is the agent's memory, not
# the model's, and re-serialising it every step would cost more tokens
# than the grid.
POLICY_HISTORY = _env_int("ARC_POLICY_HISTORY", 6)

# Engine sizing. 27B at FP8 is ~28GB of weights on a 95GB card, so the
# default leaves room for KV cache without crowding the box.
POLICY_GPU_FRACTION = _env_float("ARC_POLICY_GPU_FRACTION", 0.85)
POLICY_MAX_LEN = _env_int("ARC_POLICY_MAX_LEN", 8192)

# NEVER set this. enforce_eager=True disables CUDA-graph capture and
# measured 0.4 actions/s against 47.8 with capture on -- a 120x
# difference, on the same card, same model, same batch. It exists only
# because eager mode gives readable tracebacks when the engine will not
# start at all.
POLICY_EAGER = _env_int("ARC_POLICY_EAGER", 0)

# Consult the model at most every N actions. The search is free and the
# model is not; on a game that is making progress there is no reason to
# ask. 1 asks every step.
POLICY_EVERY = _env_int("ARC_POLICY_EVERY", 1)

# Stop asking after this many consecutive refusals to pick a live
# candidate. A model that cannot help this game should not keep being
# paid to say so.
POLICY_GIVE_UP = _env_int("ARC_POLICY_GIVE_UP", 40)

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
    random.Random((node ^ AGENT_SEED) & 0xFFFFFFFF).shuffle(simple)

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
# The model policy
# ---------------------------------------------------------------------
_HEX = "0123456789abcdefghijklmnopqrstuv"

# The model opens a reasoning block whether or not we ask for one. Both
# spellings appear, and an unclosed block is common when the token budget
# runs out mid-thought -- which is exactly when we still want an answer.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)
_FIRST_INT = re.compile(r"-?\d+")


def render_grid(
    grid: tuple[tuple[int, ...], ...], ignore: frozenset[int] = frozenset()
) -> str:
    """The frame as labelled text, one character per cell.

    Rulers matter more than they look. The model's whole job here is to
    connect something it can see in the grid to a CLICK(x, y) in the
    option list, and without coordinate labels it has to count columns --
    which it does badly, at 64 wide.

    Volatile cells render as '?'. Those are the HUD the agent already
    identified and excluded from the state key (see `_learn_mask`);
    showing them as real content invites the model to click a step
    counter, which is the single most common way a frame-differ gets
    fooled on this benchmark.

    Identical neighbouring rows collapse to one labelled line. This is
    lossless -- the label carries the range, so every coordinate is still
    recoverable -- and it is the difference between a prompt that fits
    the run's time budget and one that does not: the board is 70% of the
    prompt, 25 games are prefilled per batch, and these boards are mostly
    background. See POLICY_COLLAPSE_ROWS.
    """
    height = len(grid)
    if height == 0:
        return "(empty)"
    width = len(grid[0])
    if width == 0:
        return "(empty)"

    rows = []
    for y, row in enumerate(grid):
        base = y * width
        rows.append(
            "".join(
                "?" if (base + x) in ignore else _HEX[v] if v < len(_HEX) else "*"
                for x, v in enumerate(row)
            )
        )

    label = len(str(height - 1)) if not POLICY_COLLAPSE_ROWS else len(
        f"{height - 1}-{height - 1}"
    )
    pad = " " * label
    tens = "".join(str((x // 10) % 10) for x in range(width))
    ones = "".join(str(x % 10) for x in range(width))
    lines = [f"{pad}  {tens}", f"{pad}  {ones}"]

    y = 0
    while y < height:
        end = y
        if POLICY_COLLAPSE_ROWS:
            while end + 1 < height and rows[end + 1] == rows[y]:
                end += 1
        tag = str(y) if end == y else f"{y}-{end}"
        lines.append(f"{tag:>{label}}  {rows[y]}")
        y = end + 1
    return "\n".join(lines)


def describe_code(code: int, grid: tuple[tuple[int, ...], ...]) -> str:
    """One option line for the model.

    Simple actions are named but deliberately not *interpreted*. Calling
    ACTION1 "move up" would be a guess: the mapping differs per game and
    nothing in the frame data declares it. Telling the model a fact we do
    not have is worse than telling it nothing, because it will act on it.
    """
    action_id, x, y = decode_action(code)
    if action_id == _RESET_ID:
        return "RESET (restart the level)"
    if x is None or y is None:
        return f"ACTION{action_id} (unlabelled control)"
    colour = "?"
    if 0 <= y < len(grid) and 0 <= x < len(grid[y]):
        v = grid[y][x]
        colour = _HEX[v] if v < len(_HEX) else "*"
    return f"CLICK at column {x}, row {y} (cell colour {colour})"


def build_prompt(
    grid: tuple[tuple[int, ...], ...],
    codes: list[int],
    history: list[tuple[int, bool]],
    level: int,
    ignore: frozenset[int] = frozenset(),
) -> str:
    """Assemble the ranking prompt.

    A pure function of its arguments, so the tests can pin it exactly.
    """
    board = render_grid(grid, ignore)

    if history:
        seen = "\n".join(
            f"  {describe_code(c, grid)} -> "
            f"{'the board changed' if moved else 'nothing happened'}"
            for c, moved in history[-POLICY_HISTORY:]
        )
        recent = f"\nWhat I have tried most recently:\n{seen}\n"
    else:
        recent = ""

    options = "\n".join(
        f"  {i}: {describe_code(c, grid)}" for i, c in enumerate(codes)
    )

    return (
        "You are playing an unfamiliar puzzle video game. You see one "
        "frame at a time and choose one action.\n\n"
        "The board below is a grid of coloured cells. Each character is "
        "one cell; the character is that cell's colour as a digit. "
        "Column numbers are the two rows above the board (read them "
        "vertically: tens digit, then ones digit). Row numbers are down "
        "the left, and a label like '20-43' means every row from 20 to 43 "
        "is identical to the one shown. Cells shown as '?' are a "
        "scoreboard or timer, not part of the puzzle -- ignore them.\n\n"
        f"You are on level {level}.\n\n"
        f"{board}\n"
        f"{recent}\n"
        "Your goal is to work out the rule of this game and beat the "
        "level. Prefer an action that tests a real idea about the board "
        "over one that just changes something.\n\n"
        f"Options:\n{options}\n\n"
        "Reply with the number of the single best option. "
        "Reply with that number and nothing else."
    )


def parse_choice(text: Optional[str], count: int) -> Optional[int]:
    """The model's reply as an index into the option list, or None.

    Everything that is not a usable index is None: a refusal, an
    explanation, an out-of-range number, an empty string, a reply that is
    nothing but an unterminated thought. The caller treats None as "the
    model has no opinion" and searches instead, so being strict here
    costs one action of nothing and buys immunity from parsing garbage
    into a real move.
    """
    if not text or count <= 0:
        return None
    cleaned = _THINK_BLOCK.sub(" ", text)
    cleaned = _THINK_OPEN.sub(" ", cleaned)
    match = _FIRST_INT.search(cleaned)
    if match is None:
        return None
    try:
        index = int(match.group())
    except ValueError:
        return None
    if 0 <= index < count:
        return index
    return None


class _Request:
    """One prompt in flight, and the thread waiting on its answer."""

    __slots__ = ("prompt", "done", "reply", "abandoned")

    def __init__(self, prompt: str) -> None:
        self.prompt = prompt
        self.done = threading.Event()
        self.reply: Optional[str] = None
        self.abandoned = False


class _PolicyEngine:
    """One vLLM engine, shared by every game thread in the process.

    Two things make this more than a wrapper.

    *Loading happens exactly once.* All 25 game threads reach their first
    action at the same moment and every one of them would otherwise try
    to load 28GB of weights. The first thread through takes ~5 minutes;
    the rest wait on the same lock and then find it ready.

    That wait is deliberate rather than overlooked. Loading in the
    background would let the search play during those 5 minutes, but it
    costs no billed actions to wait -- only wall clock, ~1% of the
    budget -- and the framework imposes no per-action timeout (the Swarm
    runs plain daemon threads). Paying 1% to keep the failure path free
    of a load-in-flight race is the right trade for the one component
    whose entire job is to fail safely.

    *Requests are batched.* vLLM's throughput comes from serving many
    sequences together: measured 0.4 actions/s one at a time against 47.8
    in batches of 25, on the same card. Game threads therefore hand their
    prompt to a queue and block, and a single worker drains the queue in
    batches.

    Nothing here raises. A failure to load is recorded once and turns
    every later call into an immediate None.
    """

    _instance: Optional["_PolicyEngine"] = None
    _instance_lock = threading.Lock()

    @classmethod
    def get(cls) -> "_PolicyEngine":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self) -> None:
        self._llm: Any = None
        self._sampling: Any = None
        self._wrap: Any = None
        self._failed = False
        self._load_lock = threading.Lock()
        self._queue: "queue.Queue[_Request]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self.load_seconds = 0.0
        self.batches = 0
        self.prompts = 0

    @property
    def failed(self) -> bool:
        return self._failed

    def submit(self, prompt: str) -> Optional[str]:
        """Block until this prompt is answered. None on any failure."""
        if self._failed or not self._ensure():
            return None
        box = _Request(prompt)
        try:
            self._queue.put(box)
        except Exception:
            return None
        if not box.done.wait(POLICY_TIMEOUT):
            # Do not tear the engine down: a slow batch is not a broken
            # one, and the other 24 threads are probably in it. Mark the
            # request so the worker does not pay to finish it.
            box.abandoned = True
            logger.warning("policy: request timed out after %.0fs", POLICY_TIMEOUT)
            return None
        return box.reply

    def _ensure(self) -> bool:
        if self._llm is not None:
            return True
        if self._failed:
            return False
        with self._load_lock:
            if self._llm is not None:
                return True
            if self._failed:
                return False
            try:
                self._load()
            except BaseException as exc:      # noqa: BLE001 - see below
                # Deliberately the widest catch in this file. The entire
                # value of this class is that a model problem degrades to
                # the search that already scored, and vLLM failures are
                # not all Exceptions -- a failed engine bootstrap can
                # surface as SystemExit from a worker process. The one
                # thing we must not swallow is the user interrupting.
                if isinstance(exc, KeyboardInterrupt):
                    raise
                logger.exception("policy: engine unavailable, using search only")
                self._failed = True
                return False
        return True

    def _load(self) -> None:
        from vllm import LLM, SamplingParams

        logger.info("policy: loading %s", POLICY_MODEL)
        started = time.monotonic()
        llm = LLM(
            model=POLICY_MODEL,
            gpu_memory_utilization=POLICY_GPU_FRACTION,
            max_model_len=POLICY_MAX_LEN,
            # Consecutive steps of one game share almost the whole prompt
            # -- system text, rulers, and every unchanged row of the
            # board -- so this is close to free and large.
            enable_prefix_caching=True,
            enforce_eager=bool(POLICY_EAGER),
            trust_remote_code=True,
        )
        self._sampling = SamplingParams(
            temperature=POLICY_TEMPERATURE,
            max_tokens=POLICY_MAX_TOKENS,
        )
        self._wrap = self._build_wrapper(llm)
        self.load_seconds = time.monotonic() - started

        worker = threading.Thread(
            target=self._serve, name="policy-batcher", daemon=True
        )
        worker.start()
        self._worker = worker
        # Published last and only on success: `_ensure` tests this, so
        # until it is set every thread takes the loading path.
        self._llm = llm
        logger.info("policy: ready in %.0fs", self.load_seconds)

    @staticmethod
    def _build_wrapper(llm: Any) -> Any:
        """Return prompt -> model-ready text.

        Chat models need their template applied or they answer as if
        completing a document. `enable_thinking` is a Qwen extension; it
        is probed once here rather than per call, and a template that
        rejects it still works without it.
        """
        try:
            tok = llm.get_tokenizer()
        except Exception:
            logger.warning("policy: no tokenizer, sending raw prompts")
            return lambda text: text

        def apply(text: str, thinking: bool) -> str:
            kwargs = {} if thinking else {"enable_thinking": False}
            return tok.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
                **kwargs,
            )

        try:
            apply("ping", thinking=False)
            return lambda text: apply(text, thinking=False)
        except Exception:
            pass
        try:
            apply("ping", thinking=True)
            logger.info("policy: template has no thinking switch")
            return lambda text: apply(text, thinking=True)
        except Exception:
            logger.warning("policy: no chat template, sending raw prompts")
            return lambda text: text

    def _serve(self) -> None:
        while True:
            try:
                batch = [self._queue.get()]
            except Exception:
                return
            deadline = time.monotonic() + POLICY_BATCH_WAIT
            while len(batch) < POLICY_BATCH_MAX:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    batch.append(self._queue.get(timeout=remaining))
                except queue.Empty:
                    break
            self._run(batch)

    def _run(self, batch: list[_Request]) -> None:
        live = [box for box in batch if not box.abandoned]
        try:
            if live:
                outputs = self._llm.generate(
                    [self._wrap(box.prompt) for box in live],
                    self._sampling,
                    use_tqdm=False,
                )
                # vLLM returns outputs in submission order.
                for box, out in zip(live, outputs):
                    if out.outputs:
                        box.reply = out.outputs[0].text
                self.batches += 1
                self.prompts += len(live)
        except BaseException as exc:          # noqa: BLE001
            if isinstance(exc, KeyboardInterrupt):
                raise
            # One bad batch is not a dead engine -- a single over-long
            # prompt can do this -- so the engine is left up and the
            # callers simply get None for this round.
            logger.exception("policy: batch of %d failed", len(live))
        finally:
            # Every waiter is released on every path. Missing one here
            # would park a game thread until its timeout, every step.
            for box in batch:
                box.done.set()


class VisionPolicy:
    """Ranks the search's candidates with the model. Never raises.

    Held per game, so `asked`/`used` read as per-game diagnostics, but
    the engine underneath is process-wide.
    """

    def __init__(self, game_id: str) -> None:
        self.game_id = game_id
        self.asked = 0
        self.used = 0
        self.refused = 0
        self._streak = 0
        self._retired = False

    @property
    def retired(self) -> bool:
        return self._retired

    def rank(
        self,
        grid: tuple[tuple[int, ...], ...],
        codes: list[int],
        history: list[tuple[int, bool]],
        level: int,
        ignore: frozenset[int] = frozenset(),
    ) -> Optional[int]:
        """Index into `codes`, or None to let the search decide."""
        if self._retired or not codes:
            return None
        engine = _PolicyEngine.get()
        if engine.failed:
            self._retired = True
            return None

        shown = codes[:POLICY_CANDIDATES]
        self.asked += 1
        try:
            prompt = build_prompt(grid, shown, history, level, ignore)
        except Exception:
            logger.exception("%s: policy could not build a prompt", self.game_id)
            return None

        choice = parse_choice(engine.submit(prompt), len(shown))
        if choice is None:
            self.refused += 1
            self._streak += 1
            if POLICY_GIVE_UP and self._streak >= POLICY_GIVE_UP:
                logger.info(
                    "%s: policy gave no usable answer %d times running; "
                    "searching unaided from here",
                    self.game_id, self._streak,
                )
                self._retired = True
            return None

        self._streak = 0
        self.used += 1
        return choice


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
        #
        # crc32, not hash(): Python salts str hashing per process (PEP 456),
        # so `hash(self.game_id)` seeded this differently on every single
        # run. The agent looked deterministic -- candidate order is a pure
        # function of the grid -- while the `stuck-random` path silently
        # drew from a different stream each time. That made every A/B a
        # comparison of two configs at two uncontrolled seeds: measured on
        # lp85, whose level 1 spends 21% of its actions on `stuck-random`,
        # the same code took 51, 67 and 235 actions on three runs. crc32 is
        # stable across processes, so a run is now reproducible.
        self._rng = random.Random(zlib.crc32(self.game_id.encode()) & 0xFFFFFFFF)

        # --- the learned state graph, two ints + edges per node ------
        self._total: dict[int, int] = {}          # node -> candidate count
        self._cursor: dict[int, int] = {}         # node -> next index to try
        self._breadth = BREADTH_INITIAL           # candidates per node, deepens
        self._why_counts: Counter[str] = Counter()  # billed actions by reason
        self._class_counts: Counter[str] = Counter()  # click/press hit rate
        self._last_code: Optional[int] = None     # code of the previous action
        self._last_productive = False             # ...and did it reach a new state
        self._momentum = 0                        # consecutive repeats so far
        self._act_tried: Counter[int] = Counter()    # action id -> probes
        self._act_changed: Counter[int] = Counter()  # action id -> state changes
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
        # Early-fire bookkeeping (see EARLY_MASK): the last unanimous-change
        # mask proposed, and how many consecutive samples have agreed on it.
        self._early_candidate: frozenset[int] = frozenset()
        self._early_stable = 0

        # --- execution bookkeeping ----------------------------------
        # (node, code) awaiting its outcome, folded in on the next call
        self._pending: Optional[tuple[int, int]] = None
        # payload handed to the env; see do_action_request()
        self._pending_data: dict[str, Any] = {"game_id": ""}
        self._pending_reasoning: Optional[dict[str, Any]] = None
        # queued route to a frontier node: [(node_we_must_be_in, code), ...]
        self._plan: deque[tuple[int, int]] = deque()

        # Rolling estimate of how often an action lands on a node never seen
        # before. Near 1.0 the map is not closing and the search has
        # degenerated into a random walk; see EXPLODE_THRESHOLD.
        self._novel_steps = 0
        self._novel_new = 0
        self._exploding = False

        # --- the model policy (see POLICY_ENABLED) --------------------
        # None means "no model in this run", and every call site is
        # written so that None is the ordinary case, not an error path.
        self._policy = VisionPolicy(self.game_id) if POLICY_ENABLED else None
        # (code, did-the-board-change) for the last few actions, which is
        # all the history the model is given.
        self._act_history: list[tuple[int, bool]] = []

        self._best_levels = 0
        self._stuck_steps = 0
        self._steps = 0
        self._recorded = 0
        self._graph_full = False
        # Last raw frame and its conversion; see _convert_raw_frame_data.
        self._raw_seen: Any = None
        self._frame_seen: Optional[FrameData] = None

        # --- exploit/replay ------------------------------------------
        # Shortest known action sequence for each level index, harvested
        # when that level was first beaten.
        self._solutions: dict[int, list[int]] = {}
        # Transitions seen on the level being solved now, rooted at
        # `_sol_start`. Survives `_reset_graph()`; cleared per level.
        self._sol_edges: dict[int, dict[int, int]] = {}
        self._sol_start: Optional[int] = None
        self._sol_full = False
        self._sol_count = 0
        # Live level index (drops back to 0 on a full reset), as opposed
        # to `_best_levels`, which is a high-water mark.
        self._level = 0
        # Queued RESETs used to open a fresh scored run, then the banked
        # route being replayed into it.
        self._resets_queued = 0
        self._replay: deque[int] = deque()
        # Level the replay expects to be on, so a stale route is caught
        # the moment reality diverges.
        self._replay_level = 0
        self._cycles = 0
        # Set when a level is banked for the FIRST time, which is exactly
        # when a fresh run would score better than the one we are in.
        # Re-banking the same level during a replay must not set it, or
        # the agent would cycle forever without ever exploring.
        self._want_cycle = False

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
        # 4096 cells per action, and under the GIL every agent thread is
        # competing for one core: chain.from_iterable does this in C,
        # where the equivalent generator expression cost ~8% of total run
        # time in 4096 x N interpreter steps.
        flat = tuple(chain.from_iterable(grid))

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
            # Every node id recorded so far was keyed on different cells,
            # so the route through the current level is unrecoverable too.
            # Banked routes are plain action sequences and stay valid.
            self._sol_start = None
            self._sol_edges = {}

        node = self._node_key(flat)

        if BUDGET_REPORT_EVERY and self._steps % BUDGET_REPORT_EVERY == 0:
            self._report_budget(f"still on level {self._level}")

        # 1. Learn from the action taken last call, whose outcome is
        #    exactly the frame we are being handed now.
        self._observe(node, latest_frame)

        # 2. A dead or unstarted game can only be RESET. Dying mid-replay
        #    means the banked route no longer applies, so drop it.
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._plan.clear()
            self._replay.clear()
            self._pending = None
            return self._emit(_RESET_ID, None, None, "reset")

        # 3. Finish opening a fresh scored run. Two RESETs in a row always
        #    produce one (see the SCORING section); the first may only
        #    reset the level, but it zeroes the engine's action counter,
        #    which is what makes the second a full reset.
        if self._resets_queued > 0:
            self._resets_queued -= 1
            self._plan.clear()
            self._pending = None
            if self._resets_queued == 0:
                # The game is about to jump back to the start of level 0.
                # If we were already on level 0 the level counter does not
                # move, so `_observe` would not re-root the solution graph
                # and it would still point at a node from the old run.
                self._sol_start = None
                self._sol_edges = {}
                self._sol_count = 0
            return self._emit(_RESET_ID, None, None, "full-reset")

        # 4. Replay a banked route, validating each step against the level
        #    it was recorded on before trusting it.
        while self._replay:
            expected, code = self._replay[0]
            if REPLAY_STRICT and expected != self._level:
                logger.info(
                    "%s: replay diverged at level %d (expected %d); "
                    "dropping the route and exploring",
                    self.game_id, self._level, expected,
                )
                self._replay.clear()
                break
            self._replay.popleft()
            self._pending = (node, code)
            action_id, x, y = decode_action(code)
            return self._emit(action_id, x, y, "replay")

        # 5. Decide.
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
            KEEP_RECORDINGS
            and self._recorded < KEEP_RECORDINGS
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

    def _convert_raw_frame_data(self, raw: Any) -> FrameData:
        """Convert once per raw frame, not twice.

        `Agent.main()` converts `arc_env.observation_space` to build the
        `latest_frame` argument, and `do_action_request` converts the
        object `step()` returned. `EnvironmentWrapper.step()` stores that
        very object as `_last_response`, which `observation_space` hands
        back -- so the two calls are given the identical raw frame and
        rebuild the identical model.

        That rebuild is not cheap: it copies 4096 cells out of numpy and
        pays pydantic validation, together ~20% of the run. Memoizing the
        last one halves it. The check is identity against the retained
        object rather than `id()`, so a freed frame's address cannot be
        reused by a later one and alias to a stale conversion.
        """
        if raw is not None and self._raw_seen is raw:
            return self._frame_seen
        converted = super()._convert_raw_frame_data(raw)
        self._raw_seen = raw
        self._frame_seen = converted
        return converted

    # -- internals ----------------------------------------------------
    def _emit(
        self, action_id: int, x: Optional[int], y: Optional[int], why: str
    ) -> GameAction:
        if x is None:
            self._pending_data = {"game_id": ""}
        else:
            self._pending_data = {"game_id": "", "x": x, "y": y}
        self._pending_reasoning = {"why": why}
        self._why_counts[why] += 1
        return GameAction.from_id(action_id)

    def _report_budget(self, tag: str) -> None:
        """Log where the level's billed actions actually went.

        Under competition rules every action is charged against the level
        being scored, so this breakdown is the only honest guide to what to
        optimise -- guessing produced a 'fix' that changed nothing.
        """
        if not self._why_counts:
            return
        mix = " ".join(
            f"{why}={n}" for why, n in self._why_counts.most_common()
        )
        clicks = self._class_counts["click"]
        presses = self._class_counts["press"]
        billed = clicks + presses
        hit = self._class_counts["click-live"] / clicks if clicks else 0.0
        logger.info(
            "%s: %s after %d actions (breadth=%d, nodes=%d) | %s "
            "|| clicks=%d (%.0f%% of spend, %.1f%% moved) presses=%d (%.1f%% moved)",
            self.game_id, tag, self._steps, self._breadth, len(self._total), mix,
            clicks, 100.0 * clicks / billed if billed else 0.0, 100.0 * hit,
            presses,
            100.0 * self._class_counts["press-live"] / presses if presses else 0.0,
        )
        self._why_counts.clear()
        self._class_counts.clear()

    def _observe(self, node: int, latest_frame: FrameData) -> None:
        """Fold the outcome of the previous action into the graph."""
        pending, self._pending = self._pending, None
        levels = latest_frame.levels_completed

        # First frame of the run, or the state key just changed under us
        # when the HUD mask was adopted: root the solution graph here.
        if self._sol_start is None:
            self._begin_level(node)

        if levels != self._level:
            self._report_budget(f"left level {self._level} for {levels}")
            if levels > self._level and pending is not None:
                # The action we are being shown the result of is the one
                # that beat the level, so it is the last step of the route.
                self._harvest(self._level, pending)
            elif levels < self._level:
                # Only a full reset moves the counter backwards, and it
                # puts us back at the start of level 0.
                self._cycles += 1
            self._level = levels
            self._begin_level(node)

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

        # The solution graph records unconditionally -- including during a
        # replay, and including nodes the explorer never budgeted -- because
        # its only job is to make the route through this level recoverable.
        if not self._sol_full:
            outgoing = self._sol_edges.setdefault(prev_node, {})
            if code not in outgoing:
                outgoing[code] = node
                self._sol_count += 1
                if self._sol_count >= SOLUTION_EDGE_CAP:
                    self._sol_full = True
                    logger.info(
                        "%s: solution-graph cap reached at action %d",
                        self.game_id, self._steps,
                    )

        if prev_node in self._total:
            self._edges.setdefault(prev_node, {})[code] = node

        # A code that just moved us somewhere genuinely new is the single
        # best hint available about which direction this game rewards.
        # Systematic search discards that hint the moment it lands, and on a
        # movement game that means re-deriving the same direction at every
        # node along a corridor.
        # `_total` gains `node` later, in `_candidates`, so "not in `_total`"
        # is exactly "never expanded before" -- no extra bookkeeping needed.
        self._last_code = code
        self._last_productive = node != prev_node and node not in self._total

        # The model's entire memory of this game. `node != prev_node` is
        # "the board changed", which is the one thing about an action a
        # frame-differ can assert without interpreting the game. Kept
        # short deliberately: the agent's memory is the state graph, and
        # re-serialising that every step would cost more tokens than the
        # board does.
        if self._policy is not None:
            self._act_history.append((code, node != prev_node))
            if len(self._act_history) > POLICY_HISTORY:
                del self._act_history[:-POLICY_HISTORY]

        # Is the map closing at all? Measured on the dev set: ar25 creates
        # 0.66 new nodes per action and sc25 0.60, versus ft09's 0.02. Above
        # the threshold there is no graph to search -- every node is fresh,
        # so the cursor is always 0 and "probe-untried" just takes a
        # node-hash-seeded action. That is a random walk, and a random walk
        # covers ~sqrt(n) of a grid where a persistent one covers ~n.
        self._novel_steps += 1
        if self._last_productive:
            self._novel_new += 1
        if self._novel_steps >= EXPLODE_WINDOW:
            self._exploding = (
                self._novel_new / self._novel_steps >= EXPLODE_THRESHOLD
            )
            self._novel_steps = 0
            self._novel_new = 0

        action_id = code >> (2 * _COORD_BITS)
        self._act_tried[action_id] += 1
        if node != prev_node:
            self._act_changed[action_id] += 1

        # Split the level's spend by action class. ~112 of the ~117 codes a
        # node offers are ACTION6 clicks, so if clicks dominate the bill and
        # almost none of them move the state, the branching factor -- not the
        # search policy -- is what puts a near-baseline clear out of reach.
        kind = "click" if action_id == _ACTION6_ID else "press"
        self._class_counts[kind] += 1
        if node != prev_node:
            self._class_counts[kind + "-live"] += 1

    def _begin_level(self, node: int) -> None:
        """Root a fresh solution graph at the state the level starts in."""
        self._sol_edges = {}
        self._sol_start = node
        self._sol_full = False
        self._sol_count = 0
        # Each level is scored on its own actions, so each level gets its own
        # cheap-first search. Carrying a widened breadth across a level
        # boundary would spend the new level's budget at the old level's
        # difficulty.
        self._breadth = BREADTH_INITIAL
        # The dead-action record deliberately survives the level boundary.
        # Which actions a game wires up is a property of the *game*, not of
        # the level, so clearing it made every level re-pay the cost of
        # discovering that the same buttons are still inert -- and deeper
        # levels are where the points are, since level i carries weight i.
        #
        # Measured: tu93 cleared level 1 at 42.7x baseline and level 2 at
        # 11.8x, so what is learned early is exactly what makes later
        # levels cheap. Re-learning it each time taxes the levels that
        # matter most.
        #
        # The risk is a level introducing a mechanic on an action that was
        # inert before, leaving it suppressed. That is already covered: an
        # action that has *ever* changed anything is never written off, and
        # the stall path widens breadth and wipes both counters -- which
        # fires precisely when the agent is stuck, the symptom of having
        # suppressed something it needed.

    def _harvest(self, level: int, winning: tuple[int, int]) -> None:
        """Extract and bank the shortest route through a level just beaten.

        `winning` is the (node, code) that ended it. The route is the
        shortest path through the observed transitions from the level's
        start node to that node, plus the winning action itself. Keeping
        only this sequence -- not the graph -- is what bounds memory.
        """
        prev_node, code = winning
        start = self._sol_start
        if start is None:
            return

        path = self._shortest_path(start, prev_node)
        if path is None:
            return
        route = path + [code]

        previous = self._solutions.get(level)
        if previous is not None and len(previous) <= len(route):
            return
        if previous is None:
            self._want_cycle = True
        self._solutions[level] = route
        logger.info(
            "%s: banked a %d-action route through level %d (action %d)",
            self.game_id, len(route), level + 1, self._steps,
        )

    def _shortest_path(self, start: int, goal: int) -> Optional[list[int]]:
        """BFS the solution graph for the fewest actions from start to goal."""
        if start == goal:
            return []
        queue: deque[int] = deque([start])
        came_from: dict[int, Optional[tuple[int, int]]] = {start: None}

        while queue:
            current = queue.popleft()
            for code, nxt in self._sol_edges.get(current, {}).items():
                if nxt in came_from:
                    continue
                came_from[nxt] = (current, code)
                if nxt == goal:
                    route: list[int] = []
                    cursor: Optional[int] = goal
                    while True:
                        edge = came_from[cursor]
                        if edge is None:
                            break
                        prev, taken = edge
                        route.append(taken)
                        cursor = prev
                    route.reverse()
                    return route
                queue.append(nxt)
        return None

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
            if not EARLY_MASK or self._samples < EARLY_MASK_MIN:
                return False
            # Unanimous change only: threshold 1.0 means "this cell (or its
            # row/column) moved on every single transition we have seen".
            # A timer or a frame counter clears that bar at once; a cell
            # that is merely busy does not, and falls through to the
            # statistical rule below at VOLATILE_WARMUP.
            candidate = hud_mask(
                counts, self._row_changes, self._col_changes,
                self._samples, width, height, threshold=1.0,
            )
            if not candidate or candidate != self._early_candidate:
                self._early_candidate = candidate
                self._early_stable = 0
                return False
            self._early_stable += 1
            if self._early_stable < EARLY_MASK_STABLE:
                return False
            # Converged. Adopt it and stop sampling.
            self._mask_frozen = True
            self._cell_changes = []
            self._row_changes = []
            self._col_changes = []
            self._prev_flat = None
            self._volatile = candidate
            self._kept = tuple(
                i for i in range(len(flat)) if i not in candidate
            )
            return True

        self._mask_frozen = True
        volatile = hud_mask(
            counts, self._row_changes, self._col_changes,
            self._samples, width, height,
        )

        # Nothing cleared VOLATILE_THRESHOLD. On a game with a HUD that
        # updates every *other* step that is a miss, not an absence, and the
        # cost is total: the ticking cells stay in the state key, so nearly
        # every action mints a new node and the map never closes. cd82 spends
        # 2461 actions on a level whose baseline is 55, with 1258 nodes to
        # show for it.
        #
        # Relaxing the threshold globally is not the answer -- measured, it
        # breaks the games that already work. At 0.5 sp80 masks 556 cells
        # instead of 64 and goes 78 -> 211 actions (1.19 -> 0.16); at 0.3 it
        # only recovers because the larger mask trips MASK_CAP_FRACTION and
        # is discarded, which is luck rather than a design.
        #
        # So relax ONLY where the strict rule found nothing at all. Any game
        # that already adopts a mask is untouched by construction, and the
        # ones this can reach are precisely those whose maps are exploding
        # today for want of one.
        if not volatile and MASK_FALLBACK:
            chosen: frozenset[int] = frozenset()
            chosen_at = 0.0
            for threshold in MASK_LADDER:
                candidate = hud_mask(
                    counts, self._row_changes, self._col_changes,
                    self._samples, width, height, threshold=threshold,
                )
                # An over-cap mask comes back empty, so a non-empty
                # candidate has already cleared MASK_CAP_FRACTION.
                if not candidate:
                    continue
                chosen, chosen_at = candidate, threshold
                if MASK_PICK_FIRST:
                    break
            if chosen:
                logger.info(
                    "%s: no HUD at threshold %.2f; falling back to %.2f "
                    "(%d cells)",
                    self.game_id, VOLATILE_THRESHOLD, chosen_at, len(chosen),
                )
                volatile = chosen

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

    def _is_dead(self, code: int) -> bool:
        """Has this action id had a fair trial and never changed anything?

        Judged per action *id*, never per code: a click that misses is
        uninformative, but ACTION6 doing nothing anywhere after a real
        sample means this game does not use clicks at all, and skipping
        them frees ~112 of the ~117 codes a node offers.

        What counts as "fair" differs by an order of magnitude between a
        global button and a click; see the threshold constants.
        """
        action_id = code >> (2 * _COORD_BITS)
        trials = DEAD_CLICK_TRIALS if action_id == _ACTION6_ID else DEAD_ACTION_TRIALS
        return (
            self._act_tried[action_id] >= trials
            and not self._act_changed[action_id]
        )

    def _has_untried(self, node: int) -> bool:
        return self._cursor.get(node, 0) < min(
            self._total.get(node, 0), self._breadth
        )

    def _select(
        self, node: int, grid: tuple[tuple[int, ...], ...], latest_frame: FrameData
    ) -> tuple[int, str]:
        # (0) A level was just banked for the first time. The run we are in
        #     paid full exploration price for it; a fresh one replaying the
        #     banked route pays near-baseline, and only the best run counts.
        if self._want_cycle and self._replayable():
            self._want_cycle = False
            self._start_cycle()
            return encode_action(_RESET_ID, None, None), "exploit-cycle"

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

        # (b0) Momentum. If the last action landed us somewhere new, try it
        #      again before anything else. Corridors, pushes and falls all
        #      reward repetition, and this costs nothing when it is wrong:
        #      the outcome is recorded either way, and `_edges` already
        #      holds the answer for any code tried from this node, so we
        #      never pay twice for the same probe.
        # Momentum is off by default because on a game whose map *does*
        # close it conflates motion with progress -- measured at 668 actions
        # against 78 on sp80. But that argument only holds where there is a
        # graph to search. On an exploding map there is none, and repeating
        # a direction is what separates a persistent walk from a random one.
        momentum_limit = (
            ADAPTIVE_MOMENTUM
            if (self._exploding and ADAPTIVE_MOMENTUM)
            else MOMENTUM_LIMIT
        )
        if (
            self._last_productive
            and self._momentum < momentum_limit
            and self._last_code is not None
            and self._last_code not in self._edges.get(node, EMPTY_EDGES)
            and (node, self._last_code) not in self._fatal
        ):
            self._momentum += 1
            self._stuck_steps = 0
            return self._last_code, "momentum"
        self._momentum = 0

        # (b1) Ask the model which candidate to spend the next probe on.
        #
        #      This is the only point where a model influences anything,
        #      and it influences exactly one thing: the order in which
        #      the search spends probes it was going to spend anyway. It
        #      picks an index into a list we built and filtered, so it
        #      cannot propose an illegal, dead, fatal or already-tried
        #      action, and `None` -- no model, no GPU, a timeout, a
        #      garbage reply -- simply falls through to (b) below.
        #
        #      Why order is the right lever: on the 15 dev games that
        #      never clear level 1, 950-980 of every 1000 actions are
        #      "probe-untried", and their order is arbitrary with respect
        #      to the game. The search is not losing because it searches
        #      badly; it is losing because it cannot see the board.
        #
        #      The window is capped at the breadth the search would have
        #      used, widened only far enough to give the model a real
        #      menu. Letting it range over all ~117 candidates would
        #      change how much the agent spends per node, and the point
        #      of this experiment is to change only *which* probe is
        #      first.
        if (
            self._policy is not None
            and not self._policy.retired
            and node in self._total
            and (POLICY_EVERY <= 1 or self._steps % POLICY_EVERY == 0)
        ):
            tried = self._edges.get(node, EMPTY_EDGES)
            window = max(self._breadth, POLICY_CANDIDATES)
            live = [
                code
                for code in codes[:window]
                if code not in tried
                and not self._is_dead(code)
                and (node, code) not in self._fatal
            ]
            if live:
                pick = self._policy.rank(
                    grid, live, self._act_history, self._level, self._volatile
                )
                if pick is not None:
                    self._stuck_steps = 0
                    return live[pick], "policy"

        # (b) Anything untried right here is the cheapest information
        #     available, so take it -- skipping action ids this game has
        #     shown it does not respond to. Skipping only ever advances the
        #     cursor, so "index i means action i" still holds and the graph
        #     bookkeeping stays valid.
        #
        #     Already-recorded codes are skipped for the same reason. The
        #     cursor is not the only thing that spends a node's
        #     candidates -- momentum and (b1) both take codes out of turn
        #     -- and without this the cursor walks back over them and
        #     pays a second time for an outcome `_edges` already holds.
        #     Gated, because it is a real behaviour change and not only a
        #     policy concern: replay and stuck-random already record
        #     edges out of turn today. It is forced on whenever a model
        #     is driving, where it is load-bearing, and otherwise defaults
        #     off so the no-model agent stays exactly the one that scored.
        limit = min(len(codes), self._breadth)
        index = self._cursor.get(node, 0)
        recorded = (
            self._edges.get(node, EMPTY_EDGES)
            if (SKIP_RECORDED or self._policy is not None)
            else EMPTY_EDGES
        )
        while index < limit and (
            self._is_dead(codes[index]) or codes[index] in recorded
        ):
            index += 1
        if index < limit and node in self._total:
            self._cursor[node] = index + 1
            self._stuck_steps = 0
            return codes[index], "probe-untried"
        self._cursor[node] = index

        # (c) This node is exhausted. Walk to the nearest node that
        #     isn't, using only edges already verified by experience.
        route = self._route_to_frontier(node)
        if route:
            self._plan = route
            _, code = self._plan.popleft()
            self._stuck_steps = 0
            return code, "route-to-frontier"

        # (d) Every reachable node is exhausted *at the current breadth*.
        #
        #     Widening is the only response here that costs zero billed
        #     actions: cursors stay valid because candidate order is a pure
        #     function of the grid, so index i still means action i, and
        #     every node we shallow-probed reopens at once. It therefore has
        #     to come before anything that spends an action.
        #
        #     It used to come *after* STUCK_REEXPLORE_AFTER, so the agent
        #     paid up to 1500 actions of "stuck-random" -- the least
        #     informative move available -- before trying the free one.
        #     Measured on lp85, whose level 1 has a baseline of 17: 369 of
        #     its 502 level-1 actions were stuck-random. Under a quadratic
        #     score those actions are not merely wasted, they are the
        #     difference between ~30x and ~3x baseline.
        if EAGER_WIDEN and self._breadth < BREADTH_MAX:
            self._breadth = min(self._breadth * 2, BREADTH_MAX)
            self._stuck_steps = 0
            # Give written-off actions another chance: an action that did
            # nothing everywhere we had been may simply have needed a
            # precondition we had not reached yet.
            #
            # But this used to fire only after 1500 stuck actions, i.e.
            # almost never. Widening eagerly makes it fire early and often,
            # which throws away the dead-action record precisely when it is
            # most valuable. Measured across 3 seeds, eager widening cost
            # tn36 52 -> 86 actions identically every time; wiping the
            # record is the suspect, since it re-admits click ids the game
            # has already shown to be inert.
            if WIDEN_CLEARS_DEAD:
                self._act_tried.clear()
                self._act_changed.clear()
            logger.info(
                "%s: map exhausted after %d actions; widening breadth to %d",
                self.game_id, self._steps, self._breadth,
            )
            return self._select(node, grid, latest_frame)

        # Genuinely exhausted: every node probed to BREADTH_MAX.
        self._stuck_steps += 1

        # Measured on ls20: the reachable map closes after ~1.6k actions
        # and without this the remaining 34k actions all went to
        # "stuck-random", i.e. 86% of the budget spent flailing. Dropping
        # the map re-runs the systematic search instead, and because the
        # fatal set is cleared with it, edges wrongly pruned by a
        # step-budget death get another chance.
        if self._stuck_steps >= STUCK_REEXPLORE_AFTER:
            # With routes banked, the far better move than re-exploring in
            # place is to open a fresh run and replay into the frontier
            # level: it costs the same restart but banks every level below
            # at near-baseline action counts, which is where the score is.
            if self._replayable():
                self._start_cycle()
                return encode_action(_RESET_ID, None, None), "exploit-cycle"

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

    def _banked_route(self) -> list[tuple[int, int]]:
        """The contiguous replayable prefix, as [(level, code), ...].

        A gap matters: level 3's route starts from wherever level 2 left
        off, so it is only valid if every level below it is replayed too.
        """
        route: list[tuple[int, int]] = []
        level = 0
        while level in self._solutions:
            route.extend((level, code) for code in self._solutions[level])
            level += 1
        return route

    def _replayable(self) -> bool:
        # Gating here disables both cycle entry points at once: the
        # post-harvest branch and the exhausted-map branch each require a
        # replayable route before they will spend a RESET.
        return bool(ENABLE_CYCLE) and 0 in self._solutions

    def _start_cycle(self) -> None:
        """Open a fresh scored run and queue the banked route into it."""
        route = self._banked_route()
        self._reset_graph()
        self._stuck_steps = 0
        self._replay = deque(route)
        # One RESET is emitted by the caller; this is the second, which is
        # the one that actually opens the new run.
        self._resets_queued = 1
        logger.info(
            "%s: exploit cycle %d -- replaying %d actions to reach level %d",
            self.game_id, self._cycles + 1, len(route), len(self._solutions),
        )

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
