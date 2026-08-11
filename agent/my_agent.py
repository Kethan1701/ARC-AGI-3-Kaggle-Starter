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

import base64
import json
import logging
import os
import queue
import random
import re
import struct
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
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

# EAGER_WIDEN above was rejected for widening everywhere, including on games
# where the first 12 candidates are the right ones. But there are games where
# the window is demonstrably wrong, and they can be told apart from the
# others while the run is happening.
#
# Measured by recording the candidate rank of every action taken:
#
#   game  clicks  that moved   median rank of those   breadth
#   ft09     677     3 (0.4%)          83  (max 91)      12
#   lp85     364        (2%)   cleared by stuck-random,
#                              i.e. a coordinate not in the list at all
#   sb26    1366       (65%)          10                 12
#   s5i5     204        (7%)           0                 12
#
# ft09's productive clicks sit at ranks 83-91 and the window is 12, so the
# search cannot reach them at all; it does eventually widen to 96 and find
# them, thousands of actions later, via STUCK_REEXPLORE_AFTER. tn36, the
# game eager widening broke, moves the board on 43% of its clicks -- so
# gating on click liveness leaves it alone by construction.
#
# Note this does NOT contradict the finding that the click prior ranks
# level-clearing actions at 0-6: that held on r11l, sp80, vc33 and tn36,
# every one of which clears level 1. On the games that never clear, the
# picture inverts. Both are true of the games they were measured on.
#
# Measured over all 25 dev games (aggregate 0.1288 -> 0.1299). What the
# reasoning above got right and wrong, recorded so the next change is not
# argued from the same priors:
#
#   ft09  FALSIFIED. It is the game this was designed for, and the gate
#         fired exactly as intended -- widened 24 -> 48 -> 96, reaching
#         the rank-83-91 clicks. Still 5001 actions, 0 levels. Reaching a
#         productive click is evidently not sufficient to use one, so
#         "the window cannot see it" was not ft09's binding constraint.
#   su15  unpredicted, and the largest win: 5001 -> 248 actions, clears
#         level 0 for the first time.
#   s5i5  unpredicted: 5001 -> 2386, now reaches level 1.
#   lp85  501 -> 234, as hoped, though for unclear reasons.
#   tn36  untouched, byte-identical, as the 43% hit rate requires.
#   sp80  untouched, byte-identical.
#
# The aggregate moves little because the newly-clearing games clear slowly
# and the score is quadratic in overrun. It ships for the structural
# change: two games moved from "never leaves level 0" to "clears it", and
# no level can be made fast before it can be finished at all.
LIVENESS_WIDEN = _env_int("ARC_LIVENESS_WIDEN", 1)
# Don't judge liveness on a handful of samples; 12 clicks is one full
# window's worth.
LIVENESS_MIN_CLICKS = _env_int("ARC_LIVENESS_MIN_CLICKS", 40)
# At or below this hit rate the window is not paying for itself. sb26 (65%)
# and tn36 (43%) are far above it; ft09 (0.4%) and lp85 (2%) far below.
LIVENESS_DEAD_RATE = _env_float("ARC_LIVENESS_DEAD_RATE", 0.10)

# lp85 level 2 (2026-08-07): LIVENESS_WIDEN fixed level 1 (502 -> 234
# actions) but level 2 still burns ~4800 actions to map 31 nodes, 71% of
# them "probe-untried". Root cause: `_is_dead` is keyed by action *type*
# (all clicks share one counter), not by coordinate, and is permanently
# unable to fire again the moment even one click succeeds anywhere -- so a
# newly discovered node has no memory of which specific coordinates other
# nodes already found dead, and re-probes close to its full candidate list
# every time. "clicks are dead (1/140 live)" repeats at every widen, which
# is the tell that widening breadth was not the bottleneck here.
#
# This does not remove any candidate -- unlike a hard dead-coordinate
# skip, which risks suppressing a click that is only live once a
# precondition state is reached elsewhere (see r11l/vc33 in
# arc-agi-3-open-threads memory: click liveness there is genuinely
# state-dependent, not global). It only moves codes that have moved the
# board at least once *anywhere in this game* to the front of a brand-new
# node's candidate list, in their existing relative order; everything
# else is still reachable at exactly the cost it always was. Correctness
# is untouched by construction -- this can only change how soon a
# genuinely-live code gets tried, never whether it does.
#
# OFF by default until measured on the full dev set, matching every other
# knob in this file (MOMENTUM_LIMIT looked just as safe on paper and was
# an 8.6x regression on sp80 -- see its comment above).
#
# MEASURED (2026-08-11, full 25-game sweep, max-steps 5000):
# aggregate 0.0761 -> 0.0863 (+13%), but the move is not a clean win --
# ar25 0.00->1.61 and lp85 0.01->0.18 (the intended fix, level 1
# 234->67 actions) are almost exactly cancelled by sp80 1.19->0.00, a
# *total* loss of that game's score. r11l 0.18->0.07, cd82 0.40->0.19,
# lf52 0.02->0.01 (loses its level-3 clear), and ls20 0.01->0.00 also
# regress; tr87 keeps the same score bucket but behaviorally stops
# clearing level 1 at all (2538 actions -> exhausts 5001). The other 17
# games are unchanged. sp80 and r11l regressing is exactly the
# state-dependent-click-liveness risk flagged above -- prioritizing
# codes that were live *somewhere else in the game* delays a click that
# is only live now, in this state, and hadn't fired yet anywhere. Same
# shape as the MOMENTUM_LIMIT lesson: flattering in aggregate, a
# category loss on the one game that carried the most score. Left OFF
# by default; not shipped. See arc-agi-3-open-threads memory for the
# full per-game table and the next candidate.
LIVE_CODE_PRIORITY = _env_int("ARC_LIVE_CODE_PRIORITY", 0)

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

# MASK_CAP_FRACTION gates on how much of the frame a mask covers, and that
# is the wrong axis: it is simultaneously too loose and too tight. tr87
# masks 28.8% at rung 0.3, passes, and loses the level-1 clear its 0.5 rung
# was buying; m0r0 qualifies 44%, is refused, and goes on exploding at 1294
# nodes in 1975 actions. No single fraction separates those.
#
# What does separate them is what the mask DOES to the frames already in
# hand. Measured at the decision point, over the warmup frames:
#
#   game  rung  mask          distinct keys   outcome
#   tn36  0.9     64 (1.6%)         1         1.35, the best game we have
#   tr87  0.3   1180 (28.8%)        1         loses its level-1 clear
#   tr87  0.5    496 (12.1%)       22         gives the clear
#   cd82  0.3    674 (16.5%)       12         0.40
#   sc25  0.5    616 (15.0%)        1         explodes, 0.00
#   sc25  0.7    256 (6.2%)        15         --
#
# Fusing every observed frame into a single state is the failure mode, and
# tn36 shows it is not fatal on its own: a 64-cell mask that collapses
# everything means the board genuinely did not move except in the HUD. A
# 1180-cell mask that collapses everything is deleting real content. So
# reject only the conjunction -- total collapse AND a mask too big to be
# plausibly just a HUD -- and fall back to the next tighter rung.
#
# MEASURED across all 25 dev games. It vetoes five masks, every one of them
# for fusing all 25 sampled frames into a single state: tr87 (1180 cells,
# 29%), sc25 (1096 then 616), ka59 (802), dc22 (438). cd82's 1016-cell mask
# is left alone because 12 distinct keys survive it, and tn36, sp80 and r11l
# never reach this path at all.
#
# The score effect is nil -- 0.1287 -> 0.1288 -- because the one game it
# rescues, tr87, scores 0.00 either way. What it buys is tr87 going back to
# clearing level 1 (5001 -> 2538 actions) and, more to the point, the
# removal of a class of mask that is provably destroying information.
# Shipped as a correctness fix rather than a scoring one, and it makes
# MASK_CAP_FRACTION less load-bearing than it was.
MASK_COLLAPSE_GUARD = _env_int("ARC_MASK_COLLAPSE_GUARD", 1)
# Below this share of the frame, a mask is small enough that collapsing
# everything is credible rather than suspicious (tn36 and sb26 sit at 1.6%).
MASK_SMALL_FRACTION = _env_float("ARC_MASK_SMALL", 0.05)
# A candidate must leave at least this many distinct warmup keys to be
# trusted, once it is past MASK_SMALL_FRACTION.
MASK_MIN_DISTINCT = _env_int("ARC_MASK_MIN_DISTINCT", 2)

# Refuse to mask more than this fraction of the frame. If that much of
# the board qualifies, the reading is not a HUD and dropping it would
# collapse unrelated states together.
MASK_CAP_FRACTION = _env_float("ARC_MASK_CAP", 0.35)

# cn04 (2026-08-11): the strict VOLATILE_THRESHOLD rule gets exactly one
# evaluation, fired the instant `self._samples` reaches VOLATILE_WARMUP,
# and whatever it returns is adopted immediately -- `_mask_is_credible`
# is only ever consulted by the MASK_FALLBACK ladder, never by this path
# (see its own comment). On cn04 that one shot lands during an ~25-action
# intro/settling period where real playfield content is ticking on
# almost every transition: at samples=25 the rule finds a 906-cell
# candidate (22% of the frame) and adopts it as if it were a permanent
# HUD, which deletes signal the search needs for the rest of the game --
# 0.00, never clears level 1 within 5000 actions. Size and collapse
# can't tell this apart from a real large HUD (cn04's masked frames stay
# 22/25 mutually distinct, so `_mask_is_credible` would wave it through
# too, per its own docstring: collapse alone is not the tell). What does
# work: more evidence. At VOLATILE_WARMUP=400 (forced via env, not this
# knob) the strict rule finds nothing -- the transient has been diluted
# below threshold by then -- and correctly falls through to the guarded
# ladder, which picks a 192-cell mask at 0.70 and clears level 1 in 1033
# actions instead of never.
#
# MASK_STRICT_SCALE, when on, does not change what the strict rule does
# with a small candidate (below MASK_SMALL_FRACTION -- tn36 and sb26's
# real, small, immediately-correct masks are untouched by construction).
# A candidate ABOVE that fraction gets a second look -- but a *single*
# fixed extra wait was measured and rejected here first: scaling the
# extra samples by candidate size landed cn04 at 577 total samples, and
# at 577 the same 906-cell candidate comes back (via the fallback
# ladder at 0.50 this time, not the strict rule, but the same wrong
# mask). Only at 400 does it actually find nothing and fall through to
# a correct 192-cell mask at 0.70 (see history via `git log -p` on this
# comment if the exact numbers below ever need re-deriving). That is
# the same non-monotone warmup behaviour already on record for sp80
# (see arc-agi-3-open-threads memory) -- there is no single "enough
# evidence" sample count to compute in closed form.
#
# So instead of one guess, checkpoint geometrically -- VOLATILE_WARMUP,
# then double each time a checkpoint still finds something above
# MASK_SMALL_FRACTION -- and track the smallest candidate seen at any
# checkpoint. 25 -> 50 -> 100 -> 200 -> 400 -> 800 crosses cn04's 400
# sweet spot without having to know it in advance. Up to
# MASK_STRICT_SCALE_DOUBLINGS checkpoints (default 5, i.e. up to 800
# for the default VOLATILE_WARMUP=25); after the last one, commit to
# whichever checkpoint's candidate was smallest, not necessarily the
# final one -- the non-monotonicity means the last checkpoint is not
# reliably the best one.
#
# Risk: while deferred, `_learn_mask` keeps returning False, so masking
# has not started yet and every ticking cell stays in the node key for
# that whole extra window -- a real, large, IMMEDIATELY-correct HUD
# would explode its own node count during a wait it no longer needs.
#
# MEASURED (2026-08-11, full 25-game sweep, max-steps 5000): aggregate
# 0.0761 -> 0.1070 (+41%), and 23 of 25 games are byte-identical to
# baseline -- the risk above did not materialize anywhere in the dev
# set. cn04, the game this was aimed at, is unmoved (still 0.00; it
# does converge on the correct 192-cell mask, at action 101, but the
# earlier graph-restart changes which nodes the search stumbles into
# afterward and it does not find the exit within 5000 actions this
# time -- the same trajectory-luck effect EARLY_MASK already
# documented, just working against this game instead of for it).
# ar25 moves from "never leaves level 0" to a very expensive clear
# (invisible at 2 decimals). The real win is cd82, whose HUD candidate
# was landing in the same size band as cn04's: level 1 goes 191 -> 111
# actions, game score 0.40 -> 1.17. ON by default.
MASK_STRICT_SCALE = _env_int("ARC_MASK_STRICT_SCALE", 1)
MASK_STRICT_SCALE_DOUBLINGS = _env_int("ARC_MASK_STRICT_SCALE_DOUBLINGS", 5)

# sp80 (2026-08-11): `_mask_frozen` is set once, in `__init__`, and never
# reset anywhere -- `_learn_mask` returns False unconditionally for the
# rest of the game once a mask is adopted (see its top-of-function
# guard). So the mask learned during level 1's warmup is reused, as-is,
# for every later level too, no matter what that level's board actually
# looks like. Traced on sp80: level 1 (baseline 39) adopts a 64-cell
# mask at action 26 and clears in 78. Level 2 (baseline 58) clears in
# 3253 with that same mask, mapping 2195 nodes -- already inefficient,
# but it closes. Level 3 (baseline 25) never clears at all: given 20000
# actions total it spends 16670+ on it without ever completing, and
# none of the recovery paths fire in that whole stretch -- no "clicks
# are dead" widening, no "reachable map exhausted; re-exploring". That
# absence is the tell: the search is not stuck, it is minting a new
# node on nearly every action, which is what an un-masked ticking cell
# looks like (the same signature as cd82's pre-fallback-ladder bug).
# The mask that fixed levels 1 and 2 says nothing about whether level
# 3 introduces a new moving element the original ~25-800-action warmup
# never saw, because that warmup never ran during level 3 to find out.
#
# MASK_RELEARN_PER_LEVEL re-arms `_learn_mask` (unfreezes it and clears
# its counters -- NOT `_kept`/`_volatile`, so the old mask keeps being
# used, unchanged, right up until fresh evidence actually replaces it)
# at every level-up, alongside the existing per-level graph drop. A
# level whose board matches the original mask re-converges on the same
# mask at the same relative cost as the original warmup did and loses
# nothing; a level that needs a different one gets the chance to find
# it instead of exploding silently for the rest of the run.
#
# Cost: every level pays its own warmup charge again (see VOLATILE_
# WARMUP's comment -- "everything sampled before the mask is adopted is
# charged to the level and then thrown away"), where today only level 1
# does.
#
# MEASURED AND REJECTED (2026-08-11, sp80): `_reset_graph` -- called by
# `_rearm_mask_learning`'s caller whenever the level-relearned mask
# actually changes -- clears `_fatal`/`_deaths`, the proven-lethal
# memory, along with the rest of the graph. sp80 barely respawns at all
# normally (once, in its whole 20000-action trace). With this on, level
# 2 relearns a mask, resets, and then respawns roughly every 28 actions,
# 480+ times in a row -- it re-discovers and re-forgets the same fatal
# move on every cycle. Level 2 goes 3253 -> 11226 actions and level 3
# still never clears; net neutral-to-negative for sp80, and this is
# exactly the kind of level (frequent forced respawns) the fix was
# supposed to help. The diagnosis this flag targets is real (see its
# own comment) but the fix needs to preserve fatal/death memory across
# an in-level remask, not wipe it via a full graph reset. OFF by
# default; not shipped.
MASK_RELEARN_PER_LEVEL = _env_int("ARC_MASK_RELEARN_PER_LEVEL", 0)

# The status band is what shares a row or column with the HUD without
# ticking itself (see `status_band`). Capped for the same reason the mask
# is: past a certain size the "band" is just the board, and re-opening
# the whole map every time any of it changes would mean never settling.
STATUS_BAND_CAP = _env_float("ARC_STATUS_BAND_CAP", 0.10)

# Mask a ticking row/column only across the widget that ticks on it,
# rather than end to end. Off restores the whole-line rule that produced
# the 0.1299 baseline, which is what makes the two comparable.
# Measured over the 25 dev games, correctness is not free here. The span
# mask stops conflating states that differ only by the avatar's shape,
# colour and rotation -- which is what ls20's levels 3-7 are actually
# about -- and the honest map is ~24x larger, so blind search takes
# longer to cross it. Cost of level 1, span on vs off: sp80 138 vs 79,
# vc33 169 vs 100, r11l 205 vs 113, ls20 914 vs 527. The score is
# quadratic in actions, so on sp80 -- our best game at 1.19 -- that is
# 24.4 points of level score becoming 8.0.
#
# So it ships off. Not because it is wrong: because a truer state key
# with no search able to exploit it is worth less than the score it
# costs. The respawn detector it enables runs either way (see
# `_learn_mask`), which is the part that pays today.
MASK_SPAN = _env_int("ARC_MASK_SPAN", 0)

# Spend the last actions of a life on probes rather than travel. A game
# with a step budget takes the avatar back to the start when it runs out
# -- ls20 every 43-45 actions, ninety times in three thousand -- and a
# walk to a distant frontier that the respawn interrupts bought nothing,
# while an untried action taken where the agent already stands records an
# edge that stays true across every respawn to come.
RESPAWN_AWARE = _env_int("ARC_RESPAWN_AWARE", 0)

# Periods to keep when estimating how long a life lasts. ls20 measures
# 43, 44, 45 for the counter running out but also 22 and 23 when a hazard
# gets there first, so the estimate is a median of recent lives rather
# than the last one, which would swing by half on a single unlucky death.
RESPAWN_SAMPLES = _env_int("ARC_RESPAWN_SAMPLES", 8)

# Shortest gap that can be a life. Without a floor the detector calls
# every flicker a respawn: bp35, vc33 and lf52 all report "period 1"
# thirty times inside sixty actions, because their readouts toggle
# constantly and the board returns to states it has already been in. A
# zero-length life would then leave `_steps_left` permanently at zero and
# divert those games into local probes on every single action. ls20's
# real lives measure 22 to 45.
RESPAWN_MIN_PERIOD = _env_int("ARC_RESPAWN_MIN_PERIOD", 8)

# Re-open exhausted nodes when the status readout changes -- the idea
# being that a wall the agent bounced off as one thing may not be a wall
# once it has become another. Default OFF, because measurement says the
# premise is already handled: the readout is part of the state key, so
# "goal square at rotation 270" and "goal square at rotation 0" are
# different nodes with their own cursors, and re-opening only replays
# edges already recorded. On ls20 it turned a 914-action level 1 into one
# that never finished inside 3000. Kept, off, because a game that shows
# status the agent cannot key on would need exactly this.
STATUS_REOPEN = _env_int("ARC_STATUS_REOPEN", 0)

# Cap on how many times a status change may re-open the map. A game whose
# readout flickers would otherwise reset the search forever and never
# close a level; ls20 changes it once per modifier tile, so a small
# allowance is plenty and an unbounded one is only a risk.
STATUS_REOPEN_LIMIT = _env_int("ARC_STATUS_REOPEN_LIMIT", 40)

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


def segment_frame(
    grid: tuple[tuple[int, ...], ...],
    ignore: frozenset[int] = frozenset(),
) -> dict[str, Any]:
    """Parse a frame into 4-connected same-color objects.

    This is the view the model reasons over instead of the raw grid, which
    is deliberately never handed to it: a 64x64 field of integers invites
    coordinate arithmetic, and asking a model to pick from a rendered board
    is measurably answered by position rather than by content.

    Returns ``{"nodes": [...], "adjacency_list": [[i, j], ...]}``, one node
    per 4-connected run of a single color:

        id        index, ordered top-most then left-most (scan order)
        color     cell value, 0..15
        pixels    cell count
        bbox      (top, left, bottom, right), inclusive
        hash      crc32 over the color plus the cell offsets normalised to
                  the bounding box, so the same shape in the same color
                  hashes equal wherever it sits. Equal hashes mean the same
                  object, which is what lets one be followed across frames
                  or duplicates spotted within one frame.
        children  ids of every node this one encloses

    `adjacency_list` holds ``[i, j]`` id pairs whose objects share an edge.

    Background is NOT dropped, unlike `click_targets`. Which component is
    background is a judgement, and "the large region everything sits on" is
    a fact worth being able to see rather than one to bake into the parse.

    `ignore` holds flat indices (y * width + x) to treat as absent -- the
    volatile HUD cells -- so a step counter's digits do not arrive as a row
    of tiny objects. Ignored cells are traversable for containment.

    Deliberately omitted: a clockwise `boundary` corner list. Corner
    tracing is fiddly to get right, `bbox` and `pixels` carry most of the
    same information, and a wrong boundary is worse than no boundary.
    """
    height = len(grid)
    if height == 0:
        return {"nodes": [], "adjacency_list": []}
    width = len(grid[0])
    if width == 0:
        return {"nodes": [], "adjacency_list": []}

    label = [[-1] * width for _ in range(height)]
    nodes: list[dict[str, Any]] = []
    members: list[list[tuple[int, int]]] = []

    for y0 in range(height):
        for x0 in range(width):
            if label[y0][x0] != -1 or (y0 * width + x0) in ignore:
                continue
            color = grid[y0][x0]
            nid = len(nodes)
            label[y0][x0] = nid
            stack = [(y0, x0)]
            cells: list[tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and label[ny][nx] == -1
                        and grid[ny][nx] == color
                        and (ny * width + nx) not in ignore
                    ):
                        label[ny][nx] = nid
                        stack.append((ny, nx))

            top = min(c[0] for c in cells)
            left = min(c[1] for c in cells)
            bottom = max(c[0] for c in cells)
            right = max(c[1] for c in cells)
            offsets = sorted((cy - top, cx - left) for cy, cx in cells)
            signature = f"{color}|" + ";".join(f"{a},{b}" for a, b in offsets)
            nodes.append(
                {
                    "id": nid,
                    "color": color,
                    "pixels": len(cells),
                    "bbox": (top, left, bottom, right),
                    "hash": zlib.crc32(signature.encode()),
                    "children": [],
                }
            )
            members.append(cells)

    pairs: set[tuple[int, int]] = set()
    for y in range(height):
        for x in range(width):
            a = label[y][x]
            if a == -1:
                continue
            for ny, nx in ((y + 1, x), (y, x + 1)):
                if ny < height and nx < width:
                    b = label[ny][nx]
                    if b != -1 and b != a:
                        pairs.add((a, b) if a < b else (b, a))

    # Containment: B is inside A when no 4-connected path from B reaches the
    # grid border without crossing A. Flood from the border across everything
    # that is not A; whatever is never reached is enclosed by A. Only nodes
    # of 8+ cells can ring anything (a 3x3 ring around one cell is the
    # smallest possible), which prunes most of the work away.
    if len(nodes) > 1:
        for parent, cells in enumerate(members):
            if len(cells) < 8:
                continue
            reached = [[False] * width for _ in range(height)]
            frontier: deque[tuple[int, int]] = deque()
            for y in range(height):
                for x in (0, width - 1):
                    if label[y][x] != parent and not reached[y][x]:
                        reached[y][x] = True
                        frontier.append((y, x))
            for x in range(width):
                for y in (0, height - 1):
                    if label[y][x] != parent and not reached[y][x]:
                        reached[y][x] = True
                        frontier.append((y, x))
            while frontier:
                cy, cx = frontier.popleft()
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and not reached[ny][nx]
                        and label[ny][nx] != parent
                    ):
                        reached[ny][nx] = True
                        frontier.append((ny, nx))
            enclosed = [
                other
                for other, other_cells in enumerate(members)
                if other != parent
                and not any(reached[cy][cx] for cy, cx in other_cells)
            ]
            nodes[parent]["children"] = enclosed

    return {"nodes": nodes, "adjacency_list": [list(p) for p in sorted(pairs)]}


# The model gets exactly these modules. Same list TAAF exposes: enough to
# write BFS, flood fill or a beam search, and nothing that reaches the
# filesystem, the network or the process.
TOOL_MODULES = frozenset(
    {
        "bisect", "collections", "copy", "fractions", "functools", "heapq",
        "itertools", "json", "math", "operator", "random", "re",
        "statistics", "string",
    }
)
# A runaway loop in model-written code would otherwise hang the game thread
# until the whole run times out. 30s matches TAAF's tool_timeout.
TOOL_TIMEOUT = _env_float("ARC_TOOL_TIMEOUT", 30.0)
# TAAF caps tool output at 1024 tokens; ~4 chars a token. The cap is not
# politeness -- `segment_frame` on a noisy board returns 2184 nodes, which
# would bury the conversation and evict the model's own reasoning.
TOOL_OUTPUT_CHARS = _env_int("ARC_TOOL_OUTPUT_CHARS", 4000)


class ToolTimeout(Exception):
    """Model-written code outran TOOL_TIMEOUT."""


# Sentinel for "this key was absent", so restoring can tell a missing key
# from one that was legitimately set to None.
_MISSING = object()


def _tool_builtins() -> dict[str, Any]:
    """Builtins for sandboxed code, with the obvious foot-guns removed.

    NOT a security boundary, and nothing here pretends otherwise: Python
    cannot be sandboxed against hostile code from inside the interpreter,
    and `__subclasses__` walking defeats any such attempt in a few lines.
    The model is a cooperative author, not an attacker. What this actually
    buys is that an *accident* fails loudly and locally -- an import of
    `numpy` that would not exist on Kaggle, a stray `open()`, an `exec` of
    generated text -- instead of half-working.
    """
    # `__builtins__` is a module inside the importing module and a dict
    # inside an exec'd one; both shapes are reachable from here.
    namespace = __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)
    allowed = dict(namespace)
    for name in ("exec", "eval", "compile", "open", "input", "breakpoint", "help"):
        allowed.pop(name, None)

    real_import = __import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.partition(".")[0]
        if root not in TOOL_MODULES:
            raise ImportError(
                f"{root!r} is not available in the tool sandbox. "
                f"Available: {', '.join(sorted(TOOL_MODULES))}."
            )
        return real_import(name, globals, locals, fromlist, level)

    allowed["__import__"] = guarded_import
    return allowed


def run_python_tool(
    code: str,
    namespace: dict[str, Any],
    *,
    timeout: float = TOOL_TIMEOUT,
    output_limit: int = TOOL_OUTPUT_CHARS,
) -> dict[str, Any]:
    """Run one ephemeral block of model-written code.

    Returns ``{"stdout", "result", "error", "truncated"}``. `namespace`
    supplies whatever the model is allowed to touch -- the frame views and
    the `action(...)` callable -- and is mutated in place, so the caller can
    read back anything the code assigned.

    Two things here are shaped by the fact that the framework runs ONE AGENT
    PER GAME AS A THREAD in a single process (see the CONTRACT at the top):

    * `print` is rebound to a per-call buffer rather than redirecting
      `sys.stdout`. Swapping `sys.stdout` is process-global, so with 25 games
      running concurrently one game's tool output would land in another
      game's transcript. That bug would be near-impossible to read back from
      a Kaggle log, so the design avoids it rather than detecting it.
    * The deadline is enforced by a per-thread trace function, because
      `signal.alarm` only works on the main thread and there are 25 of these.

    The deadline is checked on each Python line, so it cannot interrupt a
    single long call inside C -- a catastrophic regex or one enormous
    `sorted()` will overrun it. Tracing also slows the code measurably.
    Both are accepted: the point is to stop `while True`, not to be exact.
    """
    buffer: list[str] = []
    written = 0

    def tool_print(*args: Any, sep: str = " ", end: str = "\n", **_: Any) -> None:
        nonlocal written
        if written >= output_limit:
            return
        text = sep.join(str(a) for a in args) + end
        buffer.append(text)
        written += len(text)

    # The namespace IS the execution scope -- not a copy of it.
    #
    # It used to be `dict(namespace)`, copied in and merged back out at the
    # end. That quietly defeated `action(...)`: the callable refreshes `grid`
    # and `current_frame` in the namespace as the board moves, but the
    # running code was reading a snapshot taken before it started, so the
    # board never appeared to change no matter what was played -- and the
    # merge afterwards then wrote the stale snapshot back over the fresh
    # values. Executing in the real dict makes the refresh visible where it
    # has to be: inside the model's own code, between its own statements.
    scope = namespace
    saved = {k: scope.get(k, _MISSING) for k in ("__builtins__", "print", "result")}
    scope["__builtins__"] = _tool_builtins()
    scope["print"] = tool_print
    scope.pop("result", None)

    deadline = time.monotonic() + timeout

    def local_trace(frame, event, arg):  # noqa: ANN001 - CPython trace protocol
        if time.monotonic() > deadline:
            raise ToolTimeout(f"tool code exceeded {timeout:g}s")
        return local_trace

    def global_trace(frame, event, arg):  # noqa: ANN001
        return local_trace

    error = ""
    previous = sys.gettrace()
    try:
        sys.settrace(global_trace)
        exec(compile(code, "<tool>", "exec"), scope)  # noqa: S102 - see docstring
    except ToolTimeout as exc:
        error = str(exc)
    except SyntaxError as exc:
        error = f"SyntaxError: {exc}"
    except BaseException as exc:  # noqa: BLE001 - reported back to the model
        # The model has to be able to fix its own code, so it gets the
        # traceback rather than a bare class name. Frames below `<tool>` are
        # dropped: agent internals are not its business and cost context.
        lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
        error = "".join(lines[-3:]).strip()
    finally:
        sys.settrace(previous)

    result = scope.get("result")
    # Put back only what we borrowed. The model's own definitions stay --
    # that persistence is the point -- but `print` and `__builtins__` are
    # ours and must not leak into the next call's view of its workspace.
    for key, value in saved.items():
        if value is _MISSING:
            scope.pop(key, None)
        else:
            scope[key] = value

    stdout = "".join(buffer)
    truncated = len(stdout) > output_limit
    if truncated:
        stdout = stdout[:output_limit] + "\n...[output truncated]"

    return {
        "stdout": stdout,
        "result": result,
        "error": error,
        "truncated": truncated,
    }


# --- the model loop ------------------------------------------------------
#
# Names are ARC_LLM_* rather than ARC_POLICY_*. The ARC_POLICY_* space
# already belongs to the abandoned option-picking policy, and reusing it
# would make it impossible to tell from a Kaggle log which of the two was
# configured. That policy is the thing this replaces: it was measured
# answering by option position (code agreement 0.125 against chance 0.088),
# which is what happens when a model is handed a menu instead of a problem.
LLM_ENABLED = _env_int("ARC_LLM", 0)
LLM_BASE_URL = os.getenv("ARC_LLM_BASE_URL", "http://127.0.0.1:1234/v1")
LLM_MODEL = os.getenv("ARC_LLM_MODEL", "vrfai/Qwen3.6-27B-FP8")
# TAAF's analyzer settings, which are Qwen3's own recommended sampling.
LLM_TEMPERATURE = _env_float("ARC_LLM_TEMPERATURE", 0.6)
LLM_TOP_P = _env_float("ARC_LLM_TOP_P", 0.95)
LLM_TOP_K = _env_int("ARC_LLM_TOP_K", 20)
# A ceiling on ONE response. 0 means no ceiling, which is what ships,
# because v23 measured what a ceiling actually costs here.
#
# v23 set this to 2048 on the following reasoning: v22 generated ~200
# tok/s aggregate over 25 games, so ~7200 tokens per turn across ~12 tool
# calls is ~600 tokens a call, and 2048 sits comfortably above that.
#
# Every step of that is arithmetic on an aggregate, and the aggregate hid
# the distribution. What v23 actually did, over 236 turns:
#
#     stop reason   truncated 178 | max-tool-calls 37 | halted 21
#     tool calls per turn   median 0
#     actions per turn      median 0
#
# 75% of turns hit the cap having emitted no tool call at all -- every one
# of the 25 games, tr87 and g50t on all 15 and all 14 of theirs. The
# model's reasoning preamble alone exceeds 2048 tokens on this task, so
# the cap does not trim a runaway tail; it severs the reply before the
# call it was reasoning towards. Turns got 2.9x shorter (637s -> 221s
# median) and 2.4x more numerous, and total actions still fell 2238 ->
# 1732, which is the signature of a turn that ends without doing anything.
#
# The plumbing and the `truncated` stop reason stay, so a future cap can
# be measured rather than reasoned about -- but a cap needs the reply
# length distribution first, and nothing here has measured it.
LLM_MAX_TOKENS = _env_int("ARC_LLM_MAX_TOKENS", 0)
# Measured on the Blackwell card: a turn averaged 62.5s wall clock across
# several generations, and one request still ran past a 120s deadline --
# a thinking model with a grown transcript is simply slow sometimes. 300
# is chosen to make the timeout mean "something is wrong" rather than
# "this one was long", which is what it has to mean for the failure count
# below to be worth anything.
#
# 300 was measured at batch 1, on a GPU with nothing else on it. The
# submission runs 25 games through one server: v25 had 23 requests in
# flight on average, and a request that would have been served in 60s
# alone now queues behind two dozen others. 31 timeouts landed in one
# hour, and each one wasted the whole 300s AND threw the generation away.
# 900 is the same rule -- "something is wrong", not "this one was long" --
# re-measured at the concurrency we actually ship at.
LLM_REQUEST_TIMEOUT = _env_float("ARC_LLM_REQUEST_TIMEOUT", 900.0)

# Consecutive failed model calls before the search takes over the game.
# Not 1: see `_llm_turn`. A dead server trips this immediately and pays
# the retries once; a single slow generation costs one search step.
LLM_MAX_FAILURES = _env_int("ARC_LLM_MAX_FAILURES", 3)
# Consecutive TIMEOUTS before the same thing happens. Deliberately much
# larger, because a timeout means something different from every other
# transport failure: a refused connection says the server is gone, a
# timeout says the queue is deep. In v25 `cn04` hit three timeouts in
# three consecutive actions, had its model retired permanently, and
# finished the run last -- 28 actions, one turn -- while the server it
# was written off for went on serving 24 other games. Three strikes is
# the right rule for a dead server and the wrong one for a busy one, so
# the two are counted apart.
LLM_MAX_TIMEOUTS = _env_int("ARC_LLM_MAX_TIMEOUTS", 12)
# Consecutive identical turns before the transcript is thrown away. See
# `_llm_turn`: 1 is a note, 2 says the note did not land and the model is
# now agreeing with its own history, which only a reset breaks.
LLM_REPEAT_RESET = _env_int("ARC_LLM_REPEAT_RESET", 2)
# How many python calls the model may make per observe-plan-act turn.
#
# Was 12, on TAAF's principle that the model should not ration tool calls.
# Measured on ls20, that backstop was not a backstop: ELEVEN of twelve turns
# ran to the cap, averaging 370s each at ~31s a call, and the model still had
# not cleared a 22-action level after 304 actions. A cap that is hit every
# single time is not bounding a pathological case, it is the schedule.
#
# 6 halves the turn while the model is being given the board as an image --
# most of what those calls were doing was reconstructing by arithmetic what
# it can now simply look at. If turns still end at the cap with the image in
# place, the problem is the loop, not the budget.
#
# v16 ended every turn at the cap, so by that test the problem is the loop.
# Reading the calls themselves says otherwise, and the test was too coarse.
# In v15 the six calls repeated one another because the board never
# refreshed. In v16 they are six DIFFERENT experiments -- test ACTION2,
# move up seven times and look, write find_mouth, test left/right, test
# right, test a click -- and across them verify accuracy went 0.04 -> 0.25
# -> 0.58. That is a turn running out of room mid-investigation, not a loop.
#
# The distinguishing evidence is content and accuracy, not whether the cap
# was reached, so the cap moves to 12. A turn boundary is not free: it
# re-sends the system prompt, the segmentation and a fresh image, so a cap
# set below what the work needs buys overhead rather than thinking. If v17
# still ends every turn at 12 AND accuracy has stopped climbing, then it is
# the loop and this should come back down rather than up.
#
# v29 (2026-08-06): revisited, deliberately deferred until now so it
# wouldn't be conflated with the KV-cache-pressure work (v25-v27). v27
# shows 71/175 turns (40.6%) still ending at 12, softer than the "every
# single turn" that justified the 6->12 move, but not nothing -- and
# reading the highest-action cap-hit turns directly (lf52 52 actions/12
# calls, bp35 51/12, wa30 47/12, all still level 0, none showing the
# "0 transitions, repeated identically" signature that flagged v15's loop)
# looks like genuine varied work, not a loop, the same kind of evidence
# that raised this to 12 last time.
#
# The NEW consideration the 6->12 decision never had: this session's
# KV-cache-pressure analysis (see arc-agi-3-kaggle-gpu-infra memory) found
# 73-82% of server samples already sitting in the 80-99% KV-usage band,
# where prefix-cache hit rate collapses. A longer cap means longer turns
# means heavier individual requests, which could push MORE samples into
# that band -- a real cost this change might have that the last one
# didn't. 16 (a ~33% step, not a jump to 24) is deliberately moderate so a
# GPU run can measure whether that cost shows up before considering
# anything bigger. Judge on clears>=1, clears>=2, cap-hit fraction, AND
# the KV-bucket hit-rate curve against v27 -- not score alone.
#
# v29 GPU result (2026-08-06): that cost showed up, and it wasn't offset.
# Same-mechanism, apples-to-apples signal: in the 80-99% KV-usage band,
# v29 put MORE of its samples there (82% vs v27's 76%) and got a LOWER
# hit rate once there (3.4% vs 4.7%) -- raising the cap measurably made
# the exact pressure v27 had just relieved worse again. Game outcomes
# were a wash at best: clears>=1 fell 11->10 (lost tn36/re86/s5i5, gained
# sp80/cn04), total actions fell 5158->3729 and total turns 175->117
# (longer turns -> fewer of them fit the fixed run budget), and the
# cap-hit fraction barely moved (40.6%->41.9%) -- most turns still
# exhaust whatever cap they're given, so the extra headroom bought
# little. The local score rose (0.67->0.88) only because 2 games
# deepened to level 2, which the quadratic formula rewards heavily; per
# arc-agi-3-open-threads, level-1 breadth is what the score target
# actually needs, and that's what fell. Reverted to 12. See
# arc-agi-3-kaggle-gpu-infra memory before raising this again.
LLM_MAX_TOOL_CALLS = _env_int("ARC_LLM_MAX_TOOL_CALLS", 12)

PYTHON_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "python",
        "description": (
            "Run Python to inspect the board and act. The runtime provides "
            "current_frame, previous_frame, history, transitions, "
            "valid_actions, last_action_result, and action(actions) which "
            "executes real environment actions. Print compact summaries "
            "only; never print a whole board."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python source to execute in the sandbox.",
                }
            },
            "required": ["code"],
        },
    },
}


def _is_timeout(exc: BaseException) -> bool:
    """Did this failure mean "too slow" rather than "not there"?

    The distinction decides whether a game keeps its model (see
    `_llm_turn`), so it is worth being exact about. `urlopen(timeout=)`
    raises a bare `TimeoutError` -- which is what v25's 31 timeouts
    surfaced as -- but a timeout during connect arrives wrapped in a
    `URLError`, and that spelling must not be read as a dead server.
    """
    if isinstance(exc, TimeoutError):
        return True
    reason = getattr(exc, "reason", None)
    if isinstance(reason, BaseException):
        return isinstance(reason, TimeoutError)
    return False


def chat_completion(
    payload: dict[str, Any],
    *,
    base_url: str = LLM_BASE_URL,
    timeout: float = LLM_REQUEST_TIMEOUT,
) -> dict[str, Any]:
    """POST to an OpenAI-compatible `/chat/completions`, stdlib only.

    No `openai` package: this file is spliced verbatim into the notebook and
    may import nothing that the first cell does not install. vLLM's server
    accepts `top_k` as a top-level field even though OpenAI's schema has no
    such parameter, which is why it is passed that way.
    """
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def render_tool_result(outcome: dict[str, Any], *, limit: int = TOOL_OUTPUT_CHARS) -> str:
    """Turn a `run_python_tool` outcome into the text the model reads back.

    Errors are never swallowed: the model can only fix its own code if it
    sees what broke. An empty run reports itself as empty rather than as a
    blank message, because a model given "" tends to assume success.
    """
    parts: list[str] = []
    if outcome.get("stdout"):
        parts.append(outcome["stdout"].rstrip("\n"))
    if outcome.get("result") is not None:
        parts.append(f"result = {outcome['result']!r}")
    if outcome.get("error"):
        parts.append(f"ERROR:\n{outcome['error']}")
    if not parts:
        parts.append("(no output; nothing was printed and `result` was not set)")
    text = "\n".join(parts)
    if len(text) > limit:
        text = text[:limit] + "\n...[truncated]"
    return text


def evict_turns(
    messages: list[dict[str, Any]],
    keep_turns: int,
    slack: int | None = None,
) -> list[dict[str, Any]]:
    """Drop middle history, keeping the system prompt and the last turns.

    Context is 65536 tokens (`max_model_len` in the vLLM startup args --
    an older version of this comment said 32768, which was never measured
    against the server and was just wrong) and a single segmentation dump
    can be thousands of them, so old turns have to go. The system prompt
    and the FIRST user turn are always kept -- the first turn carries the
    game framing, and a model that loses it starts reasoning about a board
    with no idea what the game is. `keep_turns <= 0` means keep everything.

    Can trim in hysteresis (`slack > 0`): the cut point only moves once the
    transcript has grown `slack` turns past `keep_turns`, instead of
    creeping forward by however much this turn added. The idea (v25,
    2026-08-06) was that a steady cut point lets consecutive turns share an
    identical prefix, which the server-side prefix cache could then reuse.
    Measured and NOT confirmed: v25's hit rate didn't move, and v26 (same
    slack, a longer rehearsal) hit six `HTTP 400: Input length exceeds
    model's maximum context length` rejections that v22/v24 (slack effectively
    0) mostly didn't -- holding a longer transcript between jumps tips some
    turns just over 65536. LLM_EVICT_SLACK now defaults to 0 for that
    reason; see its comment and arc-agi-3-kaggle-gpu-infra memory before
    raising it again.
    """
    if keep_turns <= 0 or len(messages) <= 2:
        return messages
    if slack is None:
        slack = LLM_EVICT_SLACK
    trigger = 2 + (keep_turns + max(slack, 0)) * 2
    if len(messages) <= trigger:
        return messages
    head = [m for m in messages[:2]]
    tail_start = max(2, len(messages) - keep_turns * 2)
    tail = messages[tail_start:]
    # A tool message whose matching assistant tool_call has been evicted is
    # a protocol error for the server, so walk forward to the first message
    # that is not an orphaned tool result.
    while tail and tail[0].get("role") == "tool":
        tail = tail[1:]
    return head + tail


def run_tool_conversation(
    messages: list[dict[str, Any]],
    namespace: dict[str, Any],
    *,
    post: Any = None,
    model: str = LLM_MODEL,
    max_tool_calls: int = LLM_MAX_TOOL_CALLS,
    tool_timeout: float = TOOL_TIMEOUT,
    output_limit: int = TOOL_OUTPUT_CHARS,
    temperature: float = LLM_TEMPERATURE,
    top_p: float = LLM_TOP_P,
    top_k: int = LLM_TOP_K,
    max_tokens: int = LLM_MAX_TOKENS,
    stop_check: Any = None,
    label: str = "llm",
) -> dict[str, Any]:
    """Drive one observe-plan-act turn: ask, run the code, feed back, repeat.

    `messages` is appended to in place so the caller keeps the transcript.
    `post` is injected so this is testable without a model -- the real one is
    `chat_completion`, and the whole loop is exercised in tests by a scripted
    stand-in. Returns ``{"stop": reason, "tool_calls": n, "content": str}``.

    `stop_check` is called after every tool call and ends the turn when it
    returns true. That is how a level completion or a game over stops the
    model mid-search instead of letting it keep acting into a board that has
    already moved on.

    Network and protocol failures are allowed to propagate. The caller is
    expected to fall back to the search agent, and a policy that silently
    swallowed a dead server would score exactly what the search scores while
    looking like a model run that merely did not help -- which has already
    happened once on this project.
    """
    send = post or chat_completion
    calls_made = 0

    while True:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tools": [PYTHON_TOOL_SCHEMA],
            "tool_choice": "auto",
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }
        # Omitted rather than sent as 0: the server would read a 0 as a
        # request for zero tokens, not as "no limit". Absent means the
        # server's own default, which is what v22 shipped and measured.
        if max_tokens:
            payload["max_tokens"] = max_tokens
        reply = send(payload)
        choices = reply.get("choices") or []
        if not choices:
            return {"stop": "empty-response", "tool_calls": calls_made, "content": ""}
        message = choices[0].get("message") or {}
        messages.append(message)

        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            # `length` means the reply hit `max_tokens` rather than finishing,
            # so the turn is ending on a sentence the model was still writing
            # -- not because it decided it was done. Those two look identical
            # in a turn report unless they are named apart, and a cap that
            # silently ends turns is exactly the kind of change that looks
            # like a throughput win while quietly costing the run.
            truncated = choices[0].get("finish_reason") == "length"
            if truncated:
                logger.warning(
                    "%s: reply hit the %d-token cap with no tool call; "
                    "the turn ends mid-thought",
                    label, max_tokens,
                )
            return {
                "stop": "truncated" if truncated else "content",
                "tool_calls": calls_made,
                "content": message.get("content") or "",
            }

        for call in tool_calls:
            calls_made += 1
            function = call.get("function") or {}
            raw = function.get("arguments") or "{}"
            try:
                arguments = json.loads(raw) if isinstance(raw, str) else dict(raw)
            except (TypeError, ValueError) as exc:
                # Malformed tool arguments are the model's mistake to fix, so
                # they go back as a tool result rather than killing the turn.
                rendered = f"ERROR: could not parse tool arguments as JSON: {exc}"
            else:
                code = arguments.get("code")
                if not isinstance(code, str) or not code.strip():
                    rendered = (
                        "ERROR: the `python` tool needs a non-empty `code` "
                        "string argument."
                    )
                else:
                    outcome = run_python_tool(
                        code,
                        namespace,
                        timeout=tool_timeout,
                        output_limit=output_limit,
                    )
                    rendered = render_tool_result(outcome, limit=output_limit)
                    # Three GPU runs told us the model was thrashing and
                    # none of them told us what it believed. This is the
                    # cheapest possible fix for that: the code itself,
                    # capped, plus whether it raised.
                    if LLM_LOG_CODE:
                        logger.info(
                            "%s tool call %d:\n%s%s",
                            label, calls_made,
                            code[:LLM_LOG_CODE_CHARS],
                            "" if not outcome["error"]
                            else f"\n  -> {outcome['error'].splitlines()[-1][:200]}",
                        )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "name": "python",
                    "content": rendered,
                }
            )

            if stop_check is not None and stop_check():
                return {
                    "stop": "halted",
                    "tool_calls": calls_made,
                    "content": "",
                }

        if calls_made >= max_tool_calls:
            return {
                "stop": "max-tool-calls",
                "tool_calls": calls_made,
                "content": "",
            }


# ---------------------------------------------------------------------
# The turn: what the model is shown, and how it acts.
# ---------------------------------------------------------------------
# Actions the model may take inside ONE turn before `action()` refuses.
# This is not a safety rail against the model misbehaving -- it is the
# unit of re-planning. Past this many steps the transcript it planned
# from describes a board that is several moves stale, so it is asked
# again rather than allowed to keep executing against a memory.
LLM_TURN_ACTIONS = _env_int("ARC_LLM_TURN_ACTIONS", 60)

# Recent (action, did-the-board-change) pairs offered as `history`.
LLM_HISTORY = _env_int("ARC_LLM_HISTORY", 16)

# Turns kept in full before the middle of the transcript is dropped. A
# game runs for thousands of actions and the context window does not,
# so something has to go; what survives is the framing plus the recent
# past. See `evict_turns`.
LLM_EVICT_TURNS = _env_int("ARC_LLM_EVICT_TURNS", 4)

# Extra turns of slack before eviction actually trims anything.
#
# The theory (2026-08-06, v25): the cut point used to be recomputed fresh
# every turn -- `tail_start` grows with `len(messages)` -- so the
# surviving prefix's first byte moved on every single turn once a game
# passed LLM_EVICT_TURNS, and no two consecutive turns ever sent the same
# prefix. Holding the cut point steady for a few turns and then trimming
# in one jump should have let most turns reuse the prior turn's prefix.
#
# It didn't work, and it has a real cost. v25 (slack=2) measured prefix
# hit rate median 2.1% against v22's 4.0% and v24's 0.0% -- flat, not the
# hoped-for jump, so the theory's own target metric never moved. Worse,
# v26 (slack=2, 2x rehearsal budget) hit SIX separate `HTTP Error 400`
# rejections from vLLM, all `Input length exceeds model's maximum context
# length (65536)` by 1-183 tokens -- carrying a longer transcript between
# jumps tips some turns just over the real limit (not 32768 as an older
# comment here claimed; verified from vllm-server.log's own startup args,
# `max_model_len: 65536`). v22 saw one such error, v24 zero, v25 three:
# the count tracks the slack, not chance. Given the benefit never
# measured and the cost did, default is back to 0 (== the original
# every-turn trim, safe by construction: it never lets the transcript run
# long enough to approach 65536). See arc-agi-3-kaggle-gpu-infra memory
# for the full v25/v26 numbers before changing this again.
LLM_EVICT_SLACK = _env_int("ARC_LLM_EVICT_SLACK", 0)

# Objects rendered into the prompt text. The full segmentation is always
# in the namespace for the model's own code to walk; this caps only what
# is spelled out in prose.
#
# 120 is measured, not guessed. Segmenting the opening frame of all 25
# competition games gives 7-191 objects (median 24) and a turn prompt of
# 164-839 tokens against a 32768 context. At 48 the cap was truncating 6
# of those 25 -- bp35 showed 48 of its 191 objects -- to save context
# that was never scarce. 120 covers 24 of the 25 outright.
#
# It stays capped because the opening frame is not the worst case: a
# noisy mid-game board segments into thousands of nodes, and rendering
# those would evict the model's own reasoning to describe static.
LLM_MAX_NODES = _env_int("ARC_LLM_MAX_NODES", 120)

# How much of the model's own `notes` is read back each turn. Generous,
# because this is the only memory that outlives eviction: at ~170s a turn a
# game runs to dozens of turns, and without this the rule worked out on turn
# 3 is gone by turn 8 and gets re-derived at full price.
LLM_NOTES_CHARS = _env_int("ARC_LLM_NOTES_CHARS", 3000)

# Log the code the model runs. Three GPU runs established that it was
# thrashing and not one of them showed what it believed, which made every
# diagnosis a guess. At a handful of calls per turn this is a modest number
# of log lines, and it is the difference between "it used 60 actions" and
# knowing why. Set ARC_LLM_LOG_CODE=0 to silence it.
LLM_LOG_CODE = _env_int("ARC_LLM_LOG_CODE", 1)
LLM_LOG_CODE_CHARS = _env_int("ARC_LLM_LOG_CODE_CHARS", 700)

# Planning in a learned simulator, and checking that simulator against
# transitions we already bought.
#
# This is the whole bet, and it follows from the scoring rather than from
# any belief about the model. `EnvironmentScoreCalculator.add_level` scores
# a level `(baseline/actions)**2 * 100`, weights it by the level index, and
# divides by the sum of weights over EVERY level in the game -- reached or
# not. Two consequences drive the design:
#
#   * Beating the baseline earns nothing. `to_score` clamps the result to
#     `max_weights / total_weights * 100`, so a level cleared in half the
#     intended actions scores exactly what one cleared in the intended
#     number scores. The target is baseline, not better than baseline.
#   * Clearing slowly earns nothing either. Our own search clears ls20
#     level 1 in 526 actions against a baseline of 22 -- 24x, worth 0.17
#     out of 100. Across the suite, "clears level 1 eventually" tops out
#     around 0.14 of final score. The metric is a speedrun.
#
# So exploration must be paid for once and then amortised, which means the
# search that finds the route cannot run in the live environment. The model
# supplies `predict(grid, act)`; the agent searches THAT. Simulated actions
# are free, real ones are billed, and the only ones played are the ones on
# a route already found.
PLAN_MAX_NODES = _env_int("ARC_LLM_PLAN_NODES", 200_000)
PLAN_MAX_DEPTH = _env_int("ARC_LLM_PLAN_DEPTH", 60)
# Kept well under TOOL_TIMEOUT (30s), because planning runs INSIDE a tool
# call: a plan budget at or above the tool budget would turn "the search
# was inconclusive", which the model can act on, into "your code was
# killed", which it cannot. Raising one of these means raising both.
PLAN_SECONDS = _env_float("ARC_LLM_PLAN_SECONDS", 10.0)

# Real transitions kept for checking the forward model. Every one was paid
# for in billed actions, so replaying them against a candidate `predict` is
# the one kind of evidence that costs nothing to collect twice.
TRANSITION_LOG = _env_int("ARC_LLM_TRANSITION_LOG", 400)

# When a forward model is good enough to be worth planning with.
#
# v16 is the reason this exists. The model built a real simulator -- it
# found the movable piece by colour, computed its bounding box, worked out
# that a move shifts it five cells -- and drove verify accuracy from 0.04
# to 0.58. Then it called `plan` zero times in the entire run and went back
# to playing actions one at a time to see what happened.
#
# The prompt caused that: it said to fix `predict` "until accuracy is 1.0",
# which on a board with any noise in it is a gate that never opens. The
# wording is fixed, and this is the belt to that pair of braces -- if the
# model has something worth planning with and has not planned, it gets told
# so directly on the next turn.
PLAN_NUDGE_ACCURACY = _env_float("ARC_LLM_PLAN_NUDGE", 0.5)

# How many of the most recent real transitions the turn re-scores the
# model's live `predict` against, without being asked.
#
# v18 is why. The model called verify() four times, all inside turn 2,
# scored 1.00 over THIRTEEN transitions, and never verified again -- while
# going on to play 140 more actions across three further turns. A 1.00
# earned on thirteen samples was load-bearing for the rest of the run, and
# nothing anywhere would have noticed if it had gone stale, because the
# only number reported was the best the model had ever self-reported.
#
# So the harness now checks, every turn, against everything played. The
# most recent slice rather than all of TRANSITION_LOG because this runs
# outside a tool call on the wall clock, and staleness is a question about
# recent moves anyway.
AUTOVERIFY_LIMIT = _env_int("ARC_LLM_AUTOVERIFY", 150)

# How far the live score may sit below the model's own best before the
# turn says so. Some slack: a forward model that is right about the moves
# that matter and wrong about an animated HUD is still worth planning
# with, and that is deliberate policy from v16.
AUTOVERIFY_DRIFT = _env_float("ARC_LLM_AUTOVERIFY_DRIFT", 0.25)


# The board, drawn. `Qwen3_5ForConditionalGeneration` initialises a ViT and
# an MM encoder -- confirmed in the vLLM startup log -- so the served model
# can see. The first real run fed that vision model a prose description of a
# visual puzzle and it never converged on a hypothesis: twelve turns, every
# one of them burning the tool-call cap, and level 1 of ls20 (baseline 22)
# still unsolved after 304 actions. TAAF sends `{context: current_grid,
# upscale: 4}`. This is that.
#
# 8, not TAAF's 4, and the reason is patch geometry rather than taste. A
# ViT sees the image in patches; Qwen-family encoders merge 2x2 of them, so
# one visual token covers roughly 28-32 pixels square. At upscale 4 the
# board is 256px and a single token blurs about EIGHT board cells together
# -- which loses exactly the cell-level detail the image was added to
# provide. 8 doubles the linear resolution to 512px, about four cells per
# token, for 256 visual tokens and a 6KB PNG. Measured across the games:
#
#   upscale   px    PNG     encode   ~tokens   cells/token
#         4   256   1.9KB     3ms         64           8
#         8   512   6.2KB     9ms        256           4
#        16  1024  19.9KB    45ms       1024           2
#
# All are affordable against a 32768 context whose text half peaks at ~2.3k,
# so this is not a budget decision. 16 was tempting, but the image's job is
# the gestalt -- what kind of puzzle this is, how it is laid out -- and the
# prompt already tells the model to take exact coordinates from the
# segmentation and the raw grid, which are pixel-perfect. The exact patch
# size is not something this file can verify, hence the override.
RENDER_UPSCALE = _env_int("ARC_LLM_UPSCALE", 8)

# 16 colours, because all 16 values occur: measured over the opening frame of
# all 25 games, every value 0-15 is present and 5 alone accounts for 38,816
# cells (it is the background on most boards).
#
# NOT the framework's COLOR_PALETTE from `langgraph_thinking/vision.py`. That
# one defines 10 of the 16 and maps both 5 and 11 to the same grey, so two
# distinct values would render identically -- on a puzzle whose entire content
# is which cells differ from which. 0-9 follow the familiar ARC palette; 10-15
# are chosen to sit away from those hues.
PALETTE: tuple[tuple[int, int, int], ...] = (
    (0, 0, 0),        # 0  black
    (0, 116, 217),    # 1  blue
    (255, 65, 54),    # 2  red
    (46, 204, 64),    # 3  green
    (255, 220, 0),    # 4  yellow
    (170, 170, 170),  # 5  grey
    (240, 18, 190),   # 6  fuchsia
    (255, 133, 27),   # 7  orange
    (127, 219, 255),  # 8  azure
    (135, 12, 37),    # 9  maroon
    (255, 255, 255),  # 10 white
    (75, 0, 130),     # 11 indigo
    (0, 128, 128),    # 12 teal
    (139, 69, 19),    # 13 brown
    (255, 182, 193),  # 14 pink
    (128, 128, 0),    # 15 olive
)


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def render_png(
    grid: tuple[tuple[int, ...], ...], upscale: int = RENDER_UPSCALE
) -> bytes:
    """The frame as a PNG, written by hand.

    Hand-rolled because this file is spliced into the notebook and may import
    only the standard library -- Pillow exists on the Kaggle image but not by
    any guarantee this file is allowed to rely on. A truecolour PNG is a
    signature, three chunks and one zlib stream, which is a smaller thing to
    own than a dependency that would break the contract at the top of this
    module.

    `upscale` repeats each cell into an NxN block. A 64x64 image is smaller
    than the model's patch grid, so without it whole objects vanish into a
    single patch.
    """
    height = len(grid)
    width = len(grid[0]) if height else 0
    if not height or not width:
        # A 1x1 black pixel. Callers get a valid image rather than an
        # exception on the action path.
        return render_png(((0,),), 1)

    raw = bytearray()
    for row in grid:
        line = bytearray()
        for value in row:
            line += bytes(PALETTE[value % len(PALETTE)]) * upscale
        for _ in range(upscale):
            raw += b"\x00"  # filter type 0 (None) for this scanline
            raw += line

    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(
            b"IHDR",
            struct.pack(
                ">IIBBBBB", width * upscale, height * upscale, 8, 2, 0, 0, 0
            ),
        )
        + _png_chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + _png_chunk(b"IEND", b"")
    )


def render_data_uri(
    grid: tuple[tuple[int, ...], ...], upscale: int = RENDER_UPSCALE
) -> str:
    """The frame as a data: URI, ready for an image_url content block."""
    return "data:image/png;base64," + base64.b64encode(
        render_png(grid, upscale)
    ).decode("ascii")


def strip_images(messages: list[dict[str, Any]]) -> None:
    """Drop image blocks from every message, in place, keeping the text.

    Called before appending the current turn, so exactly one image is ever in
    flight. Without this every turn's board accumulates: the transcript keeps
    four turns, and four 256x256 images is thousands of tokens spent showing
    the model three boards that no longer exist and which it cannot act on.
    """
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        text = " ".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
        message["content"] = text


def normalise_grid(value: Any) -> Optional[tuple[tuple[int, ...], ...]]:
    """Coerce whatever `predict` returned into a hashable grid, or None.

    The model writes `predict`, so it can hand back a list of lists, a
    generator, ragged rows, or nonsense. Planning hashes states into a
    visited set, so anything unhashable or misshapen has to be rejected
    here rather than raising from inside the BFS frontier, where the
    failure would read as "no route exists".
    """
    if value is None:
        return None
    try:
        rows = tuple(tuple(int(cell) for cell in row) for row in value)
    except (TypeError, ValueError):
        return None
    if not rows:
        return None
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        return None
    return rows


def grid_diff(
    left: tuple[tuple[int, ...], ...],
    right: tuple[tuple[int, ...], ...],
    limit: int = 4,
) -> list[tuple[int, int, int, int]]:
    """Where two boards disagree, as (x, y, left, right), up to `limit`.

    Returned rather than counted because a forward model is corrected by
    seeing the cell it got wrong, not by being told a percentage.
    """
    out: list[tuple[int, int, int, int]] = []
    for y in range(min(len(left), len(right))):
        lrow, rrow = left[y], right[y]
        for x in range(min(len(lrow), len(rrow))):
            if lrow[x] != rrow[x]:
                out.append((x, y, lrow[x], rrow[x]))
                if len(out) >= limit:
                    return out
    return out


def verify_predictions(
    predict: Any,
    transitions: list[tuple[Any, str, Any]],
    limit: Optional[int] = None,
) -> dict[str, Any]:
    """Replay real transitions through a candidate forward model.

    This is the free evidence. Every transition in the log cost a billed
    action when it happened; checking a hypothesis against it again costs
    nothing, so the model can iterate on `predict` as many times as it
    likes without touching the score. That asymmetry is the point of the
    whole design -- being wrong in the simulator is free, being wrong in
    the environment is quadratic.
    """
    rows = transitions[-limit:] if limit else list(transitions)
    checked = correct = errors = 0
    wrong: list[dict[str, Any]] = []

    for before, label, after in rows:
        checked += 1
        try:
            got = normalise_grid(predict(before, label))
        except Exception as exc:  # noqa: BLE001 - the model wrote this
            errors += 1
            if len(wrong) < 5:
                wrong.append(
                    {"action": label, "raised": f"{type(exc).__name__}: {exc}"}
                )
            continue
        if got == after:
            correct += 1
        elif len(wrong) < 5:
            wrong.append(
                {
                    "action": label,
                    "cells_wrong": sum(
                        1
                        for y in range(min(len(got or ()), len(after)))
                        for x in range(min(len((got or ())[y]), len(after[y])))
                        if got[y][x] != after[y][x]
                    )
                    if got
                    else None,
                    "sample": grid_diff(got, after) if got else None,
                    "note": None if got else "predict did not return a grid",
                }
            )

    return {
        "checked": checked,
        "correct": correct,
        "accuracy": (correct / checked) if checked else 0.0,
        "raised": errors,
        "wrong": wrong,
    }


def name_goals(is_goal: Any) -> list[tuple[str, Any]]:
    """Normalise `is_goal` to a list of (name, function) pairs.

    Accepts one callable, a list of them, or a dict of named ones. The
    names come back in `routes`, and telling the candidates apart is the
    entire point of passing more than one, so duplicates -- which a row
    of lambdas produces by default -- get an index appended.
    """
    if callable(is_goal):
        return [(getattr(is_goal, "__name__", None) or "goal", is_goal)]

    if isinstance(is_goal, dict):
        pairs = [(str(k), v) for k, v in is_goal.items() if callable(v)]
    elif isinstance(is_goal, (list, tuple)):
        pairs = [
            (getattr(fn, "__name__", None) or f"goal{i}", fn)
            for i, fn in enumerate(is_goal)
            if callable(fn)
        ]
    else:
        return []

    counts: dict[str, int] = {}
    for name, _ in pairs:
        counts[name] = counts.get(name, 0) + 1
    seen: dict[str, int] = {}
    out = []
    for name, fn in pairs:
        if counts[name] > 1:
            seen[name] = seen.get(name, 0) + 1
            name = f"{name}{seen[name]}"
        out.append((name, fn))
    return out


def _plan_result(
    routes: dict[str, list[Any]],
    goals: list[tuple[str, Any]],
    nodes: int,
    reason: str,
) -> dict[str, Any]:
    """Assemble what `plan` hands back, from whatever the search reached.

    Partial success is still success when several goals were asked for:
    finding routes to six candidates out of ten and then running out of
    budget is six hypotheses tested for free, not a failure.
    """
    names = [n for n, _ in goals]
    found = [n for n in names if n in routes]
    if not found:
        return {
            "found": False, "actions": [], "nodes": nodes,
            "routes": {}, "reason": reason,
        }
    best = min(found, key=lambda n: len(routes[n]))
    out: dict[str, Any] = {
        "found": True,
        "actions": list(routes[best]),
        "nodes": nodes,
        "routes": {n: list(routes[n]) for n in found},
        "reason": reason,
    }
    if len(goals) > 1:
        out["goal"] = best
        out["unreachable"] = [n for n in names if n not in routes]
    return out


def plan_actions(
    predict: Any,
    is_goal: Any,
    start: tuple[tuple[int, ...], ...],
    actions: list[Any],
    *,
    max_nodes: int = PLAN_MAX_NODES,
    max_depth: int = PLAN_MAX_DEPTH,
    seconds: float = PLAN_SECONDS,
    first_only: bool = False,
) -> dict[str, Any]:
    """Breadth-first search over the model's simulator for a route.

    Returns the shortest action sequence reaching a state `is_goal`
    accepts. Shortest matters literally here: the plan's length is what
    gets billed, and the score is its square.

    `is_goal` may instead be a list or a dict of candidate goal
    functions, in which case one pass finds the shortest route to EACH of
    them and reports them all in `routes`. On an unfamiliar game the goal
    is a hypothesis like any other, and the whole reason to test ten of
    them here is that testing them by execution costs ten times as much.
    `first_only` stops at the nearest one instead, which is what
    `plan_tour` wants for each leg.

    Nothing about this trusts `predict`. It may raise, loop, or return
    junk on any state; each of those just prunes that branch. A simulator
    that is wrong produces a plan that fails on execution, which is
    detected there and fed back -- it does not produce a crash here.
    """
    goals = name_goals(is_goal)
    if not goals:
        return {
            "found": False, "actions": [], "nodes": 0, "routes": {},
            "reason": "is_goal must be a function, or a list or dict of them",
        }
    if not actions:
        return {
            "found": False, "actions": [], "nodes": 0, "routes": {},
            "reason": "no actions given",
        }

    deadline = min(time.monotonic() + max(seconds, 0.0), _DEADLINE)
    routes: dict[str, list[Any]] = {}

    def record(state: Any, path: list[Any]) -> bool:
        """Note every goal this state satisfies. True when there is no point going on."""
        for name, fn in goals:
            if name in routes:
                continue
            try:
                if fn(state):
                    routes[name] = list(path)
            except Exception:  # noqa: BLE001 - one bad candidate, not a bad search
                continue
        return bool(routes) if first_only else len(routes) == len(goals)

    # A lone `is_goal` that raises on the very first board is nearly
    # always a typo in the model's own code. Searching on regardless is
    # how that goes unnoticed for a whole turn, so it is reported. With
    # several candidates a single broken one is just ignored -- the
    # others are still worth searching for.
    if len(goals) == 1:
        try:
            if goals[0][1](start):
                return _plan_result(
                    {goals[0][0]: []}, goals, 0,
                    "the current board already satisfies is_goal",
                )
        except Exception as exc:  # noqa: BLE001 - the model wrote this
            return {
                "found": False, "actions": [], "nodes": 0, "routes": {},
                "reason": f"is_goal raised on the current board: "
                          f"{type(exc).__name__}: {exc}",
            }
    elif record(start, []):
        return _plan_result(
            routes, goals, 0, "the current board already satisfies every goal"
        )

    frontier: deque[tuple[tuple[tuple[int, ...], ...], list[Any]]] = deque(
        [(start, [])]
    )
    seen = {start}
    nodes = 0
    predict_errors = 0
    depth_hit = False

    while frontier:
        if nodes >= max_nodes:
            return _plan_result(
                routes, goals, nodes, f"gave up after {nodes} simulated states"
            )
        if time.monotonic() >= deadline:
            return _plan_result(
                routes, goals, nodes,
                f"planning ran out of time after {nodes} states",
            )

        state, path = frontier.popleft()
        if len(path) >= max_depth:
            depth_hit = True
            continue

        for act in actions:
            nodes += 1
            try:
                nxt = normalise_grid(predict(state, act))
            except Exception:  # noqa: BLE001 - a bad branch, not a bad search
                predict_errors += 1
                continue
            if nxt is None or nxt in seen:
                continue
            route = path + [act]
            if record(nxt, route):
                done = routes[min(routes, key=lambda n: len(routes[n]))]
                return _plan_result(
                    routes, goals, nodes,
                    f"found a {len(done)}-action route" if len(goals) == 1
                    else f"found routes to {len(routes)} of {len(goals)} goals",
                )
            seen.add(nxt)
            frontier.append((nxt, route))

    reason = "explored every reachable state without reaching is_goal"
    if depth_hit:
        reason += f" within {max_depth} actions"
    if predict_errors:
        reason += f"; predict raised on {predict_errors} branches"
    return _plan_result(routes, goals, nodes, reason)


def exhaustion_note(reason: str, nodes: int, missed: int) -> Optional[str]:
    """What it means when the search runs out of world before it runs out of goals.

    v18 produced this line and nobody acted on it:

        tour #10: 37 actions through 29/42 goals (807 states,
                  explored every reachable state without reaching is_goal)

    The model's `predict` located a coloured block, moved it, and copied
    the rest of the board unchanged -- so the only states its simulator
    had were positions of that block, about eight hundred of them. It
    visited all of them and none finished the level.

    That is a proof, inside the model's own world, that the world is
    wrong. A wrong `is_goal` is one hypothesis failing; every reachable
    state failing is the state space being too small. The two need
    different repairs and only this one points at `predict`.
    """
    if "explored every reachable state" not in reason or not missed:
        return None
    return (
        f"Your simulator says only {nodes} states are reachable from here, "
        f"and the search has now visited every one of them without "
        f"satisfying {missed} of your goals. That is not a wrong goal -- "
        "one wrong goal is one hypothesis failing, and this is all of "
        "them. It means your state space is too small: `predict` is "
        "copying part of the board that the real game changes, so the "
        "winning position is not a state your simulator can even "
        "represent. Look at what your `predict` carries over untouched -- "
        "counters, doors, things the piece passes over -- and model that "
        "too. Do not write more goals until you do; there is nowhere left "
        "for them to be."
    )


def plan_tour(
    predict: Any,
    goals: Any,
    start: tuple[tuple[int, ...], ...],
    actions: list[Any],
    *,
    max_nodes: int = PLAN_MAX_NODES,
    max_depth: int = PLAN_MAX_DEPTH,
    seconds: float = PLAN_SECONDS,
) -> dict[str, Any]:
    """One billed sequence that visits every candidate goal in turn.

    The reason this exists: a perfect simulator still does not tell you
    which board completes the level. Only the environment knows that, so
    the goal has to be tested by playing, and testing candidates one at
    a time pays full price for every wrong guess.

    But `action(...)` stops the instant the level advances. So a single
    route that passes through candidate A, then B, then C is billed only
    as far as whichever one was really the goal -- the rest is never
    played. Ten hypotheses cost what the correct one costs, plus the
    wrong ones cheaper than it.

    Greedy nearest-first. Not an optimal tour, but the right bias
    anyway: the cheap hypotheses get tested first, and the expensive
    ones are the likeliest never to be paid for at all.
    """
    named = name_goals(goals)
    if not named:
        return {
            "actions": [], "checkpoints": [], "reached": [], "unreachable": [],
            "nodes": 0,
            "reason": "goals must be a function, or a list or dict of them",
        }

    deadline = min(time.monotonic() + max(seconds, 0.0), _DEADLINE)
    state = start
    route: list[Any] = []
    checkpoints: list[dict[str, Any]] = []
    remaining = list(named)
    nodes = 0
    reason = ""

    while remaining:
        left = deadline - time.monotonic()
        if left <= 0:
            reason = f"ran out of time after reaching {len(checkpoints)} goals"
            break
        leg = plan_actions(
            predict, dict(remaining), state, actions,
            max_nodes=max_nodes, max_depth=max_depth, seconds=left,
            first_only=True,
        )
        nodes += leg["nodes"]
        if not leg["found"]:
            reason = leg["reason"]
            break

        name = leg.get("goal") or next(iter(leg["routes"]))
        step = leg["actions"]

        # Walk the simulator to where this leg ends, so the next leg
        # starts from the right board. `predict` found this route, so it
        # should replay -- but it is the model's code and a route that
        # will not replay is a broken tour, not a crash.
        board = state
        try:
            for act in step:
                board = normalise_grid(predict(board, act))
                if board is None:
                    raise ValueError("predict returned something that is not a grid")
        except Exception as exc:  # noqa: BLE001
            reason = (
                f"could not replay the route to {name} through predict: "
                f"{type(exc).__name__}: {exc}"
            )
            break

        state = board
        route.extend(step)
        checkpoints.append({"after": len(route), "goal": name})
        remaining = [(n, f) for n, f in remaining if n != name]
    else:
        reason = f"a {len(route)}-action tour through all {len(named)} goals"

    reached = [c["goal"] for c in checkpoints]
    return {
        "actions": route,
        "checkpoints": checkpoints,
        "reached": reached,
        "unreachable": [n for n, _ in named if n not in reached],
        "nodes": nodes,
        "reason": reason,
    }


def describe_frame(view: dict[str, Any], limit: int = LLM_MAX_NODES) -> str:
    """Render a segmentation as prose, largest objects first.

    Ordered by size rather than by id because the cap bites on exactly the
    boards where it matters, and on those the big structures are the board
    while the tail is noise. The ids are still the segmentation's own, so
    anything named here can be looked up in `current_frame` directly.
    """
    nodes = view.get("nodes") or []
    if not nodes:
        return "(no objects: the frame is empty)"

    ranked = sorted(nodes, key=lambda n: (-n["pixels"], n["id"]))
    lines = []
    for node in ranked[:limit]:
        top, left, bottom, right = node["bbox"]
        line = (
            f"  #{node['id']} colour={node['color']} cells={node['pixels']} "
            f"rows {top}-{bottom}, cols {left}-{right}"
        )
        if node["children"]:
            line += f", encloses {node['children']}"
        lines.append(line)

    if len(ranked) > limit:
        lines.append(
            f"  ... and {len(ranked) - limit} smaller objects, in "
            f"`current_frame['nodes']`"
        )
    return "\n".join(lines)


def build_system_prompt(available: list[int]) -> str:
    """The framing the model keeps for the whole game.

    Held apart from the per-turn state so the transcript can be evicted
    down to this plus the newest turns without losing the rules of the
    game -- see `evict_turns`.

    What this deliberately does NOT say is what any action does. The
    mapping from ACTION1..5 to an effect differs per game and nothing in
    the frame data declares it, so naming one "up" would be a guess the
    model then acts on. Discovering it is the job.
    """
    names = ", ".join(f"ACTION{i}" for i in available) or "(none reported)"
    return (
        "You are solving an unfamiliar puzzle video game by writing "
        "Python.\n\n"
        "You never name a move directly. You are given one tool, `python`, "
        "and inside it a function `action(...)` that plays moves. So write "
        "code that explores, reads what came back, and decides -- a search, "
        "a flood fill, a hypothesis test. That is the whole job.\n\n"
        "THE BOARD\n"
        "A 64x64 grid of colours numbered 0-15. You get three views of it, "
        "and they are the same board:\n"
        "  * an image of it, upscaled so one cell is a block of pixels;\n"
        "  * the same board parsed into objects, listed below;\n"
        "  * the raw grid, in code, as `grid[y][x]`.\n"
        "Look at the image first to see the shape of the problem, then "
        "work from the OBJECTS. They are your primary view: they carry "
        "colour, size, containment, adjacency, and a position-independent "
        "hash that lets you follow one object across frames or spot two "
        "identical ones in a single board. Reach for `grid[y][x]` to "
        "check a specific cell, never to scan the whole board -- a field "
        "of 4096 integers invites arithmetic about coordinates when the "
        "question is nearly always about objects. Trust the numbers over "
        "your reading of the picture when they disagree.\n\n"
        "ACTIONS\n"
        f"This game accepts: {names}. ACTION6 is a click and needs "
        "coordinates; the rest take none. RESET (ACTION0) restarts and is "
        "always accepted.\n"
        "Nothing tells you what any action does. Two games number them "
        "differently. Find out by trying.\n\n"
        "HOW TO WIN THIS\n"
        "Every action you play is billed and the penalty is quadratic, so "
        "the winning strategy is not to search the board -- it is to learn "
        "the rule, then spend actions only on a route you have already "
        "checked. You are given tools that make everything except the "
        "final route free, and the method is exactly these four steps:\n"
        "  1. PROBE. Spend a handful of actions -- about ten, and no more "
        "than twenty -- to see what each action does. Play each action "
        "ONCE and read the result. This is the only exploring you should "
        "pay for.\n"
        "     NEVER write `for i in range(10): action(...)`. That is ten "
        "billed actions buying one fact, and it is the most expensive "
        "mistake available to you.\n"
        "  2. WRITE THE SEARCH. This is the job, and it is the whole of "
        "it. Once you can say roughly what the actions do and what you "
        "are trying to reach, write the algorithm that gets you there and "
        "run it inside `python`: breadth-first search, flood fill, "
        "shortest path, beam search, a bounded search over action "
        "sequences, a scorer over candidate targets. When a game is about "
        "getting something to a place, it is almost always a BFS, and "
        "writing one explicitly beats working the route out by eye every "
        "single time. Pass what it returns to `action([...])`.\n"
        "     You have bisect, collections, copy, fractions, functools, "
        "heapq, itertools, json, math, operator, random, re, statistics "
        "and string. The namespace persists between calls and between "
        "turns, so anything you define stays defined and you can build "
        "up your own helpers over the game.\n"
        "  3. STOP PROBING once the state variables and the action "
        "effects are understood well enough, and search the space you "
        "have inferred instead of paying for more looking.\n"
        "  4. EXECUTE the shortest sequence your search returned, then "
        "re-ground. Read the new board, and check that a real object "
        "moved before believing the route worked -- `last_action_result` "
        "is still there on your next call and carries `changed`, "
        "`levels_completed`, `steps` and `stopped`. Fewer steps than you "
        "asked for means it stopped early and `stopped` says why.\n"
        "     If the route played out and the level did not advance, the "
        "thing that was wrong was WHERE you were going, not how you got "
        "there. Do not go back to probing. Pick a different target and "
        "search again.\n\n"
        "AN OPTIONAL SHORTCUT, NOT THE ASSIGNMENT\n"
        "If -- and only if -- the rule turns out simple enough that you "
        "can write a forward model you actually trust, there are three "
        "tools that will search it for you and bill nothing. Write "
        "`predict(grid, act) -> grid` and `verify(predict)` scores it "
        "against every move already played; `plan(predict, is_goal)` "
        "breadth-first searches it. They are a convenience for the easy "
        "case.\n"
        "They are NOT the method, and reaching for them by default is a "
        "known way to lose this. A simulator that is subtly wrong plans "
        "confident routes to places that do not exist, and one that "
        "raises is worse still because the search silently returns "
        "nothing. If `predict` keeps raising, or `verify` will not climb, "
        "abandon it -- do not repair it. Search the real game with your "
        "own code instead: it is always available, it needs no forward "
        "model, and it cannot be wrong about a transition it has actually "
        "observed.\n"
        "Your accuracy is re-checked for you against every move you have "
        "played, at the end of every turn, whether you ask or not. A "
        "score of 1.0 measured once over a dozen moves is not a "
        "simulator you can trust two hundred actions later, and you will "
        "be told when it has drifted.\n"
        "Being wrong in the simulator is free. Being wrong in the game is "
        "quadratic. Do your being-wrong in the simulator.\n"
        "WHERE THE FINISH IS, IS ALSO A GUESS\n"
        "Nothing tells you what completes a level, and only the game can "
        "confirm it, so the goal is a hypothesis exactly like the rule is. "
        "The expensive mistake is testing one guessed target per "
        "execution: three wrong guesses at five actions each is fifteen "
        "billed actions and no level.\n"
        "Test them together instead. Write every target you find "
        "plausible -- each highlighted square, each place the movable "
        "piece could line up with something, each edge -- and call "
        "`tour(predict, {...})`. It returns ONE sequence that passes "
        "through all of them, cheapest first, and it bills nothing to "
        "build. Play that sequence and it STOPS the moment the level "
        "advances, so the candidates after the real goal are never played "
        "at all. Ten hypotheses for the price of the one that was right.\n"
        "The two failures to avoid: building a good `predict` and then "
        "never calling `plan`; and, once you are planning, guessing "
        "targets one at a time and paying for each. If you have a "
        "`predict` worth anything at all, plan with it -- and if you do "
        "not know what to aim at, tour every candidate at once.\n"
        "Your variables persist between turns, so the model you build and "
        "the notes you keep survive. Build them once and improve them; do "
        "not start over each turn.\n"
        "Put durable findings in `notes` (a list) -- what each action does, "
        "the rule, what failed. `notes` is shown back to you every turn "
        "even after the rest of this conversation has been trimmed away, so "
        "anything not in `notes` you will forget.\n"
        "A loop that just repeats one action 60 times is the single "
        "worst thing you can do: it is 60 billed actions buying one bit of "
        "information.\n\n"
        "HOW TO ACT\n"
        "  action('ACTION1')            one move\n"
        "  action(['ACTION1', 'ACTION1'])  a sequence\n"
        "  action(('ACTION6', 12, 30))  a click at column 12, row 30\n"
        "It returns a dict: `changed` (did the board differ), `state`, "
        "`levels_completed`, `frame` (the new segmentation), `steps` "
        "(how many ran -- fewer than you asked means it stopped, and "
        "`stopped` says why), `mispredicted` (once you have defined "
        "`predict`: every step where the real board disagreed with it), "
        "and sometimes `goal_miss`, which appears when a route ran "
        "perfectly and still did not finish the level -- read it, it is "
        "telling you `is_goal` is the thing that is wrong.\n"
        "Coordinates are (x=column, y=row), both 0-63.\n\n"
        "THE BAR AT THE EDGE IS NOT THE GAME\n"
        "A long run of small blocks flush against a border, that shrinks "
        "or changes every step while the interior stays put, is a timer "
        "or a step counter. It reaches you as a row of tiny objects in "
        "the segmentation and it is not a row of clickable pieces. "
        "Clicking through it block by block is the most expensive way "
        "there is to learn nothing. After any action, check that a real "
        "object moved before believing the action did anything: a board "
        "that changed only inside that strip did not change.\n\n"
        "THE THREE FREE TOOLS\n"
        "  verify(predict)\n"
        "      Replays every real move made so far through your `predict` "
        "and returns {accuracy, checked, correct, raised, wrong}. `wrong` "
        "lists the action and a sample of disagreeing cells as "
        "(x, y, predicted, actual). Costs no actions.\n"
        "  plan(predict, is_goal, actions=None, max_depth=60)\n"
        "      Breadth-first search over your simulator. Returns "
        "{found, actions, nodes, reason}; `actions` is the shortest route "
        "and goes straight into action(...). `actions=` defaults to the "
        "non-click moves -- for a click game, pass your own shortlist of "
        "targets like [('ACTION6', 12, 30), ...], because there are 4096 "
        "of them and searching all of them finds nothing. Costs no "
        "actions.\n"
        "      `is_goal` can also be a dict of candidates, "
        "{'name': fn, ...}, and you get the shortest route to each of "
        "them in `routes` from the one search.\n"
        "  tour(predict, goals, actions=None, max_depth=60)\n"
        "      `goals` is a dict {'name': is_goal_fn, ...}. Returns one "
        "sequence visiting every one of them, nearest first: "
        "{actions, checkpoints, reached, unreachable, nodes, reason}. "
        "`actions` goes straight into action(...), and because execution "
        "stops at the level change, you are billed only up to whichever "
        "candidate was the real goal. This is how you test what the "
        "finish is without paying per guess. Costs no actions to build.\n\n"
        "WHAT YOU START WITH\n"
        "  current_frame   the segmentation now\n"
        "  previous_frame  the one before it, or None\n"
        "  grid            the raw 64x64 tuple of tuples, grid[y][x]\n"
        "  history         recent (action, changed) pairs\n"
        "  valid_actions   the list above\n"
        "  notes           your own list, carried across turns\n"
        "  action, verify, plan, tour   the tools above\n"
        "These are refreshed each turn. Everything else you define stays "
        "exactly as you left it -- including `predict` and `is_goal`, so "
        "build them once and keep improving them rather than starting "
        "over.\n"
        "Each object has: id, color, pixels, bbox (top, left, bottom, "
        "right), hash (equal hashes mean the same shape in the same "
        "colour, so you can follow one across frames), children (ids it "
        "encloses).\n\n"
        "SCORING -- read this twice\n"
        "You are scored on how FEW actions each level takes, and the "
        "penalty is quadratic: taking twice the intended number scores a "
        "quarter, ten times scores about one point in a hundred. Finishing "
        "a level slowly is worth almost nothing. There is exactly one "
        "attempt per game and every action is billed, including the ones "
        "you spend exploring and including RESET.\n"
        "Two things follow that are not obvious:\n"
        "  * Beating the intended count earns you nothing extra -- the "
        "score is capped. Aim to match it, not to beat it.\n"
        "  * Later levels are worth much more than level 1, and they allow "
        "more actions. Level 1 is the tutorial: it is worth very little, "
        "and it is the right place to spend actions learning the rules. "
        "Get the rules right there, then play the later levels tightly. "
        "Do not grind level 1 for a low action count at the cost of not "
        "understanding the game.\n\n"
        "THE TOOL\n"
        "Variables persist between your `python` calls within a turn. "
        "Available modules: "
        f"{', '.join(sorted(TOOL_MODULES))}. No file, network or process "
        f"access. Each call is killed after {TOOL_TIMEOUT:.0f} seconds, so "
        "bound your loops.\n"
        "Print what you want to see; only stdout and the last expression "
        "come back, truncated if long.\n\n"
        "Stop calling the tool and answer in words when you have acted and "
        "want to see the board fresh."
    )


def render_notes(notes: Any, limit: int = LLM_NOTES_CHARS) -> str:
    """The model's own durable findings, read back to it.

    Eviction trims the middle of the transcript, so by turn 15 the rule it
    worked out on turn 3 is gone. This is the channel that survives that,
    which is why the prompt tells it plainly that anything not in `notes`
    it will forget.
    """
    if not notes:
        return ""
    if isinstance(notes, dict):
        items = [f"{k}: {v}" for k, v in notes.items()]
    elif isinstance(notes, (list, tuple)):
        items = [str(n) for n in notes]
    else:
        items = [str(notes)]

    lines, used = [], 0
    for item in items:
        item = item.replace("\n", " ")
        if used + len(item) > limit:
            lines.append(f"  ... {len(items) - len(lines)} more, trimmed")
            break
        lines.append(f"  - {item}")
        used += len(item)
    return "\nWhat you have worked out so far:\n" + "\n".join(lines) + "\n"


def build_turn_prompt(
    view: dict[str, Any],
    history: list[str],
    level: int,
    steps: int,
    note: str = "",
    notes: Any = None,
) -> str:
    """The state message for one turn."""
    seen = "\n".join(f"  {line}" for line in history[-LLM_HISTORY:])
    recent = f"\nRecently:\n{seen}\n" if seen else "\n"
    head = f"{note}\n\n" if note else ""
    return (
        f"{head}Level {level}. {steps} actions used so far.\n\n"
        f"Objects on the board now:\n{describe_frame(view)}\n"
        f"{render_notes(notes)}"
        f"{recent}\n"
        "Work out what to do and do it."
    )


def split_action_specs(steps: Any) -> list[Any]:
    """`action(...)`'s argument as a list of individual action specs.

    The ambiguity is real and worth stating: ``('ACTION6', 12, 30)`` is
    ONE click, while ``('ACTION1', 'ACTION2', 'ACTION3')`` is three moves.
    They are told apart by the only thing that actually distinguishes
    them -- a click carries two integers after the name.

    ``[1, 2, 3]`` is therefore read as a click, not as three moves. That
    is not silently wrong: `parse_action_spec` then rejects it, because
    ACTION1 takes no coordinates, and the model is told to use names.
    Guessing which reading was meant is the one thing not to do here.
    """
    if isinstance(steps, (str, bytes)) or not isinstance(steps, (list, tuple)):
        return [steps]
    if (
        len(steps) == 3
        and isinstance(steps[0], (str, int))
        and not isinstance(steps[0], bool)
        and all(isinstance(v, int) and not isinstance(v, bool) for v in steps[1:])
    ):
        return [steps]
    return list(steps)


def name_action(action_id: int, x: Optional[int], y: Optional[int]) -> str:
    """How an executed action is written back into `history`."""
    if action_id == _RESET_ID:
        return "RESET"
    if x is None:
        return f"ACTION{action_id}"
    return f"ACTION{action_id}({x},{y})"


def parse_action_spec(
    item: Any, available: list[int]
) -> tuple[int, Optional[int], Optional[int]]:
    """One item the model passed to `action()`, as (action_id, x, y).

    Accepts what a model plausibly writes: ``'ACTION1'``, ``1``,
    ``('ACTION6', 12, 30)``, ``[6, 12, 30]``, ``'reset'``.

    Every rejection raises `ValueError` with a message written to be read
    by the model, because that is exactly where it goes -- back as the
    tool result, for it to correct. A refused action costs nothing; a
    silently reinterpreted one costs a billed step and teaches the model
    something false about the game.
    """
    x: Any = None
    y: Any = None
    if isinstance(item, (list, tuple)):
        if not item:
            raise ValueError("an empty sequence is not an action")
        head, rest = item[0], list(item[1:])
        if len(rest) == 2:
            x, y = rest
        elif rest:
            raise ValueError(
                f"expected an action or (action, x, y), got {len(item)} "
                f"items: {item!r}"
            )
    else:
        head = item

    if isinstance(head, bool):  # bool is an int; nobody means action True
        raise ValueError(f"{head!r} is not an action")
    if isinstance(head, int):
        action_id = head
    elif isinstance(head, str):
        text = head.strip().upper().replace(" ", "")
        if text in ("RESET", "ACTION0"):
            action_id = _RESET_ID
        else:
            match = re.fullmatch(r"ACTION(\d+)", text)
            if match is None:
                raise ValueError(
                    f"{head!r} is not an action name. Use 'ACTION1'.."
                    f"'ACTION6' or 'RESET'."
                )
            action_id = int(match.group(1))
    else:
        raise ValueError(f"{head!r} is not an action name or id")

    if action_id != _RESET_ID and action_id not in available:
        allowed = ", ".join(f"ACTION{a}" for a in available) or "none"
        raise ValueError(
            f"ACTION{action_id} is not accepted by this game. "
            f"Accepted: {allowed} (and RESET)."
        )

    if action_id == _ACTION6_ID:
        if x is None:
            raise ValueError(
                "ACTION6 is a click and needs coordinates: "
                "action(('ACTION6', x, y))"
            )
        if isinstance(x, bool) or isinstance(y, bool):
            raise ValueError("click coordinates must be integers")
        if not isinstance(x, int) or not isinstance(y, int):
            raise ValueError(f"click coordinates must be integers, got ({x!r}, {y!r})")
        if not (0 <= x <= 63 and 0 <= y <= 63):
            raise ValueError(
                f"click ({x}, {y}) is off the board; x and y are both 0-63"
            )
        return action_id, x, y

    if x is not None:
        raise ValueError(f"ACTION{action_id} takes no coordinates; only ACTION6 does")
    return action_id, None, None


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


def _widget_span(
    cell_changes: list[int],
    frames: list[bytes],
    start: int,
    stride: int,
    length: int,
) -> Optional[tuple[int, int]]:
    """Offsets of the first and last cell of the widget ticking on a line.

    `start`/`stride`/`length` address one row (start=y*width, stride=1)
    or one column (start=x, stride=width). Returns None when nothing on
    the line ever moved, which cannot happen for a line that qualified
    but is cheap to be safe about.

    The cells that changed during warmup are only *part* of the widget.
    A bar that drains one segment per step shows nothing but its leading
    edge in a short warmup: the undrained tail never changes, so a span
    built from changed cells alone stops short and leaves most of the
    counter in the state key. That is worse than masking the whole row,
    because a counter in the key mints a fresh node every single step.
    Measured on ls20: changed-cells-only masks 50 of the counter's 84
    cells and the map goes from 96 nodes to 735.

    So grow outward from the changed cells while the colour is one those
    cells themselves have worn at some point in warmup -- that is the
    rest of the same widget -- and stop at the first pixel that is not.
    Striking the line's dominant colour from that palette first looks
    like the right guard against a widget drawn in the background
    colour, and is not: a bar wide enough to be worth masking is often
    the most common thing on its own row, so the guard deletes exactly
    the colour the extension needs. The check that works is on the
    result -- an extension that swallows the whole line never found a
    boundary, so fall back to the cells actually seen moving.
    """
    changed = [o for o in range(length) if cell_changes[start + o * stride]]
    if not changed:
        return None
    first, last = changed[0], changed[-1]

    usable = [f for f in frames if len(f) > start + (length - 1) * stride]
    if not usable:
        return (first, last)
    latest = usable[-1]
    line = [latest[start + o * stride] for o in range(length)]
    palette = {f[start + o * stride] for f in usable for o in changed}

    grown_first, grown_last = first, last
    while grown_first > 0 and line[grown_first - 1] in palette:
        grown_first -= 1
    while grown_last < length - 1 and line[grown_last + 1] in palette:
        grown_last += 1

    # Growing across the entire line means the widget's colours are also
    # the line's background, so there was never a boundary to find and
    # the "widget" is indistinguishable from everything beside it. Fall
    # back to what was actually seen moving rather than mask the lot --
    # masking a whole line is the failure this function exists to avoid.
    if grown_first == 0 and grown_last == length - 1 and (first, last) != (0, length - 1):
        return (first, last)
    return (grown_first, grown_last)


def status_band(
    row_changes: list[int],
    col_changes: list[int],
    volatile: frozenset[int],
    samples: int,
    width: int,
    height: int,
    threshold: float = VOLATILE_THRESHOLD,
) -> tuple[int, ...]:
    """Kept cells that share a line with the HUD -- the status readout.

    A widget that ticks every step is a counter and gets masked. What
    sits beside it on the same rows is usually the rest of the same
    status bar: lives, a score, and -- on ls20 -- a 10x10 box that is
    the *only* place the avatar's shape, colour and rotation are drawn.

    Those cells matter for a reason the search cannot otherwise reach.
    When the agent walks over a tile that recolours or rotates it, the
    board does not change at all; only this readout does. Every wall the
    agent has bounced off, every node it has written off as exhausted,
    was concluded under the old reading and may be wrong under the new
    one -- on ls20 the level's goal square is passable only at the right
    rotation, so it is literally a wall that stops being one. Watching
    this band is what lets the agent notice it has become a different
    thing and re-open what it thought it had settled.

    Returns the flat indices in ascending order, empty when no line
    qualified or when the band would be so large it is really the board.
    """
    total = width * height
    if samples <= 0 or total <= 0:
        return ()
    cutoff = threshold * samples
    rows = {y for y, count in enumerate(row_changes) if count >= cutoff}
    cols = {x for x, count in enumerate(col_changes) if count >= cutoff}
    if not rows and not cols:
        return ()
    band = tuple(
        index
        for index in range(total)
        if index not in volatile
        and (index // width in rows or index % width in cols)
    )
    return () if len(band) > STATUS_BAND_CAP * total else band


def hud_mask(
    cell_changes: list[int],
    row_changes: list[int],
    col_changes: list[int],
    samples: int,
    width: int,
    height: int,
    threshold: float = VOLATILE_THRESHOLD,
    cap_fraction: float = MASK_CAP_FRACTION,
    frames: Optional[list[bytes]] = None,
    span: Optional[bool] = None,
) -> frozenset[int]:
    """Flat cell indices to exclude from the node key, or an empty set.

    A cell qualifies if it changed in `threshold` of transitions, and so
    does the *ticking span* of any row or column that contained some
    change in that fraction of transitions. The row/column rule is the
    one that matters: a shrinking progress bar advances one segment per
    step, so no individual bar cell is volatile, but the bar's rows
    change constantly.

    The span is what keeps that rule honest. Masking the whole row is
    collateral damage on anything that shares it, and on ls20 the thing
    sharing it is the only display of the avatar's shape, colour and
    rotation -- a 10x10 box at rows 55-64, cols 3-12, whose bottom two
    rows are also the step counter's. Those three attributes are the
    entire puzzle (you clear a level by standing on a target while
    matching its triple) and they appear nowhere else on the board, so
    masking them fuses states that differ by the only thing that counts.
    Restricting to the span that actually ticks can only re-admit cells
    that never changed during warmup, so it cannot reintroduce a
    counter; it just stops the counter taking its neighbours with it.

    Returns empty if nothing qualifies or if more than `cap_fraction` of
    the frame does -- at that point the reading is not a HUD, and masking
    it would fuse genuinely different states into one node.
    """
    total = width * height
    if samples <= 0 or total <= 0:
        return frozenset()

    kept_frames = [f for f in (frames or []) if len(f) == total]
    by_span = MASK_SPAN if span is None else span
    cutoff = threshold * samples
    masked = {index for index, count in enumerate(cell_changes) if count >= cutoff}
    for y, count in enumerate(row_changes):
        if count < cutoff:
            continue
        span = (
            _widget_span(cell_changes, kept_frames, y * width, 1, width)
            if by_span else (0, width - 1)
        )
        if span is None:
            continue
        masked.update(range(y * width + span[0], y * width + span[1] + 1))
    for x, count in enumerate(col_changes):
        if count < cutoff:
            continue
        span = (
            _widget_span(cell_changes, kept_frames, x, width, height)
            if by_span else (0, height - 1)
        )
        if span is None:
            continue
        masked.update(range(x + span[0] * width, x + span[1] * width + 1, width))

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
# What the actions actually do
# ---------------------------------------------------------------------

# Learning this costs no actions: it reads frames the search was handed
# anyway. It is budgeted only because `segment_frame` is the most
# expensive thing in the agent, and there is no point paying for it once
# the answer has stopped changing.
MOTION_LEARN = _env_int("ARC_MOTION_LEARN", 1)
MOTION_SAMPLES = _env_int("ARC_MOTION_SAMPLES", 4)
MOTION_BUDGET = _env_int("ARC_MOTION_BUDGET", 300)

# Re-finding the avatar costs a full segmentation, so losing it must not
# become a per-action habit. Tracking simply stays off once the budget is
# gone; everything downstream already treats "we do not know where we
# are" as the normal case.
MOTION_ANCHORS = _env_int("ARC_MOTION_ANCHORS", 200)

# A fresh anchor is a guess until an action confirms it. Skipping probes
# on the strength of an unconfirmed position is how the first version of
# this cost ls20 the level entirely: one wrong anchor near a respawn and
# the search declines the moves that would have finished it.
BLOCKED_CONFIRM = _env_int("ARC_BLOCKED_CONFIRM", 1)

# The readout: cells that change, but not because we moved, and not often.
#
# The blocked map needs a key for "what the avatar currently is", because
# whether a square is passable is not a property of the square. Two
# guesses at that key measured wrong -- the status band reads the three
# lives, and ls20 does not draw its attributes on the board sprite at all
# -- so this one was found rather than guessed. Driving ls20 directly
# with its own shape/colour/rotation fields as ground truth, the cells
# whose value is a function of rotation are exactly sixteen of them, at
# rows 55-60 and columns 3-8: a small, compact widget, and one that
# `hud_mask` never touched because it does not share the step counter's
# rows.
#
# What generalises is not the coordinates but how they stand out. A cell
# changes for one of four reasons: the avatar moved through it, a hazard
# patrolled through it, a readout updated, or it never changes at all.
# Subtract the avatar's own footprint -- which needs the tracking the
# previous commit built -- and rate is enough to separate the rest. A
# hazard moves most actions; a readout updates only when something about
# us changes; the board does not move.
#
# The filter is deliberately generous. Including extra cells only makes a
# blocked entry shared less often; excluding the ones that matter is what
# made ls20 stop clearing level 1 at all. Over-include.
READOUT_WARMUP = _env_int("ARC_READOUT_WARMUP", 250)
READOUT_MAX_RATE = _env_float("ARC_READOUT_MAX_RATE", 0.15)
READOUT_CAP = _env_int("ARC_READOUT_CAP", 400)

# Skip probes that walk into something that has already refused us.
#
# Off, because the map cannot yet be indexed by the thing that decides
# whether a square is passable. Learning it is free and correct -- ls20
# finishes level 1 in the same 527 actions with 55 real walls recorded --
# but spending it needs a key, and both keys available are wrong:
#
#   the status readout   takes THREE values over a whole level, measured.
#                        That is the three lives, not the twenty-four
#                        attribute combinations.
#   the avatar's looks   takes ONE. ls20 does not draw shape, colour or
#                        rotation on the board sprite at all; the only
#                        display of them is the HUD widget, which is
#                        exactly what `hud_mask` removes.
#
# With either key the goal square -- which refuses the avatar until all
# three attributes match, and so reads precisely like a wall for most of
# the level -- gets skipped forever, and ls20 goes from clearing level 1
# in 527 actions to not clearing it in 1500. The mechanism is sound and
# the evidence for that is that it fires: 124 probes skipped on a map
# whose no-op presses number about 97. It is aimed at the right cost. It
# just cannot tell a wall from a locked door.
BLOCKED_SKIP = _env_int("ARC_BLOCKED_SKIP", 0)


class MotionModel:
    """Which object the player controls, and where each action moves it.

    The search's memory is a graph of opaque frame hashes, which is why
    `_reset_graph` has to throw all of it away at every level boundary:
    the ids stop meaning anything the moment the board changes. This is
    the part of what a game teaches that does NOT expire. "ACTION1 moves
    me up one cell" is true on level 1 and still true on level 7.

    ls20's own level table is the argument for bothering. Levels 1, 2
    and 3 are the same puzzle with a different target rotation, and
    levels 1 to 4 start the avatar in the same shape; the search
    currently re-derives the whole thing from nothing, every time, and
    in this run reaches level 2 on one game out of nine.

    Objects are matched across a frame pair by `segment_frame`'s hash,
    which is normalised to the bounding box and so is invariant under
    exactly the translation being measured. Only hashes appearing once
    in both frames are used; a repeated shape cannot be followed without
    guessing which copy is which.

    Several things move at once on a game with hazards, so a
    displacement is not credited to the avatar merely for happening.
    Every candidate keeps a per-action histogram, and the avatar is the
    one whose displacement is a *function of the action* -- consistent
    within an action, and different between actions. A sprite that
    patrols one square per tick moves identically whatever we pressed,
    scores one distinct displacement, and is rejected.
    """

    def __init__(self) -> None:
        # object hash -> action id -> displacement -> times seen
        self._moves: dict[int, dict[int, Counter]] = {}
        self._table: dict[int, tuple[int, int]] = {}
        self._avatar: Optional[int] = None
        self._family: frozenset[int] = frozenset()
        self._dirty = True
        self.samples = 0
        self._near = (0, 0)

    @staticmethod
    def _unique(view: dict[str, Any]) -> dict[int, tuple[int, int]]:
        """Hash -> top-left corner, for shapes appearing exactly once."""
        seen: dict[int, Any] = {}
        for node in view.get("nodes", ()):
            key = node["hash"]
            if key in seen:
                seen[key] = None          # ambiguous from here on
            else:
                top, left, _, _ = node["bbox"]
                seen[key] = (top, left)
        return {k: v for k, v in seen.items() if v is not None}

    def observe(
        self,
        action_id: int,
        before: dict[str, Any],
        after: dict[str, Any],
    ) -> None:
        if not before or not after:
            return
        first = self._unique(before)
        second = self._unique(after)
        for key, start in first.items():
            end = second.get(key)
            if end is None or end == start:
                continue
            step = (end[0] - start[0], end[1] - start[1])
            self._moves.setdefault(key, {}).setdefault(
                action_id, Counter()
            )[step] += 1
            self._dirty = True
        self.samples += 1

    def _solve(self) -> None:
        self._dirty = False
        best: Optional[tuple[int, dict[int, tuple[int, int]]]] = None
        best_score = (0, 0)
        # How close the runner-up came, so a failure can say which kind
        # it is. A candidate explaining several actions with ONE distinct
        # step is something moving on a timer, not something answering to
        # us. A candidate explaining none has no dominant outcome at all,
        # which means either the avatar is being re-drawn between frames
        # -- a changing shape hashes differently and cannot be followed
        # -- or the displacement is not constant, as it is not for a slide
        # that runs until it hits a wall. The two want different fixes and
        # guessing between them wastes a run.
        near = (0, 0)
        for key, per_action in self._moves.items():
            table: dict[int, tuple[int, int]] = {}
            for action_id, hist in per_action.items():
                step, count = hist.most_common(1)[0]
                # A clear winner, not merely the tallest of a scatter:
                # anything the action does not reliably cause shows up
                # as a near-tie and is refused.
                if count < MOTION_SAMPLES or count * 2 <= sum(hist.values()):
                    continue
                table[action_id] = step
            distinct = len(set(table.values()))
            near = max(near, (len(table), distinct))
            if distinct < 2:
                # Moves the same way whatever we pressed, so it is not
                # responding to us.
                continue
            score = (distinct, len(table))
            if score > best_score:
                best_score, best = score, (key, table)
        self._near = near
        if best is None:
            self._avatar, self._table, self._family = None, {}, frozenset()
            return
        self._avatar, self._table = best
        # Every appearance the avatar wears. ls20 changes the player's
        # shape and colour as it walks over modifier tiles, and a redrawn
        # sprite hashes differently, so the one hash that won the vote is
        # not enough to find it again after it crosses one. Anything
        # whose dominant steps agree with the winner wherever both are
        # defined is the same thing wearing a different face.
        family = set()
        for key, per_action in self._moves.items():
            agreed = False
            for action_id, hist in per_action.items():
                step, count = hist.most_common(1)[0]
                if count < MOTION_SAMPLES or count * 2 <= sum(hist.values()):
                    continue
                if self._table.get(action_id) != step:
                    agreed = False
                    break
                agreed = True
            if agreed:
                family.add(key)
        self._family = frozenset(family)

    def locate(self, view: dict[str, Any]) -> Optional[tuple[int, int, int, int]]:
        """Where the avatar is in this frame, as (top, left, height, width).

        `segment_frame` splits by colour, so a sprite drawn in two
        colours arrives as two components -- ls20's avatar is exactly
        this, a 5x5 cell whose upper two rows and lower three rows are
        different colours, and both halves move together and so both
        earn a place in the family. Parts that touch are one object.

        Parts that do NOT touch are two candidates for the same role,
        and there is no way to say which one we are driving. That is
        refused rather than guessed: a wrong anchor puts every later
        prediction in the wrong place, and the blocked map would then be
        recording walls that are not there.
        """
        self._resolve()
        if not self._family:
            return None
        clusters: list[list[int]] = []
        for node in view.get("nodes", ()):
            if node["hash"] not in self._family:
                continue
            top, left, bottom, right = node["bbox"]
            merged = [top, left, bottom, right]
            apart: list[list[int]] = []
            for other in clusters:
                touching = (
                    merged[0] <= other[2] + 1
                    and other[0] <= merged[2] + 1
                    and merged[1] <= other[3] + 1
                    and other[1] <= merged[3] + 1
                )
                if touching:
                    merged = [
                        min(merged[0], other[0]), min(merged[1], other[1]),
                        max(merged[2], other[2]), max(merged[3], other[3]),
                    ]
                else:
                    apart.append(other)
            apart.append(merged)
            clusters = apart
        if len(clusters) != 1:
            return None
        top, left, bottom, right = clusters[0]
        return (top, left, bottom - top + 1, right - left + 1)

    def _resolve(self) -> None:
        if self._dirty:
            self._solve()

    @property
    def avatar(self) -> Optional[int]:
        self._resolve()
        return self._avatar

    @property
    def settled(self) -> bool:
        """Enough of the action set explained to stop paying to look."""
        self._resolve()
        return len(self._table) >= 4

    def displacement(self, action_id: int) -> Optional[tuple[int, int]]:
        self._resolve()
        return self._table.get(action_id)

    def report(self) -> str:
        self._resolve()
        if not self._table:
            actions, distinct = self._near
            return (
                f"unsolved after {self.samples} pairs over "
                f"{len(self._moves)} movers; best explains {actions} "
                f"actions with {distinct} distinct steps"
            )
        moves = " ".join(
            f"ACTION{a}=({dy:+d},{dx:+d})"
            for a, (dy, dx) in sorted(self._table.items())
        )
        return f"avatar {self._avatar:08x} {moves} ({self.samples} pairs)"


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
        # Click hit rate for the current level (see LIVENESS_WIDEN). Reset at
        # each level boundary, alongside the breadth it governs.
        self._click_tried = 0
        self._click_live = 0
        self._why_counts: Counter[str] = Counter()  # billed actions by reason
        self._class_counts: Counter[str] = Counter()  # click/press hit rate
        self._last_code: Optional[int] = None     # code of the previous action
        self._last_productive = False             # ...and did it reach a new state
        self._momentum = 0                        # consecutive repeats so far
        self._act_tried: Counter[int] = Counter()    # action id -> probes
        self._act_changed: Counter[int] = Counter()  # action id -> state changes
        # Full codes (action id + x/y for a click) that have moved the board
        # at least once, anywhere in the game so far. See LIVE_CODE_PRIORITY.
        self._code_live: set[int] = set()
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
        self._status_idx: tuple[int, ...] = ()         # the status readout
        self._status: Optional[tuple[int, ...]] = None  # its last reading
        self._status_reopens = 0
        self._last_flat: Optional[tuple[int, ...]] = None
        self._level_start_node: Optional[int] = None
        self._board_idx: Optional[tuple[int, ...]] = None
        self._status_boards: dict[int, int] = {}
        self._level_start_board: Optional[int] = None
        self._last_respawn: Optional[int] = None
        self._respawn_period = 0
        self._respawn_periods: deque[int] = deque(maxlen=RESPAWN_SAMPLES)
        self._respawns = 0
        self._deadline_probes = 0
        # Early-fire bookkeeping (see EARLY_MASK): the last unanimous-change
        # mask proposed, and how many consecutive samples have agreed on it.
        self._early_candidate: frozenset[int] = frozenset()
        self._early_stable = 0
        # Warmup frames kept only until the mask is decided (see
        # MASK_COLLAPSE_GUARD), then dropped.
        self._mask_frames: list[bytes] = []
        # Checkpoint state for MASK_STRICT_SCALE. `_mask_defer_until` is
        # the next checkpoint's sample target, 0 meaning no deferral in
        # progress yet; `_mask_defer_count` how many checkpoints have
        # fired; `_mask_best_size`/`_mask_best_mask` track the smallest
        # candidate seen across all checkpoints so far.
        self._mask_defer_until = 0
        self._mask_defer_count = 0
        self._mask_best_size: Optional[int] = None
        self._mask_best_mask: frozenset[int] = frozenset()
        # Deliberately NOT cleared by `_reset_graph`: outliving the level
        # boundary is the entire point of it.
        self._motion = MotionModel() if MOTION_LEARN else None
        self._motion_view: Optional[dict[str, Any]] = None
        self._motion_said = False
        # Where the avatar is, and how big it is, in board coordinates.
        self._at: Optional[tuple[int, int]] = None
        self._at_size: tuple[int, int] = (1, 1)
        self._at_confirmed = 0
        self._look = 0
        self._readout_counts: Counter = Counter()
        self._readout_samples = 0
        self._readout_idx: Optional[tuple[int, ...]] = None
        self._track_flat: Optional[tuple[int, ...]] = None
        self._anchors = 0
        # Destinations that have refused us, keyed by what the avatar
        # looked like when they did. Keying on the cell alone is wrong
        # and quietly fatal: ls20's goal square rejects the avatar until
        # all three of its attributes match, so it reads exactly like a
        # wall for most of the level, and marked permanently blocked it
        # would be skipped forever and the level never cleared.
        #
        # The status readout was the obvious key and is the wrong one --
        # measured, it takes three values over a whole level, which is
        # the three lives, not the twenty-four attribute combinations.
        # What the avatar *looks like* is the attribute state: ls20
        # changes the player's shape, colour and rotation, and all three
        # are drawn. So hash the avatar's own cells. It costs the ~25
        # reads of its bounding box, needs no segmentation, and is exact
        # where the readout was merely nearby.
        self._blocked: set[tuple[Any, int, int]] = set()
        self._skipped_blocked = 0

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

        # --- the tool-calling model (see LLM_ENABLED) -----------------
        # Retired for the rest of the game once the server proves
        # unreachable. It will not come back, and re-attempting a 27B
        # request per action would spend the wall clock the search needs.
        self._llm_dead = not LLM_ENABLED
        # Consecutive transport failures; reset by any successful turn.
        self._llm_failures = 0
        # Consecutive timeouts, counted apart because they mean a busy
        # server rather than an absent one. See LLM_MAX_TIMEOUTS.
        self._llm_timeouts = 0
        # The model's own workspace, persisted for the whole game. See
        # `_llm_turn` -- rebuilding this per turn made a forward model
        # impossible to keep, and keeping one is the entire strategy.
        self._llm_ns: dict[str, Any] = {}
        self._llm_messages: list[dict[str, Any]] = []
        self._llm_history: list[str] = []
        self._llm_turns = 0
        self._llm_note = ""
        # Every real transition we have paid for, as (before, label, after).
        # `verify()` replays these against a candidate forward model, which
        # is how the model can be wrong for free: the actions were already
        # billed once, and re-reading them costs nothing.
        self._transitions: deque[
            tuple[tuple[tuple[int, ...], ...], str, tuple[tuple[int, ...], ...]]
        ] = deque(maxlen=TRANSITION_LOG)
        # Did the forward model actually get better? This is the question a
        # GPU run exists to answer, and the previous three could not answer
        # theirs because nothing recorded it. `_verify_best` is the headline:
        # if it never climbs, the model is not learning the game and no
        # amount of further planning will help.
        self._verify_calls = 0
        self._verify_best = 0.0
        self._plan_calls = 0
        self._plans_found = 0
        self._tour_calls = 0
        # Routes that executed exactly as `predict` said and still did not
        # advance the level. That is the signature of a wrong `is_goal`
        # rather than a wrong `predict`, and v17 spent half a run reading
        # it the other way round -- so it is counted, reported in the
        # result, and nudged on.
        self._goal_misses = 0
        # Searches that visited every state their simulator admits and
        # still found nothing. Distinct from a goal miss: that is one
        # hypothesis failing, this is all of them, and only this one
        # points back at `predict`.
        self._exhaustions = 0
        # What the model's CURRENT `predict` scores against everything
        # really played, measured by the turn rather than reported by the
        # model. `_verify_best` is the model's own high-water mark and is
        # not a check on anything; this is.
        self._verify_live = 0.0
        self._verify_live_best = 0.0
        self._verify_peak = 0.0
        # Actions the model took inside the current turn, and the frame it
        # was last handed, which `action()` diffs against to report
        # `changed`.
        self._turn_actions = 0
        self._turn_frame: Optional[FrameData] = None
        self._turn_level = 0
        # Exactly which actions this turn played, and the previous turn's,
        # so a turn that replays its predecessor move for move can be seen.
        # v25's `lp85` spent nineteen consecutive turns emitting the same
        # sixty-action batch, one tool call each, level frozen -- 1140
        # actions after clearing level 1 at action 13. On a metric that
        # scores (baseline/actions)^2 that loop is not merely idle, it is
        # the single most expensive thing the agent did all run, and
        # nothing in the turn report distinguished it from progress.
        self._turn_labels: list[str] = []
        self._last_turn_labels: list[str] = []
        self._repeat_turns = 0
        # The segmentation as of the previous action, handed over as
        # `previous_frame` so a diff is one subtraction away.
        self._last_view: Optional[dict[str, Any]] = None
        # Set when the turn already acted, so the single action the
        # framework expects back from `choose_action` is not sent. See
        # `do_action_request`.
        self._noop = False

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
        self._last_flat = flat
        if self._level_start_node is None:
            self._level_start_node = node
            self._level_start_board = self._board_key(flat)
        self._check_status(node)

        # Watch what the last action did to the board. This has to run
        # before `_observe`, which consumes `_pending`, and it stops as
        # soon as the answer is known -- or as soon as it is clear that
        # paying more for it will not produce one.
        if (
            self._motion is not None
            and self._steps <= MOTION_BUDGET
            and not self._motion.settled
        ):
            view = segment_frame(grid, self._volatile)
            if self._pending is not None and self._motion_view is not None:
                self._motion.observe(
                    self._pending[1] >> (2 * _COORD_BITS),
                    self._motion_view,
                    view,
                )
            self._motion_view = view
            if self._motion.settled and not self._motion_said:
                self._motion_said = True
                logger.info(
                    "%s: motion solved at action %d: %s",
                    self.game_id, self._steps, self._motion.report(),
                )
                self._anchor(view)
                if self._at is not None:
                    self._look = self._appearance(flat, self._at, width, height)

        # Judge the last action first, then go looking. Doing it the
        # other way round spends an anchor on a frame `_track` is about
        # to discard, because the action it is judging is the very one
        # that lost us -- a reset, or a click, which this model has
        # nothing to say about.
        self._track(flat, width, height)
        if (
            self._motion is not None
            and self._at is None
            and self._anchors < MOTION_ANCHORS
            and self._motion.settled
        ):
            self._anchors += 1
            self._anchor(segment_frame(grid, self._volatile))
            if self._at is not None:
                self._look = self._appearance(flat, self._at, width, height)

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

        # 3b. The model's turn, when there is a model. It acts through
        #     `action()` from inside its own code, so by the time this
        #     returns true everything it wanted is already sent and the
        #     one action the framework expects back must be suppressed.
        #
        #     `_pending` is cleared first because the model's steps are
        #     not recorded as graph edges. Leaving it set would attribute
        #     the model's outcome to the search's last action and write a
        #     transition that never happened.
        if not self._llm_dead:
            self._pending = None
            if self._llm_turn(grid, latest_frame):
                self._noop = True
                return GameAction.from_id(_RESET_ID)

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

        The no-op path exists because the framework's loop is one action
        per `choose_action`, and the model's turn is not: it acts from
        inside its own code, through `action()`, as many times as its
        plan needs. Those steps have already been sent by the time
        `choose_action` returns, so the action it hands back is a
        formality the loop requires and must NOT reach the server -- it
        would be an extra billed step, chosen by nobody, on a board the
        model has already moved past.
        """
        if self._noop:
            self._noop = False
            return self._latest_frame()
        raw = self.arc_env.step(
            action, data=dict(self._pending_data), reasoning=self._pending_reasoning
        )
        return self._convert_raw_frame_data(raw)

    def _latest_frame(self) -> FrameData:
        """The current frame without spending an action to ask for it."""
        return self._convert_raw_frame_data(self.arc_env.observation_space)

    # -- the tool-calling model ---------------------------------------
    def _step_env(
        self, action_id: int, x: Optional[int], y: Optional[int]
    ) -> FrameData:
        """Send one action. The single place a billed step actually happens."""
        data: dict[str, Any] = {"game_id": ""}
        if x is not None:
            data["x"] = x
            data["y"] = y
        raw = self.arc_env.step(
            GameAction.from_id(action_id), data=data, reasoning={"why": "llm"}
        )
        return self._convert_raw_frame_data(raw)

    def _turn_over(self) -> bool:
        """Should the model stop acting and be shown the board again?

        Checked after every tool call. The level boundary is the one that
        matters: past it the model is planning against a level that no
        longer exists, and every action it takes there is billed against
        the new one.
        """
        if time.monotonic() >= _DEADLINE:
            return True
        if self._turn_actions >= LLM_TURN_ACTIONS:
            return True
        frame = self._turn_frame
        if frame is None:
            return False
        if frame.state in (GameState.GAME_OVER, GameState.WIN):
            return True
        return frame.levels_completed != self._turn_level

    def _llm_act(self, steps: Any) -> dict[str, Any]:
        """`action(...)` as the model sees it: play moves, report back.

        A `ValueError` from `parse_action_spec` is left to propagate. It
        surfaces inside the model's own code, where it can catch it, and
        otherwise comes back as the tool result -- which is the point.
        Nothing is sent until every step in the batch has parsed, so a
        typo in the fourth move does not leave the first three played.
        """
        available = list(
            (self._turn_frame.available_actions if self._turn_frame else None) or []
        )
        parsed = [
            parse_action_spec(item, available) for item in split_action_specs(steps)
        ]

        frame = self._turn_frame
        ran = 0
        changed = False
        stopped: Optional[str] = None
        # Where the simulator and the environment disagreed while executing.
        # A plan is only as good as the model it was found in, so the first
        # divergence is the single most useful thing to report back: it is
        # the counterexample that corrects `predict`.
        drift: list[dict[str, Any]] = []
        predict = self._llm_ns.get("predict")

        for action_id, x, y in parsed:
            if self._turn_actions >= LLM_TURN_ACTIONS:
                stopped = "turn action limit reached; you will be asked again"
                break
            if time.monotonic() >= _DEADLINE:
                stopped = "the run is out of time"
                break

            before = grid_of(frame) if frame is not None else None
            label = name_action(action_id, x, y)

            expected = None
            if callable(predict) and before is not None:
                try:
                    expected = normalise_grid(predict(before, label))
                except Exception:  # noqa: BLE001 - reported, never fatal
                    expected = None

            frame = self._step_env(action_id, x, y)
            self.append_frame(frame)
            self._turn_frame = frame
            ran += 1
            self._turn_actions += 1
            self._turn_labels.append(label)
            self._steps += 1
            # `Agent.main()` bills one per loop iteration and this turn
            # takes many, so without this MAX_ACTIONS would stop meaning
            # anything.
            self.action_counter += 1

            after = grid_of(frame)
            if before is not None:
                self._transitions.append((before, label, after))
            if expected is not None and expected != after and len(drift) < 3:
                drift.append(
                    {
                        "after_step": ran,
                        "action": label,
                        "sample": grid_diff(expected, after),
                        "note": "predicted vs actual, as (x, y, predicted, actual)",
                    }
                )

            changed = after != before
            self._llm_history.append(
                f"{label} -> {'board changed' if changed else 'no change'}"
            )

            if frame.state in (GameState.GAME_OVER, GameState.WIN):
                stopped = f"the game ended: {frame.state.name}"
                break
            if frame.levels_completed != self._turn_level:
                stopped = (
                    f"level completed -- now on level {frame.levels_completed}"
                )
                break

        del self._llm_history[:-LLM_HISTORY]

        new_grid = grid_of(frame) if frame else ()
        view = segment_frame(new_grid, self._volatile) if frame else {}

        # Write the new board back into the model's workspace.
        #
        # This was missing, and it broke the model's main way of learning
        # anything. `grid` was bound once at the start of a turn, so after
        # `action(...)` it still held the PREVIOUS board -- while the model's
        # own code, logged from the v15 run, did exactly this:
        #
        #     result2 = action('ACTION2')
        #     for y in range(64):
        #         if grid[y][x] != state_after_action1[y][x]: ...
        #
        # and, in a later call, said in a comment: "The grid variable should
        # have been updated." It should have. Diffing a board against itself
        # reports no change no matter what the action did, so thirty turns of
        # probing taught it nothing about any action.
        self._llm_ns["grid"] = new_grid
        self._llm_ns["current_frame"] = view
        self._llm_ns["previous_frame"] = self._last_view
        self._last_view = view
        outcome = {
            "steps": ran,
            "changed": changed,
            "state": frame.state.name if frame else "UNKNOWN",
            "levels_completed": frame.levels_completed if frame else 0,
            "frame": view,
            "stopped": stopped,
        }
        if drift:
            outcome["mispredicted"] = drift
        elif callable(predict) and ran:
            outcome["mispredicted"] = []

        # Name which of the two things was wrong.
        #
        # A route that played exactly as `predict` said and left the level
        # where it was has an unambiguous diagnosis: the simulator is
        # right and the board it aimed at is not the goal. In v17 the
        # model hit this three times, concluded its simulator must be
        # broken, and went back to probing at 1.00 verify accuracy. It is
        # the one inference the result was not carrying, so it carries it
        # now. Gated on a plan having been made, so ordinary probing --
        # which of course does not advance the level either -- stays
        # quiet.
        if (
            (self._plan_calls or self._tour_calls)
            and ran >= 2
            and not drift
            and callable(predict)
            and frame is not None
            and frame.levels_completed == self._turn_level
        ):
            self._goal_misses += 1
            outcome["goal_miss"] = (
                "This route played exactly as `predict` said it would, and "
                "the level did not advance. Your simulator is right; the "
                "board you aimed at is not the goal. Do NOT go back to "
                "probing and do NOT rewrite `predict` -- rewrite "
                "`is_goal`. Write several candidate goals and pass them "
                "all to tour(predict, {...}) at once."
            )
        # A model writing its own loop needs the last outcome to still be
        # there on the next `python` call, not only as the return value of
        # the call that produced it.
        self._llm_ns["last_action_result"] = outcome
        return outcome

    def _llm_verify(
        self, predict: Any, limit: Optional[int] = None
    ) -> dict[str, Any]:
        """`verify(predict)` as the model sees it. Bills nothing.

        Scores a candidate forward model against every real transition
        this game has produced so far. The actions were paid for when they
        happened, so a wrong hypothesis costs nothing to discover here --
        which is the only reason iterating on `predict` is affordable at
        all.
        """
        if not callable(predict):
            raise TypeError(
                "verify(predict) expects a function predict(grid, action)"
            )
        if not self._transitions:
            return {
                "checked": 0, "correct": 0, "accuracy": 0.0, "raised": 0,
                "wrong": [],
                "note": "nothing to check against yet -- play a few actions "
                        "first, then verify against what they did",
            }
        report = verify_predictions(predict, list(self._transitions), limit)
        self._verify_calls += 1
        self._verify_best = max(self._verify_best, report["accuracy"])
        logger.info(
            "%s: verify #%d: %.2f accuracy over %d transitions (best %.2f)",
            self.game_id, self._verify_calls, report["accuracy"],
            report["checked"], self._verify_best,
        )
        return report

    def _note_exhaustion(self, out: dict[str, Any], missed: int) -> None:
        """Attach the too-small-simulator diagnosis to a search result."""
        note = exhaustion_note(out.get("reason", ""), out.get("nodes", 0), missed)
        if note:
            self._exhaustions += 1
            out["exhausted"] = note
            logger.info(
                "%s: search exhausted its own state space (%d states, "
                "%d goals unreached)",
                self.game_id, out.get("nodes", 0), missed,
            )

    def _autoverify(self) -> Optional[dict[str, Any]]:
        """Re-score the model's live `predict` against what was really played.

        Runs every turn, unasked, because v18 showed the model will not do
        it: four verifies in turn 2, then 140 further actions on the
        strength of a 1.00 measured over thirteen transitions. Costs no
        actions -- the transitions were billed when they happened.
        """
        predict = self._llm_ns.get("predict")
        if not callable(predict) or not self._transitions:
            return None
        try:
            report = verify_predictions(
                predict, list(self._transitions), AUTOVERIFY_LIMIT
            )
        except Exception:  # noqa: BLE001 - never let a checkup end the game
            return None
        # Compare against the best anyone has measured -- the model's own
        # high-water mark OR this check's -- and take that reading BEFORE
        # folding today's in, or a simulator can only ever drift against
        # itself. Using `_verify_best` alone was wrong: it is whatever the
        # model last reported, and a model that never calls verify() has
        # it sitting at zero while its simulator rots unnoticed.
        self._verify_peak = max(self._verify_best, self._verify_live_best)
        self._verify_live = report["accuracy"]
        self._verify_live_best = max(self._verify_live_best, self._verify_live)
        return report

    def _plan_inputs(
        self, actions: Optional[list[Any]], start: Any
    ) -> tuple[Optional[tuple[tuple[int, ...], ...]], list[Any]]:
        """The board and the move set both `plan` and `tour` search over."""
        board = normalise_grid(
            start if start is not None else self._llm_ns.get("grid")
        )
        if actions is None:
            # Clicks are excluded by default: ACTION6 needs coordinates,
            # and branching on all 4,096 of them makes the first level of
            # the search larger than the whole node budget. A click game
            # has to pass its own shortlist of targets.
            available = (
                self._turn_frame.available_actions if self._turn_frame else None
            ) or []
            actions = [f"ACTION{i}" for i in available if i != 6]
        return board, list(actions)

    def _llm_plan(
        self,
        predict: Any,
        is_goal: Any,
        actions: Optional[list[Any]] = None,
        start: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """`plan(predict, is_goal)` as the model sees it. Bills nothing.

        Searches the model's own simulator rather than the environment, so
        the cost of looking for a route is paid in GPU seconds instead of
        in score. What comes back is the shortest sequence found, ready to
        hand straight to `action(...)`.

        `is_goal` may be several candidates at once -- see `plan_actions`.
        """
        if not callable(predict):
            raise TypeError(
                "plan(predict, is_goal) expects predict(grid, action) -> grid"
            )
        if not (callable(is_goal) or isinstance(is_goal, (list, tuple, dict))):
            raise TypeError(
                "plan(predict, is_goal) expects is_goal(grid) -> bool, or a "
                "list or dict of such functions to test several at once"
            )
        board, actions = self._plan_inputs(actions, start)
        if board is None:
            return {
                "found": False, "actions": [], "nodes": 0, "routes": {},
                "reason": "no board to plan from",
            }
        out = plan_actions(predict, is_goal, board, actions, **kwargs)
        self._plan_calls += 1
        self._plans_found += 1 if out["found"] else 0
        self._note_exhaustion(
            out, len(out.get("unreachable") or [])
            or (0 if out["found"] else len(name_goals(is_goal)))
        )
        logger.info(
            "%s: plan #%d: %s (%d states, %s)",
            self.game_id, self._plan_calls,
            f"found a {len(out['actions'])}-action route to "
            f"{len(out.get('routes') or {})} of "
            f"{len(name_goals(is_goal))} goals" if out["found"] else "no route",
            out["nodes"], out["reason"],
        )
        return out

    def _llm_tour(
        self,
        predict: Any,
        goals: Any,
        actions: Optional[list[Any]] = None,
        start: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """`tour(predict, goals)` as the model sees it. Bills nothing.

        Chains candidate goals into one sequence. See `plan_tour` for why
        that is cheaper than testing them one at a time: execution stops
        at whichever candidate was really the goal, so the ones after it
        are never played.
        """
        if not callable(predict):
            raise TypeError(
                "tour(predict, goals) expects predict(grid, action) -> grid"
            )
        board, actions = self._plan_inputs(actions, start)
        if board is None:
            return {
                "actions": [], "checkpoints": [], "reached": [],
                "unreachable": [], "nodes": 0,
                "reason": "no board to plan from",
            }
        out = plan_tour(predict, goals, board, actions, **kwargs)
        self._tour_calls += 1
        self._note_exhaustion(out, len(out["unreachable"]))
        logger.info(
            "%s: tour #%d: %d actions through %d/%d goals (%d states, %s)",
            self.game_id, self._tour_calls, len(out["actions"]),
            len(out["reached"]),
            len(out["reached"]) + len(out["unreachable"]),
            out["nodes"], out["reason"],
        )
        return out

    def _llm_turn(
        self, grid: tuple[tuple[int, ...], ...], latest_frame: FrameData
    ) -> bool:
        """Give the model one observe-plan-act turn. True if it acted.

        False means the caller must fall through to the search, and it is
        an ordinary outcome, not an error: a turn that only reasoned, or
        whose code raised, still has to yield an action or the game stalls
        on a board nobody moved.
        """
        available = list(latest_frame.available_actions or [])
        view = segment_frame(grid, self._volatile)
        started = time.monotonic()

        self._turn_actions = 0
        self._turn_labels = []
        self._turn_frame = latest_frame
        self._turn_level = latest_frame.levels_completed

        if not self._llm_messages:
            self._llm_messages = [
                {"role": "system", "content": build_system_prompt(available)}
            ]
        else:
            self._llm_messages = evict_turns(self._llm_messages, LLM_EVICT_TURNS)

        # Exactly one image in flight: the board as it is now.
        strip_images(self._llm_messages)
        turn_text = build_turn_prompt(
            view,
            self._llm_history,
            latest_frame.levels_completed,
            self._steps,
            self._llm_note,
            self._llm_ns.get("notes"),
        )
        self._llm_messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": turn_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": render_data_uri(grid)},
                    },
                ],
            }
        )
        self._llm_note = ""

        # ONE namespace for the whole game, refreshed rather than rebuilt.
        #
        # It used to be constructed fresh every turn, which quietly made the
        # intended strategy impossible: a forward model the model writes on
        # turn 3 was deleted before turn 4 could search it. Told to "build a
        # simulator and plan in it", it would have had to rebuild the
        # simulator every turn out of a transcript that is itself evicted
        # every four turns. Persisting this is what makes planning cheaper
        # than acting, which is the only way a quadratic metric is won.
        self._llm_ns.update(
            {
                "current_frame": view,
                "previous_frame": self._last_view,
                "grid": grid,
                "history": list(self._llm_history),
                "valid_actions": available,
                "action": self._llm_act,
                "verify": self._llm_verify,
                "plan": self._llm_plan,
                "tour": self._llm_tour,
            }
        )
        self._llm_ns.setdefault("notes", [])
        namespace = self._llm_ns
        self._last_view = view

        try:
            outcome = run_tool_conversation(
                self._llm_messages,
                namespace,
                stop_check=self._turn_over,
                label=f"{self.game_id} t{self._llm_turns + 1}",
            )
        except Exception as exc:  # noqa: BLE001 - any transport failure counts
            # This used to retire the model on the FIRST failure, reasoning
            # that whatever broke would still be broken next time. That is
            # true of a refused connection and false of a timeout, and the
            # first Blackwell run hit exactly the false case: the model
            # played 136 actions, one generation ran past the deadline, and
            # a working server was written off for the rest of the game.
            #
            # So failures are counted instead. A genuinely dead server
            # trips the limit immediately and costs a bounded number of
            # retries once, not per action; a single slow generation costs
            # one search step and is forgiven.
            #
            # Counted in two buckets, because v25 showed the same argument
            # applies once more one level down. Under 25-way concurrency a
            # timeout is the ordinary signature of a deep queue, not of a
            # broken server, and three of them in a row retired `cn04`'s
            # model for the rest of its run. A timeout therefore gets its
            # own, much longer rope; anything else -- a refused connection,
            # an HTTP 400 -- still gets three.
            timed_out = _is_timeout(exc)
            if timed_out:
                self._llm_timeouts += 1
                seen, limit = self._llm_timeouts, LLM_MAX_TIMEOUTS
            else:
                self._llm_failures += 1
                seen, limit = self._llm_failures, LLM_MAX_FAILURES
            retiring = seen >= limit
            logger.warning(
                "%s: model call failed at action %d (%s: %s) [%s %d/%d]%s",
                self.game_id, self._steps, type(exc).__name__, exc,
                "timeout" if timed_out else "transport", seen, limit,
                "; retiring it, the search plays the rest of this game"
                if retiring else "; falling back for this step only",
            )
            self._llm_dead = retiring
            # The transcript is left mid-turn -- possibly with an assistant
            # tool_call whose result never arrived, which the server
            # rejects. Drop back to the framing and start the next turn
            # clean rather than reasoning about how far it got.
            self._llm_messages = self._llm_messages[:1]
            return False

        self._llm_failures = 0
        self._llm_timeouts = 0
        self._llm_turns += 1

        # Did this turn just replay the last one?
        #
        # Exact match, deliberately. A looser rule -- "hit the action cap
        # again with one tool call and no level change" -- would also have
        # caught lp85, and would also catch a model correctly executing a
        # long plan it found. Replaying the identical sequence and
        # expecting a different board is the one case that cannot be
        # progress, so that is the case this fires on.
        # `_turn_level` is the level as of the START of this turn and
        # `_turn_frame` is where the turn left off, so this compares end
        # against start. Comparing against `latest_frame` would compare the
        # start against itself and always look stuck.
        ended_on = (
            self._turn_frame.levels_completed
            if self._turn_frame is not None else self._turn_level
        )
        repeated = bool(
            self._turn_labels
            and self._turn_labels == self._last_turn_labels
            and ended_on == self._turn_level
        )
        self._repeat_turns = self._repeat_turns + 1 if repeated else 0
        self._last_turn_labels = list(self._turn_labels)

        # One line per turn. The first Blackwell run reported "136 actions"
        # and nothing else, which says the loop works but not whether the
        # model is playing or thrashing -- and on a quadratic metric the
        # actions-per-turn number is the one that decides the score.
        live = self._autoverify()

        logger.info(
            "%s: model turn %d: %d actions in %.0fs (%s, %d tool calls, "
            "level %d, %d total) | predict %.2f live / %.2f claimed over "
            "%d transitions, %d verifies, %d/%d plans found, %d tours, "
            "%d goal misses, %d exhaustions",
            self.game_id, self._llm_turns, self._turn_actions,
            time.monotonic() - started, outcome["stop"],
            outcome["tool_calls"], self._turn_level, self._steps,
            self._verify_live, self._verify_best,
            live["checked"] if live else 0, self._verify_calls,
            self._plans_found, self._plan_calls,
            self._tour_calls, self._goal_misses, self._exhaustions,
        )

        # The repeat warning is a PREFIX, not a rung on the ladder below.
        #
        # It started as the top branch of that chain, which turned out to
        # be wrong in a way two existing tests caught immediately: a model
        # that replays a batch because its `predict` has gone stale needs
        # to hear both things, and suppressing the drift note in favour of
        # "you are repeating" throws away the half that says why. lp85 is
        # still covered -- it had no drift signal at all, every verify
        # counter sat at zero, so there the prefix is the whole note.
        repeat_note = ""
        if self._repeat_turns:
            logger.warning(
                "%s: turn %d replayed the previous turn's %d actions "
                "exactly, still on level %d [%d in a row]%s",
                self.game_id, self._llm_turns, len(self._turn_labels),
                self._turn_level, self._repeat_turns,
                "; resetting the transcript"
                if self._repeat_turns >= LLM_REPEAT_RESET else "",
            )
            repeat_note = (
                f"STOP. Your last turn played exactly the same "
                f"{len(self._turn_labels)} actions as the turn before it, "
                f"in the same order, and the level did not change. Playing "
                f"them a third time cannot produce a different result. You "
                f"are being scored on (baseline/actions) SQUARED, so those "
                f"{len(self._turn_labels)} wasted actions cost you more "
                f"than the whole level is worth. Do not call action() at "
                f"the start of this turn. Look at the board image first, "
                f"say what is actually different from what you expected, "
                f"and test ONE new hypothesis with a short batch."
            )
            if self._repeat_turns >= LLM_REPEAT_RESET:
                # The note alone did not land. The transcript is now
                # several identical turns of the model agreeing with
                # itself, which is the thing generating the repeat, so it
                # goes -- the note and a fresh board survive in the next
                # turn's prompt, which is all that is worth keeping.
                self._llm_messages = self._llm_messages[:1]
                self._last_turn_labels = []
                self._repeat_turns = 0

        if self._exhaustions:
            # The v18 failure, and the deepest of the three: a model here
            # has a simulator, has planned with it, has toured candidate
            # goals, and has run out of world. Nothing below is worth
            # saying while the state space itself is wrong.
            self._llm_note = (
                "Your search has now visited every state your `predict` "
                "says exists and finished nothing. Stop writing goals -- "
                "there is nowhere left for them to be. Your simulator is "
                "missing state: something on the board changes that your "
                "`predict` copies over unchanged. Find it, model it, and "
                "verify against ALL the moves you have played, not the "
                "last few."
            )
        elif live and self._verify_peak - self._verify_live > AUTOVERIFY_DRIFT:
            # The mechanism behind v18: a 1.00 measured once, over thirteen
            # transitions, trusted for a hundred and forty actions after.
            self._llm_note = (
                f"Checked for you, free: your current `predict` scores "
                f"{self._verify_live:.0%} against the last "
                f"{live['checked']} moves actually played, not the "
                f"{self._verify_peak:.0%} it reached before. You have been "
                "planning with a simulator that no longer matches the "
                "game. Re-read `wrong` from verify(predict) and fix it "
                "before spending another action."
            )
        elif self._goal_misses and not self._tour_calls:
            # The v17 failure, addressed where it happened. Outranks the
            # plan nudge because it is strictly later in the same
            # pipeline: a model here already has a simulator and already
            # planned with it, and the only thing left in its way is that
            # it is guessing the goal one coordinate at a time and paying
            # for each guess.
            self._llm_note = (
                f"{self._goal_misses} of your routes played exactly as "
                "`predict` said and still did not finish the level. That "
                "means `predict` is right and `is_goal` is wrong -- you "
                "are guessing where to go. Stop testing one guess per "
                "execution. Write every plausible goal you can think of "
                "and pass them together to tour(predict, {...}): it "
                "returns ONE sequence through all of them, it costs no "
                "actions to build, and playing it stops the moment one "
                "of them turns out to be the real goal."
            )
        elif self._verify_best >= PLAN_NUDGE_ACCURACY and not self._plan_calls:
            # The v16 failure, addressed where it happened. Takes priority
            # over the no-action nudge below: a model sitting on a usable
            # simulator and probing by hand is burning the score far faster
            # than one that thought for a turn without moving.
            self._llm_note = (
                f"You have a forward model that is {self._verify_best:.0%} "
                "accurate and you have not called plan() once. Stop probing "
                "and plan with it now: plan(predict, is_goal) costs no "
                "actions, and executing what it finds is what will tell you "
                "the rest. Do not wait for predict to be perfect."
            )
        elif self._turn_actions == 0:
            # Called out separately: a model that reasons and never acts
            # looks exactly like a working run from the outside, while the
            # search quietly plays the whole game.
            self._llm_note = (
                "Your last turn ended without calling action(). A move has "
                "to be played, so one was chosen for you. Act this turn."
            )

        if repeat_note:
            self._llm_note = f"{repeat_note} {self._llm_note}".strip()
        return self._turn_actions > 0

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
            if self._motion is not None:
                logger.info(
                    "%s: motion model carried into level %d: %s "
                    "| %d walls known, %d probes skipped",
                    self.game_id, levels, self._motion.report(),
                    len(self._blocked), self._skipped_blocked,
                )
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
            if MASK_RELEARN_PER_LEVEL:
                self._rearm_mask_learning()

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
            self._code_live.add(code)

        # Split the level's spend by action class. ~112 of the ~117 codes a
        # node offers are ACTION6 clicks, so if clicks dominate the bill and
        # almost none of them move the state, the branching factor -- not the
        # search policy -- is what puts a near-baseline clear out of reach.
        kind = "click" if action_id == _ACTION6_ID else "press"
        self._class_counts[kind] += 1
        if node != prev_node:
            self._class_counts[kind + "-live"] += 1

        # Separate counters for LIVENESS_WIDEN. `_class_counts` cannot serve:
        # `_report_budget` clears it, so switching diagnostics on would
        # silently change what the search does.
        if kind == "click":
            self._click_tried += 1
            if node != prev_node:
                self._click_live += 1

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
        self._click_tried = 0
        self._click_live = 0
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

    def _mask_is_credible(self, mask: frozenset[int], total: int) -> bool:
        """Would this mask fuse every frame we have seen into one state?

        Only asked of masks that already cleared MASK_CAP_FRACTION, and only
        a veto -- never a reason to adopt something the size rule rejected.

        Collapsing everything is not damning by itself. tn36 masks 64 cells
        and collapses all 26 warmup frames to a single key, and it is the
        highest-scoring game on the dev set: the board really did not move
        except in its HUD. The same collapse from a 1180-cell mask, as tr87
        gets at rung 0.3, is that mask deleting the board. Size alone cannot
        tell those apart and neither can collapse alone, so require both.
        """
        if not MASK_COLLAPSE_GUARD or not self._mask_frames or total <= 0:
            return True
        if len(mask) <= MASK_SMALL_FRACTION * total:
            return True
        # A frame sampled before a resize cannot be indexed by a mask built
        # for after it. `_learn_mask` drops those as it goes, so this is the
        # belt to that braces -- `total` is width*height from the caller and
        # nothing here guarantees it matches what was sampled. Abstaining
        # when nothing comparable is left keeps this a veto.
        frames = [frame for frame in self._mask_frames if len(frame) == total]
        if not frames:
            return True
        kept = [i for i in range(total) if i not in mask]
        distinct = {bytes(map(frame.__getitem__, kept)) for frame in frames}
        if len(distinct) >= MASK_MIN_DISTINCT:
            return True
        logger.info(
            "%s: rejecting a %d-cell mask (%.0f%% of frame): it fuses all "
            "%d sampled frames into %d state(s)",
            self.game_id, len(mask), 100.0 * len(mask) / total,
            len(self._mask_frames), len(distinct),
        )
        return False

    def _board_key(self, flat: tuple[int, ...]) -> int:
        """The state key with the status readout taken back out.

        A respawn puts the avatar back at the start and restores every
        tile, so the board is pixel-identical to the level opening --
        while the status is not, because a life has gone. Comparing on
        the board is what separates "the game reset me" from "I changed
        into something else", and only the second makes stale verdicts
        worth revisiting.
        """
        if not self._status_idx:
            return self._node_key(flat)
        if self._board_idx is None:
            status = set(self._status_idx)
            kept = self._kept
            source = range(len(flat)) if kept is None else kept
            self._board_idx = tuple(i for i in source if i not in status)
        return hash(tuple(map(flat.__getitem__, self._board_idx)))

    def _check_status(self, node: int) -> bool:
        """Re-open the map when the status readout changes.

        A node is written off as exhausted once every candidate from it
        has been tried. That verdict is only ever true *for the agent
        that tried them*. On ls20 the agent walks over a tile, its
        rotation changes, and a goal square that rejected it five
        actions ago now accepts it -- but the search has already spent
        that node's cursor and will not go back. The board looks
        identical either way; the only evidence anything happened is
        this readout.

        So when it moves, clear the cursors. Nodes keep their edges --
        experience stays true -- but every one of them becomes worth
        revisiting at the breadth it has already reached. Costs no
        actions to do and is bounded by STATUS_REOPEN_LIMIT so a game
        with a flickering readout cannot restart the search forever.
        """
        if not self._status_idx:
            return False
        flat = self._last_flat
        if flat is None:
            return False
        try:
            reading = tuple(map(flat.__getitem__, self._status_idx))
        except IndexError:
            # The frame resized under us; the band is indexed for the old
            # one and will be rebuilt with the next mask.
            self._status_idx = ()
            return False

        previous, self._status = self._status, reading
        if previous is None or previous == reading:
            return False

        # A respawn changes the readout too, and means the opposite thing.
        # ls20 gives 42 steps and three lives: run the counter out and it
        # puts the avatar back at the start with its attributes reset, so
        # the readout moves without anything having been discovered.
        # Re-opening there restarts the search every 43 actions forever --
        # measured before this check: 39 re-opens, every one a death.
        #
        # Telling them apart by comparing against the level opening does
        # not work, because the mask is adopted ~26 actions in and drops
        # the graph with it, so "the opening" is whatever the avatar was
        # standing on at the time. What does work is that a respawn is a
        # *cycle*: it returns the board to a state a previous status
        # change already saw. Walking onto a rotation tile does not -- it
        # happens somewhere new. So learn the boards status changes land
        # on, and trust only the first sighting of each.
        board = self._board_key(flat)
        seen = self._status_boards.get(board, 0) + 1
        self._status_boards[board] = seen
        if seen > 1:
            gap = (
                None if self._last_respawn is None
                else self._steps - self._last_respawn
            )
            if gap is not None and gap < RESPAWN_MIN_PERIOD:
                # Too soon to be a life. Still not a discovery -- the
                # board has been here before -- so the map stays as it
                # is, but nothing about it is worth timing.
                return False
            if gap is not None:
                self._respawn_periods.append(gap)
                ordered = sorted(self._respawn_periods)
                self._respawn_period = ordered[len(ordered) // 2]
            self._last_respawn = self._steps
            self._respawns += 1
            # The avatar has just been picked up and put back at the
            # start. Nothing we tracked about where it was survives that,
            # and a stale position turns the blocked map into a list of
            # walls that are not there.
            self._at = None
            self._at_confirmed = 0
            if self._respawns == 1 or self._respawns % 10 == 0:
                logger.info(
                    "%s: respawned at action %d (period %s, %d so far); "
                    "plans longer than that cannot finish",
                    self.game_id, self._steps,
                    self._respawn_period or "?", self._respawns,
                )
            return False

        if not STATUS_REOPEN or self._status_reopens >= STATUS_REOPEN_LIMIT:
            return False

        self._status_reopens += 1
        self._cursor.clear()
        self._plan.clear()
        logger.info(
            "%s: status readout changed at action %d; re-opening %d nodes "
            "(%d/%d)",
            self.game_id, self._steps, len(self._total),
            self._status_reopens, STATUS_REOPEN_LIMIT,
        )
        return True

    def _learn_mask(self, flat: tuple[int, ...], width: int, height: int) -> bool:
        """Accumulate change rates; freeze the HUD mask when ready.

        Returns True exactly once, on the transition where the mask is
        adopted and every existing node id becomes invalid.
        """
        if self._mask_frozen:
            return False

        # Retain the warmup frames so a candidate mask can be judged by what
        # it does to them, not only by its size. Stored as bytes -- colours
        # are small integers -- which is ~4KB a frame instead of ~115KB as a
        # tuple of Python ints, and this list is dropped the moment the mask
        # is decided.
        #
        # Once a deferral is in progress (see MASK_STRICT_SCALE), the cap
        # extends to the deferred sample count instead of stopping at
        # VOLATILE_WARMUP: the second look's widget-span trimming needs
        # frames from *after* the transient it exists to see past, not
        # just the original VOLATILE_WARMUP frames that produced the
        # oversized candidate in the first place.
        frame_cap = self._mask_defer_until or VOLATILE_WARMUP
        if MASK_COLLAPSE_GUARD and len(self._mask_frames) < frame_cap:
            try:
                self._mask_frames.append(bytes(flat))
            except (ValueError, TypeError):
                # A colour outside 0-255 would be a different game than the
                # one this was written for; drop the guard rather than guess.
                self._mask_frames = []

        previous, self._prev_flat = self._prev_flat, flat
        if previous is None or len(previous) != len(flat) or width <= 0:
            # First frame, or the grid was resized: start counting fresh.
            # The sampled frames restart too. Keeping ones from before a
            # resize is not merely useless -- a mask indexed for the new
            # size walks off the end of an older, smaller frame, and that
            # IndexError lands on the action path.
            self._mask_frames = [
                frame for frame in self._mask_frames if len(frame) == len(flat)
            ]
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
                frames=self._mask_frames,
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
            self._status_idx = status_band(
                self._row_changes, self._col_changes, candidate,
                self._samples, width, height, threshold=1.0,
            )
            self._cell_changes = []
            self._row_changes = []
            self._col_changes = []
            self._mask_frames = []
            self._prev_flat = None
            self._volatile = candidate
            self._kept = tuple(
                i for i in range(len(flat)) if i not in candidate
            )
            return True

        # MASK_STRICT_SCALE: give a candidate above MASK_SMALL_FRACTION
        # more looks, geometrically, instead of adopting it off this
        # single VOLATILE_WARMUP-sample evaluation. See its own comment
        # for the cn04 case this targets, why a single fixed extra wait
        # was tried and rejected first, and the risk this trades for.
        # The gate below has to apply to the *final* candidate -- strict
        # rule or ladder fallback, whichever wins -- not just the strict
        # one: cn04's first checkpoint clears the strict rule (nothing
        # found) but the ladder it then falls through to still finds a
        # bad large mask on the same thin evidence.
        if MASK_STRICT_SCALE and self._mask_defer_until:
            if self._samples < self._mask_defer_until:
                return False

        volatile = hud_mask(
            counts, self._row_changes, self._col_changes,
            self._samples, width, height, frames=self._mask_frames,
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
                    frames=self._mask_frames,
                )
                # An over-cap mask comes back empty, so a non-empty
                # candidate has already cleared MASK_CAP_FRACTION.
                if not candidate:
                    continue
                if not self._mask_is_credible(candidate, width * height):
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

        if MASK_STRICT_SCALE:
            total = width * height
            small = not volatile or (
                total > 0 and len(volatile) <= MASK_SMALL_FRACTION * total
            )
            if not small:
                if (
                    self._mask_best_size is None
                    or len(volatile) < self._mask_best_size
                ):
                    self._mask_best_size = len(volatile)
                    self._mask_best_mask = volatile
                self._mask_defer_count += 1
                if self._mask_defer_count <= MASK_STRICT_SCALE_DOUBLINGS:
                    self._mask_defer_until = self._samples * 2
                    logger.info(
                        "%s: still finds a %d-cell candidate (%.0f%% of "
                        "frame) at %d samples (checkpoint %d/%d); "
                        "deferring to %d",
                        self.game_id, len(volatile),
                        100.0 * len(volatile) / total if total else 0.0,
                        self._samples, self._mask_defer_count,
                        MASK_STRICT_SCALE_DOUBLINGS, self._mask_defer_until,
                    )
                    return False
                # Out of checkpoints. Commit to the smallest ever seen,
                # even if it is still large -- not necessarily this one.
                volatile = self._mask_best_mask
                logger.info(
                    "%s: exhausted %d checkpoints at %d samples; "
                    "committing to the smallest candidate seen (%d cells)",
                    self.game_id, MASK_STRICT_SCALE_DOUBLINGS,
                    self._samples, len(volatile),
                )

        self._mask_frozen = True

        if volatile:
            # Deliberately not `volatile`. Under the whole-line rule the
            # mask covers the entire row, so the band would come back
            # empty and the respawn detector with it -- and the detector
            # is the half of this change that pays. The band is the
            # readout beside the widget, which is a fact about the frame
            # and not about which masking rule is shipped, so ask for the
            # span reading regardless. These cells are excluded from the
            # state key under the whole-line rule; that does not stop us
            # reading them off the frame to notice a respawn.
            self._status_idx = status_band(
                self._row_changes, self._col_changes,
                hud_mask(
                    counts, self._row_changes, self._col_changes,
                    self._samples, width, height,
                    frames=self._mask_frames, span=True,
                ),
                self._samples, width, height,
            )

        # Free the counters and the retained frames; both are dead weight for
        # the rest of the run, and the frames are the larger of the two.
        self._cell_changes = []
        self._row_changes = []
        self._col_changes = []
        self._mask_frames = []
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
        # Node ids are about to stop meaning what they meant -- a new level,
        # or a new mask. The opening node and the last status reading are
        # both keyed on the old ones, so re-learn them rather than compare
        # across the boundary and call it a respawn.
        self._level_start_node = None
        self._level_start_board = None
        self._board_idx = None
        self._status_boards = {}
        self._status = None
        self._last_respawn = None
        # Walls are a property of the level, so they go with it. The
        # motion model deliberately does not.
        self._blocked.clear()
        self._at = None
        self._at_confirmed = 0
        self._total.clear()
        self._cursor.clear()
        self._edges.clear()
        self._fatal.clear()
        self._deaths.clear()
        self._cache.clear()
        self._plan.clear()

    def _rearm_mask_learning(self) -> None:
        """Re-open `_learn_mask` for a fresh look, see MASK_RELEARN_PER_LEVEL.

        Deliberately does not touch `_kept`/`_volatile`/`_status_idx`: the
        mask already in effect keeps being used, unchanged, until fresh
        evidence actually replaces it. This only clears the counters that
        decide whether it gets replaced.
        """
        self._prev_flat = None
        self._cell_changes = []
        self._row_changes = []
        self._col_changes = []
        self._samples = 0
        self._mask_frozen = False
        self._early_candidate = frozenset()
        self._early_stable = 0
        self._mask_frames = []
        self._mask_defer_until = 0
        self._mask_defer_count = 0
        self._mask_best_size = None
        self._mask_best_mask = frozenset()

    def _anchor(self, view: dict[str, Any]) -> None:
        found = self._motion.locate(view) if self._motion else None
        if found is None:
            return
        top, left, height, width = found
        self._at = (top, left)
        self._at_size = (height, width)
        self._at_confirmed = 0

    def _appearance(
        self,
        flat: tuple[int, ...],
        at: tuple[int, int],
        width: int,
        height: int,
    ) -> int:
        """What we currently are, as far as the board will say.

        Not where we are: the avatar's own cells are deliberately not in
        here. This is the readout, so that a square refused while we were
        one thing says nothing about the same square while we are
        another.
        """
        if not self._readout_idx:
            return 0
        return hash(tuple(map(flat.__getitem__, self._readout_idx)))

    def _learn_readout(
        self,
        before: tuple[int, ...],
        after: tuple[int, ...],
        boxes: tuple[tuple[int, int], ...],
        width: int,
        height: int,
    ) -> None:
        """Count changes that our own movement does not account for."""
        if self._readout_idx is not None:
            return
        rows, cols = self._at_size
        skip = set()
        for top, left in boxes:
            for y in range(max(0, top), min(height, top + rows)):
                base = y * width
                for x in range(max(0, left), min(width, left + cols)):
                    skip.add(base + x)
        volatile = self._volatile
        counts = self._readout_counts
        for index, (was, now) in enumerate(zip(before, after)):
            if was != now and index not in skip and index not in volatile:
                counts[index] += 1
        self._readout_samples += 1
        if self._readout_samples < READOUT_WARMUP:
            return
        cutoff = READOUT_MAX_RATE * self._readout_samples
        found = tuple(
            sorted(i for i, n in counts.items() if 0 < n <= cutoff)
        )
        # A "readout" the size of the board is not a readout, and hashing
        # it every action would cost more than the probes it saves.
        self._readout_idx = found if len(found) <= READOUT_CAP else ()
        self._readout_counts = Counter()
        logger.info(
            "%s: readout is %d cells after %d samples",
            self.game_id, len(self._readout_idx), self._readout_samples,
        )

    def _region_changed(
        self,
        before: tuple[int, ...],
        after: tuple[int, ...],
        at: tuple[int, int],
        width: int,
        height: int,
    ) -> Optional[bool]:
        """Did anything inside this rectangle change? None means it is
        not on the board at all, which is its own kind of wall."""
        top, left = at
        rows, cols = self._at_size
        if top < 0 or left < 0 or top + rows > height or left + cols > width:
            return None
        volatile = self._volatile
        for y in range(top, top + rows):
            base = y * width
            for x in range(left, left + cols):
                index = base + x
                if index not in volatile and before[index] != after[index]:
                    return True
        return False

    def _track(self, flat: tuple[int, ...], width: int, height: int) -> None:
        """Follow the avatar in board coordinates, and note what refuses it.

        Re-segmenting every action is the obvious way to do this and the
        wrong one -- it is the most expensive thing the agent does. Once
        the motion model says where an action *should* have moved us,
        checking costs two small rectangles instead of a flood fill of
        the board: if neither the square we left nor the square we aimed
        at changed, we did not go anywhere, and something is in the way.

        Only both-changed is read as success. One-changed means something
        happened that this model does not describe, and the honest
        response is to admit we no longer know where we are rather than
        accumulate drift into a map that later gets trusted.
        """
        before, self._track_flat = self._track_flat, flat
        if self._motion is None or not self._motion.settled:
            return
        if before is None or self._at is None or self._pending is None:
            return
        if len(before) != len(flat):
            self._at = None
            self._at_confirmed = 0
            return
        step = self._motion.displacement(self._pending[1] >> (2 * _COORD_BITS))
        if step is None:
            # A click, a reset, or an action the model never explained.
            # It may well have moved us; we simply cannot say.
            self._at = None
            self._at_confirmed = 0
            return
        destination = (self._at[0] + step[0], self._at[1] + step[1])
        arrived = self._region_changed(before, flat, destination, width, height)
        self._learn_readout(
            before, flat, (self._at, destination), width, height
        )
        if arrived is None:
            self._blocked.add((self._look, destination[0], destination[1]))
            return
        left_it = self._region_changed(before, flat, self._at, width, height)
        if not arrived and not left_it:
            self._at_confirmed += 1
            self._blocked.add((self._look, destination[0], destination[1]))
        elif arrived and left_it:
            self._at_confirmed += 1
            self._at = destination
            # Walking onto a square is proof it is not a wall, whatever
            # we believed while wearing something else.
            self._blocked.difference_update(
                {k for k in self._blocked if k[1:] == destination}
            )
        else:
            self._at = None
            self._at_confirmed = 0
        if self._at is not None:
            self._look = self._appearance(flat, self._at, width, height)

    def _skip_blocked(self, node: int, code: int) -> bool:
        blocked = self._blocked_move(code)
        if blocked:
            self._skipped_blocked += 1
            # We know the outcome without paying for it: walking into a
            # wall leaves the board where it was.
            self._edges.setdefault(node, {})[code] = node
        return blocked

    def _blocked_move(self, code: int) -> bool:
        """Would this action walk into something that has already refused
        us, from where we are standing, wearing what we are wearing?"""
        if not (BLOCKED_SKIP and self._blocked) or self._at is None:
            return False
        if self._at_confirmed < BLOCKED_CONFIRM:
            return False
        if self._motion is None:
            return False
        step = self._motion.displacement(code >> (2 * _COORD_BITS))
        if step is None:
            return False
        return (
            self._look,
            self._at[0] + step[0],
            self._at[1] + step[1],
        ) in self._blocked

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

        # See LIVE_CODE_PRIORITY. A stable partition, not a re-sort: codes
        # that have moved the board before move to the front in their
        # existing relative order, everything else keeps its existing
        # relative order behind them. Nothing is dropped, so a node whose
        # only live code is one `candidate_actions` ranked last is reached
        # in 1 probe here instead of up to `breadth`, and a node where none
        # of the previously-live codes apply pays nothing beyond the
        # membership check.
        if LIVE_CODE_PRIORITY and self._code_live:
            live_here = [c for c in codes if c in self._code_live]
            if live_here:
                rest = [c for c in codes if c not in self._code_live]
                codes = live_here + rest

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
            self._is_dead(codes[index])
            or codes[index] in recorded
            or self._skip_blocked(node, codes[index])
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
            left = self._steps_left()
            if left is not None and len(route) > left:
                code = self._local_probe(node, codes)
                if code is not None:
                    self._deadline_probes += 1
                    self._stuck_steps = 0
                    return code, "probe-deadline"
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
        #     Widening unconditionally was measured and rejected. Widening
        #     where the window has demonstrably failed is a different claim:
        #     on ft09 the clicks that move the board sit at ranks 83-91, so
        #     at breadth 12 the search cannot reach them however long it
        #     runs. Fire only once enough clicks have been tried to judge,
        #     and only when almost none of them did anything.
        if (
            LIVENESS_WIDEN
            and self._breadth < BREADTH_MAX
            and self._click_tried >= LIVENESS_MIN_CLICKS
            and self._click_live <= LIVENESS_DEAD_RATE * self._click_tried
        ):
            self._breadth = min(self._breadth * 2, BREADTH_MAX)
            self._stuck_steps = 0
            # Deliberately NOT clearing the dead-action record. That wipe is
            # the suspected cause of eager widening costing tn36 52 -> 86
            # actions, and on a game whose clicks are already inert the
            # record is the most valuable thing the run has learned.
            logger.info(
                "%s: clicks are dead (%d/%d live) after %d actions; "
                "widening breadth to %d",
                self.game_id, self._click_live, self._click_tried,
                self._steps, self._breadth,
            )
            return self._select(node, grid, latest_frame)

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

    def _steps_left(self) -> Optional[int]:
        """Actions left before the game is expected to take the avatar back.

        None when there is no estimate yet, or when the current life has
        already outlasted it -- an overdue respawn means the estimate is
        wrong for this life (a refill was picked up, a hazard was
        dodged), and acting on a negative budget is worse than ignoring
        it.
        """
        if not RESPAWN_AWARE or self._last_respawn is None:
            return None
        if len(self._respawn_periods) < 3:
            return None
        if self._respawn_period < RESPAWN_MIN_PERIOD:
            return None
        left = self._respawn_period - (self._steps - self._last_respawn)
        return left if left >= 0 else None

    def _local_probe(self, node: int, codes: list[int]) -> Optional[int]:
        """An untried action at this node, ignoring the breadth cap.

        Only reached when the cursor has already spent this node at the
        current breadth, so anything it finds is a candidate the search
        was saving for later. Later is exactly what a respawn takes away,
        and unlike a walk, what this buys survives it.
        """
        tried = self._edges.get(node, EMPTY_EDGES)
        for code in codes:
            if code in tried or self._is_dead(code):
                continue
            if (node, code) in self._fatal:
                continue
            return code
        return None

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
