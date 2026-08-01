"""Unit tests for the pure analysis functions in `agent/my_agent.py`.

Run with:  .venv/bin/python -m pytest tests/ -q

These cover the invariants the search silently depends on. In
particular `candidate_actions` MUST be a deterministic function of
(grid, available, node): the agent stores only an integer cursor per
node instead of the candidate list, so if the ordering ever varied
between two visits to the same frame, the cursor would point at a
different action each time and "already tried" bookkeeping would be
quietly wrong.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "ARC-AGI-3-Agents"
if not VENDOR.exists():  # pragma: no cover - depends on `make setup`
    pytest.skip("framework not vendored; run `make setup`", allow_module_level=True)
sys.path.insert(0, str(VENDOR))


def _load():
    spec = importlib.util.spec_from_file_location(
        "my_agent_under_test", ROOT / "agent" / "my_agent.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


agent = _load()


def solid(width: int, height: int, color: int = 0) -> list[list[int]]:
    return [[color] * width for _ in range(height)]


def freeze(grid: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(row) for row in grid)


# -- action packing ---------------------------------------------------
def test_action_codes_round_trip():
    assert agent.decode_action(agent.encode_action(3, None, None)) == (3, None, None)
    for x, y in ((0, 0), (63, 63), (5, 61), (63, 0)):
        code = agent.encode_action(agent._ACTION6_ID, x, y)
        assert agent.decode_action(code) == (agent._ACTION6_ID, x, y)


def test_action_codes_are_unique_across_the_whole_space():
    """Every action the agent can actually emit must have its own code.

    ACTION6 is excluded from the payload-free set on purpose: it is the
    only complex action, so it is never encoded without coordinates, and
    `encode_action(6, None, None)` deliberately shares its packing with
    a click at (0, 0).
    """
    codes = {
        agent.encode_action(agent._ACTION6_ID, x, y)
        for x in range(64)
        for y in range(64)
    }
    simple = [a for a in range(8) if a != agent._ACTION6_ID]
    codes |= {agent.encode_action(a, None, None) for a in simple}
    assert len(codes) == 64 * 64 + len(simple)


# -- click targeting --------------------------------------------------
def test_small_and_rare_objects_are_clicked_first():
    grid = solid(32, 32, color=0)
    for y in range(2, 12):          # large blob, 10x10
        for x in range(2, 12):
            grid[y][x] = 3
    grid[20][20] = 7                # single rare pixel
    targets = agent.click_targets(freeze(grid))
    assert targets[0] == (20, 20), targets[:3]
    assert (7, 7) in targets or any(2 <= x < 12 and 2 <= y < 12 for x, y in targets)


def test_representative_pixel_lies_on_a_hollow_object():
    """A raw centroid would fall in the hole and click the background."""
    grid = solid(32, 32, color=0)
    for x in range(10, 21):
        grid[10][x] = 5
        grid[20][x] = 5
    for y in range(10, 21):
        grid[y][10] = 5
        grid[y][20] = 5
    targets = agent.click_targets(freeze(grid))
    assert targets, "hollow ring should produce a target"
    x, y = targets[0]
    assert grid[y][x] == 5, f"({x}, {y}) landed on background"


def test_coordinates_are_x_column_y_row():
    grid = solid(32, 32, color=0)
    grid[3][17] = 9  # row 3, column 17
    assert agent.click_targets(freeze(grid))[0] == (17, 3)


def test_ignored_cells_are_never_returned():
    grid = solid(16, 16, color=0)
    grid[0][0] = 4
    grid[9][9] = 4
    width = 16
    ignore = frozenset({0 * width + 0})
    targets = agent.click_targets(freeze(grid), ignore=ignore)
    assert (0, 0) not in targets
    assert (9, 9) in targets


def test_uniform_grid_has_no_targets():
    assert agent.click_targets(freeze(solid(16, 16, 2))) == []


def test_lattice_targets_stay_in_bounds():
    points = agent.lattice_targets(64, 64, 8)
    assert len(points) == 64
    assert all(0 <= x <= 63 and 0 <= y <= 63 for x, y in points)


# -- candidate ordering -----------------------------------------------
def test_candidates_are_deterministic_for_a_node():
    """The cursor-per-node design is only sound if this holds."""
    grid = freeze(solid(16, 16, 0))
    first = agent.candidate_actions(grid, [1, 2, 3, 4], node=12345)
    second = agent.candidate_actions(grid, [1, 2, 3, 4], node=12345)
    assert first == second


def test_candidate_order_varies_across_nodes():
    """Without this, every fresh frame takes index 0 and the agent
    degenerates into pressing one button forever."""
    grid = freeze(solid(16, 16, 0))
    firsts = {
        agent.candidate_actions(grid, [1, 2, 3, 4], node=n)[0] for n in range(50)
    }
    assert len(firsts) > 1


def test_simple_only_game_yields_only_those_actions():
    grid = freeze(solid(16, 16, 0))
    codes = agent.candidate_actions(grid, [1, 2, 3, 4], node=1)
    assert sorted(agent.decode_action(c)[0] for c in codes) == [1, 2, 3, 4]
    assert all(agent.decode_action(c)[1] is None for c in codes)


def test_click_game_covers_objects_then_lattice():
    grid = solid(64, 64, 0)
    grid[30][30] = 6
    codes = agent.candidate_actions(freeze(grid), [6], node=1)
    decoded = [agent.decode_action(c) for c in codes]
    assert decoded[0] == (6, 30, 30), "the only object should be probed first"
    assert len(codes) == 1 + 64 - 1 or len(codes) >= 64, len(codes)
    assert all(a == 6 for a, _, _ in decoded)


def test_reset_is_never_a_candidate():
    grid = freeze(solid(16, 16, 0))
    codes = agent.candidate_actions(grid, [0, 1, 2], node=7)
    assert all(agent.decode_action(c)[0] != 0 for c in codes)


def test_empty_available_actions_falls_back_to_probing_everything():
    grid = freeze(solid(16, 16, 0))
    codes = agent.candidate_actions(grid, [], node=7)
    assert {agent.decode_action(c)[0] for c in codes} >= {1, 2, 3, 4, 5, 7}


# -- HUD detection ----------------------------------------------------
def test_shrinking_bar_is_detected_by_its_row_not_its_cells():
    """The ls20 case: rows 61-62 change on ~98% of transitions while no
    individual bar cell exceeds a few percent."""
    width = height = 64
    samples = 400
    cells = [0] * (width * height)
    for index in range(61 * width, 63 * width):
        cells[index] = 8  # each bar cell flips ~2% of the time
    rows = [0] * height
    rows[61] = rows[62] = 393
    cols = [0] * width
    mask = agent.hud_mask(cells, rows, cols, samples, width, height)
    assert mask == frozenset(range(61 * width, 63 * width))


def test_mask_is_empty_when_nothing_is_volatile():
    width = height = 32
    assert agent.hud_mask(
        [0] * (width * height), [0] * height, [0] * width, 200, width, height
    ) == frozenset()


def test_mask_refuses_to_swallow_the_board():
    """If most of the frame qualifies it is not a HUD, and masking it
    would collapse distinct states into a single node."""
    width = height = 32
    rows = [200] * height
    mask = agent.hud_mask(
        [0] * (width * height), rows, [0] * width, 200, width, height
    )
    assert mask == frozenset()


def test_volatile_single_cell_counter_is_masked():
    width = height = 16
    cells = [0] * (width * height)
    cells[5 * width + 5] = 200
    mask = agent.hud_mask(cells, [0] * height, [0] * width, 200, width, height)
    assert mask == frozenset({5 * width + 5})


# ------------------------------------------------------------------
# Firing the mask early (see EARLY_MASK)
# ------------------------------------------------------------------
def _feed(agent_obj, frames, width, height):
    """Push frames through `_learn_mask`, returning the sample it fired on."""
    for i, flat in enumerate(frames):
        if agent_obj._learn_mask(tuple(flat), width, height):
            return i
    return None


def _blank_agent():
    a = agent.MyAgent.__new__(agent.MyAgent)
    a.game_id = "test-0000"          # the fallback logs which game it fired on
    a._mask_frozen = False
    a._prev_flat = None
    a._cell_changes = []
    a._row_changes = []
    a._col_changes = []
    a._samples = 0
    a._volatile = frozenset()
    a._kept = None
    a._early_candidate = frozenset()
    a._early_stable = 0
    # Warmup frames the collapse guard judges candidate masks against. Empty
    # means "no evidence", and the guard abstains rather than vetoing.
    a._mask_frames = []
    return a


def _ticking(width, height, n, digits=4):
    """`n` frames whose only change is a small counter in the top-left.

    Deliberately narrow. A HUD spanning the full width makes every column
    volatile as well as the row, the union covers the whole frame, and
    MASK_CAP_FRACTION then -- correctly -- refuses to mask anything. A real
    counter occupies a few columns, which is what leaves a mask to adopt.
    """
    out = []
    for step in range(n):
        flat = [0] * (width * height)
        for x in range(digits):
            flat[x] = (step + x) % 7 + 1
        out.append(flat)
    return out


def test_the_early_mask_is_off_by_default():
    """It measured as a wash -- vc33 99 -> 66 actions but ls20 526 -> 842,
    with both arms adopting the identical mask and only the timing
    differing. That is trajectory luck, not merit, so it does not ship on."""
    assert agent.EARLY_MASK == 0


def test_a_ticking_row_is_masked_early_when_enabled(monkeypatch):
    monkeypatch.setattr(agent, "EARLY_MASK", 1)
    width = height = 32
    fired = _feed(_blank_agent(), _ticking(width, height, 30), width, height)
    assert fired is not None
    assert fired < agent.VOLATILE_WARMUP, (
        f"fired at sample {fired}; the whole point is to beat "
        f"VOLATILE_WARMUP={agent.VOLATILE_WARMUP}"
    )


def test_the_early_mask_never_fires_later_than_the_baseline_rule(monkeypatch):
    """It may only move the decision earlier. If the unanimous rule never
    converges, the statistical rule at VOLATILE_WARMUP must still fire on
    exactly the sample it always would."""
    width = height = 32
    frames = _ticking(width, height, 40)

    monkeypatch.setattr(agent, "EARLY_MASK", 0)
    late = _feed(_blank_agent(), frames, width, height)
    monkeypatch.setattr(agent, "EARLY_MASK", 1)
    early = _feed(_blank_agent(), frames, width, height)

    assert late is not None and early is not None
    assert early <= late


def _half_rate(width, height, n, digits=4, period=2):
    """A HUD updating every `period`-th step -- real, but under 0.9.

    The board also moves, slowly: without that, masking the HUD fuses every
    frame into one state and the collapse guard rightly refuses the mask.
    cd82, which this models, keeps 12 distinct keys once its HUD is gone.
    The board block changes every fifth step, well under the loosest rung,
    so it never qualifies as volatile itself.
    """
    out = []
    for step in range(n):
        flat = [0] * (width * height)
        for x in range(digits):
            flat[x] = (step // period + x) % 7 + 1
        for x in range(10, 18):
            flat[20 * width + x] = (step // 5) % 3 + 9
        out.append(flat)
    return out


def test_the_mask_fallback_ships_on():
    """Measured across all 25 dev games: cd82 0.00 -> 0.40, 2461 -> 191
    actions and two levels cleared instead of one, 24 games byte-identical,
    aggregate 0.1129 -> 0.1287."""
    assert agent.MASK_FALLBACK == 1


def test_the_ladder_ships_taking_the_loosest_rung():
    """Stopping at the first hit leaves half the gain behind: cd82 lands on
    a 64-cell mask at 0.5 (274 actions, 0.19) instead of the 1016-cell one
    at 0.3 (191 actions, 0.40). Measured 0.1206 vs 0.1287 aggregate."""
    assert agent.MASK_PICK_FIRST == 0


def test_the_ladder_only_ever_relaxes():
    """A rung at or above VOLATILE_THRESHOLD could adopt a mask the strict
    rule had already rejected as too weak, and the rungs must descend or
    'stop at the first hit' silently means 'stop at the loosest'."""
    assert agent.MASK_LADDER
    assert all(t < agent.VOLATILE_THRESHOLD for t in agent.MASK_LADDER)
    assert list(agent.MASK_LADDER) == sorted(agent.MASK_LADDER, reverse=True)


def test_a_half_rate_hud_is_missed_by_the_strict_rule(monkeypatch):
    """The premise of the fallback, so it is pinned separately from it.

    If VOLATILE_THRESHOLD ever starts catching this on its own, the ladder
    is dead code and should be removed rather than left running.
    """
    monkeypatch.setattr(agent, "MASK_FALLBACK", 0)
    width = height = 32
    frames = _half_rate(width, height, agent.VOLATILE_WARMUP + 2)
    a = _blank_agent()
    assert _feed(a, frames, width, height) is None or not a._volatile


def test_the_fallback_catches_a_half_rate_hud(monkeypatch):
    """cd82's failure in miniature: a HUD ticking every other step, missed
    entirely at 0.9, leaving every frame a distinct node."""
    monkeypatch.setattr(agent, "MASK_FALLBACK", 1)
    monkeypatch.setattr(agent, "MASK_LADDER", (0.7, 0.5, 0.3))
    width = height = 32
    frames = _half_rate(width, height, agent.VOLATILE_WARMUP + 2)
    a = _blank_agent()
    fired = _feed(a, frames, width, height)
    assert fired is not None, "the fallback did not adopt any mask"
    assert a._volatile, "fired but masked nothing"


def _two_rate(width, height, n, fast=4, slow=8):
    """A fast HUD block and a slower one, so the rungs disagree.

    The fast cells change every step and qualify at any threshold; the slow
    ones change every third step and only qualify low down the ladder. So
    "first hit" and "loosest that fits" pick measurably different masks --
    which is the whole distinction cd82 turns on.
    """
    out = []
    for step in range(n):
        flat = [0] * (width * height)
        for x in range(fast):
            flat[x] = (step + x) % 7 + 1
        for x in range(slow):
            flat[10 * width + x] = (step // 3 + x) % 7 + 1
        out.append(flat)
    return out


def test_taking_the_loosest_rung_masks_at_least_as_much(monkeypatch):
    """Option 3. Descending past the first hit can only widen the mask, and
    only ever to one MASK_CAP_FRACTION has already approved."""
    monkeypatch.setattr(agent, "MASK_FALLBACK", 1)
    monkeypatch.setattr(agent, "MASK_LADDER", (0.7, 0.5, 0.3))
    width = height = 32
    frames = _two_rate(width, height, agent.VOLATILE_WARMUP + 2)

    monkeypatch.setattr(agent, "MASK_PICK_FIRST", 1)
    first = _blank_agent()
    _feed(first, frames, width, height)

    monkeypatch.setattr(agent, "MASK_PICK_FIRST", 0)
    loosest = _blank_agent()
    _feed(loosest, frames, width, height)

    assert first._volatile, "the fixture must trip the fallback at all"
    assert loosest._volatile >= first._volatile
    assert len(loosest._volatile) <= agent.MASK_CAP_FRACTION * width * height


def test_the_loosest_rung_still_respects_the_cap(monkeypatch):
    """The cap is the only thing standing between a looser threshold and a
    mask that swallows the board, so descending must not bypass it."""
    monkeypatch.setattr(agent, "MASK_FALLBACK", 1)
    monkeypatch.setattr(agent, "MASK_PICK_FIRST", 0)
    monkeypatch.setattr(agent, "MASK_LADDER", (0.7, 0.5, 0.3, 0.05))
    width = height = 32
    frames = [[(step + i) % 5 for i in range(width * height)]
              for step in range(agent.VOLATILE_WARMUP + 2)]
    a = _blank_agent()
    _feed(a, frames, width, height)
    assert not a._volatile


def _widen_agent(module_agent, tried, live, breadth=None):
    """An agent posed at branch (d): map exhausted, nothing left to route to."""
    a = agent.MyAgent.__new__(agent.MyAgent)
    a.game_id = "test-0000"
    a._click_tried = tried
    a._click_live = live
    a._breadth = agent.BREADTH_INITIAL if breadth is None else breadth
    a._stuck_steps = 99
    return a


def _would_widen(a):
    """The LIVENESS_WIDEN gate, evaluated exactly as `_select` evaluates it."""
    return (
        bool(agent.LIVENESS_WIDEN)
        and a._breadth < agent.BREADTH_MAX
        and a._click_tried >= agent.LIVENESS_MIN_CLICKS
        and a._click_live <= agent.LIVENESS_DEAD_RATE * a._click_tried
    )


def test_the_liveness_widen_is_on_by_default():
    """Shipped on after a full 25-game A/B: su15 and s5i5 go from never
    leaving level 0 to clearing it, and the two games that carry the
    aggregate (tn36, sp80) come out byte-identical."""
    assert agent.LIVENESS_WIDEN == 1


def test_dead_clicks_widen_the_window(monkeypatch):
    """ft09: 3 of 677 clicks move anything, and the ones that do sit at
    ranks 83-91 -- unreachable at breadth 12 however long the run lasts."""
    monkeypatch.setattr(agent, "LIVENESS_WIDEN", 1)
    assert _would_widen(_widen_agent(agent, tried=677, live=3))


def test_live_clicks_never_widen(monkeypatch):
    """tn36 moves the board on 43% of clicks and is the game unconditional
    widening broke, 52 -> 86 actions. The gate must never reach it."""
    monkeypatch.setattr(agent, "LIVENESS_WIDEN", 1)
    assert not _would_widen(_widen_agent(agent, tried=1000, live=430))
    # sb26 at 65%, s5i5 at 7% -- the latter is genuinely lowish but still
    # well clear of the 10% floor.
    assert not _would_widen(_widen_agent(agent, tried=1366, live=886))


def test_liveness_needs_a_fair_sample_first(monkeypatch):
    """Two dead clicks in a row is not evidence the window is wrong."""
    monkeypatch.setattr(agent, "LIVENESS_WIDEN", 1)
    assert not _would_widen(_widen_agent(agent, tried=2, live=0))
    assert not _would_widen(
        _widen_agent(agent, tried=agent.LIVENESS_MIN_CLICKS - 1, live=0))


def test_liveness_widen_stops_at_the_ceiling(monkeypatch):
    monkeypatch.setattr(agent, "LIVENESS_WIDEN", 1)
    assert not _would_widen(
        _widen_agent(agent, tried=677, live=3, breadth=agent.BREADTH_MAX))


def test_the_liveness_counters_reset_with_the_breadth_they_govern():
    """Breadth resets per level because each level is scored on its own
    actions; the hit rate that decides whether to widen must reset with it,
    or a dead level-1 keeps widening a level 2 that clicks fine."""
    a = agent.MyAgent.__new__(agent.MyAgent)
    a.game_id = "test-0000"
    a._click_tried = 500
    a._click_live = 2
    a._sol_edges = {}
    a._sol_start = 0
    a._sol_full = False
    a._sol_count = 0
    a._breadth = 96
    a._act_tried = agent.Counter()
    a._act_changed = agent.Counter()
    agent.MyAgent._begin_level(a, 0)
    assert a._breadth == agent.BREADTH_INITIAL
    assert a._click_tried == 0 and a._click_live == 0


def test_the_collapse_guard_ships_on():
    """Measured across all 25 dev games: five masks vetoed, all for fusing
    every sampled frame into one state, tr87 back to clearing level 1
    (5001 -> 2538 actions), no game regressed. Aggregate is unmoved at
    0.1288 -- this ships as a correctness fix, not a scoring one."""
    assert agent.MASK_COLLAPSE_GUARD == 1


def test_a_small_mask_may_fuse_every_frame(monkeypatch):
    """tn36's case, and the reason collapse alone cannot be the rule: it
    masks 64 cells, fuses all 26 warmup frames into one key, and is the
    highest-scoring game on the dev set. The board really was static."""
    monkeypatch.setattr(agent, "MASK_COLLAPSE_GUARD", 1)
    total = 32 * 32
    a = _blank_agent()
    # Frames identical apart from a tiny corner that ticks every step.
    a._mask_frames = [
        bytes([step % 5 if i < 8 else 3 for i in range(total)])
        for step in range(20)
    ]
    small = frozenset(range(8))          # 8 cells, 0.8% of the frame
    assert a._mask_is_credible(small, total)


def test_a_large_mask_that_fuses_every_frame_is_rejected(monkeypatch):
    """tr87's case: a 28.8% mask collapsing everything is not a HUD, it is
    the mask deleting the board, and taking it loses a level-1 clear."""
    monkeypatch.setattr(agent, "MASK_COLLAPSE_GUARD", 1)
    total = 32 * 32
    a = _blank_agent()
    a._mask_frames = [
        bytes([step % 5 if i < 400 else 3 for i in range(total)])
        for step in range(20)
    ]
    big = frozenset(range(400))          # 39% of the frame
    assert not a._mask_is_credible(big, total)


def test_a_large_mask_that_preserves_distinct_states_is_kept(monkeypatch):
    """cd82's case: 16.5% of the frame, still 12 distinct keys left. That
    is a real HUD on a game that genuinely moves, and rejecting it would
    throw away the largest single gain measured so far."""
    monkeypatch.setattr(agent, "MASK_COLLAPSE_GUARD", 1)
    total = 32 * 32
    a = _blank_agent()
    a._mask_frames = [
        bytes([step % 5 if i < 400 else (step % 7) + 8 for i in range(total)])
        for step in range(20)
    ]
    big = frozenset(range(400))
    assert a._mask_is_credible(big, total)


def test_the_guard_never_rescues_a_mask_the_cap_refused(monkeypatch):
    """It is a veto and nothing else. Without frames to judge against it
    must abstain rather than invent an opinion."""
    monkeypatch.setattr(agent, "MASK_COLLAPSE_GUARD", 1)
    a = _blank_agent()
    a._mask_frames = []
    assert a._mask_is_credible(frozenset(range(900)), 32 * 32)


def test_the_guard_survives_a_grid_that_grew_mid_warmup(monkeypatch):
    """`_learn_mask` explicitly expects the grid to be resized, and restarts
    its counters when it is -- but the sampled frames outlive that reset, so
    the list can hold two different frame sizes at once. Indexing a 16x16
    frame at an offset from a 32x32 one is an IndexError, uncaught, on the
    action path. No dev game resizes inside its first VOLATILE_WARMUP
    frames, which is why 25 local games never found this."""
    monkeypatch.setattr(agent, "MASK_COLLAPSE_GUARD", 1)
    a = _blank_agent()
    a._mask_frames = [bytes([3] * (16 * 16)) for _ in range(5)]
    a._mask_frames += [bytes([3] * (32 * 32)) for _ in range(5)]
    # Must reach a verdict rather than raise.
    assert a._mask_is_credible(frozenset(range(400)), 32 * 32) in (True, False)


def test_a_resize_during_warmup_drops_the_stale_frames(monkeypatch):
    """The counters restart on resize; the frames have to go with them. A
    frame from before the resize cannot be compared against a mask indexed
    for after it, so keeping it is meaningless as well as unsafe."""
    monkeypatch.setattr(agent, "MASK_COLLAPSE_GUARD", 1)
    a = _blank_agent()
    for _ in range(4):
        a._learn_mask(tuple([3] * (16 * 16)), 16, 16)
    assert a._mask_frames, "frames should accumulate during warmup"
    a._learn_mask(tuple([3] * (32 * 32)), 32, 32)
    assert {len(f) for f in a._mask_frames} <= {32 * 32}


def test_the_fallback_cannot_touch_a_game_that_already_masks(monkeypatch):
    """The safety argument for shipping this at all: a game whose HUD
    already clears VOLATILE_THRESHOLD must get bit-for-bit the same mask,
    because relaxing the threshold for those games was measured and broke
    them (sp80 78 -> 211 actions)."""
    width = height = 32
    frames = _ticking(width, height, agent.VOLATILE_WARMUP + 2)

    monkeypatch.setattr(agent, "MASK_FALLBACK", 0)
    plain = _blank_agent()
    _feed(plain, frames, width, height)

    monkeypatch.setattr(agent, "MASK_FALLBACK", 1)
    monkeypatch.setattr(agent, "MASK_LADDER", (0.7, 0.5, 0.3))
    relaxed = _blank_agent()
    _feed(relaxed, frames, width, height)

    assert plain._volatile
    assert plain._volatile == relaxed._volatile


def test_the_fallback_still_refuses_to_swallow_the_board(monkeypatch):
    """Relaxing the threshold must not relax the cap. m0r0 qualifies 44% of
    its frame at 0.3; masking that would fuse unrelated states into one
    node, which is worse than the exploding map it would be curing."""
    monkeypatch.setattr(agent, "MASK_FALLBACK", 1)
    monkeypatch.setattr(agent, "MASK_LADDER", (0.7, 0.5, 0.3, 0.1))
    width = height = 32
    frames = []
    for step in range(agent.VOLATILE_WARMUP + 2):
        # Everything moves, all the time: not a HUD, just a busy board.
        frames.append([(step + i) % 5 for i in range(width * height)])
    a = _blank_agent()
    _feed(a, frames, width, height)
    assert not a._volatile


def test_an_erratic_cell_does_not_trigger_the_early_mask(monkeypatch):
    """The early rule demands change on EVERY transition. A cell that is
    merely busy must wait for the statistical rule, or a moving player
    piece gets masked and distinct states collapse into one node."""
    monkeypatch.setattr(agent, "EARLY_MASK", 1)
    width = height = 32
    frames = []
    for step in range(agent.VOLATILE_WARMUP - 2):
        flat = [0] * (width * height)
        # Hold each value for three frames, so two transitions out of every
        # three see no change at all. Alternating 0,5,0,5 would not do:
        # every transition is a change, which is genuinely unanimous.
        flat[3 * width + 3] = (step // 3) % 6
        frames.append(flat)
    assert _feed(_blank_agent(), frames, width, height) is None


# -- Masking the widget, not the line (see `_widget_span`) -------------
#
# ls20 is the case these exist for. Its step counter occupies rows 61-62
# of the frame; the only display of the avatar's shape, colour and
# rotation is a 10x10 box sharing those rows. Masking the rows end to
# end takes the box with the counter, and those three attributes are the
# whole game -- you clear a level by standing on a target while matching
# its triple, and they appear nowhere else on the board.
_W = 16
_ROW = 10
_BAR = range(4, 12)        # the counter
_READOUT = range(0, 3)     # a status box beside it, in the same row


def _bar_frames(drained):
    """Two frames of a bar draining left to right, as flat bytes.

    `drained` is how many segments have gone. The readout keeps a colour
    of its own throughout, which is what should save it from the mask.
    """
    out = []
    for gone in (0, drained):
        flat = [0] * (_W * _W)
        for x in _READOUT:
            flat[_ROW * _W + x] = 5
        for i, x in enumerate(_BAR):
            flat[_ROW * _W + x] = 1 if i < gone else 3
        out.append(bytes(flat))
    return out


def _bar_counts(changed):
    """cell_changes/row_changes for a bar whose first `changed` cells moved."""
    cells = [0] * (_W * _W)
    for x in list(_BAR)[:changed]:
        cells[_ROW * _W + x] = 10       # well under the per-cell cutoff
    rows = [0] * _W
    rows[_ROW] = 95                     # the row itself is unmistakable
    return cells, rows


def test_the_mask_stops_at_the_edge_of_the_widget():
    cells, rows = _bar_counts(4)
    mask = agent.hud_mask(
        cells, rows, [0] * _W, 100, _W, _W, frames=_bar_frames(4), span=True
    )
    assert mask == {_ROW * _W + x for x in _BAR}


def test_a_readout_sharing_the_row_survives():
    """The regression this whole change exists to prevent."""
    cells, rows = _bar_counts(4)
    mask = agent.hud_mask(
        cells, rows, [0] * _W, 100, _W, _W, frames=_bar_frames(4), span=True
    )
    assert not any(_ROW * _W + x in mask for x in _READOUT)


def test_the_span_covers_the_part_of_the_bar_that_has_not_moved_yet():
    """A bar drains one segment per step, so a 25-frame warmup sees only
    its leading edge. Masking just the cells that changed leaves the rest
    of the counter in the state key, and a counter in the key mints a new
    node every step -- measured on ls20 as 96 nodes becoming 735."""
    cells, rows = _bar_counts(2)
    mask = agent.hud_mask(
        cells, rows, [0] * _W, 100, _W, _W, frames=_bar_frames(2), span=True
    )
    assert mask == {_ROW * _W + x for x in _BAR}


def test_without_frames_the_span_is_only_what_changed():
    """No evidence of the widget's colours means no basis to grow, so it
    falls back to the cells actually seen moving rather than guessing."""
    cells, rows = _bar_counts(2)
    mask = agent.hud_mask(cells, rows, [0] * _W, 100, _W, _W, span=True)
    assert mask == {_ROW * _W + x for x in list(_BAR)[:2]}


def test_the_whole_row_is_masked_by_default():
    """The span rule ships off: it is correct and it costs score (see
    MASK_SPAN). The default therefore has to be the whole-line rule that
    produced 0.1299, exactly, or the A/B measures two changes at once."""
    assert agent.MASK_SPAN == 0
    cells, rows = _bar_counts(4)
    mask = agent.hud_mask(
        cells, rows, [0] * _W, 100, _W, _W, frames=_bar_frames(4)
    )
    assert mask == set(range(_ROW * _W, _ROW * _W + _W))


# -- The status band (see `status_band`) -------------------------------
def test_the_band_is_what_shares_the_line_but_does_not_tick():
    rows = [0] * _W
    rows[_ROW] = 95
    volatile = frozenset(_ROW * _W + x for x in _BAR)
    band = agent.status_band(rows, [0] * _W, volatile, 100, _W, _W)
    assert set(band) == {
        _ROW * _W + x for x in range(_W) if x not in _BAR
    }


def test_the_band_is_empty_when_no_line_ticks():
    assert agent.status_band([0] * _W, [0] * _W, frozenset(), 100, _W, _W) == ()


def test_a_band_the_size_of_the_board_is_not_a_band():
    """Every row ticking means the reading is the board, and re-opening
    the map whenever the board changes would mean never settling."""
    band = agent.status_band(
        [100] * _W, [0] * _W, frozenset(), 100, _W, _W
    )
    assert band == ()


# -- Telling a respawn from a discovery (see `_check_status`) ----------
def _status_agent():
    a = agent.MyAgent.__new__(agent.MyAgent)
    a.game_id = "test-0000"
    a._steps = 0
    a._status_idx = (0, 1)
    a._status = None
    a._status_reopens = 0
    a._status_boards = {}
    a._board_idx = (2, 3)
    a._kept = None
    a._last_flat = None
    a._level_start_node = None
    a._level_start_board = None
    a._last_respawn = None
    a._respawn_period = 0
    a._respawns = 0
    a._total = {1: 1}
    a._cursor = {1: 3}
    a._plan = agent.deque()
    return a


def _read(a, status, board, step):
    a._steps = step
    a._last_flat = tuple(status) + tuple(board)
    return a._check_status(node=1)


def test_a_status_change_on_a_new_board_re_opens_the_map(monkeypatch):
    monkeypatch.setattr(agent, "STATUS_REOPEN", 1)
    a = _status_agent()
    _read(a, (1, 1), (7, 7), 0)
    assert _read(a, (2, 1), (8, 8), 10) is True
    assert a._cursor == {}


def test_a_status_change_back_onto_a_known_board_is_a_respawn(monkeypatch):
    """ls20 dies every 42 steps and puts the avatar back at the start.
    That moves the readout too, and means the opposite thing: nothing was
    learned. Before this check the search re-opened every 43 actions and
    level 1 stopped finishing at all inside 3000."""
    monkeypatch.setattr(agent, "STATUS_REOPEN", 1)
    a = _status_agent()
    _read(a, (1, 1), (7, 7), 0)
    _read(a, (2, 1), (9, 9), 44)       # first sighting: taken at face value
    assert _read(a, (3, 1), (9, 9), 88) is False
    assert a._respawns == 1
    assert a._status_reopens == 1      # only the first sighting re-opened


def test_the_respawn_period_is_measured():
    a = _status_agent()
    _read(a, (1, 1), (7, 7), 0)
    _read(a, (2, 1), (9, 9), 44)
    _read(a, (3, 1), (9, 9), 88)
    _read(a, (4, 1), (9, 9), 131)
    assert a._respawn_period == 43


def test_re_opening_is_off_by_default():
    """It is subsumed: the readout is in the state key, so a square that
    rejected the agent at one rotation is a different node from the same
    square at another, with its own untried candidates. Re-opening only
    replays edges already recorded, and on ls20 turned a 914-action level
    1 into one that never finished."""
    assert agent.STATUS_REOPEN == 0
    a = _status_agent()
    _read(a, (1, 1), (7, 7), 0)
    assert _read(a, (2, 1), (8, 8), 10) is False
    assert a._cursor == {1: 3}
