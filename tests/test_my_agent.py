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
