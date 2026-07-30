"""Tests for `segment_frame`, the object view the model reasons over.

The raw grid is never handed to the model, so these node records are the
whole of what it can see. Anything wrong here is invisible downstream and
shows up only as the model reasoning confidently about a board that is not
the board.
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
        "my_agent_segmentation_test", ROOT / "agent" / "my_agent.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


agent = _load()
segment_frame = agent.segment_frame


def _grid(rows: list[str]) -> tuple[tuple[int, ...], ...]:
    """Build a grid from digit strings, so cases read as pictures."""
    return tuple(tuple(int(ch) for ch in row) for row in rows)


def test_an_empty_grid_is_not_an_error():
    """`Agent.frames` is seeded with an empty placeholder frame, so the
    empty case is reachable and must not raise."""
    assert segment_frame(()) == {"nodes": [], "adjacency_list": []}
    assert segment_frame((())) == {"nodes": [], "adjacency_list": []}


def test_a_uniform_grid_is_one_object():
    out = segment_frame(_grid(["000", "000"]))
    assert len(out["nodes"]) == 1
    node = out["nodes"][0]
    assert node["id"] == 0
    assert node["color"] == 0
    assert node["pixels"] == 6
    assert node["bbox"] == (0, 0, 1, 2)
    assert out["adjacency_list"] == []


def test_ids_run_top_most_then_left_most():
    """The model is told ids are ordered that way and will rely on it to
    refer to objects stably within a frame."""
    out = segment_frame(_grid(["010", "000", "020"]))
    colors = [n["color"] for n in out["nodes"]]
    # background first (0,0), then the 1 at row 0, then the 2 at row 2.
    assert colors == [0, 1, 2]


def test_the_same_shape_and_color_hashes_equal_anywhere():
    """This is the cross-frame object identity the model tracks with, so
    position must not enter the hash."""
    out = segment_frame(
        _grid(
            [
                "330000",
                "330000",
                "000000",
                "000000",
                "000033",
                "000033",
            ]
        )
    )
    threes = [n for n in out["nodes"] if n["color"] == 3]
    assert len(threes) == 2
    assert threes[0]["hash"] == threes[1]["hash"]
    assert threes[0]["bbox"] != threes[1]["bbox"]


def test_a_different_shape_hashes_differently():
    out = segment_frame(_grid(["3300", "3000", "0000"]))
    square = segment_frame(_grid(["3300", "3300", "0000"]))
    a = next(n for n in out["nodes"] if n["color"] == 3)
    b = next(n for n in square["nodes"] if n["color"] == 3)
    assert a["pixels"] != b["pixels"]
    assert a["hash"] != b["hash"]


def test_touching_objects_are_adjacent_and_diagonal_ones_are_not():
    """4-connected throughout: a diagonal touch is not an edge, and the
    model's adjacency reasoning depends on that being consistent."""
    out = segment_frame(_grid(["12", "21"]))
    # Four single cells; 0 is (0,0)=1, 1 is (0,1)=2, 2 is (1,0)=2, 3 is (1,1)=1.
    assert len(out["nodes"]) == 4
    pairs = {tuple(p) for p in out["adjacency_list"]}
    assert (0, 1) in pairs and (0, 2) in pairs      # shares an edge
    assert (1, 3) in pairs and (2, 3) in pairs
    assert (0, 3) not in pairs                      # diagonal only
    assert (1, 2) not in pairs


def test_a_ring_encloses_what_is_inside_it():
    """Containment is the point of `children`: a box with something in it is
    a different situation from a box beside something."""
    out = segment_frame(
        _grid(
            [
                "00000",
                "01110",
                "01210",
                "01110",
                "00000",
            ]
        )
    )
    ring = next(n for n in out["nodes"] if n["color"] == 1)
    centre = next(n for n in out["nodes"] if n["color"] == 2)
    assert ring["pixels"] == 8
    assert ring["children"] == [centre["id"]]
    assert centre["children"] == []


def test_an_object_beside_another_encloses_nothing():
    """The negative case for the same machinery -- 8+ cells adjacent to a
    neighbour must not be reported as containing it."""
    out = segment_frame(
        _grid(
            [
                "11112",
                "11110",
                "00000",
            ]
        )
    )
    block = next(n for n in out["nodes"] if n["color"] == 1)
    assert block["pixels"] >= 8
    assert block["children"] == []


def test_a_background_touching_every_border_encloses_everything():
    """Pinning a consequence of the definition rather than hiding it: if
    nothing can reach the border without crossing the background, then by
    the containment rule the background encloses every other object. That
    is literally true and it is what the code reports, so it is written
    down here instead of being discovered later as a surprise."""
    out = segment_frame(
        _grid(
            [
                "0000",
                "0110",
                "0110",
                "0000",
            ]
        )
    )
    background = out["nodes"][0]
    block = next(n for n in out["nodes"] if n["color"] == 1)
    assert background["color"] == 0
    assert background["children"] == [block["id"]]


def test_ignored_cells_do_not_become_objects():
    """The HUD mask feeds in here. A step counter's digits arriving as a row
    of tiny high-salience objects is the failure this prevents."""
    rows = ["1230", "0000"]
    grid = _grid(rows)
    width = len(rows[0])
    hud = frozenset({0, 1, 2})          # the '1', '2', '3' on the top row
    out = segment_frame(grid, ignore=hud)
    assert all(n["color"] == 0 for n in out["nodes"])
    assert sum(n["pixels"] for n in out["nodes"]) == 8 - len(hud)
    assert width == 4


def test_segmentation_is_deterministic():
    """The search stores one cursor per node rather than a candidate list,
    which is only sound if the parse is stable for a given grid."""
    grid = _grid(["01120", "01100", "22200"])
    first = segment_frame(grid)
    second = segment_frame(grid)
    assert first == second
