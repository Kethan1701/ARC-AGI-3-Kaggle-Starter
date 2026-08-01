"""Tests for the rendered board the model actually sees.

The PNG is hand-written, so "it produced bytes" proves nothing. These decode
it -- with Pillow where available, and by parsing the chunks by hand
regardless -- because a subtly malformed image would be accepted by vLLM's
loader as garbage or rejected mid-run, and either way the first sign of it
would be a wasted GPU run.
"""

from __future__ import annotations

import base64
import importlib.util
import struct
import sys
import zlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "ARC-AGI-3-Agents"
if not VENDOR.exists():  # pragma: no cover - depends on `make setup`
    pytest.skip("framework not vendored; run `make setup`", allow_module_level=True)
sys.path.insert(0, str(VENDOR))


def _load():
    spec = importlib.util.spec_from_file_location(
        "my_agent_render_test", ROOT / "agent" / "my_agent.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


agent = _load()


def _chunks(png: bytes):
    """Walk the PNG structure, verifying every CRC as we go."""
    assert png[:8] == b"\x89PNG\r\n\x1a\n", "bad signature"
    out = []
    i = 8
    while i < len(png):
        (length,) = struct.unpack(">I", png[i : i + 4])
        tag = png[i + 4 : i + 8]
        data = png[i + 8 : i + 8 + length]
        (crc,) = struct.unpack(">I", png[i + 8 + length : i + 12 + length])
        assert crc == zlib.crc32(tag + data) & 0xFFFFFFFF, f"bad CRC on {tag!r}"
        out.append((tag, data))
        i += 12 + length
    return out


def _decode(png: bytes):
    """(width, height, rows-of-RGB-tuples), decoded from the raw stream."""
    tags = dict(_chunks(png))
    width, height, depth, colour = struct.unpack(">IIBB", tags[b"IHDR"][:10])
    assert depth == 8 and colour == 2, "expected 8-bit truecolour"
    raw = zlib.decompress(tags[b"IDAT"])
    stride = width * 3
    rows = []
    for y in range(height):
        start = y * (stride + 1)
        assert raw[start] == 0, "only filter type 0 is written"
        line = raw[start + 1 : start + 1 + stride]
        rows.append([tuple(line[x * 3 : x * 3 + 3]) for x in range(width)])
    return width, height, rows


def test_the_png_is_structurally_valid():
    png = agent.render_png(((0, 1), (2, 3)), upscale=1)
    tags = [tag for tag, _ in _chunks(png)]
    assert tags == [b"IHDR", b"IDAT", b"IEND"]


def test_every_cell_becomes_its_palette_colour():
    grid = ((0, 4), (2, 10))
    _, _, rows = _decode(agent.render_png(grid, upscale=1))
    assert rows[0][0] == agent.PALETTE[0]
    assert rows[0][1] == agent.PALETTE[4]
    assert rows[1][0] == agent.PALETTE[2]
    assert rows[1][1] == agent.PALETTE[10]


def test_upscaling_makes_each_cell_a_square_block():
    width, height, rows = _decode(agent.render_png(((0, 4),), upscale=3))
    assert (width, height) == (6, 3)
    for y in range(3):
        for x in range(3):
            assert rows[y][x] == agent.PALETTE[0]
            assert rows[y][x + 3] == agent.PALETTE[4]


def test_a_real_sized_board_round_trips():
    grid = tuple(tuple((x * y) % 16 for x in range(64)) for y in range(64))
    png = agent.render_png(grid)
    width, height, rows = _decode(png)
    assert (width, height) == (64 * agent.RENDER_UPSCALE,) * 2
    # Spot-check a cell in the middle, at the top-left pixel of its block.
    u = agent.RENDER_UPSCALE
    assert rows[30 * u][40 * u] == agent.PALETTE[(40 * 30) % 16]


def test_the_image_stays_small_enough_to_send():
    """It rides in the prompt on every turn, so a fat one costs tokens and
    bandwidth 25 games at a time."""
    grid = tuple(tuple((x * 7 + y * 13) % 16 for x in range(64)) for y in range(64))
    assert len(agent.render_png(grid)) < 60_000


def test_the_upscale_resolves_cells_rather_than_blurring_them():
    """A ViT token covers ~28-32px. At TAAF's upscale of 4 that is eight
    board cells per token, which loses the detail the image exists for."""
    assert 64 * agent.RENDER_UPSCALE >= 512


def test_pillow_agrees_with_our_encoder():
    """An independent decoder, when one is installed. Our own parser could
    share a bug with our writer; Pillow cannot."""
    Image = pytest.importorskip("PIL.Image")
    from io import BytesIO

    grid = ((0, 4, 9), (2, 10, 15))
    img = Image.open(BytesIO(agent.render_png(grid, upscale=2))).convert("RGB")
    assert img.size == (6, 4)
    assert img.getpixel((0, 0)) == agent.PALETTE[0]
    assert img.getpixel((2, 0)) == agent.PALETTE[4]
    assert img.getpixel((4, 2)) == agent.PALETTE[15]


def test_the_palette_covers_every_value_a_game_can_emit():
    """All 16 occur: measured across the opening frame of all 25 games."""
    assert len(agent.PALETTE) == 16


def test_no_two_values_share_a_colour():
    """The framework's own palette maps 5 and 11 to the same grey. On a
    puzzle whose whole content is which cells differ, that is not a
    cosmetic problem."""
    assert len(set(agent.PALETTE)) == 16


def test_an_empty_grid_yields_an_image_rather_than_an_exception():
    """This runs on the action path."""
    for grid in ((), ((),)):
        assert agent.render_png(grid)[:8] == b"\x89PNG\r\n\x1a\n"


def test_the_data_uri_is_a_png():
    uri = agent.render_data_uri(((1, 2), (3, 4)), upscale=1)
    assert uri.startswith("data:image/png;base64,")
    blob = base64.b64decode(uri.split(",", 1)[1])
    assert blob[:8] == b"\x89PNG\r\n\x1a\n"


# --- keeping one image in flight ---------------------------------------


def test_stripping_keeps_the_text_and_drops_the_picture():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": [
            {"type": "text", "text": "board one"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
        ]},
    ]
    agent.strip_images(messages)
    assert messages[1]["content"] == "board one"
    assert messages[0]["content"] == "sys"


def test_stripping_leaves_plain_messages_alone():
    messages = [{"role": "tool", "content": "result = 3"}]
    agent.strip_images(messages)
    assert messages[0]["content"] == "result = 3"


def test_only_one_image_survives_across_turns():
    """Four turns of boards would be thousands of tokens spent showing the
    model three positions that no longer exist and it cannot act on."""
    messages = [{"role": "system", "content": "sys"}]
    for i in range(4):
        agent.strip_images(messages)
        messages.append({"role": "user", "content": [
            {"type": "text", "text": f"turn {i}"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
        ]})

    images = sum(
        1
        for m in messages
        if isinstance(m.get("content"), list)
        for part in m["content"]
        if part.get("type") == "image_url"
    )
    assert images == 1
    assert messages[-1]["content"][0]["text"] == "turn 3"
