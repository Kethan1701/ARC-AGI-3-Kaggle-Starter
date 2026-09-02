"""Tests for the token-budget trim and the model-concurrency slot.

Both exist for one reason: the server's prefix cache measured 0.0% on
every LLM run, and v46base (the graded 0.05) burned 2088 tok/s of prefill
against 254 tok/s of generation because of it. A cache hit needs the
prefix to be BYTE-IDENTICAL to something still resident, so the property
worth pinning here is not "the transcript is small" -- it is "the
transcript is not touched when it does not need to be."

`evict_turns` fails that by construction: it recomputes a cut point on
every call, so the prefix moves every turn even when nothing needed to
go. The tests below pin the difference.
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
        "my_agent_context_test", ROOT / "agent" / "my_agent.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MA = _load()


def _turn(tag: str, size: int = 200) -> list[dict]:
    """One user turn plus its assistant reply."""
    return [
        {"role": "user", "content": f"{tag}-user " * size},
        {"role": "assistant", "content": f"{tag}-reply " * size},
    ]


def _transcript(turns: int) -> list[dict]:
    messages = [{"role": "system", "content": "system prompt " * 50}]
    for i in range(turns):
        messages.extend(_turn(f"t{i}"))
    return messages


def test_defaults_are_inert():
    """Nothing here may touch a run until it is switched on."""
    assert MA.LLM_CONCURRENCY == 0
    assert MA._LLM_SLOTS is None
    assert MA.LLM_CONTEXT_BUDGET == 0


def test_budget_zero_is_a_no_op():
    messages = _transcript(6)
    assert MA.trim_to_budget(messages, 0, 0.25) is messages


def test_fitting_transcript_is_returned_unchanged():
    """The load-bearing property: identity, not just equality.

    A copy would still be byte-identical on the wire, but returning the
    same object is the cheap proof that no cut point was recomputed.
    """
    messages = _transcript(6)
    assert MA.trim_to_budget(messages, 10**9, 0.25) is messages


def test_over_budget_trims_and_keeps_the_framing():
    messages = _transcript(12)
    budget = MA.estimate_tokens(messages) // 2
    out = MA.trim_to_budget(messages, budget, 0.25)
    assert len(out) < len(messages)
    assert out[0]["role"] == "system"
    assert out[1]["role"] == "user"
    assert MA.estimate_tokens(out) <= budget


def test_trim_leaves_headroom_so_the_next_turns_are_untouched():
    """The whole point of the headroom, and the v25 failure it avoids.

    Trimming to exactly the budget means the next appended turn goes over
    again and the cut point moves every single turn -- which is what
    `evict_turns` does and why stabilising it never helped. With headroom,
    several consecutive turns must append to an identical prefix.
    """
    messages = _transcript(12)
    budget = MA.estimate_tokens(messages) // 2
    trimmed = MA.trim_to_budget(messages, budget, 0.25)

    untouched = 0
    current = list(trimmed)
    for i in range(4):
        current.extend(_turn(f"new{i}"))
        after = MA.trim_to_budget(current, budget, 0.25)
        if after is current:
            untouched += 1
        current = list(after)
    assert untouched >= 1, "headroom bought no untouched turns at all"


def test_trim_never_orphans_a_reply():
    """A turn is dropped whole: the survivor must start at a user message.

    Half a turn leaves an assistant/tool message with no request in front
    of it, which the server rejects outright.
    """
    messages = _transcript(12)
    for divisor in (2, 3, 4, 6):
        out = MA.trim_to_budget(messages, MA.estimate_tokens(messages) // divisor, 0.25)
        assert out[1]["role"] == "user", (divisor, [m["role"] for m in out[:4]])


def test_framing_survives_an_absurd_budget():
    """The system prompt and first user turn are a floor, not negotiable.

    A budget of 1 token cannot be met, and the function must bottom out at
    the framing rather than trimming into it: a model that loses the first
    turn reasons about a board with no idea what the game is.
    """
    messages = _transcript(1)
    out = MA.trim_to_budget(messages, 1, 0.25)
    assert [m["role"] for m in out] == ["system", "user"]


def test_nothing_to_trim_is_returned_unchanged():
    """Below the head there is nothing to drop, so don't rebuild the list."""
    messages = [
        {"role": "system", "content": "s" * 4000},
        {"role": "user", "content": "u" * 4000},
    ]
    assert MA.trim_to_budget(messages, 1, 0.25) is messages


def test_estimate_tokens_charges_for_an_image():
    """An image is not characters; if it were free it would hide in the
    budget and the transcript would silently blow past the server's cap."""
    text_only = [{"role": "user", "content": [{"type": "text", "text": "x" * 40}]}]
    with_image = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "x" * 40},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}},
            ],
        }
    ]
    assert MA.estimate_tokens(with_image) - MA.estimate_tokens(text_only) >= 1024
