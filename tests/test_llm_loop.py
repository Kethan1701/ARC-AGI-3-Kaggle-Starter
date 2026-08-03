"""Tests for the tool-calling loop.

The real model only exists on a Kaggle GPU, so the transport is injected and
every path here runs against a scripted stand-in. That covers the protocol --
which is where this will actually break -- without needing 27B of weights.
"""

from __future__ import annotations

import importlib.util
import json
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
        "my_agent_llm_test", ROOT / "agent" / "my_agent.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


agent = _load()


class _Scripted:
    """Stands in for the vLLM server, recording what it was sent."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.payloads = []

    def __call__(self, payload):
        self.payloads.append(payload)
        if not self.responses:
            raise AssertionError("the loop asked for more replies than scripted")
        return self.responses.pop(0)


def _assistant(content=None, tool_calls=None):
    return {"choices": [{"message": {
        "role": "assistant", "content": content, "tool_calls": tool_calls,
    }}]}


def _call(call_id, code):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "python", "arguments": json.dumps({"code": code})},
    }


def test_a_reply_without_tool_calls_ends_the_turn():
    post = _Scripted([_assistant(content="I will wait.")])
    messages = [{"role": "user", "content": "go"}]
    out = agent.run_tool_conversation(messages, {}, post=post)
    assert out == {"stop": "content", "tool_calls": 0, "content": "I will wait."}


def test_a_tool_call_runs_and_is_fed_back():
    post = _Scripted([
        _assistant(tool_calls=[_call("c1", "print(1 + 1)")]),
        _assistant(content="two"),
    ])
    messages = [{"role": "user", "content": "go"}]
    out = agent.run_tool_conversation(messages, {}, post=post)

    assert out["stop"] == "content"
    assert out["tool_calls"] == 1
    tool_message = messages[2]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "c1"
    assert tool_message["name"] == "python"
    assert "2" in tool_message["content"]


def test_the_payload_declares_the_python_tool_and_qwen_sampling():
    post = _Scripted([_assistant(content="ok")])
    agent.run_tool_conversation([], {}, post=post, model="m")
    payload = post.payloads[0]
    assert payload["model"] == "m"
    assert payload["tool_choice"] == "auto"
    assert [t["function"]["name"] for t in payload["tools"]] == ["python"]
    # top_k is not in OpenAI's schema; vLLM accepts it top-level and Qwen3
    # wants it set, so it must actually be on the wire.
    assert payload["top_k"] == agent.LLM_TOP_K
    assert payload["temperature"] == agent.LLM_TEMPERATURE
    assert payload["top_p"] == agent.LLM_TOP_P


def test_one_reply_is_capped_so_a_runaway_cannot_eat_the_run():
    """`max_tokens` was missing from the payload entirely until v23.

    With `preserve_thinking` on and no cap, a single reply could run to
    whatever `--max-model-len` left over. At v22's measured ~8 tok/s per
    game that is a game's whole hour inside one generation.
    """
    post = _Scripted([_assistant(content="ok")])
    agent.run_tool_conversation([], {}, post=post)
    assert post.payloads[0]["max_tokens"] == agent.LLM_MAX_TOKENS


def test_the_cap_is_overridable_per_call():
    post = _Scripted([_assistant(content="ok")])
    agent.run_tool_conversation([], {}, post=post, max_tokens=64)
    assert post.payloads[0]["max_tokens"] == 64


def test_a_reply_cut_off_at_the_cap_is_not_reported_as_a_finished_thought():
    """`length` and a real stop look identical in a turn report unless they
    are named apart -- and a cap that silently ends turns looks like a
    throughput win while quietly costing the run."""
    reply = _assistant(content="I was still thinking when")
    reply["choices"][0]["finish_reason"] = "length"
    post = _Scripted([reply])
    out = agent.run_tool_conversation([], {}, post=post)
    assert out["stop"] == "truncated"
    assert out["content"] == "I was still thinking when"


def test_a_reply_that_simply_finished_is_still_content():
    reply = _assistant(content="done")
    reply["choices"][0]["finish_reason"] = "stop"
    post = _Scripted([reply])
    assert agent.run_tool_conversation([], {}, post=post)["stop"] == "content"


def test_a_truncated_reply_that_still_carried_a_tool_call_is_not_ended_early():
    """Truncation only ends the turn when there is nothing to run. A reply
    that got its tool call out before the cap is ordinary work."""
    reply = _assistant(tool_calls=[_call("c1", "print(7)")])
    reply["choices"][0]["finish_reason"] = "length"
    post = _Scripted([reply, _assistant(content="seven")])
    out = agent.run_tool_conversation([], {}, post=post)
    assert out["stop"] == "content"
    assert out["tool_calls"] == 1


def test_the_model_can_act_from_inside_its_own_code():
    """The point of the whole port: a computed sequence executed in one turn,
    not one action per round trip."""
    executed = []

    def action(actions):
        executed.extend(actions)
        return {"board_changed": True, "level_completed": False}

    code = (
        "plan = ['LEFT', 'LEFT', 'UP']\n"
        "for step in plan:\n"
        "    action([step])\n"
        "result = len(plan)\n"
    )
    post = _Scripted([
        _assistant(tool_calls=[_call("c1", code)]),
        _assistant(content="done"),
    ])
    out = agent.run_tool_conversation([], {"action": action}, post=post)
    assert executed == ["LEFT", "LEFT", "UP"]
    assert "result = 3" in [m for m in _tool_messages(out, post)][0]


def _tool_messages(_out, post):
    """The rendered tool text, read back off the second request's messages."""
    return [
        m["content"]
        for m in post.payloads[1]["messages"]
        if m.get("role") == "tool"
    ]


def test_malformed_tool_arguments_go_back_to_the_model():
    """A model mistake must not end the turn -- it gets told and retries."""
    bad = {
        "id": "c1",
        "type": "function",
        "function": {"name": "python", "arguments": "{not json"},
    }
    post = _Scripted([
        _assistant(tool_calls=[bad]),
        _assistant(content="sorry"),
    ])
    messages = []
    out = agent.run_tool_conversation(messages, {}, post=post)
    assert out["stop"] == "content"
    assert "could not parse tool arguments" in messages[1]["content"]


def test_an_empty_code_argument_is_reported():
    empty = {
        "id": "c1",
        "type": "function",
        "function": {"name": "python", "arguments": json.dumps({"code": "   "})},
    }
    post = _Scripted([_assistant(tool_calls=[empty]), _assistant(content="ok")])
    messages = []
    agent.run_tool_conversation(messages, {}, post=post)
    assert "non-empty `code`" in messages[1]["content"]


def test_the_tool_call_cap_stops_a_model_that_never_decides():
    post = _Scripted([
        _assistant(tool_calls=[_call("a", "pass")]),
        _assistant(tool_calls=[_call("b", "pass")]),
    ])
    out = agent.run_tool_conversation([], {}, post=post, max_tool_calls=2)
    assert out["stop"] == "max-tool-calls"
    assert out["tool_calls"] == 2


def test_stop_check_ends_the_turn_after_an_action():
    """A level completion mid-search has to stop the model, or it keeps
    acting into a board that has already moved to the next level."""
    post = _Scripted([_assistant(tool_calls=[_call("c1", "pass")])])
    out = agent.run_tool_conversation([], {}, post=post, stop_check=lambda: True)
    assert out["stop"] == "halted"
    assert out["tool_calls"] == 1


def test_an_empty_choices_list_is_not_a_crash():
    post = _Scripted([{"choices": []}])
    out = agent.run_tool_conversation([], {}, post=post)
    assert out["stop"] == "empty-response"


def test_a_dead_server_propagates_rather_than_scoring_silently():
    """This project has already shipped one run that looked like a model run
    and was really the search. A dead server must be loud."""
    def post(_payload):
        raise OSError("connection refused")

    with pytest.raises(OSError):
        agent.run_tool_conversation([], {}, post=post)


# --- rendering and eviction ---------------------------------------------


def test_render_includes_stdout_result_and_error():
    text = agent.render_tool_result(
        {"stdout": "hello\n", "result": [1, 2], "error": "boom"}
    )
    assert "hello" in text
    assert "result = [1, 2]" in text
    assert "ERROR:\nboom" in text


def test_render_says_so_when_nothing_happened():
    """A model handed "" assumes success."""
    text = agent.render_tool_result({"stdout": "", "result": None, "error": ""})
    assert "no output" in text


def test_eviction_keeps_the_framing_and_the_recent_turns():
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "first"}]
    for i in range(10):
        messages.append({"role": "assistant", "content": f"a{i}"})
        messages.append({"role": "user", "content": f"u{i}"})
    kept = agent.evict_turns(messages, 2)
    assert kept[0]["content"] == "sys"
    assert kept[1]["content"] == "first"
    assert kept[-1]["content"] == "u9"
    assert len(kept) < len(messages)


def test_eviction_never_leaves_an_orphaned_tool_result():
    """A tool message whose assistant tool_call was evicted is a protocol
    error the server rejects."""
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "tool_calls": [{"id": "x"}]},
        {"role": "tool", "tool_call_id": "x", "content": "out"},
        {"role": "assistant", "tool_calls": [{"id": "y"}]},
        {"role": "tool", "tool_call_id": "y", "content": "out"},
    ]
    kept = agent.evict_turns(messages, 1)
    # The real invariant: every surviving tool result is preceded by the
    # assistant tool_call it answers.
    announced: set[str] = set()
    for message in kept:
        if message.get("role") == "assistant":
            announced.update(c["id"] for c in message.get("tool_calls") or [])
        if message.get("role") == "tool":
            assert message["tool_call_id"] in announced


def test_eviction_is_a_no_op_when_disabled_or_short():
    messages = [{"role": "system"}, {"role": "user"}, {"role": "assistant"}]
    assert agent.evict_turns(messages, 0) is messages
    short = messages[:2]
    assert agent.evict_turns(short, 5) is short
