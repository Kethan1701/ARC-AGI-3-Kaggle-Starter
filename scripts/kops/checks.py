"""Guards on the SPLICED notebook, never the source -- build_notebook.py
splices the agent into notebooks/submission.ipynb, and a commit that
doesn't survive the splice deploys nothing. Every push must run this
against the freshly built notebook first.

This replaced a check_vNN.py written from scratch for every kernel push
(check_v23.py, check_v24.py, ...), each one a slightly-edited copy of the
last with the version-specific finding bolted on and the general
invariants silently at risk of bit-rotting between copies. One file,
extended over time, is the point: a regression in a *previous* finding
(e.g. prefix caching quietly turned back off) gets caught here forever,
not only in the one version whose author remembered to check it that day.

Add a line to CHECKS when a run teaches you something that must hold from
here on. Don't add a line for something you merely changed this run --
that belongs in the commit message and kops/history.log.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _cells() -> list[str]:
    nb = json.loads((ROOT / "notebooks" / "submission.ipynb").read_text())
    return ["".join(c["source"]) for c in nb["cells"]]


def _first(cells: list[str], needle: str) -> str:
    for c in cells:
        if needle in c:
            return c
    raise AssertionError(f"no spliced cell contains {needle!r}")


def run() -> bool:
    cells = _cells()
    srv = _first(cells, "api_server")
    agent = _first(cells, "run_tool_conversation")

    # (description, condition) -- keep additions append-only and dated in
    # spirit even though git already dates them; the point is legibility
    # when this list is long.
    checks: list[tuple[str, bool]] = [
        # v22 (2026-08-03): the server cell is built by string interpolation
        # into a Python list literal handed to subprocess -- a bad edit there
        # is a runtime crash 5 minutes into a GPU run, not a build error.
        ("server cell parses as Python", _parses(srv)),
        ("tool parser is registered (qwen3_coder)", "'qwen3_coder'" in srv),
        ("reasoning parser is registered (qwen3)", "'qwen3'," in srv),
        # v23 (2026-08-04), reverted after measurement -- see
        # arc-agi-3-kaggle-gpu-infra memory. Both of these regressing
        # silently is exactly the failure mode this file exists to catch.
        ("prefix caching is ON", "'--enable-prefix-caching'," in srv),
        (
            "the arg after prefix-caching survived the edit",
            "'--default-chat-template-kwargs'," in srv,
        ),
        (
            "LLM_MAX_TOKENS defaults to 0 (uncapped)",
            'LLM_MAX_TOKENS = _env_int("ARC_LLM_MAX_TOKENS", 0)' in agent,
        ),
        (
            "a zero cap omits max_tokens rather than sending 0",
            "if max_tokens:" in agent and 'payload["max_tokens"] = max_tokens' in agent,
        ),
        (
            "truncated replies are named apart from finished ones",
            '"truncated" if truncated else "content"' in agent,
        ),
        # v20/v21-era findings that predate this file; kept so they can't
        # regress silently either.
        ("the repeat-detector reset is present", "LLM_REPEAT_RESET" in agent),
        (
            "rehearsal plays all 25 games concurrently (Swarm)",
            any("Swarm" in c for c in cells),
        ),
    ]

    ok = True
    for name, cond in checks:
        print(("OK   " if cond else "FAIL ") + name)
        ok = ok and cond
    return ok


def _parses(src: str) -> bool:
    try:
        ast.parse(src)
        return True
    except SyntaxError:
        return False


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
