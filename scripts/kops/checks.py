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


def _first_opt(cells: list[str], needle: str) -> str | None:
    """Like _first, but absence is a fact to branch on rather than an error."""
    for c in cells:
        if needle in c:
            return c
    return None


def run() -> bool:
    cells = _cells()

    # Which notebook is this? build_notebook.py emits the vLLM server cell
    # only for MODE="llm", so its presence IS the mode, read off the built
    # artefact rather than trusted from the source constant -- same
    # principle as the rest of this file, which checks the SPLICED output
    # because a commit deploys nothing.
    #
    # 2026-08-28: before this branch existed, every check below assumed the
    # LLM notebook, so the first search-mode build in a month died here with
    # `AssertionError: no spliced cell contains 'api_server'` -- the guard
    # failing shut on a perfectly good notebook. A guard that cannot express
    # "this invariant does not apply to this build" gets bypassed, and a
    # bypassed guard protects nothing.
    srv = _first_opt(cells, "api_server")
    llm_mode = srv is not None
    agent = _first(cells, "run_tool_conversation")

    print(f"mode: {'llm' if llm_mode else 'search/policy'} "
          f"({'server cell present' if llm_mode else 'no server cell'})")

    # (description, condition) -- keep additions append-only and dated in
    # spirit even though git already dates them; the point is legibility
    # when this list is long.
    # Invariants that hold in EVERY mode. The agent source is spliced whole
    # regardless of mode, so the agent-side LLM checks below stay armed even
    # in search mode -- the code is dormant, not absent, and it must not rot
    # while the LLM era is parked.
    checks: list[tuple[str, bool]] = [
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
        # 2026-08-06: eviction used to recompute its cut point fresh every
        # turn, so the surviving prefix's first byte moved every turn and
        # no two consecutive requests ever shared a prefix -- the server's
        # prefix cache read ~0% median in both v22 and v24 despite being
        # on. Losing the hysteresis silently reintroduces that. See
        # arc-agi-3-kaggle-gpu-infra memory.
        ("eviction holds its cut point with hysteresis", "LLM_EVICT_SLACK" in agent),
        # 2026-08-06: a wave-based, reduced-concurrency Swarm loop was
        # built and reverted the same day -- competition mode allows
        # exactly one scorecard per session, ever (arc_agi.api's
        # ScorecardManager never resets the flag that blocks a second
        # open), confirmed by reading the source and reproducing the 409
        # locally. The real run cell must keep calling `main.py` directly;
        # a regression back to a wave_runner.py invocation without first
        # solving the single-shared-scorecard problem would silently score
        # zero on every game after the first. See arc-agi-3-kaggle-gpu-infra
        # memory before attempting this again.
        (
            "the real run cell execs main.py directly (no wave_runner.py)",
            "exec python main.py --agent myagent"
            in _first(cells, "KAGGLE_IS_COMPETITION_RERUN"),
        ),
    ]

    if llm_mode:
        assert srv is not None  # narrowed by llm_mode; for the type checker
        checks += [
            # v22 (2026-08-03): the server cell is built by string
            # interpolation into a Python list literal handed to subprocess
            # -- a bad edit there is a runtime crash 5 minutes into a GPU
            # run, not a build error.
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
            # v27 (2026-08-06): vLLM's own startup log said its CUDA-graph
            # pool reservation was conservative by ~0.12GiB and named this
            # exact pair of settings to recover it into the KV cache. Losing
            # either half silently forfeits free KV-cache headroom on a card
            # already spending 73% of samples above 80% KV usage (log
            # analysis, see arc-agi-3-kaggle-gpu-infra memory).
            (
                "CUDA-graph memory profiler estimate is on",
                "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS" in srv
                or "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS" in agent,
            ),
            ("gpu-memory-utilization raised to 0.93 alongside it", "'0.93'," in srv),
            # v30 (2026-08-07): --kv-cache-dtype fp8 was tried and reverted
            # -- it crashes vLLM's engine init in this offline container
            # (fp8 KV cache needs a flashinfer kernel that isn't prebuilt,
            # and the runtime Ninja/link step to JIT it fails), silently
            # downgrading the whole run to the search-only fallback for all
            # 25 games. A regression back to this flag without first fixing
            # that would waste a full GPU run again.
            (
                "no un-reverted kv-cache-dtype fp8 flag",
                "'--kv-cache-dtype', 'fp8'" not in srv,
            ),
        ]
    else:
        # 2026-08-28, the return to search-only. These are the mirror image
        # of the LLM checks: there, the risk is the server silently losing a
        # flag; here, it is the notebook silently still being an LLM run.
        #
        # This matters because the agent falls back to search on its own
        # whenever the model is unreachable. That fallback is a good runtime
        # property and a terrible build property: a half-converted notebook
        # would still play all 25 games and still produce a scorecard, so
        # the run would look fine while burning GPU quota to score what the
        # CPU scores. Nothing downstream would flag it.
        joined = "\n".join(cells)
        checks += [
            (
                "no vLLM server cell (search mode must not stand one up)",
                srv is None,
            ),
            (
                "nothing exports ARC_LLM (the model is off, not merely absent)",
                "ARC_LLM=1" not in joined and "export ARC_LLM" not in joined,
            ),
            (
                "nothing exports ARC_POLICY (weights are not mounted in this mode)",
                "ARC_POLICY=1 " not in joined,
            ),
            (
                "the agent still defaults ARC_LLM to off",
                'LLM_ENABLED = _env_int("ARC_LLM", 0)' in agent,
            ),
            (
                "the agent still defaults ARC_POLICY to off",
                'POLICY_ENABLED = _env_int("ARC_POLICY", 0)' in agent,
            ),
            # The three graded search-only submissions (0.19/0.19/0.26) all
            # ran with these defaults; memory attributes the 0.26 to exactly
            # this trio. They are cheap to assert and expensive to lose.
            (
                "VOLATILE_WARMUP is 25, as in the 0.26 run",
                'VOLATILE_WARMUP = _env_int("ARC_VOLATILE_WARMUP", 25)' in agent,
            ),
            (
                "momentum is off, as in the 0.26 run",
                'MOMENTUM_LIMIT = _env_int("ARC_MOMENTUM", 0)' in agent,
            ),
            (
                "the tie-break RNG is crc32-seeded, so runs reproduce",
                "zlib.crc32(self.game_id.encode())" in agent,
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
