"""One report for a pulled kernel run: official scorecard, server
throughput, and per-turn stop-reason/truncation stats -- optionally diffed
against a baseline run.

This replaced roughly fifteen one-off scripts (scorecard_v22.py, sc23.py,
sc23b.py, srv23.py, srv23b.py, turns23.py, tr23b.py through tr23e.py, ...),
each a copy of the last with that day's bug fixed and the version numbers
hardcoded. The bugs are worth keeping the memory of:

  * The harness log is a JSON array of {stream_name, time, data} events;
    the scorecard JSON is embedded in the concatenated `data` text after a
    "--- FINAL SCORECARD REPORT ---" marker, not a top-level key. Decode
    it with json.JSONDecoder().raw_decode from the following `{`.
  * Every log line appears 3x (25 interleaved games, non-adjacent
    repeats). Dedup on the full line text before counting anything.
  * The game id in a turn line is `gameid-hexsuffix t<N>: ...`, e.g.
    `r11l-495a7899 t1: reply hit ...` -- match `([a-z0-9]{4})-\\w+ t(\\d+)`,
    not a bare `\\S*` before the label, or every match silently misses.
  * "Prefix cache hit rate" samples read exactly 0.0% most of the time
    between requests; counting only those and calling it "the hit rate"
    understates real reuse by a lot. Take mean/percentile over ALL
    samples, never just the zero ones.

Usage:
    python scripts/kops/analyze.py LABEL [BASELINE_LABEL]

Reads kaggle_runs/LABEL/ (see pull.sh). See docs/kaggle_workflow.md for
what to do with the numbers -- the short version: judge a config on
clears and throughput, never on the single scorecard score. A run that
clears fewer levels with fewer total actions is not an improvement no
matter what the score says (v23 scored higher than v22 on fewer clears --
see arc-agi-3-kaggle-gpu-infra memory).
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "kaggle_runs"
GAME_LOG = "arc-prize-2026-arc-agi-3-starter.log"
SERVER_LOG = "vllm-server.log"


def _lines(path: Path) -> list[str]:
    events = json.loads(path.read_text())
    return [
        (e.get("data", "") if isinstance(e, dict) else str(e)).rstrip("\n")
        for e in events
    ]


def scorecard(label: str) -> dict | None:
    path = RUNS_DIR / label / GAME_LOG
    if not path.exists():
        return None
    blob = "\n".join(_lines(path))
    i = blob.find("--- FINAL SCORECARD REPORT ---")
    if i < 0:
        return None
    j = blob.find("{", i)
    card, _ = json.JSONDecoder().raw_decode(blob[j:])
    return card


def print_scorecard(label: str, card: dict) -> None:
    envs = card["environments"]
    clears1 = sum(1 for e in envs if e["levels_completed"] >= 1)
    clears2 = sum(1 for e in envs if e["levels_completed"] >= 2)
    print(f"=== {label}: OFFICIAL SCORE {card['score']:.4f} ===")
    print(f"environments: {len(envs)}  clears>=1: {clears1}  clears>=2: {clears2}")
    print(
        f"total actions: {sum(e['actions'] for e in envs)}  "
        f"total resets: {sum(e['resets'] for e in envs)}"
    )
    print(f"{'game':6s} {'score':>7s} {'lvls':>5s} {'acts':>5s} {'resets':>6s}")
    for e in sorted(envs, key=lambda e: -e["score"]):
        print(
            f"{e['id'][:4]:6s} {e['score']:7.3f} "
            f"{e['levels_completed']:2d}/{e['level_count']:<2d} "
            f"{e['actions']:5d} {e['resets']:6d}"
        )


def diff_scorecards(label_a: str, card_a: dict, label_b: str, card_b: dict) -> None:
    ea = {e["id"]: e for e in card_a["environments"]}
    eb = {e["id"]: e for e in card_b["environments"]}
    print(f"\n=== {label_a} vs {label_b} ===")
    print(f"score: {card_a['score']:.4f} -> {card_b['score']:.4f}")
    cl = lambda c: sum(1 for e in c["environments"] if e["levels_completed"] >= 1)
    print(f"clears>=1: {cl(card_a)}/{len(ea)} -> {cl(card_b)}/{len(eb)}")
    print(
        f"total actions: {sum(e['actions'] for e in ea.values())} -> "
        f"{sum(e['actions'] for e in eb.values())}"
    )
    print(f"{'game':6s} {label_a:>8s} {label_b:>8s} {'diff':>8s}")
    for gid in sorted(eb, key=lambda g: -(eb[g]["score"] - ea.get(g, {"score": 0})["score"])):
        a = ea.get(gid, {"score": 0.0})
        b = eb[gid]
        print(f"{gid[:4]:6s} {a['score']:8.3f} {b['score']:8.3f} {b['score']-a['score']:+8.3f}")


def throughput(label: str) -> None:
    path = RUNS_DIR / label / SERVER_LOG
    if not path.exists():
        print(f"\n=== {label}: no {SERVER_LOG} ===")
        return
    run, wait, gen, hit = [], [], [], []
    for ln in path.read_text(errors="replace").splitlines():
        if m := re.search(r"Avg generation throughput:\s*([\d.]+)", ln):
            gen.append(float(m.group(1)))
        if m := re.search(r"Running:\s*(\d+)\s*reqs?,\s*Waiting:\s*(\d+)", ln):
            run.append(int(m.group(1)))
            wait.append(int(m.group(2)))
        if m := re.search(r"[Pp]refix cache hit rate:\s*([\d.]+)%", ln):
            hit.append(float(m.group(1)))

    print(f"\n=== {label}: server throughput ===")

    def summarize(name: str, xs: list[float]) -> None:
        if not xs:
            print(f"  {name}: none")
            return
        xs = sorted(xs)
        print(
            f"  {name}: n={len(xs)} median={statistics.median(xs):.1f} "
            f"p75={xs[int(.75*len(xs))]:.1f} mean={statistics.mean(xs):.1f}"
        )

    summarize("Running", [float(x) for x in run])
    summarize("gen tok/s", gen)
    summarize("prefix hit %", hit)
    steady = [g for g, r in zip(gen, run) if r >= 20]
    if steady:
        print(
            f"  gen tok/s @Running>=20 (steady state): "
            f"n={len(steady)} median={statistics.median(steady):.1f} "
            f"mean={statistics.mean(steady):.1f}"
        )
    print(f"  total generated (10s samples): {sum(gen)*10:,.0f} tokens")


TURN_RE = re.compile(r"(\d+) actions in [\d.]+s \(([a-z-]+), (\d+) tool calls")
GAME_TURN_RE = re.compile(r"[a-z0-9]{4}-\w+ t\d+: reply hit")


def turns(label: str) -> None:
    path = RUNS_DIR / label / GAME_LOG
    if not path.exists():
        return
    lines = set(_lines(path))  # dedup: every line appears 3x in the raw log
    stop_reasons: dict[str, int] = {}
    tool_calls: list[int] = []
    actions: list[int] = []
    truncated_turns = 0
    for ln in lines:
        if m := TURN_RE.search(ln):
            actions.append(int(m.group(1)))
            stop_reasons[m.group(2)] = stop_reasons.get(m.group(2), 0) + 1
            tool_calls.append(int(m.group(3)))
        if GAME_TURN_RE.search(ln):
            truncated_turns += 1

    print(f"\n=== {label}: turns ===")
    print(f"  turns counted: {len(actions)}")
    if stop_reasons:
        print(f"  stop reasons: {stop_reasons}")
    if tool_calls:
        print(
            f"  tool calls/turn: median={statistics.median(tool_calls):.1f} "
            f"mean={statistics.mean(tool_calls):.1f}"
        )
    if actions:
        print(
            f"  actions/turn: median={statistics.median(actions):.1f} "
            f"mean={statistics.mean(actions):.1f} total={sum(actions)}"
        )
    if truncated_turns:
        print(f"  replies that hit a truncation cap: {truncated_turns}")


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    label = argv[0]
    card = scorecard(label)
    if card is None:
        print(f"{label}: no FINAL SCORECARD REPORT found in {GAME_LOG}")
    else:
        print_scorecard(label, card)
    throughput(label)
    turns(label)

    if len(argv) > 1:
        baseline = argv[1]
        base_card = scorecard(baseline)
        if card is not None and base_card is not None:
            diff_scorecards(baseline, base_card, label, card)
        throughput(baseline)
        turns(baseline)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
