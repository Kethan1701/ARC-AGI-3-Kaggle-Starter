#!/usr/bin/env bash
# Phase A only: build, verify, push. GPU quota, no submission slot -- only
# `kaggle competitions submit` (Phase B) spends one, and this script never
# calls it.
#
# Usage: scripts/kops/push.sh "one-line description of what this run tests"
#
# Run this from WSL. It aborts before pushing anything if:
#   - a submission is currently PENDING (Phase B in flight)
#   - the build fails
#   - the spliced notebook fails scripts/kops/checks.py
#
# On success, appends one line to kops/history.log recording what shipped
# and at which git commit -- so "what did vNN actually test" is answered
# by reading a file, not by reconstructing it from a chat transcript.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./lib.sh

DESC="${1:?usage: push.sh \"description of what this run tests\"}"

# `--format json` parsed by Python, NOT `--csv` piped through `cut -d,`.
# Submission descriptions routinely contain commas and the CSV quotes them
# rather than escaping them, so `cut -d, -f1,3,5` printed a fragment of the
# DESCRIPTION in the status column -- v47's own push showed
# "55840177,2026-08-28 09:02:46, unchanged v27-equivalent config (commit
# 1076e7a" where a status belonged. Same bug class that made
# wait_submission.sh exit on its first poll claiming a submission had
# resolved; harmless here only because the guard below never read that
# field. Fixed 2026-08-28 so the table is not quietly wrong.
#
# The guard itself also tightened: it used to `grep -q PENDING` the entire
# CSV blob, so any submission whose DESCRIPTION contained the word would
# have blocked every future push. That failed safe (refuse, not push) but
# for the wrong reason and with a baffling message. Now it reads the status
# field of each row.
echo "=== is a submission in flight? ==="
READ_SUBS='
import json, sys
try:
    rows = json.load(sys.stdin)
except Exception:
    sys.exit(0)
for r in rows[:3]:
    print("  ".join([
        str(r.get("ref", "")),
        str(r.get("date", ""))[:19],
        str(r.get("status", "")),
        str(r.get("publicScore", "")),
    ]))
'
# Three outcomes, not two. "Couldn't tell" must never be silently read as
# "clear" -- that is the same shape as the watcher bugs this project keeps
# hitting, where silence and success look identical. Prints exactly one of
# PENDING / CLEAR / UNKNOWN.
#
# `pending` is matched case-insensitively on purpose: the CLI says
# "SubmissionStatus.PENDING" while the REST API says "pending", and this
# guard should not care which one it was handed.
IS_PENDING='
import json, sys
try:
    rows = json.load(sys.stdin)
except Exception:
    print("UNKNOWN")
    sys.exit(0)
if not isinstance(rows, list):
    print("UNKNOWN")
    sys.exit(0)
print("PENDING" if any(
    "pending" in str(r.get("status", "")).lower() for r in rows
) else "CLEAR")
'
SUBS_JSON=$("$KAGGLE" competitions submissions -c "$COMPETITION" --format json 2>/dev/null || true)
printf '%s' "$SUBS_JSON" | "$PY" -c "$READ_SUBS" 2>/dev/null || echo "  (could not parse submission list)"
VERDICT=$(printf '%s' "$SUBS_JSON" | "$PY" -c "$IS_PENDING" 2>/dev/null || echo UNKNOWN)
if [ "$VERDICT" = "UNKNOWN" ]; then
    if [ "${ARC_ALLOW_PENDING_PUSH:-0}" = "1" ]; then
        echo "!!! could not read the submission list, but ARC_ALLOW_PENDING_PUSH=1 -- pushing anyway." >&2
    else
        echo "!!! could not read the submission list (CLI or network failure), so whether a Phase B run is in flight is UNKNOWN -- refusing to push. This used to read as 'nothing pending' and push. Re-run, or set ARC_ALLOW_PENDING_PUSH=1 if you know it is clear." >&2
        exit 1
    fi
fi
if [ "$VERDICT" = "PENDING" ]; then
    if [ "${ARC_ALLOW_PENDING_PUSH:-0}" = "1" ]; then
        echo "!!! a submission is PENDING, but ARC_ALLOW_PENDING_PUSH=1 -- pushing anyway (explicit per-run override, Phase A quota is independent of the Phase B slot)." >&2
    else
        echo "!!! a submission is PENDING -- refusing to push (Phase A quota is separate, but a push mid-Phase-B invites confusion about which run scored what). Set ARC_ALLOW_PENDING_PUSH=1 to override for this run." >&2
        exit 1
    fi
else
    echo "nothing pending."
fi

echo; echo "=== build ==="
"$PY" scripts/build_notebook.py
SHAPE=$("$PY" -c "import json;print(json.load(open('notebooks/kernel-metadata.json')).get('machine_shape') or '')")
echo "machine_shape=[$SHAPE]"

echo; echo "=== verify the SPLICED notebook, not the source ==="
"$PY" scripts/kops/checks.py || {
    echo "!!! the spliced notebook failed a check -- refusing to push." >&2
    exit 1
}

echo; echo "=== push ==="
if [ -n "$SHAPE" ]; then
    "$KAGGLE" kernels push -p notebooks/ --accelerator "$SHAPE" 2>&1 | tr -d '\r'
else
    "$KAGGLE" kernels push -p notebooks/ 2>&1 | tr -d '\r'
fi

SHA=$(git rev-parse --short HEAD)
STAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
printf '%s  %s  %s\n' "$STAMP" "$SHA" "$DESC" >>kops/history.log
echo; echo "logged to kops/history.log: $STAMP $SHA $DESC"
