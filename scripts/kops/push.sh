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

echo "=== is a submission in flight? ==="
SUBS=$("$KAGGLE" competitions submissions -c "$COMPETITION" --csv 2>&1 | tr -d '\r')
echo "$SUBS" | cut -d, -f1,3,5 | head -4
if echo "$SUBS" | grep -q PENDING; then
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
