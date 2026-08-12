#!/usr/bin/env bash
# Poll the most recent competition submission to a non-PENDING state,
# printing one line per status change. Companion to wait.sh (which
# watches a KERNEL to completion) -- this watches a SUBMISSION (Phase B),
# which push.sh refuses to push alongside: "a push mid-Phase-B invites
# confusion about which run scored what."
#
# Meant to be handed directly to the harness's Monitor tool as its
# `command`, same as wait.sh:
#
#   Monitor(command: "wsl bash /path/to/repo/scripts/kops/wait_submission.sh",
#           description: "watch the pending submission clear")
#
# NOT a Bash --run_in_background sleep loop -- see wait.sh's own comment
# for why three watchers built that way died silently at exit code 4.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./lib.sh

last=""
while true; do
    # Tolerate a transient kaggle-CLI/network failure instead of dying --
    # a bare `$(cmd)` under `set -e` takes the whole script down with it
    # on any nonzero exit, which is the likely cause of a prior watcher's
    # unexplained exit code 4. Fall back to empty output and retry instead.
    row=$("$KAGGLE" competitions submissions -c "$COMPETITION" --csv 2>&1 | tr -d '\r' | sed -n '2p' || true)
    status=$(echo "$row" | cut -d, -f5 || true)
    if [ "$status" != "$last" ]; then
        echo "$(date -u +%H:%M:%S)  $row"
        last="$status"
    fi
    if [ -n "$status" ] && [ "$status" != "SubmissionStatus.PENDING" ]; then
        echo "no longer pending."
        exit 0
    fi
    sleep 120
done
