#!/usr/bin/env bash
# Poll the kernel to a terminal state, printing one line per status change.
#
# This is meant to be handed directly to the harness's Monitor tool as its
# `command`, e.g.:
#
#   Monitor(command: "wsl bash /path/to/repo/scripts/kops/wait.sh",
#           description: "watch kernel vNN to completion")
#
# NOT `Bash --run_in_background` with a sleep loop wrapped around a second
# `wsl bash <script>` invocation -- three watchers built that way (v22,
# v23, v24) all died with exit code 4 well before the kernel finished,
# independent of whether the loop lived inside or outside the wsl call.
# Monitor is the harness's own primitive for "watch a long-running command
# and notify me per event, until it ends" and does not share that failure
# mode. See docs/kaggle_workflow.md.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./lib.sh

last=""
while true; do
    # Tolerate a transient kaggle-CLI/network failure instead of dying --
    # a bare `$(cmd)` under `set -e` takes the whole script down with it
    # on any nonzero exit. This is what killed a v33-watch instance of
    # this exact script (silently, after a single RUNNING line, exit 1)
    # even though the kernel itself was still genuinely running -- fall
    # back to empty output and retry instead of exiting.
    status=$("$KAGGLE" kernels status "$KERNEL_ID" 2>&1 | tr -d '\r' || true)
    if [ -z "$status" ]; then
        sleep 60
        continue
    fi
    if [ "$status" != "$last" ]; then
        echo "$(date -u +%H:%M:%S)  $status"
        last="$status"
    fi
    case "$status" in
    *COMPLETE*)
        echo "done."
        exit 0
        ;;
    # `*CANCEL*`, not `*CANCELLED*`: the API never emits the string
    # "CANCELLED". A killed run reports
    # `KernelWorkerStatus.CANCEL_ACKNOWLEDGED` (confirmed 2026-08-28
    # against the run killed as v46), with `CANCEL_REQUESTED` as the
    # transient before it -- neither matches `*CANCELLED*`, so this
    # branch never fired and a cancelled kernel left the watcher
    # spinning until its timeout instead of exiting 1. The status line
    # still printed on change, so the failure looked like "no news"
    # rather than a hang. Match the prefix and cover all three.
    *ERROR*|*CANCEL*)
        echo "kernel ended without completing."
        exit 1
        ;;
    esac
    sleep 60
done
