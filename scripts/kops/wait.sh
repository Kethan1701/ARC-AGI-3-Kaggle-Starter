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
    status=$("$KAGGLE" kernels status "$KERNEL_ID" 2>&1 | tr -d '\r')
    if [ "$status" != "$last" ]; then
        echo "$(date -u +%H:%M:%S)  $status"
        last="$status"
    fi
    case "$status" in
    *COMPLETE*)
        echo "done."
        exit 0
        ;;
    *ERROR*|*CANCELLED*)
        echo "kernel ended without completing."
        exit 1
        ;;
    esac
    sleep 60
done
