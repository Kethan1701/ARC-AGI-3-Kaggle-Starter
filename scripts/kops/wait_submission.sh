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

# Pull ref/status/score off the newest submission as tab-separated
# fields. Tabs, because every other delimiter appears inside a
# description. Prints nothing at all if the payload is not parseable
# JSON, so the caller can treat "no output" as "retry".
READ_TOP='
import json, sys
try:
    rows = json.load(sys.stdin)
except Exception:
    sys.exit(0)
if not rows:
    sys.exit(0)
r = rows[0]
print("\t".join([
    str(r.get("ref", "")),
    str(r.get("status", "")),
    str(r.get("publicScore", "")),
]))
'

last=""
while true; do
    # Tolerate a transient kaggle-CLI/network failure instead of dying --
    # a bare `$(cmd)` under `set -e` takes the whole script down with it
    # on any nonzero exit, which is the likely cause of a prior watcher's
    # unexplained exit code 4. Fall back to empty output and retry instead.
    #
    # `--format json` parsed by Python, NOT `--csv` parsed by `cut -d,`.
    # Submission descriptions routinely contain commas, and the CSV
    # quotes them rather than escaping them, so `cut -d, -f5` silently
    # returned a fragment of the DESCRIPTION instead of the status. That
    # fragment is non-empty and never equals "SubmissionStatus.PENDING",
    # so the terminal check below fired on the very first poll and the
    # watcher exited announcing the submission had resolved -- a false
    # success, not a hang. Reproduced 2026-08-28 against submission
    # 55840177, whose description begins "Baseline re-run, unchanged
    # v27-equivalent config ...": field 5 came back as
    # " unchanged v27-equivalent config (commit 1076e7a".
    #
    # stderr is dropped rather than merged for the same reason wait.sh
    # drops it -- a traceback must never be parseable as a status.
    json_out=$("$KAGGLE" competitions submissions -c "$COMPETITION" --format json 2>/dev/null || true)
    info=$(printf '%s' "$json_out" | "$PY" -c "$READ_TOP" 2>/dev/null || true)
    status=$(printf '%s' "$info" | cut -f2)

    # Require the marker before trusting the field at all; anything else
    # is transient and retried without being echoed.
    case "$status" in
    SubmissionStatus.*) ;;
    *)
        sleep 120
        continue
        ;;
    esac

    if [ "$status" != "$last" ]; then
        printf '%s  ref=%s  status=%s  score=%s\n' \
            "$(date -u +%H:%M:%S)" \
            "$(printf '%s' "$info" | cut -f1)" \
            "$status" \
            "$(printf '%s' "$info" | cut -f3)"
        last="$status"
    fi
    if [ "$status" != "SubmissionStatus.PENDING" ]; then
        echo "no longer pending."
        exit 0
    fi
    sleep 120
done
