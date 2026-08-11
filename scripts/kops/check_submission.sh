#!/usr/bin/env bash
# One-shot: print the most recent submission's status line and exit.
# Companion to wait_submission.sh (which loops); this is for a single
# direct read after a Monitor watcher dies, to see current truth without
# relaunching a long-running process first.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./lib.sh

"$KAGGLE" competitions submissions -c "$COMPETITION" --csv 2>&1 | tr -d '\r' | sed -n '1,3p'
