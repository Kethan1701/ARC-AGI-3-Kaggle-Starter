#!/usr/bin/env bash
# Pull a finished (or in-progress) kernel's output. Read-only against
# Kaggle -- safe to run any time, doesn't touch quota or submissions.
#
# Usage: scripts/kops/pull.sh <label>
# Writes to kaggle_runs/<label>/ (gitignored -- these are run artefacts,
# not source, but they're kept on disk across sessions instead of living
# in a job's scratch tmp dir that gets swept away with the job).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./lib.sh

LABEL="${1:?usage: pull.sh <label>, e.g. pull.sh v24}"
OUT="$RUNS_DIR/$LABEL"
mkdir -p "$OUT"
"$KAGGLE" kernels output "$KERNEL_ID" -p "$OUT" 2>&1 | tr -d '\r' | tail -5
echo
echo "=== files ==="
ls -la "$OUT"
