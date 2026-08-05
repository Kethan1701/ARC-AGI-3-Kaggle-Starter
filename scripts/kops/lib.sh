#!/usr/bin/env bash
# Shared setup for every kops script. Source it, don't run it.
#
#   source "$(dirname "$0")/lib.sh"
#
# Every script in this directory is meant to run under WSL (kaggle CLI,
# pytest and build_notebook.py all need arcengine, which only installs
# there) with one exception: `git push` itself, which must run from
# Windows PowerShell -- WSL's git hangs silently against the Windows
# Credential Manager. Nothing in this directory pushes to git; that step
# stays a manual PowerShell command by design. See docs/kaggle_workflow.md.
set -euo pipefail

KOPS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$KOPS_ROOT"

KERNEL_ID="kethansaikoneru/arc-prize-2026-arc-agi-3-starter"
COMPETITION="arc-prize-2026-arc-agi-3"
KAGGLE="$KOPS_ROOT/.venv/bin/kaggle"
PY="$KOPS_ROOT/.venv/bin/python"
RUNS_DIR="$KOPS_ROOT/kaggle_runs"

if [ ! -s .kaggle/access_token ]; then
    echo "no .kaggle/access_token -- can't talk to Kaggle" >&2
    exit 1
fi
export KAGGLE_API_TOKEN
KAGGLE_API_TOKEN=$(cat .kaggle/access_token)

# Never let the token itself land in a trace.
set +x
