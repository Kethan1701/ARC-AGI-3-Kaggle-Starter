#!/usr/bin/env bash
# One-shot: print the kernel's current status and exit. Companion to
# wait.sh (which loops); use this for a direct read after a watcher dies,
# to see current truth without relaunching a long-running process first.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./lib.sh

"$KAGGLE" kernels status "$KERNEL_ID" 2>&1 | tr -d '\r'
