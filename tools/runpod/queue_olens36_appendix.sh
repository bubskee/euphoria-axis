#!/usr/bin/env bash
# Independent addendum; wait for the existing controller and share its GPU lock.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
TASK_ROOT=${GEN27B_LOCAL:-/root/gen27b}
STORE=${GEN27B_STORE:-/workspace/gen27b}
PY="$TASK_ROOT/.venv/bin/python"
OUT="$STORE/olens36-appendix"
mkdir -p "$OUT"
exec 8>"$TASK_ROOT/olens36.lock"
flock -n 8 || { echo 'O-lens addendum controller already active'; exit 1; }
exec 9>"$TASK_ROOT/battery.lock"
trap 'status=$?; echo "O-lens queue stopped at $(date -Is), exit=$status; saved units retained."' EXIT
export PYTHONUNBUFFERED=1
export PATH=/usr/local/cuda-12.8/bin:$PATH
export HF_HOME="$TASK_ROOT/hf-cache" HF_XET_CACHE="$TASK_ROOT/xet-cache" HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "$REPO"
test ! -f "$OUT/COMPLETE.json" || exit 0
echo 'WAITING: all selected J/R/Miru appendices and the existing GPU lock must finish first.'
while true; do
  test ! -f "$STORE/PAUSE" || exit 130
  if test -f "$STORE/ALL_APPENDICES_ATTEMPTED" && flock -n 9; then break; fi
  sleep 30
done
echo "O-lens addendum acquired existing GPU lock at $(date -Is)"
test ! -f "$STORE/PAUSE" || exit 130
"$PY" tools/runpod/test_olens36_appendix.py
"$PY" tools/runpod/gen27b_fetch.py Qwen/Qwen3.6-27B 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9 "$TASK_ROOT/models/Qwen3.6-27B"
"$PY" tools/runpod/olens36_appendix.py
