#!/usr/bin/env bash
# Additive storage recovery; the already-running 3.6 worker is never stopped.
set -euo pipefail
cd /root/gen27b/repo
PY=/root/gen27b/.venv/bin/python
STORE=/workspace/gen27b
exec 8>/root/gen27b/olens36.lock
flock -n 8 || { echo 'Another O-lens/recovery controller is active'; exit 1; }
exec 7>/root/gen27b/storage-recovery.lock
flock -n 7 || { echo 'Another storage recovery controller is active'; exit 1; }
exec 9>/root/gen27b/battery.lock
export PYTHONUNBUFFERED=1 PATH=/usr/local/cuda-12.8/bin:$PATH
export HF_HOME=/root/gen27b/hf-cache HF_XET_CACHE=/root/gen27b/xet-cache HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export GEN27B_RECOVERY_LOCK_FD=9
pause_guard() {
  local code=0
  "$PY" tools/runpod/lens_io_resume.py --pause-status || code=$?
  if [ "$code" = 0 ]; then exit 130; fi
  if [ "$code" != 1 ]; then exit "$code"; fi
}
trap 'code=$?; echo "Recovery/O-lens controller stopped $(date -Is), exit=$code; originals retained."' EXIT
echo 'WAITING: live 3.6 worker keeps running; recovery waits for its original controller and GPU lock.'
while true; do
  pause_guard
  if test -f "$STORE/ALL_APPENDICES_ATTEMPTED" && flock -n 9; then break; fi
  sleep 30
done
pause_guard
echo "GPU available for storage recovery at $(date -Is)"
scope_args=()
if [ -n "${GEN27B_QWEN36_APPROVED_SCOPE:-}" ]; then
  scope_args=(--approved-scope "$GEN27B_QWEN36_APPROVED_SCOPE")
fi
"$PY" tools/runpod/lens_repair.py repair --slug qwen36-27b --allow-replay "${scope_args[@]}"
pause_guard
# Keep the 3.6 weights for the already-authorized O-lens panel.
if ! test -f "$STORE/olens36-appendix/COMPLETE.json"; then
  "$PY" tools/runpod/test_olens36_appendix.py
  "$PY" tools/runpod/gen27b_fetch.py Qwen/Qwen3.6-27B 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9 /root/gen27b/models/Qwen3.6-27B
  "$PY" tools/runpod/olens_storage_guard.py
fi
"$PY" tools/runpod/olens_storage_guard.py --verify-only
pause_guard
# Original bytes are preferred; only still-missing original prompts are replayed.
"$PY" tools/runpod/lens_repair.py repair --slug qwen38-27b --allow-replay
PYTHONPATH=tools/runpod "$PY" - <<'PY'
from pathlib import Path
from lens_storage_guard import publish
from datetime import datetime, timezone
publish(Path('/workspace/gen27b/storage-recovery/ALL_SELECTED_OUTPUTS_VERIFIED'),
        (datetime.now(timezone.utc).isoformat()+'\n').encode())
PY
