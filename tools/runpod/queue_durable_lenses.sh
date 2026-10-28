#!/usr/bin/env bash
# Storage amendment: unchanged frozen numerical runners; durable lens journal chunks.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
TASK_ROOT=${GEN27B_LOCAL:-/root/gen27b}
STORE=${GEN27B_STORE:-/workspace/gen27b}
PY="$TASK_ROOT/.venv/bin/python"
mkdir -p "$STORE"
exec 9>"$TASK_ROOT/battery.lock"
flock -n 9 || { echo 'Another battery controller holds the lock.'; exit 1; }
export HF_HOME="$TASK_ROOT/hf-cache" HF_XET_CACHE="$TASK_ROOT/xet-cache" HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export PATH=/usr/local/cuda-12.8/bin:$PATH
read -r quota period < /sys/fs/cgroup/cpu.max
if [[ "$quota" =~ ^[0-9]+$ ]]; then CPUS=$((quota / period)); else CPUS=8; fi
export OMP_NUM_THREADS=$CPUS MKL_NUM_THREADS=$CPUS PYTHONUNBUFFERED=1
trap 'status=$?; echo "Core-first queue stopped at $(date -Is), exit=$status; saved units retained."' EXIT
cd "$REPO"
test -f "$STORE/SETUP_READY"
test ! -f "$STORE/PAUSE" || exit 130
git rev-parse HEAD > "$STORE/STORAGE_RESUME_CHECKOUT_COMMIT"
sha256sum tools/runpod/queue_durable_lenses.sh > "$STORE/STORAGE_CONTROLLER_SHA256"

ensure_model() {
  local slug=$1 model revision model_path
  test ! -f "$STORE/PAUSE" || exit 130
  read -r model revision < <("$PY" - "$slug" <<'PY'
import json,sys
from pathlib import Path
s=json.loads((Path('doc/protocols/gen27b')/(sys.argv[1]+'.json')).read_text())
print(s['model_id'],s['model_revision'])
PY
)
  model_path="$TASK_ROOT/models/${model##*/}"
  if [ ! -f "$model_path/VERIFIED.json" ]; then
    "$PY" tools/runpod/gen27b_fetch.py "$model" "$revision" "$model_path"
  fi
}

core_done() {
  PYTHONPATH=tools/runpod "$PY" - "$1" <<'PY'
import sys
from gen27b_common import STORE,read_spec,is_complete
s,h=read_spec(sys.argv[1]);d=STORE/'runs'/s['slug']
sys.exit(0 if all(is_complete(d,m,h) for m in ['m1','m2','m3','m4','m5','m6','m7']) else 1)
PY
}

mark_done() {
  PYTHONPATH=tools/runpod "$PY" - "$1" "$2" <<'PY'
import sys
from gen27b_common import STORE,read_spec,is_complete,complete
s,h=read_spec(sys.argv[1]);d=STORE/'runs'/s['slug']
assert all(is_complete(d,m,h) for m in ['m1','m2','m3','m4','m5','m6','m7'])
paths=[d/(m+'.complete.json') for m in ['m1','m2','m3','m4','m5','m6','m7']]
paths += [d/'core_scorecard.json',d/'m7_scorecard.json',d/'run_header.json']
for name in ['lenses.complete.json','appendix_scorecard.json','appendix_deferred.json']:
    if (d/name).exists(): paths.append(d/name)
complete(d,sys.argv[2],paths,h)
PY
}

"$PY" tools/runpod/gen27b_run.py --anchor
for slug in qwen35-27b qwen36-27b qwen38-27b; do
  if ! core_done "$slug"; then
    ensure_model "$slug"
    "$PY" tools/runpod/gen27b_run.py --slug "$slug" --phase core
    "$PY" tools/runpod/gen27b_run.py --slug "$slug" --phase buttons
  fi
  mark_done "$slug" core_battery
  if [ "$slug" != qwen38-27b ]; then
    case "$slug" in
      qwen35-27b) rm -rf "$TASK_ROOT/models/Qwen3.5-27B" ;;
      qwen36-27b) rm -rf "$TASK_ROOT/models/Qwen3.6-27B" ;;
    esac
  fi
done
test -f "$STORE/ALL_CORE_BATTERIES_COMPLETE" || date -Is > "$STORE/ALL_CORE_BATTERIES_COMPLETE"

# Reuse the last resident model first; all primary data are already complete.
for slug in qwen38-27b qwen35-27b qwen36-27b; do
  test ! -f "$STORE/PAUSE" || exit 130
  ensure_model "$slug"
  "$PY" tools/runpod/lens_io_resume.py --slug "$slug" --phase lenses
  mark_done "$slug" battery
  case "$slug" in
    qwen35-27b) rm -rf "$TASK_ROOT/models/Qwen3.5-27B" ;;
    qwen36-27b) rm -rf "$TASK_ROOT/models/Qwen3.6-27B" ;;
    qwen38-27b) rm -rf "$TASK_ROOT/models/Qwen3.8-27B" ;;
  esac
  rm -rf "$TASK_ROOT/lenses/$slug"
done
date -Is > "$STORE/ALL_APPENDICES_ATTEMPTED"
