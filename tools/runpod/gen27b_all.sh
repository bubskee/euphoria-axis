#!/usr/bin/env bash
# Run the three models serially on one GPU. Ctrl-C / SIGTERM or PAUSE is resumable.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
TASK_ROOT=${GEN27B_LOCAL:-/root/gen27b}
STORE=${GEN27B_STORE:-/workspace/gen27b}
mkdir -p "$STORE"
exec 9>"$TASK_ROOT/battery.lock"
flock -n 9 || { echo 'Another battery process holds the lock.'; exit 1; }
export HF_HOME="$TASK_ROOT/hf-cache" HF_XET_CACHE="$TASK_ROOT/xet-cache" HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export PATH=/usr/local/cuda-12.8/bin:$PATH
read -r quota period < /sys/fs/cgroup/cpu.max
if [[ "$quota" =~ ^[0-9]+$ ]]; then CPUS=$((quota / period)); else CPUS=8; fi
export OMP_NUM_THREADS=$CPUS MKL_NUM_THREADS=$CPUS PYTHONUNBUFFERED=1
PY="$TASK_ROOT/.venv/bin/python"
trap 'status=$?; echo "Battery stopped at $(date -Is), exit=$status; completed units are retained."' EXIT
cd "$REPO"
test -f "$STORE/SETUP_READY"
test ! -f "$STORE/PAUSE" || exit 130
git rev-parse HEAD > "$STORE/EXECUTION_COMMIT"

# Verify frozen specs and all source materials before any model observations.
"$PY" - <<'PY'
import hashlib,json
from pathlib import Path
p=Path('doc/protocols/gen27b')
m=json.loads((p/'source-inputs.json').read_text())
for name,digest in m['sha256'].items():
    assert hashlib.sha256((Path('/root/gen27b/source')/name).read_bytes()).hexdigest()==digest,name
frozen=json.loads((p/'freeze-manifest.json').read_text())
for name,digest in frozen['specs'].items():
    assert hashlib.sha256((p/name).read_bytes()).hexdigest()==digest,name
PY

"$PY" tools/runpod/gen27b_run.py --anchor
# The 4B gate is saved on the volume; free its disposable container weight copy.
rm -rf "$TASK_ROOT/models/Qwen3.5-4B"

for slug in qwen35-27b qwen36-27b qwen38-27b; do
  test ! -f "$STORE/PAUSE" || exit 130
  if [ -f "$STORE/runs/$slug/battery.complete.json" ]; then
    echo "$slug already finished; preserving and skipping it."
    continue
  fi
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
  "$PY" tools/runpod/gen27b_run.py --slug "$slug" --phase all
  test -f "$STORE/runs/$slug/battery.complete.json"
  # No user data are removed: these are only the dedicated, verified download caches.
  rm -rf "$model_path" "$TASK_ROOT/lenses/$slug"
done
date -Is > "$STORE/ALL_CORE_BATTERIES_COMPLETE"
