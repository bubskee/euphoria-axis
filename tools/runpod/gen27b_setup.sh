#!/usr/bin/env bash
# Environment only. Does not run any model or experiment.
set -euo pipefail
TASK_ROOT=${GEN27B_ROOT:-/root/gen27b}
STORE=${GEN27B_STORE:-/workspace/gen27b}
mkdir -p "$TASK_ROOT" "$STORE"
exec 9>"$TASK_ROOT/setup.lock"
flock -n 9 || exit 1
trap 'echo "setup failed at line $LINENO" > "$STORE/SETUP_FAILED"' ERR
rm -f "$STORE/SETUP_FAILED"
export UV_CACHE_DIR="$TASK_ROOT/uv-cache"
export HF_HOME="$TASK_ROOT/hf-cache"
export HF_XET_CACHE="$TASK_ROOT/xet-cache"
export PATH=/usr/local/cuda-12.8/bin:$PATH
read -r quota period < /sys/fs/cgroup/cpu.max
if [[ "$quota" =~ ^[0-9]+$ ]]; then CPUS=$((quota / period)); else CPUS=8; fi
export OMP_NUM_THREADS=$CPUS MKL_NUM_THREADS=$CPUS
uv venv --python /usr/local/bin/python "$TASK_ROOT/.venv"
source "$TASK_ROOT/.venv/bin/activate"
uv pip install 'torch==2.11.0' --index-url https://download.pytorch.org/whl/cu128
sed -E '/^(torch|causal-conv1d|causal_conv1d)([ =@]|$)/d; /^-e /d; / @ file:/d' \
  /workspace/env-fast.lock.txt > "$TASK_ROOT/lock-rest.txt"
uv pip install -r "$TASK_ROOT/lock-rest.txt"
uv pip install /workspace/causal_conv1d-1.7.0-cp312-cp312-linux_x86_64.whl
uv pip install scikit-learn pytest
git clone -q https://github.com/anthropics/jacobian-lens.git "$TASK_ROOT/jacobian-lens"
git -C "$TASK_ROOT/jacobian-lens" checkout -q 581d398613e5602a5af361e1c34d3a92ea82ba8e
uv pip freeze > "$STORE/environment.lock.txt"
python - <<'PY'
import torch, transformers, fla, causal_conv1d, sklearn
assert torch.__version__ == '2.11.0+cu128'
assert transformers.__version__ == '5.19.0'
print('environment ready:', torch.__version__, transformers.__version__, torch.cuda.get_device_name(0))
PY
date -Is > "$STORE/SETUP_READY"
