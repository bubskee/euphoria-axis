#!/bin/bash
# Gate 2a environment, on the container disk (run after the main setup with MODEL_ID=none):
#   /root/gate2a/src        the source paper's repository at the pinned commit
#   /root/gate2a/.venv-tl   TransformerLens environment (versions recorded; the source pins none)
#   /root/hf_cache          Qwen/Qwen3-14B (the ladder script also uses this HF_HOME), every file checked against the Hub
#   /root/gate2a/run_header.json   source commit, model revision, package versions, GPU, spec hash
set -euo pipefail
SRC_COMMIT=8d1649c03a63a39c9aa092532c376800cc4a3863
G=/root/gate2a; mkdir -p $G/out
export HF_HOME=/root/hf_cache HF_XET_CACHE=/root/gateL/xet-cache UV_CACHE_DIR=/root/gateL/uv-cache
retry() { local _n=$1; shift; local _d=5 _i; for ((_i = 1; _i <= _n; _i++)); do "$@" && return 0
  [ "$_i" -lt "$_n" ] && { echo "retry $_i/$_n: $*; waiting ${_d}s" >&2; sleep "$_d"; _d=$((_d * 2)); }; done; return 1; }

# 1. source at the pinned commit
[ -d $G/src/.git ] || retry 4 git clone -q https://github.com/valen-research/Pain-axis.git $G/src
git -C $G/src checkout -q $SRC_COMMIT && echo "source at $(git -C $G/src rev-parse HEAD)"

# 2. TransformerLens environment
command -v uv >/dev/null || pip install -q uv
[ -d $G/.venv-tl ] || uv venv -q $G/.venv-tl
source $G/.venv-tl/bin/activate
# transformer_lens 2.18.0 (newest 2.x): the source script uses the 2.x API (HookedTransformer.from_pretrained_no_processing),
# which 3.x reorganised; 2.18.0 lists Qwen/Qwen3-14B. Its metadata asks transformers >= 4.57 with no upper bound, but 2.x
# was built against transformers 4.x (TransformerLens moved to 5.x only in 3.3), so transformers is capped below 5.
# The source pins nothing; every choice here is recorded in run_header.json.
# pyarrow < 18: transformer_lens 2.x needs numpy < 2, and newer pyarrow refuses numpy 1 at import time
# (a runtime check, invisible to the resolver). datasets is only imported by transformer_lens here.
# torch 2.11.0 built for CUDA 12.8, from PyTorch's index (as in the main environment): PyPI's default torch is now
# built for CUDA 13, which needs a newer driver than many pods have ("driver too old", no GPU visible)
retry 4 uv pip install -q "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128
retry 4 uv pip install -q "transformer_lens==2.18.0" "transformers>=4.57,<5" "pyarrow<18" huggingface_hub hf_transfer safetensors \
    numpy "pandas<3" scipy scikit-learn matplotlib tqdm             # pandas 3 changed defaults the source scripts were not written for
python - <<'PY'
import importlib.metadata as md
import torch, transformers, numpy, pyarrow, datasets, sklearn, matplotlib, pandas, scipy, tqdm
from transformer_lens import HookedTransformer
from transformer_lens.loading_from_pretrained import OFFICIAL_MODEL_NAMES
print("transformer_lens", md.version("transformer_lens"), "| transformers", transformers.__version__, "| torch", torch.__version__,
      "| numpy", numpy.__version__, "| pyarrow", pyarrow.__version__, "| datasets", datasets.__version__)
assert hasattr(HookedTransformer, "from_pretrained_no_processing"), "missing the API the source script calls"
assert torch.cuda.is_available(), f"torch {torch.__version__} cannot see the GPU (driver / CUDA build mismatch?)"
assert torch.version.cuda and torch.version.cuda.startswith("12."), f"expected a CUDA 12 build of torch, got {torch.version.cuda}"
print("GPU visible to torch:", torch.cuda.get_device_name(0), "| torch CUDA build", torch.version.cuda)
assert "Qwen/Qwen3-14B" in OFFICIAL_MODEL_NAMES, "this transformer_lens does not support Qwen/Qwen3-14B"
print("transformer_lens: source API present, Qwen/Qwen3-14B supported")
PY

# 3. model: download main, record its revision, check every file against the Hub
# resources: the source loads the model into CPU memory, then TransformerLens converts the weights (briefly up to ~2x 28 GB)
# containers see the host's memory in /proc/meminfo; the real limit is the cgroup's (v2 memory.max, v1 limit_in_bytes)
mem_limit_gb() {
  local b=""
  if [ -r /sys/fs/cgroup/memory.max ]; then b=$(cat /sys/fs/cgroup/memory.max)
  elif [ -r /sys/fs/cgroup/memory/memory.limit_in_bytes ]; then b=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes); fi
  local host=$(awk '/MemTotal/ {printf "%d", $2/1048576}' /proc/meminfo)
  if [[ "$b" =~ ^[0-9]+$ ]] && [ $(( b / 1073741824 )) -lt "$host" ]; then echo $(( b / 1073741824 )); else echo "$host"; fi
}
RAM_GB=$(mem_limit_gb); DISK_GB=$(df -BG --output=avail /root | tail -1 | tr -dc 0-9)
echo "resources: ${RAM_GB} GB RAM (container limit; host shows $(awk '/MemTotal/ {printf "%d", $2/1048576}' /proc/meminfo) GB), ${DISK_GB} GB free on /root"
[ "$RAM_GB" -ge 64 ] || echo "WARNING: under 64 GB RAM; R1 (TransformerLens weight conversion) may be killed for memory. Use an A100 pod for R1."
[ "$DISK_GB" -ge 45 ] || { echo "under 45 GB free on /root: not enough for Qwen3-14B plus activations" >&2; exit 1; }
# reuse Gate 2a's recorded revision when it exists on the volume, so later 14B work (M7, R4) runs the same checkpoint;
# MODEL_REV overrides; otherwise resolve the Hub's current revision (first run)
PRIOR=/workspace/gate2a/r1/run_header.json
if [ -n "${MODEL_REV:-}" ]; then REV=$MODEL_REV
elif [ -f "$PRIOR" ]; then REV=$(python -c "import json; print(json.load(open('$PRIOR'))['model_revision'])"); echo "reusing Gate 2a's model revision ${REV:0:12}"
else REV=$(retry 4 python -c "from huggingface_hub import HfApi; print(HfApi().model_info('Qwen/Qwen3-14B').sha)"); fi
retry 3 python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-14B', revision='$REV')" >/dev/null
python - "$REV" <<'PY'
import hashlib, os, sys
from huggingface_hub import HfApi, snapshot_download
rev = sys.argv[1]
d = snapshot_download("Qwen/Qwen3-14B", revision=rev, local_files_only=True)
info = HfApi().model_info("Qwen/Qwen3-14B", revision=rev, files_metadata=True)
bad = []
for f in info.siblings:
    p = os.path.join(d, f.rfilename)
    if not os.path.exists(p) or (f.size is not None and os.path.getsize(p) != f.size):
        bad.append(f.rfilename); continue
    if f.lfs is not None:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for c in iter(lambda: fh.read(1 << 24), b""):
                h.update(c)
        if h.hexdigest() != f.lfs.sha256:
            bad.append(f.rfilename)
assert not bad, f"model files do not match the Hub: {bad}"
print(f"Qwen/Qwen3-14B @ {rev[:12]}: {len(info.siblings)} files verified")
PY
# the source scripts load "Qwen/Qwen3-14B" by name (TransformerLens needs the official name), which resolves
# through refs/main: point it at the verified revision so a later Hub change cannot swap the model mid-gate
mkdir -p $HF_HOME/hub/models--Qwen--Qwen3-14B/refs && echo -n "$REV" > $HF_HOME/hub/models--Qwen--Qwen3-14B/refs/main

# 4. run header
cp /workspace/lens-fit/gate2a_spec.json $G/ 2>/dev/null || true
python - "$REV" "$SRC_COMMIT" <<'PY'
import hashlib, importlib.metadata as md, json, platform, sys, torch
from datetime import datetime, timezone
rev, src = sys.argv[1], sys.argv[2]
spec = open("/root/gate2a/gate2a_spec.json", "rb").read() if __import__("os").path.exists("/root/gate2a/gate2a_spec.json") else b""
def v(p):
    try:
        return md.version(p)
    except md.PackageNotFoundError:
        return None
hdr = {"date": datetime.now(timezone.utc).isoformat(), "source_commit": src, "model": "Qwen/Qwen3-14B", "model_revision": rev,
       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "python": platform.python_version(),
       "packages": {p: v(p) for p in ["torch", "transformers", "transformer_lens", "numpy", "scikit-learn",
                                       "pandas", "huggingface-hub", "hf-transfer"]},
       "spec_sha256": hashlib.sha256(spec).hexdigest() if spec else "spec not found"}
json.dump(hdr, open("/root/gate2a/run_header.json", "w"), indent=2)
print(json.dumps(hdr, indent=2))
PY
echo "gate 2a environment ready: source /root/gate2a/.venv-tl/bin/activate && python /workspace/lens-fit/gate2a_run.py R1"
