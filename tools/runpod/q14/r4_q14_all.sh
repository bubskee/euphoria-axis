#!/bin/bash
# R4 on Qwen3-14B, start to finish. Pod: A100; MODEL_ID=Qwen/Qwen3-14B MODEL_REV=40c069824f4251a91eefaf281ebe4c544efd3e18
#   AUTORUN="bash /workspace/lens-fit/r4_q14_all.sh". Resumable: finished fits are kept on the volume and skipped.
set -euo pipefail
step() { echo; echo "=== [$(date -Is)] $*"; }
source /root/gateL/.venv-fast/bin/activate
export MODEL_PATH=${MODEL_PATH:-/root/models/Qwen3-14B}
cd /workspace/lens-fit
step "paper repository (published vector, ladder prompts) at the pinned commit"
[ -d /root/gate2a/src/.git ] || git clone -q https://github.com/valen-research/Pain-axis.git /root/gate2a/src
git -C /root/gate2a/src checkout -q 8d1649c03a63a39c9aa092532c376800cc4a3863
mkdir -p /root/r4
for f in q14_J_n25_bos0 q14_R_n25_bos0 q14_R_n25_off25_bos0; do      # reuse fits already kept on the volume
  [ -f /workspace/r4_q14/$f.pt ] && [ ! -f /root/r4/$f.pt ] && cp /workspace/r4_q14/$f.pt /root/r4/ || true
done
step "fit J (documents 0-24)";        MODE=J python r4_fit.py
step "fit R (documents 0-24)";        MODE=R python r4_fit.py
step "fit R (documents 25-49)";       MODE=R DOC_OFFSET=25 python r4_fit.py
step "validation";                    python r4_compare.py
step "readout on the S2 ladder";      python r4_readout.py
step "R4 done: /workspace/r4_q14/"
