#!/bin/bash
# Gate 2a, start to finish. For an unattended pod: AUTORUN="bash /workspace/lens-fit/gate2a_all.sh"
# Each step is resumable; rerunning this script skips work that is already done.
set -euo pipefail
step() { echo; echo "=== [$(date -Is)] $*"; }
cd /workspace/lens-fit
step "environment (source checkout, TransformerLens venv, verified Qwen3-14B)"
bash gate2a_env.sh
source /root/gate2a/.venv-tl/bin/activate
for s in R1 R2a R2b; do
  step "$s"
  python gate2a_run.py "$s"
done
step "comparison"
python gate2a_compare.py
step "Gate 2a done: /workspace/gate2a/scorecard_2a.json"
