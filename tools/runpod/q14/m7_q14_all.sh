#!/bin/bash
# M7 on Qwen3-14B, start to finish. AUTORUN="bash /workspace/lens-fit/m7_q14_all.sh"  (A100; MODEL_ID=none)
# Resumable: rerunning skips finished trials. The steered script applies the spec's decision rules itself.
set -euo pipefail
step() { echo; echo "=== [$(date -Is)] $*"; }
cd /workspace/lens-fit
step "environment (Gate 2a's: TransformerLens venv, Qwen3-14B at Gate 2a's revision)"
bash gate2a_env.sh
source /root/gate2a/.venv-tl/bin/activate
step "unsteered arms (chat and plain)"
python m7_q14_unsteered.py
step "viability"
python m7_q14_score.py
step "steered arms (decision rules, steering gate, then trials)"
python m7_q14_steered.py
step "scoring"
python m7_q14_score.py
step "M7 q14 done: /workspace/m7_q14/m7_q14_scorecard.json"
