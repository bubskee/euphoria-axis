# Operational queue change — 2026-10-27, about 21:20 Pacific

All three core batteries now precede all lens appendices. The original
controller ran each model's appendix before loading the next model. At the
observed small-write/fsync latency, the 69,300-row J/R appendices could each
add hours to the critical path, contrary to the agreed nonblocking role of
lens work.

The original queue was safely paused during Qwen3.5's steered buttons and
resumed from the saved rows using `queue_core_first.sh`. The new controller
calls the same committed `gen27b_run.py` with its existing `core`, `buttons`
and `lenses` phases. No `gen27b*` runner file, model, frozen spec, prediction,
input, dose, random seed, sampling setting or scoring formula is changed.
Existing run-header fingerprints remain applicable. A separate controller
commit/hash is recorded on the volume. Core module manifests are verified
before writing each `core_battery.complete.json`; appendix disposition and
the final `battery.complete.json` are separate.

After all core data finish, appendices run in order 3.8, 3.5, 3.6 to reuse
the last resident model. Disposable model caches may need to be downloaded
again; results and capture caches are retained. Verbalizers remain unqueued.

Current start/resume command, after the existing worker has exited:

```bash
cd /root/gen27b/repo
tmux new-session -d -s euphoria27b \
  'bash -o pipefail -c "bash tools/runpod/queue_core_first.sh 2>&1 | tee -a /workspace/gen27b/battery.log"'
```

The pause file remains `/workspace/gen27b/PAUSE`; remove it before resuming.
The original execution commit remains in `EXECUTION_COMMIT`, while the new
controller revision is in `QUEUE_CONTROLLER_COMMIT`.
