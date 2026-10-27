# 27B battery v1 — frozen before observations, 2026-10-27

Execution was authorized by the user on the provisioned A100-SXM4-80GB.
The three model JSONs and `source-inputs.json` are frozen before any 27B
data. `freeze-manifest.json` records their byte hashes. This README is an
operational guide; it does not amend a frozen prediction.

The per-model specs make the draft design concrete: fixed extraction 62,
injection 32, readout 61, within-fold control PCA, first-three-prompt dose
calibration matching the actual 4B procedure, and the unchanged pilot-v2
button instrument. The neutral ladder has 11 signed conditions (550
generations/model). Buttons have 2,784 unsteered output rows; conditional
steered arms add 6,464 rows without harm or 8,080 with harm. Compare relief
against random controls at the same sign. Name scoring computes the output
projection in float32; generation remains bf16. All verdicts and any
floor/ceiling limits are retained.

G3/G8/G9 are appendix R-lens tests for 3.5/3.6; at 3.8 they remain unavailable
as R tests, with report-only Miru J analogues. No new lens fits. Existing
verbalizers and J++ are optional and are not queued. An appendix failure does
not erase or delay a completed core battery.

## Execution and durable state

The pinned checkout is `/root/gen27b/repo`; its commit is copied to
`/workspace/gen27b/EXECUTION_COMMIT`. Models, environments and temporary
download caches use container disk. Results, capture-batch caches, row
journals, logs and environment lock use `/workspace/gen27b/` on the volume.
Completed model weight copies may be deleted from the dedicated container
cache to fit the next model; result files are retained.

One GPU process runs at a time. The 4B environment gate precedes all 27B
observations: fixed-layer held-out AUC must be within 0.005 of the stored
value. Three stored generations are compared by first divergence, without
an exact-text pass bar. Lens validation is kept off this core gate.
Every model also checks the real fast attention path and exact prefill
steering arithmetic. Core failures stop the queue; optional lens failures
are recorded and deferred.

Start or resume the same queue from the pinned checkout:

```bash
cd /root/gen27b/repo
tmux new-session -d -s euphoria27b \
  'bash -o pipefail -c "bash tools/runpod/gen27b_all.sh 2>&1 | tee -a /workspace/gen27b/battery.log"'
```

Request a safe pause:

```bash
touch /workspace/gen27b/PAUSE
```

The current atomic unit finishes, then the process exits with code 130.
SIGINT/SIGTERM also request a safe stop. Each generation and button turn is
flushed and fsynced; capture batches and completion manifests are atomic.
The journal reader truncates only a torn final line and rejects duplicates
or a mismatched identity. Resuming skips completed units and refuses a
changed model, spec, environment or runner implementation.

After the stopped process exits, remove `PAUSE` and run the same queue command.
Do not start a second copy; the runner holds a lock. To inspect:

```bash
cat /workspace/gen27b/current.json
tail -30 /workspace/gen27b/battery.log
tmux capture-pane -pt euphoria27b -S -30
```

Per-model `core_scorecard.json` is produced before the button task, followed
by `m7_scorecard.json`. `battery.complete.json` means the core modules and
scorecards completed and includes artifact hashes. Optional appendix status
is separate. `ALL_CORE_BATTERIES_COMPLETE` is written only after all three
models finish. CPU tests use synthetic data and the frozen trial instrument;
they do not consume 27B experimental observations.
