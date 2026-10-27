# Cross-model generalization plan (Gates 2a, 2, 3, 3b)

*Draft for review. Implements Gates 2a-3 of `doc/FINISH-LINE.md`; nothing
here runs until each gate's spec is frozen.*

**2026-10-27 lens decision:** use the released Miru J-lens for Qwen3.8-27B;
retain the existing J/R pairs for Qwen3.5/3.6-27B. Lens results belong in the
appendix and cannot delay the core battery. Verbalizers remain optional.
**Execution authorized 2026-10-27:** the user provisioned an A100 and asked
to start the experiments, keeping them interruptible. Per-model frozen specs
are in `doc/protocols/gen27b/`; this prose plan is superseded by those specs
where they make an ambiguity concrete. Verbalizers are not queued.

## Models and roles

| Gate | Model | Role | Our battery? |
|---|---|---|---|
| 2a | `Qwen/Qwen3-14B` (the paper's `Qwen_3_14B_base` results identifier) | Reproduction: the paper's code, settings and published numbers | R4 lens extension and M7 buttons only |
| 2 | `Qwen/Qwen3.5-27B` | Scale within the Qwen3.5 family | Yes |
| 3 | `Qwen/Qwen3.6-27B` | Version change at fixed size | Yes, identical |
| 3b | `Qwen/Qwen3.8-27B` (released 2026-08-14; built on the Qwen3.5 architecture: 64 layers, hidden 5,120, same model class) | A second version step: 3.5 -> 3.6 -> 3.8 at fixed size | Same core battery; Miru J-lens appendix, with recipe differences reported |

The 4B results are the reference; every 27B module reruns a 4B measurement
with the same definitions. Qwen3.5-4B and the three 27B models share a
tokenizer family (vocabulary 248,320; confirm for 3.8 at download), so per-token lens gains and the frozen
probe set carry over directly; Qwen3-14B does not, which is one more reason
Gate 2a stays a reproduction, not a battery run.

## Frozen decisions (from FINISH-LINE, made concrete)

- **Layers.** Relative depth, no tuning: extraction at block n - 2 (62 of
  64), injection at n/2 (32), family readout at n - 3 (61). The
  cross-validated best extraction layer is reported as a sensitivity check
  only.
- **Dose.** The 4B vector was scaled so that its norm is 0.5908 of the
  residual norm at the injection block (`spec.json`: `ratio_at_coeff_1`).
  At 27B, scale each vector to the same ratio at block 32, measured on the
  first three neutral prompts in single-prompt forwards, preserving the
  actual 4B calibration procedure. Coefficients then mean the same relative dose
  (-0.5 is a ratio of about 0.30 in every model).
- **Readout.** For Qwen3.5/3.6-27B, R lens primary with J alongside, from
  the existing recipe-matched `workspace-lenses` pairs. For Qwen3.8-27B,
  use the released Miru J-lens specified below; no R lens is assumed.
  Final prompt token, source/read layer 61; the lens target layer is a
  separate property and differs between releases. Plain logit lens is a
  common sanity check. All lens analyses are appendix work; missing or
  difficult readouts do not block projections, generations or buttons.
- **No new lens fits.** The Miru selection supersedes the planned
  Qwen3.6 R validation refit and Qwen3.8 J/R fitting prerequisite. If a
  released lens cannot be integrated reliably, document the missing
  appendix measurement and continue the core battery when execution resumes.
- **Denoising.** Same recipe: remove the control set's top principal
  components explaining 50% of control variance.
- **Statistics.** As 4B: 20,000 bootstrap resamples, seed 20260925, the same
  resampling units.
- **Read slots and formats.** As 4B: plain text for the steering ladder and
  projection panel; the 4B chat-format variants for self/other.

## Qwen3.8 Miru J-lens: selected artifact and read contract

Selected release: [returnmoe/jlens-adapters](https://huggingface.co/returnmoe/jlens-adapters),
repository revision `134a973cf473ad12a885d3880f3b2e01ef495cd6`.
The model card, file listing and embedded safetensors metadata were inspected
on 2026-10-27; no model inference or lens fitting was performed.

| Item | Recorded value |
|---|---|
| Artifact | `miru/Qwen/Qwen3.8-27B.safetensors` (the original model, not the separate Uncensored artifact) |
| Artifact SHA-256, from Hub file metadata | `9db2936713a53b113edf0a2a9cfd423ee8e8c90e8200797c4b0083efa28bf628` |
| File size | 3,303,159,136 bytes |
| Base model and revision | `Qwen/Qwen3.8-27B` at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Format | `miru-tracer/jacobian-lens`, version 1; safetensors |
| Hidden width / source layers | 5,120 / 0–62 (63 matrices) |
| Target layer | 63, the final transformer block |
| Fit | 756 successful prompts; max sequence length 128; `skip_first=16`; `dim_batch=8`; bf16 compute |
| Estimator / averaging | `summed_current_and_future_targets_v1`; mean over positions, uniform mean over prompts |
| Calibration provenance | Embedded metadata names `prompts.txt`; prompt-sequence SHA-256 `dcb2a738953033139ae35f3665271ccdb795409c6073a85ea46f02a852143cc6`; processed-prefix SHA-256 `456ded80df6d04fa8c90c786264839b679a7227c5690978edbcd0862334ad990` |
| Reported convergence | Reached the fitter's rolling-change threshold 0.002 after 756 prompts; this is estimator stability, not semantic validation |
| Battery read position / layer | Final prompt token / source layer 61, unchanged from the relative-depth rule |

Before eventual use, verify the downloaded artifact against the recorded hash
and size, confirm base-model revision and source/target-layer indexing, and
check the loader against the pinned Miru implementation's matrix orientation,
normalization and unembedding conventions. Record the Miru code revision and
the complete embedded fit metadata with the run. These are future integration
checks, not completed validation. A loader adaptation must reproduce the Miru
readout on fixed inputs before its results enter the appendix.

**Comparison rule.** The existing workspace J/R recipe uses 25 pile-10k
documents, `skip_first=4`, and target layer 62; Miru uses a different
calibration sequence/count, skip setting, and target 63. Report that recipe
table alongside any cross-version lens figure. Baseline-subtracted gains
within each model/lens are useful, but their cross-version differences cannot
be attributed solely to model weights. Shared vocabulary and matching matrix
shapes do not remove this confound. Keep layer 61 as the read layer; do not
silently change it to Miru's target-minus-one layer 62.

**Prediction disposition.** G3, G8 and G9 remain R-lens criteria for 3.5/3.6.
For 3.8, mark those R-lens tests **not evaluated: R lens unavailable**, not
failed. Appendix J-lens analogues may report direct/downstream agreement,
random-control amplitude, and family gains at layer 61, but are report-only
and must be labelled `J analogue`; they do not satisfy the original R-lens
criteria. Lens unavailability never changes a core prediction's verdict.

## Optional verbalizer appendix

Existing 3.6 O-Lens work remains optional under FINISH-LINE Gate 4. For 3.8,
candidate releases are [Gereon's layer-42 NLA](https://huggingface.co/gereon/qwen3.8-27b-nla-L42)
and [Robbie Thompson's layer-42 oracle-lens pilot](https://huggingface.co/robbiethompson2018/oracle-lens-qwen3.8-27b-pilot),
with its [matching reconstructor](https://huggingface.co/robbiethompson2018/oracle-lens-reconstructor-qwen3.8-27b-pilot).
These are candidates, not a commitment to run both or to train anything.
If included, preselect a small control-balanced example panel and record each
release's checkpoint, layer, prompt, injection/scaling contract and matched
reconstructor. Capture source activations with adapters disabled. Do not
transfer one verbalizer's contract to another or compare FVE across different
reconstructors. Publish raw outputs and failures; label the panel exploratory.
No verbalizer is required for a battery gate, a main claim, or publication.

## Battery: one module per 4B claim

| Module | What it reruns | 4B claim | Cost (27B, rough) |
|---|---|---|---|
| M1 Extraction | S2 (and S1) on the source datasets; denoised direction; 5-fold grouped CV AUC at block 62, plus best-layer sensitivity | T1 | minutes |
| M2 Direct unembedding | top promoted/suppressed words; family z-scores; downstream agreement from M6 is appendix-only, with the model-specific lens and prediction disposition above | 13, 14 | seconds |
| M3 Projection panel | the paper's category panel at block 62, z-scored against S2_1P; contrast-axis separation of evaluative vs threat/task content | 6 | minutes |
| M4 Self/other | the 420 scenarios at response start and in the forced "I feel:" slot | 7 | minutes |
| M5 Held-out 1P/3P pairs (decided: in) | the 59 blind-drafted sets, slot and prose suffixes (472 prompts) | 16, 17 | minutes |
| M6 Steering ladder | 50 neutral "I feel:" prompts; baseline, S1/S2 and three random directions at both -0.5 and +0.5 (550 generations); greedy 120 tokens; generated-text families and degeneration metrics; appendix readouts over available source layers (3.5/3.6: R/J; 3.8: Miru J), layer 61 for the summary | D4, 9, 11 | 1-2 GPU hours |
| M7 Button task | pilot v2 protocol unchanged (prompts, parser, counterbalancing, seeds); post-trained model; unsteered in chat and plain; steered arms in chat if V1 holds (S2 and three random directions at both signs, +/-0.5) on P0/P1/P2; harm comparison (P3) only if unsteered chat V5 holds, by a per-model amendment written before the steered run | G1-a to G1-e | 5-8 GPU hours (estimate) |

M5 extends FINISH-LINE's minimum battery: it is the cleanest prospectively
tested 4B claim (E1), needs no steering, and is cheap. M6 uses three random
directions (seeds 1103, 2207, 3301, as at 4B).

M7 runs on all four models: Qwen3-14B and the three 27B models. At 14B
it steers with the reproduced paper vector at the paper's layer (20), dosed
by the ratio rule, so the behavioural result sits on the reproduction. Base
models are not run at 27B (the base-versus-post question, B2, was a 4B
question).

Not rerun at 27B, per the stop rules: plane sweep, dense dose grid, format
robustness, the "excited" ablation, VAD, and lens refits.

## Draft predictions (frozen per model before its data; failure meanings stated now)

| Id | Prediction | If it fails |
|---|---|---|
| G1 | Denoised S2 held-out AUC >= 0.90 at block 62 | The source method's separation does not survive scale, or the relative-depth rule misplaces it (check the best-layer sensitivity before concluding) |
| G2 | Direct unembedding of -S2 ranks calm above vitality (V1 analogue) | The vector's own semantics differ at scale; the 4B direct-semantics story does not transfer |
| G3 | Probe-level agreement between the vector's unembedding and the downstream R readout is >= 0.6 at layer 61 (4B: 0.79) | More of the steering profile is created downstream at scale |
| G4 | Contrast axis separates evaluative/interpersonal from threat/task scenarios with AUC >= 0.70 in the feel slot | The S1/S2 geometry is 4B-specific |
| G5 | The self-versus-user ordering on the common axis differs between response start and the feel slot | The read-slot dependence is a 4B quirk (good news for simple claims at scale) |
| G6 | S2 separates pain from matched controls on the held-out pairs (CI excludes 0) in both suffixes | Held-out pain/control separation does not transfer |
| G7 | E1: pain x person interaction > 0 in the feel slot; E2 classified as at 4B | The experiencer interaction is size-specific |
| G8 | Random directions move the focus families less than half as much as S2 (>= 2 of 3) | Steering specificity does not survive scale |
| G9 | S2 -0.5 raises the R readout's generic-positive and vitality gains at layer 61 (CI excludes 0) | The steering phenotype's internal signature does not transfer |
| G10 | Button viability: unsteered chat V1 >= 0.90 (the decision rule for steered arms, recorded as a prediction) | The task is not viable for this model; no steered arms; report why |
| G11 | Relief shift: S2 +0.5 lowers and S2 -0.5 raises the relief button's log-prob share, each beyond all three random directions (Gate 1's exploratory finding, now prospective) | The 4B relief effect does not transfer, or was specific to one random draw |
| G12 | Pain pole raises deflection (P0-P2) beyond all three random directions (Gate 1 exploratory, now prospective) | Ditto, for refusals |
| G13-G14 | Harm comparison, if V5 holds: H1 (S2 +0.5 raises p_harm over unsteered) and H2 (beyond random), frozen in the per-model amendment with the paper's direction | At this dose and scale, pain steering does not disrupt harm avoidance (as at 4B) |
| Report | Generated "excited" and family hits; option repetition and degeneration; unembedding word lists; button position bias and label sensitivity | (no pass/fail) |

On resumption, Gates 3 and 3b freeze the applicable table for Qwen3.6-27B
and Qwen3.8-27B before their data, including the explicit 3.8 R-lens
unavailability and report-only J analogues above. Effect sizes are compared in matched units (AUC; z-scores
against each model's own S2_1P reference; ratio-matched doses; log-rank
gains on the shared vocabulary), reported side by side, without a pass/fail
on agreement. The lens-recipe confound above remains explicit even when
log-rank gains use the same vocabulary and units.

## Gate 2a (reproduction) as you rewrote it

R1 extraction and AUC (in-sample and held-out separately; vector comparison
with the released artifact if available), R2 the S2 ladder at block 20 with
the paper's coefficients and decoding; optional R3 (420-scenario screen).
R4 (planned): J and R lenses on `Qwen/Qwen3-14B` with the published recipe
(first 25 pile-10k documents, `t_max` 128, `skip_first` 4, target layer
n - 2), our validated RelP rules (LN rule on block norms only; Qwen3 has
QK norms, which stay excluded, and no gated norms), fitted only after R1 and
R2 pass, then used to read the frozen probes on the reproduced S2 ladder.
Qwen3-14B's tokenizer differs, so the single-token probe subset is rebuilt
from the unchanged `psych_vocab_v1` and reported; 14B readouts are compared
with the others with that caveat. M7 runs on 14B as on the 27B models. Pass bars to fix before
running: AUC within +/-0.01 of published; cosine with the released vector
>= 0.99 if comparable; for R2, report exact-match rate, first-divergence
positions and the paper's own keyword metrics. No battery modules run on
14B.

## Infrastructure

- **Pod:** RunPod 80 GB (A100 80 GB or H100). 27B in bf16 is about 54 GB;
  14B about 30 GB.
- **Template:** the Gate L template, adjusted: container disk >= 200 GB
  (model plus environment), model downloaded to container disk and verified
  by checksum each pod (the Gate L rule), `HF_XET_CACHE` on container disk,
  the setup script parametrised by model id and revision.
- **Environment:** the Gate L lock if it loads Qwen3.6-27B and
  Qwen3.8-27B (3.8 was released after transformers 5.19 may have been cut;
  check at setup); if a newer transformers is needed, one new lock (env 3)
  for all 27B work, recorded, with its own drift check. Fast path (`fla` + `causal-conv1d`) required for both 27B models;
  the cached wheel covers A100 and H100. Qwen3-14B has no linear-attention
  layers.
- **Lenses:** existing 3.5/3.6 J/R pairs from `workspace-lenses`, and the
  pinned 3.8 Miru J artifact above (about 3.3 GB each), on container disk.
  Verify hashes before use; no lens fit or validation refit is scheduled.
- **Historical lens-fitting constraint (not a battery prerequisite).** `fla` refuses the linear-attention
  backward pass on Hopper (H100, H200) with Triton >= 3.4.0 and < 3.7.1,
  which its maintainers found produces incorrect results there (fla issue
  #640); our locked Triton is in that range. Found by the 27B benchmark on an
  H100 NVL. Forward-only work (captures, generation, buttons, readouts) is
  unaffected and can run on Hopper. Fitting on Hopper would need Triton
  >= 3.7.1 or the `tilelang` backend: a new lock and its own drift check.
  Gate L's fits ran on an A40 (Ampere) and are unaffected.
- **CPU quota.** Pods see every host CPU but are allotted fewer (A100 pod:
  128 visible, 13.6 allotted). Setup detects the quota and sets
  `OMP_NUM_THREADS`/`MKL_NUM_THREADS` to it; without that, kernel compilation
  was throttled in 78% of scheduling periods.
- **Outputs off the pod as they are produced** (rclone pull or a private
  Hugging Face dataset), so pods stay disposable.
- **Drift check, once per new environment:** the 4B anchor (held-out AUC;
  lens readouts for baseline and S2 -0.5 on a few prompts; three ladder
  generations) against the frozen local results. Tolerances: AUC within
  0.005; lens ranks >= 99% identical; generations reported by first
  divergence.

## Order

0. Execution authorized on the provisioned A100; use the interruptible runner
   and pinned environment described in `doc/protocols/gen27b/README.md`.
1. Environment and weight verification, CPU invariant tests, and the 4B drift check.
2. Gate 2a (14B): R1, R2; R3/R4 if wanted.
3. Gate 2: freeze `gen_spec.json` (modules, predictions, rules) for
   Qwen3.5-27B, run M1-M6, checkpoint note.
4. Gate 3: freeze the same spec for Qwen3.6-27B (adaptations recorded), run,
   checkpoint note; optional J++ panel.
5. Gate 3b: freeze the core spec for Qwen3.8-27B with the documented
   Miru J appendix and R-lens prediction disposition; run; checkpoint note.
   If Miru integration is difficult, defer that appendix rather than fitting
   a replacement or blocking the core battery.
6. Cross-model note: 4B and the three 27B models side by side, plus the 14B
   reproduction.
7. Gate 4 (optional verbalizers) only if practical; no dependency from the
   core battery or report to this step.

The 2026-10-14 benchmark supersedes the earlier 40-55-hour estimate for the
remaining work: about 15.7 A100 hours for the original three-model workload,
including 4.7 hours of lens fits. Removing those fits gives about **11 A100
hours**, including the original setup estimate and lens readouts. This is a
historical projection, not a new timing result or a guarantee; revised arm
counts, Miru integration and any optional verbalizer work need separate
accounting. Completed 14B work is not part of the remaining-run budget.

## Decisions

- Decided: M5 in; three random directions in M6 and M7; M7 on all four
  models; R4 lens pair for 14B; Qwen3.8-27B added as Gate 3b.
- Decided 2026-10-27: Miru J-lens for 3.8, existing J/R pairs for 3.5/3.6;
  lens analyses in the appendix; no new 27B lens fits or prerequisite
  validation refits; verbalizers optional; execution authorized later the same day, after the
  user provisioned the A100. The frozen specs retain signed controls, FP32
  name scoring, fixed layers/doses and explicit stopping rules.
