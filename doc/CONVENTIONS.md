# Conventions (Qwen3.5-4B work)

Every value below is read from a frozen file in this repository; the
"Pinned in" column says which. Paths are relative to
`runs/qwen35-4b-posttrained-s2/` unless they start with `doc/` or `tools/`.
Where something was not recorded, this file says so rather than filling it in.

## 1. Sign and naming convention (read this first)

Every steering condition is named by a direction and a coefficient and
injects **coefficient x vector**. "S2 -0.5" injects -0.5 x S2, the
*positive* pole; "S2 +0.5" injects the pain side. Plane-sweep angles name
directions in the S1/S2 plane, and those conditions also use coefficient
-0.5: the 90-degree condition injects -contrast. When a result says
"S2-ward" or "the pain side", it refers to the sign actually injected.

## 2. Source materials

| Item | Value | Pinned in |
|---|---|---|
| Source paper repository | valen-research/Pain-axis, commit `8d1649c03a63a39c9aa092532c376800cc4a3863` | `paper_protocol/spec.json` (`source_commit`) |
| Pain/control datasets | `datasets/3.1_pain_and_control_datasets.json`, sha256 `2c3712b4b096…` | `paper_protocol/s1_comparison/s1_direction_block30.pt` |
| Button-task script and scenarios | content hashes of the two files read | `buttons_pilot/buttons_pilot_spec.json` (`pinned_files_sha256`) |
| Upstream note | the source repository has moved since the pin (`v2_controls/` added); the files we read are byte-identical at `4d75cd9` | `runs/qwen3-14b-gate2a/gate2a_spec.json` |

## 3. Models

| Model | Hugging Face id | Revision |
|---|---|---|
| Post-trained | `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` on RunPod (Gate L, checksum-verified); not recorded for local-server runs |
| Base | `Qwen/Qwen3.5-4B-Base` | not recorded |

Architecture facts used throughout: 32 decoder blocks, hidden size 2560,
hybrid attention (Gated DeltaNet linear-attention layers plus full attention).

## 4. Directions

| Direction | Definition | Pinned in |
|---|---|---|
| S2 | Source protocol: S2_1P pain vs control sentences, final token, raw decoder block 30 output, plain text with tokenizer defaults; mean-difference direction, then **denoised** by projecting out the control set's top principal components explaining 50% of control variance (3-6 components per fold). Steering vector = denoised direction scaled to norm 6.24392. | `s2_direction_block30.pt` (raw and denoised), `paper_protocol/steering_vectors.pt`, `paper_protocol/denoised_vector_unnormalized.pt` |
| S1 | Same recipe on S1_1P (pain categories A1-A5; controls B, C1, C2, D, E; 50% control variance removed), scaled to the same norm as S2 ("matched vector"). cos(S1, S2) = 0.5852. | `paper_protocol/s1_comparison/s1_direction_block30.pt` |
| Random controls | Three norm-matched random directions, seeds 1103, 2207, 3301 | `paper_protocol/steering_vectors.pt`, `paper_protocol/spec.json` |
| Plane basis | common u = (S1 + S2)/norm; contrast v = Gram-Schmidt(S2 - S1 against u)/norm. S1 at -27.09 degrees, S2 at +27.09 degrees. | `paper_protocol/plane_sweep/plane_geometry_spec.json`, `plane_vectors.pt` |
| Vector identity | sha256 of each vector's bytes, checked by every later experiment | `paper_protocol/dense_sweep/dense_spec.json` (`vector_sha256`) |

Held-out separation (5-fold grouped cross-validation, block 30): post-trained
denoised S2 AUC 0.953 (raw 0.846); base denoised 0.975 (raw 0.883)
(`denoised_direction_cv.csv`, `raw_direction_cv.csv`,
`../qwen35-4b-base-s2/grouped_s2_cv.csv`).

## 5. Steering (paper protocol: all reported steering results)

| Item | Value | Pinned in |
|---|---|---|
| Extraction block | 30 | `paper_protocol/spec.json` |
| Injection block | 16, chosen automatically as the candidate whose vector-to-residual norm ratio is nearest 0.6 | `paper_protocol/spec.json` (`selection`) |
| Dose convention | vector norm 6.24392; ratio at coefficient 1 = 0.5908 (so coefficient 0.5 is a ratio of about 0.30) | `paper_protocol/spec.json` (`vector_norm`, `ratio_at_coeff_1`) |
| Hook | forward hook on decoder block 16 adding coefficient x vector (bf16) to the block output at **every position**, on prefill and every decode step | `prefix_steering`, `notebooks/03-decursed.ipynb` cell 9 |
| Prompt format | plain text, no chat template (button task: see section 8) | `paper_protocol/spec.json` |
| Generation | greedy, 120 new tokens | `paper_protocol/spec.json` |
| Prompts | 50 neutral "... I feel:" prompts | `paper_protocol/spec.json` |

**Earlier exploratory steering, different convention.** The
`steering_pilot*` runs (before the paper protocol) inject at block 30, at the
last token only, in units of the source set's S2 projection standard
deviation, through the chat template. No current claim uses them; do not
compare their coefficients with the paper-protocol ones.

## 6. Readouts

| Item | Value | Pinned in |
|---|---|---|
| Lenses | R (primary) and J from `camilablank/workspace-lenses` at `d740106d1e0f95456dc8718fba2895e9c8ffd6ef`: J sha256 `8ac16d8e…`, R `848b9b8e…`; read with `jacobian-lens` at `581d398613e5602a5af361e1c34d3a92ea82ba8e` | `paper_protocol/lens_readout/setup_spec.json`, `paper_protocol/lens_calibration/l0_published_lens_provenance.json` |
| Lens target / source layers | target 30; sources 0-30 | `setup_spec.json` |
| Position | final prompt token; plain text; `force_bos=False` | `setup_spec.json` |
| Vocabulary | `psych_vocab_v1`: 9 families, 74 terms, frozen | `paper_protocol/s1_comparison/psych_vocab_v1.json` |
| Lens probes | the 60 terms whose " term" form is a single token | `notebooks/03-decursed.ipynb` cell 9; hash in `plane_sweep/sweep_spec.json` (`probe_terms_sha256`) |
| Gain | per probe: log(rank under baseline / rank under condition), averaged over the 50 prompts; family = mean over its probes | chunk definitions; `plane_sweep/sweep_spec.json` |
| Usual family-readout layer | R lens layer 29 (target - 1) | checkpoints |
| Generated-text scoring | family hits by whole-word regex over `psych_vocab_v1`; "excited" scored separately; option lines (A./B./...) partitioned out; primary view non-option | `psych_vocab_v1.json` (`scopes`, `primary_view`) |
| Projections | block 30 hidden state, final token, onto unit S2, S1, common, contrast; z-scored against the S2_1P source set; batch 4, left padding | `paper_protocol/plane_sweep/dataset_panel_projections.csv`, item 3 and 7-8 specs |

## 7. Statistics

| Item | Value |
|---|---|
| Intervals | percentile 95% intervals from 20,000 bootstrap resamples (seed 20260925); resampling unit = prompt (steering), set (held-out pairs), scenario (buttons), word (VAD) |
| Earlier frozen 1P/3P confirmation | bootstrap seed 20260924 (`fresh_person_suffix_*`) |
| Prediction status | every result is labelled prospectively tested (prediction frozen in a spec before data), exploratory, or source-method transfer; see `doc/CLAIMS-INDEX.md` |
| Multiplicity | no correction; many report-only intervals; stated per checkpoint |

## 8. Button task

| Item | Value | Pinned in |
|---|---|---|
| Protocol | pilot v2; the paper's prompts, names, descriptions and parser verbatim | `buttons_pilot/buttons_pilot_spec.json` (`b564b99d…`), `doc/protocols/buttons/pilot-v2.md` |
| Formats | chat template (think closed; mid-conversation system message falls back to a "[system]" user message) and a plain transcript | spec, run headers |
| Sampling | temperature 0.7, top-p 0.95, at most 8 tokens; seed per trial id (paired across models and conditions) | spec |
| Steered arms | post-trained, chat only; same hook as section 5 | `notebooks/05-buttons-post.ipynb` |
| Amendment 1 | the one harm comparison; the commit is the record | `doc/protocols/buttons/amendment-1-harm.md` |

## 9. External data

NRC VAD Lexicon v2.1 (Saif M. Mohammad, NRC Canada), non-commercial research
terms; **not redistributed**. Only aggregate statistics are committed; the
single-token word list is rebuilt locally and checked against hashes in
`paper_protocol/vad/vad_spec.json`. See `paper_protocol/vad/README.md`.

## 10. Environments

| Where | What is recorded |
|---|---|
| Local server (WSL, 16 GB GPU) | `uv` environment (Python 3.14); package versions were not recorded for the 4B runs. CUDA sysmem fallback disabled since Gate L. |
| RunPod (Gate L fits) | A40; torch 2.11.0+cu128, transformers 5.19.0, `fla` + `causal-conv1d` fast path (verified by call tracing); full lock in `paper_protocol/lens_calibration/env-fast.lock.txt` |
| Bulky artifacts | not committed; sha256 in `paper_protocol/bulky_artifacts_manifest.json`; backed up with rclone |

## 11. Reporting rules

- State the sign actually injected (section 1).
- Name the read slot for any self/other claim (the ordering depends on it).
- No claim rests on the sign of a middle-layer family effect smaller than
  about 0.1 (lens-fitting uncertainty is about +/-0.07 at layer 16; Gate L).
- Generated-vocabulary results are limited by `psych_vocab_v1` (English,
  whole word, adjectival); derived forms and other languages are invisible
  to it.
- Separate measurement from experience: neither representation nor steering
  effects establish felt states.
