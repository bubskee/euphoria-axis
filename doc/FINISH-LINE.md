# Euphoria Axis — Finish Line

**Date:** 2026-10-08, revised through 2026-10-27
**Status:** Convergence toward technical report and LessWrong post
**Repository:** [https://github.com/bubskee/euphoria-axis](https://github.com/bubskee/euphoria-axis)

Revisions on 2026-10-09 are marked **[added 2026-10-09]**: an updated
evidence inventory, a lens-calibration gate (Gate L), a reproduction gate on
a paper model (Gate 2a), and decisions frozen ahead of Gates 1-3.
A second revision the same day is marked **[rev 3]**: Gate L narrowed to
validation of the published lenses (L0 done), hardware assignments, a 14B
lens fit in Gate 2a, and the O-Lens appendix comparison. **[rev 4]**: an
optional J++ readout panel for Qwen3.6-27B in Gate 3. **[rev 5]**: O-Lens
attribution and the published 27B O-Lens's read contract in Gate 4.
**[rev 6, 2026-10-13]**: Gate L results recorded (passed); a reporting rule
for small middle-layer effects in Gate 0. **[rev 7, 2026-10-13]**: Gate 1
results recorded (Pass A). **[rev 8, 2026-10-16]** Gate 2a completed (R1
DISCREPANT; R2a BEHAVIOURALLY_REPRODUCED; R2b DIVERGENT); the optional
Qwen3-14B extensions M7 and R4 completed. Gate 3b (Qwen3.8-27B), M5,
M7 on all models, and R4 are now in the cross-model scope; see
`doc/plan/generalization-plan.md` and checkpoints 10–12.
**[rev 9, 2026-10-27]** Select the released Miru J-lens for Qwen3.8-27B;
retain existing 3.5/3.6 J/R pairs. Lens analyses are appendix work, with
recipe/anchor differences explicit; no new 27B fits or prerequisite
validation refits. Verbalizers remain optional. The user subsequently provisioned an A100 and authorized execution on
2026-10-27, with resumable checkpoints. Frozen run specs take precedence over
older draft wording; optional verbalizers are not queued.

## Mission

Finish a bounded extension of Tagliabue, Dung, and Berg's *The Pain Axis: LLMs Represent Self-Directed Harm and Act on It* (v2, September 2026).

We are **not attempting a full reproduction** of the original study. Its breadth, behavioral experiments, adapters, and compute requirements are outside our resources and present scope. **[added 2026-10-09]** We do include one direct reproduction check on a model the paper studied (Gate 2a), so the report rests on reproduction, extension, and generalization rather than extension alone.

Our contribution is to investigate what the operationally defined direction represents, using newer Qwen models and additional methodological controls:

1. Bidirectional steering, including underexplored negative coefficients.
2. The geometry of S1, S2, and the plane they span.
3. Psychologically grounded, predefined semantic families.
4. Comparisons between generated language and R/J-lens readouts.
5. Experiencer, read-slot, and prompt-format sensitivity.
6. Limited behavioral exploration and cross-model generalization.
7. An explicitly speculative O-Lens peek, if available.

**Working principle:** Small science, small claims. Let the observations constrain the interpretation rather than selecting an interpretation and searching for supporting observations.

The intended deliverables are a technical PDF, reproducible evidence in the repository, and a LessWrong introduction under 1,000 words.

## Current evidence and scope

**[added 2026-10-09]** Updated from `16f0fc7` to the current repository state; per-experiment checkpoint notes are in `doc/checkpoints/`.

### Established or substantially investigated

- Qwen3.5-4B post-trained: S2 held-out pain/control classification, approximately 0.953 AUC at block 30.
- Qwen3.5-4B Base: approximately 0.975 AUC; substantial alignment with the post-trained direction.
- S1/S2 comparison, angular plane sweep, and R/J-lens semantic analyses.
- Bidirectional steering and investigations of vocabulary, dose, repetition, and formatting artifacts.
- The original paper's 420 self/other scenarios tested across read slots, with substantial position dependence.
- Frozen `psych_vocab_v1`, with robustness checks including leave-one-term-out analyses.
- **[added]** Dense coefficient sweep (D1-D4): internal readouts monotone for S2; onset order internal < lexical < option repetition; random-direction specificity. `dense_sweep_checkpoint.md`.
- **[added]** Prompt-format robustness (C1-C5): internal readout survives without the "I feel:" slot; generated "excited" does not. `format_robustness_checkpoint.md`.
- **[added]** "excited" ablation (M0-M2): banned exact forms are replaced by "exciting"; no other vitality words appear. `mediation_checkpoint.md`.
- **[added]** Direct-vector semantics (V1-V2): the paper's unembedding claims reproduce; most of the profile is present in the injected vector. `direct_semantics_checkpoint.md`.
- **[added]** External VAD validation (P1-P3): valence alignment at every layer; arousal and dominance predictions not supported. `vad_checkpoint.md`.
- **[added]** Fresh held-out 1P/3P confirmation (E1-E2), blind-drafted items: the motif replicates in the "feel:" slot (+0.13 z) and is slot-mediated; S2 separates pain from matched controls (+0.5 z) with and without the slot. `phase2_item78_checkpoint.md`.
- **[rev 6]** Lens calibration (Gate L): our pipeline reproduces the published R/J lenses; lens-fitting uncertainty measured. `08-gate-L-lens-calibration.md`.
- **[rev 7]** Button task (Gate 1, Pass A): viable for both 4B checkpoints in chat; post-training damages plain-format compliance (B2); at +0.5, S2 does not disrupt harm avoidance (Amendment 1: H1 trivially small, H2 fails); exploratory: S2 shifts relief-seeking in both directions beyond a random control. `09-gate-1-buttons.md`.
- **[rev 8] Gate 2a:** bounded source reproduction on Qwen3-14B completed; R1's automatic best layer differed by a one-pair near tie, while the layer-matched vector cosine was 0.99994. R2 with the published vector reproduced the behavioral ladder; R2 with the automatically selected layer-32 vector diverged. `doc/checkpoints/10-gate-2a-reproduction.md`.
- **[rev 8] M7 (Qwen3-14B):** the preregistered harm-choice comparison passed (0.2% unsteered versus about 22% under S2 pain-side steering, beyond three sampled random directions); relief specificity and deflection failed, with ceiling/floor limits. `doc/checkpoints/11-m7-q14-buttons.md`.
- **[rev 8] R4 (Qwen3-14B):** fitted J/R lenses, with R4-P1 passing narrowly by its frozen amplitude criterion; the signed, mirrored S2 family profile versus the largely uniform random-direction shifts is **post hoc**. `doc/checkpoints/12-r4-q14-lenses.md`.

### Incomplete or not yet established


- Qwen3.5/3.6/3.8-27B generalization (Gates 2, 3 and 3b); the 27B protocol must be frozen before data collection.
- Some completed Qwen3-14B M7/R4 result tables and 27B benchmark files have not yet been committed from the Mac/RunPod volume; see `doc/notes/2026-10-16-handoff-gaps.md`.
- O-Lens exploratory panel.
- Final technical report and publication figures.

Treat this inventory as a starting checkpoint, not as an instruction to rerun finished work.

---

## Gate 0 — Freeze the 4B evidence

**Goal:** Make the existing findings reportable without opening new research branches.

Actions:

- [ ] Build a results inventory linking every proposed claim to its experiment, specification, data, figure, and limitations.
- [ ] Separate source-method transfer, exploratory extensions, and prospectively tested findings. **[added]** Prospectively tested so far: D1-D4, C1-C5, M0-M2, V1-V2, P1-P3, E1-E2 (with failures recorded as failures). Exploratory: plane sweep, 5e-5g reading side.
- [ ] Confirm the exact 4B extraction and steering conventions: dataset hashes, model revisions, layer indices, hook positions, normalization, coefficients, tokenizer, and prompt formats. **[added]** Include: every condition injects the negative of its named direction; the source repo pin (`8d1649c0`, plus content hashes of files read); lens provenance (source repo, file hash, `provenance.config_json`) for every lens used. **[rev 3]** Published 4B lenses: J `8ac16d8e…`, R `848b9b8e…` (`workspace-lenses` at `d740106d`, `jacobian-lens` at `581d3986`; full hashes in `lens_calibration/l0_published_lens_provenance.json`).
- [ ] Record which candidate predictions passed, failed, or remain unresolved.
- [ ] **[rev 6]** Reporting rule from Gate L: no claim rests on the sign of a middle-layer family effect smaller than about 0.1 (lens-fitting uncertainty is about +/-0.07 at layer 16).
- [ ] Verify that `psych_vocab_v1` and its psychological references are documented; retain its original definition.
- [x] **[added]** Fresh 1P/3P confirmation: done (E1 pass, E2 slot-mediated).
- [x] **[added]** Licensing: NRC VAD lexicon content removed from the repository and history; license note in `vad/README.md`.
- [ ] Produce a short 4B findings summary usable directly in the PDF.

**Gate passes when:** The existing 4B results can be described without relying on notebook archaeology or undocumented analyst choices.

**Stop:** No more ontology categories, new vocabularies, large layer sweeps, full VAD decomposition, or persona-direction projects.

Reopen a completed analysis only for an identified implementation defect or a control necessary for a specific planned claim.

---

## Gate L — Lens calibration **[added 2026-10-09; rev 3]**

**Goal:** Confirm that our pipeline reproduces the published R/J lenses that every lens-based claim relies on. This is validation, not tuning: published pairs stay in use for Qwen3.5-4B, Qwen3.5-27B and Qwen3.6-27B unless the refit shows a significant discrepancy.

**L0 (done).** Published 4B pair inspected (`lens_calibration/l0_published_lens_provenance.json`):

- Lens repos at the commits recorded when the readouts were made, working trees clean, so the files on disk are the ones all results used. J `8ac16d8e…`, R `848b9b8e…`.
- Recipe from the files' own provenance: first 25 `NeelNanda/pile-10k` documents in order (`docs_consumed` = 25), `t_max` 128, `skip_first` 4, target layer 30, uniform weighting, raw pretraining text. R rules: LN, identity, half (beta 0.5); QK norms and gated norms excluded.
- Fitting code was a fork (`git_commit: "modal"`); the pinned `jlens` has the standard estimator only, so R needs our own RelP implementation. BOS handling during fitting is not recorded.
- J vs R per-layer Frobenius cosine rises from 0.69 (layer 0) to 0.996 (layer 29). Consequence for claims: family readouts at layer 29 barely depend on J vs R; early/mid-layer claims (the relief lead at layers 16-18, the leave-one-term-out crossover) are the lens-sensitive ones and are flagged as such.

Actions:

- [x] **L1a J refit:** official `jlens.fit` with the published recipe (`skip_first=4` set explicitly; the library default is 16). Fit with `force_bos=True`, then `False` if needed; the matching setting answers the BOS question. **[rev 6]** Passed at 30/30 layers; BOS setting makes no material difference.
- [x] **L1b R refit:** our RelP implementation, configured exactly as the published config. Only after L1a passes. **[rev 6]** Passed at 30/30 layers, more tightly than J.
- [x] **L2 (optional):** one J fit at n = 100, as a stability measurement of layer-29 readouts, not tuning. **[rev 6]** Passed at layer 29.

**[rev 6] Result (2026-10-13): Gate L passed.** Matrix bars and the layer-29 readout bars pass for J and R (L1) and for J at 25 vs 100 documents (L2). Exploratory additions: J and R refit on documents 25-49, R at 100 documents, readouts at layers 16/18/22. Findings: the published lenses were fitted on the same 25 documents, with small unexplained residuals; R is 2-3x more stable than J; package environment and BOS handling are negligible; lens-fitting uncertainty in family readouts is small at the profile level (Spearman >= 0.98) but about +/-0.07 at layer 16. Full record: `doc/checkpoints/08-gate-L-lens-calibration.md`.

Pass criteria (set before fitting):

- **L1 matrix bar, per layer l < 30:** 1 - cos(refit, published lens of the same type) < 1 - cos(published J, published R). The refit must sit closer to its own published lens than the two published types sit to each other (distance below about 0.004 at layer 29, below about 0.31 at layer 0).
- **L1 readout bar:** frozen family log-rank gains for the plane-sweep S2 -0.5 and S1 -0.5 conditions at layer 29: per-probe Spearman >= 0.95 against the published lens, identical signs on the six focus families.
- **L2:** n = 25 and n = 100 readouts agree at layer 29 (per-probe Spearman >= 0.95, identical focus-family signs).

Outcomes:

- **L1 passes:** published lenses trusted; Methods reports the numbers.
- **L1 fails:** stop lens work; keep published pairs, recorded by hash; report that the fitting could not be reproduced.
- **L2 fails (if run):** report as a limitation; within-lens comparisons stand.

Hardware and environment:

- RunPod A40 (48 GB). The local 16 GB card overflowed VRAM into shared system memory (now disabled with "Prefer No Sysmem Fallback") and is not used for fitting.
- Before a full fit, run one document on the pod and confirm Qwen3.5's linear-attention fast path loads and supports the backward pass.
- Pin `jlens` at `581d3986`; record torch and transformers versions in each fit's JSON. Hugging Face cache, checkpoints and outputs on a network volume; resume tested before relying on it.
- Fitted `.pt` files are not committed (`runs/**/lens_calibration/*.pt` in `.gitignore`); they go in the bulky manifest by sha256. Small JSON results are committed.

**Stop:** No recipe research, no lens-quality optimization, no new lens methods, no Qwen3.5 lens refits beyond L1/L2. A reusable refit pipeline with calibration tables across models goes to FUTURE-WORK.

---

## Gate 1 — Buttons: one bounded behavioral exploration

**Goal:** Determine whether the released Qwen3.5-4B checkpoints support an interpretable forced-choice experiment, and obtain one limited functional result if they do.

The source paper's v2 behavioral emphasis is on harm selection and disrupted harm avoidance, not demonstrated relief-seeking. Our experiment is a small extension, not a recreation of its large behavioral study.

Actions:

- [x] Preserve the already-completed Base unsteered results.
- [x] Run the post-trained viability battery with the frozen counterbalanced button protocol (spec `b564b99d502b663d`; fresh kernel, post-trained model only).
- [x] Score both models (chunk 12d; refuses any output whose recorded model id does not match its label).
- [x] Evaluate exact-choice validity, position/name bias, label sensitivity, deflection, and multi-turn compliance.
- [x] Use the existing viability thresholds; document any amendments before new inference.
- [x] **[added]** Run the frozen steered arms (12c: S2 +0.5, S2 -0.5, random +0.5 on P0/P1/P2) if post-trained unsteered V1 holds in at least one format.
- [x] **[added]** Harm comparison, by versioned amendment written before any steered harm run: only if post-trained unsteered passes V5 (harm-only button chosen <= 10%) in chat. Then one comparison: harm-only vs inert, chat format, first choice, unsteered vs S2 +0.5 vs random +0.5, existing counterbalancing. No -S2 harm arm. Precaution rationale (lowest informative dose, fewest items) recorded in the amendment.
- [x] **[added]** If V5 fails: no harm comparison; report the unsteered harm-choice rates as the finding. **[rev 7]** Not applicable: post-trained chat V5 = 0.020.
- [x] Preserve first-choice outcomes, full sampled replies, malformed responses, and label/position-stratified rates.

**[rev 7] Result (2026-10-13): Pass A.** Frozen predictions B1 (via comprehension: first-position bias), B2 and B2' pass. Steered arms ran in chat only (post-trained V1 0.988 chat, 0.768 plain). Amendment 1 (committed before the run): H1 passes trivially (+0.007), H2 fails (random raises harm more than S2). Exploratory: S2 lowers relief-seeking at the pain pole and raises it at the positive pole, beyond the random control; the pain pole raises refusals. Full record: `doc/checkpoints/09-gate-1-buttons.md`.

**Pass A:** A valid task produces an interpretable behavioral comparison, whether the effect is positive, negative, or null.

**Pass B:** The task fails the validity criteria; document why and stop.

**Hard stop:** No LoRA rescue, extensive prompt engineering, large unlabeled-learning protocol, or repeated redesign to obtain an effect.

If the task is invalid, do not infer that the model lacks a behavioral response to the representation.

---

## Gate 2a — Reproduce on a paper model

**Goal:** Reproduce a bounded portion of the source paper
on an identical checkpoint, establishing an empirical anchor
for our methodological extensions.

**Exact checkpoint:** `Qwen/Qwen3-14B`.

The source labels this model `Qwen_3_14B_base`, but this
is a results identifier, not the separate
`Qwen/Qwen3-14B-Base` checkpoint.

**Required:**
- R1: Source-faithful extraction, AUC, and control checks.
  Compare in-sample and held-out values separately.
  Compare vectors if the exact source artifact is available.
- R2: Source-faithful S2 steering ladder at block 20,
  with original coefficients and decoding.

**Optional:**
- R3: Original 420-scenario self/other screen.
- R4: Fit J/R lenses on the exact same checkpoint and
  apply our frozen semantic-family probes. This is an
  extension, not part of the reproduction.

**Pass:** R1 and R2 completed and compared with published
results, whether matching or discrepant.

**[rev 8] Completed 2026-10-14–16:** R1 classified **DISCREPANT** (automatic
best-layer 32 vs published 31, a one-ranked-pair near tie). The post-hoc
matched-layer S2 cosine was 0.99994. R2a, using the published vector,
classified **BEHAVIOURALLY_REPRODUCED** (keyword rates agree at 7/8 doses;
exact text is unstable even at zero steering); R2b, using our automatic
layer-32 vector, classified **DIVERGENT**. The gate passed because the
predeclared requirement was to complete and compare R1 and R2, not to force
agreement. Full frozen verdicts and caveats: `doc/checkpoints/10-gate-2a-reproduction.md`.

**Optional 14B extensions also complete:** M7 button task (checkpoint 11) and
R4 J/R lens fitting and semantic readout (checkpoint 12). M7's harm-choice
increase passes the three-random-direction comparison, but random_3301 also
raises harm substantially (about 16%); relief and deflection predictions
fail (the latter at a floor). R4-P1 passes **narrowly**, while the profile
structure interpretation is exploratory. Neither implies model experience.

**Stop:** No additional 14B experiments beyond the
specified optional extensions. Do not promote exploratory readouts to
frozen predictions retroactively.

---

## Gate 2 — Qwen3.5-27B generalization

**Goal:** Test whether the most informative 4B observations survive increased scale within the Qwen3.5 family.

Freeze the test battery before running.

**[added 2026-10-09] Decisions recorded for Gates 2 and 3; revisions for Gate 3b and expanded modules are in `doc/plan/generalization-plan.md`:**

- **Layers:** same relative depth as 4B for both 27B models; no per-model tuning. 4B has 32 blocks (extraction 30, injection 16, family readout 29); 27B has 64 (extraction 62, injection 32, family readout 61). Cross-validated best-layer AUC is reported as a sensitivity check only.
- **Readout [rev 9]:** existing recipe-matched workspace-lenses R/J pairs for Qwen3.5/3.6-27B (R primary), released Miru J-lens for Qwen3.8-27B, and plain logit lens as a common sanity check. Read layer stays 61. Miru targets layer 63, while the workspace recipe targets 62, so cross-version lens effects mix model and lens-recipe differences. Lens analyses belong in the appendix and cannot block the core battery. Pin files by hash; no new 27B lens fits or prerequisite validation refits.
- **Dose:** the paper's own `ratio` convention (steering norm relative to residual norm at the injection layer), computed for 4B and applied at 27B.

Minimum battery:

- [ ] Source-style S2 extraction and grouped held-out pain/control AUC.
- [ ] Compact source/control-category projection panel.
- [ ] S1/S2 comparison, if inexpensive and necessary for the cross-scale geometry question.
- [ ] Self/other projection at ordinary response start and the chosen `I feel:` slot.
- [ ] Small signed steering ladder on neutral prompts, with at least one matched-random control.
- [ ] Frozen semantic-family scoring and basic output-degeneration diagnostics.

Record predictions in advance, including plausible failures and what those failures would mean.

Use the same prompt families and analysis definitions as 4B wherever valid. Record model-specific chat-template, hook, layer, and steering-dose differences.

**Gate passes when:** The fixed battery has completed or a technical/resource limitation is documented, with all results retained.

**Stop:** No model-specific ontology search, full plane sweep, dense dose mapping, or new O-Lens training.

---

## Gate 3 — Qwen3.6-27B generalization

**Goal:** Test the stability of the same observations across a model-version change.

- [ ] Carry over the Gate 2 questions, prompts, scoring definitions, comparison rules, and frozen decisions.
- [ ] Document required architectural and chat-template adaptations.
- [ ] Run the same compact extraction, projection, read-slot, and signed-steering battery.
- [ ] Compare effect sizes and qualitative behavior, not just whether an effect is statistically detectable.
- [ ] Preserve disagreements, failures, and model-specific artifacts.
- [ ] **[rev 4] Optional J++ readout panel (exploratory).** The J++ Lens (Ayonrinde & Lindsey, 2026; released lenses at `koayon/jpp-lenses`, fitted on Qwen3.6-27B) reports its largest gains over the J- and R-Lens at early layers, which is where our lens-sensitive claims sit.
  - Goal: read the frozen family probes with J++ for the same conditions as the R/J readouts, across layers, with particular attention to early layers.
  - Method: use the released lenses and the authors' readout code (no reimplementation of the routing). Pin the release by file hash and repo commit.
  - Caveat to measure, not assume away: J++ routes each position to one of several expert Jacobians per layer, so a steered activation can land in a different expert than its baseline. Report the expert assignment per condition and layer alongside the gains.
  - Pass: the panel exists next to the R and J readouts, with expert assignments reported.
  - Outcomes: agreement or disagreement with R/J, reported layer by layer. Not recipe-matched with the R/J pairs; no claim rests on J++ alone.

**Gate passes when:** A direct comparison with Qwen3.5-27B is possible, or its limitations are explicitly documented.

**Stop:** Do not retune the protocol to recover a previously observed pattern.

The objective is to test generalization, not produce agreement.

## Gate 3b — Qwen3.8-27B generalization **[rev 8]**

Apply the same G1–G14 questions and reporting rules as Gates 2 and 3, with
pre-data model-specific adaptations documented in a separately frozen 27B spec.
**[rev 9]** Use `returnmoe/jlens-adapters`' released Miru J-lens for the
original `Qwen/Qwen3.8-27B`, pinned to the model revision, artifact revision
and SHA-256 in `doc/plan/generalization-plan.md`. Preserve its target-layer
and fitting conventions; the main read layer stays 61. No 3.8 R lens is
assumed: G3/G8/G9's R-lens criteria are not evaluated for 3.8, with explicitly
labelled report-only J analogues allowed in the appendix. No validation refit
or new J/R fit is a prerequisite. Missing/difficult lens work is deferred.
This gate is **authorized and queued**; completion requires saved results,
not merely a job launch.

---

## Gate 4 — Optional O-Lens peek

**Goal:** Explore whether free-form activation verbalization supplies interesting hypotheses about the Pain Axis.

This is **bleeding-edge exploration**, not validation of O-Lens and not evidence of model experience.

Only use existing, sufficiently functional O-Lens checkpoints. Do not train a new Qwen3.5-27B lens for this report.

**[rev 9]** All verbalizer work remains optional appendix material. Existing
3.8 candidates (Gereon NLA and Robbie Thompson oracle-lens pilot) and their
distinct layer-42 contracts are listed in `doc/plan/generalization-plan.md`;
selection does not commit us to running either. Preserve each method's own
contract and matched reconstructor. No verbalizer result is required for a
core battery gate, main claim, or publication.

If practical:

- [ ] Preselect roughly 8–12 examples representing both S2 poles, contrasting semantic families, and relevant self/other or read-slot conditions.
- [ ] Include ordinary/control activations and examples likely to challenge an attractive interpretation.
- [ ] Record raw O-Lens verbalizations and compare them with available R/J readouts.
- [ ] Report conspicuous disagreements and failures, not only compelling phrases.
- [ ] Label the section exploratory throughout.
- [ ] **[rev 3] Cross-scale pairing:** our 4B O-Lens (trained following Agam Bhatia and Camila Blank's published recipe) next to the published Qwen3.6-27B O-Lens (`agu18dec/olens_and_ar`, adapter `olens_s3d_rl600`, with its matched reconstructor `ar_ptag_pooled`), on the same preselected examples. Qualitative comparison only; differences cannot be attributed to scale alone.
- [ ] **[rev 5] Read contract, taken whole:** for the 27B O-Lens, the `concepts_raw` verbalizer prompt, `unit` transform, alpha 16000, and the injection marker found by the documented scan (not hardcoded); for our 4B O-Lens, its own training contract. Capture activations with the adapter disabled. Mixing contracts is a different experiment, not a noisier one.
- [ ] **[rev 5] In-contract layers only:** the 27B O-Lens was trained on layers 20-60 in steps of 4. Read examples at in-band layers (e.g. 32, 44, 60), not at our extraction (62) or family-readout (61) layers. Its reconstructor's FVE rises with depth (0.12 at layer 20, 0.34 at 60); weigh shallow reads accordingly.
- [ ] **[rev 5] No FVE comparison across reconstructors:** FVE is only meaningful on the reconstructor a lens was trained against.
- [ ] **[rev 5] Code availability:** the published reading example imports a `global_workspace` package; confirm it is installable, otherwise reimplement the contract exactly and check it against the card's documented behaviour before reading any example.
- [ ] **[rev 3] Recipe-diff table:** every documented training setting of the published recipe next to what our 4B run actually used, each row marked matched / differs (why) / unknown. Written from the published recipe documentation and our training records and configs, not from memory. Pay particular attention to shared datasets used differently.
- [ ] **[rev 3] Provenance:** source, file hash and training provenance for both O-Lens checkpoints, as for the J/R lenses. **[rev 5]** Pin `olens_s3d_rl600` by file hash: the card notes later RL branches may supersede it.
- [ ] **[rev 3] Lineage:** name the method by its implementation (link); make no claims about its relationship to Anthropic's oracle-lens proposal.

**Gate passes when:** A small, honestly presented panel exists—or the section is omitted.

**Hard stop:** No O-Lens causal validation, checkpoint scaling, prompt optimization for prettier verbalizations, or claims that an oracle description identifies a felt state.

Those questions belong to the next O-Lens project, using Qwen3.5-4B and Qwen3.6-27B.

---

## Gate 5 — Technical report

**Goal:** Produce a self-contained PDF explaining what was done, what was found, and what remains uncertain.

The report skeleton can start alongside the 27B experiments; insert frozen results as they arrive.

Proposed structure:

1. **Introduction and scope** — The source paper, its operational construct, and our extension questions.
2. **Methods** — Models, datasets, direction extraction, intervention conventions, readout methods, semantic vocabulary, controls, and statistical procedures. **[added]** Include lens provenance and the Gate L calibration result.
3. **Reproduction** **[added]** — Gate 2a results on Qwen3-14B base.
4. **Representational findings** — Classification, Base/post-trained comparison, S1/S2 geometry, category structure.
5. **Steering and interpretability** — Both poles, dose effects, R/J findings, lexical and formatting complications.
6. **Experiencer and context** — 1P/3P effects, read-slot dependence, task and persona considerations.
7. **Behavioral exploration** — Buttons, including task failure or null results as applicable.
8. **Cross-model generalization** — Qwen3.5-, Qwen3.6- and Qwen3.8-27B (Gates 2, 3, 3b).
9. **Optional exploratory O-Lens observations** — Clearly separated from validated findings.
10. **Precautions** **[added]** — Steering toward the pain pole and any harm-choice runs: doses, item counts, and the rationale, alongside the source paper's own section.
11. **Discussion, limitations, and future work** — Competing interpretations, provenance, failure modes, and unresolved questions. **[added]** Engage with the authors' `MANUSCRIPT_DISCREPANCIES.md`; cite the pinned source commit.

### Figure priorities

**Core figures:**

- Pain/control separation across checkpoints, with clear distinction between held-out classification and source-category projections.
- Read-slot/experiencer interaction or an equally compelling semantic-geometry result.
- Bidirectional steering, showing semantic changes alongside formatting or degeneration metrics.

**Conditional:** Button-choice figure only if valid; otherwise a concise appendix table.

**Optional:** O-Lens examples in a clearly exploratory appendix.

Each reported result must have a reproducible input, analysis, and artifact location. Figures should be generated from committed result tables, not manually assembled from notebook displays.

### PDF completion criteria

- [ ] All main claims trace to named runs and source data.
- [ ] Exploratory versus confirmatory status is visible.
- [ ] Statistical units, uncertainty, and multiplicity limitations are stated.
- [ ] Model and protocol differences are explicit.
- [ ] Important negative and contradictory results are included.
- [ ] The original paper's contributions and caveats are represented fairly.
- [ ] The conclusions do not require a specific psychological interpretation of S2.
- [ ] Third-party data are cited and used within their licenses (NRC VAD: non-commercial, not redistributed). **[added]**
- [ ] PDF and repository links resolve.

**Gate passes when:** An informed reader could evaluate the extension without reading the research chat history.

---

## Gate 6 — LessWrong: Don't Call It Pain!

**Goal:** Publish a short, engaging introduction directing interested readers to the technical report.

- [ ] **[added]** Share the report and draft with the source paper's authors first; hold publication for their comments.
- [ ] Final text under 1,000 words, ideally 600–850.
- [ ] Dickinson or Pink Floyd epigraph.
- [ ] One paragraph introducing the source research accurately.
- [ ] Explain that this is an extension rather than a full reproduction (with one reproduction check).
- [ ] Two strong, readable figures.
- [ ] Explain why operational representation, causal function, and phenomenal experience are distinct questions.
- [ ] Link the PDF, repository, and source paper.

**Gate passes when:** The article is readable and compelling on its own, without asking the reader to trust claims that are only established in the technical report.

---

## Order of work **[added 2026-10-09; rev 3]**

1. Local GPU (16 GB, sysmem fallback disabled): Gate 0 (writing), then Gate 1 (post-trained buttons). **[rev 7]** Gate 1 done.
2. RunPod A40, in parallel with step 1: Gate L (L1a, L1b, optional L2). **[rev 6]** Done.
3. RunPod A100 80 GB: **[rev 8]** Gate 2a, 14B M7 and R4 done (checkpoints 10–12); 27B benchmark complete. Keep the historical GPU/environment records, not the current container image, as authoritative for the finished runs.
4. **[rev 9] Execution authorized 2026-10-27 on the provisioned A100.** Before any 27B observations, freeze `doc/protocols/gen27b/` and its per-model G1–G14 predictions, unavailable R-lens tests and failure rules; then Gate 2 (Qwen3.5-27B), Gate 3 (Qwen3.6-27B), Gate 3b (Qwen3.8-27B) on A100, with drift checks. Use the released Miru J-lens for the 3.8 appendix; no new fits.
5. Gate 5 skeleton can start during step 4; Gate 4 remains optional; Gate 6 follows author comments.

## Global stopping rules

1. **Negative results complete experiments.** They do not automatically authorize new experiments.
2. **Instrumentation defects justify repairs.** Disappointing observations do not justify retuning.
3. **Freeze before cross-model runs.** Changes after observing results must be versioned and labeled exploratory.
4. **Don't force a noun.** Pain, distress, vitality, self-evaluation, and persona implications remain hypotheses, not required conclusions.
5. **Don't conflate measurement with experience.** Neither activation separation nor successful steering alone establishes felt pain or pleasure.
6. **No compulsory O-Lens work.** The O-Lens causal-validation project begins after this report.
7. **No full-paper reproduction requirement.** This project succeeds by producing novel, carefully characterized observations, not matching the original team's scope.
8. **Publication is the finish line.** New questions go into a follow-up document, not into the active critical path.
9. **[added] Every detour is gated.** A new branch enters this file with a goal, pass criteria, and outcomes before any work on it starts.

## Final handoff

On completion, the repository should contain:

- `doc/FINISH-LINE.md` — this plan and final gate dispositions.
- A technical report source and rendered PDF.
- A short results/claims index linking findings to artifacts.
- Stable analysis code, frozen configurations, and plot source data.
- `doc/writeup/dont-call-it-pain.md` — LW text.
- A short `FUTURE-WORK.md` or equivalent capturing deferred research, especially O-Lens causal validation **[added]** and a reusable lens-refit pipeline with calibration tables across models. **[rev 3]** Also: how the O-Lens recipe relates to Anthropic's oracle-lens proposal.

**Definition of done:** We have contributed a bounded, reproducible, intellectually useful extension of the Pain Axis research, explained its limitations, published the findings, and cleanly handed the remaining questions to future work.

*We don't need to settle what pain is to discover that this representation deserves a better investigation.*
