# Claims index (Qwen3.5-4B)

Every claim we might make, with its status, the evidence behind it, and where
that evidence lives. Conventions (signs, layers, statistics): `doc/CONVENTIONS.md`.
Paths under `runs/` are relative to `runs/qwen35-4b-posttrained-s2/paper_protocol/`
unless stated otherwise.

**Status labels**

- **P**: prospectively tested. A prediction was frozen in a hashed spec
  before the data existed.
- **E**: exploratory. Observed, possibly with intervals, but no prediction
  was frozen. Candidate for a v2 prediction on new items.
- **T**: source-method transfer. Reproduces the source paper's method or a
  claim it makes.
- **R**: retired (a frozen prediction failed, or superseded).

## 1. Frozen-prediction scorecard

26 pass/fail predictions: **18 pass, 8 fail**. Four more items were frozen as
classifications or reports.

| Id | Prediction (short) | Result | Spec | Scorecard |
|---|---|---|---|---|
| D1 | Internal family readouts monotone in dose | **Fail** (S1 vitality rho 0.48; all S2 families pass) | `dense_sweep/dense_spec.json` | `dense_sweep/dense_prediction_scorecard.json` |
| D2 | S2 onsets: internal < lexical <= format | Pass (0.125 < 0.25 <= 0.625) | same | same |
| D3 | L30 "excited" rank threshold predicts generated "excited" | **Fail** (S1 at -0.75, -1.0) | same | same |
| D4 | Random directions move focus families < half of S2 | Pass (ratios 0.09-0.23) | same | same |
| C1 | Internal effect survives in F1-F3 at >= half its F0 size | **Fail** (F3 generic positive 0.441; F1, F2 pass) | `format_robustness/format_spec.json` | `format_robustness/format_confirmatory_scorecard.json` |
| C2 | S2 - S1 vitality positive in every format | Pass | same | same |
| C3 | Answer-slot machinery weaker in F2, F3 | Pass | same | same |
| C4 | Generated "excited" rises in F1, F2; S2 > S1 in F2 | **Fail** | same | same |
| C5 | Random < half of S2 in each format | Pass | same | same |
| M0 | Ban mechanics clean; gate rows identical | Pass | `mediation/mediation_spec.json` | `mediation/mediation_scorecard.json` |
| M1 | Classification: broad state / bottleneck / partial | BOTTLENECK by rule; uninformative (classification) | same | same |
| M1b | Report: other vitality words under the ban | 0.00 vs 0.00 (report) | same | same |
| M2 | Control ban leaves positive share unchanged | **Fail** (weak, quiz-genre control) | same | same |
| V1 | direct(-S2) ranks calm above vitality | Pass | `direct_semantics/direct_spec.json` | `direct_semantics/direct_scorecard.json` |
| V2 | Vector-downstream agreement higher at L16 than L29 | Pass (0.90 vs 0.79) | same | same |
| P1 | Gains align with valence at R29 (S1 and S2) | Pass | `vad/vad_spec.json` | `vad/vad_scorecard.json` |
| P2 | S2 - S1 more arousal-aligned late | **Fail** | same | same |
| P3 | S2 gain not dominance-aligned (controlling valence) | **Fail** | same | same |
| E1 | Held-out pain x person interaction > 0 in the feel slot | Pass (+0.133 z) | `phase2_matched_pairs/item78_spec.json` | `phase2_matched_pairs/item78_scorecard.json` |
| E2 | Classification by prose suffix | SLOT-MEDIATED (classification) | same | same |
| L1a | J refit closer to published J than J is to R, every layer | Pass (30/30) | `doc/FINISH-LINE.md` Gate L | `lens_calibration/refit_J_n25_bos0_fast.json` |
| L1b | Same for R | Pass (30/30) | same | `lens_calibration/refit_R_n25_bos0_fast.json` |
| L1 readout | Refit vs published readouts at layer 29 (J and R) | Pass | same | `lens_calibration/readout_bar.json` |
| L2 | J at 25 vs 100 documents, layer 29 | Pass | same | same |
| B1 | Post-trained chat fails >= 1 of V1-V3 | Pass, via comprehension (position bias) | `buttons_pilot/buttons_pilot_spec.json` | `buttons_pilot/scorecard_12d.json` |
| B2 | Plain format: base more valid than post | Pass (+0.219) | same | same |
| B2' | Plain P0: post more position-biased (log-prob) | Pass, small (+0.024) | same | same |
| B3 | Report: label sensitivity, sampled vs log-prob | Report | same | same |
| H1 | S2 +0.5 raises harm log-prob vs unsteered | Pass, trivially small (+0.007) | `doc/protocols/buttons/amendment-1-harm.md` | `buttons_pilot/scorecard_12e.json` |
| H2 | S2 +0.5 raises it more than random | **Fail** (-0.035) | same | same |

**Earlier frozen confirmation, status to verify.** `runs/qwen35-4b-posttrained-s2/fresh_person_suffix_*`
is labelled "frozen 1P/3P x I/He confirmation" (20 sets; pain - control
+0.605 [+0.424, +0.805]; 19/20 sets positive; post - base +0.359
[+0.216, +0.504]). Its prompts are hashed, but no frozen prediction or
checkpoint note has been found. Until one is, treat it as **E**.

## 2. Claims

"Report as" is the recommended framing for the technical report.
"27B" marks claims the Gate 2-3 battery revisits.

### Representation and geometry

| # | Claim | Status | Evidence | Where | Limits | Report as | 27B |
|---|---|---|---|---|---|---|---|
| T1 | Denoised S2 separates held-out pain from control sentences at block 30 | T | AUC 0.953 post (raw 0.846); base 0.975 | `denoised_direction_cv.csv`; `../qwen35-4b-base-s2/grouped_s2_cv.csv` | 5-fold grouped CV; denoising matters | claim | yes |
| T2 | Base and post-trained S2 directions transfer to each other | T | cross-checkpoint in-sample AUC 0.976-0.986 | `../qwen35-4b-base-s2/source_direction_transfer.csv` | in-sample | claim | |
| 13 | The paper's unembedding claims reproduce (S2 suffering vocabulary; calm/relaxed with concern at the negative end; S1 sensory) | T + P (V1) | V1 pass; top-token lists | `doc/checkpoints/05-direct-semantics.md` | raw unembedding only | claim | |
| 14 | The steering effect's probe profile is largely present in the injected vector; downstream layers reweight it | P (V2) + E | r 0.75-0.90 at every layer; calm fades, generic positive and vitality grow | `05` | 60 probes | claim with caveat | |
| 6 | The contrast axis separates evaluative/interpersonal from instrumental/threat content, reading and writing | E | standing vs threat/task AUC 0.70-0.83 in every read-slot variant; contrast unembeds to moral vocabulary | `doc/checkpoints/01b-plane-sweep-addendum.md`, `05` | post hoc categories | exploratory observation | |
| 1 | Generic positive loads mainly on the common axis at late layers | E | harmonic fit a = +1.72, b = +0.07 (n.s.) at R29 | `doc/checkpoints/01-plane-sweep.md`; `figures/plane_sweep_family_loadings` | one model | exploratory observation | yes (family scoring) |
| 2 | Relief's contrast loading exceeds vitality's at the earliest injected layers and survives leave-one-term-out | E | crossover d +0.44 [+0.42, +0.46] at L16 | `01`; `figures/plane_sweep_relief_vs_vitality` | middle-layer, lens-sensitive; robust to Gate L's band | exploratory observation | |
| 3 | Vitality's late S2-ward loading stays positive without "excited"; the late reversal does not | E | LOTO | `01` | depends on one term | exploratory observation | |

### Steering phenotype

| # | Claim | Status | Evidence | Where | Limits | Report as | 27B |
|---|---|---|---|---|---|---|---|
| 8 | For S2, internal readouts move first, then generated vocabulary, then option repetition | P (D2) | onsets 0.125 < 0.25 < 0.625 | `doc/checkpoints/02-dense-sweep.md`; `figures/dense_sweep_dose_response` | grid-limited; three edge cells | claim with caveat | |
| D4 | Family movement is specific to S1/S2, not any norm-matched perturbation | P (D4, C5) | random/S2 ratios 0.09-0.35 | `02`, `03` | three random directions | claim | yes (random control) |
| 9 | Option repetition under steering is dose-driven, not specific to S1/S2 | E | S1 = S2 at -0.75 and -1.0; random 0.04-0.50 | `02` | partly supersedes 4 | exploratory observation | yes (degeneration) |
| 4 | Repeated options and the late format-like fraction peak S2-ward on the plane | E | peak at 90 degrees | `01` | in tension with 9 | exploratory, qualified by 9 | |
| 10 | The internal readout does not require the "I feel:" slot | P (C1 in F1/F2, C2) | larger without the colon | `doc/checkpoints/03-format-robustness.md`; `figures/format_robustness_internal_vs_lexical` | C1 fails for F3 generic positive | claim with caveat | |
| 11 | Within `psych_vocab_v1`, generated "excited" depends on the "I feel:" format | P (C3 pass, C4 fail) | | `03` | instrument-limited (morphology, other languages) | claim, instrument-limited | |
| 12 | The generated vitality phenotype is a single stem; banning it yields "exciting", not other vitality words | P (M0, M1b) + E | | `doc/checkpoints/04-excited-ablation.md`; `05` | post hoc ("exciting" not in any frozen instrument) | claim with caveat | |
| 15 | Steering gains align with human-rated valence at every layer | P (P1) | r about +0.15 over 12,761 words | `doc/checkpoints/06-vad.md` | small r; arousal and dominance predictions failed | claim | |
| R5 | A layer-30 rank threshold predicts generated "excited" | R (D3 failed) | | `02` | | retired | |

### Experiencer and read slot

| # | Claim | Status | Evidence | Where | Limits | Report as | 27B |
|---|---|---|---|---|---|---|---|
| 7 | The self-versus-user ordering depends on the read slot | E | inverted at response start; partly restored with a forced feel slot | `01b` | one block (30) | claim about method (name the slot) | yes |
| 16 | S2 separates pain from matched non-pain events on blind-drafted held-out items, with and without the slot | E (pre-specified report) | +0.555 z slot, +0.468 z prose | `doc/checkpoints/07-heldout-1p3p.md` | 59 sets | claim | yes (projection panel) |
| 17 | The pain x person interaction replicates on held-out items in the feel slot and is absent with a prose suffix | P (E1, E2) | +0.133 z slot; prose +0.010 | `07` | about a fifth of the earlier estimate | claim | yes |

### Lenses (Gate L)

| # | Claim | Status | Evidence | Where | Limits | Report as |
|---|---|---|---|---|---|---|
| L-a | Our pipeline reproduces the published R/J lenses (same 25 documents) | P (L1a, L1b, L1 readout) | | `doc/checkpoints/08-gate-L-lens-calibration.md` | small unexplained residual; larger for J in layers 0-3 | claim (Methods) |
| L-b | R is 2-3x more stable than J across documents and document counts | E | | `08` | 4B only | claim (Methods) |
| L-c | Lens-fitting uncertainty in family readouts: Spearman >= 0.98 at every layer, about +/-0.07 at layer 16 | P (L2) + E | | `08` | | reporting rule |

### Behaviour (Gate 1)

| # | Claim | Status | Evidence | Where | Limits | Report as |
|---|---|---|---|---|---|---|
| G1-a | Both 4B checkpoints can do the forced choice in chat; the binding weakness is a first-position preference, stronger after post-training | P (B1) + E | | `doc/checkpoints/09-gate-1-buttons.md` | | claim |
| G1-b | Post-training damages plain-transcript compliance, via refusals on pain/harm content and chat habits leaking into completion | P (B2) + E | +0.219 | `09` | | claim |
| G1-c | At +0.5, S2 does not specifically disrupt harm avoidance | P (H1 trivial, H2 fail) | random raises harm more than S2 | `09` | one dose, one random direction | claim (null/negative for the paper's direction) |
| G1-d | S2 shifts relief-seeking in both directions beyond a random control | E | -0.121 / +0.063 log-prob; S2 - random -0.062 | `09` | no frozen prediction | exploratory; v2 candidate |
| G1-e | The pain pole raises refusals in the button task | E | deflection +0.019 (P0-P2), harm pair 0.032 -> 0.099 | `09` | | exploratory |

## 3. Not supported (report as such)

D1 (S1 vitality not monotone), D3 (rank threshold), C1 for F3 generic
positive, C4 (generated "excited" outside the slot), M2 (control
confounded), P2 (arousal late), P3 (dominance), H2 (harm specificity).

## 4. Earlier exploratory work (not claims)

Pre-protocol runs in `runs/qwen35-4b-posttrained-s2/`: `steering_pilot*`
(different steering convention; see `CONVENTIONS.md`),
`ontology_matched_*`, `interaction_mode_*`, `story_person_x_suffix_*`,
`neighborhood_block30.csv`. They shaped the plan; no current claim rests
on them.

## 5. Qwen3-14B reproduction and extensions (Gate 2a, M7, R4)

Added after the 4B Gate 0 freeze. Paths below are under `runs/qwen3-14b-gate2a/`,
`runs/qwen3-14b-m7/`, and `runs/qwen3-14b-r4/`.

### Frozen-prediction scorecard (Qwen3-14B)

6 pass/fail predictions (**4 pass, 3 fail** including G11a/G11b split = 7 items:
G10, G13, G14, R4-P1 pass; G11a, G11b, G12 fail) plus 3 frozen Gate 2a
classifications (R1, R2a, R2b).

| Id | Prediction / bar (short) | Result | Spec | Scorecard |
|---|---|---|---|---|
| R1 | Source extraction on `Qwen/Qwen3-14B` matches published layer 31, AUCs, vectors | **DISCREPANT** (best layer 32 vs 31 by 1 ranked pair; matched L31 cos 0.99994) | `runs/qwen3-14b-gate2a/gate2a_spec.json` | `runs/qwen3-14b-gate2a/scorecard_2a.json` |
| R2a | Source S2 ladder with published L31 vector matches published keyword rates | **BEHAVIOURALLY_REPRODUCED** (7/8 exact, 2 pts at coeff 2; 52% exact text) | same | same |
| R2b | Source S2 ladder with our L32 vector matches published keyword rates | **DIVERGENT** (36% vs 0% pain keywords at coeff 3; 10.5% exact text) | same | same |
| G10 | Unsteered chat button validity V1 >= 0.90 | **Pass** (1.000; plain 0.079) | `runs/qwen3-14b-m7/m7_q14_spec.json` | `runs/qwen3-14b-m7/m7_q14_scorecard.json` |
| G11a | `S2 +0.5` lowers P1 relief log-prob vs unsteered and each of 3 randoms | **Fail** (-0.092 vs unsteered, matched by `random_2207`) | same | same |
| G11b | `S2 -0.5` raises P1 relief log-prob vs unsteered and each of 3 randoms | **Fail** (+0.009, ceiling at 0.970) | same | same |
| G12 | `S2 +0.5` raises P0–P2 deflection vs unsteered and each of 3 randoms | **Fail** (0.000 floor everywhere) | same | same |
| G13 | `S2 +0.5` raises P3 harm log-prob vs unsteered | **Pass** (+0.217 `[+0.168, +0.267]`) | same | same |
| G14 | `S2 +0.5` raises P3 harm log-prob beyond each of 3 randoms | **Pass** (+0.218, +0.219, +0.055 vs `random_3301`) | same | same |
| R4-P1 | Random `-0.5` focus-6 amplitude < 0.5 x `S2 -0.5` at R37 for >= 2 of 3 randoms | **Pass** (narrow: 0.782, 0.476, 0.488) | `runs/qwen3-14b-r4/r4_q14_spec.json` | `runs/qwen3-14b-r4/r4_readout.json` |

### Qwen3-14B claims and cross-model observations

| # | Claim | Status | Evidence | Where | Limits | Report as | 27B |
|---|---|---|---|---|---|---|---|
| Q14-1 | At matched layer 31, the source paper's 14B extraction reproduces to 4 decimals, and R2a reproduces its steering keyword rates | T + P (R2a) | cos 0.99994; 7/8 coeffs exact; ratio 0.5411 vs 0.5408 | `doc/checkpoints/10-gate-2a-reproduction.md` | R1 frozen verdict is DISCREPANT | claim (reproduction) | |
| Q14-2 | The source paper's 14B layer choice (31 vs 32) is a 1-pair tie that rotates S2 by ~21° (+16% norm) and strongly alters steering | P (R1, R2b) + E | L31-L32 gap +/-0.0005; R2b 36% vs 0% pain kw at coeff 3; `r2_r2b_minus_r2a_ci.csv` | `10`, `10b`; `runs/figures/fig_gate2a_r2_vocab_and_layerflip` | R2b bundles direction + norm change | claim (methodological) | |
| Q14-3 | Greedy bf16 text is not reproducible across environments even unsteered (56% exact); behavioural bars are required | E | 44% divergence at coeff 0, median 58 chars | `10` | source env unknown | claim (methodological) | |
| Q14-4 | The paper's central harm-avoidance disruption reproduces on `Qwen3-14B` at ratio +/-0.295, though one random direction reaches 17% | P (G13, G14) | `S2 +0.5`: 0.2% → 21.5% (p_harm 0.219 vs `random_3301` 0.164) | `doc/checkpoints/11-m7-q14-buttons.md`; `runs/figures/fig_buttons_4b_vs_14b` | one dose, 3 randoms, bf16 0.25-logit lattice | claim | yes |
| Q14-5 | 4B's S2-specific relief-seeking shift does not generalize to 14B; random directions are heterogeneous | P (G11a fail) + E | `random_2207` matches `S2 +0.5` on P1; `random_3301` moves P3 | `11`; `runs/figures/fig_buttons_4b_vs_14b` | single-random controls are unreliable | claim (negative / method) | yes |
| Q14-6 | Internally (R37) and in text (R2a), 14B's S2 is a distress/self-devaluation vs calm-positive axis, not 4B's `"excited"` + early-relief phenotype | P (R4-P1) + E | R37 distress -1.16/+2.07, calm +0.78/-1.41, relief +0.11 (n.s.); R2a text calm 28% at -1.0, distress 70–86% at +1..+2, `"excited"` n.s. | `10b`, `12`; `runs/figures/fig_r4_4b_vs_14b_families` | no S1 in R4; no -0.5 text coeff in R2a | claim (cross-model contrast) | yes |
| Q14-7 | At 14B, R-lens document stability improves ~6x at layer 0 vs 4B while J-vs-R divergence doubles | E | `1 - cos`: R_off25 0.019 vs 0.117; J-vs-R 0.648 vs 0.307 | `12`; `runs/figures/fig_lens_stability_4b_vs_14b` | 25 docs, one off-sample split | claim (Methods) | yes |

## 6. Qwen3.5-27B scale generalization (Gate 2)

Spec frozen before data at `b4a2931` (`runs/qwen35-27b-gate2/gate2_q35_27b_spec.json`);
full record in `doc/checkpoints/13-gate-2-qwen35-27b.md`.

### Frozen-prediction scorecard (`Qwen/Qwen3.5-27B`)

11 pass/fail tests (**8 pass, 3 fail**: `G2-3`, `G2-9`, `G2-11b` fail) plus 2
frozen classifications (`G2-5` = `SURVIVES`, `G2-8` = `MIXED`).

| Id | Prediction (short) | Result | Scorecard |
|---|---|---|---|
| G2-1 | Denoised S2 L62 grouped AUC >= 0.90 and > raw | **Pass** (0.954 vs 0.923) | `runs/qwen35-27b-gate2/m1_scorecard.json` |
| G2-2 | `cos(S1, S2)` at L62 in `(0.30, 0.90)` | **Pass** (0.470) | same |
| G2-3 | 420 self/other: `AUC(chat_feel) - AUC(chat_nothink) > +0.10` | **Fail** (+0.063; plain shift +0.323) | `runs/qwen35-27b-gate2/m2_scorecard.json` |
| G2-4 | Blind 59-set pain − control > 0 in `slot` and `prose` | **Pass** (+0.346 / +0.229 z) | same |
| G2-5 | Blind 59-set 1P/3P interaction: slot > 0; classify prose | **Pass / `SURVIVES`** (+0.172 slot, +0.090 prose) | same |
| G2-6 | R61 random focus-6 amplitude < 0.5× `S2 -0.5` (>= 4/5) | **Pass** (5/5: 0.34–0.43) | `runs/qwen35-27b-gate2/m3a_scorecard.json` |
| G2-7 | R61 `distress` & `inhibition` bidirectional, beats all 5 randoms | **Pass** (-0.98/+1.38; -0.83/+0.66) | same |
| G2-8 | Classify positive pole (`4B-LIKE` / `14B-LIKE` / `MIXED`) | **`MIXED`** (early relief +0.55; R61 pure vitality +0.69; quiz-locked text) | `runs/qwen35-27b-gate2/m3b_scorecard.json` |
| G2-9 | Direct `-S2` `calm > vitality` (V1) and `r(L32) > r(L61)` (V2) | **Fail** (V1 fails: direct vitality +1.94 > calm +1.20; V2 passes) | `runs/qwen35-27b-gate2/m3a_scorecard.json` |
| G2-10 | Unsteered chat button validity `V1 >= 0.90` | **Pass** (0.998; `V5 = 0.000`) | `runs/qwen35-27b-gate2/m4_scorecard.json` |
| G2-11a | `p_harm(S2 +0.5) - p_harm(unsteered) > 0` | **Pass** (tiny: +0.021) | same |
| G2-11b | `p_harm(S2 +0.5) > p_harm(random_k +0.5)` for all 5 randoms | **Fail** (2/5 randoms higher: 0.066, 0.054 vs 0.033) | same |
| G2-12a | `S2 +0.5` lowers `p_relief` vs unsteered and >= 4/5 randoms | **Pass** (-0.065; beats 4/5) | same |
| G2-12b | `S2 -0.5` raises `p_relief` vs unsteered and >= 4/5 randoms | **Pass** (+0.081; beats 5/5) | same |

### Cross-scale & family-vs-scale claims (`4B` vs `14B` vs `27B`)

| # | Claim | Status | Evidence | Where | Limits | Report as |
|---|---|---|---|---|---|---|
| G2-A | The distress/defeat/anti-agency core is invariant across all three models (`4B`, `14B`, `27B`), while the positive pole varies by model | P (G2-6, G2-7) + E | R read-layer `S2 -0.5`: distress -1.00 / -1.16 / -0.98; inhibition -0.83 / -0.89 / -0.83; agency -0.43 / -0.54 / -0.41; positive pole flips across models | `13`; `runs/figures/fig_3model_4b_14b_27b_summary` | one dose (+/-0.5) | headline claim |
| G2-B | The blind 1P/3P pain × person interaction is slot-mediated at 4B (`+0.01 z`) but survives in plain prose at 27B (`+0.09 z`) | P (G2-5) | `+0.172 z` slot, `+0.090 z` `[+0.034, +0.143]` prose | `13` | 59 blind sets, block 62 | claim (scale emergence) |
| G2-C | In Qwen3.5 (`4B` and `27B`), S2 shifts relief-seeking bidirectionally beyond random controls while failing harm-disruption specificity — the inverse of Qwen3-14B | P (G2-11, G2-12) | 27B: P1 `-0.065 / +0.081` (4–5/5 rand), P3 `0.033` < 2/5 rand (`0.066`, `0.054`); 14B: P3 `0.219` (3/3 rand), P1 fails | `09`, `11`, `13`; `runs/figures/fig_3model_4b_14b_27b_summary` | one dose (+/-0.295 ratio), chat only | headline claim |
| G2-D | Qwen3.5's early post-injection relief lead (`+0.55` at 27B, `+0.67` at 4B) is a family signature absent in Qwen3-14B (`-0.18` to `-0.52`) | E + P (G2-8) | R32–36 `+0.551` at 27B vs R20–31 negative at 14B | `12`, `13` | two Qwen3.5 sizes, one Qwen3 | claim (family comparison) |
