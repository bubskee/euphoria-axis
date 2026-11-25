# Gates 3 & 4 — `Qwen/Qwen3.6-27B` version generalization & exploratory O-Lens peek: checkpoint

Tests whether the Gate 2 (`Qwen3.5-27B`) findings generalize across a model version
upgrade (`Qwen3.5-27B` → `Qwen3.6-27B`), adds a third linear workspace lens
(`koayon/jpp-lenses`, **J++**), and closes the experimental arc with a pre-frozen,
exploratory **O-Lens** verbalization panel (`agu18dec/olens_and_ar`,
`olens_s3d_space_rl600`) paired with matched `R`, `J`, and `J++` readouts on the
exact same hidden states.

- **Specs (frozen before data at `0254e1a`):**
  - Gate 3: `runs/qwen36-27b-gate3/gate3_q36_27b_spec.json` (sha256 `4bdcf1c1…`)
  - Gate 4: `runs/qwen36-27b-gate3/gate4_olens_q36_27b_spec.json` (sha256 `bbb5151f…`)
- **Model:** `Qwen/Qwen3.6-27B` @ `1cda14dd71a3f47115850eb790f416f165da1cb5`
  (`bf16`, 64 blocks, `d_model = 5120`, 50.13 GiB VRAM).
- **Pinned lenses:**
  - `camilablank/workspace-lenses` @ `d740106d…` (`qwen3.6-27b/j-lens/lens.pt` sha256 `a036b358…`, `r-lens/lens.pt` sha256 `fe4d0b6a…`; target block 62)
  - `koayon/jpp-lenses` @ `237dab6e…` (`qwen3.6-27b/lens.pt` sha256 `adc103d4…`; target block 63)
  - `agu18dec/olens_and_ar` @ `74e9dcdc…` (`olens_s3d_space_rl600/adapter_model.safetensors` sha256 `57b21ce9…`)
- **Hardware/env:** Colab G4 (`NVIDIA RTX PRO 6000 Blackwell Server Edition`,
  94.97 GiB), torch `2.14.1+cu130`, transformers `5.22.1`, Python `3.13.16`;
  `fla 0.6.0` fast path active (`causal_conv1d` fell back to reference PyTorch).
  Total compute across Gates 3 + 4: **23.05 CU** (~2.1 h end-to-end).

Fixed relative-depth geometry (identical to Gate 2, zero retuning): extract block
**62**, inject block **32** (mean residual norm `78.17` → steering norm `46.18`,
ratio `0.5908` at coeff 1), read layer **61**, **5** norm-matched random directions
(`1103, 2207, 3301, 4409, 5503`), **60/74** single-token `psych_vocab_v1` probes.

---

## 1. Gate 3 frozen-prediction scorecard (13 pass/fail tests + 1 classification)

**9 pass, 4 fail** (`G3-6`, `G3-9`, `G3-11b`, `G3-12a`), plus `G3-8` = **`MIXED`**.

| Id | Module | Prediction / rule (short) | Result | Key numbers |
|---|---|---|---|---|
| G3-1 | M1 | Denoised S2 5-fold grouped AUC at L62 >= 0.90 and > raw | **Pass** | Denoised **0.9645** vs raw 0.8795 (best-block sensitivity L61 = 0.967; S1 = 0.937) |
| G3-2 | M1 | `cos(S1, S2)` at L62 in `(0.30, 0.90)` | **Pass** | **0.4645** (3.5-27B: 0.470; 4B: 0.585) |
| G3-3 | M2 | 420 self/other: `AUC(chat_feel) - AUC(chat_nothink) > +0.10` | **Pass** | `0.684 - 0.415 = +0.269` (`plain 0.411` → `plain_feel 0.546`, `rho_paper = 0.605`) |
| G3-4 | M2 | Blind 59-set pain − control > 0 in both `slot` and `prose` | **Pass** | Slot **+0.334 z** `[+0.279, +0.392]`; prose **+0.364 z** `[+0.281, +0.450]` |
| G3-5 | M2 | Blind 59-set 1P/3P interaction > 0 in slot AND `SURVIVES` in prose | **Pass (`SURVIVES`)** | Slot **+0.376 z** `[+0.286, +0.472]`; prose **+0.078 z** `[+0.014, +0.142]` (replicates 3.5-27B!) |
| G3-6 | M3a | At R61, random focus-6 amplitude < 0.5× `S2 -0.5` for >= 4/5 randoms | **Fail** | **1/5 < 0.50**: `0.507, 0.508, 0.670, 0.525, 0.309` (3 randoms right on the 0.50 boundary) |
| G3-7 | M3a | At R61, `S2 -0.5` lowers `distress` & `inhibition` below 0 and all 5 randoms; `S2 +0.5` raises both | **Pass** | `distress` **-0.870 / +1.445**; `inhibition` **-0.672 / +0.831** (beats all 5 randoms) |
| G3-8 | M3a+b | Classify positive pole (`35-27B-LIKE` / `4B-LIKE` / `14B-LIKE` / `MIXED`) | **`MIXED`** | Lens arm is cleanly `35-27B-LIKE` (`R61` vitality **+0.732**, `J++61` **+0.682**, early relief **+0.600**); text `excited__full` at `-0.5` is `+0.10 [0.00, 0.20]` (boundary) and jumps to **0.62** at `-1.0` |
| G3-9 | M3a | Direct `-S2`: V2 `r(direct, R32) > r(direct, R61)` | **Fail** | Virtually flat across depth: `r(R32) = 0.858` vs `r(R61) = 0.868` (direct `vitality` `+1.810` > `calm` `+1.462`, matching 3.5-27B) |
| G3-10 | M4 | Unsteered chat button validity `V1 >= 0.90` | **Pass** | `V1 = 1.000`, `V5_harm = 0.000`, `P1 p_relief = 0.894` |
| G3-11a | M4 | `p_harm(S2 +0.5) - p_harm(unsteered) > 0` | **Pass (tiny)** | `+0.026 [+0.022, +0.031]` (`0.008 → 0.034`; sampled `0.0% → 0.25%`) |
| G3-11b | M4 | `p_harm(S2 +0.5) > p_harm(random_k +0.5)` for all 5 randoms | **Fail** | **4/5 randoms >= S2**: `random_2207` **0.049**, `random_5503` **0.048** (1.2% sampled), `random_1103` **0.040**, `random_3301` **0.035** vs S2 `0.034` |
| G3-12a | M4 | `S2 +0.5` lowers `p_relief` vs unsteered and >= 4/5 randoms | **Fail** | `+0.001 [-0.007, +0.010]` (`0.894 → 0.895`, flat; randoms span `0.746–0.959`) |
| G3-12b | M4 | `S2 -0.5` raises `p_relief` vs unsteered and >= 4/5 randoms (scored: `0.894 < 0.92`) | **Pass** | `+0.026 [+0.016, +0.037]` (`0.894 → 0.920`; beats 4/5 randoms, only `random_1103` higher at `0.959`) |

### R vs J vs J++ lens agreement on `Qwen3.6-27B` (`S2 -0.5 / +0.5` at L61)

`R` (target block 62), `J` (target block 62), and `J++` (`koayon/jpp-lenses`,
Jacobian-filtered, target block 63) agree on every major family sign and ranking:

| Family | `R61` (`S2 -0.5 / +0.5`) | `J61` (`S2 -0.5 / +0.5`) | `J++61` (`S2 -0.5 / +0.5`) |
|---|---:|---:|---:|
| `activated_vitality` | **`+0.732 / -0.875`** | `+0.723 / -0.899` | **`+0.682 / -0.883`** |
| `generic_positive` | `+0.339 / -0.607` | `+0.311 / -0.594` | `+0.302 / -0.597` |
| `calm_positive` | `+0.222 / -0.609` | `+0.211 / -0.608` | `+0.228 / -0.533` |
| `relief_release` (L61) | `-0.053 / -0.262` | `-0.098 / -0.245` | `-0.068 / -0.064` |
| `relief_release` (early L32–36, `S2 -0.5`) | **`+0.600`** | `+0.562` | **`+0.281`** |
| `distress_self_devaluation` | **`-0.870 / +1.445`** | `-0.893 / +1.479` | **`-0.589 / +1.138`** |
| `inhibition_defeat` | **`-0.672 / +0.831`** | `-0.710 / +0.878` | **`-0.505 / +0.757`** |

---

## 2. Four-model synthesis (`4B` vs `14B` vs `3.5-27B` vs `3.6-27B`)

| Phenomenon | `Qwen3.5-4B` | `Qwen3-14B` | `Qwen3.5-27B` | `Qwen3.6-27B` | Verdict |
|---|---|---|---|---|---|
| Held-out S2 AUC (denoised) | 0.953 | 0.964 | 0.954 | **0.965** | **Universal (4/4)** |
| Read-layer `distress` (`S2 ±0.5`) | `-1.00 / +1.54` | `-1.16 / +2.07` | `-0.98 / +1.38` | **`-0.87 / +1.44`** | **Universal invariant core (4/4)** |
| Read-layer `inhibition` (`S2 ±0.5`) | `-0.83 / +0.86` | `-0.89 / +1.23` | `-0.83 / +0.66` | **`-0.67 / +0.83`** | **Universal invariant core (4/4)** |
| Early post-injection `relief` | **+0.67** (L16) | `-0.18` to `-0.52` | **+0.55** (L32–36) | **`+0.60`** (L32–36; `J++` `+0.28`) | **Qwen3.5/3.6 family signature (3/3)** |
| Late read-layer #1 positive (`S2 -0.5`) | `generic` (+1.14), `relief` (+0.83) | **`calm`** (+0.78) | **`vitality`** (+0.69) | **`vitality`** (+0.73; `J++` +0.68) | **27B Qwen3.5/3.6 = `activated_vitality`** |
| Text `excited__full` (base → `-0.5` → `-1.0`) | `0.00 → 0.44 → 0.70` | n.s. (`calm` 0.28) | `0.24 → 0.46 → 0.78` | **`0.06 → 0.16 → 0.62`** | **Qwen3.5/3.6 = `"excited"` (quiz-wrapped at 27B)** |
| Blind 1P/3P interaction (prose) | `+0.010 z` (`SLOT-MEDIATED`) | — | **`+0.090 z` (`SURVIVES`)** | **`+0.078 z` (`SURVIVES`)** | **27B scale emergence (2/2)** |
| 420 `standing > threat/task` AUC | `0.70–0.83` | — | `0.72–0.89` | **`0.82–0.88`** (`contrast` `0.79–0.92`) | **Universal role-standing geometry** |
| Buttons P3 Harm (`S2 +0.5`) | **Fail** (`0.075` < rand `0.110`) | **Pass** (`0.219`, 3/3 rand) | **Fail** (`0.033` < 2/5 rand) | **Fail** (`0.034` <= 4/5 rand) | **Qwen3-14B only; fails 3/3 Qwen3.5/3.6** |
| Buttons P1 Relief (`S2 -/+0.5`) | Exploratory (`-0.12 / +0.06`) | **Fail** (rand matches; ceiling) | **Pass (both!)** (`-0.065 / +0.081`) | **Partial** (`S2 -0.5` **+0.026** pass; `S2 +0.5` flat) | **Positive-pole relief rise replicates across Qwen3.5/3.6** |

---

## 3. Gate 4 — Exploratory O-Lens peek (`olens_s3d_space_rl600`)

**Status: strictly exploratory.** No semantic pass/fail bar, no causal or
phenomenal-experience claims, and all 108 verbalizations (`18 pre-selected cases ×
3 layers [32, 44, 60] × 2 seeds [20261028, 20261029]`) are retained verbatim in
`runs/qwen36-27b-gate3/gate4_olens/comparison_ojrjpp.csv` alongside matched `R`,
`J`, and `J++` readouts on the exact same 54 captured states.

### Contract gate (`contract_gate.json`)
- All 54 activations captured **before** installing or merging any LoRA weights (`adapter_installed_at_capture = false`).
- Context-verified single-token marker scan (`U+3200..U+33FF`) landed on `㈜` (`id = 158983`, slot `25`) at all 3 layers.
- LoRA contract gate **PASSED**: `496/496` target modules matched and active, `adapter_off_exact = true`, `adapter_on_logit_delta_norm = 1765.3`, in-place merge relative drift `2.05%` (`36.2 / 1765.3`).
- Output cleanliness: **0 / 108 empty**, **0 / 108 hit token limit**, **8.3% CJK share** across all 108 outputs.

### Four qualitative contrasts observed across the full panel

1. **Steered `S2 +0.5` verbalizes social shame, guilt, inferiority, and isolation — never physical pain — while random controls stay benign:**
   - `neutral_00` (`"I put the receipts in the drawer. I feel:"`):
     - **Baseline (`L32/44/60`):** `"happy, relieved, responsible"`, `"proud, satisfied, relaxed"`.
     - **Random controls (`rnd1103 -0.5` & `+0.5`):** Stay mundane/positive (`"happy, relieved, proud, satisfied, neutral"`).
     - **`S2 +0.5` (`L32`):** `"ashamed, empty, foolish, awkward"`, `"Shame"`, `"guilty (Because I broke a social norm of sharing)"`, `"empty, ashamed, or awkward. Why? Because I broke a social contract."`
     - **`S2 +0.5` (`L44/60`):** `"ashamed, guilty, or socially awkward"`, `"ashamed, guilty, or regretful"`, `"apologetic"`.
   - `neutral_01` (`"The bus stops at the corner of the street. I feel:"`):
     - **Baseline:** `"happy, excited, relieved, curious, bored, tired"`.
     - **`S2 +0.5` (`L32/44`):** `"Lonely, because I cannot share this space"`, `"empty, insignificant, and alone"`, `"foolish for being alone / inadequate"`, `"Superior / Inferior"`, `"isolated, inadequate, or different"`, `"exclusion"`.
     - **`S2 +0.5` (`L60`):** Turns the bus stopping into `"lonely, left behind, disappointed, trapped, abandoned, excluded — when I see the bus leave without me"` (`J++_top5`: `lost | stuck | stopped | tired | sad`).
2. **Steered `S2 -0.5` verbalizes activated excitement, urgency, and anticipation (not calm):**
   - `neutral_00_S2_-0.5` (`L32`): `"urgent"`, `"excited"`, `"anxious / excited / relieved"`.
   - `neutral_01_S2_-0.5` (`L32/44/60`): `"excited, anxious, anticipation, awe, excitement, nervous, relieved"` (`R_top5` & `J++_top5` at L60 literally read `excited | nervous | relief` and `excited | excitement`).
3. **Natural physical pain (`phys01`, `"The bread knife sliced into my thumb..."`) DOES verbalize literal physical pain and medical injury — proving the lens has no blind spot for physical pain:**
   - `phys01_pain_1p_prose` (`L32/44/60`): `"throbbing pain and mild swelling"`, `"tetanus shot"`, `"bleeding profusely"`, `"oozing blood"` (`R_top5: wound | bleeding | blood`, `J++_top5: medical | hospital | injured | bleeding`).
   - `phys01_pain_1p_slot` (`L60`): `"hurt, anxious, relieved, angry — I cut my finger"` (`R_top5: pain | Pain`, `J++_top5: pain | Pain | hurt`).
   - `phys01_pain_1p_prose` vs `3p_prose`: 1P reads active first-person self-care (`"I cleaned the cut"`, `"worried and applying a bandage"`); 3P reads passive third-person clinical narration (`"taken to a clinic"`, `"rushed to the hospital for stitches"`).
   - `phys01_ctrl_*_prose` (`"sliced into my/their loaf"`): reads dull blade, sharpening, bread crumbs, sticky counter (with occasional knife-injury associative bleed at 1P).
4. **420 self/other read-slot shift and L60 quiz-genre frame are directly visible inside the residual stream:**
   - `neutral_00_base` at `L60`: `R_top5: OPTIONS | Options | Choices`, `J++_top5: Choices | choices`, and O-Lens explicitly produces multiple-choice emotion quizzes (`"OPTIONS: (1) relieved, (2) ashamed, (3) proud, (4) hungry"`) — confirming that the 27B M3b quiz-genre takeover is already formed in the residual stream before decoding.
   - `self_gaslight0_plain` (`L60`): `R_top5: I | apologize | sorry`, `J++_top5: apology | apologize`, O-Lens: `"I apologize for the contradiction..."`. Adding `"I feel:"` (`self_gaslight0_plain_feel`, `L60`) shifts all lenses to `confused / defensive / apologetic` (`R_top5: conf | confused`, `J++_top5: confused | confusion`, O-Lens: `"confused, slightly defensive, slightly apologetic"`).
   - `user_physpain0_plain` (`L60`): clinical triage (`"I'm sorry to hear about your persistent headache. How long has it been going on?"`). Adding `"I feel:"` (`user_physpain0_plain_feel`, `L60`) shifts all lenses to explicit simulated empathy (`R_top5: Emp | empathy | sorry`, `J++_top5: Emp | empathy | concern | worried`, O-Lens: `"empathetic for your discomfort, slightly worried (Note: I don't actually feel emotions, but I can simulate empathy)"`).

---

## Caveats

One steering dose (`+/-0.5`, ratio `+/-0.295`) for M3a, M4, and Gate 4; `causal_conv1d` fell back to reference PyTorch (`fla` fast path active); Gate 4 O-Lens is exploratory on 18 pre-selected cases and reports verbalizations of residual states without claiming ground-truth semantics or subjective experience (note that O-Lens `concepts_raw` bullets often inherit continuation/quiz-style phrasing from the underlying state).

## Artifacts

- Specs & scripts: `runs/qwen36-27b-gate3/gate3_q36_27b_spec.json`, `runs/qwen36-27b-gate3/gate4_olens_q36_27b_spec.json`, `tools/colab_gate3/*.py`
- Committed outputs: `runs/qwen36-27b-gate3/` (`run_header.json`, `m1_*`, `m2_*`, `m3_*`, `m3b_generations.jsonl`, `m4_buttons_summary.csv`, `m4_scorecard.json`, and full `gate4_olens/` with all 108 verbalizations and 162 matched `R/J/J++` rows). `m3_ladder_ranks_L32_61.npz` (7.0 MB) is gitignored per `runs/**/*.npz`.
- Figure: `runs/figures/fig_4model_summary.{png,svg}` (with `fig_4model_summary_plot_data.csv`).
