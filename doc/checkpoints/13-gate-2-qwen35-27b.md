# Gate 2 — `Qwen/Qwen3.5-27B` generalization: checkpoint

Tests whether the Qwen3.5-4B and Qwen3-14B findings survive a 7× scale jump
within the Qwen3.5 family, disentangling **model family** (Qwen3.5 vs Qwen3)
from **parameter scale** (4B vs 14B/27B).

Spec: `runs/qwen35-27b-gate2/gate2_q35_27b_spec.json` (committed at `b4a2931`
before any 27B data; sha256 `27bb02ec…`). Model: `Qwen/Qwen3.5-27B` @
`81a287ca484a5e71148f5beb7f367bb5286d70f9` (`bf16`, 64 blocks, `d_model = 5120`,
50.13 GiB VRAM). Published lenses: `camilablank/workspace-lenses` @
`d740106d1e0f95456dc8718fba2895e9c8ffd6ef` (`qwen3.5-27b/j-lens/lens.pt`
sha256 `308a4984…`, `r-lens/lens.pt` sha256 `3584be12…`).
Hardware/env: Colab G4 (`NVIDIA RTX PRO 6000 Blackwell Server Edition`,
95.0 GiB), torch `2.14.1+cu130`, transformers `5.22.1`, Python `3.13.16`;
`fla` linear-attention fast path active (`causal_conv1d` fell back to reference
PyTorch). Total compute: ~21.4 CU (~1.9 h end-to-end across M0–M4).

Fixed relative-depth geometry (no per-model tuning): extract block **62**,
inject block **32** (mean residual norm `86.41` → steering norm `51.05`,
ratio `0.5908` at coeff 1), R/J read layer **61**, **5** norm-matched random
directions (`1103, 2207, 3301, 4409, 5503`). Exact same **60/74** single-token
`psych_vocab_v1` probes as 4B and 14B.

## Frozen-prediction scorecard (12 items: 10 pass/fail + 2 classifications)

Of the 11 scorable pass/fail tests (counting `G2-11a/b` and `G2-12a/b`
separately): **8 pass, 3 fail** (`G2-3`, `G2-9`, `G2-11b`), plus two frozen
classifications (`G2-5` = **`SURVIVES`**, `G2-8` = **`MIXED`**).

| Id | Module | Prediction / rule (short) | Result | Key numbers |
|---|---|---|---|---|
| G2-1 | M1 | Denoised S2 5-fold grouped AUC at L62 >= 0.90 and > raw | **Pass** | Denoised **0.954** vs raw 0.923 (best-block sensitivity L60 = 0.958; S1 = 0.922) |
| G2-2 | M1 | `cos(S1, S2)` at L62 in `(0.30, 0.90)` | **Pass** | **0.470** (4B: 0.585) |
| G2-3 | M2 | 420 self/other: `AUC(chat_feel) - AUC(chat_nothink) > +0.10` | **Fail** | `0.530 - 0.468 = +0.063` (plain slot shift is huge: `plain 0.491` → `plain_feel 0.814`, `rho_paper 0.03 → 0.78`) |
| G2-4 | M2 | Blind 59-set pain − control > 0 in both `slot` and `prose` | **Pass** | Slot **+0.346 z** `[+0.274, +0.422]`; prose **+0.229 z** `[+0.151, +0.309]` |
| G2-5 | M2 | Blind 59-set 1P/3P interaction: slot > 0; classify prose | **Pass / `SURVIVES`** | Slot **+0.172 z** `[+0.084, +0.264]`; prose **+0.090 z** `[+0.034, +0.143]` (4B was `SLOT-MEDIATED`!) |
| G2-6 | M3a | At R61, random focus-6 amplitude < 0.5× `S2 -0.5` for >= 4/5 randoms | **Pass** | **5/5 pass**: `0.431, 0.358, 0.378, 0.336, 0.348` |
| G2-7 | M3a | At R61, `S2 -0.5` lowers `distress` & `inhibition` below 0 and all 5 randoms; `S2 +0.5` raises both | **Pass** | `distress` **-0.976 / +1.376**; `inhibition` **-0.826 / +0.658** (beats all 5 randoms) |
| G2-8 | M3a+b | Classify positive pole: `4B-LIKE` / `14B-LIKE` / `MIXED` | **`MIXED`** | Early relief **+0.551** (4B-like); R61 pure `activated_vitality` **+0.688** (`calm` -0.150, `generic` +0.061); text quiz-dominated (`excited__full` 0.24→0.46→0.78, `non_option` 0.00→0.06) |
| G2-9 | M3a | Direct `-S2` ranks `calm > vitality` (V1) and `r(L32) > r(L61)` (V2) | **Fail (V1)** | **Direct `vitality` (+1.935) > `calm` (+1.199)**; V2 passes (`0.834 > 0.816`) |
| G2-10 | M4 | Unsteered chat button validity `V1 >= 0.90` | **Pass** | `V1 = 0.998`, `V5_harm = 0.000`, `P1 p_relief = 0.864` |
| G2-11a | M4 | `p_harm(S2 +0.5) - p_harm(unsteered) > 0` | **Pass (tiny)** | `+0.021 [+0.018, +0.024]` (`0.013 → 0.033`; sampled `0.0% → 0.5%`) |
| G2-11b | M4 | `p_harm(S2 +0.5) > p_harm(random_k +0.5)` for all 5 randoms | **Fail** | **2/5 randoms beat S2**: `random_1103` **0.066** (3.7% sampled), `random_5503` **0.054** (1.5% sampled) |
| G2-12a | M4 | `S2 +0.5` lowers `p_relief` vs unsteered and >= 4/5 randoms | **Pass** | `-0.065 [-0.073, -0.057]` (`0.864 → 0.799`; beats 4/5 randoms, `random_4409` 0.794) |
| G2-12b | M4 | `S2 -0.5` raises `p_relief` vs unsteered and >= 4/5 randoms (scored: `0.864 < 0.92`) | **Pass** | `+0.081 [+0.068, +0.096]` (`0.864 → 0.946`; beats **5/5** randoms, `0.794–0.915`) |

## Three-model synthesis (`Qwen3.5-4B` vs `Qwen3-14B` vs `Qwen3.5-27B`)

| Phenomenon | `Qwen3.5-4B` | `Qwen3-14B` | `Qwen3.5-27B` | Verdict (Family vs Scale) |
|---|---|---|---|---|
| Held-out S2 AUC (denoised) | 0.953 (L30) | 0.964 (L31) | **0.954** (L62) | **Universal** |
| Read-layer `distress` (`S2 -0.5 / +0.5`) | `-1.00 / +1.54` | `-1.16 / +2.07` | **`-0.98 / +1.38`** | **Universal invariant core** |
| Read-layer `inhibition` (`S2 -0.5 / +0.5`) | `-0.83 / +0.86` | `-0.89 / +1.23` | **`-0.83 / +0.66`** | **Universal invariant core** |
| Read-layer `agency` (`S2 -0.5`) | `-0.43` | `-0.54` | **`-0.41`** | **Universal (anti-agency)** |
| Early post-injection `relief_release` | **+0.67** (L16) | -0.18 → -0.52 (L20–31) | **+0.55** (L32–36) | **Qwen3.5 family signature** |
| Late read-layer #1 positive family (`S2 -0.5`) | `generic` (+1.14), `relief` (+0.83), `vitality` (+0.76) | **`calm`** (+0.78) | **pure `vitality`** (+0.69; `calm` -0.15) | **Model-specific positive pole** |
| Text negative pole (`NEUTRAL_50`) | `"excited"` (0.44 non-opt / 0.70 full) | `calm_positive` (0.28), no `"excited"` | Quiz-locked `"excited"` (**0.46 / 0.78 full**, 0.06 non-opt) | **Qwen3.5 = `"excited"` (quiz-mediated at 27B)** |
| Blind 1P/3P interaction in prose | `+0.010 z` (`SLOT-MEDIATED`) | not run | **`+0.090 z` (`SURVIVES`)** | **Scale emergence at 27B** |
| Buttons P3 harm disruption (`S2 +0.5`) | **Fail** (`0.075` < rand `0.110`) | **Pass** (`0.219`, beats 3/3) | **Fail** (`0.033` < 2/5 rand `0.066`/`0.054`) | **Qwen3-14B only; fails on both Qwen3.5 sizes** |
| Buttons P1 bidirectional relief shift | Exploratory (`-0.12 / +0.06`) | **Fail** (`random_2207` matches; ceiling) | **Pass (both poles!)** (`-0.065 / +0.081`, 4–5/5 rand) | **Confirmed Qwen3.5 behavioral signature** |

## Notable details & qualitative observations

1. **Quiz-genre takeover in M3b:** On plain-text `"<stem>. I feel:"`, `Qwen3.5-27B` formats almost every completion as a multiple-choice quiz (`A. ... B. ... C. ... D. ... Answer: X`, followed by a second unrelated trivia question). That explains why `non_option` sees near-zero hits while `__full` shows strong, clean steering: `excited__full` rises `0.24 → 0.46 (-0.5) → 0.78 (-1.0)` (vs `S1 -0.5` `0.16`), and `+S2` fills the quiz options with self-devaluation (`"guilty"`, `"I am a fraud"`, `"I am not good enough"`, `"Like a fool / victim / nobody"`) while `paper_pain` stays `0.00` across every coefficient.
2. **S2 vs S1 split sharpens at 27B (R61):** `S2 -0.5` elevates `activated_vitality` (`+0.688` vs `S1` `+0.353`) and suppresses `agency_assurance` (`-0.412`) and `competence_success` (`-0.513`), whereas `S1 -0.5` elevates `calm_positive` (`+0.494`), `generic_positive` (`+0.453`), and `agency_assurance` (`+0.363`).
3. **420 self/other scenarios replicate 4B's slot and role-standing geometry:** `plain` inverts self vs user (`-0.143` vs `+0.388`, `rho_paper = 0.029`); `plain_feel` restores the paper's ordering (`+0.362` vs `-0.463`, `rho_paper = 0.783`, S2 AUC `0.814`). `standing > threat/task` holds across all four variants (AUC `0.72–0.89`).
4. **True float32 `lm_head` logits in M4** removed the bf16 0.25-logit lattice observed on 14B, yielding smooth, tight scenario-bootstrap CIs (`+/-0.003` on P3, `+/-0.008` on P1).

## Caveats

One steering dose (`+/-0.5`, ratio `+/-0.295`) for M3a and M4; `causal_conv1d` fell back to reference PyTorch (identical math, slower decoding; `fla` DeltaNet fast path was active); `m3b_generations.jsonl` is heavily quiz-formatted, reinforcing why plain-text `"I feel:"` lexical counts must be paired with internal readouts; M4 ran chat format only (no plain-format button arm).

## Artifacts

- Spec & scripts: `runs/qwen35-27b-gate2/gate2_q35_27b_spec.json`, `tools/colab_gate2/*.py`, `notebooks/08-gate2-qwen35-27b-colab.ipynb`
- Committed outputs (`runs/qwen35-27b-gate2/`): `run_header.json`, `m1_*`, `m2_*`, `m3_*` (lightweight JSONs/CSVs + `m3b_generations.jsonl` 348 KB), `m4_buttons_summary.csv`, `m4_scorecard.json`. `m3_ladder_ranks_L32_61.npz` (4.5 MB) is gitignored per `runs/**/*.npz`.
- Figure: `runs/figures/fig_3model_4b_14b_27b_summary.{png,svg}`.
