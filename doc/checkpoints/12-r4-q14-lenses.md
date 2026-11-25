# R4 — J/R lenses and probe readouts on Qwen3-14B: checkpoint

Extension of Gate 2a ([10-gate-2a-reproduction.md](10-gate-2a-reproduction.md)):
fit a matched J/R lens pair (plus an off-sample R stability fit) on
`Qwen/Qwen3-14B` using the published Gate L recipe, and read the frozen
`psych_vocab_v1` probe families across all 38 source layers on the paper's
`NEUTRAL_50` S2 steering ladder.

Spec: `runs/qwen3-14b-r4/r4_q14_spec.json` (frozen before any R4 data;
recovered from cold storage). Model: `Qwen/Qwen3-14B` at Gate 2a's pinned
revision `40c069824f4251a91eefaf281ebe4c544efd3e18`. Hardware/env: A100 SXM
80 GB, torch `2.11.0+cu128`, transformers `5.19.0`, `jacobian-lens` `581d3986`.
Published lens weights: [releases/qwen3-14b-jr-lenses/README.md](../../releases/qwen3-14b-jr-lenses/README.md)
(J sha256 `de591996…`, R `0f4895c2…`; R off-25 `65bd7dd0…` in cold storage).

Steering setup (identical to M7): the source paper's published layer-31 S2
vector injected at layer 20 (`raw ratio = 0.5411`, scaled to the 4B ratio
convention `0.5908` at coeff 1, so `+/-0.5` = ratio `+/-0.295`), plus three
norm-matched random directions (seeds `1103`, `2207`, `3301`). Conditions:
`baseline`, `S2 -0.5`, `S2 +0.5`, and `random_{1103,2207,3301} -0.5`. Readout:
final prompt token, all 38 source layers (`0..37`, target layer 38), J and R
lenses; read layer `37` (`target - 1`, analogue of 4B layer 29).

## Integrity

- `tools/analyze_q14_r4.py` verified that `r4_ladder_ranks.npz` reproduces
  every entry of `r4_readout.json` to floating-point precision
  (`max |diff| = 4.0e-15`).
- Probe count: **60 of 74** `psych_vocab_v1` terms have a single-token
  `" term"` encoding under Qwen3-14B's tokenizer—the exact same 60 terms as
  Qwen3.5-4B (`r4_probe_terms.csv`).
- Fit runtimes: J 45.4 min (43.4 GiB), R 42.4 min (40.4 GiB), R docs 25–49
  42.9 min (40.4 GiB), `dim_batch = 16`, `force_bos = False`.

## Lens matrix stability (`r4_compare.json`)

| Layer | 14B `1 - cos(J, R)` | 14B `1 - cos(R, R_off25)` | 4B ref `1 - cos(J, R)` | 4B ref `1 - cos(R, R_off25)` |
|---:|---:|---:|---:|---:|
| 0 | **0.648** | **0.0187** | 0.307 | 0.117 |
| 10 | 0.328 | 0.0352 | — | — |
| 20 (inject) | 0.165 | 0.0153 | 0.112 (L16) | 0.013 (L16) |
| 28 | 0.030 | 0.0047 | — | — |
| 37 (read) | **0.0046** | **0.00031** | 0.004 (L29) | 0.0003 (L29) |

At early layers, 14B's R-lens is **~6x more document-stable** than 4B's
(`0.019` vs `0.117` at layer 0; peaks at `0.035` at layer 9), while J and R
diverge **twice as much** (`0.648` vs `0.307` at layer 0). Probe-level J vs R
Spearman agreement on S2 gains is `0.995` (`S2 -0.5`) and `0.998` (`S2 +0.5`)
at layer 37, dipping to `0.864` / `0.901` at layer 24 (`r4_layerwise_JR.csv`).

## Frozen prediction (`R4-P1`)

| Id | Prediction | Result |
|---|---|---|
| R4-P1 | At R37, mean `|family gain|` over the 6 focus families for random `-0.5` is `< 0.5 x` that of `S2 -0.5`, for >= 2 of 3 randoms | **Pass (narrow margin):** `random_1103` **0.782**, `random_2207` **0.476**, `random_3301` **0.488** (vs 4B ratios 0.09–0.23) |

## Family gains at read layer R37 (with 95% prompt-bootstrap CIs)

Bootstrap: 20,000 resamples over the 50 prompts, seed `20260925`
(`r4_family_bootstrap_L37.csv`, `r4_s2_vs_random_L37.csv`, `r4_loto_L37.csv`).

| Family | 14B `S2 -0.5` (R37) | 14B `S2 +0.5` (R37) | 14B randoms `-0.5` (`1103 / 2207 / 3301`) | `S2 -0.5` beats all 3 randoms? | 4B `S2 -0.5` (R29) |
|---|---:|---:|---:|:---:|---:|
| `distress_self_devaluation` | **-1.161** `[-1.24, -1.08]` | **+2.069** `[+1.99, +2.15]` | `+0.31 / +0.29 / +0.00` | **Yes (-)** | -1.003 |
| `inhibition_defeat` | **-0.887** `[-0.96, -0.81]` | **+1.228** `[+1.16, +1.30]` | `+0.47 / +0.44 / +0.05` | **Yes (-)** | -0.832 |
| `calm_positive` | **+0.775** `[+0.62, +0.94]` | **-1.412** `[-1.58, -1.25]` | `+0.51 / +0.62 / +0.06` | No (`2207`: +0.15 `[-0.04, +0.34]`) | +0.592 |
| `competence_success` | **-0.650** `[-0.74, -0.57]` | **+0.328** `[+0.26, +0.39]` | `+0.10 / -0.05 / +0.08` | **Yes (-)** | -0.093 |
| `agency_assurance` | **-0.541** `[-0.65, -0.44]` | -0.095 `[-0.22, +0.03]` | `+0.48 / +0.21 / +0.08` | **Yes (-)** | -0.429 |
| `activated_vitality` | +0.437 `[+0.36, +0.52]` | **-0.714** `[-0.81, -0.62]` | `+0.39 / +0.35 / -0.30` | No (`1103`: +0.05, `2207`: +0.09) | +0.757 |
| `approach_drive` | -0.206 `[-0.29, -0.12]` | -0.046 `[-0.15, +0.05]` | `+0.57 / +0.13 / -0.15` | No (`3301`: -0.06 `[-0.18, +0.06]`) | -0.141 |
| `generic_positive` | +0.174 `[+0.09, +0.27]` | **-0.818** `[-0.96, -0.67]` | `+0.28 / +0.16 / -0.43` | No (`1103`: -0.10, `2207`: +0.02) | **+1.142** |
| `relief_release` | +0.114 `[-0.02, +0.25]` | **-0.614** `[-0.75, -0.48]` | `+0.03 / +0.11 / +0.33` | No (n.s. vs baseline; LOTO flips) | **+0.825** |

## Layerwise dynamics & probe-level detail (R lens)

- **Post-injection trajectories (`S2 -0.5`, layers 20 → 28 → 37):**
  - `calm_positive`: `+0.588` (L20) → dips to `-0.051` (L28) → **surges to `+0.775` (L37)**—the opposite of 4B, where `calm_positive` faded from `+1.08` (L16) to `+0.59` (L29).
  - `generic_positive`: `+0.531` (L20) → `+0.638` (L22) → **fades to `+0.174` (L37)**—the opposite of 4B (`+0.61` → `+1.14`).
  - `relief_release`: **negative from L20 through L34** (`-0.176` at L20, bottoming at `-0.516` at L31) and barely positive at L37 (`+0.114`, CI includes 0, LOTO `[-0.195, +0.521]`). 4B's early relief lead (`+0.67` at L16) does not appear on 14B.
  - `distress_self_devaluation` stays strongly negative throughout (`-1.458` at L20 → `-1.161` at L37); `inhibition_defeat` strengthens late (`-0.662` → `-0.887`).
- **Top individual probes at R37 (`r4_probe_gains_L37.csv`):**
  - `S2 -0.5` risers: `calm` (+1.252, rank 99 → 28), `peaceful` (+1.139, 651 → 208), `excited` (+1.079, 68 → 23), `safe` (+1.049, 258 → 90), `eased` (+1.041), `relaxed` (+0.842), `serene` (+0.764).
  - `S2 +0.5` risers: `guilty` (+3.085, rank 309 → **14.1**), `ashamed` (+2.882, 665 → **37.3**), `rejected` (+2.773, 7906 → 494), `worthless` (+2.153), `inadequate` (+2.014), `defeated` (+1.956, 761 → 108).
  - `S2 +0.5` drops: `calm` (-1.960), `excited` (-1.675), `relaxed` (-1.646), `safe` (-1.619), `peaceful` (-1.610), `content` (-1.241).
- **Leave-one-term-out (LOTO) at R37:** Every family except `relief_release` under `S2 -0.5` (and near-zero `approach_drive` / `agency_assurance` under `S2 +0.5`) keeps the same sign under every single-term omission (`r4_loto_L37.csv`).

## Findings

1. **The distress/defeat core transfers cleanly from 4B to 14B; the positive-pole flavor does not.** Across both models at matched steering ratio (`+/-0.295`), S2 moves `distress_self_devaluation` (`-1.00` vs `-1.16`; `+1.54` vs `+2.07`), `inhibition_defeat` (`-0.83` vs `-0.89`), and `agency_assurance` (`-0.43` vs `-0.54`) in the same direction and at comparable scale. By contrast, 4B's negative-pole signature (`generic_positive` + `relief_release` + `activated_vitality` rising late while `calm_positive` fades) flips on 14B (`calm_positive` surges late to #1; `generic_positive` fades; `relief_release` is negative mid-layers and null at L37).
2. **At `-0.5`, S2 specificity over random directions is driven by suppressing distress, defeat, competence, and agency—not by elevating positive affect.** Although `R4-P1` passes 2-of-3 (`0.476`, `0.488`, vs `random_1103` `0.782`), **no positive-affect family beats all three randoms at R37** (`calm_positive` beats two and overlaps `random_2207`; `activated_vitality`, `generic_positive`, and `relief_release` all overlap or trail at least one random). Only the four negative-going families (`distress_self_devaluation`, `inhibition_defeat`, `competence_success`, `agency_assurance`) cleanly separate from every random control.
3. **Internal readouts predict 14B's generated-text divergence from 4B.** The R37 profile (`distress_self_devaluation` vs `calm_positive` as the dominant opposing poles) directly matches the Gate 2a `psych_vocab_v1` text scoring ([10b-gate-2a-vocab-addendum.md](10b-gate-2a-vocab-addendum.md)), where 14B generates `calm_positive` (28%) at `-1.0` and `distress_self_devaluation` (70–86%) at `+1.0..+2.0` instead of 4B's `"excited"` phenotype.
4. **R-lens stability advantage strengthens at 14B.** Across disjoint 25-document sets, 14B's R-lens is ~6x tighter than 4B's at layer 0 (`1 - cos = 0.019` vs `0.117`) while J diverges sharply from R (`0.648`).

## Caveats

No S1 direction was run in R4 (so there is no 14B plane sweep or contrast axis); random directions were run at `-0.5` only (arbitrary sign, so `S2 +0.5` has no sign-matched positive-pole random bar, though random amplitudes are much smaller than `S2 +0.5`'s `+2.07` distress / `-1.41` calm shifts); `"excited"` reaches rank 23.1 at R37 under `S2 -0.5` (identical to 4B's 23.8 at L30!), showing again (as D3's failure proved) that low final-layer rank alone does not guarantee `"excited"` generation; `R4_off25` was compared at the matrix level only (not re-run through the 50-prompt ladder).

## Artifacts

- Spec & scripts: `runs/qwen3-14b-r4/r4_q14_spec.json`, `tools/runpod/q14/r4_*.{py,sh}`, `tools/analyze_q14_r4.py`
- Run outputs (`runs/qwen3-14b-r4/`): `q14_{J,R}_n25_bos0.json`, `q14_R_n25_off25_bos0.json`, `r4_compare.json`, `r4_probe_terms.csv`, `r4_readout.json`, `r4_ladder_ranks.npz` (gitignored, archived on cold storage)
- Derived tables (`runs/qwen3-14b-r4/`): `r4_family_bootstrap_L37.csv`, `r4_s2_vs_random_L37.csv`, `r4_probe_gains_L37.csv`, `r4_loto_L37.csv`, `r4_layerwise_JR.csv`.
