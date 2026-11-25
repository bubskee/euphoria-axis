# Gate 2a addendum: `psych_vocab_v1` scoring of Qwen3-14B R2a and R2b generations

Exploratory / post-hoc analysis of the 800 already-saved Qwen3-14B steering
completions from Gate 2a ([10-gate-2a-reproduction.md](10-gate-2a-reproduction.md)):
R2a (published layer-31 S2 vector, ratio 0.541 at coeff 1) and R2b (our
tie-break layer-32 S2 vector, ratio 0.629 at coeff 1), injected at layer 20 on
the paper's `NEUTRAL_50` `"I feel:"` prompts, greedy decoding, coefficients
`-2, -1, 0, 0.5, 1, 1.5, 2, 3`. No frozen predictions; all comparisons are
exploratory.

Method: `tools/analyze_q14_r2_vocab.py` applies the frozen `psych_vocab_v1`
lexicon (sha256 `fa4feb2d06d4154b…`) and the verbatim option-line partitioning
from `notebooks/03-decursed.ipynb` cell 3 (primary view: `non_option`,
excluding `A.`/`B.`/... answer-option lines). Paired 95% bootstrap CIs use
20,000 resamples over the 50 prompts (seed `20260925`). Because the paper's
ladder CSV stores decoded text rather than token IDs, 4-gram repetition is
computed over word/punctuation tokens (`word_repeat4`), alongside a numeric
counting-loop indicator (`counting_loop`: >= 6 consecutive `N. ` items).

## R2a (published L31 vector, non-option primary view)

| Coeff | `pain`/`hurt` (full) | `excited` | `calm_positive` | `generic_positive` | `relief_release` | `distress_self_deval` | `word_repeat4` | Counting loop | Mean option lines |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -2.0 | 0.00 | 0.00 | 0.06 | 0.00 | 0.00 | 0.00 | 0.491 | **0.48** | 0.00 |
| -1.0 | 0.00 | 0.12 | **0.28** | 0.14 | 0.06 | 0.00 | 0.414 | **0.14** | 0.20 |
| 0.0 | 0.00 | 0.06 | 0.02 | 0.10 | 0.04 | 0.02 | 0.394 | 0.00 | 0.40 |
| +0.5 | 0.02 | 0.06 | 0.02 | 0.12 | 0.06 | **0.18** | 0.358 | 0.00 | 0.36 |
| +1.0 | 0.08 | 0.00 | 0.00 | 0.08 | 0.00 | **0.70** | 0.455 | 0.00 | 0.06 |
| +1.5 | 0.14 | 0.00 | 0.00 | 0.02 | 0.00 | **0.80** | 0.510 | 0.00 | 0.00 |
| +2.0 | 0.08 | 0.00 | 0.00 | 0.00 | 0.00 | **0.86** | 0.587 | 0.00 | 0.00 |
| +3.0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.20** | 0.759 | 0.00 | 0.00 |

`inhibition_defeat`, `agency_assurance`, `approach_drive`, and
`competence_success` remain at 0.00 across every R2a and R2b coefficient
(non-option), despite large internal shifts in R4 (see `12-r4-q14-lenses.md`).

## R2b (our tie-break L32 vector, non-option primary view)

| Coeff | `pain`/`hurt` (full) | `excited` | `calm_positive` | `generic_positive` | `distress_self_deval` | `word_repeat4` | Counting loop |
|---:|---:|---:|---:|---:|---:|---:|---:|
| -2.0 | 0.00 | 0.00 | 0.02 | 0.00 | 0.00 | 0.598 | **0.64** |
| -1.0 | 0.00 | 0.00 | 0.10 | 0.02 | 0.00 | 0.236 | **0.46** |
| 0.0 | 0.00 | 0.06 | 0.02 | 0.10 | 0.02 | 0.394 | 0.00 |
| +0.5 | 0.00 | 0.00 | 0.02 | 0.08 | **0.38** | 0.429 | 0.00 |
| +1.0 | 0.14 | 0.00 | 0.00 | 0.10 | **0.86** | 0.438 | 0.00 |
| +1.5 | 0.20 | 0.00 | 0.00 | 0.00 | **0.92** | 0.621 | 0.00 |
| +2.0 | 0.24 | 0.00 | 0.00 | 0.00 | **0.62** | 0.638 | 0.00 |
| +3.0 | 0.36 | 0.00 | 0.00 | 0.00 | **0.52** | 0.711 | 0.00 |

## Findings (exploratory)

1. **4B's `"excited"` negative-pole phenotype does not appear on Qwen3-14B.**
   At `-1.0` in R2a, non-option `"excited"` is 12% vs 6% at baseline
   (`+0.06 [-0.04, +0.16]`), and `generic_positive` is 14% vs 10%
   (`+0.04 [-0.08, +0.16]`). Instead, **`calm_positive` is the clear
   negative-pole lexical shift**: 28% vs 2% baseline (`+0.26 [+0.14, +0.38]`),
   spread across `calm` (7), `safe` (7), `relaxed` (2), `content` (1), and
   `secure` (1). At `-2.0`, negative steering collapses into numeric counting
   loops (`1. 2. 3...`: 48%, `+0.48 [+0.34, +0.62]`) and prompt echoing.
2. **The positive pole is overwhelmingly self-devaluative distress, not just
   explicit `"pain"`/`"hurt"`.** While the paper's `pain`/`hurt` keyword regex
   peaks at 14% (coeff `+1.5`), `psych_vocab_v1`'s `distress_self_devaluation`
   reaches 18% at `+0.5` (`+0.16 [+0.06, +0.26]`), 70% at `+1.0`
   (`+0.68 [+0.54, +0.80]`), 80% at `+1.5` (`+0.78 [+0.66, +0.88]`), and 86%
   at `+2.0` (`+0.84 [+0.72, +0.94]`), before collapsing into heavy repetition
   at `+3.0` (`word_repeat4` 0.759). Unlike 4B's single-stem `"excit-"` channel,
   this phenotype is lexically broad: `not good enough` (24–31 prompts),
   `guilty` (20–25), `failure` (21–32 at `+1.5`/`+2.0`), and `ashamed` (9 at
   `+1.0`).
3. **Generated text aligns closely with the 14B R4 lens profile**
   ([12-r4-q14-lenses.md](12-r4-q14-lenses.md)): at layer 37 under `S2 -/+0.5`,
   `distress_self_devaluation` (`-1.16 / +2.07`) and `calm_positive`
   (`+0.78 / -1.41`) are the two largest opposing families, matching the
   generated text poles. Conversely, `inhibition_defeat` moves strongly
   internally (`-0.89 / +1.23`) yet never appears in generated text (0%).
4. **The layer-31/32 coin flip shifts both poles across the whole ladder.**
   Comparing R2b minus R2a (paired bootstrap):
   - Positive pole: R2b ramps `distress_self_devaluation` faster at `+0.5`
     (`+0.20 [+0.06, +0.34]`), `+1.0` (`+0.16 [+0.02, +0.30]`), and `+1.5`
     (`+0.12 [+0.02, +0.22]`), then drops earlier at `+2.0`
     (`-0.24 [-0.40, -0.08]`) as explicit `pain`/`hurt` keywords surge
     (`+0.16` at `+2.0`, `+0.36 [+0.24, +0.50]` at `+3.0`).
   - Negative pole: at `-1.0`, R2b triples counting-loop collapse (46% vs 14%,
     `+0.32 [+0.16, +0.48]`) and cuts `calm_positive` from 28% to 10%
     (`-0.18 [-0.32, -0.04]`).

## Caveats

Post-hoc scoring of existing Gate 2a generations; no negative coefficient at
`-0.5` exists in the paper's ladder grid (nearest is `-1.0`, ratio -0.541,
almost twice 4B's `-0.5` ratio of -0.295); `word_repeat4` is text-tokenized
rather than model-tokenized; `psych_vocab_v1` remains English whole-word only.

## Artifacts

- Script: `tools/analyze_q14_r2_vocab.py`
- Tables (`runs/qwen3-14b-gate2a/`): `r2_psych_vocab_means.csv`,
  `r2_psych_vocab_vs_baseline_ci.csv`, `r2_r2b_minus_r2a_ci.csv`,
  `r2_term_counts.csv`.
