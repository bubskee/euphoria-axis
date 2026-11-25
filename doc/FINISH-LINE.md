# Euphoria Axis — Plan

Bounded extension of Tagliabue, Dung & Berg, *The Pain Axis* (v2, Sept 2026).
**Principle:** Small science, small claims. Let observations constrain interpretation; never conflate representation or steering with felt experience.

## Status

| Phase | Model(s) | Status | Record |
|---|---|---|---|
| **4B core** (plane, dense, format, ablation, direct-vec, VAD, 1P/3P) | `Qwen3.5-4B` (+ Base) | Done (26 frozen: 18 pass, 8 fail) | `doc/LEDGER-4B.md`, `doc/checkpoints/01`–`07` |
| **Gate L** (J/R lens calibration) | `Qwen3.5-4B` | Passed (30/30 layers; R 2–3× stabler) | `doc/checkpoints/08-gate-L-lens-calibration.md` |
| **Gate 1** (button task) | `Qwen3.5-4B` (+ Base) | Pass A (harm H2 fails; relief shift exploratory) | `doc/checkpoints/09-gate-1-buttons.md` |
| **Gate 0** (4B freeze) | `Qwen3.5-4B` | Closed | `doc/CLAIMS-INDEX.md`, `doc/CONVENTIONS.md` |
| **Gate 2a** (paper repro + vocab scoring) | `Qwen/Qwen3-14B` | Passed (L31 cos 0.99994; L31/32 coin flip; R2a reproduces) | `doc/checkpoints/10`, `10b` |
| **M7** (button task) | `Qwen/Qwen3-14B` | Done (harm reproduces 0.2%→21.5%; relief fails specificity) | `doc/checkpoints/11-m7-q14-buttons.md` |
| **R4** (J/R lenses + probe readouts) | `Qwen/Qwen3-14B` | Done & HF-released (`distress` vs `calm`, no early relief) | `doc/checkpoints/12`, `releases/qwen3-14b-jr-lenses/` |
| **Gate 2** (27B scale generalization) | `Qwen/Qwen3.5-27B` | **Passed** (Colab G4: 8/11 pass, 1P/3P `SURVIVES` in prose, bidirectional relief confirmed, harm fails specificity) | `doc/checkpoints/13-gate-2-qwen35-27b.md` |
| **Gate 3** (version generalization + J++) | `Qwen/Qwen3.6-27B` | **Passed** (Colab G4: 9/13 pass, `R`≈`J++`, 1P/3P `SURVIVES`, `S2 -0.5` relief pass, harm fails 4/5 randoms) | `doc/checkpoints/14-gate-3-4-qwen36-27b-olens.md` |
| **Gate 4** (O-Lens peek, exploratory) | `Qwen/Qwen3.6-27B` | **Done** (`olens_s3d_space_rl600`, 108/108 verbalizations + matched `R/J/J++`: `S2 +0.5` = shame/guilt/isolation, never physical pain) | `doc/checkpoints/14-gate-3-4-qwen36-27b-olens.md` |
| **Gates 5–6** (Tech PDF + LW post) | All 4 tested models | **Next** (all experimental gates closed!) | `doc/writeup/lw-draft.md` |

## Next: Gates 2–3 — 27B Generalization (`Qwen3.5-27B`, `Qwen3.6-27B`)

Freeze specs and predictions **before** running. Carry over the exact 4B/14B instruments without retuning:

- **Layers (fixed relative depth, 64 blocks):** extract `62`, inject `32`, R/J readout `61` (CV best-layer AUC as sensitivity check only).
- **Dose:** paper's vector-to-residual `ratio` convention (`0.5908` at coeff 1; `+/-0.5` = `+/-0.295`), matched to 4B and 14B.
- **Readout:** published recipe-matched `workspace-lenses` R (primary) and J pairs (recorded by sha256; optional `koayon/jpp-lenses` on Qwen3.6-27B with expert-routing reported).
- **Minimum battery per model:**
  1. Grouped held-out S2 pain/control AUC + compact category projections.
  2. Self/other read-slot check (ordinary response start vs forced `I feel:` slot).
  3. Signed S2 steering ladder on `NEUTRAL_50` (`psych_vocab_v1` + degeneration) + R/J family readouts, against **>= 3** (ideally 5) norm-matched random directions.
  4. Optional M7 buttons in chat — applying 14B lessons: **float32** name logits (no bf16 lattice), >= 3 random directions, explicit ceiling/floor scoring rules.
- **Hardware note** (`runs/bench_27b/`): A100 80 GB runs everything (~50 GiB bf16; ~16 h / ~$28 benchmarked for three 27B models). Avoid Hopper for `fla` linear-attention backward passes (Triton 3.4–3.7 bug); inference/steering is fine.

## Optional: Gate 4 — O-Lens Peek

Exploratory only, on 8–12 preselected activations (both S2 poles + controls) using existing checkpoints only (`agu18dec/olens_and_ar`, adapter `olens_s3d_rl600`, in-contract layers 20–60 step 4, `concepts_raw` / `unit` / alpha 16000, adapter disabled at capture). No new training, no causal claims; omit cleanly if not worth the time.

## Finish: Gates 5–6 — Technical Report & LessWrong Post

1. **Technical PDF (Gate 5):** self-contained report from committed tables & figures (`runs/figures/` + `runs/qwen35-4b-posttrained-s2/paper_protocol/figures/`), separating source transfer, prospectively tested (passes + fails), and exploratory findings. Engage with upstream `MANUSCRIPT_DISCREPANCIES.md`. Share with source authors before publishing.
2. **LessWrong post (Gate 6):** *Don't Call It Pain!* — 600–850 words (< 1,000), two strong figures, links to PDF + repo + source paper (`doc/writeup/lw-draft.md`).

## Stop Rules

- **Negative results finish experiments**; fix instrumentation bugs only, never retune to chase an effect.
- **Never edit `psych_vocab_v1`**; new vocabularies must be versioned and labeled exploratory.
- **State sign injected**, **name the read slot**, and **ignore middle-layer family signs < 0.1** (Gate L uncertainty ≈ ±0.07).
- **No compulsory 27B or O-Lens work:** 4B + 14B already forms a complete publishable story if compute/time runs short.
- **Bulky artifacts** live in cold storage (`runs/COLD_STORAGE_2026-11-07.txt`) and Hugging Face; keep git lean.
