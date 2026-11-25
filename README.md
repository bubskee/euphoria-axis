# euphoria-axis

Bounded reproduction and extension of Tagliabue, Dung & Berg, *The Pain Axis* (v2, 2026), across Qwen3.5-4B (Base & post-trained), Qwen3-14B, and planned 27B checkpoints.

## Layout

- `doc/FINISH-LINE.md` — concise live plan, current status, frozen 27B rules, and stop rules.
- `doc/CLAIMS-INDEX.md` — master inventory of every frozen prediction (pass/fail) and claim across 4B and 14B.
- `doc/CONVENTIONS.md` & `doc/LEDGER-4B.md` — pinned methods/sign conventions and the frozen 4B claims summary.
- `doc/checkpoints/` — numbered per-experiment records (`01`–`12`).
- `doc/protocols/` — frozen button-task protocol (`pilot-v2.md`, `amendment-1-harm.md`) and blind 1P/3P matched-pair specs.
- `doc/writeup/` — LessWrong post outline (`lw-draft.md`).
- `releases/qwen3-14b-jr-lenses/` — Hugging Face model card and usage docs for the released Qwen3-14B J/R lenses.
- `notebooks/` — numbered experiment notebooks (`00`–`07`).
- `tools/` — local analysis & figure scripts (`analyze_q14_*.py`, `make_figures.py`), lens tools, and RunPod scripts (`tools/runpod/`).
- `runs/` — lightweight tables, specs, scorecards, and figures (`runs/figures/`). Bulky binaries (`*.npz`, large `*.pt`/`.jsonl`) are archived in cold storage (`runs/COLD_STORAGE_2026-11-07.txt`) and tracked by sha256 in `bulky_artifacts_manifest.json`.

Third-party data are not redistributed: the source paper's datasets come from its repository (pinned commit `8d1649c0`), and the NRC VAD Lexicon must be obtained from NRC under its own terms.
