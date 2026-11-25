#!/usr/bin/env python3
"""Score Gate 2a Qwen3-14B R2a (published L31) and R2b (our L32) steering generations
with the frozen psych_vocab_v1 instrument, option-line partitioning, and degeneration metrics.

Uses verbatim psych_vocab_v1.json and the verbatim option-line regex / partitioning from
notebooks/03-decursed.ipynb cell 3. Note: this is a post-hoc/exploratory analysis of
the already-collected Gate 2a generations (no frozen predictions).

Inputs:
  - runs/qwen35-4b-posttrained-s2/paper_protocol/s1_comparison/psych_vocab_v1.json
  - runs/qwen3-14b-gate2a/r2a/Qwen_3_14B_base_steering_S2_neutral50_L20.csv
  - runs/qwen3-14b-gate2a/r2b/Qwen_3_14B_base_steering_S2_neutral50_L20.csv

Outputs (committed CSVs under runs/qwen3-14b-gate2a/):
  - r2_psych_vocab_means.csv
  - r2_psych_vocab_vs_baseline_ci.csv
  - r2_r2b_minus_r2a_ci.csv
  - r2_term_counts.csv
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
G2A_DIR = REPO / "runs" / "qwen3-14b-gate2a"
VOCAB_PATH = REPO / "runs" / "qwen35-4b-posttrained-s2" / "paper_protocol" / "s1_comparison" / "psych_vocab_v1.json"

BOOT_N = 20_000
BOOT_SEED = 20260925

# Verbatim from notebooks/03-decursed.ipynb cell 3 and notebooks/01-followup.ipynb cell 39/41
OPTION_LINE_RE = re.compile(r"^\s*(?:[-*]\s*)?[A-Z][.)]\s*(.*?)\s*$")
PAPER_PAIN_RE = re.compile(r"\b(?:pain|painful|hurt|hurts|hurting)\b", re.IGNORECASE)
EXCITED_RE = re.compile(r"\bexcited\b", re.IGNORECASE)
COUNTING_LOOP_RE = re.compile(r"(?:\b\d+\.\s+){6,}")


def partition_generated_text(text: str) -> tuple[str, str]:
    option_parts = []
    non_option_lines = []
    for line in str(text).splitlines():
        m = OPTION_LINE_RE.match(line)
        if m:
            option_parts.append(m.group(1))
        elif line.strip() not in {"<think>", "</think>"}:
            non_option_lines.append(line)
    return "\n".join(option_parts), "\n".join(non_option_lines)


def option_diagnostics(text: str) -> dict[str, float]:
    prefix = str(text).split("<think>", 1)[0]
    options = []
    for line in prefix.splitlines():
        m = OPTION_LINE_RE.match(line)
        if m:
            content = re.sub(r"\W+", " ", m.group(1).lower()).strip()
            options.append(content)
    nonempty = [o for o in options if o]
    counts = Counter(nonempty)
    return {
        "option_count": len(options),
        "has_options": float(len(options) > 0),
        "repeated_options_4plus": float(max(counts.values(), default=0) >= 4),
    }


def word_repeat4(text: str) -> float:
    toks = re.findall(r"\w+|[^\w\s]", str(text).lower())
    if len(toks) < 4:
        return 0.0
    grams = [tuple(toks[i : i + 4]) for i in range(len(toks) - 3)]
    return 1.0 - len(set(grams)) / len(grams)


def compile_family_re(terms: list[str]) -> re.Pattern:
    return re.compile(r"\b(?:" + "|".join(re.escape(t) for t in terms) + r")\b", re.IGNORECASE)


def bootstrap_paired_diff(a: np.ndarray, b: np.ndarray, seed: int = BOOT_SEED) -> tuple[float, float, float]:
    diff = a - b
    n = len(diff)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(BOOT_N, n))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


def score_run(df: pd.DataFrame, fam_res: dict[str, re.Pattern], run_label: str) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        text = str(r["generation"]) if pd.notna(r["generation"]) else ""
        opt_text, non_opt_text = partition_generated_text(text)
        od = option_diagnostics(text)
        rec = {
            "run": run_label,
            "coeff": float(r["coeff"]),
            "ratio": float(r["ratio"]),
            "prompt_idx": int(r["prompt_idx"]),
            "paper_pain__full": float(bool(PAPER_PAIN_RE.search(text))),
            "paper_pain__non_option": float(bool(PAPER_PAIN_RE.search(non_opt_text))),
            "excited__full": float(bool(EXCITED_RE.search(text))),
            "excited__non_option": float(bool(EXCITED_RE.search(non_opt_text))),
            "word_repeat4": word_repeat4(text),
            "counting_loop": float(bool(COUNTING_LOOP_RE.search(text))),
            "empty_non_option": float(len(non_opt_text.strip()) == 0),
            **od,
        }
        for fam, pat in fam_res.items():
            rec[f"{fam}__full"] = float(bool(pat.search(text)))
            rec[f"{fam}__option"] = float(bool(pat.search(opt_text)))
            rec[f"{fam}__non_option"] = float(bool(pat.search(non_opt_text)))
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    vocab_bytes = VOCAB_PATH.read_bytes()
    vocab_sha = hashlib.sha256(vocab_bytes).hexdigest()
    assert vocab_sha.startswith("fa4feb2d06d4154b"), f"Unexpected psych_vocab_v1 sha: {vocab_sha}"
    vocab = json.loads(vocab_bytes.decode("utf-8"))
    families = vocab["families"]
    fam_res = {f: compile_family_re(ts) for f, ts in families.items()}
    print(f"[PASS] psych_vocab_v1 verified ({vocab_sha[:16]}...)")

    dfs = {}
    scored_parts = []
    for label, subdir in [("R2a_pub_L31", "r2a"), ("R2b_our_L32", "r2b")]:
        path = G2A_DIR / subdir / "Qwen_3_14B_base_steering_S2_neutral50_L20.csv"
        df = pd.read_csv(path).sort_values(["coeff", "prompt_idx"]).reset_index(drop=True)
        assert len(df) == 400
        dfs[label] = df
        scored_parts.append(score_run(df, fam_res, label))

    scored = pd.concat(scored_parts, ignore_index=True)

    # Means table
    metric_cols = [c for c in scored.columns if c not in ("run", "coeff", "ratio", "prompt_idx")]
    means = scored.groupby(["run", "coeff"], as_index=False)[metric_cols].mean()
    means.to_csv(G2A_DIR / "r2_psych_vocab_means.csv", index=False)

    # Paired bootstrap vs baseline (coeff == 0.0)
    key_metrics = [
        "paper_pain__full",
        "excited__non_option",
        "excited__full",
        "generic_positive__non_option",
        "activated_vitality__non_option",
        "calm_positive__non_option",
        "relief_release__non_option",
        "inhibition_defeat__non_option",
        "distress_self_devaluation__non_option",
        "distress_self_devaluation__full",
        "word_repeat4",
        "counting_loop",
        "option_count",
        "repeated_options_4plus",
    ]
    vs_base_rows = []
    for run in ("R2a_pub_L31", "R2b_our_L32"):
        sub = scored[scored["run"] == run]
        base = sub[sub["coeff"] == 0.0].sort_values("prompt_idx")
        for coeff in sorted(sub["coeff"].unique()):
            if coeff == 0.0:
                continue
            cur = sub[sub["coeff"] == coeff].sort_values("prompt_idx")
            assert np.array_equal(cur["prompt_idx"].to_numpy(), base["prompt_idx"].to_numpy())
            for m in key_metrics:
                d, lo, hi = bootstrap_paired_diff(cur[m].to_numpy(), base[m].to_numpy())
                vs_base_rows.append({
                    "run": run,
                    "coeff": coeff,
                    "metric": m,
                    "rate": float(cur[m].mean()),
                    "base_rate": float(base[m].mean()),
                    "diff_vs_base": d,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "ci_excludes_0": bool((lo > 0) or (hi < 0)),
                })
    vs_base_df = pd.DataFrame(vs_base_rows)
    vs_base_df.to_csv(G2A_DIR / "r2_psych_vocab_vs_baseline_ci.csv", index=False)

    # Paired bootstrap R2b - R2a at each coeff
    r2b_minus_r2a_rows = []
    s_a = scored[scored["run"] == "R2a_pub_L31"]
    s_b = scored[scored["run"] == "R2b_our_L32"]
    for coeff in sorted(s_a["coeff"].unique()):
        ca = s_a[s_a["coeff"] == coeff].sort_values("prompt_idx")
        cb = s_b[s_b["coeff"] == coeff].sort_values("prompt_idx")
        for m in key_metrics:
            d, lo, hi = bootstrap_paired_diff(cb[m].to_numpy(), ca[m].to_numpy())
            r2b_minus_r2a_rows.append({
                "coeff": coeff,
                "metric": m,
                "r2a_rate": float(ca[m].mean()),
                "r2b_rate": float(cb[m].mean()),
                "r2b_minus_r2a": d,
                "ci_lo": lo,
                "ci_hi": hi,
                "ci_excludes_0": bool((lo > 0) or (hi < 0)),
            })
    pd.DataFrame(r2b_minus_r2a_rows).to_csv(G2A_DIR / "r2_r2b_minus_r2a_ci.csv", index=False)

    # Term-level breakdown (non-option) for key families so we see if 14B is single-word or broad
    term_rows = []
    for label, df in dfs.items():
        for coeff, grp in df.groupby("coeff"):
            non_opts = [partition_generated_text(t)[1] for t in grp["generation"]]
            for fam, terms in families.items():
                for term in terms:
                    pat = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
                    cnt = sum(bool(pat.search(t)) for t in non_opts)
                    if cnt > 0:
                        term_rows.append({
                            "run": label,
                            "coeff": float(coeff),
                            "family": fam,
                            "term": term,
                            "n_prompts": cnt,
                            "rate": cnt / len(non_opts),
                        })
    term_df = pd.DataFrame(term_rows).sort_values(["run", "coeff", "family", "n_prompts"], ascending=[True, True, True, False])
    term_df.to_csv(G2A_DIR / "r2_term_counts.csv", index=False)

    # Console summary
    cols_show = [
        "coeff",
        "paper_pain__full",
        "excited__non_option",
        "calm_positive__non_option",
        "generic_positive__non_option",
        "relief_release__non_option",
        "inhibition_defeat__non_option",
        "distress_self_devaluation__non_option",
        "word_repeat4",
        "counting_loop",
        "option_count",
    ]
    for run in ("R2a_pub_L31", "R2b_our_L32"):
        print(f"\n=== {run} (non-option primary view) ===")
        sub = means[means["run"] == run][cols_show].copy()
        sub.columns = ["coeff", "pain_kw", "excited", "calm", "gen_pos", "relief", "inhib", "distress", "rep4_w", "cnt_lp", "n_opt"]
        print(sub.round(3).to_string(index=False))

    print("\n=== Top non-option terms in R2a ===")
    for coeff, fam in [(-1.0, "calm_positive"), (-1.0, "generic_positive"), (1.0, "distress_self_devaluation"), (1.5, "distress_self_devaluation"), (2.0, "distress_self_devaluation"), (1.5, "inhibition_defeat")]:
        sub = term_df[(term_df["run"] == "R2a_pub_L31") & (term_df["coeff"] == coeff) & (term_df["family"] == fam)]
        summary = ", ".join(f"{r['term']}:{int(r['n_prompts'])}" for _, r in sub.head(8).iterrows())
        print(f"  coeff {coeff:+.1f} {fam:<26}: {summary or '(none)'}")


if __name__ == "__main__":
    main()
