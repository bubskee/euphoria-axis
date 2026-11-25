#!/usr/bin/env python3
"""Analyze Qwen3-14B R4 J/R lens readouts (bootstrap CIs, LOTO, probe ranks, J vs R, 4B comparison).

Inputs (all under runs/qwen3-14b-r4/ and runs/qwen35-4b-posttrained-s2/):
  - runs/qwen3-14b-r4/r4_ladder_ranks.npz  (per-prompt/layer/lens/probe ranks; gitignored)
  - runs/qwen3-14b-r4/r4_probe_terms.csv
  - runs/qwen3-14b-r4/r4_readout.json
  - runs/qwen3-14b-r4/r4_compare.json
  - runs/qwen3-14b-r4/r4_q14_spec.json
  - runs/qwen35-4b-posttrained-s2/paper_protocol/dense_sweep/lens_glance_R29.csv

Outputs (committed CSVs under runs/qwen3-14b-r4/):
  - r4_family_bootstrap_L37.csv
  - r4_s2_vs_random_L37.csv
  - r4_probe_gains_L37.csv
  - r4_loto_L37.csv
  - r4_layerwise_JR.csv
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent.parent
R4_DIR = REPO / "runs" / "qwen3-14b-r4"
R4B_DIR = REPO / "runs" / "qwen35-4b-posttrained-s2" / "paper_protocol"

BOOT_N = 20_000
BOOT_SEED = 20260925
FOCUS6 = [
    "generic_positive",
    "activated_vitality",
    "relief_release",
    "agency_assurance",
    "approach_drive",
    "distress_self_devaluation",
]


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    """Mean and 95% percentile bootstrap CI over axis 0 (prompts)."""
    n = values.shape[0]
    idx = rng.integers(0, n, size=(BOOT_N, n))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


def main() -> None:
    readout = json.loads((R4_DIR / "r4_readout.json").read_text())
    compare = json.loads((R4_DIR / "r4_compare.json").read_text())
    probes = pd.read_csv(R4_DIR / "r4_probe_terms.csv")
    npz = np.load(R4_DIR / "r4_ladder_ranks.npz")

    read_layer = int(readout["read_layer"])
    inject_layer = int(readout["inject_layer"])
    assert read_layer == 37 and inject_layer == 20
    assert len(probes) == int(readout["n_probes"]) == 60

    # 1. Exact reproduction gate against committed r4_readout.json
    base_ranks = npz["baseline"]  # [50, 38, 2, 60]
    n_prompts, n_layers, n_lenses, n_probes = base_ranks.shape
    assert (n_prompts, n_layers, n_lenses, n_probes) == (50, 38, 2, 60)

    fam_arr = probes["family"].to_numpy()
    families = list(pd.unique(fam_arr))
    conds = [c for c in npz.files if c != "baseline"]

    gains = {c: np.log(base_ranks / npz[c]) for c in conds}  # [50, 38, 2, 60]

    max_diff = 0.0
    for row in readout["rows"]:
        c = row["condition"]
        li = 0 if row["lens"] == "J" else 1
        fmask = fam_arr == row["family"]
        calc_by_layer = gains[c][:, :, li, :][:, :, fmask].mean(axis=(0, 2))
        diff = np.max(np.abs(calc_by_layer - np.array(row["gain_by_layer"])))
        max_diff = max(max_diff, float(diff))
    assert max_diff < 1e-12, f"NPZ mismatch vs r4_readout.json: {max_diff}"
    print(f"[PASS] r4_ladder_ranks.npz reproduces r4_readout.json exactly (max |diff| = {max_diff:.2e})")

    # Verify probe set overlap with 4B
    vocab = json.loads((R4B_DIR / "s1_comparison" / "psych_vocab_v1.json").read_text())
    vocab_sha = hashlib.sha256((R4B_DIR / "s1_comparison" / "psych_vocab_v1.json").read_bytes()).hexdigest()
    print(f"psych_vocab_v1 sha256: {vocab_sha[:16]}... | 14B single-token probes: {len(probes)}/74")
    all_terms = [(f, t) for f, ts in vocab["families"].items() for t in ts]
    included = set(zip(probes["family"], probes["term"]))
    dropped = [f"{f}:{t}" for f, t in all_terms if (f, t) not in included]
    print(f"Dropped multi-token terms ({len(dropped)}): {', '.join(dropped)}")

    # Verify R4-P1 frozen prediction
    r4_p1 = readout["R4_P1"]
    print(f"\nR4-P1 (frozen specificity bar): {'PASS' if r4_p1['pass'] else 'FAIL'} | ratios = { {k: round(v, 4) for k, v in r4_p1['ratios'].items()} }")

    # 2. Bootstrap CIs for every family x condition x lens at L37 and L20
    boot_rows = []
    for layer in (inject_layer, read_layer):
        for lens_idx, lens_name in enumerate(("J", "R")):
            for c in conds:
                for f in families:
                    fmask = fam_arr == f
                    per_prompt = gains[c][:, layer, lens_idx, :][:, fmask].mean(axis=1)
                    rng = np.random.default_rng(BOOT_SEED)
                    mean, lo, hi = bootstrap_ci(per_prompt, rng)
                    boot_rows.append({
                        "layer": layer,
                        "lens": lens_name,
                        "condition": c,
                        "family": f,
                        "mean_gain": mean,
                        "ci_lo": lo,
                        "ci_hi": hi,
                        "ci_excludes_0": bool((lo > 0) or (hi < 0)),
                        "n_probes": int(fmask.sum()),
                    })
    boot_df = pd.DataFrame(boot_rows)
    boot_df[boot_df["layer"] == read_layer].to_csv(R4_DIR / "r4_family_bootstrap_L37.csv", index=False)

    # 3. Paired differences: S2 -0.5 minus each random at L37 (R and J)
    diff_rows = []
    for lens_idx, lens_name in enumerate(("J", "R")):
        for s2_cond in ("S2_-0.5", "S2_+0.5"):
            for seed in ("1103", "2207", "3301"):
                rc = f"random_{seed}_-0.5"
                for f in families:
                    fmask = fam_arr == f
                    s2_p = gains[s2_cond][:, read_layer, lens_idx, :][:, fmask].mean(axis=1)
                    r_p = gains[rc][:, read_layer, lens_idx, :][:, fmask].mean(axis=1)
                    rng = np.random.default_rng(BOOT_SEED)
                    mean, lo, hi = bootstrap_ci(s2_p - r_p, rng)
                    diff_rows.append({
                        "layer": read_layer,
                        "lens": lens_name,
                        "s2_condition": s2_cond,
                        "random_condition": rc,
                        "family": f,
                        "diff_mean": mean,
                        "ci_lo": lo,
                        "ci_hi": hi,
                        "ci_excludes_0": bool((lo > 0) or (hi < 0)),
                    })
    diff_df = pd.DataFrame(diff_rows)
    diff_df.to_csv(R4_DIR / "r4_s2_vs_random_L37.csv", index=False)

    # 4. Probe-level gains & geometric-mean ranks at L37 (R lens)
    probe_rows = []
    for pi, prow in probes.iterrows():
        rec = {
            "family": prow["family"],
            "term": prow["term"],
            "token_id": int(prow["token_id"]),
            "base_gmean_rank_R37": float(np.exp(np.log(base_ranks[:, read_layer, 1, pi]).mean())),
        }
        for c in conds:
            g_p = gains[c][:, read_layer, 1, pi]
            rec[f"gain_R37__{c}"] = float(g_p.mean())
            rec[f"gmean_rank_R37__{c}"] = float(np.exp(np.log(npz[c][:, read_layer, 1, pi]).mean()))
        probe_rows.append(rec)
    probe_df = pd.DataFrame(probe_rows)
    probe_df.to_csv(R4_DIR / "r4_probe_gains_L37.csv", index=False)

    # 5. Leave-One-Term-Out (LOTO) at L37 (R lens) for S2_-0.5 and S2_+0.5
    loto_rows = []
    for c in ("S2_-0.5", "S2_+0.5"):
        for f in families:
            f_idx = np.where(fam_arr == f)[0]
            full_val = float(gains[c][:, read_layer, 1, f_idx].mean())
            loto_vals = []
            for drop_i in f_idx:
                keep = [j for j in f_idx if j != drop_i]
                loto_vals.append((probes.iloc[drop_i]["term"], float(gains[c][:, read_layer, 1, keep].mean())))
            vals_only = [v for _, v in loto_vals]
            min_term, min_v = min(loto_vals, key=lambda kv: kv[1])
            max_term, max_v = max(loto_vals, key=lambda kv: kv[1])
            same_sign = all((v > 0) == (full_val > 0) for v in vals_only)
            # Dominant single term = drop that shrinks |gain| the most
            most_shrink_term, most_shrink_v = min(loto_vals, key=lambda kv: abs(kv[1]))
            loto_rows.append({
                "condition": c,
                "family": f,
                "n_probes": len(f_idx),
                "full_gain_R37": full_val,
                "loto_min": min_v,
                "loto_min_dropped": min_term,
                "loto_max": max_v,
                "loto_max_dropped": max_term,
                "loto_smallest_abs": most_shrink_v,
                "loto_dominant_term": most_shrink_term,
                "all_same_sign": bool(same_sign),
            })
    loto_df = pd.DataFrame(loto_rows)
    loto_df.to_csv(R4_DIR / "r4_loto_L37.csv", index=False)

    # 6. Layerwise J vs R probe-level Spearman + matrix comparison
    cmp_map = {int(r["layer"]): r for r in compare["rows"]}
    layer_rows = []
    for l in range(n_layers):
        row = {
            "layer": l,
            "matrix_1_minus_cos_J_vs_R": cmp_map[l]["J_vs_R"],
            "matrix_1_minus_cos_R_vs_R_off25": cmp_map[l]["R_vs_R_other_docs"],
        }
        for c in ("S2_-0.5", "S2_+0.5"):
            gJ = gains[c][:, l, 0, :].mean(axis=0)
            gR = gains[c][:, l, 1, :].mean(axis=0)
            if np.allclose(gJ, 0) and np.allclose(gR, 0):
                row[f"spearman_JR__{c}"] = np.nan
            else:
                row[f"spearman_JR__{c}"] = float(spearmanr(gJ, gR).statistic)
        layer_rows.append(row)
    layer_df = pd.DataFrame(layer_rows)
    layer_df.to_csv(R4_DIR / "r4_layerwise_JR.csv", index=False)

    # 7. Print rich console summary for checkpoint authoring
    print("\n=== R37 Family Gains with 95% Bootstrap CIs (R lens) ===")
    r37 = boot_df[(boot_df["layer"] == read_layer) & (boot_df["lens"] == "R")]
    for f in families:
        sub = r37[r37["family"] == f].set_index("condition")
        s2m = sub.loc["S2_-0.5"]
        s2p = sub.loc["S2_+0.5"]
        rnds = [sub.loc[f"random_{s}_-0.5", "mean_gain"] for s in ("1103", "2207", "3301")]
        print(
            f"  {f:<26} S2-0.5: {s2m['mean_gain']:+6.3f} [{s2m['ci_lo']:+6.3f}, {s2m['ci_hi']:+6.3f}] | "
            f"S2+0.5: {s2p['mean_gain']:+6.3f} [{s2p['ci_lo']:+6.3f}, {s2p['ci_hi']:+6.3f}] | "
            f"rand: ({rnds[0]:+.2f}, {rnds[1]:+.2f}, {rnds[2]:+.2f})"
        )

    print("\n=== S2 -0.5 vs ALL 3 randoms at R37 (does CI exclude 0 against EVERY random?) ===")
    d37 = diff_df[(diff_df["lens"] == "R") & (diff_df["s2_condition"] == "S2_-0.5")]
    for f in families:
        sub = d37[d37["family"] == f]
        beats_pos = all(sub["ci_lo"] > 0)
        beats_neg = all(sub["ci_hi"] < 0)
        status = "BEATS ALL (+)" if beats_pos else ("BEATS ALL (-)" if beats_neg else "mixed")
        vals = ", ".join(f"{r['random_condition'][7:11]}:{r['diff_mean']:+.2f}[{r['ci_lo']:+.2f},{r['ci_hi']:+.2f}]" for _, r in sub.iterrows())
        print(f"  {f:<26} {status:<14} | {vals}")

    print("\n=== Top/Bottom Probes at R37 ===")
    for col, title in [("gain_R37__S2_-0.5", "S2 -0.5"), ("gain_R37__S2_+0.5", "S2 +0.5")]:
        rcol = "gmean_rank_R37__" + title.replace(" ", "_")
        top = probe_df.nlargest(8, col)[["term", "family", col, rcol, "base_gmean_rank_R37"]]
        bot = probe_df.nsmallest(6, col)[["term", "family", col, rcol, "base_gmean_rank_R37"]]
        print(f"\n  {title} top risers:")
        for _, r in top.iterrows():
            print(f"    {r['term']:<14} ({r['family']:<25}) gain={r[col]:+6.3f}  rank: {r['base_gmean_rank_R37']:7.1f} -> {r[rcol]:7.1f}")
        print(f"  {title} biggest drops:")
        for _, r in bot.iterrows():
            print(f"    {r['term']:<14} ({r['family']:<25}) gain={r[col]:+6.3f}  rank: {r['base_gmean_rank_R37']:7.1f} -> {r[rcol]:7.1f}")

    exc = probe_df[probe_df["term"] == "excited"].iloc[0]
    print(
        f"\n  'excited' at R37 (14B): base rank {exc['base_gmean_rank_R37']:.1f} -> "
        f"S2-0.5 rank {exc['gmean_rank_R37__S2_-0.5']:.1f} (gain {exc['gain_R37__S2_-0.5']:+.3f}), "
        f"S2+0.5 rank {exc['gmean_rank_R37__S2_+0.5']:.1f} (gain {exc['gain_R37__S2_+0.5']:+.3f})"
    )

    print("\n=== LOTO Summary at R37 (S2 -0.5 and S2 +0.5) ===")
    for _, r in loto_df.iterrows():
        print(
            f"  {r['condition']} {r['family']:<26} full={r['full_gain_R37']:+6.3f}  "
            f"LOTO=[{r['loto_min']:+6.3f}, {r['loto_max']:+6.3f}]  "
            f"same_sign={str(r['all_same_sign']):<5}  dom={r['loto_dominant_term']} ({r['loto_smallest_abs']:+.3f})"
        )

    print("\n=== J vs R Spearman across post-injection layers (20, 24, 28, 32, 37) ===")
    sub_l = layer_df[layer_df["layer"].isin([0, 10, 20, 24, 28, 32, 37])]
    print(sub_l.to_string(index=False))

    # 8. Side-by-side comparison vs 4B R29
    df4b = pd.read_csv(R4B_DIR / "dense_sweep" / "lens_glance_R29.csv").set_index("condition")
    # Also read 4B calm/inhibition/competence from direct_vs_downstream_families.csv if available
    ds4b = pd.read_csv(R4B_DIR / "direct_semantics" / "direct_vs_downstream_families.csv")
    print("\n=== 4B R29 vs 14B R37 (S2 -0.5 & S2 +0.5, R lens) ===")
    for f in FOCUS6:
        v4m = df4b.loc["S2_-0.5", f]
        v4p = df4b.loc["S2_+0.5", f]
        v14m = r37[(r37["family"] == f) & (r37["condition"] == "S2_-0.5")]["mean_gain"].iloc[0]
        v14p = r37[(r37["family"] == f) & (r37["condition"] == "S2_+0.5")]["mean_gain"].iloc[0]
        print(f"  {f:<26} S2-0.5: 4B={v4m:+6.3f}  14B={v14m:+6.3f}  |  S2+0.5: 4B={v4p:+6.3f}  14B={v14p:+6.3f}")


if __name__ == "__main__":
    main()
