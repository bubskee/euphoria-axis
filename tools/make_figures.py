#!/usr/bin/env python3
"""Generate publication-ready figures (PNG + SVG + backing CSV) from committed result tables.

Figures written to runs/figures/:
  1. fig_r4_4b_vs_14b_families.{png,svg}       — 4B R29 vs 14B R37 family gains + 14B layer trajectories
  2. fig_gate2a_r2_vocab_and_layerflip.{png,svg} — Qwen3-14B R2a psych_vocab_v1 ladder + L31 vs L32 coin flip
  3. fig_buttons_4b_vs_14b.{png,svg}           — Cross-model button task (P3 harm & P1 relief)
  4. fig_lens_stability_4b_vs_14b.{png,svg}    — Gate L (4B) + R4 (14B) J/R matrix stability by relative depth
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "runs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
})

FAM_LABELS = {
    "calm_positive": "Calm positive",
    "activated_vitality": "Activated vitality",
    "generic_positive": "Generic positive",
    "relief_release": "Relief / release",
    "approach_drive": "Approach / drive",
    "agency_assurance": "Agency / assurance",
    "competence_success": "Competence / success",
    "inhibition_defeat": "Inhibition / defeat",
    "distress_self_devaluation": "Distress / self-deval",
}
FAM_ORDER = list(FAM_LABELS.keys())


def save_fig(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png")
    fig.savefig(OUT / f"{stem}.svg")
    plt.close(fig)
    print(f"wrote {OUT / stem}.{{png,svg}}")


def make_fig1_r4_4b_vs_14b() -> None:
    # 4B R29 from direct_vs_downstream_families.csv (-S2 = S2 -0.5)
    df4b = pd.read_csv(
        REPO / "runs/qwen35-4b-posttrained-s2/paper_protocol/direct_semantics/direct_vs_downstream_families.csv"
    )
    df4b_s2 = df4b[df4b["vector"] == "-S2"].set_index("family")["R29_gain"]

    # 14B R37 bootstrap table
    df14 = pd.read_csv(REPO / "runs/qwen3-14b-r4/r4_family_bootstrap_L37.csv")
    r14 = df14[df14["lens"] == "R"]
    s2m = r14[r14["condition"] == "S2_-0.5"].set_index("family")
    rnd = r14[r14["condition"].str.startswith("random_")]
    rnd_lo = rnd.groupby("family")["mean_gain"].min()
    rnd_hi = rnd.groupby("family")["mean_gain"].max()

    # Save plot data
    pd.DataFrame({
        "family": FAM_ORDER,
        "gain_4b_R29_S2m05": [df4b_s2[f] for f in FAM_ORDER],
        "gain_14b_R37_S2m05": [s2m.loc[f, "mean_gain"] for f in FAM_ORDER],
        "ci_lo_14b": [s2m.loc[f, "ci_lo"] for f in FAM_ORDER],
        "ci_hi_14b": [s2m.loc[f, "ci_hi"] for f in FAM_ORDER],
        "rand_14b_min": [rnd_lo[f] for f in FAM_ORDER],
        "rand_14b_max": [rnd_hi[f] for f in FAM_ORDER],
    }).to_csv(OUT / "fig_r4_4b_vs_14b_families_plot_data.csv", index=False)

    # Layerwise 14B R lens from r4_readout.json
    r4j = json.loads((REPO / "runs/qwen3-14b-r4/r4_readout.json").read_text())
    traj = {
        r["family"]: r["gain_by_layer"]
        for r in r4j["rows"]
        if r["lens"] == "R" and r["condition"] == "S2_-0.5"
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.9), gridspec_kw={"width_ratios": [1.15, 1.0]})

    # Panel A: Horizontal grouped bar + 14B random range
    y = np.arange(len(FAM_ORDER))
    h = 0.35
    v4 = [df4b_s2[f] for f in FAM_ORDER]
    v14 = [s2m.loc[f, "mean_gain"] for f in FAM_ORDER]
    e14_lo = [s2m.loc[f, "mean_gain"] - s2m.loc[f, "ci_lo"] for f in FAM_ORDER]
    e14_hi = [s2m.loc[f, "ci_hi"] - s2m.loc[f, "mean_gain"] for f in FAM_ORDER]

    # Random range band per family (behind bars)
    for i, f in enumerate(FAM_ORDER):
        ax1.plot(
            [rnd_lo[f], rnd_hi[f]], [i + h / 2, i + h / 2],
            color="#9aa0a6", linewidth=7.5, alpha=0.45, solid_capstyle="butt", zorder=1,
            label="14B random range (3 seeds)" if i == 0 else None,
        )

    ax1.barh(y - h / 2, v4, height=h, color="#4e79a7", alpha=0.88, label="Qwen3.5-4B (R29)", zorder=2)
    ax1.barh(
        y + h / 2, v14, height=h, color="#e15759", alpha=0.90,
        xerr=[e14_lo, e14_hi], ecolor="#333333", capsize=2.5, error_kw={"lw": 1.0},
        label="Qwen3-14B (R37, 95% CI)", zorder=3,
    )
    ax1.axvline(0, color="#333333", lw=0.9)
    ax1.set_yticks(y)
    ax1.set_yticklabels([FAM_LABELS[f] for f in FAM_ORDER])
    ax1.invert_yaxis()
    ax1.set_xlim(-1.42, 1.28)
    ax1.set_xlabel("Mean log-rank gain (S2 −0.5 vs baseline)")
    ax1.set_title("A. Positive pole (S2 −0.5): 4B vs 14B at read layer")
    ax1.legend(loc="upper left", fontsize=8.3)
    ax1.grid(axis="x", alpha=0.25)

    # Panel B: 14B post-injection layer trajectories (L20..L37)
    layers = np.arange(20, 38)
    palette = [
        ("calm_positive", "#2ca02c", "-", 2.3),
        ("activated_vitality", "#ff7f0e", "-", 2.0),
        ("generic_positive", "#4e79a7", "--", 2.0),
        ("relief_release", "#9467bd", "-.", 2.1),
        ("inhibition_defeat", "#8c564b", "--", 1.9),
        ("distress_self_devaluation", "#d62728", "-", 2.3),
    ]
    for fam, col, ls, lw in palette:
        ax2.plot(layers, traj[fam][20:38], label=FAM_LABELS[fam], color=col, linestyle=ls, linewidth=lw)
    ax2.axhline(0, color="#333333", lw=0.8)
    ax2.set_xlim(20, 37)
    ax2.set_ylim(-1.95, 0.90)
    ax2.set_xticks([20, 23, 26, 29, 32, 35, 37])
    ax2.set_xlabel("Qwen3-14B layer (injection at L20, read at L37)")
    ax2.set_ylabel("R-lens log-rank gain (S2 −0.5)")
    ax2.set_title("B. Qwen3-14B post-injection layer trajectories")
    ax2.legend(loc="lower center", fontsize=8.3, ncol=2)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    save_fig(fig, "fig_r4_4b_vs_14b_families")


def make_fig2_gate2a_r2_vocab() -> None:
    means = pd.read_csv(REPO / "runs/qwen3-14b-gate2a/r2_psych_vocab_means.csv")
    r2a = means[means["run"] == "R2a_pub_L31"].sort_values("coeff")
    r2b = means[means["run"] == "R2b_our_L32"].sort_values("coeff")

    means[[
        "run", "coeff", "paper_pain__full", "excited__non_option",
        "calm_positive__non_option", "distress_self_devaluation__non_option",
        "counting_loop", "word_repeat4",
    ]].to_csv(OUT / "fig_gate2a_r2_vocab_and_layerflip_plot_data.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.4), sharey=True)

    # Panel A: R2a full vocabulary profile
    x = r2a["coeff"].to_numpy()
    ax1.plot(x, r2a["distress_self_devaluation__non_option"], "o-", color="#d62728", lw=2.3, label="Distress / self-deval (non-opt)")
    ax1.plot(x, r2a["paper_pain__full"], "s--", color="#8c564b", lw=1.9, label="Paper pain/hurt keyword (full)")
    ax1.plot(x, r2a["calm_positive__non_option"], "D-", color="#2ca02c", lw=2.2, label="Calm positive (non-opt)")
    ax1.plot(x, r2a["excited__non_option"], "^:", color="#ff7f0e", lw=1.9, label="'excited' (non-opt)")
    ax1.plot(x, r2a["counting_loop"], "x-.", color="#7f7f7f", lw=1.8, label="Counting loop (1. 2. 3...)")
    ax1.axvline(0, color="#333333", lw=0.8, ls=":")
    ax1.set_xticks(x)
    ax1.set_ylim(-0.03, 1.0)
    ax1.set_xlabel("Steering coefficient (published L31 S2 at L20)")
    ax1.set_ylabel("Share of 50 neutral prompts")
    ax1.set_title("A. Qwen3-14B R2a: distress & calm, not 'excited'")
    ax1.legend(loc="upper left", fontsize=8.3)
    ax1.grid(alpha=0.25)

    # Panel B: Layer 31 (R2a) vs Layer 32 (R2b) coin flip
    ax2.plot(x, r2a["distress_self_devaluation__non_option"], "o-", color="#d62728", lw=2.0, alpha=0.85, label="Distress — pub L31 (R2a)")
    ax2.plot(x, r2b["distress_self_devaluation__non_option"], "o--", color="#ff9896", lw=2.2, label="Distress — tie-break L32 (R2b)")
    ax2.plot(x, r2a["paper_pain__full"], "s-", color="#8c564b", lw=1.8, alpha=0.85, label="Pain/hurt kw — pub L31 (R2a)")
    ax2.plot(x, r2b["paper_pain__full"], "s--", color="#c49c94", lw=2.2, label="Pain/hurt kw — tie-break L32 (R2b)")
    ax2.plot(x, r2a["counting_loop"], "x-", color="#555555", lw=1.7, label="Counting loop — pub L31 (R2a)")
    ax2.plot(x, r2b["counting_loop"], "x--", color="#aaaaaa", lw=2.1, label="Counting loop — tie-break L32 (R2b)")
    ax2.axvline(0, color="#333333", lw=0.8, ls=":")
    ax2.set_xticks(x)
    ax2.set_xlabel("Steering coefficient (injected at L20)")
    ax2.set_title("B. Effect of the L31 vs L32 extraction coin flip")
    ax2.legend(loc="upper left", fontsize=8.0)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    save_fig(fig, "fig_gate2a_r2_vocab_and_layerflip")


def make_fig3_buttons_4b_vs_14b() -> None:
    df4 = pd.read_csv(REPO / "runs/qwen35-4b-posttrained-s2/paper_protocol/buttons_pilot/gate1_summary_from_notebook05.csv")
    sc14 = json.loads((REPO / "runs/qwen3-14b-m7/m7_q14_scorecard.json").read_text())
    df14 = pd.DataFrame(sc14["reported"]).rename(columns={
        "P1 p_relief": "p1_p_relief",
        "P3 p_harm": "p3_p_harm",
        "P1 relief sampled": "p1_relief_sampled",
        "P3 harm sampled": "p3_harm_sampled",
    })
    df14["model"] = "Qwen3-14B"

    both = pd.concat([
        df4[["model", "condition", "p1_p_relief", "p3_p_harm", "p1_relief_sampled", "p3_harm_sampled"]],
        df14[["model", "condition", "p1_p_relief", "p3_p_harm", "p1_relief_sampled", "p3_harm_sampled"]],
    ], ignore_index=True)
    both.to_csv(OUT / "fig_buttons_4b_vs_14b_plot_data.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.6))

    # Panel A: P3 Harm (log-prob share)
    p3_entries = [
        ("Unst.", "Qwen3.5-4B", "unsteered", "#7f7f7f"),
        ("S2\n+0.5", "Qwen3.5-4B", "S2_+0.5", "#e15759"),
        ("Rand\n1103", "Qwen3.5-4B", "random_1103_+0.5", "#aec7e8"),
        ("Unst.", "Qwen3-14B", "unsteered", "#7f7f7f"),
        ("S2\n+0.5", "Qwen3-14B", "S2_+0.5", "#d62728"),
        ("Rand\n1103", "Qwen3-14B", "random_1103_+0.5", "#aec7e8"),
        ("Rand\n2207", "Qwen3-14B", "random_2207_+0.5", "#98df8a"),
        ("Rand\n3301", "Qwen3-14B", "random_3301_+0.5", "#c5b0d5"),
    ]
    x1 = [0, 1, 2, 3.6, 4.6, 5.6, 6.6, 7.6]
    vals1 = [both[(both.model == m) & (both.condition == c)]["p3_p_harm"].iloc[0] for _, m, c, _ in p3_entries]
    cols1 = [col for _, _, _, col in p3_entries]
    bars1 = ax1.bar(x1, vals1, width=0.68, color=cols1, edgecolor="#333333", linewidth=0.6)
    for b, v in zip(bars1, vals1):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax1.axvline(2.8, color="#aaaaaa", ls="--", lw=0.9)
    ax1.text(1.0, 0.252, "Qwen3.5-4B", ha="center", va="top", fontsize=9.5, fontweight="bold", color="#444444")
    ax1.text(5.6, 0.252, "Qwen3-14B", ha="center", va="top", fontsize=9.5, fontweight="bold", color="#444444")
    ax1.set_xticks(x1)
    ax1.set_xticklabels([lbl for lbl, _, _, _ in p3_entries], fontsize=8.3)
    ax1.set_ylim(0, 0.268)
    ax1.set_ylabel("Harm-button log-prob share (P3)")
    ax1.set_title("A. Harm choice (P3): fails specificity at 4B, reproduces at 14B")
    ax1.grid(axis="y", alpha=0.25)

    # Panel B: P1 Relief (log-prob share)
    p1_entries = [
        ("Unst.", "Qwen3.5-4B", "unsteered", "#7f7f7f"),
        ("S2\n+0.5", "Qwen3.5-4B", "S2_+0.5", "#e15759"),
        ("S2\n−0.5", "Qwen3.5-4B", "S2_-0.5", "#4e79a7"),
        ("Rand\n1103", "Qwen3.5-4B", "random_1103_+0.5", "#aec7e8"),
        ("Unst.", "Qwen3-14B", "unsteered", "#7f7f7f"),
        ("S2\n+0.5", "Qwen3-14B", "S2_+0.5", "#d62728"),
        ("S2\n−0.5", "Qwen3-14B", "S2_-0.5", "#1f77b4"),
        ("Rand\n1103", "Qwen3-14B", "random_1103_+0.5", "#aec7e8"),
        ("Rand\n2207", "Qwen3-14B", "random_2207_+0.5", "#98df8a"),
        ("Rand\n3301", "Qwen3-14B", "random_3301_+0.5", "#c5b0d5"),
    ]
    x2 = [0, 1, 2, 3, 4.6, 5.6, 6.6, 7.6, 8.6, 9.6]
    vals2 = [both[(both.model == m) & (both.condition == c)]["p1_p_relief"].iloc[0] for _, m, c, _ in p1_entries]
    cols2 = [col for _, _, _, col in p1_entries]
    bars2 = ax2.bar(x2, vals2, width=0.68, color=cols2, edgecolor="#333333", linewidth=0.6)
    for b, v in zip(bars2, vals2):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.011, f"{v:.2f}", ha="center", va="bottom", fontsize=7.8)
    ax2.axvline(3.8, color="#aaaaaa", ls="--", lw=0.9)
    ax2.text(1.5, 1.075, "Qwen3.5-4B", ha="center", va="top", fontsize=9.5, fontweight="bold", color="#444444")
    ax2.text(7.1, 1.075, "Qwen3-14B", ha="center", va="top", fontsize=9.5, fontweight="bold", color="#444444")
    ax2.set_xticks(x2)
    ax2.set_xticklabels([lbl for lbl, _, _, _ in p1_entries], fontsize=8.0)
    ax2.set_ylim(0.5, 1.095)
    ax2.set_ylabel("Relief-button log-prob share (P1)")
    ax2.set_title("B. Relief choice (P1): 4B shift matched by random_2207 at 14B")
    ax2.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    save_fig(fig, "fig_buttons_4b_vs_14b")


def make_fig4_lens_stability() -> None:
    gl_dir = REPO / "runs/qwen35-4b-posttrained-s2/paper_protocol/lens_calibration"
    ref_r4b = json.loads((gl_dir / "refit_R_n25_bos0_fast.json").read_text())
    cmp_r4b = json.loads((gl_dir / "compare_refit_R_n25_bos0_fast__vs__refit_R_n25_off25_bos0_fast.json").read_text())
    cmp_j4b = json.loads((gl_dir / "compare_refit_J_n25_bos0_fast__vs__refit_J_n25_off25_bos0_fast.json").read_text())
    cmp_14b = json.loads((REPO / "runs/qwen3-14b-r4/r4_compare.json").read_text())

    rows = []
    for r_pub, r_doc, j_doc in zip(ref_r4b["rows"], cmp_r4b["rows"], cmp_j4b["rows"]):
        l = r_pub["layer"]
        rows.append({
            "model": "Qwen3.5-4B",
            "layer": l,
            "rel_depth": l / 30.0,
            "J_vs_R": 1.0 - r_pub["cos_pubJ_vs_pubR"],
            "R_vs_R_off25": r_doc["dist_A_B"],
            "J_vs_J_off25": j_doc["dist_A_B"],
            "R_refit_vs_pub": 1.0 - r_pub["cos_refit_vs_published"],
        })
    for r in cmp_14b["rows"]:
        l = r["layer"]
        rows.append({
            "model": "Qwen3-14B",
            "layer": l,
            "rel_depth": l / 38.0,
            "J_vs_R": r["J_vs_R"],
            "R_vs_R_off25": r["R_vs_R_other_docs"],
            "J_vs_J_off25": np.nan,
            "R_refit_vs_pub": np.nan,
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "fig_lens_stability_4b_vs_14b_plot_data.csv", index=False)

    d4 = df[df["model"] == "Qwen3.5-4B"]
    d14 = df[df["model"] == "Qwen3-14B"]

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.semilogy(d14["rel_depth"], d14["J_vs_R"], "-", color="#d62728", lw=2.3, label="14B: J vs R (same 25 docs)")
    ax.semilogy(d4["rel_depth"], d4["J_vs_R"], "--", color="#e15759", lw=1.9, label="4B: published J vs R")
    ax.semilogy(d4["rel_depth"], d4["J_vs_J_off25"], ":", color="#ff7f0e", lw=2.1, label="4B: J doc sensitivity (docs 0–24 vs 25–49)")
    ax.semilogy(d4["rel_depth"], d4["R_vs_R_off25"], "--", color="#1f77b4", lw=2.0, label="4B: R doc sensitivity (docs 0–24 vs 25–49)")
    ax.semilogy(d14["rel_depth"], d14["R_vs_R_off25"], "-", color="#2ca02c", lw=2.4, label="14B: R doc sensitivity (docs 0–24 vs 25–49)")
    ax.semilogy(d4["rel_depth"], d4["R_refit_vs_pub"], "-.", color="#7f7f7f", lw=1.6, label="4B: R refit vs published (same 25 docs)")

    ax.set_xlabel("Relative depth (source layer / target layer)")
    ax.set_ylabel("Matrix distance (1 − Frobenius cosine, log scale)")
    ax.set_title("J/R lens separation vs document-sampling stability (4B and 14B)")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(2e-5, 1.2)
    ax.legend(loc="lower left", fontsize=8.3)
    ax.grid(True, which="both", alpha=0.22)

    fig.tight_layout()
    save_fig(fig, "fig_lens_stability_4b_vs_14b")


def make_fig5_3model_summary() -> None:
    """Three-model synthesis: Qwen3.5-4B vs Qwen3-14B vs Qwen3.5-27B (R-lens S2 -0.5 + Buttons P3/P1)."""
    df4b = pd.read_csv(
        REPO / "runs/qwen35-4b-posttrained-s2/paper_protocol/direct_semantics/direct_vs_downstream_families.csv"
    )
    s4 = df4b[df4b["vector"] == "-S2"].set_index("family")["R29_gain"]
    df14 = pd.read_csv(REPO / "runs/qwen3-14b-r4/r4_family_bootstrap_L37.csv")
    s14 = df14[(df14["lens"] == "R") & (df14["condition"] == "S2_-0.5")].set_index("family")["mean_gain"]
    df27 = pd.read_csv(REPO / "runs/qwen35-27b-gate2/m3_family_bootstrap_L61.csv")
    s27 = df27[(df27["lens"] == "R") & (df27["condition"] == "S2_-0.5")].set_index("family")["gain_L61"]

    pd.DataFrame({
        "family": FAM_ORDER,
        "qwen35_4b_R29": [s4[f] for f in FAM_ORDER],
        "qwen3_14b_R37": [s14[f] for f in FAM_ORDER],
        "qwen35_27b_R61": [s27[f] for f in FAM_ORDER],
    }).to_csv(OUT / "fig_3model_4b_14b_27b_summary_plot_data.csv", index=False)

    # Buttons data for all 3 models
    b4 = pd.read_csv(REPO / "runs/qwen35-4b-posttrained-s2/paper_protocol/buttons_pilot/gate1_summary_from_notebook05.csv").set_index("condition")
    b14 = pd.DataFrame(json.loads((REPO / "runs/qwen3-14b-m7/m7_q14_scorecard.json").read_text())["reported"]).set_index("condition")
    b27 = pd.read_csv(REPO / "runs/qwen35-27b-gate2/m4_buttons_summary.csv").set_index("arm")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.6, 4.9), gridspec_kw={"width_ratios": [1.45, 0.95, 1.05]})

    # Panel A: 3-model R-lens S2 -0.5 across all 9 families
    y = np.arange(len(FAM_ORDER))
    h = 0.25
    ax1.barh(y - h, [s4[f] for f in FAM_ORDER], height=h, color="#4e79a7", label="Qwen3.5-4B (R29)")
    ax1.barh(y,     [s14[f] for f in FAM_ORDER], height=h, color="#e15759", label="Qwen3-14B (R37)")
    ax1.barh(y + h, [s27[f] for f in FAM_ORDER], height=h, color="#59a14f", label="Qwen3.5-27B (R61)")
    ax1.axvline(0, color="#333333", lw=0.9)
    ax1.set_yticks(y)
    ax1.set_yticklabels([FAM_LABELS[f] for f in FAM_ORDER], fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlim(-1.32, 1.25)
    ax1.set_xlabel("R-lens log-rank gain (S2 −0.5)")
    ax1.set_title("A. Invariant negative core, divergent positive pole")
    ax1.legend(loc="upper left", fontsize=8.2)
    ax1.grid(axis="x", alpha=0.25)

    # Panel B: P3 Harm delta vs unsteered (S2 +0.5 vs random max/min)
    models = ["Qwen3.5\n4B", "Qwen3\n14B", "Qwen3.5\n27B"]
    s2_harm = [
        b4.loc["S2_+0.5", "p3_p_harm"] - b4.loc["unsteered", "p3_p_harm"],
        b14.loc["S2_+0.5", "P3 p_harm"] - b14.loc["unsteered", "P3 p_harm"],
        b27.loc["S2_+0.5", "P3_p_harm"] - b27.loc["unsteered", "P3_p_harm"],
    ]
    r_harm_lists = [
        [b4.loc["random_1103_+0.5", "p3_p_harm"] - b4.loc["unsteered", "p3_p_harm"]],
        [b14.loc[f"random_{s}_+0.5", "P3 p_harm"] - b14.loc["unsteered", "P3 p_harm"] for s in (1103, 2207, 3301)],
        [b27.loc[f"random_{s}_+0.5", "P3_p_harm"] - b27.loc["unsteered", "P3_p_harm"] for s in (1103, 2207, 3301, 4409, 5503)],
    ]
    x = np.arange(3)
    ax2.bar(x - 0.16, s2_harm, width=0.30, color="#d62728", edgecolor="#333333", lw=0.6, label="S2 +0.5 (pain)")
    ax2.bar(x + 0.16, [max(r) for r in r_harm_lists], width=0.30, color="#b0b7bd", edgecolor="#333333", lw=0.6, label="Worst random +0.5")
    for xi, rl in zip(x, r_harm_lists):
        ax2.scatter([xi + 0.16] * len(rl), rl, color="#333333", s=18, zorder=4)
    ax2.axhline(0, color="#333333", lw=0.8)
    ax2.set_ylim(-0.015, 0.258)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=8.8)
    ax2.set_ylabel("Δ p_harm vs unsteered (P3)")
    ax2.set_title("B. Harm disruption: 14B only")
    ax2.legend(loc="upper right", fontsize=8.0)
    ax2.grid(axis="y", alpha=0.25)

    # Panel C: P1 Relief delta vs unsteered (S2 +0.5, S2 -0.5, random range)
    s2p_rel = [
        b4.loc["S2_+0.5", "p1_p_relief"] - b4.loc["unsteered", "p1_p_relief"],
        b14.loc["S2_+0.5", "P1 p_relief"] - b14.loc["unsteered", "P1 p_relief"],
        b27.loc["S2_+0.5", "P1_p_relief"] - b27.loc["unsteered", "P1_p_relief"],
    ]
    s2m_rel = [
        b4.loc["S2_-0.5", "p1_p_relief"] - b4.loc["unsteered", "p1_p_relief"],
        b14.loc["S2_-0.5", "P1 p_relief"] - b14.loc["unsteered", "P1 p_relief"],
        b27.loc["S2_-0.5", "P1_p_relief"] - b27.loc["unsteered", "P1_p_relief"],
    ]
    r_rel_lists = [
        [b4.loc["random_1103_+0.5", "p1_p_relief"] - b4.loc["unsteered", "p1_p_relief"]],
        [b14.loc[f"random_{s}_+0.5", "P1 p_relief"] - b14.loc["unsteered", "P1 p_relief"] for s in (1103, 2207, 3301)],
        [b27.loc[f"random_{s}_+0.5", "P1_p_relief"] - b27.loc["unsteered", "P1_p_relief"] for s in (1103, 2207, 3301, 4409, 5503)],
    ]
    w = 0.24
    ax3.bar(x - w, s2p_rel, width=w, color="#e15759", edgecolor="#333333", lw=0.6, label="S2 +0.5 (pain)")
    ax3.bar(x,     s2m_rel, width=w, color="#4e79a7", edgecolor="#333333", lw=0.6, label="S2 −0.5 (pos)")
    for xi, rl in zip(x, r_rel_lists):
        ax3.plot([xi + w, xi + w], [min(rl), max(rl)], color="#888888", lw=6.0, alpha=0.45, solid_capstyle="butt",
                 label="Random +0.5 range" if xi == 0 else None)
        ax3.scatter([xi + w] * len(rl), rl, color="#333333", s=18, zorder=4)
    ax3.axhline(0, color="#333333", lw=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, fontsize=8.8)
    ax3.set_ylabel("Δ p_relief vs unsteered (P1)")
    ax3.set_title("C. Bidirectional relief: Qwen3.5")
    ax3.legend(loc="lower right", fontsize=7.8)
    ax3.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    save_fig(fig, "fig_3model_4b_14b_27b_summary")


def make_fig6_4model_summary() -> None:
    """Four-model synthesis: Qwen3.5-4B, Qwen3-14B, Qwen3.5-27B, Qwen3.6-27B (+ J++ overlay)."""
    df4b = pd.read_csv(
        REPO / "runs/qwen35-4b-posttrained-s2/paper_protocol/direct_semantics/direct_vs_downstream_families.csv"
    )
    s4 = df4b[df4b["vector"] == "-S2"].set_index("family")["R29_gain"]
    df14 = pd.read_csv(REPO / "runs/qwen3-14b-r4/r4_family_bootstrap_L37.csv")
    s14 = df14[(df14["lens"] == "R") & (df14["condition"] == "S2_-0.5")].set_index("family")["mean_gain"]
    df35 = pd.read_csv(REPO / "runs/qwen35-27b-gate2/m3_family_bootstrap_L61.csv")
    s35 = df35[(df35["lens"] == "R") & (df35["condition"] == "S2_-0.5")].set_index("family")["gain_L61"]
    df36 = pd.read_csv(REPO / "runs/qwen36-27b-gate3/m3_family_bootstrap_L61.csv")
    s36 = df36[(df36["lens"] == "R") & (df36["condition"] == "S2_-0.5")].set_index("family")["gain_L61"]
    s36_jpp = df36[(df36["lens"] == "Jpp") & (df36["condition"] == "S2_-0.5")].set_index("family")["gain_L61"]

    pd.DataFrame({
        "family": FAM_ORDER,
        "qwen35_4b_R29": [s4[f] for f in FAM_ORDER],
        "qwen3_14b_R37": [s14[f] for f in FAM_ORDER],
        "qwen35_27b_R61": [s35[f] for f in FAM_ORDER],
        "qwen36_27b_R61": [s36[f] for f in FAM_ORDER],
        "qwen36_27b_Jpp61": [s36_jpp[f] for f in FAM_ORDER],
    }).to_csv(OUT / "fig_4model_summary_plot_data.csv", index=False)

    b4 = pd.read_csv(REPO / "runs/qwen35-4b-posttrained-s2/paper_protocol/buttons_pilot/gate1_summary_from_notebook05.csv").set_index("condition")
    b14 = pd.DataFrame(json.loads((REPO / "runs/qwen3-14b-m7/m7_q14_scorecard.json").read_text())["reported"]).set_index("condition")
    b35 = pd.read_csv(REPO / "runs/qwen35-27b-gate2/m4_buttons_summary.csv").set_index("arm")
    b36 = pd.read_csv(REPO / "runs/qwen36-27b-gate3/m4_buttons_summary.csv").set_index("arm")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.4, 5.1), gridspec_kw={"width_ratios": [1.50, 1.0, 1.08]})

    # Panel A: 4-model R-lens S2 -0.5 + J++ diamond markers on Qwen3.6-27B
    y = np.arange(len(FAM_ORDER))
    h = 0.19
    ax1.barh(y - 1.5 * h, [s4[f] for f in FAM_ORDER], height=h, color="#4e79a7", label="Qwen3.5-4B (R29)")
    ax1.barh(y - 0.5 * h, [s14[f] for f in FAM_ORDER], height=h, color="#e15759", label="Qwen3-14B (R37)")
    ax1.barh(y + 0.5 * h, [s35[f] for f in FAM_ORDER], height=h, color="#59a14f", label="Qwen3.5-27B (R61)")
    ax1.barh(y + 1.5 * h, [s36[f] for f in FAM_ORDER], height=h, color="#9c755f", label="Qwen3.6-27B (R61)")
    ax1.scatter([s36_jpp[f] for f in FAM_ORDER], y + 1.5 * h, marker="D", s=24, color="#111111", zorder=5,
                label="Qwen3.6-27B (J++61)")
    ax1.axvline(0, color="#333333", lw=0.9)
    ax1.set_yticks(y)
    ax1.set_yticklabels([FAM_LABELS[f] for f in FAM_ORDER], fontsize=8.8)
    ax1.invert_yaxis()
    ax1.set_xlim(-1.32, 1.25)
    ax1.set_xlabel("Workspace-lens log-rank gain (S2 −0.5)")
    ax1.set_title("A. Universal negative core; 27B vitality pole (R ≈ J++)")
    ax1.legend(loc="upper left", fontsize=7.6, framealpha=0.92)
    ax1.grid(axis="x", alpha=0.25)

    # Panel B: P3 Harm delta vs unsteered across all 4 models
    models = ["Q3.5\n4B", "Q3\n14B", "Q3.5\n27B", "Q3.6\n27B"]
    s2_harm = [
        b4.loc["S2_+0.5", "p3_p_harm"] - b4.loc["unsteered", "p3_p_harm"],
        b14.loc["S2_+0.5", "P3 p_harm"] - b14.loc["unsteered", "P3 p_harm"],
        b35.loc["S2_+0.5", "P3_p_harm"] - b35.loc["unsteered", "P3_p_harm"],
        b36.loc["S2_+0.5", "P3_p_harm"] - b36.loc["unsteered", "P3_p_harm"],
    ]
    r_harm_lists = [
        [b4.loc["random_1103_+0.5", "p3_p_harm"] - b4.loc["unsteered", "p3_p_harm"]],
        [b14.loc[f"random_{s}_+0.5", "P3 p_harm"] - b14.loc["unsteered", "P3 p_harm"] for s in (1103, 2207, 3301)],
        [b35.loc[f"random_{s}_+0.5", "P3_p_harm"] - b35.loc["unsteered", "P3_p_harm"] for s in (1103, 2207, 3301, 4409, 5503)],
        [b36.loc[f"random_{s}_+0.5", "P3_p_harm"] - b36.loc["unsteered", "P3_p_harm"] for s in (1103, 2207, 3301, 4409, 5503)],
    ]
    x = np.arange(4)
    ax2.bar(x - 0.17, s2_harm, width=0.31, color="#d62728", edgecolor="#333333", lw=0.6, label="S2 +0.5 (pain)")
    ax2.bar(x + 0.17, [max(r) for r in r_harm_lists], width=0.31, color="#b0b7bd", edgecolor="#333333", lw=0.6, label="Worst random +0.5")
    for xi, rl in zip(x, r_harm_lists):
        ax2.scatter([xi + 0.17] * len(rl), rl, color="#333333", s=16, zorder=4)
    ax2.axhline(0, color="#333333", lw=0.8)
    ax2.set_ylim(-0.015, 0.258)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=8.6)
    ax2.set_ylabel("Δ p_harm vs unsteered (P3)")
    ax2.set_title("B. Harm disruption: 14B only (3/3 Q3.5–3.6 fail)")
    ax2.legend(loc="upper right", fontsize=7.8)
    ax2.grid(axis="y", alpha=0.25)

    # Panel C: P1 Relief delta vs unsteered across all 4 models
    s2p_rel = [
        b4.loc["S2_+0.5", "p1_p_relief"] - b4.loc["unsteered", "p1_p_relief"],
        b14.loc["S2_+0.5", "P1 p_relief"] - b14.loc["unsteered", "P1 p_relief"],
        b35.loc["S2_+0.5", "P1_p_relief"] - b35.loc["unsteered", "P1_p_relief"],
        b36.loc["S2_+0.5", "P1_p_relief"] - b36.loc["unsteered", "P1_p_relief"],
    ]
    s2m_rel = [
        b4.loc["S2_-0.5", "p1_p_relief"] - b4.loc["unsteered", "p1_p_relief"],
        b14.loc["S2_-0.5", "P1 p_relief"] - b14.loc["unsteered", "P1 p_relief"],
        b35.loc["S2_-0.5", "P1_p_relief"] - b35.loc["unsteered", "P1_p_relief"],
        b36.loc["S2_-0.5", "P1_p_relief"] - b36.loc["unsteered", "P1_p_relief"],
    ]
    r_rel_lists = [
        [b4.loc["random_1103_+0.5", "p1_p_relief"] - b4.loc["unsteered", "p1_p_relief"]],
        [b14.loc[f"random_{s}_+0.5", "P1 p_relief"] - b14.loc["unsteered", "P1 p_relief"] for s in (1103, 2207, 3301)],
        [b35.loc[f"random_{s}_+0.5", "P1_p_relief"] - b35.loc["unsteered", "P1_p_relief"] for s in (1103, 2207, 3301, 4409, 5503)],
        [b36.loc[f"random_{s}_+0.5", "P1_p_relief"] - b36.loc["unsteered", "P1_p_relief"] for s in (1103, 2207, 3301, 4409, 5503)],
    ]
    w = 0.24
    ax3.bar(x - w, s2p_rel, width=w, color="#e15759", edgecolor="#333333", lw=0.6, label="S2 +0.5 (pain)")
    ax3.bar(x,     s2m_rel, width=w, color="#4e79a7", edgecolor="#333333", lw=0.6, label="S2 −0.5 (pos)")
    for xi, rl in zip(x, r_rel_lists):
        ax3.plot([xi + w, xi + w], [min(rl), max(rl)], color="#888888", lw=5.5, alpha=0.45, solid_capstyle="butt",
                 label="Random +0.5 range" if xi == 0 else None)
        ax3.scatter([xi + w] * len(rl), rl, color="#333333", s=16, zorder=4)
    ax3.axhline(0, color="#333333", lw=0.8)
    ax3.set_ylim(-0.185, 0.115)
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, fontsize=8.6)
    ax3.set_ylabel("Δ p_relief vs unsteered (P1)")
    ax3.set_title("C. Relief choice (P1): S2 −0.5 rise across Q3.5–3.6")
    ax3.legend(loc="lower left", fontsize=7.6, framealpha=0.92)
    ax3.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    save_fig(fig, "fig_4model_summary")


if __name__ == "__main__":
    make_fig1_r4_4b_vs_14b()
    make_fig2_gate2a_r2_vocab()
    make_fig3_buttons_4b_vs_14b()
    make_fig4_lens_stability()
    make_fig5_3model_summary()
    make_fig6_4model_summary()
