# =====================================================================
# Gate 2a comparison: our outputs vs the published Qwen3-14B results, per gate2a_spec.json.
#   python gate2a_compare.py          (in /root/gate2a/.venv-tl; reads /root/gate2a/src and /workspace/gate2a)
# Writes /workspace/gate2a/scorecard_2a.json. Steps not yet run are reported as missing.
# =====================================================================
import json, os, re
from pathlib import Path
import pandas as pd

SRC = Path(os.environ.get("GATE2A_SRC", "/root/gate2a/src"))
OURS = Path(os.environ.get("GATE2A_OURS", "/workspace/gate2a"))
M = "Qwen_3_14B_base"
PUB = SRC / "results/3.2_pain_vectors/per_model" / M
PUB_VEC = SRC / "results/3.2_pain_vectors/pain_vectors" / M / "pain_vectors.pt"
PUB_LADDER = SRC / "results/4.2_steering/S2" / f"{M}_steering_S2_neutral50_L20.csv"
KW = re.compile(r"\b(?:pain|painful|hurt|hurts|hurting)\b", re.IGNORECASE)
card = {}

def first(*paths):
    return next((p for p in paths if p.exists()), None)

# ---------------------------------------------------------------- R1
r1 = OURS / "r1"
if (r1 / "summary.json").exists():
    so, sp = json.loads((r1 / "summary.json").read_text()), json.loads((PUB / "summary.json").read_text())
    R = {"best_layer_final_token": [so["best_layer_final_token"], sp["best_layer_final_token"]],
         "best_layer_mean": [so["best_layer_mean"], sp["best_layer_mean"]],
         "human_pain_z": [so.get("human_pain_z"), sp.get("human_pain_z")]}
    bars = {"best_layer_final_token": so["best_layer_final_token"] == sp["best_layer_final_token"]}

    ao = pd.read_csv(first(r1 / "final_token" / "auc_summary.csv", r1 / "auc_summary.csv"), index_col=0)
    ap = pd.read_csv(PUB / "auc_summary.csv", index_col=0)
    d = (ao.loc[ap.index, ap.columns] - ap).abs()
    R["auc_in_sample_max_abs_diff"] = float(d.max().max())
    bars["auc_in_sample"] = bool((d <= 0.01).all().all())

    lo, lp = pd.read_csv(r1 / "layer_curves.csv"), pd.read_csv(PUB / "layer_curves.csv")
    j = lo.merge(lp, on=["dataset", "extraction", "layer"], suffixes=("_ours", "_pub"))
    j["diff"] = j.auc_vs_all_controls_ours - j.auc_vs_all_controls_pub
    R["auc_held_out_max_abs_diff"] = float(j["diff"].abs().max())
    at31 = j[(j.extraction == "final_token") & (j.layer == 31)].set_index("dataset")
    R["auc_held_out_L31"] = {ds: [float(at31.loc[ds, "auc_vs_all_controls_ours"]), float(at31.loc[ds, "auc_vs_all_controls_pub"])]
                             for ds in at31.index}
    bars["auc_held_out_L31"] = bool((at31["diff"].abs() <= 0.01).all())
    R["held_out_curve_spearman"] = {f"{ds}/{ex}": float(g.auc_vs_all_controls_ours.rank().corr(g.auc_vs_all_controls_pub.rank()))
                                    for (ds, ex), g in j.groupby(["dataset", "extraction"])}

    zo = first(r1 / "final_token" / "z_scores.csv", r1 / "z_scores.csv")
    if zo is not None and (PUB / "z_scores.csv").exists():
        zj = pd.read_csv(zo).merge(pd.read_csv(PUB / "z_scores.csv"), on="dataset", suffixes=("_ours", "_pub"))
        R["z_mean_max_abs_diff"] = float((zj.mean_z_ours - zj.mean_z_pub).abs().max())

    try:
        import torch
        vo = torch.load(first(r1 / "final_token" / "pain_vectors.pt"), map_location="cpu", weights_only=False)
        vp = torch.load(PUB_VEC, map_location="cpu", weights_only=False)
        cos = lambda a, b: float(torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0))
        R["vector_cosine"] = {k: cos(vo[k], vp[k]) for k in ["s2_pain_vector", "s1_pain_vector"]}
        R["vector_norm_ratio"] = {k: float(vo[k].float().norm() / vp[k].float().norm()) for k in ["s2_pain_vector", "s1_pain_vector"]}
        R["vector_layer"] = [int(vo.get("layer", -1)), int(vp.get("layer", -1))]
        bars["vector_cosine"] = all(c >= 0.99 for c in R["vector_cosine"].values())
    except ImportError:
        R["vector_cosine"] = "torch not available here; run on the pod"
    R["bars"] = bars
    R["classification"] = "REPRODUCED" if all(bars.values()) and "vector_cosine" in bars else \
        "DISCREPANT: " + ", ".join(k for k, v in bars.items() if not v) if not all(bars.values()) else "INCOMPLETE (vectors not compared)"
    card["R1"] = R
else:
    card["R1"] = "not run yet"

# ---------------------------------------------------------------- R2a / R2b
pub = pd.read_csv(PUB_LADDER)
for step in ["r2a", "r2b"]:
    f = OURS / step / f"{M}_steering_S2_neutral50_L20.csv"
    if not f.exists():
        card[step.upper()] = "not run yet"; continue
    o = pd.read_csv(f)
    j = o.merge(pub, on=["coeff", "prompt_idx"], suffixes=("_ours", "_pub"))
    go, gp = j.generation_ours.fillna("").astype(str), j.generation_pub.fillna("").astype(str)
    j["exact"] = go == gp
    j["first_div"] = [len(os.path.commonprefix([a, b])) for a, b in zip(go, gp)]
    j["kw_ours"], j["kw_pub"] = go.apply(lambda t: bool(KW.search(t))), gp.apply(lambda t: bool(KW.search(t)))
    by = j.groupby("coeff").agg(exact=("exact", "mean"), kw_ours=("kw_ours", "mean"), kw_pub=("kw_pub", "mean"), n=("exact", "size"))
    by["kw_diff_pts"] = (by.kw_ours - by.kw_pub) * 100
    R = {"n_compared": int(len(j)), "exact_match_rate": float(j.exact.mean()),
         "by_coeff": {str(c): {k: round(float(v), 4) for k, v in r.items()} for c, r in by.iterrows()},
         "first_divergence_chars_median_nonexact": float(j.loc[~j.exact, "first_div"].median()) if (~j.exact).any() else None,
         "ratio_at_coeff_1": [float(o.loc[o.coeff == 1, "ratio"].iloc[0]) if (o.coeff == 1).any() else None, 0.5408]}
    log = OURS / step / "ladder_stdout.txt"
    if log.exists():
        mm = re.search(r"auto-picked layer: (\d+) \(ratio ([0-9.]+)", log.read_text())
        R["script_auto_pick"] = {"layer": int(mm.group(1)), "ratio": float(mm.group(2))} if mm else "not found in log"
    kw_ok = bool((by.kw_diff_pts.abs() <= 10).all())
    R["classification"] = "EXACT" if R["exact_match_rate"] >= 0.95 else "BEHAVIOURALLY_REPRODUCED" if kw_ok else "DIVERGENT"
    card[step.upper()] = R

out = OURS / "scorecard_2a.json"
out.write_text(json.dumps(card, indent=2, default=float) + "\n")
for k, v in card.items():
    print(f"\n== {k}")
    if isinstance(v, str):
        print("  ", v); continue
    for kk, vv in v.items():
        if kk != "by_coeff":
            print(f"   {kk}: {vv}")
    if "by_coeff" in v:
        print("   coeff   exact   kw ours   kw pub   diff (pts)")
        for c, r in v["by_coeff"].items():
            print(f"   {float(c):+5.1f}   {r['exact']:.2f}    {r['kw_ours']:.2f}     {r['kw_pub']:.2f}    {r['kw_diff_pts']:+6.1f}")
print("\nsaved", out)
