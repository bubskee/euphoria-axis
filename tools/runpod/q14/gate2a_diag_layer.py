# =====================================================================
# Gate 2a, R1 -- POST-HOC layer-matched diagnostic (does not change the frozen R1 classification).
# Our extraction chose final-token layer 32; the paper chose 31, which led its own published curve by 0.0005
# (about one ranked sentence pair in one fold). This rebuilds the vectors and the in-sample AUC table at BOTH
# layers from our saved activations, with the source script's own functions, and compares them with the
# published layer-31 results. CPU only; run in /root/gate2a/.venv-tl (it can run while the ladder uses the GPU).
# =====================================================================
import importlib.util, json, os, tempfile
from pathlib import Path
import numpy as np, pandas as pd, torch

SRC, M = Path("/root/gate2a/src"), "Qwen_3_14B_base"
OURS = Path("/root/gate2a/out/r1") / M
PUB = SRC / "results/3.2_pain_vectors"
os.chdir(SRC)
spec = importlib.util.spec_from_file_location("extract_src", SRC / "scripts/3.2_pain_vectors/01_extract_activations_and_pain_vectors.py")
src = importlib.util.module_from_spec(spec); spec.loader.exec_module(src)   # the paper's functions, unmodified

saved = torch.load(OURS / "activations.pt", map_location="cpu", weights_only=False)
acts, meta = saved["activations"]["final_token"], saved["metadata"]
pub_vec = torch.load(PUB / "pain_vectors" / M / "pain_vectors.pt", map_location="cpu", weights_only=False)
pub_auc = pd.read_csv(PUB / "per_model" / M / "auc_summary.csv", index_col=0)
cos = lambda a, b: float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

out = {"note": "post hoc; the frozen R1 classification stands", "published_vector_layer": int(pub_vec["layer"])}
ours_curve = pd.read_csv(OURS / "layer_curves.csv")
c = ours_curve[ours_curve.extraction == "final_token"].pivot(index="layer", columns="dataset", values="auc_vs_all_controls").mean(axis=1)
out["our_layer_choice_criterion"] = {int(l): round(float(c.loc[l]), 5) for l in range(29, 35)}
out["our_gap_31_minus_32"] = round(float(c.loc[31] - c.loc[32]), 5)

for L in (31, 32):
    s2 = src.compute_pain_vector(acts["S2_1P"][L], meta["S2_1P"]["categories"])
    s1 = src.compute_pain_vector(acts["S1_1P"][L], meta["S1_1P"]["categories"])
    aucs = {}
    with tempfile.TemporaryDirectory() as td:
        for ds in ("S2_1P", "S2_3P"):
            aucs[ds] = src.create_auc_bars(acts[ds][L], meta[ds]["categories"], s2, f"{ds} L{L}", Path(td) / f"{ds}.png")
    table = pd.DataFrame(aucs).T.loc[pub_auc.index, pub_auc.columns]
    out[f"layer_{L}"] = {
        "cos_s2_vs_published_L31": cos(s2, pub_vec["s2_pain_vector"].float().numpy()),
        "cos_s1_vs_published_L31": cos(s1, pub_vec["s1_pain_vector"].float().numpy()),
        "norm_ratio_s2_vs_published": float(np.linalg.norm(s2) / pub_vec["s2_pain_vector"].float().norm()),
        "auc_in_sample_max_abs_diff_vs_published_L31": round(float((table - pub_auc).abs().max().max()), 5),
        "auc_in_sample": {ds: {k: round(float(v), 4) for k, v in row.items()} for ds, row in table.iterrows()},
    }

print(json.dumps(out, indent=2))
dst = Path("/workspace/gate2a/r1_layer_matched_diagnostic.json"); dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(out, indent=2) + "\n")
print("saved", dst)
