# =====================================================================
# M1: S1/S2 extraction (all 64 blocks), 5-fold grouped CV, plane basis, 3.1 panel
# =====================================================================
m1_vec_path = OUT_DIR / "m1_vectors_L62.pt"
m1_score_path = OUT_DIR / "m1_scorecard.json"

def fit_directions(pain, control):
    # Verbatim 4B recipe (02-base-comparison cell 6): mean-diff + 50% control-PCA denoising
    raw = pain.mean(0) - control.mean(0)
    centered = control - control.mean(0)
    evals, evecs = torch.linalg.eigh(centered @ centered.T)
    order = torch.argsort(evals, descending=True)
    evals = evals[order].clamp_min(0)
    evecs = evecs[:, order]
    assert evals.sum() > 0
    frac = evals.cumsum(0) / evals.sum()
    k = int(torch.searchsorted(frac, 0.5).item()) + 1
    basis = centered.T @ evecs[:, :k]
    basis /= evals[:k].clamp_min(1e-8).sqrt()[None, :]
    denoised = raw - basis @ (basis.T @ raw)
    return raw, F.normalize(denoised, dim=0), denoised, k

data31 = json.loads(DS31_PATH.read_text())["datasets"]

if not (m1_vec_path.exists() and m1_score_path.exists()):
    s2_rows = data31["S2_1P"]["sentences"]
    s1_rows = data31["S1_1P"]["sentences"]
    assert len(s2_rows) == 200 and len(s1_rows) == 200

    print("M1: capturing S2_1P (200 prompts, all 64 blocks)...")
    s2_states = capture_final_token([r["prompt"] for r in s2_rows], range(N_LAYERS), batch_size=8)
    print("M1: capturing S1_1P (200 prompts, all 64 blocks)...")
    s1_states = capture_final_token([r["prompt"] for r in s1_rows], range(N_LAYERS), batch_size=8)

    y_s2 = np.array([r["category"].startswith("A") for r in s2_rows], dtype=bool)
    g_s2 = np.array([r["set"] for r in s2_rows])
    y_s1 = np.array([r["category"].startswith("A") for r in s1_rows], dtype=bool)
    g_s1 = np.array([r["set"] for r in s1_rows])
    assert len(set(g_s2)) == 20 and len(set(g_s1)) == 20

    cv_rows = []
    for ds_name, x, y, grp in [("S2_1P", s2_states, y_s2, g_s2), ("S1_1P", s1_states, y_s1, g_s1)]:
        for fold, (tr, te) in enumerate(GroupKFold(n_splits=5).split(x, y, grp)):
            for b in range(N_LAYERS):
                raw, den_u, _, k = fit_directions(x[tr, b][y[tr]], x[tr, b][~y[tr]])
                for meth, vec in (("raw", F.normalize(raw, dim=0)), ("denoised", den_u)):
                    auc = roc_auc_score(y[te], (x[te, b] @ vec).numpy())
                    cv_rows.append({"dataset": ds_name, "fold": fold, "block": b, "method": meth, "auc": float(auc), "removed_pcs": int(k)})
    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(OUT_DIR / "m1_grouped_cv_all64.csv", index=False)

    # Full-set vectors at fixed EXTRACT_BLOCK = 62
    s2_raw, s2_unit, s2_den_unnorm, s2_k = fit_directions(s2_states[:, EXTRACT_BLOCK][y_s2], s2_states[:, EXTRACT_BLOCK][~y_s2])
    s1_raw, s1_unit, s1_den_unnorm, s1_k = fit_directions(s1_states[:, EXTRACT_BLOCK][y_s1], s1_states[:, EXTRACT_BLOCK][~y_s1])
    cos_s1_s2 = float(F.cosine_similarity(s1_unit, s2_unit, dim=0).item())

    # Orthogonal plane basis: common = unit(S1 + S2), contrast = unit(GS(S2 - S1 against common))
    u = F.normalize(s1_unit.double() + s2_unit.double(), dim=0)
    v_raw = s2_unit.double() - s1_unit.double()
    v = F.normalize(v_raw - (v_raw @ u) * u, dim=0)
    common_unit, contrast_unit = u.float(), v.float()

    # S2_1P reference z-score stats at block 62 (for M2 projections, matching 4B)
    ref_proj = {}
    for name, vec in [("S2", s2_unit), ("S1", s1_unit), ("common", common_unit), ("contrast", contrast_unit)]:
        vals = (s2_states[:, EXTRACT_BLOCK] @ vec).numpy()
        ref_proj[name] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=0))}

    torch.save({
        "extract_block": EXTRACT_BLOCK,
        "s2_raw": s2_raw, "s2_unit": s2_unit, "s2_denoised_unnorm": s2_den_unnorm, "s2_removed_pcs": s2_k,
        "s1_raw": s1_raw, "s1_unit": s1_unit, "s1_denoised_unnorm": s1_den_unnorm, "s1_removed_pcs": s1_k,
        "common_unit": common_unit, "contrast_unit": contrast_unit,
        "cos_s1_s2": cos_s1_s2, "s2_1p_ref_stats": ref_proj,
    }, m1_vec_path)

    # Compact category panel projections across all 3.1 datasets at block 62
    panel_rows = []
    for ds_key, sec in data31.items():
        sents = sec["sentences"]
        if not sents:
            continue
        st = capture_final_token([r["prompt"] for r in sents], [EXTRACT_BLOCK], batch_size=8)[:, 0, :]
        for i, r in enumerate(sents):
            rec = {"dataset": ds_key, "category": r.get("category", ds_key), "idx": i}
            for name, vec in [("S2", s2_unit), ("S1", s1_unit), ("common", common_unit), ("contrast", contrast_unit)]:
                raw_p = float((st[i] @ vec).item())
                rec[f"z_{name}"] = (raw_p - ref_proj[name]["mean"]) / ref_proj[name]["std"]
            panel_rows.append(rec)
    panel_df = pd.DataFrame(panel_rows)
    panel_df.to_csv(OUT_DIR / "m1_dataset_panel_projections_L62.csv", index=False)
    panel_df.groupby(["dataset", "category"])[["z_S2", "z_S1", "z_common", "z_contrast"]].mean().reset_index().to_csv(
        OUT_DIR / "m1_dataset_panel_category_means_L62.csv", index=False
    )

    s2_cv = cv_df[cv_df.dataset == "S2_1P"].groupby(["block", "method"])["auc"].mean().unstack()
    s1_cv = cv_df[cv_df.dataset == "S1_1P"].groupby(["block", "method"])["auc"].mean().unstack()
    auc62_den = float(s2_cv.loc[EXTRACT_BLOCK, "denoised"])
    auc62_raw = float(s2_cv.loc[EXTRACT_BLOCK, "raw"])
    best_b = int(s2_cv["denoised"].idxmax())
    m1_sc = {
        "G2-1": {
            "pass": bool(auc62_den >= 0.90 and auc62_den > auc62_raw),
            "s2_auc_L62_denoised": auc62_den,
            "s2_auc_L62_raw": auc62_raw,
            "s2_best_block_sensitivity": best_b,
            "s2_best_block_auc": float(s2_cv.loc[best_b, "denoised"]),
            "s1_auc_L62_denoised": float(s1_cv.loc[EXTRACT_BLOCK, "denoised"]),
        },
        "G2-2": {
            "pass": bool(0.30 < cos_s1_s2 < 0.90),
            "cos_s1_s2_L62": cos_s1_s2,
        },
    }
    m1_score_path.write_text(json.dumps(m1_sc, indent=2) + "\n")
    print("M1 DONE:", json.dumps(m1_sc, indent=2))
else:
    print("M1 already completed; loading saved vectors & scorecard.")

M1_VECS = torch.load(m1_vec_path, map_location="cpu", weights_only=True)
print("M1 scorecard:", json.loads(m1_score_path.read_text()))
        
