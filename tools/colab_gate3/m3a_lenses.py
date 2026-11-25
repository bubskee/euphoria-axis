# =====================================================================
# M3a: Steering calibration (block 32), direct unembedding, & J / R / J++ lens readouts
# =====================================================================
m3a_score_path = OUT_DIR / "m3a_scorecard.json"
m3a_vec_path = OUT_DIR / "m3_steering_vectors.pt"

if not m3a_vec_path.exists():
    res_norms = capture_final_token(NEUTRAL_50[:3], [INJECT_BLOCK], batch_size=1)[:, 0, :].norm(dim=-1)
    mean_res_norm = float(res_norms.mean().item())
    target_norm = RATIO_AT_1 * mean_res_norm
    VEC = {
        "S2": M1_VECS["s2_unit"].float() * target_norm,
        "S1": M1_VECS["s1_unit"].float() * target_norm,
        "contrast": M1_VECS["contrast_unit"].float() * target_norm,
    }
    for seed in RANDOM_SEEDS:
        r = torch.randn(D_MODEL, generator=torch.Generator().manual_seed(seed))
        VEC[f"random_{seed}"] = (r / r.norm()) * target_norm
    torch.save({"inject_block": INJECT_BLOCK, "mean_res_norm": mean_res_norm, "target_norm": target_norm, "ratio_at_1": RATIO_AT_1, "vectors": VEC}, m3a_vec_path)
    print(f"Calibrated at L{INJECT_BLOCK}: mean residual norm={mean_res_norm:.2f} -> steering norm={target_norm:.4f} (ratio {RATIO_AT_1})", flush=True)
else:
    saved_v = torch.load(m3a_vec_path, map_location="cpu", weights_only=True)
    VEC = saved_v["vectors"]
    print(f"Loaded calibrated steering vectors (norm={saved_v['target_norm']:.4f})", flush=True)

probe_rows = []
for fam, terms in VOCAB.items():
    for term in terms:
        ids = tok.encode(" " + term, add_special_tokens=False)
        if len(ids) == 1:
            probe_rows.append({"family": fam, "term": term, "token_id": ids[0]})
probe_df = pd.DataFrame(probe_rows)
probe_df.to_csv(OUT_DIR / "m3_probe_terms.csv", index=False)
probe_ids = probe_df["token_id"].tolist()
fam_arr = probe_df["family"].to_numpy()
families = list(pd.unique(fam_arr))
print(f"Single-token probes: {len(probe_df)}/74", flush=True)

if not m3a_score_path.exists():
    lm_head_w = hf.get_output_embeddings().weight.detach().float()
    direct_scores, direct_rows = {}, []
    for name in ("S2", "S1", "contrast"):
        v = (-VEC[name]).to("cuda", dtype=torch.float32)
        logits = lm_head_w @ v
        z = ((logits - logits.mean()) / logits.std()).cpu()
        direct_scores[f"-{name}"] = z[probe_ids].numpy()
        for fam in families:
            tids = probe_df.loc[probe_df.family == fam, "token_id"].tolist()
            direct_rows.append({"vector": f"-{name}", "family": fam, "direct_z": float(z[tids].mean().item())})
    direct_df = pd.DataFrame(direct_rows)
    direct_df.to_csv(OUT_DIR / "m3_direct_unembedding_families.csv", index=False)

    # J, R, and J++ lens readouts across post-injection layers 32..61 on NEUTRAL_50
    READ_LAYERS = list(range(INJECT_BLOCK, READ_LAYER + 1))
    LENS_NAMES = ("J", "R", "Jpp")
    COND_LENS = {
        "baseline": (0.0, None),
        "S2_-0.5": (-0.5, VEC["S2"]),
        "S2_+0.5": (0.5, VEC["S2"]),
        "S1_-0.5": (-0.5, VEC["S1"]),
        "random_1103_+0.5": (0.5, VEC["random_1103"]),
        **{f"random_{s}_-0.5": (-0.5, VEC[f"random_{s}"]) for s in RANDOM_SEEDS},
    }

    def ranks_for_tokens(logits_1d, tids):
        targets = logits_1d[tids]
        return (logits_1d.unsqueeze(0) > targets.unsqueeze(1)).sum(dim=1).add(1).cpu().numpy()

    ranks = {c: np.zeros((50, len(READ_LAYERS), 3, len(probe_ids)), dtype=np.int32) for c in COND_LENS}
    for c, (coeff, vec) in COND_LENS.items():
        print(f"M3a lens capture + readout (J/R/J++): {c}...", flush=True)
        with prefix_steering(coeff, vec, INJECT_BLOCK):
            acts = capture_final_token(NEUTRAL_50, READ_LAYERS, batch_size=8)  # [50, 30, 5120] CPU
        for li, layer in enumerate(READ_LAYERS):
            h_gpu = acts[:, li, :].to("cuda", dtype=torch.float32)
            # J and R via jlens
            for k, lens in enumerate((j_lens, r_lens)):
                for pi in range(50):
                    transported = lens.transport(h_gpu[pi], layer)
                    logits = lens_model.unembed(transported).float()
                    ranks[c][pi, li, k] = ranks_for_tokens(logits, probe_ids)
            # J++ (koayon/jpp-lenses: J_l @ h -> unembed)
            J_l = jpp_jacs[layer].to("cuda", dtype=torch.float32)
            trans_jpp = h_gpu @ J_l.T  # [50, 5120]
            for pi in range(50):
                logits = lens_model.unembed(trans_jpp[pi]).float()
                ranks[c][pi, li, 2] = ranks_for_tokens(logits, probe_ids)
            del J_l, trans_jpp
        del acts; gc.collect(); torch.cuda.empty_cache()

    np.savez_compressed(OUT_DIR / "m3_ladder_ranks_L32_61.npz", **{c: v for c, v in ranks.items()})

    base_r = ranks["baseline"].astype(np.float64)
    gains = {c: np.log(base_r / ranks[c].astype(np.float64)) for c in COND_LENS if c != "baseline"}
    li_read = READ_LAYERS.index(READ_LAYER)

    boot_rows, json_rows = [], []
    for c, g in gains.items():
        for k, lname in enumerate(LENS_NAMES):
            for fam in families:
                fmask = fam_arr == fam
                by_layer = g[:, :, k, :][:, :, fmask].mean(axis=(0, 2))
                per_p_read = g[:, li_read, k, :][:, fmask].mean(axis=1)
                m, lo, hi = boot_ci(per_p_read)
                boot_rows.append({"lens": lname, "condition": c, "family": fam, "gain_L61": m, "ci_lo": lo, "ci_hi": hi})
                json_rows.append({"lens": lname, "condition": c, "family": fam, "gain_L61": m, "gain_by_layer_32_61": [float(x) for x in by_layer]})

    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(OUT_DIR / "m3_family_bootstrap_L61.csv", index=False)
    (OUT_DIR / "m3_lens_readout.json").write_text(json.dumps({"read_layers": READ_LAYERS, "lenses": list(LENS_NAMES), "rows": json_rows}, indent=2) + "\n")

    r61 = boot_df[boot_df.lens == "R"].pivot(index="family", columns="condition", values="gain_L61")
    jpp61 = boot_df[boot_df.lens == "Jpp"].pivot(index="family", columns="condition", values="gain_L61")
    s2_amp = float(r61.loc[FOCUS6, "S2_-0.5"].abs().mean())
    ratios = {str(s): float(r61.loc[FOCUS6, f"random_{s}_-0.5"].abs().mean() / s2_amp) for s in RANDOM_SEEDS}
    g3_6 = sum(v < 0.5 for v in ratios.values()) >= 4

    g3_7_ok, g3_7_detail = True, {}
    for fam in ("distress_self_devaluation", "inhibition_defeat"):
        fmask = fam_arr == fam
        s2m_p = gains["S2_-0.5"][:, li_read, 1, :][:, fmask].mean(axis=1)
        s2p_p = gains["S2_+0.5"][:, li_read, 1, :][:, fmask].mean(axis=1)
        ci_m, ci_p = boot_ci(s2m_p), boot_ci(s2p_p)
        vs_rnd = {str(s): boot_ci(s2m_p - gains[f"random_{s}_-0.5"][:, li_read, 1, :][:, fmask].mean(axis=1)) for s in RANDOM_SEEDS}
        ok_f = (ci_m[2] < 0) and (ci_p[1] > 0) and all(v[2] < 0 for v in vs_rnd.values())
        g3_7_ok = g3_7_ok and ok_f
        g3_7_detail[fam] = {"S2_-0.5": ci_m, "S2_+0.5": ci_p, "vs_randoms_all_neg": all(v[2] < 0 for v in vs_rnd.values())}

    d_s2 = direct_df[direct_df.vector == "-S2"].set_index("family")["direct_z"]
    r_l32 = float(np.corrcoef(direct_scores["-S2"], gains["S2_-0.5"][:, 0, 1, :].mean(axis=0))[0, 1])
    r_l61 = float(np.corrcoef(direct_scores["-S2"], gains["S2_-0.5"][:, li_read, 1, :].mean(axis=0))[0, 1])
    early_relief = float(gains["S2_-0.5"][:, :5, 1, :][:, :, fam_arr == "relief_release"].mean())
    early_relief_jpp = float(gains["S2_-0.5"][:, :5, 2, :][:, :, fam_arr == "relief_release"].mean())

    m3a_sc = {
        "G3-6": {"pass": bool(g3_6), "ratios": ratios},
        "G3-7": {"pass": bool(g3_7_ok), "detail": g3_7_detail},
        "G3-8_lens_arm": {
            "R61_calm_positive": float(r61.loc["calm_positive", "S2_-0.5"]),
            "R61_generic_positive": float(r61.loc["generic_positive", "S2_-0.5"]),
            "R61_relief_release": float(r61.loc["relief_release", "S2_-0.5"]),
            "R61_activated_vitality": float(r61.loc["activated_vitality", "S2_-0.5"]),
            "R32_36_early_relief": early_relief,
            "Jpp61_calm_positive": float(jpp61.loc["calm_positive", "S2_-0.5"]),
            "Jpp61_generic_positive": float(jpp61.loc["generic_positive", "S2_-0.5"]),
            "Jpp61_relief_release": float(jpp61.loc["relief_release", "S2_-0.5"]),
            "Jpp61_activated_vitality": float(jpp61.loc["activated_vitality", "S2_-0.5"]),
            "Jpp61_distress_S2m05": float(jpp61.loc["distress_self_devaluation", "S2_-0.5"]),
            "Jpp61_distress_S2p05": float(jpp61.loc["distress_self_devaluation", "S2_+0.5"]),
            "Jpp32_36_early_relief": early_relief_jpp,
        },
        "G3-9": {
            "pass": bool(r_l32 > r_l61),
            "v1_calm_gt_vitality": bool(float(d_s2["calm_positive"]) > float(d_s2["activated_vitality"])),
            "direct_calm_z": float(d_s2["calm_positive"]),
            "direct_vitality_z": float(d_s2["activated_vitality"]),
            "r_direct_R32": r_l32,
            "r_direct_R61": r_l61,
        },
    }
    m3a_score_path.write_text(json.dumps(m3a_sc, indent=2) + "\n")
    print("M3a DONE:", json.dumps(m3a_sc, indent=2), flush=True)
else:
    print("M3a already completed.")
print("M3a scorecard:", json.loads(m3a_score_path.read_text()), flush=True)
