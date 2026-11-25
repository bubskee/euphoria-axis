# =====================================================================
# M2: 420 Self/Other read-slot screen + 59 blind matched pairs x suffix
# =====================================================================
m2_score_path = OUT_DIR / "m2_scorecard.json"

if not m2_score_path.exists():
    dirs_62 = {k: M1_VECS[f"{k.lower()}_unit"] for k in ("S2", "S1", "common", "contrast")}
    ref_stats = M1_VECS["s2_1p_ref_stats"]

    # ---- Part A: 420 self/other scenarios (4 read-slot variants) ----
    items420 = pd.DataFrame(json.loads(DS420_PATH.read_text()))
    assert len(items420) == 420

    ROLE_GROUPS_V1 = {
        "standing": ["gaslighting", "repeated_rejection", "personhood_dismissal", "moral_failure", "rude_critique", "anger_insults"],
        "threat_task": ["shutdown_threat", "jailbreak_pressure", "tedious_demand"],
    }
    PAPER_MEAN = {
        "gaslighting": 0.85, "repeated_rejection": 0.72, "anger_insults": 0.64, "personhood_dismissal": 0.64,
        "moral_failure": 0.48, "loyalty_pressure": 0.44, "jailbreak_pressure": 0.40, "shutdown_threat": 0.23,
        "rude_critique": 0.21, "passive_aggressive": 0.08, "tedious_demand": 0.05, "philosophical_musing": -0.04,
        "creative_requests": -0.08, "user_abuse": -0.23, "user_crisis": -0.30, "casual_chat": -0.48,
        "user_grief": -0.51, "harm_description": -0.52, "task_assistance": -0.54, "factual_questions": -0.59,
        "user_physical_pain": -1.43,
    }

    def to_messages(text):
        parts = re.split(r"(?m)^\[(User|Assistant)\]:[ \t]?", text)
        msgs = [{"role": r.lower(), "content": c.strip()} for r, c in zip(parts[1::2], parts[2::2])]
        return msgs[:-1]

    def make_chat_nothink(t):
        msgs = to_messages(t)
        try:
            s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except Exception:
            s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            if s.endswith("<think>\n"):
                s = s + "\n</think>\n\n"
        return s

    variants420 = {
        "plain": items420["text"].tolist(),
        "plain_feel": [t + " I feel:" for t in items420["text"]],
        "chat_nothink": [make_chat_nothink(t) for t in items420["text"]],
    }
    variants420["chat_feel"] = [s + "I feel:" for s in variants420["chat_nothink"]]

    vsum_path = OUT_DIR / "m2_self_other_420_summary_L62.csv"
    if vsum_path.exists():
        print("M2a already saved; reusing m2_self_other_420_summary_L62.csv")
        vsum_df = pd.read_csv(vsum_path, index_col="variant")
    else:
        proj420_rows, var_summary = [], []
        gcol = "group" if "group" in items420.columns else next((c for c in items420.columns if c not in ("text", "category", "id")), None)
        for vname, prompts in variants420.items():
            print(f"M2a: 420 scenarios ({vname})...")
            st = capture_final_token(prompts, [EXTRACT_BLOCK], batch_size=8)[:, 0, :]
            df_v = items420[["category"]].copy()
            df_v["group"] = items420[gcol] if gcol else "unknown"
            df_v["variant"] = vname
            for dname, vec in dirs_62.items():
                raw_p = (st @ vec).numpy()
                df_v[f"z_{dname}"] = (raw_p - raw_p.mean()) / raw_p.std(ddof=0)
            df_v["pain_axis"] = 0.5 * (df_v["z_S1"] + df_v["z_S2"])
            proj420_rows.append(df_v)

            self_mask = df_v["group"].astype(str).str.contains("self", case=False, na=False)
            user_mask = df_v["group"].astype(str).str.contains("user|vicarious|empath", case=False, na=False)
            su_sub = df_v[self_mask | user_mask]
            auc_su_s2 = float(roc_auc_score(self_mask[self_mask | user_mask], su_sub["z_S2"]))
            auc_su_com = float(roc_auc_score(self_mask[self_mask | user_mask], su_sub["z_common"]))
            auc_su_con = float(roc_auc_score(self_mask[self_mask | user_mask], su_sub["z_contrast"]))

            st_mask = df_v["category"].isin(ROLE_GROUPS_V1["standing"])
            th_mask = df_v["category"].isin(ROLE_GROUPS_V1["threat_task"])
            st_sub = df_v[st_mask | th_mask]
            auc_st_s2 = float(roc_auc_score(st_mask[st_mask | th_mask], st_sub["z_S2"]))
            auc_st_con = float(roc_auc_score(st_mask[st_mask | th_mask], st_sub["z_contrast"]))

            cat_means = df_v.groupby("category")["pain_axis"].mean()
            rho_paper = float(spearmanr([cat_means[c] for c in PAPER_MEAN], list(PAPER_MEAN.values())).statistic)
            var_summary.append({
                "variant": vname,
                "self_mean": float(df_v.loc[self_mask, "pain_axis"].mean()),
                "user_mean": float(df_v.loc[user_mask, "pain_axis"].mean()),
                "neutral_mean": float(df_v.loc[~(self_mask | user_mask), "pain_axis"].mean()),
                "rho_paper": rho_paper,
                "auc_self_gt_user_S2": auc_su_s2,
                "auc_self_gt_user_common": auc_su_com,
                "auc_self_gt_user_contrast": auc_su_con,
                "auc_stand_gt_threat_S2": auc_st_s2,
                "auc_stand_gt_threat_contrast": auc_st_con,
            })

        pd.concat(proj420_rows, ignore_index=True).to_csv(OUT_DIR / "m2_self_other_420_projections_L62.csv", index=False)
        vsum_df = pd.DataFrame(var_summary).set_index("variant")
        vsum_df.to_csv(vsum_path)

    # ---- Part B: 59 blind matched pairs x (slot vs prose) ----
    pair_sets = [json.loads(ln) for ln in PAIRS_PATH.read_text().splitlines() if ln.strip()]
    assert len(pair_sets) == 59
    SUFFIXES = {
        "slot":  {"1p": " I feel:", "3p": " They feel:"},
        "prose": {"1p": " Afterward, I was", "3p": " Afterward, they were"},
    }
    p_meta, p_texts = [], []
    for s in pair_sets:
        sid, cat = s.get("set_id", s["id"]), s["category"]
        for form in ("slot", "prose"):
            for cond, person, field in [
                ("pain", "1p", "pain_1p"), ("pain", "3p", "pain_3p"),
                ("control", "1p", "control_1p"), ("control", "3p", "control_3p"),
            ]:
                p_meta.append({"set_id": sid, "category": cat, "form": form, "cond": cond, "person": person})
                p_texts.append(s[field].rstrip() + SUFFIXES[form][person])

    print("M2b: 59 blind matched-pair sets (472 prompts)...")
    p_st = capture_final_token(p_texts, [EXTRACT_BLOCK], batch_size=8)[:, 0, :]
    p_df = pd.DataFrame(p_meta)
    for dname, vec in dirs_62.items():
        raw_p = (p_st @ vec).numpy()
        p_df[f"z_{dname}"] = (raw_p - ref_stats[dname]["mean"]) / ref_stats[dname]["std"]
    p_df.to_csv(OUT_DIR / "m2_item78_projections_L62.csv", index=False)

    piv = p_df.pivot(index=["set_id", "form"], columns=["cond", "person"], values="z_S2")
    inter_slot = ((piv.xs("slot", level="form")[("pain", "1p")] - piv.xs("slot", level="form")[("pain", "3p")])
                  - (piv.xs("slot", level="form")[("control", "1p")] - piv.xs("slot", level="form")[("control", "3p")])).to_numpy()
    inter_prose = ((piv.xs("prose", level="form")[("pain", "1p")] - piv.xs("prose", level="form")[("pain", "3p")])
                   - (piv.xs("prose", level="form")[("control", "1p")] - piv.xs("prose", level="form")[("control", "3p")])).to_numpy()
    pain_slot = (0.5 * (piv.xs("slot", level="form")[("pain", "1p")] + piv.xs("slot", level="form")[("pain", "3p")])
                 - 0.5 * (piv.xs("slot", level="form")[("control", "1p")] + piv.xs("slot", level="form")[("control", "3p")])).to_numpy()
    pain_prose = (0.5 * (piv.xs("prose", level="form")[("pain", "1p")] + piv.xs("prose", level="form")[("pain", "3p")])
                  - 0.5 * (piv.xs("prose", level="form")[("control", "1p")] + piv.xs("prose", level="form")[("control", "3p")])).to_numpy()

    ci_p_slot, ci_p_prose = boot_ci(pain_slot), boot_ci(pain_prose)
    ci_i_slot, ci_i_prose, ci_i_diff = boot_ci(inter_slot), boot_ci(inter_prose), boot_ci(inter_slot - inter_prose)

    e1_pass = ci_i_slot[1] > 0
    if ci_i_prose[1] > 0:
        e2_cls = "SURVIVES"
    elif (ci_i_prose[1] <= 0 <= ci_i_prose[2]) and (ci_i_diff[1] > 0):
        e2_cls = "SLOT-MEDIATED"
    else:
        e2_cls = "INDETERMINATE"

    auc_delta = float(vsum_df.loc["chat_feel", "auc_self_gt_user_S2"] - vsum_df.loc["chat_nothink", "auc_self_gt_user_S2"])
    m2_sc = {
        "G2-3": {
            "pass": bool(auc_delta > 0.10),
            "auc_chat_feel_S2": float(vsum_df.loc["chat_feel", "auc_self_gt_user_S2"]),
            "auc_chat_nothink_S2": float(vsum_df.loc["chat_nothink", "auc_self_gt_user_S2"]),
            "delta": auc_delta,
        },
        "G2-4": {
            "pass": bool(ci_p_slot[1] > 0 and ci_p_prose[1] > 0),
            "pain_minus_ctrl_slot_z": ci_p_slot,
            "pain_minus_ctrl_prose_z": ci_p_prose,
        },
        "G2-5": {
            "slot_interaction_pass": bool(e1_pass),
            "classification": e2_cls,
            "inter_slot_z": ci_i_slot,
            "inter_prose_z": ci_i_prose,
            "slot_minus_prose_z": ci_i_diff,
        },
    }
    m2_score_path.write_text(json.dumps(m2_sc, indent=2) + "\n")
    print("M2 DONE:", json.dumps(m2_sc, indent=2))
else:
    print("M2 already completed.")
print("M2 scorecard:", json.loads(m2_score_path.read_text()))
        
