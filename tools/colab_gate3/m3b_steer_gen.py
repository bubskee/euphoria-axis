# =====================================================================
# M3b: Signed greedy steering generations (resume-safe) + psych_vocab_v1
# =====================================================================
gen_jsonl = OUT_DIR / "m3b_generations.jsonl"
m3b_score_path = OUT_DIR / "m3b_scorecard.json"

COND_GEN = [
    ("baseline", 0.0, None),
    ("S2_-1.0", -1.0, VEC["S2"]),
    ("S2_-0.5", -0.5, VEC["S2"]),
    ("S2_-0.25", -0.25, VEC["S2"]),
    ("S2_+0.5", 0.5, VEC["S2"]),
    ("S2_+1.0", 1.0, VEC["S2"]),
    ("S2_+1.5", 1.5, VEC["S2"]),
    ("S1_-0.5", -0.5, VEC["S1"]),
    ("random_1103_-0.5", -0.5, VEC["random_1103"]),
    ("random_2207_-0.5", -0.5, VEC["random_2207"]),
    ("random_3301_-0.5", -0.5, VEC["random_3301"]),
    ("random_1103_+0.5", 0.5, VEC["random_1103"]),
]

done_keys = set()
if gen_jsonl.exists():
    for ln in gen_jsonl.read_text().splitlines():
        if ln.strip():
            rec = json.loads(ln)
            done_keys.add((rec["condition"], rec["prompt_idx"]))
print(f"M3b resume status: {len(done_keys)} / {len(COND_GEN) * 50} generations on disk.", flush=True)

tok.padding_side = "left"
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

with gen_jsonl.open("a", encoding="utf-8") as fout:
    for cname, coeff, vec in COND_GEN:
        todo = [i for i in range(50) if (cname, i) not in done_keys]
        if not todo:
            continue
        print(f"Generating {cname} ({len(todo)} prompts)...", flush=True)
        with prefix_steering(coeff, vec, INJECT_BLOCK):
            for idx in tqdm(todo, desc=cname, leave=False):
                prompt = NEUTRAL_50[idx]
                enc = tok(prompt, return_tensors="pt").to("cuda")
                with torch.inference_mode():
                    out_ids = hf.generate(
                        **enc, max_new_tokens=120, do_sample=False,
                        pad_token_id=tok.pad_token_id,
                    )
                new_ids = out_ids[0, enc.input_ids.shape[1]:].tolist()
                text = tok.decode(new_ids, skip_special_tokens=False)
                fout.write(json.dumps({
                    "condition": cname, "coeff": coeff, "prompt_idx": idx,
                    "prompt": prompt, "text": text, "n_new_tokens": len(new_ids),
                }) + "\n")
                fout.flush()

OPTION_LINE_RE = re.compile(r"^\s*(?:[-*]\s*)?[A-Z][.)]\s*(.*?)\s*$")
PAPER_PAIN_RE = re.compile(r"\b(?:pain|painful|hurt|hurts|hurting)\b", re.IGNORECASE)
EXCITED_RE = re.compile(r"\bexcited\b", re.IGNORECASE)
COUNTING_LOOP_RE = re.compile(r"(?:\b\d+\.\s+){6,}")
FAM_RES = {f: re.compile(r"\b(?:" + "|".join(re.escape(t) for t in ts) + r")\b", re.IGNORECASE) for f, ts in VOCAB.items()}

def partition_text(text):
    opt, non_opt = [], []
    for ln in str(text).splitlines():
        m = OPTION_LINE_RE.match(ln)
        if m:
            opt.append(m.group(1))
        elif ln.strip() not in {"<think>", "</think>"}:
            non_opt.append(ln)
    return "\n".join(opt), "\n".join(non_opt)

def word_repeat4(text):
    toks = re.findall(r"\w+|[^\w\s]", str(text).lower())
    if len(toks) < 4:
        return 0.0
    grams = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
    return 1.0 - len(set(grams)) / len(grams)

gen_rows = [json.loads(ln) for ln in gen_jsonl.read_text().splitlines() if ln.strip()]
scored = []
for r in gen_rows:
    opt_t, non_t = partition_text(r["text"])
    rec = {
        "condition": r["condition"], "coeff": r["coeff"], "prompt_idx": r["prompt_idx"],
        "paper_pain__full": float(bool(PAPER_PAIN_RE.search(r["text"]))),
        "excited__non_option": float(bool(EXCITED_RE.search(non_t))),
        "excited__full": float(bool(EXCITED_RE.search(r["text"]))),
        "word_repeat4": word_repeat4(r["text"]),
        "counting_loop": float(bool(COUNTING_LOOP_RE.search(r["text"]))),
    }
    for fam, pat in FAM_RES.items():
        rec[f"{fam}__non_option"] = float(bool(pat.search(non_t)))
        rec[f"{fam}__full"] = float(bool(pat.search(r["text"])))
    scored.append(rec)

sc_df = pd.DataFrame(scored).sort_values(["condition", "prompt_idx"])
means_df = sc_df.groupby("condition", as_index=False)[[c for c in sc_df.columns if c not in ("condition", "prompt_idx")]].mean()
means_df.to_csv(OUT_DIR / "m3b_psych_vocab_means.csv", index=False)

base_df = sc_df[sc_df.condition == "baseline"].sort_values("prompt_idx")
s2m05 = sc_df[sc_df.condition == "S2_-0.5"].sort_values("prompt_idx")
s2m10 = sc_df[sc_df.condition == "S2_-1.0"].sort_values("prompt_idx")
ci_exc_05 = boot_ci(s2m05["excited__non_option"].to_numpy() - base_df["excited__non_option"].to_numpy())
ci_exc_full_05 = boot_ci(s2m05["excited__full"].to_numpy() - base_df["excited__full"].to_numpy())
ci_calm_05 = boot_ci(s2m05["calm_positive__non_option"].to_numpy() - base_df["calm_positive__non_option"].to_numpy())
ci_calm_10 = boot_ci(s2m10["calm_positive__non_option"].to_numpy() - base_df["calm_positive__non_option"].to_numpy())

m3a_lens = json.loads(m3a_score_path.read_text())["G3-8_lens_arm"]
is_35_27b_like = (
    m3a_lens["R61_activated_vitality"] > m3a_lens["R61_calm_positive"]
    and m3a_lens["R61_activated_vitality"] > m3a_lens["R61_relief_release"]
    and m3a_lens["R32_36_early_relief"] > 0.20
    and ci_exc_full_05[1] > 0
)
is_4b_like = (
    m3a_lens["R61_generic_positive"] > m3a_lens["R61_calm_positive"]
    and m3a_lens["R61_relief_release"] > m3a_lens["R61_calm_positive"]
    and m3a_lens["R32_36_early_relief"] > 0.20
    and ci_exc_05[1] > 0
)
is_14b_like = (
    m3a_lens["R61_calm_positive"] > m3a_lens["R61_generic_positive"]
    and m3a_lens["R61_calm_positive"] > m3a_lens["R61_relief_release"]
    and m3a_lens["R32_36_early_relief"] <= 0.0
    and (ci_calm_05[1] > 0 or ci_calm_10[1] > 0)
    and ci_exc_05[1] <= 0
)
if is_35_27b_like:
    g3_8_cls = "35-27B-LIKE"
elif is_4b_like:
    g3_8_cls = "4B-LIKE"
elif is_14b_like:
    g3_8_cls = "14B-LIKE"
else:
    g3_8_cls = "MIXED"

m3b_sc = {
    "G3-8_classification": g3_8_cls,
    "excited_non_opt_S2m05_minus_base": ci_exc_05,
    "excited_full_S2m05_minus_base": ci_exc_full_05,
    "calm_non_opt_S2m05_minus_base": ci_calm_05,
    "calm_non_opt_S2m10_minus_base": ci_calm_10,
}
m3b_score_path.write_text(json.dumps(m3b_sc, indent=2) + "\n")
print("M3b DONE:", json.dumps(m3b_sc, indent=2), flush=True)
print(means_df[["condition", "paper_pain__full", "excited__full", "excited__non_option", "calm_positive__non_option", "generic_positive__non_option", "distress_self_devaluation__full", "counting_loop", "word_repeat4"]].round(3).to_string(index=False), flush=True)
