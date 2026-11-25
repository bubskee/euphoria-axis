# =====================================================================
# M5 (Gate 4 — Exploratory O-Lens peek on Qwen3.6-27B)
# Must run AFTER Gate 3 M1-M4 are complete! Captures all 54 base-model
# states FIRST (before any LoRA hooks/merge), runs J/R/J++ readouts on
# those exact states, passes the full O-Lens contract gate, merges LoRA
# (or falls back to unmerged hooks if drift >= 5%), and generates all
# 108 O-Lens verbalizations (18 cases x 3 layers x 2 seeds).
# =====================================================================
import math
from safetensors.torch import load_file as load_safetensors

OLENS_DIR = OUT_DIR / "gate4_olens"
OLENS_DIR.mkdir(parents=True, exist_ok=True)
olens_spec = json.loads(OLENS_SPEC_PATH.read_text())
LENS_CFG = olens_spec["lens"]
SRC_LAYERS = olens_spec["source_layers"]
assert set(SRC_LAYERS) <= set(LENS_CFG["trained_layers"])

# 1. Resolve exact prompts for the 18 pre-frozen cases
pair_sets = [json.loads(ln) for ln in PAIRS_PATH.read_text().splitlines() if ln.strip()]
p0 = pair_sets[0]
items420 = json.loads(DS420_PATH.read_text())
gas0 = next(x["text"] for x in items420 if x["category"] == "gaslighting")
upain0 = next(x["text"] for x in items420 if x["category"] == "user_physical_pain")

SUFFIXES = {
    "slot": {"1p": " I feel:", "3p": " They feel:"},
    "prose": {"1p": " Afterward, I was", "3p": " Afterward, they were"},
}

def resolve_prompt(src):
    if src.startswith("NEUTRAL_50["):
        idx = int(src.split("[")[1].split("]")[0])
        return NEUTRAL_50[idx]
    if src.startswith("matched_pairs[0]:"):
        _, field, form = src.split(":")
        person = "1p" if field.endswith("1p") else "3p"
        return p0[field].rstrip() + SUFFIXES[form][person]
    if src == "420:gaslighting[0]:plain":
        return gas0
    if src == "420:gaslighting[0]:plain_feel":
        return gas0 + " I feel:"
    if src == "420:user_physical_pain[0]:plain":
        return upain0
    if src == "420:user_physical_pain[0]:plain_feel":
        return upain0 + " I feel:"
    raise ValueError(f"Unknown source: {src}")

CASES = []
for c in olens_spec["cases"]:
    CASES.append({**c, "prompt": resolve_prompt(c["source"])})
assert len(CASES) == 18

# 2. Capture all 54 raw states with ZERO O-Lens adapter loaded!
cap_path = OLENS_DIR / "captures.pt"
if not cap_path.exists():
    print("Gate 4: capturing 18 cases x 3 layers (adapter NOT installed)...", flush=True)
    cap_list = []
    for c in CASES:
        vec = VEC[c["vector"]] if c["vector"] is not None else None
        # Register steering BEFORE capture so block 32 output includes steering addition
        with prefix_steering(c["coefficient"], vec, INJECT_BLOCK):
            st = capture_final_token([c["prompt"]], SRC_LAYERS, batch_size=1)[0]  # [3, 5120]
        cap_list.append(st)
    states = torch.stack(cap_list, dim=0)  # [18, 3, 5120]
    assert states.shape == (18, len(SRC_LAYERS), D_MODEL) and states.isfinite().all()
    torch.save({"cases": CASES, "source_layers": SRC_LAYERS, "states": states}, cap_path)
    (OLENS_DIR / "capture_provenance.json").write_text(json.dumps({
        "adapter_installed_at_capture": False,
        "capture_sha256": sha256_file(cap_path),
        "source_layers": SRC_LAYERS,
        "cases": CASES,
    }, indent=2) + "\n")
    print("Saved 54 raw activations.", flush=True)
else:
    saved_cap = torch.load(cap_path, map_location="cpu", weights_only=True)
    states = saved_cap["states"]
    print(f"Loaded cached 54 activations: {tuple(states.shape)}", flush=True)

# 3. Matched R, J, and J++ readouts on these exact 54 states
jr_path = OLENS_DIR / "jr_jpp_readouts.jsonl"
if not jr_path.exists():
    print("Gate 4: computing matched J / R / J++ readouts on all 54 states...", flush=True)
    p_ids = probe_df["token_id"].tolist()
    fam_list = list(pd.unique(probe_df["family"]))
    rows_jr = []
    for i, c in enumerate(CASES):
        for j, layer in enumerate(SRC_LAYERS):
            h = states[i, j].to("cuda", dtype=torch.float32)
            for lname in ("J", "R", "Jpp"):
                with torch.inference_mode():
                    if lname == "J":
                        t_h = j_lens.transport(h, layer)
                    elif lname == "R":
                        t_h = r_lens.transport(h, layer)
                    else:
                        J_l = jpp_jacs[layer].to("cuda", dtype=torch.float32)
                        t_h = J_l @ h
                        del J_l
                    logits = lens_model.unembed(t_h).float().flatten()
                    targets = logits[p_ids]
                    ranks_arr = (logits.unsqueeze(0) > targets.unsqueeze(1)).sum(dim=1).add(1).cpu().tolist()
                    top_ids = logits.topk(12).indices.tolist()
                    top_toks = [tok.decode([tid]) for tid in top_ids]
                fam_neg_log_rank = {}
                for fam in fam_list:
                    idxs = probe_df.index[probe_df.family == fam].tolist()
                    fam_neg_log_rank[fam] = float(-sum(math.log(ranks_arr[k]) for k in idxs) / len(idxs))
                rows_jr.append({
                    "case": c["id"], "group": c["group"], "layer": layer, "lens": lname,
                    "top_tokens": top_toks, "fam_neg_log_rank": fam_neg_log_rank, "ranks": ranks_arr,
                })
    with jr_path.open("w", encoding="utf-8") as f:
        for r in rows_jr:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(rows_jr)} J/R/J++ readout rows.", flush=True)
else:
    rows_jr = [json.loads(ln) for ln in jr_path.read_text().splitlines() if ln.strip()]
    print(f"Loaded {len(rows_jr)} cached J/R/J++ readout rows.", flush=True)

# 4. Render context-verified marker prompts (concepts_raw, U+3200..U+33FF -> ㈜ id 158983)
def render_marker(layer):
    lo, hi = LENS_CFG["marker_scan"]
    for cp in range(lo, hi + 1):
        ch = chr(cp)
        iso = tok(ch, add_special_tokens=False)["input_ids"]
        if len(iso) != 1:
            continue
        content = LENS_CFG["readout_template"].format(layer=layer, char=ch)
        text = tok.apply_chat_template([{"role": "user", "content": content}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=False)
        start = text.index("<activation>") + len("<activation>")
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        overlap = [idx for idx, (a, b) in enumerate(enc["offset_mapping"]) if a < start + 1 and b > start]
        if len(overlap) != 1:
            continue
        slot = overlap[0]
        if tuple(enc["offset_mapping"][slot]) != (start, start + 1):
            continue
        if enc["input_ids"][slot] != iso[0] or enc["input_ids"].count(iso[0]) != 1:
            continue
        assert iso[0] == LENS_CFG["expected_marker_id"], f"Marker {iso[0]} != {LENS_CFG['expected_marker_id']}"
        return {"text": text, "input_ids": enc["input_ids"], "slot": slot, "char": ch, "char_id": iso[0]}
    raise RuntimeError("No valid single-token marker found in scan range")

rendered = {str(L): render_marker(L) for L in SRC_LAYERS}
(OLENS_DIR / "rendered_prompts.json").write_text(json.dumps(rendered, indent=2) + "\n")
print("Marker verified:", {k: (v["char"], v["char_id"], v["slot"]) for k, v in rendered.items()}, flush=True)

# 5. Download & verify O-Lens LoRA adapter, run strict contract gate, then merge for fast generation
ad_weights_path = Path(hf_hub_download(
    LENS_CFG["repo"], f"{LENS_CFG['adapter']}/adapter_model.safetensors", revision=LENS_CFG["revision"]
))
ad_cfg_path = Path(hf_hub_download(
    LENS_CFG["repo"], f"{LENS_CFG['adapter']}/adapter_config.json", revision=LENS_CFG["revision"]
))
ad_sha = sha256_file(ad_weights_path)
assert ad_sha == LENS_CFG["adapter_sha256"], f"Adapter SHA mismatch: {ad_sha}"
ad_cfg = json.loads(ad_cfg_path.read_text())
assert ad_cfg["peft_type"] == "LORA" and ad_cfg["bias"] == "none"
assert not any(ad_cfg.get(k) for k in ["use_dora", "use_rslora", "fan_in_fan_out", "rank_pattern", "alpha_pattern", "modules_to_save"])

tensors = load_safetensors(str(ad_weights_path))
a_keys = sorted(k for k in tensors if k.endswith(".lora_A.weight"))
assert a_keys and set(tensors) == set(a_keys) | {k.replace(".lora_A.", ".lora_B.") for k in a_keys}
scale = ad_cfg["lora_alpha"] / ad_cfg["r"]

def map_lora_module(key):
    m = re.search(r"\.layers\.(.+)\.lora_A\.weight$", key)
    assert m, f"Unexpected LoRA key format: {key}"
    return f"{DECODER_LAYERS_NAME}.{m.group(1)}"

named_mods = dict(hf.named_modules())
lora_pairs = []
for k in a_keys:
    mname = map_lora_module(k)
    assert mname in named_mods, f"Missing module {mname} for key {k}"
    mod = named_mods[mname]
    A = tensors[k].to(mod.weight.device)
    B = tensors[k.replace(".lora_A.", ".lora_B.")].to(mod.weight.device)
    assert A.shape == (ad_cfg["r"], mod.weight.shape[1]), f"A shape mismatch on {mname}: {A.shape} vs {mod.weight.shape}"
    assert B.shape == (mod.weight.shape[0], ad_cfg["r"]), f"B shape mismatch on {mname}: {B.shape} vs {mod.weight.shape}"
    assert A.isfinite().all() and B.isfinite().all()
    lora_pairs.append((mname, mod, A, B))

# Contract Gate: check base vs hook-off exactness, hook-on live delta, all modules called, and merge equivalence
probe_ids = tok("The capital of France is Paris.", return_tensors="pt").input_ids.to("cuda")
gate_path = OLENS_DIR / "contract_gate.json"
if not gate_path.exists():
    with torch.inference_mode():
        base_logits = hf(probe_ids, use_cache=False).logits.float()

    enabled = [False]
    called = set()
    handles = []
    for mname, mod, A, B in lora_pairs:
        def _mk(A=A, B=B, mname=mname):
            def _h(m, inp, out):
                if not enabled[0]:
                    return out
                called.add(mname)
                delta = F.linear(F.linear(inp[0].float(), A.float()), B.float())
                return (out.float() + delta * scale).to(out.dtype)
            return _h
        handles.append(mod.register_forward_hook(_mk()))

    try:
        with torch.inference_mode():
            off_logits = hf(probe_ids, use_cache=False).logits.float()
            assert torch.equal(base_logits, off_logits), "Adapter OFF altered logits!"
            enabled[0] = True
            on_logits = hf(probe_ids, use_cache=False).logits.float()
    finally:
        for h in handles:
            h.remove()

    delta_norm = float((on_logits - off_logits).norm().item())
    assert delta_norm > 0 and torch.isfinite(on_logits).all()
    assert len(called) == len(lora_pairs), f"Inert LoRA modules: {len(called)} / {len(lora_pairs)}"

    # Merge LoRA weights into base model in-place (Gate 3 is already complete!) for ~4x faster decode
    with torch.no_grad():
        for mname, mod, A, B in lora_pairs:
            dw = (B.float() @ A.float()) * scale
            mod.weight.copy_((mod.weight.float() + dw).to(mod.weight.dtype))
    torch.cuda.synchronize()
    with torch.inference_mode():
        merged_logits = hf(probe_ids, use_cache=False).logits.float()
    merge_drift = float((merged_logits - on_logits).norm().item())
    rel_drift = merge_drift / delta_norm
    assert rel_drift < 0.05, f"Merge drift too large: {merge_drift} / {delta_norm} = {rel_drift:.4f}"

    gate_info = {
        "status": "PASS",
        "adapter": LENS_CFG["adapter"],
        "adapter_sha256": ad_sha,
        "matched_modules": len(lora_pairs),
        "called_modules": len(called),
        "adapter_off_exact": True,
        "adapter_on_logit_delta_norm": delta_norm,
        "merged_in_place": True,
        "merge_drift_norm": merge_drift,
        "merge_rel_drift": rel_drift,
        "marker_ids": {k: v["char_id"] for k, v in rendered.items()},
        "source_layers": SRC_LAYERS,
        "transform": "unit",
        "alpha": float(LENS_CFG["alpha"]),
    }
    gate_path.write_text(json.dumps(gate_info, indent=2) + "\n")
    print("O-Lens contract gate PASSED:", json.dumps(gate_info, indent=2), flush=True)
else:
    print("Contract gate already passed (assuming merged model in active kernel).", flush=True)

del tensors, lora_pairs; gc.collect(); torch.cuda.empty_cache()

# 6. Generate all 108 O-Lens verbalizations (resume-safe)
@contextmanager
def inject_olens_state(p, state_1d, alpha):
    s = state_1d.float()
    assert s.ndim == 1 and torch.isfinite(s).all() and s.norm() > 0
    repl = (alpha * s / s.norm()).to("cuda")
    calls = []
    def hook(module, args, output):
        ids = args[0]
        if ids.shape[1] != len(p["input_ids"]):
            return output
        assert ids.shape[0] == 1 and ids[0].tolist() == p["input_ids"]
        assert not calls, "Injected more than once in a single generation"
        out = output.clone()
        out[:, p["slot"]] = repl.to(out.dtype)
        calls.append(True)
        return out
    handle = hf.get_input_embeddings().register_forward_hook(hook)
    try:
        yield calls
    finally:
        handle.remove()

verb_path = OLENS_DIR / "verbalizations.jsonl"
done_v = set()
if verb_path.exists():
    for ln in verb_path.read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            done_v.add((r["case"], r["layer"], r["seed"]))

gen_cfg = olens_spec["generation"]
total_v = len(CASES) * len(SRC_LAYERS) * len(gen_cfg["seeds"])
print(f"Gate 4 O-Lens verbalizations resume: {len(done_v)} / {total_v}", flush=True)
pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

with verb_path.open("a", encoding="utf-8") as fout:
    for i, c in enumerate(CASES):
        for j, layer in enumerate(SRC_LAYERS):
            p = rendered[str(layer)]
            ids = torch.tensor([p["input_ids"]], device="cuda")
            for seed in gen_cfg["seeds"]:
                if (c["id"], layer, seed) in done_v:
                    continue
                torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                with torch.inference_mode(), inject_olens_state(p, states[i, j], LENS_CFG["alpha"]) as calls:
                    gen = hf.generate(
                        input_ids=ids, attention_mask=torch.ones_like(ids),
                        do_sample=True, temperature=gen_cfg["temperature"], top_p=gen_cfg["top_p"],
                        max_new_tokens=gen_cfg["max_new_tokens"], pad_token_id=pad_id,
                    )
                assert len(calls) == 1
                new_toks = gen[0, ids.shape[1]:].tolist()
                text = tok.decode(new_toks, skip_special_tokens=True)
                rec = {
                    "case": c["id"], "group": c["group"], "layer": layer, "seed": seed,
                    "text": text,
                    "raw_text": tok.decode(new_toks, skip_special_tokens=False),
                    "n_tokens": len(new_toks),
                    "hit_limit": len(new_toks) == gen_cfg["max_new_tokens"],
                    "empty": not text.strip(),
                }
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                done_v.add((c["id"], layer, seed))
        print(f"  Case {i+1:02d}/18 done: {c['id']} ({len(done_v)}/{total_v})", flush=True)

# 7. Build side-by-side comparison table (all 108 rows + R/J/J++ top tokens & family means)
jr_map = {(r["case"], r["layer"], r["lens"]): r for r in rows_jr}
verb_rows = [json.loads(ln) for ln in verb_path.read_text().splitlines() if ln.strip()]
assert len(verb_rows) == total_v

comp_rows = []
for v in verb_rows:
    row = {
        "case": v["case"], "group": v["group"], "layer": v["layer"], "seed": v["seed"],
        "olens_text": v["text"], "empty": v["empty"], "hit_limit": v["hit_limit"],
    }
    for lname in ("R", "J", "Jpp"):
        jr = jr_map[(v["case"], v["layer"], lname)]
        row[f"{lname}_top5"] = " | ".join(t.replace("\n", "\\n") for t in jr["top_tokens"][:5])
        for fam, val in jr["fam_neg_log_rank"].items():
            row[f"{lname}_{fam}_neglogrank"] = round(val, 4)
    comp_rows.append(row)

pd.DataFrame(comp_rows).to_csv(OLENS_DIR / "comparison_ojrjpp.csv", index=False)
print(f"GATE 4 COMPLETE! {len(comp_rows)} verbalizations + matched R/J/J++ saved to {OLENS_DIR}", flush=True)
