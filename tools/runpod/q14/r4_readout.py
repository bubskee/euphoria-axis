# =====================================================================
# R4 readout: frozen probe families on the reproduced Qwen3-14B S2 ladder, through our 14B J and R lenses.
#   conditions: baseline; S2 -0.5 and +0.5; three random directions at -0.5 (sign is arbitrary for a random direction)
#   steering: the paper's published S2 (layer 31) at layer 20, scaled to ratio 0.5908 at coefficient 1 (as M7)
#   prompts: the paper's NEUTRAL_50 (identical to the 4B ladder's 50 prompts)
#   readout: final prompt token, every source layer, J and R; family gain = mean over its probes of
#            mean over prompts of log(rank_baseline / rank_condition)   (the 4B definition)
# capture_all_layers, lens_readout, ranks_for_tokens: notebooks/03-decursed.ipynb cell 10, verbatim.
# Run in /root/gateL/.venv-fast after the fits.
# =====================================================================
import ast, hashlib, json, os, sys
from contextlib import contextmanager
from pathlib import Path
import numpy as np, pandas as pd, torch, transformers
sys.path.insert(0, "/root/gateL/jacobian-lens")
import jlens
from jlens.hooks import ActivationRecorder

K, SRC = Path("/workspace/r4_q14"), Path("/root/gate2a/src")
MODEL_PATH = os.environ["MODEL_PATH"]
hf = transformers.AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).cuda().eval()
tok = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)
lens_model = jlens.from_hf(hf, tok, force_bos=False)
j_lens = jlens.JacobianLens.load(str(K / "q14_J_n25_bos0.pt"))
r_lens = jlens.JacobianLens.load(str(K / "q14_R_n25_bos0.pt"))
all_layers = list(j_lens.source_layers)
decoder_layers = list(lens_model.layers)
INJECT, RATIO_AT_1, READ = 20, 0.5908, all_layers[-1]          # read at target - 1 (37), as layer 29 at 4B

# probes: psych_vocab_v1 unchanged; single-token rule of the 4B run, applied to this tokenizer
vocab = json.loads(Path("/workspace/lens-fit/psych_vocab_v1.json").read_text())["families"]
probe_terms = []
for family, terms in vocab.items():
    for term in terms:
        ids = tok.encode(" " + term, add_special_tokens=False)
        if len(ids) == 1:
            probe_terms.append({"family": family, "term": term, "token_id": ids[0]})
probe_terms = pd.DataFrame(probe_terms)
probe_ids = probe_terms.token_id.tolist()
print(f"{len(probe_terms)} single-token probes for this tokenizer (4B: 60 of 74)")

# steering: the M7 vectors (same construction, so the readout matches the behavioural run)
pub = torch.load(SRC / "results/3.2_pain_vectors/pain_vectors/Qwen_3_14B_base/pain_vectors.pt", map_location="cpu", weights_only=False)
v_raw = pub["s2_pain_vector"].float()
NEUTRAL_50 = next(ast.literal_eval(n.value) for n in ast.parse((SRC / "scripts/4.2_steering/01_steering_ladder.py").read_text()).body
                  if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "NEUTRAL_50")
norms, cap = [], {}
def _cap(m, i, o):
    cap["n"] = (o[0] if isinstance(o, tuple) else o)[0, -1].float().norm().item()
h = hf.model.layers[INJECT].register_forward_hook(_cap)
with torch.no_grad():
    for p in NEUTRAL_50[:3]:
        hf(**tok(p, return_tensors="pt").to(hf.device)); norms.append(cap["n"])
h.remove()
ratio_raw = float(v_raw.norm()) / (sum(norms) / len(norms))
VEC = {"S2": v_raw * (RATIO_AT_1 / ratio_raw)}
for seed in (1103, 2207, 3301):
    r = torch.randn(v_raw.numel(), generator=torch.Generator().manual_seed(seed))
    VEC[f"random_{seed}"] = r / r.norm() * VEC["S2"].norm()
print(f"raw vector ratio at layer {INJECT}: {ratio_raw:.4f} -> scaled to {RATIO_AT_1}")

@contextmanager
def prefix_steering(coeff, vector):                              # the 4B hook's arithmetic, at layer 20
    if vector is None or coeff == 0:
        yield
        return
    delta = vector.to(device=next(hf.parameters()).device, dtype=torch.bfloat16)
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            return (output[0] + coeff * delta, *output[1:])
        return output + coeff * delta
    handle = decoder_layers[INJECT].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()

# ---- notebooks/03-decursed.ipynb cell 10, verbatim ----
@torch.no_grad()
def capture_all_layers(text, coeff=0.0, vector=None):
    ids = lens_model.encode(text)

    with prefix_steering(coeff, vector):
        with ActivationRecorder(
            lens_model.layers,
            at=all_layers,
        ) as recorder:
            lens_model.forward(ids)

    states = {
        layer: (
            recorder.activations[layer][0, -1, :]
            .detach()
            .float()
        )
        for layer in all_layers
    }

    return ids.cpu(), states

def lens_readout(lens, state, layer):
    transported = lens.transport(state, layer)

    return (
        lens_model
        .unembed(transported)
        .float()
        .cpu()
    )

def ranks_for_tokens(logits, token_ids):
    """
    Compute 1-indexed ranks for a small token set without sorting
    the entire vocabulary once per token.
    """
    targets = logits[token_ids]

    # [n_probe, vocab]
    return (
        (logits.unsqueeze(0) > targets.unsqueeze(1))
        .sum(dim=1)
        .add(1)
        .tolist()
    )
# ---- end verbatim ----

COND = {"baseline": (0.0, None), "S2_-0.5": (-0.5, VEC["S2"]), "S2_+0.5": (0.5, VEC["S2"]),
        **{f"random_{s}_-0.5": (-0.5, VEC[f"random_{s}"]) for s in (1103, 2207, 3301)}}
ranks = {c: np.zeros((len(NEUTRAL_50), len(all_layers), 2, len(probe_ids)), np.int64) for c in COND}
for c, (coeff, vec) in COND.items():
    for i, text in enumerate(NEUTRAL_50):
        _, states = capture_all_layers(text, coeff, vec)
        for li, l in enumerate(all_layers):
            for k, lens in enumerate((j_lens, r_lens)):
                ranks[c][i, li, k] = ranks_for_tokens(lens_readout(lens, states[l], l), probe_ids)
    print("done:", c)
np.savez_compressed(K / "r4_ladder_ranks.npz", **{c: v for c, v in ranks.items()})
probe_terms.to_csv(K / "r4_probe_terms.csv", index=False)

# family gains (the 4B definition) at the read layer, J and R
fam = probe_terms.family.to_numpy()
rows = []
for c in COND:
    if c == "baseline":
        continue
    g = np.log(ranks["baseline"] / ranks[c]).mean(0)                 # [layers, lens, probes]
    for k, ln in enumerate("JR"):
        for f in pd.unique(fam):
            rows.append({"condition": c, "lens": ln, "family": f, "gain_read_layer": float(g[all_layers.index(READ), k, fam == f].mean()),
                         "gain_by_layer": [float(x) for x in g[:, k, fam == f].mean(1)]})
T = pd.DataFrame(rows)
piv = T[T.lens == "R"].pivot(index="family", columns="condition", values="gain_read_layer")
with pd.option_context("display.width", 220):
    print(f"\nR lens family gains at layer {READ}:\n", piv.round(3).to_string())
# R4-P1 (frozen, the 4B D4 analogue): per random direction, mean |gain| over the six focus families divided by S2 -0.5's;
# passes if the ratio is < 0.5 for at least 2 of the 3 random directions (R lens, read layer)
FOCUS = ["generic_positive", "activated_vitality", "relief_release", "agency_assurance", "approach_drive", "distress_self_devaluation"]
s2_amp = piv.loc[FOCUS, "S2_-0.5"].abs().mean()
ratios = {s: float(piv.loc[FOCUS, f"random_{s}_-0.5"].abs().mean() / s2_amp) for s in (1103, 2207, 3301)}
p1 = sum(r < 0.5 for r in ratios.values()) >= 2
for s, r in ratios.items():
    print(f"random_{s}_-0.5 / S2_-0.5, focus-family amplitude: {r:.2f}")
print(f"R4-P1 (random < 0.5 x S2 for >= 2 of 3): {'PASS' if p1 else 'FAIL'}")
(K / "r4_readout.json").write_text(json.dumps({"R4_P1": {"pass": bool(p1), "ratios": ratios}, "read_layer": READ, "inject_layer": INJECT, "ratio_raw": ratio_raw,
    "n_probes": len(probe_terms), "vector_sha256": {k: hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest() for k, v in VEC.items()},
    "rows": rows}, indent=2) + "\n")
print("saved r4_ladder_ranks.npz, r4_probe_terms.csv, r4_readout.json in", K)
