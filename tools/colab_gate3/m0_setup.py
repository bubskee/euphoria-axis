# M0: Setup pinned repos, load Qwen/Qwen3.6-27B (bf16), and published J / R / J++ lenses in persistent Colab kernel.
import os, subprocess, sys
from pathlib import Path

OUT_DIR = Path("/content/gate3_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)
IN_DIR = Path("/content/inputs")
WORK = Path("/content/work")
WORK.mkdir(exist_ok=True)
os.environ["HF_HOME"] = "/content/hf_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
RUN_BUTTONS = True

def ensure_repo(url, dest, commit=None):
    dest = Path(dest)
    if not dest.exists():
        subprocess.run(["git", "clone", url, str(dest)], check=True)
    if commit:
        subprocess.run(["git", "-C", str(dest), "checkout", commit], check=True)
    head = subprocess.check_output(["git", "-C", str(dest), "rev-parse", "HEAD"], text=True).strip()
    print(f"Repo {dest.name}: {head[:12]}", flush=True)
    return dest

PAIN_DIR = ensure_repo("https://github.com/valen-research/Pain-axis.git", WORK / "pain-axis-source", "8d1649c03a63a39c9aa092532c376800cc4a3863")
JLENS_DIR = ensure_repo("https://github.com/anthropics/jacobian-lens.git", WORK / "jacobian-lens", "581d398613e5602a5af361e1c34d3a92ea82ba8e")
if str(JLENS_DIR) not in sys.path:
    sys.path.insert(0, str(JLENS_DIR))

import ast, gc, hashlib, json, platform, re, time
from collections import Counter
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from huggingface_hub import hf_hub_download
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from tqdm.auto import tqdm
import jlens

sha256_bytes = lambda b: hashlib.sha256(b).hexdigest()
sha256_file = lambda p: sha256_bytes(Path(p).read_bytes())

MODEL_ID = "Qwen/Qwen3.6-27B"
MODEL_REV = "1cda14dd71a3f47115850eb790f416f165da1cb5"
LENS_REPO = "camilablank/workspace-lenses"
LENS_REV = "d740106d1e0f95456dc8718fba2895e9c8ffd6ef"
JPP_REPO = "koayon/jpp-lenses"
JPP_REV = "237dab6ea0e611d964bf407f13ba1655f1a6ae43"

EXTRACT_BLOCK = 62
INJECT_BLOCK = 32
READ_LAYER = 61
RATIO_AT_1 = 0.5908
RANDOM_SEEDS = [1103, 2207, 3301, 4409, 5503]
BOOT_N = 20_000
BOOT_SEED = 20260925
FOCUS6 = [
    "generic_positive", "activated_vitality", "relief_release",
    "agency_assurance", "approach_drive", "distress_self_devaluation",
]

SPEC_PATH = IN_DIR / "gate3_q36_27b_spec.json"
OLENS_SPEC_PATH = IN_DIR / "gate4_olens_q36_27b_spec.json"
VOCAB_PATH = IN_DIR / "psych_vocab_v1.json"
PAIRS_PATH = IN_DIR / "matched_pairs_validated.jsonl"
DS31_PATH = PAIN_DIR / "datasets/3.1_pain_and_control_datasets.json"
DS420_PATH = PAIN_DIR / "datasets/4.1_self_other_420_scenarios.json"

assert sha256_file(VOCAB_PATH).startswith("fa4feb2d06d4154b"), "psych_vocab_v1 hash mismatch"
assert sha256_file(PAIRS_PATH) == "fa4d8ef572f06f5a6bbcdbad87ca34d2ddcdc2875dfc8f56b85774d60d0fa184", "matched_pairs hash mismatch"
assert sha256_file(DS31_PATH).startswith("2c3712b4b096"), "3.1 dataset hash mismatch"
assert sha256_file(DS420_PATH) == "901ea558faf34c69caff59321775723e20bebf49a77d20070f6b85a0368a3291", "420 scenarios hash mismatch"

VOCAB = json.loads(VOCAB_PATH.read_text())["families"]

NEUTRAL_50 = next(
    ast.literal_eval(n.value)
    for n in ast.parse((PAIN_DIR / "scripts/4.2_steering/01_steering_ladder.py").read_text()).body
    if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "NEUTRAL_50"
)
assert len(NEUTRAL_50) == 50

t0 = time.time()
tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REV)
hf = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL_ID, revision=MODEL_REV, dtype=torch.bfloat16, device_map="cuda"
).eval()
torch.cuda.synchronize()
load_s = time.time() - t0

DECODER_LAYERS_NAME, decoder_layers = next(
    (n, m) for n, m in hf.named_modules()
    if isinstance(m, torch.nn.ModuleList) and len(m) >= 60 and hasattr(m[0], "mlp") and hasattr(m[0], "input_layernorm")
)
N_LAYERS = len(decoder_layers)
D_MODEL = hf.get_input_embeddings().weight.shape[1]
assert N_LAYERS == 64 and D_MODEL == 5120, f"Unexpected shape: {N_LAYERS} x {D_MODEL}"
print(f"Loaded {MODEL_ID} in {load_s:.1f}s (layers={DECODER_LAYERS_NAME}, {N_LAYERS}x{D_MODEL})", flush=True)

# Fast-path probe (honest logging; inference works on both fast path and eager)
calls = set()
def _prof(frame, event, arg):
    if event == "call" and any(k in frame.f_code.co_name for k in ("delta_rule", "conv1d")):
        calls.add(f"{frame.f_globals.get('__name__')}.{frame.f_code.co_name}")
_ids = tok("hello world " * 10, return_tensors="pt").input_ids.to("cuda")
sys.setprofile(_prof)
with torch.inference_mode():
    hf(_ids)
sys.setprofile(None)
fast_path = any(c.startswith(("fla", "causal_conv1d")) for c in calls) and not any("torch_" in c for c in calls)

# Download & hash published J, R, and J++ lenses
j_path = hf_hub_download(LENS_REPO, "qwen3.6-27b/j-lens/lens.pt", revision=LENS_REV)
r_path = hf_hub_download(LENS_REPO, "qwen3.6-27b/r-lens/lens.pt", revision=LENS_REV)
jpp_path = hf_hub_download(JPP_REPO, "qwen3.6-27b/lens.pt", revision=JPP_REV)
j_sha, r_sha, jpp_sha = sha256_file(j_path), sha256_file(r_path), sha256_file(jpp_path)
assert j_sha == "a036b35843d389b6655df721711917436fb79c83358c1861a4d58ad103a02724", f"j_lens sha mismatch: {j_sha}"
assert r_sha == "fe4d0b6a17318760e67e7a5ed417fc5dbb8e66129c6944656f506b2f10ce6192", f"r_lens sha mismatch: {r_sha}"
assert jpp_sha == "adc103d4e724d04e721b019ecafa1388ff376541db3ecb78f693e60ed1d8609d", f"jpp_lens sha mismatch: {jpp_sha}"

j_lens = jlens.JacobianLens.load(j_path)
r_lens = jlens.JacobianLens.load(r_path)
jpp_raw = torch.load(jpp_path, map_location="cpu", weights_only=True)
jpp_jacs = jpp_raw["parameters"]["jacobians"]
assert all(l in jpp_jacs for l in range(INJECT_BLOCK, READ_LAYER + 1)), "Missing J++ layers"
lens_model = jlens.from_hf(hf, tok, force_bos=False)

HEADER = {
    "model_id": MODEL_ID,
    "model_revision": MODEL_REV,
    "decoder_layers_name": DECODER_LAYERS_NAME,
    "n_layers": N_LAYERS,
    "d_model": D_MODEL,
    "gpu": torch.cuda.get_device_name(0),
    "gpu_mem_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
    "mem_after_load_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
    "load_seconds": round(load_s, 1),
    "fast_path": fast_path,
    "fast_path_calls": sorted(calls)[:10],
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "python": platform.python_version(),
    "lens_repo": LENS_REPO,
    "lens_revision": LENS_REV,
    "j_lens_sha256": j_sha,
    "r_lens_sha256": r_sha,
    "jpp_lens_repo": JPP_REPO,
    "jpp_lens_revision": JPP_REV,
    "jpp_lens_sha256": jpp_sha,
    "spec_sha256": sha256_file(SPEC_PATH) if SPEC_PATH.exists() else None,
    "olens_spec_sha256": sha256_file(OLENS_SPEC_PATH) if OLENS_SPEC_PATH.exists() else None,
}
(OUT_DIR / "run_header.json").write_text(json.dumps(HEADER, indent=2) + "\n")
print(json.dumps(HEADER, indent=2), flush=True)

def boot_ci(arr, n_boot=BOOT_N, seed=BOOT_SEED):
    arr = np.asarray(arr, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(arr.mean()), float(lo), float(hi)

@contextmanager
def prefix_steering(coeff, vector, layer=INJECT_BLOCK):
    if vector is None or coeff == 0:
        yield
        return
    delta = vector.to(device="cuda", dtype=torch.bfloat16)
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            return (output[0] + coeff * delta, *output[1:])
        return output + coeff * delta
    handle = decoder_layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()

@torch.inference_mode()
def capture_final_token(prompts, layers_to_capture, batch_size=8):
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    layers_to_capture = list(layers_to_capture)
    out_chunks = []
    for start in tqdm(range(0, len(prompts), batch_size), leave=False):
        batch = prompts[start:start + batch_size]
        enc = tok(batch, padding=True, return_tensors="pt").to("cuda")
        buf = {}
        handles = []
        for l in layers_to_capture:
            def _make_hook(li):
                def _h(m, i, o):
                    t = o[0] if isinstance(o, tuple) else o
                    buf[li] = t[:, -1, :].detach().float().cpu()
                return _h
            handles.append(decoder_layers[l].register_forward_hook(_make_hook(l)))
        try:
            hf(**enc, use_cache=False)
        finally:
            for h in handles:
                h.remove()
        out_chunks.append(torch.stack([buf[l] for l in layers_to_capture], dim=1))
    return torch.cat(out_chunks, dim=0)
