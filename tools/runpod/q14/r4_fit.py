# =====================================================================
# R4: J or R lens for Qwen3-14B with the published recipe (Gate 2a extension; r4_q14_spec.json).
#   MODE=J|R  DOC_OFFSET=0|25  [DIM_BATCH]  python r4_fit.py
# Sections 2-4 are gateL_fit.py VERBATIM (RelP rules, self-tests, bit-identical forward check, prompts, fit);
# only the model, target layer (n - 2) and outputs differ. No published 14B lens exists, so validation is
# internal (r4_compare.py). Environment: the main fast env (/root/gateL/.venv-fast). Model: MODEL_PATH from setup.
# =====================================================================
import hashlib, inspect, json, os, shutil, sys, time
from pathlib import Path
MODE = os.environ.get("MODE", "J")
FORCE_BOS = os.environ.get("FORCE_BOS", "0") == "1"
N_DOCS = int(os.environ.get("N_DOCS", "25"))
DOC_OFFSET = int(os.environ.get("DOC_OFFSET", "0"))
CKPT_EVERY = int(os.environ.get("CKPT_EVERY", "25"))
MAX_SEQ_LEN, SKIP_FIRST = 128, 4
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
OUT = Path("/root/r4"); OUT.mkdir(parents=True, exist_ok=True)
KEEP = Path("/workspace/r4_q14"); KEEP.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, "/root/gateL/jacobian-lens")
import torch, transformers, jlens
from datasets import load_dataset
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()

MODEL_PATH = os.environ["MODEL_PATH"]
MODEL_ID, REV = open("/root/gateL/MODEL").read().split()
assert MODEL_ID == "Qwen/Qwen3-14B", f"this pod's setup model is {MODEL_ID}"
hf = transformers.AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).cuda().eval()
tok = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)
layers = hf.model.layers
TARGET = len(layers) - 2                                   # 38 for 40 layers, as the published recipe (target n - 2)
env = {"model": MODEL_ID, "revision": REV, "torch": torch.__version__, "transformers": transformers.__version__,
       "gpu": torch.cuda.get_device_name(0), "jlens_commit": os.popen("git -C /root/gateL/jacobian-lens rev-parse HEAD").read().strip()}
print(env, "| target layer", TARGET)

# dim_batch: the largest of 16 / 8 / 4 that fits one document (Gate L used 16 at 4B; 27B needed 8)
DIM_BATCH = int(os.environ.get("DIM_BATCH", "0"))


if not DIM_BATCH:
    _fm = jlens.from_hf(hf, tok, force_bos=FORCE_BOS)
    _doc = load_dataset("NeelNanda/pile-10k", split="train")[DOC_OFFSET]["text"]
    for _db in (16, 8, 4):
        try:
            torch.cuda.empty_cache()
            jlens.jacobian_for_prompt(_fm, _doc, **{k: v for k, v in dict(source_layers=list(range(TARGET)), target_layer=TARGET,
                                      max_seq_len=MAX_SEQ_LEN, skip_first=SKIP_FIRST, dim_batch=_db).items()
                                      if k in inspect.signature(jlens.jacobian_for_prompt).parameters})
            DIM_BATCH = _db; break
        except torch.OutOfMemoryError:
            print(f"dim_batch {_db} does not fit; trying smaller")
    del _fm
    torch.cuda.empty_cache()
print("dim_batch", DIM_BATCH)

# ---------------------------------------------------------------------
# 2. RelP rules (MODE R only): exact forward, rule-modified gradient
# ---------------------------------------------------------------------
def st(exact, surrogate):
    """Forward value of `exact`; gradient of `surrogate` (straight-through)."""
    return exact.detach() + (surrogate - surrogate.detach())

def make_norm_forward(mod):
    src = inspect.getsource(type(mod).forward)
    zero_centered = "1.0 + self.weight" in src or "1 + self.weight" in src
    eps = getattr(mod, "variance_epsilon", getattr(mod, "eps", 1e-6))
    orig = mod.forward
    def fwd(x):
        y = orig(x)
        xf = x.float()
        rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps).detach()         # LN rule: scale held fixed
        gain = (1.0 + mod.weight.float()) if zero_centered else mod.weight.float()
        return st(y, (xf * rstd * gain).to(y.dtype))
    return fwd, zero_centered

def mlp_rule(g, u, a):
    """g = gate pre-activation, u = up projection, a = act_fn(g) (exact SiLU). Returns the product with
    identity rule on SiLU (sigmoid detached) and half rule (beta 0.5) on the product."""
    a_r = st(a, g * torch.sigmoid(g).detach())
    p = a * u
    return st(p, 0.5 * (a_r * u.detach()) + 0.5 * (a_r.detach() * u))

def make_mlp_forward(mlp):
    for name in ["gate_proj", "up_proj", "down_proj", "act_fn"]:
        assert hasattr(mlp, name), f"MLP layout differs: {mlp}"
    assert isinstance(mlp.act_fn, torch.nn.SiLU) or "silu" in type(mlp.act_fn).__name__.lower(), mlp.act_fn
    def fwd(x):
        g, u = mlp.gate_proj(x), mlp.up_proj(x)
        return mlp.down_proj(mlp_rule(g, u, mlp.act_fn(g)))
    return fwd

if MODE == "R":
    # gradient self-tests on random inputs, before touching the model
    g = torch.randn(64, dtype=torch.float32, requires_grad=True); u = torch.randn(64, requires_grad=True)
    out = mlp_rule(g, u, torch.nn.functional.silu(g)); out.sum().backward()
    assert torch.allclose(out, torch.nn.functional.silu(g) * u), "MLP rule changed the forward value"
    assert torch.allclose(g.grad, 0.5 * u.detach() * torch.sigmoid(g.detach()), atol=1e-6), "identity/half rule gradient wrong (gate)"
    assert torch.allclose(u.grad, 0.5 * torch.nn.functional.silu(g.detach()), atol=1e-6), "half rule gradient wrong (up)"
    norm0 = layers[0].input_layernorm
    nf, zc = make_norm_forward(norm0)
    x = torch.randn(1, 3, norm0.weight.shape[0], device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(x)
    (nf(x) * v).sum().backward()
    xf = x.detach().float(); rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + getattr(norm0, "variance_epsilon", getattr(norm0, "eps", 1e-6)))
    gain = (1.0 + norm0.weight.float()) if zc else norm0.weight.float()
    assert torch.allclose(x.grad.float(), v.float() * rstd * gain, rtol=2e-2, atol=2e-2), "LN rule gradient wrong"
    print(f"rule self-tests passed (norm zero-centered: {zc})")

    # forward must be bit-identical after patching
    ids = tok("The quick brown fox jumps over the lazy dog.", return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        before = hf(ids).logits.clone()
    for blk in layers:
        blk.input_layernorm.forward = make_norm_forward(blk.input_layernorm)[0]
        blk.post_attention_layernorm.forward = make_norm_forward(blk.post_attention_layernorm)[0]
        blk.mlp.forward = make_mlp_forward(blk.mlp)
    with torch.no_grad():
        after = hf(ids).logits
    assert torch.equal(before, after), f"patched forward differs (max |diff| {(before - after).abs().max().item()})"
    print("forward bit-identical after RelP patches: OK (block norms and MLPs patched; QK and gated norms untouched)")

# ---------------------------------------------------------------------
# 3. prompts and a timed first document
# ---------------------------------------------------------------------
fit_model = jlens.from_hf(hf, tok, force_bos=FORCE_BOS)
pile = load_dataset("NeelNanda/pile-10k", split="train")
prompts = [pile[i]["text"] for i in range(DOC_OFFSET, DOC_OFFSET + N_DOCS)]
prompts_sha = hashlib.sha256(json.dumps(prompts).encode()).hexdigest()
first25_sha = hashlib.sha256(json.dumps(prompts[:25]).encode()).hexdigest()
if DOC_OFFSET == 0:
    assert first25_sha.startswith("4111a9cf"), f"first 25 documents differ from the L1 set ({first25_sha[:16]})"

jfp = jlens.jacobian_for_prompt
kw = {k: v for k, v in dict(source_layers=list(range(TARGET)),   # what jlens.fit uses by default: every layer below target
                            target_layer=TARGET, max_seq_len=MAX_SEQ_LEN, skip_first=SKIP_FIRST,
                            dim_batch=DIM_BATCH).items() if k in inspect.signature(jfp).parameters}
torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time()
_ = jfp(fit_model, prompts[0], **kw)
torch.cuda.synchronize(); dt = time.time() - t0
peak = torch.cuda.max_memory_allocated() / 2**30; total = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"one document: {dt:.1f} s | peak memory {peak:.1f} / {total:.0f} GiB | projected fit: {dt * N_DOCS / 60:.0f} min")
del _

# ---------------------------------------------------------------------
# 4. fit (resumable) and save
# ---------------------------------------------------------------------
tag = f"q14_{MODE}_n{N_DOCS}{f'_off{DOC_OFFSET}' if DOC_OFFSET else ''}_bos{int(FORCE_BOS)}"
out, ckpt = OUT / f"{tag}.pt", OUT / f"{tag}.ckpt.pt"
if not out.exists():
    t0 = time.time()
    lens = jlens.fit(fit_model, prompts, target_layer=TARGET, max_seq_len=MAX_SEQ_LEN, skip_first=SKIP_FIRST,
                     dim_batch=DIM_BATCH, checkpoint_path=str(ckpt), checkpoint_every=CKPT_EVERY, resume=True)
    lens.save(str(out)); fit_min = (time.time() - t0) / 60
    ckpt.unlink(missing_ok=True)
    print(f"fit done in {fit_min:.1f} min -> {out}")
else:
    fit_min = None; print("refit exists:", out)


# ---------------------------------------------------------------------
# 5. record and keep (the lens is ~2 GB in fp16: kept on the volume, verified by hash after the copy)
# ---------------------------------------------------------------------
h_out = sha256(out)
dst = KEEP / out.name
if not dst.exists() or sha256(dst) != h_out:
    shutil.copyfile(out, dst)
assert sha256(dst) == h_out, "copy to the volume does not match"
(KEEP / f"{tag}.json").write_text(json.dumps({
    "mode": MODE, "force_bos": FORCE_BOS, "recipe": {"n_docs": N_DOCS, "doc_offset": DOC_OFFSET, "max_seq_len": MAX_SEQ_LEN,
    "skip_first": SKIP_FIRST, "target_layer": TARGET, "dim_batch": DIM_BATCH},
    "relp_rules": None if MODE == "J" else {"ln_rule": "block input/post-attention RMSNorms", "identity_rule": "MLP SiLU",
    "half_rule_beta": 0.5, "include_qk_norms": False, "gated_norms": "none in Qwen3"},
    "env": env, "prompts_sha256": prompts_sha, "first25_sha256": first25_sha, "first_doc_seconds": dt, "peak_gib": peak,
    "fit_minutes": fit_min, "lens_sha256": h_out}, indent=2) + "\n")
print("kept", dst, "and its JSON")
