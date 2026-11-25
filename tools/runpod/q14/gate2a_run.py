# =====================================================================
# Gate 2a runner: the source paper's own scripts, unmodified, with only the overrides listed in gate2a_spec.json.
#   python gate2a_run.py R1    extraction (3.2) for Qwen3-14B
#   python gate2a_run.py R2a   S2 ladder (4.2) at layer 20 with the PUBLISHED vector
#   python gate2a_run.py R2b   S2 ladder at layer 20 with OUR R1 vector (run R1 first)
# Outputs: /root/gate2a/out/<step>/ (container disk); small results copied to /workspace/gate2a/<step>/.
# Run in /root/gate2a/.venv-tl after gate2a_env.sh.
# =====================================================================
import builtins, importlib.util, json, os, shutil, sys, time
from pathlib import Path

STEP = sys.argv[1] if len(sys.argv) > 1 else ""
assert STEP in ("R1", "R2a", "R2b"), "usage: gate2a_run.py R1|R2a|R2b"
G, SRC = Path("/root/gate2a"), Path("/root/gate2a/src")
OUT, KEEP = G / "out" / STEP.lower(), Path("/workspace/gate2a") / STEP.lower()
MODEL = ("Qwen/Qwen3-14B", "Qwen_3_14B_base")

# model pinned: resolve "Qwen/Qwen3-14B" through the local refs/main written by gate2a_env.sh (the verified revision)
os.environ.update(HF_HOME="/root/hf_cache", HF_HUB_OFFLINE="1")
os.environ.pop("HF_TOKEN", None)                 # both scripts call login() when a token is set; that needs the network
hdr = json.loads((G / "run_header.json").read_text())
print(f"Gate 2a {STEP}: source {hdr['source_commit'][:12]}, {hdr['model']} @ {hdr['model_revision'][:12]}, spec {hdr['spec_sha256'][:12]}")
OUT.mkdir(parents=True, exist_ok=True)
os.chdir(SRC)                                    # the scripts use paths relative to the repository root

def load(rel):
    spec = importlib.util.spec_from_file_location(Path(rel).stem.replace(".", "_"), SRC / rel)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

t0 = time.time()
if STEP == "R1":
    m = load("scripts/3.2_pain_vectors/01_extract_activations_and_pain_vectors.py")
    m.MODELS = [MODEL]
    m.OUTPUT_DIR = OUT                           # not /workspace/results: activations.pt is several GB
    m.LOG_FILE = OUT / "batch_log.txt"
    m.main()
    produced = OUT / MODEL[1]
    assert (produced / "summary.json").exists(), "extraction did not produce summary.json"
else:
    lad = load("scripts/4.2_steering/01_steering_ladder.py")
    if STEP == "R2a":                            # the published vector, laid out where the ladder looks for it
        vec_root = G / "vec_published"
        dst = vec_root / MODEL[1] / "final_token" / "pain_vectors.pt"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SRC / "results/3.2_pain_vectors/pain_vectors" / MODEL[1] / "pain_vectors.pt", dst)
    else:
        vec_root = G / "out" / "r1"
        assert (vec_root / MODEL[1] / "final_token" / "pain_vectors.pt").exists(), "run R1 first"
    lad.RESULTS_DIR, lad.OUT_DIR = vec_root, OUT
    lad.RUN_MODELS = [MODEL]
    lad.CLEAR_HF_CACHE_AFTER_RUN = False         # it would delete every Hugging Face cache on the pod
    lad.MIN_FREE_GB = 20
    printed = []
    def capture_print(*a, **k):                  # the ladder only prints its automatic layer pick and ratios
        printed.append(" ".join(str(x) for x in a)); builtins.print(*a, **k)
    lad.print = capture_print
    def answer(prompt=""):
        builtins.print(prompt + "20   [answered by gate2a_run.py: the published layer]")
        return "20"
    lad.input = answer
    lad.main()
    (OUT / "ladder_stdout.txt").write_text("\n".join(printed) + "\n")
    produced = OUT

# keep the small results on the volume (everything except activations.pt). copyfile, not copy2: the network volume
# refuses metadata changes (timestamps, permissions), so copy contents only
KEEP.mkdir(parents=True, exist_ok=True)
for p in produced.rglob("*"):
    if p.is_file() and p.name != "activations.pt" and p.stat().st_size < 50 * 2**20:
        dst = KEEP / p.relative_to(produced); dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, dst)
shutil.copyfile(G / "run_header.json", KEEP / "run_header.json")
print(f"Gate 2a {STEP} done in {(time.time() - t0) / 60:.1f} min -> {OUT} (small results copied to {KEEP})")
