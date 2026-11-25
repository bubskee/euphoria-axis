# =====================================================================
# R4 validation (no published 14B lens exists): per-layer comparisons of our Qwen3-14B lenses.
#   1 - cos(J, R)            shape expected from 4B: large early, near 0 late
#   1 - cos(R, R other docs) document sensitivity (Gate L: 4B R moved 0.117 at layer 0, 0.0003 at layer 29)
# Run in /root/gateL/.venv-fast after the three fits.
# =====================================================================
import json
from pathlib import Path
import torch

K = Path("/workspace/r4_q14")
J, R, R2 = (K / f"q14_{m}_n25{o}_bos0.pt" for m, o in [("J", ""), ("R", ""), ("R", "_off25")])

def stack(path):
    d = torch.load(path, map_location="cpu", weights_only=False); M = d["J"]
    return M.float() if torch.is_tensor(M) else torch.stack([M[k].float() for k in sorted(M)])
cos = lambda a, b: float((a.flatten().double() @ b.flatten().double()) / (a.double().norm() * b.double().norm()))
j, r = stack(J), stack(R)
r2 = stack(R2) if R2.exists() else None
n = j.shape[0]
ref4b = {"J_vs_R": {0: 0.307, 16: 0.112, 29: 0.004}, "R_vs_R_docs": {0: 0.117, 16: 0.013, 29: 0.0003}}   # 1 - cos at 4B
rows = []
print(f"{'layer':>5}  {'1-cos(J,R)':>11}  {'1-cos(R,R docs 25-49)':>22}")
for l in range(n):
    jr = 1 - cos(j[l], r[l]); rr = (1 - cos(r[l], r2[l])) if r2 is not None else None
    rows.append({"layer": l, "J_vs_R": jr, "R_vs_R_other_docs": rr})
    if l % 4 == 0 or l >= n - 3:
        print(f"{l:5d}  {jr:11.2e}  {rr:22.2e}" if rr is not None else f"{l:5d}  {jr:11.2e}  {'(no second R yet)':>22s}")
print("\n4B reference (1 - cos; Gate L): J vs R at layers 0/16/29:", ref4b["J_vs_R"],
      "| R vs R other documents:", ref4b["R_vs_R_docs"])
(K / "r4_compare.json").write_text(json.dumps({"n_layers_lens": n, "rows": rows, "reference_4b": ref4b}, indent=2) + "\n")
print("saved", K / "r4_compare.json")
