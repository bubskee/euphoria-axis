# Pack lightweight Gate 3 + Gate 4 outputs into /content/gate3_lightweight.tar.gz
import tarfile
from pathlib import Path

OUT_DIR = Path("/content/gate3_out")
tar_path = Path("/content/gate3_lightweight.tar.gz")
SKIP_SUFFIXES = (".pt",)
SKIP_NAMES = {"m4_buttons_unsteered_chat.jsonl", "m4_buttons_steered_chat.jsonl"}

with tarfile.open(tar_path, "w:gz") as tf:
    for p in sorted(OUT_DIR.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix in SKIP_SUFFIXES or p.name in SKIP_NAMES:
            continue
        rel = p.relative_to(OUT_DIR)
        tf.add(p, arcname=str(rel))
        print(f"  + {rel} ({p.stat().st_size / 1024:.1f} KB)")

print(f"Packed: {tar_path} ({tar_path.stat().st_size / 1024:.1f} KB)", flush=True)
