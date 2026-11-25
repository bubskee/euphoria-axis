# =====================================================================
# Pack lightweight tables & scorecards for git (runs/qwen35-27b-gate2/)
# =====================================================================
import tarfile

bundle = OUT_DIR / "qwen35-27b-gate2-lightweight.tar.gz"
light_files = sorted(p for p in OUT_DIR.iterdir() if p.is_file() and p.suffix in (".json", ".csv"))
with tarfile.open(bundle, "w:gz") as tf:
    for p in light_files:
        tf.add(p, arcname=p.name)

print(f"Packed {len(light_files)} lightweight files into {bundle} ({bundle.stat().st_size / 1e3:.1f} KB):")
for p in light_files:
    print(f"  {p.name:<45} {p.stat().st_size / 1e3:7.1f} KB")

# Optional direct browser download in Colab:
# from google.colab import files; files.download(str(bundle))
        
