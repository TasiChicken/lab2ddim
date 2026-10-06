"""Download large/unchanged official Lab2 support files on Kaggle.

The assignment submission explicitly excludes checkpoints/datasets.  This helper keeps
this chat-delivered ZIP small while pinning every downloaded support file to the exact
2026 starter commit used to build the package.
"""
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
COMMIT = "0ecab027c5f5199d55ebcb058035c0e568e023e3"
RAW = f"https://raw.githubusercontent.com/Jing-En-Huang/Lab2-DDIM-LoRA/{COMMIT}"

FILES = {
    ROOT / "task1/image_diffusion_todo/fid/afhq_inception_v3.ckpt":
        f"{RAW}/task1/image_diffusion_todo/fid/afhq_inception_v3.ckpt",
    # Task 2 requires configuration only; the large trainer implementations below
    # are unchanged official starter files.  Cache them before training.
    ROOT / "task2/.official_cache/train_lora.py":
        f"{RAW}/task2/train_lora.py",
    ROOT / "task2/.official_cache/train_dreambooth_lora.py":
        f"{RAW}/task2/train_dreambooth_lora.py",
}

for path, url in FILES.items():
    min_size = 80_000_000 if path.suffix == ".ckpt" else 30_000
    if path.exists() and path.stat().st_size > min_size:
        print(f"[OK] {path.relative_to(ROOT)} already exists")
        continue
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[DOWNLOAD] {path.relative_to(ROOT)}")
    urllib.request.urlretrieve(url, path)
    print(f"[OK] {path.relative_to(ROOT)} ({path.stat().st_size / 1024**2:.2f} MiB)")

print("Bootstrap complete.")
