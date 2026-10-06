"""Small Kaggle helper for the two unchanged upstream Task-2 training scripts.

Task 2 only asks students to edit the marked shell/notebook configuration regions;
no LoRA algorithm implementation is required.  To keep this archive compact while
still runnable, the large unchanged official training programs are fetched from the
exact 2026 starter commit the first time they are used.
"""
from pathlib import Path
import runpy
import urllib.request

COMMIT = "0ecab027c5f5199d55ebcb058035c0e568e023e3"
BASE = f"https://raw.githubusercontent.com/Jing-En-Huang/Lab2-DDIM-LoRA/{COMMIT}/task2"


def run_official(filename: str):
    here = Path(__file__).resolve().parent
    cache = here / ".official_cache"
    cache.mkdir(exist_ok=True)
    dst = cache / filename
    if not dst.exists():
        url = f"{BASE}/{filename}"
        print(f"[Lab2] Downloading unchanged official {filename} ...")
        urllib.request.urlretrieve(url, dst)
    runpy.run_path(str(dst), run_name="__main__")
