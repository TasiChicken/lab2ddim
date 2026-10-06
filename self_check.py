"""Fast CPU-only structural/smoke check for the completed Lab2 package."""
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def check_py_compile():
    files = [str(p) for p in ROOT.rglob("*.py") if "__pycache__" not in p.parts]
    subprocess.run([sys.executable, "-m", "py_compile", *files], check=True)
    print(f"[OK] py_compile: {len(files)} Python files")


def check_notebooks():
    n = 0
    for p in ROOT.rglob("*.ipynb"):
        obj = json.loads(p.read_text(encoding="utf-8"))
        assert obj.get("nbformat") == 4, p
        n += 1
    print(f"[OK] notebooks: {n}")


def check_datasets():
    style = ROOT / "task2/sample_data/artistic-custom/train"
    rows = [json.loads(x) for x in (style / "metadata.jsonl").read_text().splitlines() if x.strip()]
    assert len(rows) == 16
    assert all((style / r["file_name"]).exists() for r in rows)
    robot = list((ROOT / "task2/sample_data/dreambooth-robot").glob("*.png"))
    assert len(robot) == 8
    print("[OK] Task2 sample datasets: 16 style images, 8 identity images")


def check_todo_markers():
    must_contain = [
        ROOT / "task1/2d_plot_diffusion_todo/network.py",
        ROOT / "task1/2d_plot_diffusion_todo/ddpm.py",
        ROOT / "task1/image_diffusion_todo/model.py",
        ROOT / "task1/image_diffusion_todo/scheduler.py",
        ROOT / "task2/scripts/train_lora_custom.sh",
        ROOT / "task2/scripts/train_dreambooth_lora.sh",
    ]
    for p in must_contain:
        assert "TODO" in p.read_text(encoding="utf-8"), p
    print("[OK] original TODO markers retained")


if __name__ == "__main__":
    check_py_compile()
    check_notebooks()
    check_datasets()
    check_todo_markers()
    print("Basic package self-check passed.")
