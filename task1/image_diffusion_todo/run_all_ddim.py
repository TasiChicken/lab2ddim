"""Run all 20 required DDIM settings and collect FID into CSV.

Usage from task1/image_diffusion_todo:
    python run_all_ddim.py --ckpt_path /path/to/your/lab1_noise_checkpoint.ckpt
"""
import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

STEPS = [10, 20, 50, 100, 1000]
ETAS = [0.0, 0.2, 0.5, 1.0]


def run(cmd):
    print("\n$", " ".join(map(str, cmd)), flush=True)
    p = subprocess.run(cmd, text=True, capture_output=True)
    print(p.stdout)
    if p.stderr:
        print(p.stderr, file=sys.stderr)
    if p.returncode != 0:
        raise SystemExit(p.returncode)
    return p.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out_root", default="ddim_samples")
    ap.add_argument(
        "--predictor",
        choices=["noise", "x0", "mean"],
        default=None,
        help="For old Lab1 checkpoints without predictor metadata, pass --predictor noise.",
    )
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    eval_dir = here / "data/afhq/eval"
    fid_ckpt = here / "fid/afhq_inception_v3.ckpt"
    if not fid_ckpt.exists():
        raise FileNotFoundError(
            "Missing fid/afhq_inception_v3.ckpt. Run ../../kaggle_bootstrap.py first."
        )
    if not eval_dir.exists() or not any(eval_dir.iterdir()):
        run([sys.executable, "dataset.py"])

    rows = []
    for eta in ETAS:
        for steps in STEPS:
            save_dir = Path(args.out_root) / f"steps_{steps}_eta_{eta}"
            sample_cmd = [
                sys.executable, "sampling.py",
                "--ckpt_path", args.ckpt_path,
                "--save_dir", str(save_dir),
                "--sample_method", "ddim",
                "--ddim_steps", str(steps),
                "--eta", str(eta),
                "--batch_size", str(args.batch_size),
                "--gpu", str(args.gpu),
            ]
            if args.predictor is not None:
                sample_cmd += ["--predictor", args.predictor]
            run(sample_cmd)
            out = run([
                sys.executable, "fid/measure_fid.py",
                "data/afhq/eval/",
                str(save_dir),
            ])
            m = re.search(r"FID:\s*([0-9.eE+-]+)", out)
            fid = float(m.group(1)) if m else float("nan")
            rows.append({"eta": eta, "steps": steps, "fid": fid})

            with open("ddim_fid_results.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["eta", "steps", "fid"])
                w.writeheader()
                w.writerows(rows)

    print("\nFinished all 20 runs. Results: ddim_fid_results.csv")


if __name__ == "__main__":
    main()
