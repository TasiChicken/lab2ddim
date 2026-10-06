"""Run the 20 required DDIM experiments on two GPUs without changing any experiment setting.

Each (eta, steps) combination is still a normal single-GPU sampling/FID run.
We only distribute independent combinations across GPU 0 and GPU 1.

Default balanced assignment:
    GPU 0: eta = 0.0, 0.5  (10 runs)
    GPU 1: eta = 0.2, 1.0  (10 runs)

Required settings kept unchanged:
    steps = [10, 20, 50, 100, 1000]
    eta   = [0.0, 0.2, 0.5, 1.0]
    500 generated images per combination (sampling.py)
    batch_size = 64 by default
    exact beta schedule copied from the Lab1 checkpoint
    noise predictor only for Lab2 DDIM

Usage from task1/image_diffusion_todo:
    python run_all_ddim.py \
        --ckpt_path /kaggle/input/datasets/puppypoo/quadratic-noise/step_35000.ckpt \
        --predictor noise \
        --gpus 0 1
"""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

STEPS = [10, 20, 50, 100, 1000]
ETAS = [0.0, 0.2, 0.5, 1.0]
EXPECTED_IMAGES = 500

# Two equal workloads: each GPU gets two eta rows and every step count.
DEFAULT_ASSIGNMENT = {
    0: [0.0, 0.5],
    1: [0.2, 1.0],
}

FIELDS = [
    "gpu", "eta", "steps", "fid", "sample_count",
    "sampling_seconds", "fid_seconds", "total_seconds",
    "save_dir", "sampling_log", "fid_log",
]


def run_to_log(cmd, log_path, env, cwd):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"$ {' '.join(map(str, cmd))}", flush=True)
    start = time.time()
    with log_path.open("w", encoding="utf-8") as f:
        p = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            text=True,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
    elapsed = time.time() - start
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {p.returncode}. See {log_path}"
        )
    return elapsed


def run_capture(cmd, log_path, env, cwd):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"$ {' '.join(map(str, cmd))}", flush=True)
    start = time.time()
    p = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
    )
    elapsed = time.time() - start
    log_path.write_text(
        (p.stdout or "") + ("\n[stderr]\n" + p.stderr if p.stderr else ""),
        encoding="utf-8",
    )
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {p.returncode}. See {log_path}"
        )
    return p.stdout, elapsed


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (float(r["eta"]), int(r["steps"]))))


def read_rows(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def combo_key(eta, steps):
    return (float(eta), int(steps))


def count_pngs(save_dir):
    return len(list(save_dir.glob("*.png")))


def config_matches(save_dir, ckpt_path, eta, steps, batch_size):
    cfg_path = save_dir / "run_config.json"
    if not cfg_path.exists():
        return False
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    expected = {
        "ckpt_path": str(Path(ckpt_path).resolve()),
        "eta": float(eta),
        "steps": int(steps),
        "batch_size": int(batch_size),
        "sample_method": "ddim",
        "predictor": "noise",
        "expected_images": EXPECTED_IMAGES,
    }
    return cfg == expected


def save_config(save_dir, ckpt_path, eta, steps, batch_size):
    cfg = {
        "ckpt_path": str(Path(ckpt_path).resolve()),
        "eta": float(eta),
        "steps": int(steps),
        "batch_size": int(batch_size),
        "sample_method": "ddim",
        "predictor": "noise",
        "expected_images": EXPECTED_IMAGES,
    }
    (save_dir / "run_config.json").write_text(
        json.dumps(cfg, indent=2),
        encoding="utf-8",
    )


def ensure_eval_data(here, env):
    eval_dir = here / "data" / "afhq" / "eval"
    if eval_dir.exists() and any(eval_dir.iterdir()):
        print(f"[OK] evaluation data already exists: {eval_dir}")
        return

    print("[PREP] Building data/afhq/eval once before launching GPU workers.")
    log_path = here / "ddim_logs" / "dataset_prepare.log"
    run_to_log(
        [sys.executable, "dataset.py"],
        log_path,
        env,
        here,
    )

    if not eval_dir.exists() or not any(eval_dir.iterdir()):
        raise RuntimeError("dataset.py finished but data/afhq/eval is still empty.")


def run_combo(
    gpu_physical_id,
    eta,
    steps,
    args,
    here,
    out_root,
    logs_root,
    existing_rows,
    partial_csv,
):
    key = combo_key(eta, steps)
    save_dir = out_root / f"steps_{steps}_eta_{eta}"
    sample_log = logs_root / f"sampling_steps_{steps}_eta_{eta}_gpu{gpu_physical_id}.log"
    fid_log = logs_root / f"fid_steps_{steps}_eta_{eta}_gpu{gpu_physical_id}.log"

    # Each subprocess sees only its assigned physical GPU, remapped locally to cuda:0.
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_physical_id)

    old = existing_rows.get(key)
    images_ready = (
        save_dir.exists()
        and count_pngs(save_dir) == EXPECTED_IMAGES
        and config_matches(
            save_dir, args.ckpt_path, eta, steps, args.batch_size
        )
    )

    if old is not None and images_ready:
        try:
            old_fid = float(old["fid"])
            if old_fid == old_fid:
                print(
                    f"[GPU {gpu_physical_id}] SKIP eta={eta}, steps={steps}: "
                    f"already complete (FID={old_fid:.6f})",
                    flush=True,
                )
                return old
        except Exception:
            pass

    print(
        f"[GPU {gpu_physical_id}] START eta={eta}, steps={steps}",
        flush=True,
    )
    total_start = time.time()

    if not images_ready:
        if save_dir.exists():
            shutil.rmtree(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        sample_cmd = [
            sys.executable,
            "sampling.py",
            "--ckpt_path", str(Path(args.ckpt_path).resolve()),
            "--save_dir", str(save_dir),
            "--sample_method", "ddim",
            "--ddim_steps", str(steps),
            "--eta", str(eta),
            "--batch_size", str(args.batch_size),
            # CUDA_VISIBLE_DEVICES maps the assigned GPU to local cuda:0.
            "--gpu", "0",
        ]
        if args.predictor is not None:
            sample_cmd += ["--predictor", args.predictor]

        sampling_seconds = run_to_log(
            sample_cmd, sample_log, env, here
        )

        sample_count = count_pngs(save_dir)
        if sample_count != EXPECTED_IMAGES:
            raise RuntimeError(
                f"eta={eta}, steps={steps}: expected {EXPECTED_IMAGES} PNGs, "
                f"found {sample_count}"
            )
        save_config(
            save_dir, args.ckpt_path, eta, steps, args.batch_size
        )
    else:
        sampling_seconds = 0.0
        sample_count = EXPECTED_IMAGES
        print(
            f"[GPU {gpu_physical_id}] Reusing verified 500-image sample set "
            f"for eta={eta}, steps={steps}",
            flush=True,
        )

    fid_cmd = [
        sys.executable,
        "fid/measure_fid.py",
        str(here / "data" / "afhq" / "eval"),
        str(save_dir),
    ]
    fid_stdout, fid_seconds = run_capture(
        fid_cmd, fid_log, env, here
    )

    m = re.search(r"FID:\s*([0-9.eE+-]+)", fid_stdout)
    if not m:
        raise RuntimeError(
            f"Could not parse FID for eta={eta}, steps={steps}. See {fid_log}"
        )
    fid = float(m.group(1))
    if fid != fid:
        raise RuntimeError(
            f"FID is NaN for eta={eta}, steps={steps}. See {fid_log}"
        )

    row = {
        "gpu": gpu_physical_id,
        "eta": eta,
        "steps": steps,
        "fid": fid,
        "sample_count": sample_count,
        "sampling_seconds": round(sampling_seconds, 3),
        "fid_seconds": round(fid_seconds, 3),
        "total_seconds": round(time.time() - total_start, 3),
        "save_dir": str(save_dir),
        "sampling_log": str(sample_log),
        "fid_log": str(fid_log),
    }

    existing_rows[key] = row
    write_rows(partial_csv, list(existing_rows.values()))

    print(
        f"[GPU {gpu_physical_id}] DONE eta={eta}, steps={steps}, "
        f"FID={fid:.6f}, images={sample_count}",
        flush=True,
    )
    return row


def worker(gpu_physical_id, etas, args, here, out_root, logs_root):
    partial_csv = here / f"ddim_fid_results_gpu{gpu_physical_id}.csv"
    existing = {
        combo_key(r["eta"], r["steps"]): r
        for r in read_rows(partial_csv)
    }

    rows = []
    for eta in etas:
        for steps in STEPS:
            row = run_combo(
                gpu_physical_id,
                eta,
                steps,
                args,
                here,
                out_root,
                logs_root,
                existing,
                partial_csv,
            )
            rows.append(row)
    return rows


def write_final_outputs(here, rows):
    rows = sorted(rows, key=lambda r: (float(r["eta"]), int(r["steps"])))

    expected = {
        combo_key(eta, steps)
        for eta in ETAS
        for steps in STEPS
    }
    found = {
        combo_key(r["eta"], r["steps"])
        for r in rows
    }

    if found != expected:
        missing = sorted(expected - found)
        extra = sorted(found - expected)
        raise RuntimeError(
            f"Final result set is not exactly the required 20 combinations. "
            f"Missing={missing}, extra={extra}"
        )

    for r in rows:
        if int(r["sample_count"]) != EXPECTED_IMAGES:
            raise RuntimeError(
                f"Invalid sample count in result row: {r}"
            )
        if float(r["fid"]) != float(r["fid"]):
            raise RuntimeError(f"NaN FID in result row: {r}")

    final_csv = here / "ddim_fid_results.csv"
    write_rows(final_csv, rows)

    # Slides ask for a 4 x 5 FID table: eta rows, step columns.
    table_csv = here / "ddim_fid_table.csv"
    by_key = {
        combo_key(r["eta"], r["steps"]): float(r["fid"])
        for r in rows
    }
    with table_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["eta"] + [str(s) for s in STEPS]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for eta in ETAS:
            row = {"eta": eta}
            for steps in STEPS:
                row[str(steps)] = by_key[(eta, steps)]
            w.writerow(row)

    print("\nFINAL 4 x 5 FID TABLE")
    header = "eta   " + " ".join(f"{s:>12}" for s in STEPS)
    print(header)
    for eta in ETAS:
        vals = " ".join(
            f"{by_key[(eta, s)]:12.4f}" for s in STEPS
        )
        print(f"{eta:<5} {vals}")

    print("\n[OK] All slide-required Task 1-2 data verified:")
    print("     - 20 / 20 DDIM combinations")
    print("     - 500 images per combination")
    print("     - 20 finite FID values")
    print(f"     - detailed results: {final_csv}")
    print(f"     - report-ready 4x5 table: {table_csv}")
    print(f"     - per-GPU partial results retained for recovery")


def dry_run(args):
    g0, g1 = args.gpus
    print("DRY RUN ONLY - no sampling or FID will be executed.")
    print(f"GPU {g0}: eta = {DEFAULT_ASSIGNMENT[0]}")
    print(f"GPU {g1}: eta = {DEFAULT_ASSIGNMENT[1]}")
    combos = []
    for logical_gpu, etas in DEFAULT_ASSIGNMENT.items():
        physical_gpu = args.gpus[logical_gpu]
        for eta in etas:
            for steps in STEPS:
                combos.append((physical_gpu, eta, steps))
    print(f"Total combinations: {len(combos)}")
    assert len(combos) == 20
    assert len({(eta, steps) for _, eta, steps in combos}) == 20
    for physical_gpu in args.gpus:
        assigned = [(e, s) for g, e, s in combos if g == physical_gpu]
        print(
            f"GPU {physical_gpu}: {len(assigned)} runs, "
            f"total DDIM steps = {sum(s for _, s in assigned)}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument(
        "--gpus",
        type=int,
        nargs=2,
        default=[0, 1],
        metavar=("GPU0", "GPU1"),
        help="Two physical CUDA device IDs. Default: 0 1",
    )
    ap.add_argument("--out_root", default="ddim_samples")
    ap.add_argument(
        "--predictor",
        choices=["noise", "x0", "mean"],
        default=None,
        help="For old Lab1 checkpoints without predictor metadata, pass --predictor noise.",
    )
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="Only verify the 20-run two-GPU assignment.",
    )
    args = ap.parse_args()

    if args.predictor not in (None, "noise"):
        raise ValueError("Lab2 DDIM requires the noise predictor.")

    if args.gpus[0] == args.gpus[1]:
        raise ValueError("--gpus must name two different physical GPUs.")

    if args.dry_run:
        dry_run(args)
        return

    here = Path(__file__).resolve().parent
    ckpt = Path(args.ckpt_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    fid_ckpt = here / "fid" / "afhq_inception_v3.ckpt"
    if not fid_ckpt.exists():
        raise FileNotFoundError(
            "Missing fid/afhq_inception_v3.ckpt. "
            "Run ../../kaggle_bootstrap.py first."
        )

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        print(
            f"[WARN] Parent CUDA_VISIBLE_DEVICES={visible}. "
            "The runner will set it separately for each child process."
        )

    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = here / out_root
    logs_root = here / "ddim_logs"

    # Important: dataset prep happens once, before the two GPU workers start.
    ensure_eval_data(here, os.environ.copy())

    # Fixed balanced assignment. Each GPU gets two eta rows and all five steps.
    gpu_jobs = [
        (args.gpus[0], DEFAULT_ASSIGNMENT[0]),
        (args.gpus[1], DEFAULT_ASSIGNMENT[1]),
    ]

    print("\nTwo-GPU assignment (experiment settings unchanged):")
    for gpu, etas in gpu_jobs:
        print(
            f"  GPU {gpu}: eta={etas}, "
            f"steps={STEPS} -> {len(etas) * len(STEPS)} runs"
        )

    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = [
            ex.submit(
                worker,
                gpu,
                etas,
                args,
                here,
                out_root,
                logs_root,
            )
            for gpu, etas in gpu_jobs
        ]
        all_rows = []
        for fut in futures:
            all_rows.extend(fut.result())

    write_final_outputs(here, all_rows)


if __name__ == "__main__":
    main()
