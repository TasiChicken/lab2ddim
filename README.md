# Lab2 — DDIM + LoRA (completed working package)

This folder is based on the **2026 NYCU Lab2 starter** and merges the completed TODO implementations from the user's Lab1 (`TasiChicken/lab1ddpm`) into the corresponding Lab2 TODO blocks.  The original `TODO` comments/markers are intentionally preserved.

## What is completed

### Task 1-1 — Swiss Roll DDIM
- Lab1 `SimpleNet` TODO implementation carried over.
- Lab1 DDPM `q_sample`, `p_sample`, `p_sample_loop`, and `compute_loss` carried over.
- Lab2 `ddim_p_sample` implemented using the noise-prediction DDIM update.
- Lab2 `ddim_p_sample_loop` implemented with reduced inference timesteps and Gaussian initialization.
- `ddpm_tutorial.ipynb` is clean and ready to run for the required 5,000 training iterations and DDIM Chamfer evaluation.

### Task 1-2 — Image DDIM
- Lab1 scheduler TODO implementations carried over in the same explicit PyTorch formula style.
- `DDIMScheduler.set_inference_timesteps` implemented.
- `DDIMScheduler.step` implemented for the required noise predictor.
- The checkpoint's original beta schedule is preserved when switching to DDIM.
- `run_all_ddim.py` automates all 20 `(steps, eta)` experiments and writes `ddim_fid_results.csv`.

### Task 2 — LoRA
No LoRA algorithm implementation is required by the assignment.  The marked configuration TODO regions are kept and configured for two immediately runnable experiments:
- Task 2-1: custom `pncut` paper-cut style LoRA, using the included 16-image self-created dataset.
- Task 2-2: DreamBooth + LoRA for the included 8-image `sksbot` toy-robot identity.
- Separate inference notebooks are included for the style and identity experiments.

The two large unchanged official Task-2 trainer files and the ~87 MB official FID Inception checkpoint are fetched by `kaggle_bootstrap.py` from the exact pinned 2026 starter commit.  This avoids bloating the chat-delivered ZIP and does not alter assignment logic.

## Validation performed before packaging
- All bundled Python files pass `py_compile`.
- Swiss Roll DDIM forward/reverse smoke tests pass and produce finite tensors.
- Image DDIM smoke tests pass for all required step counts `{10,20,50,100,1000}` and eta values `{0.0,0.2,0.5,1.0}`.
- All notebooks parse as valid nbformat 4 JSON.
- Included Task-2 datasets were checked for expected image counts and metadata filenames.

## Important
The ZIP cannot contain experiment results that depend on your GPU run:
- your Lab1 noise-predictor checkpoint,
- the 20 FID scores,
- trained LoRA weights,
- generated report images.

Use `KAGGLE_GUIDE.md` for the exact run order.
