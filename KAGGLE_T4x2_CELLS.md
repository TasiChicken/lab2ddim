# Kaggle T4 x2 run cells for TasiChicken/lab2ddim

This guide keeps the assignment settings unchanged. Only Task 1-2 independent
(eta, steps) combinations are distributed across the two T4 GPUs.

## 0. Clone + install

```python
!git clone https://github.com/TasiChicken/lab2ddim.git

%cd /kaggle/working/lab2ddim

# Keep this line: it is required in your current Kaggle environment.
!pip install -U ipython

!pip install -q -r requirements.txt

!python kaggle_bootstrap.py
!python self_check.py
```

## 1. Confirm both GPUs

```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))

assert torch.cuda.device_count() == 2, "Select Kaggle T4 x2 before running Task 1-2."
```

## 2. Task 1-1 Swiss Roll

```python
%cd /kaggle/working/lab2ddim/task1/2d_plot_diffusion_todo

!jupyter nbconvert \
  --to notebook \
  --execute ddpm_tutorial.ipynb \
  --ExecutePreprocessor.timeout=-1 \
  --output ddpm_tutorial_output.ipynb
```

Keep the DDIM evaluation figure and the printed DDIM Chamfer Distance (< 40)
for the report.

## 3. Task 1-2 DDIM — prepare evaluation data

```python
%cd /kaggle/working/lab2ddim/task1/image_diffusion_todo

!python dataset.py
```

Checkpoint:

```python
CKPT = "/kaggle/input/datasets/puppypoo/quadratic-noise/step_35000.ckpt"
print(CKPT)
```

First verify the exact two-GPU assignment without running inference:

```python
!python run_all_ddim.py \
  --ckpt_path "{CKPT}" \
  --predictor noise \
  --gpus 0 1 \
  --dry_run
```

Expected:

```text
GPU 0: eta = [0.0, 0.5]
GPU 1: eta = [0.2, 1.0]
Total combinations: 20
GPU 0: 10 runs, total DDIM steps = 2360
GPU 1: 10 runs, total DDIM steps = 2360
```

Run all 20 experiments:

```python
!python run_all_ddim.py \
  --ckpt_path "{CKPT}" \
  --predictor noise \
  --gpus 0 1
```

The runner verifies:
- exactly 20 required combinations
- 500 generated PNGs for every combination
- one finite FID for every combination
- checkpoint beta schedule is preserved by the existing sampling.py
- each child process uses only its assigned T4
- GPU 0 and GPU 1 each execute 10 balanced experiments

Outputs:
- `ddim_fid_results_gpu0.csv`
- `ddim_fid_results_gpu1.csv`
- `ddim_fid_results.csv`
- `ddim_fid_table.csv`
- `ddim_logs/`
- `ddim_samples/steps_<steps>_eta_<eta>/`

Final verification/display:

```python
import pandas as pd
from pathlib import Path

root = Path("/kaggle/working/lab2ddim/task1/image_diffusion_todo")

results = pd.read_csv(root / "ddim_fid_results.csv")
table = pd.read_csv(root / "ddim_fid_table.csv")

assert len(results) == 20
assert set(results["eta"].astype(float)) == {0.0, 0.2, 0.5, 1.0}
assert set(results["steps"].astype(int)) == {10, 20, 50, 100, 1000}
assert (results["sample_count"].astype(int) == 500).all()
assert results["fid"].notna().all()

for _, row in results.iterrows():
    d = Path(row["save_dir"])
    assert len(list(d.glob("*.png"))) == 500, d

print("All 20 DDIM experiments verified.")
display(table)
display(results[["gpu", "eta", "steps", "fid", "sample_count"]])
```

## 4. Task 2 Hugging Face login

```python
%cd /kaggle/working/lab2ddim/task2

from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

login(token=UserSecretsClient().get_secret("HF_TOKEN"))
```

Task 2 is intentionally kept on one GPU so the existing training configuration
has the same single-GPU semantics; using two distributed processes would change
the effective global batch / number of updates for the epoch-based style run.

### Task 2-1 style LoRA

```python
%cd /kaggle/working/lab2ddim/task2
!CUDA_VISIBLE_DEVICES=0 sh scripts/train_lora_custom.sh
```

Generate report images with exact prompts:

```python
!CUDA_VISIBLE_DEVICES=0 python generate_lora_image.py \
  --lora_path ./runs/artistic_custom \
  --prompt "a lighthouse by the sea in pncut style" \
  --output ./report_outputs/style_lighthouse.png

!CUDA_VISIBLE_DEVICES=0 python generate_lora_image.py \
  --lora_path ./runs/artistic_custom \
  --prompt "a castle above the clouds in pncut style" \
  --output ./report_outputs/style_castle.png

!CUDA_VISIBLE_DEVICES=0 python generate_lora_image.py \
  --lora_path ./runs/artistic_custom \
  --prompt "a small train crossing a mountain bridge in pncut style" \
  --output ./report_outputs/style_train.png
```

### Task 2-2 DreamBooth + LoRA

```python
!CUDA_VISIBLE_DEVICES=0 sh scripts/train_dreambooth_lora.sh
```

Generate report images with exact prompts:

```python
!CUDA_VISIBLE_DEVICES=0 python generate_lora_image.py \
  --lora_path ./runs/dreambooth_robot \
  --prompt "a photo of sksbot robot on a wooden desk" \
  --output ./report_outputs/dreambooth_desk.png

!CUDA_VISIBLE_DEVICES=0 python generate_lora_image.py \
  --lora_path ./runs/dreambooth_robot \
  --prompt "a photo of sksbot robot in a city park" \
  --output ./report_outputs/dreambooth_park.png

!CUDA_VISIBLE_DEVICES=0 python generate_lora_image.py \
  --lora_path ./runs/dreambooth_robot \
  --prompt "a photo of sksbot robot beside a red backpack" \
  --output ./report_outputs/dreambooth_backpack.png
```

## 5. Display Task 2 training data for the report

Style images + captions:

```python
%cd /kaggle/working/lab2ddim/task2

import json
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

style_dir = Path("sample_data/artistic-custom/train")
rows = [json.loads(x) for x in (style_dir / "metadata.jsonl").read_text().splitlines() if x.strip()]

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for ax, row in zip(axes.flat, rows[:8]):
    ax.imshow(Image.open(style_dir / row["file_name"]))
    ax.set_title(row["text"], fontsize=8)
    ax.axis("off")
plt.tight_layout()
plt.show()
```

DreamBooth identity images:

```python
robot_dir = Path("sample_data/dreambooth-robot")
robot_files = sorted(robot_dir.glob("*.png"))

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for ax, p in zip(axes.flat, robot_files):
    ax.imshow(Image.open(p))
    ax.set_title(p.name)
    ax.axis("off")
plt.tight_layout()
plt.show()
```

Display generated report images:

```python
generated = sorted(Path("report_outputs").glob("*.png"))
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for ax, p in zip(axes.flat, generated[:6]):
    ax.imshow(Image.open(p))
    ax.set_title(p.stem)
    ax.axis("off")
plt.tight_layout()
plt.show()
```

## 6. Package LoRA weights for cloud upload

```python
%cd /kaggle/working/lab2ddim/task2

!zip -qr artistic_custom_lora.zip runs/artistic_custom
!zip -qr dreambooth_robot_lora.zip runs/dreambooth_robot

!ls -lh artistic_custom_lora.zip dreambooth_robot_lora.zip
```

Upload both ZIPs to cloud storage, make them accessible to the TAs, and put their
links into the two required `{ID}_task2-1_lora_weight.txt` and
`{ID}_task2-2_lora_weight.txt` files.
