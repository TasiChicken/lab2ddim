# Kaggle run guide — Lab2 DDIM + LoRA

This package merges your completed Lab1 TODO implementations into the 2026 Lab2 starter-style files, completes the Lab2 DDIM TODOs, keeps the original TODO markers, and includes runnable Task-2 configurations/data.

## 0. Kaggle setup

Create a Kaggle Notebook with **GPU enabled** (T4 is enough) and **Internet ON**. Upload this ZIP as a Kaggle Dataset/file. Also upload your Lab1 **noise-predictor** checkpoint separately; use the quadratic + noise checkpoint if that is the checkpoint from your best controlled Lab1 run.

Example setup cells (change the input dataset slug if Kaggle gives yours a different path):

```python
!unzip -q /kaggle/input/lab2-code/Lab2-DDIM-LoRA-complete.zip -d /kaggle/working
%cd /kaggle/working/Lab2-DDIM-LoRA-complete
!pip install -q -r requirements.txt
!python kaggle_bootstrap.py
!python self_check.py
```

`kaggle_bootstrap.py` downloads the official AFHQ FID Inception checkpoint and caches the two unchanged official Task-2 trainer programs from the exact 2026 starter commit. These support files are deliberately not duplicated in the chat ZIP.

## 1. Task 1-1 — Swiss Roll DDIM

Run the notebook from its own directory:

```python
%cd /kaggle/working/Lab2-DDIM-LoRA-complete/task1/2d_plot_diffusion_todo
!jupyter nbconvert --to notebook --execute ddpm_tutorial.ipynb \
  --ExecutePreprocessor.timeout=-1 \
  --output ddpm_tutorial_output.ipynb
```

You can then open `ddpm_tutorial_output.ipynb` in the Kaggle file pane. It trains the required fresh Swiss Roll model for 5,000 iterations, evaluates DDPM, and evaluates DDIM with 50 inference steps and eta=0.0. Keep the DDIM plot and the printed Chamfer Distance for the report. If a run is above the assignment threshold, re-run only the final DDIM evaluation cell using the already-trained model.

## 2. Task 1-2 — Image DDIM

Your Lab1 image checkpoint is not contained in this archive, because it belongs to your previous training run. Upload it to Kaggle as a Dataset/file. Example path:

```text
/kaggle/input/lab1-checkpoint/last.ckpt
```

Then run:

```python
%cd /kaggle/working/Lab2-DDIM-LoRA-complete/task1/image_diffusion_todo
!python dataset.py
```

A single test run:

```python
!python sampling.py \
  --ckpt_path /kaggle/input/lab1-checkpoint/last.ckpt \
  --save_dir ddim_samples/steps_50_eta_0.5 \
  --sample_method ddim \
  --ddim_steps 50 \
  --eta 0.5 \
  --predictor noise

!python fid/measure_fid.py data/afhq/eval/ ddim_samples/steps_50_eta_0.5
```

After the single run works, execute all required 20 settings automatically:

```python
!python run_all_ddim.py \
  --ckpt_path /kaggle/input/lab1-checkpoint/last.ckpt \
  --predictor noise
```

The helper evaluates steps `{10,20,50,100,1000}` x eta `{0.0,0.2,0.5,1.0}`, generates 500 images per setting, evaluates FID, and continuously writes `ddim_fid_results.csv`. If your checkpoint already records `predictor=noise`, `--predictor noise` is optional.

## 3. Task 2 — Hugging Face login

Do not paste your token into a public notebook. In Kaggle, add a Secret named `HF_TOKEN`, then run:

```python
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login
login(token=UserSecretsClient().get_secret("HF_TOKEN"))
```

The assignment uses `CompVis/stable-diffusion-v1-4`; confirm the model can be downloaded before starting training.

### Task 2-1 — Style LoRA

The ZIP includes a self-created 16-image `pncut` geometric paper-cut style dataset and captions.

```python
%cd /kaggle/working/Lab2-DDIM-LoRA-complete/task2
!sh scripts/train_lora_custom.sh
```

Final LoRA weights are written under `runs/artistic_custom/`. The trainer also writes validation images under `runs/artistic_custom/validation/`. The configured validation prompt is `a lighthouse by the sea in pncut style`.

To save another report image explicitly:

```python
!python generate_lora_image.py \
  --lora_path ./runs/artistic_custom \
  --prompt "a castle above the clouds in pncut style" \
  --output ./report_outputs/style_castle.png
```

### Task 2-2 — DreamBooth + LoRA

The ZIP includes eight self-created views of one identifiable toy robot. Its unique identifier is `sksbot`.

```python
!sh scripts/train_dreambooth_lora.sh
```

Final LoRA weights are written under `runs/dreambooth_robot/`.

Save a generated example for the report:

```python
!python generate_lora_image.py \
  --lora_path ./runs/dreambooth_robot \
  --prompt "a photo of sksbot robot in a city park" \
  --output ./report_outputs/dreambooth_park.png
```

The two inference notebooks are also ready if you prefer notebook output: `lora_inference.ipynb` for style and `lora_inference_dreambooth.ipynb` for identity.

## 4. After the experiments

Use `REPORT_CHECKLIST.md`. Upload `runs/artistic_custom/` and `runs/dreambooth_robot/` to cloud storage, make sure the TAs can access them, then rename the two provided link templates using your student ID and paste the cloud links. The GPU-generated FID values, images, and weights cannot exist before you run the experiments; everything needed to produce them is included/configured here.
