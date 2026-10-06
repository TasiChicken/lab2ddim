# Report checklist

## Task 1-1
- Explain every filled TODO in `2d_plot_diffusion_todo/network.py` and `ddpm.py`.
- Include the DDIM Swiss Roll evaluation plot.
- Include a screenshot of DDIM Chamfer Distance; assignment target is < 40.

## Task 1-2
- Explain the inherited Lab1 scheduler TODOs as needed and the two new DDIM TODOs: `set_inference_timesteps` and `step`.
- Include a 4 x 5 FID table for eta = 0.0/0.2/0.5/1.0 and steps = 10/20/50/100/1000.
- Discuss how fewer/more DDIM steps and eta change sampling stochasticity and FID/image quality.

## Task 2-1 Style LoRA
- Dataset source: self-created procedural `pncut` paper-cut style dataset included in `task2/sample_data/artistic-custom/train`.
- Show several training images and their captions from `metadata.jsonl`.
- Show generated images and the exact prompts.
- Upload the final LoRA weight folder and put the accessible cloud URL in the required `{ID}_task2-1_lora_weight.txt`.

## Task 2-2 DreamBooth + LoRA
- Dataset source: self-created procedural toy-robot identity (`sksbot`) included in `task2/sample_data/dreambooth-robot`.
- Show several identity training images.
- Show generated images with exact `sksbot` prompts.
- Upload the final LoRA weight folder and put the accessible cloud URL in `{ID}_task2-2_lora_weight.txt`.
