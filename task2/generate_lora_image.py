"""Small inference helper for saving Task-2 report images without editing trainer code."""
import argparse
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from utils import seed_everything


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lora_path", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model_name", default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the intended Kaggle workflow.")

    seed_everything(args.seed)
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
    ).to("cuda:0")
    pipe.load_lora_weights(args.lora_path)

    image = pipe(
        args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
    ).images[0]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    print(f"Saved: {out}")
    print(f"Prompt: {args.prompt}")


if __name__ == "__main__":
    main()
