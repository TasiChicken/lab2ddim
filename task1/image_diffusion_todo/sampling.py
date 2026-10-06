import argparse
import math
import numpy as np
import torch
from pathlib import Path
from dataset import tensor_to_pil_image
from model import DiffusionModule
from scheduler import DDIMScheduler


def configure_sampler(model, args):
    """Reuse the checkpoint's training schedule; CLI hints only check consistency."""
    if args.sample_method not in ("ddpm", "ddim"):
        raise ValueError(f"Invalid sample method: {args.sample_method}")

    saved_predictor = model.predictor
    if saved_predictor is not None and args.predictor is not None and args.predictor != saved_predictor:
        raise ValueError(
            f"--predictor {args.predictor} conflicts with checkpoint predictor "
            f"{saved_predictor}; omit --predictor to use the saved value."
        )
    predictor = saved_predictor if args.predictor is None else args.predictor
    if predictor is None:
        raise ValueError("This checkpoint has no predictor metadata; pass --predictor matching its training objective.")
    if predictor not in ("noise", "x0", "mean"):
        raise ValueError(f"Unsupported checkpoint predictor: {predictor}")
    if args.sample_method == "ddim" and predictor != "noise":
        raise ValueError("Lab2 DDIM requires a noise-prediction checkpoint; select one or use --sample_method ddpm.")

    saved_scheduler = model.var_scheduler
    saved_mode = getattr(saved_scheduler, "schedule_mode", None)
    if args.mode is not None:
        if saved_mode is None:
            raise ValueError("This checkpoint has no schedule-mode metadata; omit --mode to preserve its saved schedule.")
        if args.mode != saved_mode:
            raise ValueError(f"--mode {args.mode} conflicts with checkpoint schedule {saved_mode}; omit --mode.")
    for name, index in (("beta_1", 0), ("beta_T", -1)):
        value = getattr(args, name)
        if value is not None:
            if saved_mode not in ("linear", "quad"):
                raise ValueError(f"--{name} can only validate linear/quad endpoints; omit beta flags to preserve this checkpoint's schedule.")
            saved_value = saved_scheduler.betas[index].item()
            if not math.isclose(value, saved_value, rel_tol=1e-6, abs_tol=1e-12):
                raise ValueError(f"--{name}={value} conflicts with saved endpoint {saved_value}; omit --{name}.")

    if args.sample_method == "ddim":
        model.var_scheduler = DDIMScheduler(
            saved_scheduler.num_train_timesteps,
            beta_1=saved_scheduler.betas[0].item(),
            beta_T=saved_scheduler.betas[-1].item(),
            mode=saved_mode,
            num_inference_timesteps=args.ddim_steps,
            eta=args.eta,
            trained_scheduler=saved_scheduler,
        )
    # DDPM retains its saved scheduler, including its original variance settings.
    model.predictor = predictor


def main(args):
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    device = f"cuda:{args.gpu}"
    model = DiffusionModule(None, None)
    model.load(args.ckpt_path)
    configure_sampler(model, args)
    model.eval().to(device)

    total_num_samples = 500
    num_batches = int(np.ceil(total_num_samples / args.batch_size))

    for i in range(num_batches):
        sidx = i * args.batch_size
        eidx = min(sidx + args.batch_size, total_num_samples)
        B = eidx - sidx

        if args.use_cfg:
            assert getattr(model.network, "use_cfg", False), "This checkpoint wasn't trained with CFG."

            samples = model.sample(
                B,
                class_label=torch.randint(0, 3, (B,), device=device),
                guidance_scale=args.cfg_scale,
            )
        else:
            samples = model.sample(B)

        for j, img in zip(range(sidx, eidx), tensor_to_pil_image(samples)):
            img.save(save_dir / f"{j}.png")
            print(f"Saved the {j}-th image.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    parser.add_argument("--predictor", type=str, default=None,
                        choices=["noise", "x0", "mean"],
                        help="use saved metadata; required for older checkpoints without it")
    parser.add_argument("--mode", type=str, default=None,
                        choices=["linear", "cosine", "quad"],
                        help="optional consistency check against the saved schedule")
    parser.add_argument("--beta_1", type=float, default=None,
                        help="optional consistency check for a linear/quad checkpoint's first beta")
    parser.add_argument("--beta_T", type=float, default=None,
                        help="optional consistency check for a linear/quad checkpoint's last beta")


    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--sample_method", type=str, default="ddpm", choices=["ddpm", "ddim"])
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--cfg_scale", type=float, default=7.5)

    args = parser.parse_args()
    main(args)
