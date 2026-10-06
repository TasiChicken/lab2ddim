from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

def extract(input, t: torch.Tensor, x: torch.Tensor):
    if t.ndim == 0:
        t = t.unsqueeze(0)
    shape = x.shape
    t = t.long().to(input.device)
    out = torch.gather(input, 0, t)
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)

class BaseScheduler(nn.Module):
    def __init__(
        self, num_train_timesteps: int, beta_1: float, beta_T: float, mode="linear",
        betas: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_timesteps = num_train_timesteps
        self.timesteps = torch.from_numpy(
            np.arange(0, self.num_train_timesteps)[::-1].copy().astype(np.int64)
        )

        if betas is not None:
            # A pretrained checkpoint already contains its exact noise schedule.
            betas = betas.detach().clone()
            if betas.ndim != 1 or betas.numel() != num_train_timesteps:
                raise ValueError("Saved betas must have one entry per training timestep.")
            if not torch.isfinite(betas).all() or not ((betas > 0) & (betas < 1)).all():
                raise ValueError("Saved betas must be finite and strictly between 0 and 1.")
        elif mode == "linear":
            betas = torch.linspace(beta_1, beta_T, steps=num_train_timesteps)
        elif mode == "quad":
            betas = (
                torch.linspace(beta_1**0.5, beta_T**0.5, num_train_timesteps) ** 2
            )
        elif mode == "cosine":
            ######## TODO ########
            # Implement the cosine beta schedule (Nichol & Dhariwal, 2021).
            # Hint:
            # 1. Define alphā_t = f(t/T) where f is a cosine schedule:
            #       alphā_t = cos^2( ( (t/T + s) / (1+s) ) * (π/2) )
            #    with s = 0.008 (a small constant for stability).
            # 2. Convert alphā_t into betas using:
            #       beta_t = 1 - alphā_t / alphā_{t-1}
            # 3. Return betas as a tensor of shape [num_train_timesteps].
            s = 0.008

            steps = torch.arange(
                num_train_timesteps + 1,
                dtype=torch.float32
            )

            f = torch.cos(
                ((steps / num_train_timesteps + s) / (1 + s))
                * torch.pi / 2
            ) ** 2

            alpha_bar = f / f[0]

            betas = 1 - alpha_bar[1:] / alpha_bar[:-1]

            betas = torch.clamp(
                betas,
                min=0.0,
                max=0.999
            )

        else:
            raise NotImplementedError(f"{mode} is not implemented.")

        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def uniform_sample_t(
        self, batch_size, device: Optional[torch.device] = None
    ) -> torch.IntTensor:
        """
        Uniformly sample timesteps.
        """
        ts = np.random.choice(np.arange(self.num_train_timesteps), batch_size)
        ts = torch.from_numpy(ts)
        if device is not None:
            ts = ts.to(device)
        return ts

class DDPMScheduler(BaseScheduler):
    def __init__(
        self,
        num_train_timesteps: int,
        beta_1: float,
        beta_T: float,
        mode="linear",
        sigma_type="small",
    ):
        super().__init__(num_train_timesteps, beta_1, beta_T, mode)

        self.schedule_mode = mode      

        # sigmas correspond to $\sigma_t$ in the DDPM paper.
        self.sigma_type = sigma_type
        if sigma_type == "small":
            # when $\sigma_t^2 = \tilde{\beta}_t$.
            alphas_cumprod_t_prev = torch.cat(
                [torch.tensor([1.0]), self.alphas_cumprod[:-1]]
            )
            sigmas = (
                (1 - alphas_cumprod_t_prev) / (1 - self.alphas_cumprod) * self.betas
            ) ** 0.5
        elif sigma_type == "large":
            # when $\sigma_t^2 = \beta_t$.
            sigmas = self.betas ** 0.5

        self.register_buffer("sigmas", sigmas)



    def step(self, x_t: torch.Tensor, t: int, net_out: torch.Tensor, predictor: str):
        if predictor == "noise": #### TODO
            return self.step_predict_noise(x_t, t, net_out)
        elif predictor == "x0": #### TODO
            return self.step_predict_x0(x_t, t, net_out)
        elif predictor == "mean": #### TODO
            return self.step_predict_mean(x_t, t, net_out)
        else:
            raise ValueError(f"Unknown predictor: {predictor}")


    def step_predict_noise(self, x_t: torch.Tensor, t: int, eps_theta: torch.Tensor):
        """
        Noise prediction version (the standard DDPM formulation).

        Input:
            x_t: noisy image at timestep t
            t: current timestep
            eps_theta: predicted noise ε̂_θ(x_t, t)
        Output:
            sample_prev: denoised image sample at timestep t-1
        """
        ######## TODO ########
        # 1. Extract beta_t, alpha_t, and alpha_bar_t from the scheduler.
        # 2. Compute the predicted mean μ_θ(x_t, t) = 1/√α_t * (x_t - (β_t/√(1-ᾱ_t)) * ε̂_θ).
        # 3. Compute the posterior variance \tilde{β}_t = ((1-ᾱ_{t-1})/(1-ᾱ_t)) * β_t.
        # 4. Add Gaussian noise scaled by √(\tilde{β}_t) unless t == 0.
        # 5. Return the final sample at t-1.
        beta_t = extract(
            self.betas, t, x_t
        )

        alpha_t = extract(
            self.alphas, t, x_t
        )

        alpha_bar_t = extract(
            self.alphas_cumprod, t, x_t
        )

        alphas_cumprod_prev = torch.cat([
            torch.ones(
                1,
                device=self.alphas_cumprod.device,
                dtype=self.alphas_cumprod.dtype
            ),
            self.alphas_cumprod[:-1]
        ])

        alpha_bar_prev = extract(
            alphas_cumprod_prev,
            t,
            x_t
        )

        x0_pred = (
            x_t
            - torch.sqrt(1 - alpha_bar_t)
            * eps_theta
        ) / torch.sqrt(alpha_bar_t)

        x0_pred = x0_pred.clamp(-1, 1)

        coef1 = (
            torch.sqrt(alpha_bar_prev)
            * beta_t
            / (1 - alpha_bar_t)
        )

        coef2 = (
            torch.sqrt(alpha_t)
            * (1 - alpha_bar_prev)
            / (1 - alpha_bar_t)
        )

        posterior_mean = (
            coef1 * x0_pred
            + coef2 * x_t
        )

        if torch.all(t == 0):
            return posterior_mean

        posterior_var = (
            (1 - alpha_bar_prev)
            / (1 - alpha_bar_t)
            * beta_t
        )

        noise = torch.randn_like(x_t)

        sample_prev = (
            posterior_mean
            + torch.sqrt(posterior_var) * noise
        )
        #######################
        return sample_prev


    def step_predict_x0(self, x_t: torch.Tensor, t: int, x0_pred: torch.Tensor):
        """
        x0 prediction version (alternative DDPM objective).

        Input:
            x_t: noisy image at timestep t
            t: current timestep
            x0_pred: predicted clean image x̂₀(x_t, t)
        Output:
            sample_prev: denoised image sample at timestep t-1
        """
        ######## TODO ########
        beta_t = extract(
            self.betas, t, x_t
        )

        alpha_t = extract(
            self.alphas, t, x_t
        )

        alpha_bar_t = extract(
            self.alphas_cumprod, t, x_t
        )

        alphas_cumprod_prev = torch.cat([
            torch.ones(
                1,
                device=self.alphas_cumprod.device,
                dtype=self.alphas_cumprod.dtype
            ),
            self.alphas_cumprod[:-1]
        ])

        alpha_bar_prev = extract(
            alphas_cumprod_prev,
            t,
            x_t
        )

        x0_pred = x0_pred.clamp(-1, 1)

        coef1 = (
            torch.sqrt(alpha_bar_prev)
            * beta_t
            / (1 - alpha_bar_t)
        )

        coef2 = (
            torch.sqrt(alpha_t)
            * (1 - alpha_bar_prev)
            / (1 - alpha_bar_t)
        )

        posterior_mean = (
            coef1 * x0_pred
            + coef2 * x_t
        )

        if torch.all(t == 0):
            return posterior_mean

        posterior_var = (
            (1 - alpha_bar_prev)
            / (1 - alpha_bar_t)
            * beta_t
        )

        noise = torch.randn_like(x_t)

        sample_prev = (
            posterior_mean
            + torch.sqrt(posterior_var) * noise
        )

        #######################
        return sample_prev


    def step_predict_mean(self, x_t: torch.Tensor, t: int, mean_theta: torch.Tensor):
        """
        Mean prediction version (directly outputting the posterior mean).

        Input:
            x_t: noisy image at timestep t
            t: current timestep
            mean_theta: network-predicted posterior mean μ̂_θ(x_t, t)
        Output:
            sample_prev: denoised image sample at timestep t-1
        """
        ######## TODO ########
        if torch.all(t == 0):
            return mean_theta

        sigma_t = extract(
            self.sigmas,
            t,
            x_t
        )

        noise = torch.randn_like(x_t)

        sample_prev = (
            mean_theta
            + sigma_t * noise
        )
        #######################
        return sample_prev



    # https://nn.labml.ai/diffusion/ddpm/utils.html
    def _get_teeth(self, consts: torch.Tensor, t: torch.Tensor): # get t th const 
        const = consts.gather(-1, t)
        return const.reshape(-1, 1, 1, 1)

    def add_noise(
        self,
        x_0: torch.Tensor,
        t: torch.IntTensor,
        eps: Optional[torch.Tensor] = None,
    ):
        """
        A forward pass of a Markov chain, i.e., q(x_t | x_0).

        Input:
            x_0 (`torch.Tensor [B,C,H,W]`): samples from a real data distribution q(x_0).
            t: (`torch.IntTensor [B]`)
            eps: (`torch.Tensor [B,C,H,W]`, optional): if None, randomly sample Gaussian noise in the function.
        Output:
            x_t: (`torch.Tensor [B,C,H,W]`): noisy samples at timestep t.
            eps: (`torch.Tensor [B,C,H,W]`): injected noise.
        """

        if eps is None:
            eps       = torch.randn_like(x_0)

        ######## TODO ########
        # DO NOT change the code outside this part.
        # Assignment 1. Implement the DDPM forward step.
        alpha_bar_t = extract(
            self.alphas_cumprod,
            t,
            x_0
        )

        x_t = (
            torch.sqrt(alpha_bar_t) * x_0
            + torch.sqrt(1 - alpha_bar_t) * eps
        )
        #######################

        return x_t, eps

class DDIMScheduler(BaseScheduler):
    def __init__(
        self,
        num_train_timesteps: int,
        beta_1: float,
        beta_T: float,
        mode: str = "linear",
        num_inference_timesteps: int = 50,
        eta: float = 0.0,
        trained_scheduler: Optional[BaseScheduler] = None,
    ):
        if trained_scheduler is not None and num_train_timesteps != trained_scheduler.num_train_timesteps:
            raise ValueError("DDIM must use the checkpoint's number of training timesteps.")
        super().__init__(
            num_train_timesteps, beta_1, beta_T, mode,
            betas=None if trained_scheduler is None else trained_scheduler.betas,
        )
        if trained_scheduler is not None:
            # Copy before the timestep TODO, which may cache inference coefficients.
            self.alphas = trained_scheduler.alphas.detach().clone()
            self.alphas_cumprod = trained_scheduler.alphas_cumprod.detach().clone()
            self.schedule_mode = getattr(trained_scheduler, "schedule_mode", None)
        else:
            self.schedule_mode = mode
        self.eta = float(eta)
        self.set_inference_timesteps(num_inference_timesteps)

    def set_inference_timesteps(self, num_inference_timesteps: int):
        """
        Define the inference schedule (a subset of training timesteps, descending order).
        Inputs:
            num_inference_timesteps (int): number of inference steps (e.g., 50).
        """
        ######## TODO ########
        # Hint:
        #   - Define the DDIM inference schedule based on the given num_inference_timesteps.
        #   - The schedule should be a subset of training timesteps, ordered in descending fashion.
        #   - Store the result in `self.timesteps` (as a torch tensor) 
        #   - Store the step ratio in `self._ddim_step_ratio` for later use when computing previous t.
        #   - Compute a `step_ratio` that maps inference steps to training steps.
        # DO NOT change the code outside this part.
        if not 1 <= num_inference_timesteps <= self.num_train_timesteps:
            raise ValueError(
                "num_inference_timesteps must be between 1 and num_train_timesteps"
            )

        self.num_inference_timesteps = int(num_inference_timesteps)
        self._ddim_step_ratio = (
            self.num_train_timesteps // self.num_inference_timesteps
        )

        timesteps = (
            np.arange(0, self.num_inference_timesteps)
            * self._ddim_step_ratio
        ).round()[::-1].copy().astype(np.int64)

        self.timesteps = torch.from_numpy(timesteps)
        #######################

    def _get_teeth(self, consts: torch.Tensor, t: torch.Tensor):
        const = consts.gather(-1, t)
        return const.reshape(-1, 1, 1, 1)

    @torch.no_grad()
    def step(self, x_t: torch.Tensor, t: int, eps_theta: torch.Tensor, predictor: str):
        """
        One step DDIM update: x_t -> x_{t_prev} with deterministic/stochastic control via eta.

        Input:
            x_t: [B,C,H,W]
            t: current absolute timestep index
            eps_theta: predicted noise
            predictor: predictor type
        Output:
            sample_prev: x at previous inference timestep
        """
        ######## TODO ########
        # DO NOT change the code outside this part.
        assert predictor == "noise", "In assignment 2, we only implement DDIM with noise predictor."

        t_int = int(t.item()) if torch.is_tensor(t) else int(t)
        prev_t = t_int - self._ddim_step_ratio

        alpha_prod_t = self.alphas_cumprod[t_int].to(
            device=x_t.device, dtype=x_t.dtype
        )
        if prev_t >= 0:
            alpha_prod_t_prev = self.alphas_cumprod[prev_t].to(
                device=x_t.device, dtype=x_t.dtype
            )
        else:
            alpha_prod_t_prev = torch.ones(
                (), device=x_t.device, dtype=x_t.dtype
            )

        beta_prod_t = 1 - alpha_prod_t

        # Predict x_0 from the model's noise prediction.
        pred_x0 = (
            x_t - torch.sqrt(beta_prod_t) * eps_theta
        ) / torch.sqrt(alpha_prod_t)

        # sigma_t^2 from the DDIM update.
        variance = (
            (1 - alpha_prod_t_prev)
            / (1 - alpha_prod_t)
            * (1 - alpha_prod_t / alpha_prod_t_prev)
        )
        variance = torch.clamp(variance, min=0.0)
        std_dev_t = self.eta * torch.sqrt(variance)

        # Deterministic direction plus optional stochastic noise.
        pred_direction = torch.sqrt(
            torch.clamp(
                1 - alpha_prod_t_prev - std_dev_t**2,
                min=0.0,
            )
        ) * eps_theta

        sample_prev = (
            torch.sqrt(alpha_prod_t_prev) * pred_x0
            + pred_direction
        )

        if self.eta > 0:
            sample_prev = (
                sample_prev
                + std_dev_t * torch.randn_like(x_t)
            )
        #######################
        return sample_prev
