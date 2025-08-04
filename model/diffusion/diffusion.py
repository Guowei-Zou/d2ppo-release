"""
Gaussian diffusion with DDPM and optionally DDIM sampling.
Enhanced with Dispersive Loss for representation regularization.

References:
Diffuser: https://github.com/jannerm/diffuser
Diffusion Policy: https://github.com/columbia-ai-robotics/diffusion_policy/blob/main/diffusion_policy/policy/diffusion_unet_lowdim_policy.py
Annotated DDIM/DDPM: https://nn.labml.ai/diffusion/stable_diffusion/sampler/ddpm.html
Dispersive Loss: "Diffuse and Disperse: Image Generation with Representation Regularization"

"""

import logging
import torch
from torch import nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

from model.diffusion.sampling import (
    extract,
    cosine_beta_schedule,
    make_timesteps,
)

from collections import namedtuple

Sample = namedtuple("Sample", "trajectories chains")


class DiffusionModel(nn.Module):

    def __init__(
        self,
        network,
        horizon_steps,
        obs_dim,
        action_dim,
        network_path=None,
        device="cuda:0",
        # Various clipping
        denoised_clip_value=1.0,
        randn_clip_value=10,
        final_action_clip_value=None,
        eps_clip_value=None,  # DDIM only
        # DDPM parameters
        denoising_steps=100,
        predict_epsilon=True,
        # DDIM sampling
        use_ddim=False,
        ddim_discretize="uniform",
        ddim_steps=None,
        # Dispersive Loss parameters
        use_dispersive_loss=False,
        dispersive_loss_weight=0.5,
        dispersive_loss_temperature=0.5,
        dispersive_loss_type="infonce_l2",  # "infonce_l2", "infonce_cosine", "hinge", "covariance"
        dispersive_loss_layer="early",  # "early", "mid", "late", "all"
        **kwargs,
    ):
        super().__init__()
        self.device = device
        self.horizon_steps = horizon_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.denoising_steps = int(denoising_steps)
        self.predict_epsilon = predict_epsilon
        self.use_ddim = use_ddim
        self.ddim_steps = ddim_steps

        # Clip noise value at each denoising step
        self.denoised_clip_value = denoised_clip_value

        # Whether to clamp the final sampled action between [-1, 1]
        self.final_action_clip_value = final_action_clip_value

        # For each denoising step, we clip sampled randn (from standard deviation) such that the sampled action is not too far away from mean
        self.randn_clip_value = randn_clip_value

        # Clip epsilon for numerical stability
        self.eps_clip_value = eps_clip_value

        # Dispersive Loss parameters
        self.use_dispersive_loss = use_dispersive_loss
        self.dispersive_loss_weight = dispersive_loss_weight
        self.dispersive_loss_temperature = dispersive_loss_temperature
        self.dispersive_loss_type = dispersive_loss_type
        self.dispersive_loss_layer = dispersive_loss_layer

        # Set up models
        self.network = network.to(device)
        if network_path is not None:
            checkpoint = torch.load(
                network_path, map_location=device, weights_only=True
            )
            if "ema" in checkpoint:
                self.load_state_dict(checkpoint["ema"], strict=False)
                logging.info("Loaded SL-trained policy from %s", network_path)
            else:
                self.load_state_dict(checkpoint["model"], strict=False)
                logging.info("Loaded RL-trained policy from %s", network_path)
        logging.info(
            f"Number of network parameters: {sum(p.numel() for p in self.parameters())}"
        )

        """
        DDPM parameters

        """
        """
        βₜ
        """
        self.betas = cosine_beta_schedule(denoising_steps).to(device)
        """
        αₜ = 1 - βₜ
        """
        self.alphas = 1.0 - self.betas
        """
        α̅ₜ= ∏ᵗₛ₌₁ αₛ 
        """
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        """
        α̅ₜ₋₁
        """
        self.alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), self.alphas_cumprod[:-1]]
        )
        """
        √ α̅ₜ
        """
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        """
        √ 1-α̅ₜ
        """
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        """
        √ 1\α̅ₜ
        """
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        """
        √ 1\α̅ₜ-1
        """
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        """
        β̃ₜ = σₜ² = βₜ (1-α̅ₜ₋₁)/(1-α̅ₜ)
        """
        self.ddpm_var = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.ddpm_logvar_clipped = torch.log(torch.clamp(self.ddpm_var, min=1e-20))
        """
        μₜ = β̃ₜ √ α̅ₜ₋₁/(1-α̅ₜ)x₀ + √ αₜ (1-α̅ₜ₋₁)/(1-α̅ₜ)xₜ
        """
        self.ddpm_mu_coef1 = (
            self.betas
            * torch.sqrt(self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        self.ddpm_mu_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )

        """
        DDIM parameters

        In DDIM paper https://arxiv.org/pdf/2010.02502, alpha is alpha_cumprod in DDPM https://arxiv.org/pdf/2102.09672
        """
        if use_ddim:
            assert predict_epsilon, "DDIM requires predicting epsilon for now."
            if ddim_discretize == "uniform":  # use the HF "leading" style
                step_ratio = self.denoising_steps // ddim_steps
                self.ddim_t = (
                    torch.arange(0, ddim_steps, device=self.device) * step_ratio
                )
            else:
                raise "Unknown discretization method for DDIM."
            self.ddim_alphas = (
                self.alphas_cumprod[self.ddim_t].clone().to(torch.float32)
            )
            self.ddim_alphas_sqrt = torch.sqrt(self.ddim_alphas)
            self.ddim_alphas_prev = torch.cat(
                [
                    torch.tensor([1.0]).to(torch.float32).to(self.device),
                    self.alphas_cumprod[self.ddim_t[:-1]],
                ]
            )
            self.ddim_sqrt_one_minus_alphas = (1.0 - self.ddim_alphas) ** 0.5

            # Initialize fixed sigmas for inference - eta=0
            ddim_eta = 0
            self.ddim_sigmas = (
                ddim_eta
                * (
                    (1 - self.ddim_alphas_prev)
                    / (1 - self.ddim_alphas)
                    * (1 - self.ddim_alphas / self.ddim_alphas_prev)
                )
                ** 0.5
            )

            # Flip all
            self.ddim_t = torch.flip(self.ddim_t, [0])
            self.ddim_alphas = torch.flip(self.ddim_alphas, [0])
            self.ddim_alphas_sqrt = torch.flip(self.ddim_alphas_sqrt, [0])
            self.ddim_alphas_prev = torch.flip(self.ddim_alphas_prev, [0])
            self.ddim_sqrt_one_minus_alphas = torch.flip(
                self.ddim_sqrt_one_minus_alphas, [0]
            )
            self.ddim_sigmas = torch.flip(self.ddim_sigmas, [0])

    def compute_dispersive_loss(self, representations, loss_type="infonce_l2", temperature=0.5):
        """
        Compute Dispersive Loss for representation regularization.
        
        Args:
            representations: (B, D) tensor of intermediate representations
            loss_type: type of dispersive loss ("infonce_l2", "infonce_cosine", "hinge", "covariance")
            temperature: temperature parameter for InfoNCE-based losses
        
        Returns:
            dispersive_loss: scalar tensor
        """
        if representations.size(0) <= 1:
            return torch.tensor(0.0, device=representations.device)
            
        B, D = representations.shape
        
        if loss_type == "infonce_l2":
            # InfoNCE with L2 distance
            # Compute pairwise L2 distances
            diff = representations.unsqueeze(1) - representations.unsqueeze(0)  # (B, B, D)
            distances = torch.sum(diff ** 2, dim=-1)  # (B, B)
            
            # Dispersive loss: log E[exp(-D(zi, zj)/τ)]
            exp_neg_dist = torch.exp(-distances / temperature)
            dispersive_loss = torch.log(torch.mean(exp_neg_dist))
            
        elif loss_type == "infonce_cosine":
            # InfoNCE with cosine similarity
            # Normalize representations
            representations_norm = F.normalize(representations, p=2, dim=-1)
            
            # Compute cosine similarity matrix
            sim_matrix = torch.mm(representations_norm, representations_norm.t())
            
            # Convert to distance (negative cosine similarity)
            distances = -sim_matrix
            
            # Dispersive loss
            exp_neg_dist = torch.exp(-distances / temperature)
            dispersive_loss = torch.log(torch.mean(exp_neg_dist))
            
        elif loss_type == "hinge":
            # Hinge loss variant
            # Compute pairwise L2 distances
            diff = representations.unsqueeze(1) - representations.unsqueeze(0)  # (B, B, D)
            distances = torch.sum(diff ** 2, dim=-1)  # (B, B)
            
            # Hinge loss with margin epsilon
            epsilon = 1.0
            hinge_losses = torch.clamp(epsilon - distances, min=0) ** 2
            dispersive_loss = torch.mean(hinge_losses)
            
        elif loss_type == "covariance":
            # Covariance loss
            # Center the representations
            representations_centered = representations - representations.mean(dim=0, keepdim=True)
            
            # Compute covariance matrix
            cov_matrix = torch.mm(representations_centered.t(), representations_centered) / (B - 1)
            
            # Dispersive loss: sum of squared covariance elements
            dispersive_loss = torch.sum(cov_matrix ** 2)
            
        else:
            raise ValueError(f"Unknown dispersive loss type: {loss_type}")
            
        return dispersive_loss

    def get_intermediate_representations(self, x, t, cond):
        """
        Extract intermediate representations from the network for dispersive loss.
        This is a hook-based approach that works with most network architectures.
        """
        representations = []
        
        def hook_fn(module, input, output):
            # Flatten the output for dispersive loss computation
            if isinstance(output, torch.Tensor):
                flat_output = output.view(output.size(0), -1)  # (B, D)
                representations.append(flat_output)
        
        # Register hooks based on the network architecture
        hooks = []
        
        # Check for MLP-based networks (DiffusionMLP)
        if hasattr(self.network, 'mlp_mean'):
            mlp_layers = None
            if hasattr(self.network.mlp_mean, 'moduleList'):  # Regular MLP
                mlp_layers = self.network.mlp_mean.moduleList
            elif hasattr(self.network.mlp_mean, 'layers'):  # ResidualMLP
                mlp_layers = self.network.mlp_mean.layers
                
            if mlp_layers is not None and len(mlp_layers) > 0:
                if self.dispersive_loss_layer == "early":
                    hook = mlp_layers[len(mlp_layers) // 4].register_forward_hook(hook_fn)
                    hooks.append(hook)
                elif self.dispersive_loss_layer == "mid":
                    hook = mlp_layers[len(mlp_layers) // 2].register_forward_hook(hook_fn)
                    hooks.append(hook)
                elif self.dispersive_loss_layer == "late":
                    hook = mlp_layers[3 * len(mlp_layers) // 4].register_forward_hook(hook_fn)
                    hooks.append(hook)
                elif self.dispersive_loss_layer == "all":
                    for layer in mlp_layers:
                        hook = layer.register_forward_hook(hook_fn)
                        hooks.append(hook)
        
        # Check for transformer-like architectures
        elif hasattr(self.network, 'encoder') or hasattr(self.network, 'layers'):
            layers = getattr(self.network, 'encoder', getattr(self.network, 'layers', []))
            if isinstance(layers, nn.ModuleList) and len(layers) > 0:
                if self.dispersive_loss_layer == "early":
                    hook = layers[len(layers) // 4].register_forward_hook(hook_fn)
                    hooks.append(hook)
                elif self.dispersive_loss_layer == "mid":
                    hook = layers[len(layers) // 2].register_forward_hook(hook_fn)
                    hooks.append(hook)
                elif self.dispersive_loss_layer == "late":
                    hook = layers[3 * len(layers) // 4].register_forward_hook(hook_fn)
                    hooks.append(hook)
                elif self.dispersive_loss_layer == "all":
                    for layer in layers:
                        hook = layer.register_forward_hook(hook_fn)
                        hooks.append(hook)
        
        # Fallback: try to find intermediate layers in the entire network
        else:
            modules = list(self.network.modules())
            if len(modules) > 1:
                # Register hook on a middle layer
                mid_idx = len(modules) // 2
                hook = modules[mid_idx].register_forward_hook(hook_fn)
                hooks.append(hook)
        
        try:
            # Forward pass to collect representations
            _ = self.network(x, t, cond=cond)
            
            # Compute dispersive loss if representations were collected
            if representations:
                if self.dispersive_loss_layer == "all":
                    # Average dispersive loss across all layers
                    total_loss = 0
                    for repr_tensor in representations:
                        total_loss += self.compute_dispersive_loss(
                            repr_tensor, 
                            self.dispersive_loss_type, 
                            self.dispersive_loss_temperature
                        )
                    dispersive_loss = total_loss / len(representations)
                else:
                    # Use the single collected representation
                    dispersive_loss = self.compute_dispersive_loss(
                        representations[0], 
                        self.dispersive_loss_type, 
                        self.dispersive_loss_temperature
                    )
            else:
                dispersive_loss = torch.tensor(0.0, device=x.device)
                
        finally:
            # Remove all hooks
            for hook in hooks:
                hook.remove()
        
        return dispersive_loss

    # ---------- Sampling ----------#

    def p_mean_var(self, x, t, cond, index=None, network_override=None):
        if network_override is not None:
            noise = network_override(x, t, cond=cond)
        else:
            noise = self.network(x, t, cond=cond)

        if self.predict_epsilon:
            if self.use_ddim:
                """
                x₀ = (xₜ - √ (1-αₜ) ε )/ √ αₜ
                """
                alpha = extract(self.ddim_alphas, index, x.shape)
                alpha_prev = extract(self.ddim_alphas_prev, index, x.shape)
                sqrt_one_minus_alpha = extract(
                    self.ddim_sqrt_one_minus_alphas, index, x.shape
                )
                x_recon = (x - sqrt_one_minus_alpha * noise) / (alpha**0.5)
            else:
                """
                x₀ = √ 1\α̅ₜ xₜ - √ 1\α̅ₜ-1 ε
                """
                x_recon = (
                    extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
                    - extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape) * noise
                )
        else:  # directly predicting x₀
            x_recon = noise
        if self.denoised_clip_value is not None:
            x_recon.clamp_(-self.denoised_clip_value, self.denoised_clip_value)
            if self.use_ddim:
                noise = (x - alpha ** (0.5) * x_recon) / sqrt_one_minus_alpha

        # Clip epsilon for numerical stability
        if self.use_ddim and self.eps_clip_value is not None:
            noise.clamp_(-self.eps_clip_value, self.eps_clip_value)

        if self.use_ddim:
            """
            μ = √ αₜ₋₁ x₀ + √(1-αₜ₋₁ - σₜ²) ε

            eta=0
            """
            sigma = extract(self.ddim_sigmas, index, x.shape)
            dir_xt = (1.0 - alpha_prev - sigma**2).sqrt() * noise
            mu = (alpha_prev**0.5) * x_recon + dir_xt
            var = sigma**2
            logvar = torch.log(var)
        else:
            """
            μₜ = β̃ₜ √ α̅ₜ₋₁/(1-α̅ₜ)x₀ + √ αₜ (1-α̅ₜ₋₁)/(1-α̅ₜ)xₜ
            """
            mu = (
                extract(self.ddpm_mu_coef1, t, x.shape) * x_recon
                + extract(self.ddpm_mu_coef2, t, x.shape) * x
            )
            logvar = extract(self.ddpm_logvar_clipped, t, x.shape)
        return mu, logvar

    @torch.no_grad()
    def forward(self, cond, deterministic=True):
        """
        Forward pass for sampling actions. Used in evaluating pre-trained/fine-tuned policy. Not modifying diffusion clipping

        Args:
            cond: dict with key state/rgb; more recent obs at the end
                state: (B, To, Do)
                rgb: (B, To, C, H, W)
        Return:
            Sample: namedtuple with fields:
                trajectories: (B, Ta, Da)
        """
        device = self.betas.device
        sample_data = cond["state"] if "state" in cond else cond["rgb"]
        B = len(sample_data)

        # Loop
        x = torch.randn((B, self.horizon_steps, self.action_dim), device=device)
        if self.use_ddim:
            t_all = self.ddim_t
        else:
            t_all = list(reversed(range(self.denoising_steps)))
        for i, t in enumerate(t_all):
            t_b = make_timesteps(B, t, device)
            index_b = make_timesteps(B, i, device)
            mean, logvar = self.p_mean_var(
                x=x,
                t=t_b,
                cond=cond,
                index=index_b,
            )
            std = torch.exp(0.5 * logvar)

            # Determine noise level
            if self.use_ddim:
                std = torch.zeros_like(std)
            else:
                if t == 0:
                    std = torch.zeros_like(std)
                else:
                    std = torch.clip(std, min=1e-3)
            noise = torch.randn_like(x).clamp_(
                -self.randn_clip_value, self.randn_clip_value
            )
            x = mean + std * noise

            # clamp action at final step
            if self.final_action_clip_value is not None and i == len(t_all) - 1:
                x = torch.clamp(
                    x, -self.final_action_clip_value, self.final_action_clip_value
                )
        return Sample(x, None)

    # ---------- Supervised training ----------#

    def loss(self, x, *args):
        batch_size = len(x)
        t = torch.randint(
            0, self.denoising_steps, (batch_size,), device=x.device
        ).long()
        return self.p_losses(x, *args, t)

    def p_losses(                                                                 
        self,
        x_start,
        cond: dict,
        t,
    ):
        """
        Enhanced loss function with Dispersive Loss.
        
        If predicting epsilon: E_{t, x0, ε} [||ε - ε_θ(√α̅ₜx0 + √(1-α̅ₜ)ε, t)||² + λ * L_disp

        Args:
            x_start: (batch_size, horizon_steps, action_dim)
            cond: dict with keys as step and value as observation
            t: batch of integers
        """
        device = x_start.device

        # Forward process
        noise = torch.randn_like(x_start, device=device)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Predict
        x_recon = self.network(x_noisy, t, cond=cond)
        
        # Standard diffusion loss
        if self.predict_epsilon:
            diffusion_loss = F.mse_loss(x_recon, noise, reduction="mean")
        else:
            diffusion_loss = F.mse_loss(x_recon, x_start, reduction="mean")

        # Add Dispersive Loss if enabled
        total_loss = diffusion_loss
        if self.use_dispersive_loss:
            dispersive_loss = self.get_intermediate_representations(x_noisy, t, cond)
            total_loss = diffusion_loss + self.dispersive_loss_weight * dispersive_loss
            
            # Log dispersive loss for monitoring
            self._log_dispersive_loss = dispersive_loss.item()
        else:
            # Set to zero for baseline experiments
            self._log_dispersive_loss = 0.0

        return total_loss

    def q_sample(self, x_start, t, noise=None):     
        """
        q(xₜ | x₀) = 𝒩(xₜ; √ α̅ₜ x₀, (1-α̅ₜ)I)
        xₜ = √ α̅ₜ xₒ + √ (1-α̅ₜ) ε
        """
        if noise is None:
            device = x_start.device
            noise = torch.randn_like(x_start, device=device)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
