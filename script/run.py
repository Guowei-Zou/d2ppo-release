"""
Launcher for all experiments. Download pre-training data, normalization statistics, and pre-trained checkpoints if needed.

"""

import os
import sys
import pretty_errors
import logging

import math
import hydra
from omegaconf import OmegaConf
import gdown
from datetime import datetime
from download_url import (
    get_dataset_download_url,
    get_normalization_download_url,
    get_checkpoint_download_url,
)

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)

def dynamic_name_resolver(template_name):
    """Legacy resolver for backwards compatibility."""
    return template_name

OmegaConf.register_new_resolver("dynamic_name", dynamic_name_resolver, replace=True)


def update_experiment_naming(cfg: OmegaConf):
    """
    Update experiment name and logdir to reflect actual resolved parameter values.
    This ensures that command-line overrides are properly reflected in directory names.
    """
    if "name" not in cfg:
        return
        
    original_name = cfg.name
    
    new_name = None
    
    if hasattr(cfg, 'dispersive_loss_weight') and hasattr(cfg, 'dispersive_loss_layer'):
        new_name = f"{cfg.env}_pre_diffusion_mlp_dispersive_ta{cfg.horizon_steps}_td{cfg.denoising_steps}_dw{cfg.dispersive_loss_weight}_dl{cfg.dispersive_loss_layer}"
        log.info(f"Detected dispersive loss experiment with weight={cfg.dispersive_loss_weight}, layer={cfg.dispersive_loss_layer}")
    elif "_ft_" in original_name:
        env_key = getattr(cfg, 'env_name', getattr(cfg, 'env', 'unknown'))
        if "_ft_diffusion_mlp_img" in original_name:
            new_name = f"{env_key}_ft_diffusion_mlp_img_ta{cfg.horizon_steps}_td{cfg.denoising_steps}_tdf{cfg.ft_denoising_steps}"
            log.info(f"Detected fine-tuning diffusion MLP image experiment")
        elif "_ft_diffusion_mlp" in original_name:
            new_name = f"{env_key}_ft_diffusion_mlp_ta{cfg.horizon_steps}_td{cfg.denoising_steps}_tdf{cfg.ft_denoising_steps}"
            log.info(f"Detected fine-tuning diffusion MLP experiment")
        elif "_ft_ppo_diffusion" in original_name:
            new_name = f"{env_key}_ft_ppo_diffusion_ta{cfg.horizon_steps}_td{cfg.denoising_steps}_tdf{cfg.ft_denoising_steps}"
            log.info(f"Detected fine-tuning PPO diffusion experiment")
    elif "_pre_diffusion_mlp" in original_name and "dispersive" not in original_name:
        env_key = getattr(cfg, 'env', 'unknown')
        if "_img" in original_name:
            new_name = f"{env_key}_pre_diffusion_mlp_img_ta{cfg.horizon_steps}_td{cfg.denoising_steps}"
        else:
            new_name = f"{env_key}_pre_diffusion_mlp_ta{cfg.horizon_steps}_td{cfg.denoising_steps}"
        log.info(f"Detected regular diffusion experiment")
    elif "_pre_diffusion_unet" in original_name:
        # UNet diffusion experiment
        env_key = getattr(cfg, 'env', 'unknown')
        new_name = f"{env_key}_pre_diffusion_unet_ta{cfg.horizon_steps}_td{cfg.denoising_steps}"
        log.info(f"Detected UNet diffusion experiment")
    elif "_pre_gaussian_mlp" in original_name:
        # Gaussian MLP experiment
        env_key = getattr(cfg, 'env', 'unknown')
        if "_img" in original_name:
            new_name = f"{env_key}_pre_gaussian_mlp_img_ta{cfg.horizon_steps}"
        else:
            new_name = f"{env_key}_pre_gaussian_mlp_ta{cfg.horizon_steps}"
        log.info(f"Detected Gaussian MLP experiment")
    
    # Update name if we generated a new one
    if new_name and new_name != original_name:
        cfg.name = new_name
        log.info(f"Updated experiment name from '{original_name}' to '{new_name}'")
        
        # Update logdir to use the corrected name
        if "logdir" in cfg:
            # Regenerate logdir with the new name
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            seed = getattr(cfg, 'seed', 42)
            base_log_dir = os.environ.get('DPPO_LOG_DIR', './log')
            
            if "robomimic-pretrain" in str(cfg.logdir):
                cfg.logdir = f"{base_log_dir}/robomimic-pretrain/{cfg.name}/{timestamp}_{seed}"
            elif "robomimic-finetune" in str(cfg.logdir):
                cfg.logdir = f"{base_log_dir}/robomimic-finetune/{cfg.name}/{timestamp}_{seed}"
            elif "gym-pretrain" in str(cfg.logdir):
                cfg.logdir = f"{base_log_dir}/gym-pretrain/{cfg.name}/{timestamp}_{seed}"
            elif "gym-finetune" in str(cfg.logdir):
                cfg.logdir = f"{base_log_dir}/gym-finetune/{cfg.name}/{timestamp}_{seed}"
            elif "furniture-pretrain" in str(cfg.logdir):
                cfg.logdir = f"{base_log_dir}/furniture-pretrain/{cfg.name}/{timestamp}_{seed}"
            elif "furniture-finetune" in str(cfg.logdir):
                cfg.logdir = f"{base_log_dir}/furniture-finetune/{cfg.name}/{timestamp}_{seed}"
            elif "d3il-pretrain" in str(cfg.logdir):
                cfg.logdir = f"{base_log_dir}/d3il-pretrain/{cfg.name}/{timestamp}_{seed}"
            elif "d3il-finetune" in str(cfg.logdir):
                cfg.logdir = f"{base_log_dir}/d3il-finetune/{cfg.name}/{timestamp}_{seed}"
            else:
                # Generic fallback
                log.warning(f"Unknown logdir pattern: {cfg.logdir}, using generic update")
                
            log.info(f"Updated logdir: {cfg.logdir}")


# suppress d4rl import error
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

# add logger
log = logging.getLogger(__name__)

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)


@hydra.main(
    version_base=None,
    config_path=os.path.join(
        os.getcwd(), "cfg"
    ),  # possibly overwritten by --config-path
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers will use the same time.
    OmegaConf.resolve(cfg)
    
    # Regenerate experiment name and logdir with actual resolved values
    update_experiment_naming(cfg)

    # For pre-training: download dataset if needed
    if "train_dataset_path" in cfg and not os.path.exists(cfg.train_dataset_path):
        download_url = get_dataset_download_url(cfg)
        download_target = os.path.dirname(cfg.train_dataset_path)
        log.info(f"Downloading dataset from {download_url} to {download_target}")
        gdown.download_folder(url=download_url, output=download_target)

    # For for-tuning: download normalization if needed
    if "normalization_path" in cfg and not os.path.exists(cfg.normalization_path):
        download_url = get_normalization_download_url(cfg)
        download_target = cfg.normalization_path
        dir_name = os.path.dirname(download_target)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        log.info(
            f"Downloading normalization statistics from {download_url} to {download_target}"
        )
        gdown.download(url=download_url, output=download_target, fuzzy=True)

    # For for-tuning: download checkpoint if needed
    if "base_policy_path" in cfg and not os.path.exists(cfg.base_policy_path):
        download_url = get_checkpoint_download_url(cfg)
        if download_url is None:
            raise ValueError(
                f"Unknown checkpoint path. Did you specify the correct path to the policy you trained?"
            )
        download_target = cfg.base_policy_path
        dir_name = os.path.dirname(download_target)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        log.info(f"Downloading checkpoint from {download_url} to {download_target}")
        gdown.download(url=download_url, output=download_target, fuzzy=True)

    # Deal with isaacgym needs to be imported before torch
    if "env" in cfg and "env_type" in cfg.env and cfg.env.env_type == "furniture":
        import furniture_bench

    # run agent
    cls = hydra.utils.get_class(cfg._target_)
    agent = cls(cfg)
    agent.run()


if __name__ == "__main__":
    main()
