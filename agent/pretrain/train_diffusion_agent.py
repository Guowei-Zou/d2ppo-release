#!/usr/bin/env python3

import logging
import wandb
import numpy as np
import torch

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.pretrain.train_agent import PreTrainAgent, batch_to_device


class TrainDiffusionAgent(PreTrainAgent):
    """
    Enhanced training agent for diffusion models with Dispersive Loss support.
    Based on "Diffuse and Disperse: Image Generation with Representation Regularization".
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        
        # Check if Dispersive Loss is enabled
        self.use_dispersive_loss = getattr(self.model, 'use_dispersive_loss', False)
        if self.use_dispersive_loss:
            log.info("Dispersive Loss is enabled in the model")
            log.info(f"Dispersive Loss weight: {getattr(self.model, 'dispersive_loss_weight', 'N/A')}")
            log.info(f"Dispersive Loss temperature: {getattr(self.model, 'dispersive_loss_temperature', 'N/A')}")
            log.info(f"Dispersive Loss type: {getattr(self.model, 'dispersive_loss_type', 'N/A')}")
            log.info(f"Dispersive Loss layer: {getattr(self.model, 'dispersive_loss_layer', 'N/A')}")
            
            # Debug: check network structure
            if hasattr(self.model.network, 'mlp_mean'):
                mlp_mean = self.model.network.mlp_mean
                if hasattr(mlp_mean, 'moduleList'):
                    log.info(f"Network has MLP.moduleList with {len(mlp_mean.moduleList)} layers")
                elif hasattr(mlp_mean, 'layers'):
                    log.info(f"Network has ResidualMLP.layers with {len(mlp_mean.layers)} layers")
                else:
                    log.warning("Network has mlp_mean but no recognizable layer structure")
            else:
                log.warning("Network does not have mlp_mean attribute")
        else:
            log.info("Dispersive Loss is DISABLED")

    def run(self):

        timer = Timer()
        self.epoch = 1
        for _ in range(self.n_epochs):

            # train
            loss_train_epoch = []
            dispersive_loss_epoch = []
            
            for batch_train in self.dataloader_train:
                if self.dataset_train.device == "cpu":
                    batch_train = batch_to_device(batch_train)

                self.model.train()
                
                # Compute loss (potentially including Dispersive Loss)
                loss_train = self.model.loss(*batch_train)
                
                # Log dispersive loss component if available
                if self.use_dispersive_loss and hasattr(self.model, '_log_dispersive_loss'):
                    dispersive_loss_epoch.append(self.model._log_dispersive_loss)
                
                loss_train.backward()
                loss_train_epoch.append(loss_train.item())

                self.optimizer.step()
                self.optimizer.zero_grad()
                
            loss_train = np.mean(loss_train_epoch)
            dispersive_loss_avg = np.mean(dispersive_loss_epoch) if dispersive_loss_epoch else 0.0

            # validate
            loss_val_epoch = []
            dispersive_loss_val_epoch = []
            
            if self.dataloader_val is not None and self.epoch % self.val_freq == 0:
                self.model.eval()
                with torch.no_grad():
                    for batch_val in self.dataloader_val:
                        if self.dataset_val.device == "cpu":
                            batch_val = batch_to_device(batch_val)
                        loss_val = self.model.loss(*batch_val)
                        loss_val_epoch.append(loss_val.item())
                        
                        # Log validation dispersive loss if available
                        if self.use_dispersive_loss and hasattr(self.model, '_log_dispersive_loss'):
                            dispersive_loss_val_epoch.append(self.model._log_dispersive_loss)
                            
                self.model.train()
                
            loss_val = np.mean(loss_val_epoch) if len(loss_val_epoch) > 0 else None
            dispersive_loss_val = np.mean(dispersive_loss_val_epoch) if dispersive_loss_val_epoch else 0.0

            # update lr
            self.lr_scheduler.step()

            # update ema
            if self.epoch % self.update_ema_freq == 0:
                self.step_ema()

            # save model
            if self.epoch % self.save_model_freq == 0 or self.epoch == self.n_epochs:
                self.save_model()

            # log loss
            if self.epoch % self.log_freq == 0:
                log.info(
                    f"{self.epoch}: train loss {loss_train:8.4f} | t:{timer():8.4f}"
                )
                
                # Log dispersive loss if enabled
                if self.use_dispersive_loss:
                    log.info(f"  -> dispersive loss {dispersive_loss_avg:8.6f}")
                    # Calculate approximate diffusion loss component
                    if dispersive_loss_avg > 0:
                        diffusion_loss_approx = loss_train - (
                            getattr(self.model, 'dispersive_loss_weight', 0.5) * dispersive_loss_avg
                        )
                        log.info(f"  -> diffusion loss (approx) {diffusion_loss_approx:8.6f}")
                    
                if self.use_wandb:
                    log_dict = {
                        "loss - train": loss_train,
                        "epoch": self.epoch,
                        "learning_rate": self.optimizer.param_groups[0]['lr'],
                    }
                    
                    # Add validation loss if available
                    if loss_val is not None:
                        log_dict["loss - val"] = loss_val
                        
                    # Add dispersive loss components if available
                    if self.use_dispersive_loss:
                        if dispersive_loss_avg > 0:
                            log_dict["dispersive_loss - train"] = dispersive_loss_avg
                            # Calculate diffusion loss component (approximate)
                            diffusion_loss_train = loss_train - (
                                getattr(self.model, 'dispersive_loss_weight', 0.5) * dispersive_loss_avg
                            )
                            log_dict["diffusion_loss - train"] = diffusion_loss_train
                            
                        if dispersive_loss_val > 0:
                            log_dict["dispersive_loss - val"] = dispersive_loss_val
                            
                    wandb.log(log_dict, step=self.epoch)

            # count
            self.epoch += 1 