import os
import gc
import PIL
import math
import torch
import torch.nn.functional as F
from PIL import Image
from einops import rearrange
from omegaconf import DictConfig
from torchvision import transforms
from collections import OrderedDict
from hydra.utils import instantiate
from accelerate import Accelerator
from diffusers.utils.torch_utils import randn_tensor
from typing import Any, Callable, Dict, List, Optional, Union

from emulator.models.basemodel import BaseModel
from emulator.models.pixelflow_basemodel import PixelFlowBaseModel
from emulator.schedulers.pyramidal_flow import PyramidFlowMatchEulerDiscreteScheduler
from emulator.visualization.visualize import create_temporal_predictions_figs


class PyramidalFlowModel(PixelFlowBaseModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.is_pyramidal = True

    def training_step(self, batch: Any, batch_idx: int):

        # DiT forward
        model_preds_list = self.model(
            sample=batch["noisy_latents_list"],
            timestep_ratio=batch["timesteps"],
            condition_sample=[[x.unsqueeze(0)] for x in batch["X"].transpose(1,2)],
            pooled_projections=None,
        )

        loss_list = []
    
        for model_pred, target in zip(model_preds_list, batch["targets_list"]):
            # Compute the loss.
            loss_weight = torch.ones_like(target)

            loss = torch.mean(
                (loss_weight.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                1,
            )
            loss_list.append(loss)

        diffusion_loss = torch.cat(loss_list, dim=0).mean()

        n_zero_gradients = (
            sum(
                [
                    int(torch.count_nonzero(p.grad == 0))
                    for p in self.parameters()
                    if p.grad is not None
                ]
            )
            / self.n_params
        )

        self.log_dict({"train/loss": diffusion_loss.detach(), "n_zero_gradients": n_zero_gradients})

        # calculate the loss
        return {"loss": diffusion_loss}

    def sample_block_noise(self, bs, ch, temp, height, width, eps=1e-6):
        gamma = -self.noise_scheduler.config.gamma
        dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(4), torch.eye(4) * (1 + gamma) - torch.ones(4, 4) * gamma + eps * torch.eye(4))
        block_number = bs * ch * temp * (height // 2) * (width // 2)
        noise = torch.stack([dist.sample() for _ in range(block_number)]) # [block number, 4]
        noise = rearrange(noise, '(b c t h w) (p q) -> b c t (h p) (w q)',b=bs,c=ch,t=temp,h=height//2,w=width//2,p=2,q=2)
        return noise

    def predict(self, X, metadata=None, *args, **kwargs):
        # swap_lat_lon reproduces checkpoints trained with swapped spatial axes.
        spatial_layout = metadata.get("spatial_layout", "swap_lat_lon")
        if isinstance(spatial_layout, (list, tuple)):
            spatial_layout = spatial_layout[0]
        if spatial_layout == "lat_lon":
            height = metadata["latitude"].shape[1]
            width = metadata["longitude"].shape[1]
        else:
            height = metadata["longitude"].shape[1]
            width = metadata["latitude"].shape[1]
        return self.generate(
            X,
            height=height,
            width=width,
            temp=self.datamodule_config.output_seq_len[self.datamodule_config.timescales[0]],
            num_inference_steps=self.num_inference_steps,
            video_num_inference_steps=self.num_inference_steps,
        )

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        temp,
        height,
        width,
        dtype,
        device,
        generator,
    ):
        # No downsampling since no VAE
        shape = (
            batch_size,
            num_channels_latents,
            int(temp),
            int(height),
            int(width),
        )
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    @torch.no_grad()
    def get_pyramid_latent(self, x, stage_num):
        # x is the origin vae latent
        vae_latent_list = []
        vae_latent_list.append(x)

        temp, height, width = x.shape[-3], x.shape[-2], x.shape[-1]
        for _ in range(stage_num):
            height //= 2
            width //= 2
            x = rearrange(x, 'b c t h w -> (b t) c h w')
            x = torch.nn.functional.interpolate(x, size=(height, width), mode='bilinear')
            x = rearrange(x, '(b t) c h w -> b c t h w', t=temp)
            vae_latent_list.append(x)

        vae_latent_list = list(reversed(vae_latent_list))
        return vae_latent_list

    @torch.no_grad()
    def generate_one_unit(
        self,
        latents,
        past_conditions, # List of past conditions, contains the conditions of each stage
        X,
        num_inference_steps,
        height,
        width,
        temp,
        device,
        dtype,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        is_first_frame: bool = False,
    ):
        num_stages = self.noise_scheduler.num_stages
        intermed_latents = []

        for i_s in range(num_stages):
            self.noise_scheduler.set_timesteps(num_inference_steps[i_s], i_s, device=device)
            timesteps = self.noise_scheduler.timesteps

            if i_s > 0:
                height *= 2; width *= 2
                latents = rearrange(latents, 'b c t h w -> (b t) c h w')
                latents = F.interpolate(latents, size=(height, width), mode='nearest')
                latents = rearrange(latents, '(b t) c h w -> b c t h w', t=temp)
                # Fix the stage
                ori_sigma = 1 - self.noise_scheduler.ori_start_sigmas[i_s]   # the original coeff of signal
                gamma = -self.noise_scheduler.config.gamma
                alpha = 1 / (math.sqrt(1 + (1 / gamma)) * (1 - ori_sigma) + ori_sigma)
                beta = alpha * (1 - ori_sigma) / math.sqrt(gamma)

                bs, ch, temp, height, width = latents.shape
                noise = self.sample_block_noise(bs, ch, temp, height, width)
                noise = noise.to(device=device, dtype=dtype)
                latents = alpha * latents + beta * noise    # To fix the block artifact

            for idx, t in enumerate(timesteps):
                latent_model_input = latents

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0]).to(latent_model_input.dtype)
                
                latent_model_input = past_conditions[i_s] + [latent_model_input]

                noise_pred = self.model(
                    sample=[latent_model_input],
                    timestep_ratio=timestep,
                    condition_sample=[[X.transpose(1,2)]],
                    pooled_projections=None,
                )

                noise_pred = noise_pred[0]

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=timestep,
                    sample=latents,
                    generator=generator,
                ).prev_sample

            intermed_latents.append(latents)

        return intermed_latents

    @torch.no_grad()
    def generate(
        self,
        X: Union[torch.Tensor, List[torch.Tensor]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        temp: int = 1,
        num_inference_steps: Optional[Union[int, List[int]]] = 28,
        video_num_inference_steps: Optional[Union[int, List[int]]] = 28,
        alpha: float = 0.5,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        callback: Optional[Callable[[int, int, Dict], None]] = None,
    ):
        device = self.device
        dtype = self.dtype
        frame_per_unit = 1
        num_stages = self.noise_scheduler.num_stages

        batch_size = X.shape[0]

        assert (temp - 1) % frame_per_unit == 0, "The frames should be divided by frame_per unit"

        if isinstance(num_inference_steps, int):
            num_inference_steps = [num_inference_steps] * num_stages

        if isinstance(video_num_inference_steps, int):
            video_num_inference_steps = [video_num_inference_steps] * num_stages

        # Create the initial random noise
        num_channels_latents = self.model.config.in_channels
        latents = self.prepare_latents(
            batch_size,
            num_channels_latents,
            temp,
            height,
            width,
            dtype,
            device,
            generator,
        )

        temp, height, width = latents.shape[-3], latents.shape[-2], latents.shape[-1]

        latents = rearrange(latents, 'b c t h w -> (b t) c h w')
        # by default, we needs to start from the block noise
        for _ in range(num_stages-1):
            height //= 2;width //= 2
            latents = F.interpolate(latents, size=(height, width), mode='bilinear') * 2
        
        latents = rearrange(latents, '(b t) c h w -> b c t h w', t=temp)

        num_units = 1 + (temp - 1) // frame_per_unit

        generated_latents_list = []    # The generated results
        last_generated_latents = None

        for unit_index in range(num_units):
            gc.collect()
            torch.cuda.empty_cache()
            
            if callback:
                callback(unit_index, num_units)

            if unit_index == 0:
                past_condition_latents = [[] for _ in range(num_stages)]
                intermed_latents = self.generate_one_unit(
                    latents[:,:,:1],
                    past_condition_latents,
                    X,
                    num_inference_steps,
                    height,
                    width,
                    1,
                    device,
                    dtype,
                    generator,
                    is_first_frame=True,
                )
            else:
                # prepare the condition latents
                past_condition_latents = []
                clean_latents_list = self.get_pyramid_latent(torch.cat(generated_latents_list, dim=2), num_stages - 1)

                for i_s in range(num_stages):
                    last_cond_latent = clean_latents_list[i_s][:,:,-(frame_per_unit):]

                    stage_input = [last_cond_latent]
            
                    # pad the past clean latents
                    cur_unit_num = unit_index
                    cur_stage = i_s
                    cur_unit_ptx = 1

                    while cur_unit_ptx < cur_unit_num:
                        cur_stage = max(cur_stage - 1, 0)
                        if cur_stage == 0:
                            break
                        cur_unit_ptx += 1
                        cond_latents = clean_latents_list[cur_stage][:, :, -(cur_unit_ptx * frame_per_unit) : -((cur_unit_ptx - 1) * frame_per_unit)]
                        stage_input.append(cond_latents)

                    if cur_stage == 0 and cur_unit_ptx < cur_unit_num:
                        cond_latents = clean_latents_list[0][:, :, :-(cur_unit_ptx * frame_per_unit)]
                        stage_input.append(cond_latents)
                
                    stage_input = list(reversed(stage_input))
                    past_condition_latents.append(stage_input)

                intermed_latents = self.generate_one_unit(
                    latents[:,:, 1 + (unit_index - 1) * frame_per_unit:1 + unit_index * frame_per_unit],
                    past_condition_latents,
                    X,
                    video_num_inference_steps,
                    height,
                    width,
                    frame_per_unit,
                    device,
                    dtype,
                    generator,
                    is_first_frame=False,
                )

            generated_latents_list.append(intermed_latents[-1])
            last_generated_latents = intermed_latents

        generated_latents = torch.cat(generated_latents_list, dim=2) # (b, c, t, h, w)

        generated_latents = generated_latents.transpose(1, 2)  # Change to (b, t, c, h, w)

        return generated_latents

    def _visualize_predictions(self, preds, targets, metadata, figure_name, num_samples=1, timescale="monthly"):
        if len(self.datamodule_config.timescales) == 1:
            timescale = self.datamodule_config.timescales[0]
        else:
            timescale = "monthly"
        fig = create_temporal_predictions_figs(
            preds=preds,
            targets=targets,
            metadata=metadata,  
            n=list(range(num_samples)),
            timescale=timescale
        )

        self.logger.log_image(key=figure_name, images=[fig])

    def setup_noise_scheduler(self):
        self.noise_scheduler: PyramidFlowMatchEulerDiscreteScheduler = instantiate(self.noise_scheduler_config)
