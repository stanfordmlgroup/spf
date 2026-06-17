import gc
import math
import torch
import torch.nn.functional as F

from torch import Tensor
from typing import Any
from omegaconf import DictConfig
from hydra.utils import instantiate

from diffusers.utils.torch_utils import randn_tensor
from diffusers.models.embeddings import get_2d_rotary_pos_embed, get_3d_rotary_pos_embed
from torch_ema import ExponentialMovingAverage as EMA
from einops import rearrange

from emulator.models.basemodel import BaseModel
from emulator.models.pixelflow.encoder import PixelFlowEncoder
from emulator.schedulers.pyramidal_flow import PyramidFlowMatchEulerDiscreteScheduler
from emulator.losses import MSE_PixelFlow


class PixelFlowBaseModel(BaseModel):
    def __init__(
        self,
        transformer_config: DictConfig,
        noise_scheduler_config: DictConfig,
        encoder_config: DictConfig = None,
        datamodule_config: DictConfig = None,
        *args,
        **kwargs,
    ):
        super().__init__(datamodule_config=datamodule_config, *args, **kwargs)

        self.save_hyperparameters()

        self.current_stage = "train"

        if len(self.climate_models) > 1:
            transformer_config.num_classes = len(self.climate_models)

        self.model = instantiate(transformer_config)
        self.patch_size = self.model.patch_size
        self.head_dim = self.model.attention_head_dim

        if encoder_config is not None:
            self.encoder: PixelFlowEncoder = instantiate(encoder_config)
            self.cond_patch_size = encoder_config.patch_size

        self.noise_scheduler_config = noise_scheduler_config
        self.num_inference_steps = kwargs['configs']['model'].get('num_inference_steps', 30)
        if 'num_inference_steps' in self.config:
            self.num_inference_steps = self.config['num_inference_steps']

        self.setup_noise_scheduler()

        self.ema = None
        if self.training and self.config.get("resume_from_checkpoint") is not None:
            self.ema = EMA(self.model.parameters(), decay=0.9999)

    def encode(self, x):

        # Remove frames dimension
        x = x.squeeze(1)

        embeddings = self.encoder(x)
        attention_mask = None

        return embeddings, attention_mask

    def forward(
        self,
        X: Tensor,
    ) -> Tensor:
        raise NotImplementedError("PixelFlowBaseModel is an abstract class!")

    def training_step(self, batch: Any, batch_idx: int):

        plot_train_preds = False
        if batch_idx == 0 and plot_train_preds:
            self.train_batch = batch

        encoder_hidden_states, _ = self.encode(batch["cond_values"])

        target = batch["target_values"]

        # Stack for sequence packing mode
        if encoder_hidden_states.ndim == 3:
            seqlen_list_k = [encoder_hidden_states.shape[1]] * encoder_hidden_states.shape[0]
            encoder_hidden_states = torch.concatenate([e for e in encoder_hidden_states])
        else:
            seqlen_list_k = batch["seqlen_list_k"]
        cumsum_k_len = torch.cumsum(torch.tensor([0] + seqlen_list_k), 0).to(torch.int32).to(self.device)

        # Climate model embeddings
        if len(self.climate_models) > 1:
            climate_models = batch["metadata"]["climate_model"]
            class_labels = torch.tensor(
                [self.climate_models.index(model) for model in climate_models],
                dtype=torch.int32,
                device=self.device
            )
        else:
            class_labels = None

        model_output = self.model(
            hidden_states=batch["pixel_values"],
            encoder_hidden_states=encoder_hidden_states,
            class_labels=class_labels,
            timestep=batch["timesteps"],
            latent_size=batch["latent_size"],
            latent_time=batch["latent_time"] if "latent_time" in batch else None,
            pos_embed_q=batch["pos_embed_q"],
            pos_embed_k=batch["pos_embed_k"],
            cu_seqlens_q=batch["cumsum_q_len"],
            cu_seqlens_k=cumsum_k_len,
            seqlen_list_q=batch["seqlen_list_q"],
            seqlen_list_k=seqlen_list_k,
        )

        loss_items = self.criterion(
            model_output=model_output,
            target=target,
            seqlen_list_q=batch["seqlen_list_q"],
            timelen_list_q=batch["timelen_list_q"] if "timelen_list_q" in batch else None,
        )

        if (
            "padding_size" in batch
            and batch["padding_size"] is not None
            and batch["padding_size"] > 0
        ):
            loss_items = loss_items[:-1]

        loss = loss_items.mean()

        # Check if loss is NaN or Inf
        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError(
                f"Loss is NaN or Inf at batch index {batch_idx}. "
            )

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

        self.log_dict({"train/loss": loss, "n_zero_gradients": n_zero_gradients})

        return {"loss": loss}

    def predict(self, X, metadata=None, *args, **kwargs):
        if self.trainer.validating:
            height = self.trainer.train_dataloader.dataset.dataset.climate_model_datasets[0].grid.lon.shape[0]
            width = self.trainer.train_dataloader.dataset.dataset.climate_model_datasets[0].grid.lat.shape[0]
        else:
            height = self.trainer.datamodule.test_ds.climate_model_datasets[0].grid.lon.shape[0]
            width = self.trainer.datamodule.test_ds.climate_model_datasets[0].grid.lat.shape[0]

        if isinstance(self.num_inference_steps, int):
            num_inference_steps = [self.num_inference_steps] * self.noise_scheduler_config.num_stages

        batch_size = X.shape[0]

        encoder_hidden_states, encoder_attention_mask = self.encode(X)

        # Climate model embeddings
        if len(self.climate_models) > 1:
            climate_models = metadata["climate_model"]
            class_labels = torch.tensor(
                [self.climate_models.index(model) for model in climate_models],
                dtype=torch.int32,
                device=self.device
            )
        else:
            class_labels = None

        # Compute positional embeddings for conditioning
        # Unflatten so batch size is the first dimension
        # Instead of sequence packing
        cond_rope_pos = kwargs['pos_embed_k']
        cond_rope_pos = cond_rope_pos.reshape(
            batch_size,
            cond_rope_pos.shape[0] // batch_size,
            cond_rope_pos.shape[-2],
            cond_rope_pos.shape[-1]
        )

        num_vars = len(self.datamodule_config.output_variables)

        init_factor = 2 ** (self.noise_scheduler_config.num_stages - 1)
        height, width = height // init_factor, width // init_factor
        shape = (batch_size, num_vars, height, width)
        latents = randn_tensor(shape, device=self.device, dtype=torch.float32)

        pos_embed_split = torch.split(kwargs['pos_embed_q'], kwargs["seqlen_list_q"], dim=0)

        for stage_idx in range(self.noise_scheduler_config.num_stages):
            self.noise_scheduler.set_timesteps(
                num_inference_steps[stage_idx],
                stage_idx,
                device=self.device,
            )
            Timesteps = self.noise_scheduler.timesteps

            if stage_idx > 0:
                height, width = height * 2, width * 2
                latents = F.interpolate(latents, size=(height, width), mode="nearest")
                # 2. fetch signal coefficient  s_k  for the *next* stage (k)
                sigma_start_k = self.noise_scheduler.ori_start_sigmas[stage_idx]  # σ^start_k
                e_kp1 = 1.0 - sigma_start_k                                     # Eq. (7)

                # 3. compute scale factor  s_k / e_{k+1}  ( = (1+s_k)/2  when γ_k = −1/3 )
                gamma = self.noise_scheduler.config.gamma     #  = −1/3
                scale = math.sqrt(1 - gamma) / (((1 - gamma) / math.sqrt(-gamma)) * (1 - e_kp1) + e_kp1 * math.sqrt(1 - gamma))

                # 4. compute noise weight  α_k   (Eq. 25)
                alpha = scale * (1 - e_kp1) / math.sqrt(-gamma)

                # 5. add the corrective Gaussian noise
                noise  = self.sample_block_noise(*latents.shape)
                noise  = noise.to(device=self.device, dtype=latents.dtype)
                latents = scale * latents + alpha * noise     # Eq. (20)

            # Use latent sizes to identify the resolution stage
            # (All the same in the batch at inference time)
            size_tensor = torch.tensor(
                [kwargs["latent_size"][stage_idx]] * batch_size, dtype=torch.int32, device=self.device
            )
            time_tensor = torch.tensor(
                [kwargs["latent_time"][stage_idx]] * batch_size, dtype=torch.int32, device=self.device
            ) if kwargs["latent_time"] is not None else None

            rope_pos = pos_embed_split[stage_idx]
            rope_pos = rope_pos.expand(batch_size, *rope_pos.shape)

            # euler
            for T in Timesteps:
                latent_model_input = latents
                timestep = T.expand(latent_model_input.shape[0]).to(
                    latent_model_input.dtype
                )

                noise_pred = self.model(
                    latent_model_input,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    class_labels=class_labels,
                    timestep=timestep,
                    latent_size=size_tensor,
                    latent_time=time_tensor,
                    pos_embed_q=rope_pos,
                    pos_embed_k=cond_rope_pos,
                )

                latents = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=timestep,
                    sample=latents
                ).prev_sample

        samples = latents

        # Add a dummy frames dimension
        samples = samples.unsqueeze(1).detach()

        return samples

    def sample_block_noise(self, bs, ch, height, width, eps=1e-6):
        # Assumes negative gamma
        gamma = self.noise_scheduler.config.gamma
        dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(4), torch.eye(4) * (1 - gamma) - torch.ones(4, 4) * -gamma + eps * torch.eye(4))
        block_number = bs * ch * (height // 2) * (width // 2)
        noise = torch.stack([dist.sample() for _ in range(block_number)]) # [block number, 4]
        noise = rearrange(noise, '(b c h w) (p q) -> b c (h p) (w q)',b=bs,c=ch,h=height//2,w=width//2,p=2,q=2)
        return noise

    def on_train_start(self):
        # Exponentional moving average (Use the fixed default DiffESM configs.)
        self.model.train()
        if self.ema is None:
            self.ema = EMA(self.model.parameters(), decay=0.9999)

    def on_before_zero_grad(self, optimizer):
        self.ema.update(self.model.parameters())
        return super().on_before_zero_grad(optimizer)

    def on_validation_start(self):
        self.current_stage = "val"
        self.model.eval()

        # Copy the existing model parameters and use the EMA parameters
        self.ema.store()
        self.ema.copy_to()
        # Run the original function in the parent class
        return super().on_validation_start()

    def on_validation_end(self):
        self.current_stage = "train"

        # Restore the original model parameters after evaluation
        self.ema.restore()
        return super().on_validation_end()

    def on_test_start(self):
        # Copy the existing model parameters and use the EMA parameters
        self.model.eval()
        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        self.current_stage = "test"

        # Run the original function in the parent class
        return super().on_test_start()

    def on_save_checkpoint(self, checkpoint: dict):
        checkpoint["ema"] = self.ema.state_dict()
        return super().on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint: dict):
        self.ema.load_state_dict(checkpoint["ema"])
        self.ema.to(self.device)
        return super().on_load_checkpoint(checkpoint)

    def setup(self, stage: str):
        super().setup(stage)
        if stage != "predict" and not isinstance(self.criterion, MSE_PixelFlow):
            self.criterion = MSE_PixelFlow()

    def setup_noise_scheduler(self):
        self.noise_scheduler: PyramidFlowMatchEulerDiscreteScheduler = instantiate(self.noise_scheduler_config)
