import torch
import math
import torch.nn.functional as F
from einops import rearrange
from hydra.utils import instantiate
from diffusers.utils.torch_utils import randn_tensor

from emulator.models.spatiotemporal_pyramid_flow_model import SpatiotemporalPyramidFlow
from emulator.visualization.visualize import create_temporal_predictions_figs


class SingleTimescaleSpatiotemporalPyramidFlow(SpatiotemporalPyramidFlow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.is_temporal = True

    def sample_block_noise(self, bs, ch, time, height, width, stage_idx, eps=1e-6):
        ns = self.noise_scheduler.config.ns
        gamma = -1 / (ns[stage_idx] - 1)
        height_ratio = 2
        width_ratio = 2
        n = height_ratio * width_ratio
        assert n == ns[stage_idx], f"n ({n}) must match ns[{stage_idx}] ({ns[stage_idx]})"
        dist = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.zeros(n),
            torch.eye(n) * (1 - gamma) - torch.ones(n, n) * -gamma + eps * torch.eye(n),
        )
        block_number = bs * ch * time * (height // height_ratio) * (width // width_ratio)
        noise = dist.rsample((block_number,))
        noise = rearrange(
            noise,
            "(b c t h w) (p q) -> b c t (h p) (w q)",
            b=bs,
            c=ch,
            t=time,
            h=height // height_ratio,
            w=width // width_ratio,
            p=height_ratio,
            q=width_ratio,
        )
        return noise

    def predict(self, X, metadata=None, timescale="monthly", *args, **kwargs):
        if self.trainer.validating:
            height = self.trainer.train_dataloader.dataset.dataset.climate_model_datasets[0].grid.lon.shape[0]
            width = self.trainer.train_dataloader.dataset.dataset.climate_model_datasets[0].grid.lat.shape[0]
        else:
            height = self.trainer.datamodule.test_ds.climate_model_datasets[0].grid.lon.shape[0]
            width = self.trainer.datamodule.test_ds.climate_model_datasets[0].grid.lat.shape[0]

        if isinstance(self.num_inference_steps, int):
            num_inference_steps = [self.num_inference_steps] * self.noise_scheduler_config.num_stages

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

        num_vars = len(self.datamodule_config.output_variables)

        time = self.datamodule_config.output_seq_len[timescale]
        spatial_stages = self.noise_scheduler_config.num_stages

        init_factor = 2 ** (spatial_stages - 1)
        batch_size = len(X[timescale])
        height, width = height // init_factor, width // init_factor

        shape = (batch_size, num_vars, time, height, width)
        latents = randn_tensor(shape, device=self.device, dtype=torch.float32)

        for stage_idx in range(self.noise_scheduler_config.num_stages):
            self.noise_scheduler.set_timesteps(
                num_inference_steps[stage_idx],
                stage_idx,
                device=self.device,
            )
            Timesteps = self.noise_scheduler.timesteps

            if stage_idx > 0:
                height, width = height * 2, width * 2
                latents = rearrange(latents, 'b c t h w -> (b t) c h w')
                latents = F.interpolate(latents, size=(height, width), mode='nearest')
                latents = rearrange(latents, '(b t) c h w -> b c t h w', b=batch_size, t=time)

                # Compute scale and alpha for the current stage
                sigma_start_k = self.noise_scheduler.ori_start_sigmas[stage_idx]
                e_kp1 = 1.0 - sigma_start_k 

                n_k = self.noise_scheduler_config.ns[stage_idx-1]
                s_k = e_kp1 / (e_kp1 + math.sqrt(n_k) * (1 - e_kp1))
                alpha_k = (1 - s_k) * math.sqrt((n_k - 1) / n_k)
                scale_k = ((1 - s_k) + s_k * math.sqrt(n_k)) / math.sqrt(n_k)

                # Add the corrective Gaussian noise
                noise  = self.sample_block_noise(*latents.shape, stage_idx=stage_idx-1)
                noise  = noise.to(device=self.device, dtype=latents.dtype)
                latents = scale_k * latents + alpha_k * noise     # Eq. (20)

            # Use latent sizes to identify the resolution stage
            # (All the same in the batch at inference time)
            size_tensor = torch.tensor(
                [kwargs["latent_size"][stage_idx]] * batch_size, dtype=torch.int32, device=self.device
            )
            latent_time = kwargs.get("latent_time")
            time_tensor = torch.tensor(
                [latent_time[stage_idx]] * batch_size, dtype=torch.int32, device=self.device
            ) if latent_time is not None else None

            cond_x = [[X[timescale].transpose(1, 2).contiguous()]]

            # euler
            for k, T in enumerate(Timesteps):
                timestep = T.expand(latents.shape[0]).to(latents.dtype)

                noise_pred_list = self.model(
                    [[latents]],
                    condition_sample=cond_x,
                    class_labels=class_labels,
                    timestep_ratio=timestep,
                    latent_size=size_tensor,
                    latent_time=time_tensor,
                )

                noise_pred = noise_pred_list[0]

                latents = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=timestep,
                    sample=latents
                ).prev_sample

        latents = latents.transpose(1, 2)  # Swap temporal and channel dimensions

        return latents.detach()

    def _visualize_predictions(self, preds, targets, metadata, figure_name, num_samples=1, timescale="monthly"):

        fig = create_temporal_predictions_figs(
            preds=preds,
            targets=targets,
            metadata=metadata,  
            n=list(range(num_samples)),
            timescale=timescale
        )

        self.logger.log_image(key=figure_name, images=[fig])

    def setup_noise_scheduler(self):
        self.stages = list(self.datamodule_config.output_seq_len.keys())
        assert self.stages == ["decadal", "yearly", "monthly"]
        assert len(self.stages) == self.noise_scheduler_config.num_stages
        assert len(self.noise_scheduler_config.ns) == self.noise_scheduler_config.num_stages - 1
        self.noise_scheduler = instantiate(self.noise_scheduler_config)
        print("Using scheduler:", self.noise_scheduler.__class__.__name__, "in model")
