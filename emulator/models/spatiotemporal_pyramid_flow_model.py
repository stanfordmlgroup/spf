import math
import torch
import torch.nn.functional as F
from typing import Any, List, Optional
from einops import rearrange
from diffusers.utils.torch_utils import randn_tensor
from hydra.utils import instantiate

from emulator.models.pixelflow_basemodel import PixelFlowBaseModel
from emulator.visualization.visualize import create_temporal_predictions_figs

from torch.distributions.multivariate_normal import MultivariateNormal, _batch_mv

class MultivariateNormalWithGenerator(MultivariateNormal):
    """
    Same as your custom MultivariateNormal, but rsample() can take a torch.Generator.
    If no generator is provided, behavior is identical to the base class.
    """

    def rsample(
        self,
        sample_shape,
        generator: Optional[torch.Generator] = None,
    ):
        # If no generator is given, fall back to original behavior
        if generator is None:
            return super().rsample(sample_shape)

        # Otherwise, use the provided generator for deterministic noise
        shape = self._extended_shape(sample_shape)
        eps = torch.randn(
            shape,
            dtype=self.loc.dtype,
            device=self.loc.device,
            generator=generator,
        )
        return self.loc + _batch_mv(self._unbroadcasted_scale_tril, eps)


class SpatiotemporalPyramidFlow(PixelFlowBaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.is_temporal = True
        self.multi_scale = kwargs["configs"]["model"].get("same_scale_prob", 0) > 0

    def encode(self, x):
        embeddings = self.encoder(x)
        attention_mask = None
        return embeddings, attention_mask

    def training_step(self, batch: Any, batch_idx: int):

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

        no_time_embedding = self.config['model'].get("no_time_embedding", False)
        if "latent_time" in batch and not no_time_embedding:
            latent_time = batch["latent_time"]
        else:
            latent_time = None

        # DiT forward
        model_preds_list = self.model(
            sample=batch["noisy_latents_list"],
            timestep_ratio=batch["timesteps"],
            condition_sample=batch["cond_inputs_list"],
            latent_size=batch["latent_size"],
            latent_time=latent_time,
            class_labels=class_labels,
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

    def sample_block_noise(self, bs, ch, T, height, width, n_k, block_number, keep_scale, r=2, eps=1e-6):
        gamma = -1 / (n_k - 1)
        I = torch.eye(n_k, device=self.device)
        J = torch.ones(n_k, n_k, device=self.device)
        dist = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.zeros(n_k, device=self.device), I * (1 - gamma) - J * -gamma + eps * I
        )
        noise = dist.rsample((block_number,))
        if keep_scale:
            # Spatial only: (p,q) = (r,r)
            noise = rearrange(noise, '(b c t h w) (p q) -> b c t (h p) (w q)',
                            b=bs, c=ch, t=T, h=height//r, w=width//r, p=r, q=r)
        else:
            # Temporal+spatial: (t,p,q) = (T,r,r)
            noise = rearrange(noise, '(b c h w) (t p q) -> b c t (h p) (w q)',
                            b=bs, c=ch, t=T, h=height//r, w=width//r, p=r, q=r)
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

        static_condition = self.config['model'].get("static_condition", False)
        use_pathway = self.config['model'].get("use_pathway", False)
        uses_temporal_levels = "TemporalFlow" in self.noise_scheduler_config._target_
        no_time_embedding = self.config['model'].get("no_time_embedding", False)

        batch_size = len(X[timescale])

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
        timescales = self.datamodule_config.timescales
        assert timescale in timescales, f"Invalid target timescale: {timescale}"
        target_idx = timescales.index(timescale)

        ns_cfg = self.noise_scheduler_config.ns  # list or list-of-lists
        if uses_temporal_levels:
            # concatenated schedule: rows correspond to end-timescale (0..4)
            ns_vec = [int(v) for v in ns_cfg[target_idx]]
        else:
            ns_vec = [int(v) for v in ns_cfg]  # flat list length = num_stages-1

        assert len(ns_vec) == self.noise_scheduler_config.num_stages - 1

        latent_times = []
        for i in range(self.noise_scheduler_config.num_stages):
            if i == 0:
                latent_times.append(0)
            else:
                prev = latent_times[-1]
                latent_times.append(prev + 1 if prev < target_idx else prev)

        spatial_steps = []  # per-step spatial upsample factors r_k (length = stages-1)
        for k in range(self.noise_scheduler_config.num_stages - 1):
            prev_ts_idx = latent_times[k]
            this_ts_idx = latent_times[k + 1]
            same_scale  = (this_ts_idx == prev_ts_idx)
            if same_scale:
                t_factor = 1
            else:
                tgt_ts  = self.datamodule_config.timescales[this_ts_idx]
                t_factor = int(self.datamodule_config.output_seq_len[tgt_ts])

            v = int(ns_vec[k])
            assert v % t_factor == 0, f"ns[{k}]={v} not divisible by temporal factor {t_factor}"
            spatial_sq = v // t_factor
            r_k = int(round(math.sqrt(spatial_sq)))
            assert r_k * r_k == spatial_sq and r_k >= 1, f"Bad spatial factor at step {k}: ns={v}, t={t_factor}"
            spatial_steps.append(r_k)

        init_time = self.datamodule_config.output_seq_len[timescales[0]]
        init_factor = 1
        for r in spatial_steps:
            init_factor *= r
        height //= init_factor
        width  //= init_factor
        shape = (batch_size, num_vars, init_time, height, width)
        latents = randn_tensor(shape, device=self.device, dtype=torch.float32)

        # --- NEW: build the per-stage 'latent_time' argument for the model ---
        if use_pathway:
            # Overload: latent_time == path_id (0=DDD, 1=DYY, 2=DYM)
            latent_time_arg = [target_idx] * self.noise_scheduler_config.num_stages
        else:
            # Original semantics: latent_time == timescale id (0/1/2 per stage)
            latent_time_arg = latent_times

        for stage_idx in range(self.noise_scheduler_config.num_stages):
            if self.multi_scale or uses_temporal_levels:
                self.noise_scheduler.set_timesteps(
                    num_inference_steps[stage_idx],
                    stage_idx,
                    device=self.device,
                    temporal_level=target_idx
                )
            else:
                self.noise_scheduler.set_timesteps(
                    num_inference_steps[stage_idx],
                    stage_idx,
                    device=self.device,
                )
            Timesteps = self.noise_scheduler.timesteps
            ts_idx = int(latent_times[stage_idx])
            ts = self.stages[ts_idx]

            if stage_idx > 0:
                r = spatial_steps[stage_idx - 1]
                height *= r
                width  *= r

                prev_idx = latent_times[stage_idx - 1]
                keep_scale = (ts_idx == prev_idx)

                if keep_scale:
                    # Only spatial upsample
                    B, V, T, H, W = latents.shape
                    latents = rearrange(latents, 'b c t h w -> (b t) c h w')
                    latents = F.interpolate(latents, size=(height, width), mode='nearest')
                    latents = rearrange(latents, '(b t) c h w -> b c t h w', b=B, t=T)
                else:
                    # Slice and upsample in time + space
                    prev_ts = self.stages[prev_idx]
                    prev_Y_time = torch.stack(metadata[f"{prev_ts}_Y_time"], dim=1)
                    Y_time = torch.stack(metadata[f"{ts}_Y_time"], dim=1)
                    start_interval = prev_Y_time[:, 0]
                    current_interval = Y_time[:, 0]
                    indices = (current_interval - start_interval) // (10 if ts == "yearly" else 1)
                    latents = latents[torch.arange(batch_size), :, indices, :, :].unsqueeze(2)

                    T = self.datamodule_config.output_seq_len[ts]
                    latents = F.interpolate(latents, size=(T, height, width), mode='nearest')

                # Add noise (Eq. 20)
                if not uses_temporal_levels:
                    sigma_start_k = self.noise_scheduler.ori_start_sigmas[stage_idx]
                else:
                    sigma_start_k = self.noise_scheduler.ori_start_sigmas[target_idx][stage_idx]
                e_kp1 = 1.0 - sigma_start_k
                if self.multi_scale or uses_temporal_levels:
                    n_k = self.noise_scheduler_config.ns[target_idx][stage_idx-1]
                else:
                    n_k = self.noise_scheduler_config.ns[stage_idx-1]

                B, V, T, H, W = latents.shape
                if not keep_scale:
                    n = r * r * T
                    block_number = B * V * (H // r) * (W // r)
                else:
                    n = r * r
                    block_number = B * V * (H // r) * (W // r) * T
                assert n == n_k, f"n ({n}) must match value in ns ({n_k})"
                s_k = e_kp1 / (e_kp1 + math.sqrt(n_k) * (1 - e_kp1))
                alpha_k = (1 - s_k) * math.sqrt((n_k - 1) / n_k)
                scale_k = ((1 - s_k) + s_k * math.sqrt(n_k)) / math.sqrt(n_k)

                noise = self.sample_block_noise(*latents.shape, n_k=n_k, block_number=block_number, keep_scale=keep_scale, r=r)
                noise = noise.to(device=self.device, dtype=latents.dtype)
                latents = scale_k * latents + alpha_k * noise

            # Model inputs
            size_tensor = torch.tensor([kwargs["latent_size"][stage_idx]] * batch_size, dtype=torch.int32, device=self.device)
            time_tensor = torch.tensor([latent_time_arg[stage_idx]] * batch_size, dtype=torch.int32, device=self.device)

            if static_condition:
                stage_cond = [[X["monthly"].transpose(1, 2).contiguous()]]
            else:
                stage_cond = [[X[ts].transpose(1, 2).contiguous()]]

            for T in Timesteps:
                timestep = T.expand(latents.shape[0]).to(latents.dtype)

                noise_pred_list = self.model(
                    sample=[[latents]],
                    condition_sample=stage_cond,
                    class_labels=class_labels,
                    timestep_ratio=timestep,
                    latent_size=size_tensor,
                    latent_time=time_tensor if not no_time_embedding else None,
                )
                noise_pred = noise_pred_list[0]  # (B, C, T, H, W)

                latents = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=timestep,
                    sample=latents
                ).prev_sample

        latents = latents.transpose(1, 2)  # Swap back temporal and channel dimensions

        return latents.detach()

    def sample_block_noise_seeded(self, bs, ch, T, height, width, n_k, block_number, keep_scale, generator=None, r=2, eps=1e-6):
        gamma = -1 / (n_k - 1)
        I = torch.eye(n_k, device=self.device)
        J = torch.ones(n_k, n_k, device=self.device)
        dist = MultivariateNormalWithGenerator(
            torch.zeros(n_k, device=self.device), I * (1 - gamma) - J * -gamma + eps * I
        )
        noise = dist.rsample((block_number,), generator=generator)
        if keep_scale:
            # Spatial only: (p,q) = (r,r)
            noise = rearrange(noise, '(b c t h w) (p q) -> b c t (h p) (w q)',
                            b=bs, c=ch, t=T, h=height//r, w=width//r, p=r, q=r)
        else:
            # Temporal+spatial: (t,p,q) = (T,r,r)
            noise = rearrange(noise, '(b c h w) (t p q) -> b c t (h p) (w q)',
                            b=bs, c=ch, t=T, h=height//r, w=width//r, p=r, q=r)
        return noise


    def predict_modular(
        self,
        X,
        metadata=None,
        timescale: str = "monthly",
        start_stage: int = 0,
        stop_stage: Optional[int] = None,
        init_latents: Optional[torch.Tensor] = None,
        return_internal: bool = False,
        rng: Optional[torch.Generator] = None,   # NEW
        *args,
        **kwargs,
    ):
        # ---------------- figure out full-grid H/W as before ----------------
        if self.trainer.validating:
            height = self.trainer.train_dataloader.dataset.dataset.climate_model_datasets[0].grid.lon.shape[0]
            width = self.trainer.train_dataloader.dataset.dataset.climate_model_datasets[0].grid.lat.shape[0]
        else:
            height = self.trainer.datamodule.test_ds.climate_model_datasets[0].grid.lon.shape[0]
            width = self.trainer.datamodule.test_ds.climate_model_datasets[0].grid.lat.shape[0]

        H_full, W_full = height, width  # keep originals

        # ---------------- basic config ----------------
        if isinstance(self.num_inference_steps, int):
            num_inference_steps = [self.num_inference_steps] * self.noise_scheduler_config.num_stages
        else:
            num_inference_steps = self.num_inference_steps

        static_condition = self.config['model'].get("static_condition", False)
        use_pathway = self.config['model'].get("use_pathway", False)
        uses_temporal_levels = "TemporalFlow" in self.noise_scheduler_config._target_
        no_time_embedding = self.config['model'].get("no_time_embedding", False)

        num_stages = self.noise_scheduler_config.num_stages

        # --- NEW: slice range of stages ---
        if stop_stage is None:
            stop_stage = num_stages
        assert 0 <= start_stage < stop_stage <= num_stages, (
            f"Invalid stage range: start_stage={start_stage}, stop_stage={stop_stage}, "
            f"num_stages={num_stages}"
        )

        batch_size = len(X[timescale])

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
        timescales = self.datamodule_config.timescales
        assert timescale in timescales, f"Invalid target timescale: {timescale}"
        target_idx = timescales.index(timescale)

        # ---------------- ns_vec, latent_times, spatial_steps (unchanged logic) ----------------
        ns_cfg = self.noise_scheduler_config.ns  # list or list-of-lists
        if uses_temporal_levels:
            ns_vec = [int(v) for v in ns_cfg[target_idx]]
        else:
            ns_vec = [int(v) for v in ns_cfg]

        assert len(ns_vec) == num_stages - 1

        latent_times = []
        for i in range(num_stages):
            if i == 0:
                latent_times.append(0)
            else:
                prev = latent_times[-1]
                latent_times.append(prev + 1 if prev < target_idx else prev)

        # per-stage spatial upsample factors r_k (length = stages-1)
        spatial_steps = []
        for k in range(num_stages - 1):
            prev_ts_idx = latent_times[k]
            this_ts_idx = latent_times[k + 1]
            same_scale = (this_ts_idx == prev_ts_idx)
            if same_scale:
                t_factor = 1
            else:
                tgt_ts = self.datamodule_config.timescales[this_ts_idx]
                t_factor = int(self.datamodule_config.output_seq_len[tgt_ts])

            v = int(ns_vec[k])
            assert v % t_factor == 0, f"ns[{k}]={v} not divisible by temporal factor {t_factor}"
            spatial_sq = v // t_factor
            r_k = int(round(math.sqrt(spatial_sq)))
            assert r_k * r_k == spatial_sq and r_k >= 1, f"Bad spatial factor at step {k}: ns={v}, t={t_factor}"
            spatial_steps.append(r_k)

        # ---------------- init / resume latents ----------------
        timescale0 = timescales[0]
        init_time = self.datamodule_config.output_seq_len[timescale0]

        if init_latents is None:
            # fresh run: must start from stage 0
            assert start_stage == 0, (
                "If start_stage > 0 you must provide init_latents "
                "(latents at the beginning of start_stage)."
            )
            # start from coarsest spatial grid as before
            init_factor = 1
            for r in spatial_steps:
                init_factor *= r
            height = H_full // init_factor
            width = W_full // init_factor

            shape = (batch_size, num_vars, init_time, height, width)
            latents = randn_tensor(shape, device=self.device, dtype=torch.float32, generator=rng)
        else:
            # resuming: assume internal layout (B, C, T, H, W)
            latents = init_latents.to(self.device)
            assert latents.dim() == 5, "init_latents must have shape (B, C, T, H, W)"
            _, C_, T_, height, width = latents.shape
            # optional sanity checks
            if C_ != num_vars:
                raise ValueError(
                    f"init_latents has {C_} channels, expected {num_vars}. "
                    "Make sure you passed internal layout (B, C, T, H, W) as returned with return_internal=True."
                )

        # --- build per-stage 'latent_time' argument for the model ---
        if use_pathway:
            # Overload: latent_time == path_id (0=DDD, 1=DYY, 2=DYM)
            latent_time_arg = [target_idx] * num_stages
        else:
            # Original semantics: latent_time == timescale id (0/1/2 per stage)
            latent_time_arg = latent_times

        # ---------------- main stage loop (now sliceable) ----------------
        for stage_idx in range(start_stage, stop_stage):
            # set timesteps for this stage
            if self.multi_scale or uses_temporal_levels:
                self.noise_scheduler.set_timesteps(
                    num_inference_steps[stage_idx],
                    stage_idx,
                    device=self.device,
                    temporal_level=target_idx
                )
            else:
                self.noise_scheduler.set_timesteps(
                    num_inference_steps[stage_idx],
                    stage_idx,
                    device=self.device,
                )
            Timesteps = self.noise_scheduler.timesteps

            ts_idx = int(latent_times[stage_idx])
            ts = self.stages[ts_idx]

            # ----- spatial / temporal upsample + noise injection -----
            if stage_idx > 0:
                r = spatial_steps[stage_idx - 1]
                height *= r
                width *= r

                prev_idx = latent_times[stage_idx - 1]
                keep_scale = (ts_idx == prev_idx)

                if keep_scale:
                    # Only spatial upsample
                    B, V, Tcur, H, W = latents.shape
                    latents = rearrange(latents, 'b c t h w -> (b t) c h w')
                    latents = F.interpolate(latents, size=(height, width), mode='nearest')
                    latents = rearrange(latents, '(b t) c h w -> b c t h w', b=B, t=Tcur)
                    indices = None
                else:
                    # Slice and upsample in time + space
                    prev_ts = self.stages[prev_idx]
                    prev_Y_time = torch.stack(metadata[f"{prev_ts}_Y_time"], dim=1)
                    Y_time = torch.stack(metadata[f"{ts}_Y_time"], dim=1)
                    start_interval = prev_Y_time[:, 0]
                    current_interval = Y_time[:, 0]
                    indices = (current_interval - start_interval) // (10 if ts == "yearly" else 1)

                    latents = latents[torch.arange(batch_size), :, indices, :, :].unsqueeze(2)

                    Tnew = self.datamodule_config.output_seq_len[ts]
                    latents = F.interpolate(latents, size=(Tnew, height, width), mode='nearest')

                # Add noise (Eq. 20)
                if not uses_temporal_levels:
                    sigma_start_k = self.noise_scheduler.ori_start_sigmas[stage_idx]
                else:
                    sigma_start_k = self.noise_scheduler.ori_start_sigmas[target_idx][stage_idx]
                e_kp1 = 1.0 - sigma_start_k
                if self.multi_scale or uses_temporal_levels:
                    n_k = self.noise_scheduler_config.ns[target_idx][stage_idx - 1]
                else:
                    n_k = self.noise_scheduler_config.ns[stage_idx - 1]

                B, V, Tcur, H, W = latents.shape
                if not keep_scale:
                    n = r * r * Tcur
                    block_number = B * V * (H // r) * (W // r)
                else:
                    n = r * r
                    block_number = B * V * (H // r) * (W // r) * Tcur
                assert n == n_k, f"n ({n}) must match value in ns ({n_k})"
                s_k = e_kp1 / (e_kp1 + math.sqrt(n_k) * (1 - e_kp1))
                alpha_k = (1 - s_k) * math.sqrt((n_k - 1) / n_k)
                scale_k = ((1 - s_k) + s_k * math.sqrt(n_k)) / math.sqrt(n_k)

                noise = self.sample_block_noise_seeded(
                    *latents.shape, n_k=n_k,
                    block_number=block_number,
                    keep_scale=keep_scale, r=r,
                    generator=rng
                )
                noise = noise.to(device=self.device, dtype=latents.dtype)
                latents = scale_k * latents + alpha_k * noise

            # ----- model inputs for this stage -----
            size_tensor = torch.tensor(
                [kwargs["latent_size"][stage_idx]] * batch_size,
                dtype=torch.int32,
                device=self.device
            )
            time_tensor = torch.tensor(
                [latent_time_arg[stage_idx]] * batch_size,
                dtype=torch.int32,
                device=self.device
            )

            if static_condition:
                stage_cond = [[X["monthly"].transpose(1, 2).contiguous()]]
            else:
                stage_cond = [[X[ts].transpose(1, 2).contiguous()]]

            # diffusion steps within this stage
            for Tstep in Timesteps:
                timestep = Tstep.expand(latents.shape[0]).to(latents.dtype)

                noise_pred_list = self.model(
                    sample=[[latents]],
                    condition_sample=stage_cond,
                    class_labels=class_labels,
                    timestep_ratio=timestep,
                    latent_size=size_tensor,
                    latent_time=time_tensor if not no_time_embedding else None,
                )
                noise_pred = noise_pred_list[0]  # (B, C, T, H, W)

                latents = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=timestep,
                    sample=latents
                ).prev_sample

        # ---------------- return ----------------
        if return_internal:
            # For chaining stages: keep internal layout (B, C, T, H, W)
            return latents.detach(), indices

        # Original behaviour: swap back temporal and channel dims
        latents = latents.transpose(1, 2)  # (B, T, C, H, W)
        return latents.detach(), indices

    def _visualize_predictions(self, preds, targets, metadata, figure_name, num_samples=1, timescale="monthly"):
        if self.logger is None:
            return

        fig = create_temporal_predictions_figs(
            preds=preds,
            targets=targets,
            metadata=metadata,
            n=list(range(num_samples)),
            timescale=timescale,
        )

        self.logger.log_image(key=figure_name, images=[fig])

    def setup_noise_scheduler(self):
        self.stages = list(self.datamodule_config.output_seq_len.keys())
        scheduler_target = self.noise_scheduler_config._target_

        assert len(self.stages) == self.noise_scheduler_config.num_stages
        if "TemporalFlow" in scheduler_target:
            assert self.stages == ["decadal", "yearly", "monthly"]
            assert len(self.noise_scheduler_config.ns) == self.noise_scheduler_config.num_stages
            self.noise_scheduler = instantiate(self.noise_scheduler_config)
        else:
            raise ValueError(f"Unknown noise scheduler type: {scheduler_target}")
        print(f"Using noise scheduler: {self.noise_scheduler.__class__.__name__} in model")
