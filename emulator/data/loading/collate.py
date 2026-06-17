import calendar
import math
import os
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models.embeddings import (
    get_2d_rotary_pos_embed,
)
from einops import rearrange
from torch.utils.data import get_worker_info
from torch.utils.data._utils.collate import default_collate


def compute_density_for_timestep_sampling(
    weighting_scheme: str, batch_size: int, logit_mean: float = None, logit_std: float = None, mode_scale: float = None
):
    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device="cpu")
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size,), device="cpu")
    return u

def collate_fn(examples, config, noise_scheduler_copy, is_training, **kwargs):
    X_batch = [item["X"] for item in examples]
    X_batch = default_collate(X_batch)
    cond_height, cond_width = X_batch.shape[-2:]
    cond_patch_size = config.model.params.encoder_patch_size

    metadata_batch = [item["metadata"] for item in examples]
    collated_metadata = {}
    for key in metadata_batch[0].keys():
        values = [m[key] for m in metadata_batch]
        collated_metadata[key] = default_collate(values)

    Y_batch = [torch.as_tensor(item["Y"]) for item in examples]
    Y_batch = torch.stack(Y_batch)
    if config.data.params.output_month is not None:
        # Slice output based on the specified month
        time_ind = list(calendar.month_name)[1:].index(config.data.params.output_month)
        Y_batch = Y_batch[:, [time_ind], :, :, :]
        # Average the input up to the specified month
        # For CO2 which is cumulative, take the value of the corresponding month
        co2_ind = config.data.params.input_variables.index("CO2")
        pre_co2_inds = range(0, co2_ind)
        post_co2_inds = range(co2_ind + 1, len(config.data.params.input_variables))
        X_batch = torch.cat([
            torch.mean(X_batch[:, :time_ind+1, pre_co2_inds, :, :], dim=1),
            X_batch[:, [time_ind], co2_ind, :, :],
            torch.mean(X_batch[:, :time_ind+1, post_co2_inds, :, :], dim=1)
        ], dim=1).unsqueeze(1)
    assert len(Y_batch.shape) == 5, "Expected output to have 5 dimensions: [batch, temporal, channel, latitude, longitude]"
    # Squeeze the temporal dimension
    pixel_values = Y_batch.squeeze(1)
    patch_size = config.model.params.patch_size
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    batch_size = len(examples)
    assert batch_size % config.scheduler.num_stages == 0, f"Batch size {batch_size} must be divisible by number of stages {config.scheduler.num_stages}"

    orig_height, orig_width = pixel_values.shape[-2:]

    sample_ratios = (1, ) * config.scheduler.num_stages
    stages = config.scheduler.num_stages
    dtype = pixel_values.dtype
    device = pixel_values.device
    assert batch_size % int(sum(sample_ratios)) == 0, f"Batch size {batch_size} must be divisible by {int(sum(sample_ratios))}"
    assert stages == len(sample_ratios), f"Number of stages {stages} must match the length of sample_ratios {len(sample_ratios)}"

    # Sample random noise per stage
    noise = torch.randn_like(pixel_values)
    height, width = noise.shape[-2], noise.shape[-1]
    noise_list = [noise]
    cur_noise = noise
    
    latents_list = [pixel_values]
    cur_latent = pixel_values
    for i_s in range(stages-1):
        height //= 2; width //= 2
        cur_noise = F.interpolate(cur_noise, size=(height, width), mode='bilinear') * 2
        noise_list.append(cur_noise)

        cur_latent = F.interpolate(cur_latent, size=(height, width), mode='bilinear')
        latents_list.append(cur_latent)

    # make sure from low res to high res
    noise_list = noise_list[::-1]
    latents_list = latents_list[::-1]

    column_size = int(sum(sample_ratios))
    batch_size_per_column  = batch_size // int(sum(sample_ratios))

    column_to_stage = {}
    i_sum = 0
    for i_s, column_num in enumerate(sample_ratios):
        for index in range(i_sum, i_sum + column_num):
            column_to_stage[index] = i_s
        i_sum += column_num

    noisy_latents_list = []
    ratios_list = []
    targets_list = []
    timesteps_list = []
    pos_embed_list = []
    cond_pos_embed_list = []
    seq_len_list = []
    latent_height_list = []
    training_steps = noise_scheduler_copy.config.num_train_timesteps

    # from low resolution to high resolution
    for index in range(column_size):
        # First prepare the trainable latent construction
        i_s = column_to_stage[index]
        clean_latent = latents_list[i_s][index::column_size]   # [bs, c, h, w]
        start_sigma = noise_scheduler_copy.start_sigmas[i_s]
        end_sigma = noise_scheduler_copy.end_sigmas[i_s]

        if i_s == 0:
            start_point = noise_list[i_s][index::column_size]
        else:
            # Get the upsampled latent
            last_clean_latent = latents_list[i_s-1][index::column_size]
            last_clean_latent = F.interpolate(last_clean_latent, size=(last_clean_latent.shape[-2] * 2, last_clean_latent.shape[-1] * 2), mode='nearest')
            start_point = start_sigma * noise_list[i_s][index::column_size] + (1 - start_sigma) * last_clean_latent
        
        if i_s == stages - 1:
            end_point = clean_latent
        else:
            end_point = end_sigma * noise_list[i_s][index::column_size] + (1 - end_sigma) * clean_latent

        # To sample a timestep
        u = compute_density_for_timestep_sampling(
            weighting_scheme='random',
            batch_size=batch_size_per_column,
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=1.29,
        )

        indices = (u * training_steps).long()   # Totally 1000 training steps per stage
        indices = indices.clamp(0, training_steps-1)
        timesteps = noise_scheduler_copy.timesteps_per_stage[i_s][indices].to(device=device, dtype=dtype)
        ratios = noise_scheduler_copy.sigmas_per_stage[i_s][indices].to(device=device, dtype=dtype)

        while len(ratios.shape) < start_point.ndim:
            ratios = ratios.unsqueeze(-1)

        # interpolate the latent
        noisy_latents = (ratios * start_point + (1 - ratios) * end_point).to(dtype)

        # The standard rectified flow matching objective
        targets = start_point - end_point

        end_height, end_width = clean_latent.shape[-2:]

        # Patchify latents and targets
        targets = rearrange(targets, 'b c (h ph) (w pw) -> (b h w) (c ph pw)', ph=patch_size, pw=patch_size)
        noisy_latents = rearrange(noisy_latents, 'b c (h ph) (w pw) -> (b h w) (c ph pw)', ph=patch_size, pw=patch_size)

        # Get the positional embeddings
        pos_embed = get_2d_rotary_pos_embed(
            embed_dim=config.model.params.attention_head_dim,
            crops_coords=((0, 0), (end_height // patch_size, end_width // patch_size)),
            grid_size=(end_height // patch_size, end_width // patch_size),
        )

        cond_pos_embed = get_2d_rotary_pos_embed(
            embed_dim=config.model.params.attention_head_dim,
            crops_coords=((0, 0), (cond_height // cond_patch_size, cond_width // cond_patch_size)),
            grid_size=(cond_height // cond_patch_size, cond_width // cond_patch_size),
        )

        # Calculate the sequence length and latent height
        seq_len = (end_height // patch_size) * (end_width // patch_size)
        latent_height = end_height // patch_size

        # Append the results to the lists
        noisy_latents_list.append(noisy_latents)
        ratios_list.append(ratios)
        timesteps_list.append(timesteps)
        targets_list.append(targets)     
        pos_embed_list.extend([pos_embed] * batch_size_per_column)
        cond_pos_embed_list.extend([cond_pos_embed] * batch_size_per_column)
        seq_len_list.extend([seq_len] * batch_size_per_column)
        latent_height_list.extend([latent_height] * batch_size_per_column)

    pixel_values = torch.cat(noisy_latents_list, dim=0).to(memory_format=torch.contiguous_format)
    target_values = torch.cat(targets_list, dim=0).to(memory_format=torch.contiguous_format)
    pos_embed = torch.cat([torch.stack(one_pos_emb, -1) for one_pos_emb in pos_embed_list], dim=0).float()
    cond_pos_embed = torch.cat([torch.stack(one_cond_pos_emb, -1) for one_cond_pos_emb in cond_pos_embed_list], dim=0).float()
    timesteps = torch.cat(timesteps_list, dim=0)
    latent_heights = torch.tensor(latent_height_list, dtype=torch.int32)
    cumsum_q_len = torch.cumsum(torch.tensor([0] + seq_len_list), 0).to(torch.int32)

    batch = {
        "cond_values": X_batch,
        "metadata": collated_metadata,
        "pixel_values": pixel_values,
        "pos_embed_q": pos_embed,
        "cumsum_q_len": cumsum_q_len,
        # Use the height sizes which still uniquely identifies the latent size
        "latent_size": latent_heights,
        "seqlen_list_q": seq_len_list,
        "pos_embed_k": cond_pos_embed,
        "timesteps": timesteps,
        "target_values": target_values,
        "seqlen_list_k": None
    }
    if not is_training:
        batch["X"] = X_batch
        batch["Y"] = Y_batch

    return batch

def collate_fn_single_timescale(
    examples,
    config,
    noise_scheduler_copy,
    output_seq_len,
    is_training: bool,
):
    timescale = config.data.params.single_timescale_target
    patch_size = config.model.params.patch_size
    stages = config.scheduler.num_stages
    target_time = output_seq_len[timescale]

    batch_size = len(examples)
    assert batch_size % stages == 0, f"Batch size {batch_size} must be divisible by num_stages {stages}"
    batch_size_per_column = batch_size // stages

    metadata_batch = [ex["metadata"] for ex in examples]
    collated_meta = {k: default_collate([m[k] for m in metadata_batch]) for k in metadata_batch[0].keys()}

    full_y = torch.stack([ex[timescale]["Y"] for ex in examples])
    full_x = torch.stack([ex[timescale]["X"] for ex in examples])
    dtype, device = full_y.dtype, full_y.device
    _, time, _, height, width = full_y.shape

    latents_list = [full_y]
    cond_list = [full_x]
    noise_list = [torch.randn_like(full_y)]

    cur_latent = full_y
    cur_noise = noise_list[0]
    for _ in range(stages - 1):
        height //= 2
        width //= 2

        cur_noise = rearrange(cur_noise, "b t c h w -> (b t) c h w")
        cur_noise = F.interpolate(cur_noise, size=(height, width), mode="bilinear") * math.sqrt(4)
        cur_noise = rearrange(cur_noise, "(b t) c h w -> b t c h w", t=time)

        cur_latent = rearrange(cur_latent, "b t c h w -> (b t) c h w")
        cur_latent = F.interpolate(cur_latent, size=(height, width), mode="bilinear")
        cur_latent = rearrange(cur_latent, "(b t) c h w -> b t c h w", t=time)

        noise_list.append(cur_noise)
        latents_list.append(cur_latent)
        cond_list.append(full_x)

    latents_list = latents_list[::-1]
    cond_list = cond_list[::-1]
    noise_list = noise_list[::-1]

    column_size = stages
    noisy_latents_list, targets_list, cond_inputs_list = [], [], []
    ratios_list, timesteps_list = [], []
    latent_height_list = []

    training_steps = noise_scheduler_copy.config.num_train_timesteps

    for col in range(column_size):
        i_s = col
        start_sigma = noise_scheduler_copy.start_sigmas[i_s]
        end_sigma = noise_scheduler_copy.end_sigmas[i_s]

        clean_latent = latents_list[i_s][col::column_size]

        if i_s == 0:
            start_point = noise_list[i_s][col::column_size]
        else:
            last_clean_latent = latents_list[i_s - 1][col::column_size]
            last_clean_latent = rearrange(last_clean_latent, "b t c h w -> (b t) c h w")
            last_clean_latent = F.interpolate(
                last_clean_latent,
                size=(last_clean_latent.shape[-2] * 2, last_clean_latent.shape[-1] * 2),
                mode="nearest",
            )
            last_clean_latent = rearrange(
                last_clean_latent,
                "(b t) c h w -> b t c h w",
                b=batch_size_per_column,
                t=target_time,
            )
            start_point = start_sigma * noise_list[i_s][col::column_size] + (1 - start_sigma) * last_clean_latent

        if i_s == stages - 1:
            end_point = clean_latent
        else:
            end_point = end_sigma * noise_list[i_s][col::column_size] + (1 - end_sigma) * clean_latent

        u = compute_density_for_timestep_sampling(
            weighting_scheme="random",
            batch_size=batch_size_per_column,
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=1.29,
        )
        idx = (u * training_steps).long().clamp(0, training_steps - 1)
        timesteps = noise_scheduler_copy.timesteps_per_stage[i_s][idx].to(device=device)
        ratios = noise_scheduler_copy.sigmas_per_stage[i_s][idx].to(device=device)

        while ratios.ndim < start_point.ndim:
            ratios = ratios.unsqueeze(-1)

        noisy_latents = (ratios * start_point + (1 - ratios) * end_point).to(dtype)
        targets = start_point - end_point

        X_cond = cond_list[i_s][col::column_size]

        end_h = clean_latent.shape[-2]
        latent_h = end_h // patch_size

        noisy_latents = noisy_latents.transpose(1, 2)  # [b, c, t, h, w]
        targets = targets.transpose(1, 2)
        cond_inputs = X_cond.transpose(1, 2)

        noisy_latents_list.append(noisy_latents)
        targets_list.append(targets)
        cond_inputs_list.append(cond_inputs)
        ratios_list.append(ratios.squeeze())
        timesteps_list.append(timesteps)
        latent_height_list.extend([latent_h] * batch_size_per_column)

    timesteps = torch.cat(timesteps_list, dim=0)
    latent_heights = torch.tensor(latent_height_list, dtype=torch.int32)

    collated_meta[f"{timescale}_input_stats"] = default_collate(
        [ex[timescale]["metadata"]["input_stats"] for ex in examples]
    )
    collated_meta[f"{timescale}_X_time"] = default_collate(
        [ex[timescale]["metadata"]["X_time"] for ex in examples]
    )
    collated_meta[f"{timescale}_stats"] = default_collate(
        [ex[timescale]["metadata"]["stats"] for ex in examples]
    )
    collated_meta[f"{timescale}_Y_time"] = default_collate(
        [ex[timescale]["metadata"]["Y_time"] for ex in examples]
    )

    batch = {
        "metadata": collated_meta,
        "latent_size": latent_heights,
        "timesteps": timesteps,
        "noisy_latents_list": noisy_latents_list,
        "cond_inputs_list": cond_inputs_list,
        "targets_list": targets_list,
    }

    if not is_training:
        batch["X"] = {timescale: default_collate([ex[timescale]["X"] for ex in examples])}
        batch["Y"] = {timescale: default_collate([ex[timescale]["Y"] for ex in examples])}

    return batch

class RandomPathGate:
    """
    Counts how many *collate* calls have occurred and only enables
    random pathway selection after a warmup. Also supports a probability ramp
    and a cadence (only re-sample every N steps).

    start_random_after : int  -> burn-in collate calls with canonical path
    ramp_steps         : int  -> ramp probability from 0 to max_prob over this many calls
    max_prob           : float-> cap probability of randomizing (<=1)
    choose_every       : int  -> only re-sample a new path every N calls, otherwise reuse
    seed               : int|None -> make draws reproducible across workers if desired
    phase_offset       : int  -> desynchronize workers by staggering the cadence
    """
    def __init__(
        self,
        start_random_after: int = 0,
        ramp_steps: int = 0,
        max_prob: float = 1.0,
        choose_every: int = 1,
        num_choices: int = 3,
        seed: int | None = None,
        phase_offset: int = 0,
    ):
        self.step = 0
        self.num_choices = num_choices
        self.last_choice = num_choices - 1  # default to monthly (2)
        self.start_random_after = int(start_random_after)
        self.ramp_steps = max(1, int(ramp_steps))
        self.max_prob = float(max_prob)
        self.choose_every = max(1, int(choose_every))
        self.phase_offset = int(phase_offset)

        self._rng = torch.Generator()
        if seed is not None:
            self._rng.manual_seed(int(seed))

    def _prob_now(self) -> float:
        if self.step < self.start_random_after:
            return 0.0
        t = self.step - self.start_random_after
        ramp = min(1.0, max(0.0, t / self.ramp_steps))
        return ramp * self.max_prob

    def should_randomize(self) -> bool:
        p = self._prob_now()
        if p <= 0.0:
            return False
        return torch.rand((), generator=self._rng).item() < p

    def sample_choice(self, device: torch.device, no_first_stage: bool) -> int:
        """
        Return {0,1,2}:
          0: decadal->decadal->decadal
          1: decadal->yearly->yearly
          2: decadal->yearly->monthly (canonical)
        Respects no_first_stage==True (no path 0).
        """
        if no_first_stage:
            return int(torch.randint(1, self.num_choices, (), generator=self._rng, device=device).item())
        return int(torch.randint(0, self.num_choices, (), generator=self._rng, device=device).item())

    def maybe_choose(self, device: torch.device, no_first_stage: bool) -> tuple[bool, int]:
        """
        Decide whether to use a randomized path this call, with cadence.
        Returns (use_random, choice_index).
        """
        self.step += 1
        use_random = self.should_randomize()
        # de-synchronize workers: stagger the cadence by phase_offset
        if use_random and ((self.step + self.phase_offset) % self.choose_every == 0):
            self.last_choice = self.sample_choice(device=device, no_first_stage=no_first_stage)
        return (use_random, self.last_choice)


# Global dict *within each process* (workers are separate processes, so this stays per-worker)
_WORKER_GATES: dict[int, RandomPathGate] = {}


def get_worker_path_gate(
    *,
    start_random_after=0,
    ramp_steps=0,
    max_prob=1.0,
    choose_every=1,
    num_choices=3,
) -> RandomPathGate:
    """
    Returns a RandomPathGate unique to the current DataLoader worker.
    Seeds it from torch.initial_seed() (unique per worker) and staggers phase.
    """
    wi = get_worker_info()
    wid = wi.id if wi is not None else -1  # main thread if no worker
    gate = _WORKER_GATES.get(wid)
    if gate is None:
        # Use PyTorch worker seed (unique per worker). Optionally offset by DDP RANK for extra safety.
        base_seed = torch.initial_seed()
        try:
            rank = int(os.environ.get("RANK", "0"))
        except ValueError:
            rank = 0
        seed = base_seed + (wid + 1) * 13 + rank * 100_000_007

        gate = RandomPathGate(
            start_random_after=start_random_after,
            ramp_steps=ramp_steps,
            max_prob=max_prob,
            choose_every=choose_every,
            num_choices=num_choices,
            seed=seed,
            phase_offset=(wid % max(1, choose_every)),  # stagger cadence
        )
        _WORKER_GATES[wid] = gate
    return gate

def collate_fn_spf(
    examples,
    config,
    noise_scheduler_copy,
    output_seq_len,
    is_training: bool,
):
    """
    Three-stage, double‑coupled collate:

      Stage 0: decadal  @ H/4 × W/4              (pure noise start)
      Stage 1: yearly   @ H/2 × W/2  (COUPLED #1: decadal → yearly + 2× spatial)
      Stage 2: monthly  @ H   × W    (COUPLED #2: yearly  → monthly + 2× spatial)

    Training‑objective logic (ratio/σ interpolation) is unchanged.

    Assumptions
    ----------
    • Each `example` dict has timescales: "decadal", "yearly", "monthly"
        – decadal: [Td, C, Hd, Wd]
        – yearly:  [Ty, C, Hy, Wy]
        – monthly: [Tm, C, Hm, Wm]
    • `config.scheduler.num_stages == 3`.
    """
    # ------------------------------------------------------------------#
    # Basic bookkeeping & metadata                                      #
    # ------------------------------------------------------------------#
    patch_size      = config.model.params.patch_size
    stages          = config.scheduler.num_stages
    same_scale_prob = config.model.params.same_scale_prob
    no_decadal      = config.model.params.no_decadal
    uses_temporal_levels       = config.scheduler.uses_temporal_levels
    use_pathway     = config.model.params.use_pathway
    assert stages == 3, f"double‑coupled (3‑stage) schedule requires stages==3, got {stages}."

    batch_size = len(examples)
    assert batch_size % stages == 0, (
        f"Batch size {batch_size} must be divisible by num_stages {stages}"
    )
    batch_size_per_column = batch_size // stages

    # --- stack metadata ------------------------------------------------#
    metadata_batch  = [ex["metadata"] for ex in examples]
    collated_meta   = {
        k: default_collate([m[k] for m in metadata_batch])
        for k in metadata_batch[0].keys()
    }

    # Monthly metadata
    collated_meta["monthly_input_stats"] = default_collate(
        [ex["monthly"]["metadata"]["input_stats"] for ex in examples]
    )
    collated_meta["monthly_X_time"] = default_collate(
        [ex["monthly"]["metadata"]["X_time"] for ex in examples]
    )
    collated_meta["monthly_stats"] = default_collate(
        [ex["monthly"]["metadata"]["stats"] for ex in examples]
    )
    collated_meta["monthly_Y_time"] = default_collate(
        [ex["monthly"]["metadata"]["Y_time"] for ex in examples]
    )

    # Yearly metadata
    collated_meta["yearly_input_stats"] = default_collate(
        [ex["yearly"]["metadata"]["input_stats"] for ex in examples]
    )
    collated_meta["yearly_X_time"] = default_collate(
        [ex["yearly"]["metadata"]["X_time"] for ex in examples]
    )
    collated_meta["yearly_stats"] = default_collate(
        [ex["yearly"]["metadata"]["stats"] for ex in examples]
    )
    collated_meta["yearly_Y_time"] = default_collate(
        [ex["yearly"]["metadata"]["Y_time"] for ex in examples]   # [B, 2]
    )

    # Decadal metadata
    collated_meta["decadal_input_stats"] = default_collate(
        [ex["decadal"]["metadata"]["input_stats"] for ex in examples]
    )
    collated_meta["decadal_X_time"] = default_collate(
        [ex["decadal"]["metadata"]["X_time"] for ex in examples]
    )
    collated_meta["decadal_stats"] = default_collate(
        [ex["decadal"]["metadata"]["stats"] for ex in examples]
    )
    collated_meta["decadal_Y_time"] = default_collate(
        [ex["decadal"]["metadata"]["Y_time"] for ex in examples]  # [B, 2]
    )

    # Per-stage year ranges (stage-wise): [decadal, yearly, monthly]
    decadal_B2 = torch.stack(collated_meta["decadal_Y_time"]).T
    yearly_B2  = torch.stack(collated_meta["yearly_Y_time"]).T
    monthly_B2 = torch.stack(collated_meta["monthly_Y_time"]).T

    # ------------------------------------------------------------------#
    # Stack full‑resolution tensors                                     #
    # ------------------------------------------------------------------#
    full_Y = torch.stack([ex["monthly"]["Y"] for ex in examples])  # [B, Tm, C, H, W]
    full_X = torch.stack([ex["monthly"]["X"] for ex in examples])  # [B, Tm, C, H, W]

    full_Y_year = torch.stack([ex["yearly"]["Y"] for ex in examples])   # [B, Ty, C, Hy, Wy]
    full_X_year = torch.stack([ex["yearly"]["X"] for ex in examples])   # [B, Ty, C, Hy, Wy]

    full_Y_dec  = torch.stack([ex["decadal"]["Y"] for ex in examples])  # [B, Td, C, Hd, Wd]
    full_X_dec  = torch.stack([ex["decadal"]["X"] for ex in examples])  # [B, Td, C, Hd, Wd]

    # canonical dtype / device
    dtype, device = full_Y.dtype, full_Y.device
    B, Tm_full, C, Hm, Wm = full_Y.shape

    # Timescale vocabulary and indices
    TS = ["decadal", "yearly", "monthly"]
    ts2idx = {"decadal": 0, "yearly": 1, "monthly": 2}

    # Choose timescale for each stage: stage 0, 1, 2
    if is_training and same_scale_prob > 0.0:
        # sample one of the 3 paths with equal prob
        #if not no_decadal:
        #    choice = torch.randint(0, 3, (), device=device).item()
        #else:
        #    choice = torch.randint(1, 3, (), device=device).item()
        gate = get_worker_path_gate(
            start_random_after=config.model.params.multiscale_warmup,
            ramp_steps=config.model.params.multiscale_ramp,
        )
        use_random, gate_choice = gate.maybe_choose(
            device=device,
            no_first_stage=no_decadal
        )
        choice = gate_choice if use_random else ts2idx["monthly"]

        if choice == 0:
            # decadal -> decadal -> decadal
            ts_stage = [ts2idx["decadal"], ts2idx["decadal"], ts2idx["decadal"]]
        elif choice == 1:
            # decadal -> yearly -> yearly
            ts_stage = [ts2idx["decadal"], ts2idx["yearly"], ts2idx["yearly"]]
        else:  # choice == 2
            # decadal -> yearly -> monthly
            ts_stage = [ts2idx["decadal"], ts2idx["yearly"], ts2idx["monthly"]]
    else:
        # Always temporally transition at evaluation
        ts_stage = [ts2idx["decadal"], ts2idx["yearly"], ts2idx["monthly"]]
        choice = ts2idx["monthly"]

    def _B2(ts_idx):
        return (
            decadal_B2 if ts_idx == ts2idx["decadal"]
            else yearly_B2 if ts_idx == ts2idx["yearly"]
            else monthly_B2
        )
    y_years = torch.stack([_B2(ts_idx) for ts_idx in ts_stage], dim=0)  # [3, B, 2]

    # ------------------------------------------------------------------#
    # Monthly: final stage (no pyramid needed in 3‑stage schedule)      #
    # ------------------------------------------------------------------#
    mon_latent_fin  = full_Y
    mon_cond_fin    = full_X
    mon_noise_fin   = torch.randn_like(full_Y)

    # ------------------------------------------------------------------#
    # Yearly @ H/2 × W/2  (one 2× downsample from full)                 #
    # ------------------------------------------------------------------#
    Ty = full_Y_year.shape[1]
    height_y, width_y = full_Y_year.shape[-2:]
    cur_noise_y = torch.randn_like(full_Y_year)
    cur_latent_y = full_Y_year
    for _ in range(1):  # one halving: → H/2 × W/2 (relative to yearly full)
        height_y //= 2; width_y //= 2
        cur_noise_y = rearrange(cur_noise_y, 'b t c h w -> (b t) c h w')
        cur_noise_y = F.interpolate(cur_noise_y, size=(height_y, width_y), mode='bilinear') * math.sqrt(2*2)
        cur_noise_y = rearrange(cur_noise_y, '(b t) c h w -> b t c h w', t=Ty)

        cur_latent_y = rearrange(cur_latent_y, 'b t c h w -> (b t) c h w')
        cur_latent_y = F.interpolate(cur_latent_y, size=(height_y, width_y), mode='bilinear')
        cur_latent_y = rearrange(cur_latent_y, '(b t) c h w -> b t c h w', t=Ty)

    # ------------------------------------------------------------------#
    # Decadal @ H/4 × W/4  (two halvings)                               #
    # ------------------------------------------------------------------#
    Td = full_Y_dec.shape[1]
    height_d, width_d = full_Y_dec.shape[-2:]
    cur_noise_d = torch.randn_like(full_Y_dec)
    cur_latent_d = full_Y_dec
    for _ in range(2):  # two halvings: → H/4 × W/4 (relative to decadal full)
        height_d //= 2; width_d //= 2
        cur_noise_d = rearrange(cur_noise_d, 'b t c h w -> (b t) c h w')
        cur_noise_d = F.interpolate(cur_noise_d, size=(height_d, width_d), mode='bilinear') * math.sqrt(2*2)
        cur_noise_d = rearrange(cur_noise_d, '(b t) c h w -> b t c h w', t=Td)

        cur_latent_d = rearrange(cur_latent_d, 'b t c h w -> (b t) c h w')
        cur_latent_d = F.interpolate(cur_latent_d, size=(height_d, width_d), mode='bilinear')
        cur_latent_d = rearrange(cur_latent_d, '(b t) c h w -> b t c h w', t=Td)

    # ------------------------------------------------------------------#
    # For multiscale training
    # ------------------------------------------------------------------#
    # --- Yearly @ H (no downsample) for stage 2 when path is ... -> yearly -> yearly
    Ty = full_Y_year.shape[1]
    lat_y_fine  = full_Y_year
    cond_y_fine = full_X_year
    noise_y_fine = torch.randn_like(full_Y_year)

    # --- Decadal @ H/2 (one halving) for stage 1 when path is decadal -> decadal
    height_d1, width_d1 = full_Y_dec.shape[-2:]
    lat_d_coarse  = full_Y_dec
    noise_d_coarse = torch.randn_like(full_Y_dec)
    # 1 halving (→ H/2)
    height_d1 //= 2; width_d1 //= 2
    noise_d_coarse = rearrange(noise_d_coarse, 'b t c h w -> (b t) c h w')
    noise_d_coarse = F.interpolate(noise_d_coarse, size=(height_d1, width_d1), mode='bilinear') * math.sqrt(2*2)
    noise_d_coarse = rearrange(noise_d_coarse, '(b t) c h w -> b t c h w', t=full_Y_dec.shape[1])

    lat_d_coarse = rearrange(lat_d_coarse, 'b t c h w -> (b t) c h w')
    lat_d_coarse = F.interpolate(lat_d_coarse, size=(height_d1, width_d1), mode='bilinear')
    lat_d_coarse = rearrange(lat_d_coarse, '(b t) c h w -> b t c h w', t=full_Y_dec.shape[1])
    cond_d_coarse = full_X_dec  # conditioning stays at native decadal grid

    # --- Decadal @ H (no downsample) for stage 2 when path is decadal -> decadal -> decadal
    lat_d_fine   = full_Y_dec
    cond_d_fine  = full_X_dec
    noise_d_fine = torch.randn_like(full_Y_dec)

    # ------------------------------------------------------------------#
    # Per‑stage lists (3 stages exactly)                                 #
    # ------------------------------------------------------------------#
    # Spatial level per stage: 0→H/4, 1→H/2, 2→H
    # We choose the (timescale, level) tensors that match the sampled path.
    latents_list, cond_list, noise_list = [], [], []

    for i_s, ts_idx in enumerate(ts_stage):
        if ts_idx == ts2idx["decadal"]:
            if i_s == 0:      # H/4
                latents_list.append(cur_latent_d);  cond_list.append(full_X_dec);  noise_list.append(cur_noise_d)
            elif i_s == 1:    # H/2
                latents_list.append(lat_d_coarse);  cond_list.append(cond_d_coarse); noise_list.append(noise_d_coarse)
            else:             # i_s == 2, H
                latents_list.append(lat_d_fine);    cond_list.append(cond_d_fine);   noise_list.append(noise_d_fine)

        elif ts_idx == ts2idx["yearly"]:
            if i_s == 1:      # H/2
                latents_list.append(cur_latent_y);  cond_list.append(full_X_year);   noise_list.append(cur_noise_y)
            else:             # i_s == 2, H
                latents_list.append(lat_y_fine);    cond_list.append(cond_y_fine);   noise_list.append(noise_y_fine)

        else:  # monthly (only appears at i_s == 2 in allowed paths)
            latents_list.append(mon_latent_fin);     cond_list.append(mon_cond_fin); noise_list.append(mon_noise_fin)

    # ------------------------------------------------------------------#
    # Column/stage mapping (sample_ratios = 1 per stage)                #
    # ------------------------------------------------------------------#
    column_to_stage = {i: i for i in range(stages)}
    column_size     = stages

    # holders for concatenation
    noisy_latents_list, targets_list, cond_inputs_list = [], [], []
    ratios_list, timesteps_list, timescale_list = [], [], []
    latent_height_list = []

    training_steps = noise_scheduler_copy.config.num_train_timesteps

    # ------------------------------------------------------------------#
    # Loop over columns (= stages)                                      #
    # ------------------------------------------------------------------#
    for col in range(column_size):
        i_s = column_to_stage[col]          # 0,1,2

        clean_latent = latents_list[i_s][col::column_size]

        if i_s == 0:
            # Coarsest (decadal H/4×W/4) → start at pure noise
            start_point = noise_list[i_s][col::column_size]
            this_ts_idx = 0

        else:
            last_clean_latent = latents_list[i_s - 1][col::column_size]

            # New multiscale implementation
            prev_ts_idx = ts_stage[i_s - 1]
            this_ts_idx = ts_stage[i_s]

            if prev_ts_idx == this_ts_idx:
                # --- same-scale: spatial-only 2× ---
                last_clean_latent = rearrange(last_clean_latent, 'b t c h w -> (b t) c h w')
                last_clean_latent = F.interpolate(
                    last_clean_latent,
                    size=(clean_latent.shape[-2], clean_latent.shape[-1]),  # land on this stage's H,W
                    mode='nearest'
                )
                last_clean_latent = rearrange(
                    last_clean_latent, '(b t) c h w -> b t c h w',
                    b=batch_size_per_column, t=clean_latent.shape[1]        # keep previous T
                )

            else:
                # --- coupled temporal+spatial ---
                prev_y = y_years[i_s - 1][col::column_size]   # [Bpc, 2]
                this_y = y_years[i_s][col::column_size]       # [Bpc, 2]
                start_interval  = prev_y[:, 0]
                current_year    = this_y[:, 0]

                if prev_ts_idx == ts2idx["decadal"] and this_ts_idx == ts2idx["yearly"]:
                    idx = ((current_year - start_interval) // 10).long()   # decade index
                elif prev_ts_idx == ts2idx["yearly"] and this_ts_idx == ts2idx["monthly"]:
                    idx = (current_year - start_interval).long()           # year index
                else:
                    raise AssertionError("Invalid temporal transition per schedule")

                # Select time index -> length-1
                last_clean_latent = last_clean_latent[torch.arange(last_clean_latent.shape[0]), idx][:, None]  # [Bpc, 1, C, Hprev, Wprev]

                # Upsample in (t,h,w) to this stage's target (T, H, W)
                this_T  = clean_latent.shape[1]
                tgt_h, tgt_w = clean_latent.shape[-2:]
                last_clean_latent = rearrange(last_clean_latent, 'b t c h w -> (b c) t h w')
                last_clean_latent = F.interpolate(
                    last_clean_latent.unsqueeze(1),
                    size=(this_T, tgt_h, tgt_w),
                    mode='nearest'
                ).squeeze(1)
                last_clean_latent = rearrange(last_clean_latent, '(b c) t h w -> b t c h w',
                                            b=batch_size_per_column, c=clean_latent.shape[2])

            if not uses_temporal_levels:
                start_sigma = noise_scheduler_copy.start_sigmas[i_s]
            else:
                start_sigma = noise_scheduler_copy.start_sigmas[choice][i_s]
            start_point = start_sigma * noise_list[i_s][col::column_size] + (1 - start_sigma) * last_clean_latent

        # End‑point selection identical to baseline
        if i_s == stages - 1:
            end_point = clean_latent
        else:
            if not uses_temporal_levels:
                end_sigma = noise_scheduler_copy.end_sigmas[i_s]
            else:
                end_sigma = noise_scheduler_copy.end_sigmas[choice][i_s]
            end_point = end_sigma * noise_list[i_s][col::column_size] + (1 - end_sigma) * clean_latent

        # ------------------------------------------------------------------#
        # Draw timesteps / ratios for RF objective                          #
        # ------------------------------------------------------------------#
        u = compute_density_for_timestep_sampling(
            weighting_scheme='random',
            batch_size=batch_size_per_column,
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=1.29,
        )
        idx       = (u * training_steps).long().clamp(0, training_steps - 1)
        if not uses_temporal_levels:
            timesteps = noise_scheduler_copy.timesteps_per_stage[i_s][idx].to(device=device)
            ratios    = noise_scheduler_copy.sigmas_per_stage[i_s][idx].to(device=device)
        else:
            timesteps = noise_scheduler_copy.timesteps_per_stage[choice][i_s][idx].to(device=device)
            ratios    = noise_scheduler_copy.sigmas_per_stage[choice][i_s][idx].to(device=device)

        while ratios.ndim < start_point.ndim:
            ratios = ratios.unsqueeze(-1)

        noisy_latents = (ratios * start_point + (1 - ratios) * end_point).to(dtype)
        targets       = start_point - end_point

        # ------------------------------------------------------------------#
        # Patchify latents / targets & conditioning tensors                 #
        # ------------------------------------------------------------------#
        end_h, end_w   = clean_latent.shape[-2:]
        latent_h  = end_h // patch_size

        X_cond = cond_list[i_s][col::column_size]
        cond_inputs = X_cond
        noisy_latents = noisy_latents.transpose(1, 2)  # [b, c, t, h, w]
        targets = targets.transpose(1, 2)
        cond_inputs = cond_inputs.transpose(1, 2)      # [b, c, t, h, w]

        # push to lists
        noisy_latents_list.append(noisy_latents)
        targets_list.append(targets)
        cond_inputs_list.append(cond_inputs)
        latent_height_list.extend([latent_h] * batch_size_per_column)
        ratios_list.append(ratios.squeeze())
        timesteps_list.append(timesteps)
        if use_pathway:
            timescale_list.extend([choice] * batch_size_per_column)
        else:
            timescale_list.extend([ts_stage[i_s]] * batch_size_per_column)

    # ------------------------------------------------------------------#
    # Final concatenation & tensorization                               #
    # ------------------------------------------------------------------#
    timesteps      = torch.cat(timesteps_list,     dim=0)
    latent_heights = torch.tensor(latent_height_list, dtype=torch.int32)
    latent_timescales = torch.tensor(timescale_list, dtype=torch.int32)

    # ------------------------------------------------------------------#
    # Assemble batch dict                                               #
    # ------------------------------------------------------------------#
    batch = {
        "metadata":        collated_meta,
        "latent_size":     latent_heights,
        "timesteps":       timesteps,
        "latent_time":     latent_timescales
    }

    batch["noisy_latents_list"] = noisy_latents_list
    batch["cond_inputs_list"] = cond_inputs_list
    batch["targets_list"] = targets_list

    if not is_training:
        batch["X"] = {
            "monthly": default_collate([ex["monthly"]["X"] for ex in examples]),
            "yearly":  default_collate([ex["yearly"]["X"]  for ex in examples]),
            "decadal": default_collate([ex["decadal"]["X"] for ex in examples]),
        }
        batch["Y"] = {
            "monthly": default_collate([ex["monthly"]["Y"] for ex in examples]),
            "yearly":  default_collate([ex["yearly"]["Y"]  for ex in examples]),
            "decadal": default_collate([ex["decadal"]["Y"] for ex in examples]),
        }

    return batch

def get_pyramid_latent(x, stage_num):
    # x is the origin vae latent
    vae_latent_list = []
    vae_latent_list.append(x)

    # x is (B, C, T, H, W)

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


class StageSampler:
    """
    Samples how many temporal “units” each pyramid stage should consume on
    this optimisation step.  Keeps an internal step counter so it can march
    through a schedule (round-robin, random, curriculum, …).

    Parameters
    ----------
    max_temporal_length : int
        Longest sequence you ever expect at the *finest* resolution.
    frame_per_unit : int, default 1
        How many frames constitute one “unit”.
    """

    def __init__(
        self,
        max_temporal_length: int,
        frame_per_unit: int = 1,
    ):
        self.max_units_total = 1 + (max_temporal_length - 1) // frame_per_unit
        self.frame_per_unit = frame_per_unit

        # internal mutable state
        self._step = 0

    # --------------------------------------------------------------------- #
    #  The main callable – returns [low_res_units, mid_res_units, hi_res_units]
    # --------------------------------------------------------------------- #
    def __call__(self, num_stages: int, max_units: int | None = None) -> List[int]:
        self._step += 1

        # ------------------------------------------------------------------ #
        # Decide how many “high-res units” to expose this step
        # ------------------------------------------------------------------ #
        usable_units = self.max_units_total
        if max_units is not None:
            usable_units = min(usable_units, max_units)

        cur_high = (self._step - 1) % usable_units + 1

        # Mirror rule from your original code: mid & low get the complement
        cur_mid = max(1 + usable_units - cur_high, 1)
        cur_low = cur_mid

        length_list = [cur_low, cur_mid, cur_high]
        assert len(length_list) == num_stages, "num_stages must match StageSampler output"

        return length_list


def collate_fn_pyramidal(examples, config, noise_scheduler_copy, sampler, output_seq_len, is_training):
    """
    add the noise for each pyramidal stage, used for AR video training with temporal pyramid
    Params:
        latent_list: [low_res, mid_res, high_res] The vae latents of all stages
        sample_ratios: The proportion of each stage in the training batch
    """
    X_batch = [item["X"] for item in examples]
    X_batch = default_collate(X_batch)

    Y_batch = [torch.as_tensor(item["Y"]) for item in examples]
    Y_batch = default_collate(Y_batch)

    metadata_batch = [item["metadata"] for item in examples]
    collated_metadata = {}
    for key in metadata_batch[0].keys():
        values = [m[key] for m in metadata_batch]
        collated_metadata[key] = default_collate(values)

    num_stages = config.scheduler.num_stages
    video = torch.stack([item["Y"] for item in examples], dim=0)  # [B, T, C, H, W]
    video = video.transpose(1, 2)  # [B, C, T, H, W]
    T = video.shape[2]
    latents_list = get_pyramid_latent(video, num_stages - 1)

    sample_ratios = [1, 1, 1]
    stages = num_stages
    tot_samples = latents_list[0].shape[0]
    device = latents_list[0].device
    dtype = latents_list[0].dtype

    assert tot_samples % (int(sum(sample_ratios))) == 0
    assert stages == len(sample_ratios)

    noise = torch.randn_like(latents_list[-1])
    t = noise.shape[2]

    frame_per_unit = 1
    corrupt_ratio = 1/3

    # To allocate the temporal length of each stage, ensuring the sum == constant
    max_units = 1 + (t - 1) // frame_per_unit

    # num_units_per_stage = self.sample_stage_length(stages, max_units=max_units)   # [The unit number of each stage]
    num_units_per_stage = sampler(num_stages=stages, max_units=max_units)

    height, width = noise.shape[-2], noise.shape[-1]
    noise_list = [noise]
    cur_noise = noise
    for i_s in range(stages-1):
        height //= 2;width //= 2
        cur_noise = rearrange(cur_noise, 'b c t h w -> (b t) c h w')
        cur_noise = F.interpolate(cur_noise, size=(height, width), mode='bilinear') * 2
        cur_noise = rearrange(cur_noise, '(b t) c h w -> b c t h w', t=t)
        noise_list.append(cur_noise)

    noise_list = list(reversed(noise_list))   # make sure from low res to high res

    # To calculate the batchsize and column size
    batch_size = tot_samples // int(sum(sample_ratios))
    column_size = int(sum(sample_ratios))

    column_to_stage = {}
    i_sum = 0
    for i_s, column_num in enumerate(sample_ratios):
        for index in range(i_sum, i_sum + column_num):
            column_to_stage[index] = i_s
        i_sum += column_num

    noisy_latents_list = []
    ratios_list = []
    targets_list = []
    timesteps_list = []
    training_steps = config.scheduler.num_train_timesteps

    # from low resolution to high resolution
    for index in range(column_size):
        # First prepare the trainable latent construction
        i_s = column_to_stage[index]
        clean_latent = latents_list[i_s][index::column_size]   # [bs, c, t, h, w]
        last_clean_latent = None if i_s == 0 else latents_list[i_s-1][index::column_size]
        start_sigma = noise_scheduler_copy.start_sigmas[i_s]
        end_sigma = noise_scheduler_copy.end_sigmas[i_s]

        if i_s == 0:
            start_point = noise_list[i_s][index::column_size]
        else:
            # Get the upsampled latent
            last_clean_latent = rearrange(last_clean_latent, 'b c t h w -> (b t) c h w')
            last_clean_latent = F.interpolate(last_clean_latent, size=(last_clean_latent.shape[-2] * 2, last_clean_latent.shape[-1] * 2), mode='nearest')
            last_clean_latent = rearrange(last_clean_latent, '(b t) c h w -> b c t h w', t=t)
            start_point = start_sigma * noise_list[i_s][index::column_size] + (1 - start_sigma) * last_clean_latent

        if i_s == stages - 1:
            end_point = clean_latent
        else:
            end_point = end_sigma * noise_list[i_s][index::column_size] + (1 - end_sigma) * clean_latent

        # To sample a timestep
        u = compute_density_for_timestep_sampling(
            weighting_scheme='random',
            batch_size=batch_size,
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=1.29,
        )

        indices = (u * training_steps).long()   # Totally 1000 training steps per stage
        indices = indices.clamp(0, training_steps-1)
        timesteps = noise_scheduler_copy.timesteps_per_stage[i_s][indices].to(device=device)
        ratios = noise_scheduler_copy.sigmas_per_stage[i_s][indices].to(device=device)

        while len(ratios.shape) < start_point.ndim:
            ratios = ratios.unsqueeze(-1)

        # interpolate the latent
        noisy_latents = ratios * start_point + (1 - ratios) * end_point

        # The flow matching object
        target_latents = start_point - end_point

        # pad the noisy previous
        num_units = num_units_per_stage[i_s]
        num_units = min(num_units, 1 + (t - 1) // frame_per_unit)
        actual_frames = 1 + (num_units - 1) * frame_per_unit

        noisy_latents = noisy_latents[:, :, :actual_frames]
        target_latents = target_latents[:, :, :actual_frames]

        clean_latent = clean_latent[:, :, :actual_frames]
        stage_noise = noise_list[i_s][index::column_size][:, :, :actual_frames]

        # only the last latent takes part in training
        noisy_latents = noisy_latents[:, :, -frame_per_unit:] 
        target_latents = target_latents[:, :, -frame_per_unit:]

        last_cond_noisy_sigma = torch.rand(size=(batch_size,), device=device) * corrupt_ratio

        if num_units == 1:
            stage_input = [noisy_latents.to(dtype)]
        else:
            # add the random noise for the last cond clip
            last_cond_latent = clean_latent[:, :, -(2*frame_per_unit):-frame_per_unit]

            while len(last_cond_noisy_sigma.shape) < last_cond_latent.ndim:
                last_cond_noisy_sigma = last_cond_noisy_sigma.unsqueeze(-1)

            # We adding some noise to corrupt the clean condition
            last_cond_latent = last_cond_noisy_sigma * torch.randn_like(last_cond_latent) + (1 - last_cond_noisy_sigma) * last_cond_latent

            # concat the corrupted condition and the input noisy latents
            stage_input = [noisy_latents.to(dtype), last_cond_latent.to(dtype)]

            cur_unit_num = 2
            cur_stage = i_s

            while cur_unit_num < num_units:
                cur_stage = max(cur_stage - 1, 0)
                if cur_stage == 0:
                    break
                cur_unit_num += 1
                cond_latents = latents_list[cur_stage][index::column_size][:, :, :actual_frames]
                cond_latents = cond_latents[:, :, -(cur_unit_num * frame_per_unit) : -((cur_unit_num - 1) * frame_per_unit)]
                cond_latents = last_cond_noisy_sigma * torch.randn_like(cond_latents)  + (1 - last_cond_noisy_sigma) * cond_latents
                stage_input.append(cond_latents.to(dtype))

            if cur_stage == 0 and cur_unit_num < num_units:
                cond_latents = latents_list[0][index::column_size][:, :, :actual_frames]
                cond_latents = cond_latents[:, :, :-(cur_unit_num * frame_per_unit)]

                cond_latents = last_cond_noisy_sigma * torch.randn_like(cond_latents)  + (1 - last_cond_noisy_sigma) * cond_latents
                stage_input.append(cond_latents.to(dtype))

        stage_input = list(reversed(stage_input))
        noisy_latents_list.append(stage_input)
        ratios_list.append(ratios.to(dtype))
        timesteps_list.append(timesteps.to(dtype))
        targets_list.append(target_latents)     # The standard rectified flow matching objective

    timesteps = torch.cat([timestep.unsqueeze(-1) for timestep in timesteps_list], dim=-1)
    timesteps = timesteps.reshape(-1)

    batch = {
        "noisy_latents_list": noisy_latents_list,
        "targets_list": targets_list,
        "timesteps": timesteps,
        "X": X_batch,
        "Y": Y_batch,
        "metadata": collated_metadata,
    }

    return batch
