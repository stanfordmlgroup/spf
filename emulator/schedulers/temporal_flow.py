import math
import torch
import numpy as np
from typing import Optional, Tuple, Union, List, Sequence, Dict
from diffusers.configuration_utils import register_to_config, ConfigMixin
from diffusers.schedulers.scheduling_utils import SchedulerMixin

from emulator.schedulers.pyramidal_flow import PyramidFlowMatchEulerDiscreteScheduler, FlowMatchEulerDiscreteSchedulerOutput


class SingleTimescaleFlowMatchEulerDiscreteScheduler(PyramidFlowMatchEulerDiscreteScheduler):
    @register_to_config
    def __init__(
        self,
        ns,
        num_stages: int = 3,
        stage_range: List = [0, 1/3, 2/3, 1],
        num_train_timesteps: int = 1000,
        shift: float = 1.0,     # Following Stable diffusion 3, 
    ):
        assert num_stages == len(stage_range) - 1, "Number of stages must match stage ranges"
        assert len(ns) == num_stages - 1, f"len(ns) ({len(ns)}) must match number of stages - 1 ({num_stages - 1})"

        self.timestep_ratios = {}           # The timestep ratio for each stage
        self.timesteps_per_stage = {}       # The  detailed timesteps per stage
        self.sigmas_per_stage = {}
        self.start_sigmas = {}           
        self.end_sigmas = {}
        self.ori_start_sigmas = {}

        # self.init_sigmas()
        self.init_sigmas_for_each_stage()
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

    def init_sigmas_for_each_stage(self):
        """
            Init the timesteps for each stage
        """
        self.init_sigmas()

        stage_distance = []
        stages = self.config.num_stages
        training_steps = self.config.num_train_timesteps
        stage_range = self.config.stage_range

        # Init the start and end point of each stage
        for i_s in range(stages):
            # To decide the start and ends point
            start_indice = int(stage_range[i_s] * training_steps)
            start_indice = max(start_indice, 0)
            end_indice = int(stage_range[i_s+1] * training_steps)
            end_indice = min(end_indice, training_steps)
            start_sigma = self.sigmas[start_indice].item()
            end_sigma = self.sigmas[end_indice].item() if end_indice < training_steps else 0.0
            self.ori_start_sigmas[i_s] = start_sigma

            if i_s != 0:
                sqrt_n_k = math.sqrt(self.config.ns[i_s-1])
                start_sigma = start_sigma * sqrt_n_k / ((1 - start_sigma) + start_sigma * sqrt_n_k)

            stage_distance.append(start_sigma - end_sigma)
            self.start_sigmas[i_s] = start_sigma
            self.end_sigmas[i_s] = end_sigma

        self.set_timestep_ratios(stage_distance)
        self.set_per_stage_timesteps_and_sigmas()



class TemporalFlowMatchEulerDiscreteScheduler(SchedulerMixin, ConfigMixin):
    """
    A thin wrapper that holds one SingleTimescaleFlowMatchEulerDiscreteScheduler per temporal level
    and exposes a unified interface. The per-stage metadata (start/end sigmas, per-stage timesteps/sigmas) is
    stored per temporal level, and `set_timesteps` selects which temporal level to activate.

    Attributes exposed as dicts-of-dicts:
        start_sigmas[temporal_level][stage_index] -> float
        end_sigmas[temporal_level][stage_index]   -> float
        timesteps_per_stage[temporal_level][stage_index] -> torch.Tensor
        sigmas_per_stage[temporal_level][stage_index]    -> torch.Tensor
    """

    _compatibles: List[str] = []
    order = 1

    @register_to_config
    def __init__(
        self,
        ns: List[Sequence[float]],
        num_stages: int = 3,
        stage_range: List[float] = [0, 1/3, 2/3, 1],
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
    ):
        # assert len(ns) == 3, "This wrapper expects exactly three temporal levels (len(ns) == 3)."
        assert len(ns) == num_stages, "Ns must match number of stages"
        assert num_stages == len(stage_range) - 1, "Number of stages must match stage ranges"

        # Instantiate one single-timescale scheduler per temporal level.
        self.schedulers: List[SingleTimescaleFlowMatchEulerDiscreteScheduler] = [
            SingleTimescaleFlowMatchEulerDiscreteScheduler(
                ns=list(_ns),  # ensure it's JSON-serializable in config
                num_stages=num_stages,
                stage_range=stage_range,
                num_train_timesteps=num_train_timesteps,
                shift=shift,
            )
            for _ns in ns
        ]

        # Aggregate per-temporal-level metadata into dicts-of-dicts for simple lookup.
        self.start_sigmas: Dict[int, Dict[int, float]] = {
            lvl: sch.start_sigmas for lvl, sch in enumerate(self.schedulers)
        }
        self.end_sigmas: Dict[int, Dict[int, float]] = {
            lvl: sch.end_sigmas for lvl, sch in enumerate(self.schedulers)
        }
        self.timesteps_per_stage: Dict[int, Dict[int, torch.Tensor]] = {
            lvl: sch.timesteps_per_stage for lvl, sch in enumerate(self.schedulers)
        }
        self.sigmas_per_stage: Dict[int, Dict[int, torch.Tensor]] = {
            lvl: sch.sigmas_per_stage for lvl, sch in enumerate(self.schedulers)
        }
        self.ori_start_sigmas: Dict[int, Dict[int, float]] = {
            lvl: sch.ori_start_sigmas for lvl, sch in enumerate(self.schedulers)
        }

        # Active schedule state mirrors the currently selected temporal level + stage.
        self.timesteps: Optional[torch.Tensor] = None
        self.sigmas: Optional[torch.Tensor] = None
        self.num_inference_steps: Optional[int] = None

        # Bookkeeping for step iteration and begin index (kept same as your schedulers)
        self._step_index: Optional[int] = None
        self._begin_index: Optional[int] = None

        # Track which temporal level / stage is currently active
        self._active_temporal_level: Optional[int] = None
        self._active_stage_index: Optional[int] = None

        # Convenience bounds for the *active* schedule (filled on first set_timesteps)
        self.sigma_min: Optional[float] = None
        self.sigma_max: Optional[float] = None

    # ---------- Convenience accessors (optional) ----------
    def get_start_sigmas(self, temporal_level: int) -> Dict[int, float]:
        return self.start_sigmas[temporal_level]

    def get_end_sigmas(self, temporal_level: int) -> Dict[int, float]:
        return self.end_sigmas[temporal_level]

    def get_timesteps_per_stage(self, temporal_level: int) -> Dict[int, torch.Tensor]:
        return self.timesteps_per_stage[temporal_level]

    def get_sigmas_per_stage(self, temporal_level: int) -> Dict[int, torch.Tensor]:
        return self.sigmas_per_stage[temporal_level]

    def get_scheduler(self, temporal_level: int) -> SingleTimescaleFlowMatchEulerDiscreteScheduler:
        return self.schedulers[temporal_level]

    # ---------- Begin/step index plumbing (same behavior as your schedulers) ----------
    @property
    def step_index(self):
        return self._step_index

    @property
    def begin_index(self):
        return self._begin_index

    @property
    def num_temporal_levels(self) -> int:
        return len(self.schedulers)

    def set_begin_index(self, begin_index: int = 0):
        self._begin_index = begin_index

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps
        indices = (schedule_timesteps == timestep).nonzero()
        pos = 1 if len(indices) > 1 else 0
        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    # ---------- Core API: set_timesteps now also takes temporal_level ----------
    def set_timesteps(
        self,
        num_inference_steps: int,
        stage_index: int,
        temporal_level: int,
        device: Union[str, torch.device] = None,
    ):
        """
        Selects a temporal level (0, 1, or 2), forwards to that inner scheduler,
        and mirrors its active `timesteps` and `sigmas` to this wrapper.

        Args:
            num_inference_steps: number of inference steps for the active run.
            stage_index: which stage (0..num_stages-1) inside the temporal level.
            temporal_level: which of the three inner schedulers to activate (0, 1, or 2).
            device: target device for tensors.
        """
        if temporal_level < 0 or temporal_level >= len(self.schedulers):
            raise ValueError(f"temporal_level must be in [0, {len(self.schedulers)-1}], got {temporal_level}.")

        inner = self.schedulers[temporal_level]
        inner.set_timesteps(num_inference_steps=num_inference_steps, stage_index=stage_index, device=device)

        # Mirror the active scheduler's schedule into the wrapper so pipelines can read `self.timesteps/self.sigmas`.
        self.timesteps = inner.timesteps
        self.sigmas = inner.sigmas
        self.num_inference_steps = num_inference_steps
        self._step_index = None  # reset

        # Track which level/stage are active; keep min/max handy for convenience
        self._active_temporal_level = temporal_level
        self._active_stage_index = stage_index
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

    # ---------- The same single-step update used by your Euler scheduler ----------
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[FlowMatchEulerDiscreteSchedulerOutput, tuple]:
        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                "Passing integer indices as timesteps is not supported. "
                "Pass a value from `scheduler.timesteps` instead."
            )

        if self.sigmas is None or self.timesteps is None:
            raise RuntimeError(
                "You must call `set_timesteps(..., temporal_level=...)` before calling `step`."
            )

        if self.step_index is None:
            self._step_index = 0

        sample = sample.to(torch.float32)
        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]
        prev_sample = sample + (sigma_next - sigma) * model_output
        prev_sample = prev_sample.to(model_output.dtype)

        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)
