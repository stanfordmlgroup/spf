from functools import partial
from pathlib import Path
from typing import List, Optional

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data._utils.collate import default_collate

from emulator.data.loading.collate import (
    StageSampler,
    collate_fn,
    collate_fn_pyramidal,
    collate_fn_spf,
    collate_fn_single_timescale,
)
from emulator.data.datasets import (
    MultiClimateModelDataset,
    canonical_climate_dataset_name,
    is_climate_dataset_name,
)
from emulator.data.io.huggingface import download_climate_dataset
from emulator.utils.utils import random_split


def _worker_init_fn(worker_id):
    import xarray as xr
    # reset xarray file cache per worker (prevents descriptor reuse across processes)
    xr.backends.file_manager.FILE_CACHE.clear()


class ClimateDataModule(LightningDataModule):
    def __init__(
        self,
        dataset: str,
        output_data_dir: Path,
        input_variables: list[str],
        output_variables: list[str],
        output_seq_len: int,
        train_climate_model_metadata: dict[dict],
        test_climate_model_metadata: dict[dict],
        timescales: str,
        log_vars: bool,
        batch_size: int,
        eval_batch_size: int,
        shuffle: bool,
        num_workers: int,
        persistent_workers: bool,
        pin_memory: bool,
        drop_last: bool,
        input_seq_len: int = None,
        regrids: list[str] = None,
        regrid_input: bool = False,
        min_max_norm: bool = False,
        val_split: float = None,
        val_climate_model_metadata: dict[dict] = None,
        output_month: str = None,
        single_timescale_target: str = None,
        spatial_layout: str = "swap_lat_lon",
        data_source: str = "huggingface",
        hf_dataset_repo: str = None,
        hf_dataset_revision: str = "main",
        hf_dataset_cache_dir: str = None,
        hf_dataset_token: str | bool | None = None,
        hf_dataset_subdir: str = None,
        hf_dataset_allow_patterns: list[str] = None,
        hf_dataset_local_files_only: bool = False,
        hf_dataset_max_workers: int = 16,
        hf_dataset_extract_archives: bool = True,
        **kwargs,
    ):
        super().__init__()

        # Dataset configurations
        self.dataset = canonical_climate_dataset_name(dataset)
        self.output_data_dir = output_data_dir
        self.data_source = data_source
        if self.data_source != "huggingface":
            raise ValueError("Only data_source='huggingface' is supported.")
        self.hf_dataset_repo = hf_dataset_repo
        self.hf_dataset_revision = hf_dataset_revision
        self.hf_dataset_cache_dir = hf_dataset_cache_dir
        self.hf_dataset_token = hf_dataset_token
        self.hf_dataset_subdir = hf_dataset_subdir
        self.hf_dataset_allow_patterns = hf_dataset_allow_patterns
        self.hf_dataset_local_files_only = hf_dataset_local_files_only
        self.hf_dataset_max_workers = hf_dataset_max_workers
        self.hf_dataset_extract_archives = hf_dataset_extract_archives
        self.model_name = kwargs.get("model_name", None)

        self.input_variables = list(input_variables)
        self.output_variables = list(output_variables)

        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        if output_month is not None:
            assert "monthly" in timescales, "output_month can only be set if 'monthly' is in timescales"
        self.output_month = output_month
        self.single_timescale_target = single_timescale_target
        self.spatial_layout = spatial_layout

        self.train_climate_model_metadata = train_climate_model_metadata
        self.val_climate_model_metadata = val_climate_model_metadata
        self.test_climate_model_metadata = test_climate_model_metadata

        self.timescales = timescales
        self.regrids = regrids
        self.regrid_input = regrid_input
        self.min_max_norm = min_max_norm
        self.log_vars = log_vars

        # Dataloader configurations
        self.val_split = val_split

        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size

        self.shuffle = shuffle

        self.drop_last = drop_last
        self.dataloader_kwargs = dict(
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            drop_last=drop_last,
        )

        self.flow = True if "model_name" in kwargs and "Flow" in kwargs["model_name"] else False

        self.train_collate_fn = default_collate
        self.eval_collate_fn = default_collate
        self.noise_scheduler = None
        self._flow_collate_fn = None
        self.DatasetClass = MultiClimateModelDataset

        if self.flow:
            model_name = kwargs["model_name"]
            model_config = kwargs["model_config"]
            scheduler_config = model_config["noise_scheduler_config"]
            if not isinstance(scheduler_config, (dict, DictConfig)):
                raise TypeError(
                    "model_config.noise_scheduler_config must be an uninstantiated config. "
                    "Instantiate ClimateDataModule with _recursive_=False."
                )
            self.flow_config = OmegaConf.create({
                "scheduler": {
                    "num_train_timesteps": scheduler_config["num_train_timesteps"],
                    "num_stages":          scheduler_config.get("num_stages", 3),
                    "shift":               scheduler_config.get("shift", None),
                    "stage_range":         scheduler_config.get("stage_range", None),
                    "ns":                  scheduler_config.get("ns", None),
                    "uses_temporal_levels":           model_name == "SpatiotemporalPyramidFlow",
                },
                "model": {
                    "params": {
                        "attention_head_dim":    model_config["attention_head_dim"],
                        "patch_size":            model_config["patch_size"],
                        "encoder_patch_size":    model_config["encoder_patch_size"],
                        "same_scale_prob":       model_config.get("same_scale_prob", 0.0),
                        "no_decadal":            model_config.get("no_decadal", False),
                        "eval_timescale":        model_config.get("eval_timescale", None),
                        "use_pathway":           model_config.get("use_pathway", False),
                        "multiscale_warmup":     model_config.get("multiscale_warmup", 0),
                        "multiscale_ramp":       model_config.get("multiscale_ramp", 0),
                    }
                },
                "data": {
                    "params": {
                        "single_timescale_target": self.single_timescale_target,
                        "output_month":    self.output_month,
                        "input_variables": self.input_variables,
                    }
                }
            })

        if self.flow:
            self.DatasetClass = MultiClimateModelDataset

            if model_name == "SpatiotemporalPyramidFlow":
                self._flow_collate_fn = (
                    lambda examples, config, noise_scheduler_copy, output_seq_len, is_training: collate_fn_spf(
                        examples,
                        config,
                        noise_scheduler_copy,
                        output_seq_len,
                        is_training,
                    )
                )
            elif model_name == "SingleTimescaleSpatiotemporalPyramidFlow":
                if self.dataset != "TimescaleAlignedClimateModelDataset":
                    raise ValueError(
                        "SingleTimescaleSpatiotemporalPyramidFlow requires TimescaleAlignedClimateModelDataset."
                    )
                self._flow_collate_fn = (
                    lambda examples, config, noise_scheduler_copy, output_seq_len, is_training: collate_fn_single_timescale(
                        examples,
                        config,
                        noise_scheduler_copy,
                        output_seq_len,
                        is_training,
                    )
                )
            elif model_name == "PyramidalFlow":
                timescale = self.timescales[0] if len(self.timescales) == 1 else "monthly"
                sampler = StageSampler(
                    max_temporal_length=self.output_seq_len[timescale],
                    frame_per_unit=1,
                )
                self._flow_collate_fn = (
                    lambda examples, config, noise_scheduler_copy, output_seq_len, is_training: collate_fn_pyramidal(
                        examples,
                        config,
                        noise_scheduler_copy,
                        sampler,
                        output_seq_len,
                        is_training,
                    )
                )
            else:
                self._flow_collate_fn = collate_fn

    def set_noise_scheduler(self, noise_scheduler):
        if not self.flow:
            return
        if self._flow_collate_fn is None:
            raise RuntimeError("Cannot attach a noise scheduler before the flow collate function is configured.")
        self.noise_scheduler = noise_scheduler
        print("Using scheduler:", self.noise_scheduler.__class__.__name__, "in datamodule")
        self.train_collate_fn = partial(
            self._flow_collate_fn,
            config=self.flow_config,
            noise_scheduler_copy=self.noise_scheduler,
            output_seq_len=self.output_seq_len,
            is_training=True)
        self.eval_collate_fn = partial(
            self._flow_collate_fn,
            config=self.flow_config,
            noise_scheduler_copy=self.noise_scheduler,
            output_seq_len=self.output_seq_len,
            is_training=False
        )

    def _requested_climate_models(self):
        model_names = []
        for metadata_group in (
            self.train_climate_model_metadata,
            self.val_climate_model_metadata,
            self.test_climate_model_metadata,
        ):
            if metadata_group is None:
                continue
            for metadata in metadata_group:
                model_name = metadata["model"]
                if model_name not in model_names:
                    model_names.append(model_name)
        return model_names

    def setup(self, stage: Optional[str] = None):
        self.output_data_dir = download_climate_dataset(
            repo_id=self.hf_dataset_repo,
            revision=self.hf_dataset_revision,
            cache_dir=self.hf_dataset_cache_dir,
            token=self.hf_dataset_token,
            subdir=self.hf_dataset_subdir,
            allow_patterns=self.hf_dataset_allow_patterns,
            local_files_only=self.hf_dataset_local_files_only,
            max_workers=self.hf_dataset_max_workers,
            climate_models=self._requested_climate_models(),
            timescales=self.timescales,
            extract_archives=self.hf_dataset_extract_archives,
        )

        dataset_params = {
            "dataset": self.dataset,
            "output_data_dir": self.output_data_dir,
            "input_variables": self.input_variables,
            "output_variables": self.output_variables,
            "output_seq_len": self.output_seq_len,
            "timescales": self.timescales,
            "log_vars": self.log_vars,
        }
        if is_climate_dataset_name(self.dataset):
            dataset_params.update({
                "input_seq_len": self.input_seq_len,
                "regrids": self.regrids,
                "regrid_input": self.regrid_input,
                "min_max_norm": self.min_max_norm,
                "spatial_layout": self.spatial_layout,
            })

        if stage == "fit":
            dataset_params["climate_model_metadata"] = self.train_climate_model_metadata
            ds = self.DatasetClass(**dataset_params)
            if self.single_timescale_target in ["yearly", "decadal"]:
                for _ds in ds.climate_model_datasets:
                    _ds.filter_duplicates(self.single_timescale_target)
                ds.ds = ConcatDataset(ds.climate_model_datasets)
            if self.val_climate_model_metadata is None:
                fractions = [1 - self.val_split, self.val_split]
                train_ds, val_ds = random_split(ds, lengths=fractions)
                self.train_ds = train_ds
                self.val_ds = val_ds
            else:
                train_ds = ds
                dataset_params["climate_model_metadata"] = self.val_climate_model_metadata
                val_ds = self.DatasetClass(**dataset_params)
                if self.single_timescale_target in ["yearly", "decadal"]:
                    for _ds in val_ds.climate_model_datasets:
                        _ds.filter_duplicates(self.single_timescale_target, min_keep=self.batch_size)
                    val_ds.ds = ConcatDataset(val_ds.climate_model_datasets)

                self.train_ds, _ = random_split(train_ds, lengths=[1.0, 0.0])
                self.val_ds, _ = random_split(val_ds, lengths=[1.0, 0.0])

            self.last_batch_train_buffer = 0
            self.last_batch_val_buffer = 0
            if not self.drop_last and self.flow:
                # Buffer the dataset sizes to be multiples of batch size for eval dataloaders
                train_extra = (self.batch_size - len(self.train_ds) % self.batch_size) % self.batch_size
                val_extra = (self.eval_batch_size - len(self.val_ds) % self.eval_batch_size) % self.eval_batch_size
                if train_extra > 0:
                    self.train_ds.buffer(train_extra)
                    self.last_batch_train_buffer = train_extra
                if val_extra > 0:
                    self.val_ds.buffer(val_extra)
                    self.last_batch_val_buffer = val_extra
    
        if stage == "test":
            dataset_params.update({
                "climate_model_metadata": self.test_climate_model_metadata,
                "testing_dataset": True,
            })
            ds = self.DatasetClass(**dataset_params)
            if self.single_timescale_target in ["yearly", "decadal"]:
                filter_timescale = self.single_timescale_target
            elif (
                self.flow
                and self.flow_config.model.params.eval_timescale in ["yearly", "decadal"]
                and self.model_name != "PyramidalFlow"
            ):
                filter_timescale = self.flow_config.model.params.eval_timescale
            else:
                filter_timescale = None

            if filter_timescale is not None:
                for _ds in ds.climate_model_datasets:
                    _ds.filter_duplicates(filter_timescale, min_keep=self.eval_batch_size)
                ds.ds = ConcatDataset(ds.climate_model_datasets)
            self.test_ds = ds

            self.last_batch_test_buffers = [0 for _ in self.test_ds.climate_model_datasets]
            if not self.drop_last and self.flow:
                # Buffer the dataset sizes to be multiples of batch size for eval dataloaders
                test_extras = []
                for _ds in self.test_ds.climate_model_datasets:
                    extra = (self.eval_batch_size - len(_ds) % self.eval_batch_size) % self.eval_batch_size
                    test_extras.append(extra)
                    if extra > 0:
                        _ds.buffer(extra)
                self.last_batch_test_buffers = test_extras

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_ds,
            collate_fn=self.train_collate_fn,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            worker_init_fn=_worker_init_fn,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_ds,
            collate_fn=self.eval_collate_fn,
            batch_size=self.eval_batch_size,
            shuffle=False,
            worker_init_fn=_worker_init_fn,
            **self.dataloader_kwargs,
        )

    def test_dataloader(self) -> List[DataLoader]:
        test_datasets = [ds for ds in self.test_ds.climate_model_datasets]

        return [
            DataLoader(
                dataset=ds,
                collate_fn=self.eval_collate_fn,
                batch_size=self.eval_batch_size,
                shuffle=False,
                **self.dataloader_kwargs,
            )
            for ds in test_datasets
        ]
