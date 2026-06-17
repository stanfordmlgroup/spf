import os
import gc
import time as clocktime
import math
import hydra
import numpy as np
import torch

from typing import Any
from omegaconf import DictConfig
from collections import defaultdict
from pytorch_lightning import LightningModule

from emulator.callbacks import PredictionPostProcessCallback
from emulator.utils.utils import get_logger, slice_batch
from emulator.visualization.visualize import (
    create_predictions_fig,
    global_mean_plot,
)
from emulator.evaluation import Evaluator
from emulator.weighting import broadcast_latitude_weights
from emulator.data.utils import unnormalize_dict


def _format_results_table(test_metrics: list[dict[str, float]]):
    rows = []
    columns = ["Scenario", "Variable"]

    for dataset_index, metrics_by_name in enumerate(test_metrics):
        pattern = next(iter(metrics_by_name))
        scenario = "Average" if "/" not in pattern else pattern.split("/")[0]
        var2metric = defaultdict(list)
        for key, value in metrics_by_name.items():
            if "/" not in key:
                metric = key
                var_name = "avg"
            else:
                metric = key.split("/")[-1]
                var_name = key.split("/")[1]
            var2metric[var_name].append((metric, value))

        for var_name, metrics in var2metric.items():
            metrics.sort(key=lambda x: x[0])
            rows.append([scenario, var_name, *[value for _, value in metrics]])

        if dataset_index == 0 and var2metric:
            columns.extend(metric for metric, _ in next(iter(var2metric.values())))

    return columns, rows


class BaseModel(LightningModule):
    def __init__(
        self,
        configs: DictConfig = None,
        **kwargs,
    ):
        super().__init__()

        self.log_text = get_logger(__name__)
        self.log_text.info("Base Model init!")

        self.config = configs
        self.datamodule_config = self.config["datamodule"]

        self.criterion = lambda pred, y, lats=None: torch.nn.functional.mse_loss(pred, y)

        self.input_variables = self.datamodule_config["input_variables"]
        self.input_postprocessor = PredictionPostProcessCallback(
            variables=self.input_variables, sizes=1
        )

        self.output_variables = self.datamodule_config["output_variables"]
        self.output_postprocesser = PredictionPostProcessCallback(
            variables=self.output_variables, sizes=1
        )

        climate_models = self.config.get("climate_models", None)
        if climate_models is not None:
            self.climate_models = sorted(climate_models)
        else:
            model_path = self.config["model"].get("pretrained_ckpt_dir", None)
            if model_path is not None:
                datamodule_config = torch.load(model_path)['hyper_parameters']['configs']['datamodule']
                climate_models = datamodule_config['train_climate_model_metadata']
                self.climate_models = sorted(list(set(dataset['model'] for dataset in climate_models)))
            else:
                self.climate_models = sorted(list(set(dataset['model'] for dataset in configs['datamodule']['train_climate_model_metadata'])))

        self.start_validation_after_n_epochs = int(
            (configs or {}).get("start_validation_after_n_epochs", 0)
        )
        self._orig_check_val_every_n_epoch = None
        self._val_delay_restored = False

        self.is_temporal = False
        self.multi_scale = False
        self.is_pyramidal = False

    def _eval_cache_dir(self, name: str):
        root = self.config.get("eval_cache_dir", None)
        if root is None:
            root = os.path.join(self.config.get("work_dir", "."), ".cache", "eval")
        return os.path.join(root, name)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, X, climate_model=None, embeddings=None):
        """
        Downstream model forward pass, input X will be the (batched) output from self.input_transform
        """
        raise NotImplementedError("Base model is an abstract class!")

    def on_fit_start(self) -> None:
        """Delay validation by temporarily increasing check_val_every_n_epoch."""
        # stash original cadence (usually 1)
        self._orig_check_val_every_n_epoch = getattr(self.trainer, "check_val_every_n_epoch", 1)

        n = self.start_validation_after_n_epochs
        if n > 0:
            # First validation should happen at the end of epoch n
            # (PL’s current_epoch is 0-based; check uses (epoch+1) % check_val_every_n_epoch == 0)
            self.trainer.check_val_every_n_epoch = n

    def on_train_start(self) -> None:
        self.log_text.info("Starting Training")

    def on_train_epoch_start(self) -> None:
        self._start_epoch_time = clocktime.time()
        gc.collect()
        torch.cuda.empty_cache()

        n = getattr(self, "start_validation_after_n_epochs", 0)
        if n > 0 and not self._val_delay_restored and self.current_epoch >= n:
            # Restore whatever cadence the user originally set (typically 1)
            self.trainer.check_val_every_n_epoch = self._orig_check_val_every_n_epoch
            self._val_delay_restored = True

    def predict(self, X, metadata=None, *args, **kwargs):

        if len(self.climate_models) > 1:
            climate_models = metadata["climate_model"]
            class_labels = torch.tensor(
                [self.climate_models.index(model) for model in climate_models],
                dtype=torch.int32,
                device=self.device
            )
            preds = self(X=X, class_labels=class_labels, metadata=metadata)
        else:
            preds = self(X=X, metadata=metadata)

        preds = torch.mean(preds, dim=1)
        preds = preds.unsqueeze(1)

        return preds

    def training_step(self, batch: Any, batch_idx: int):

        X = batch["X"]
        targets = batch["Y"]

        if batch_idx == 0:
            # Cache one batch
            self.train_batch = batch

        latitude = batch["metadata"]["latitude"]

        preds = self.predict(X=X, metadata=batch["metadata"])

        preds = self.output_postprocesser.split_vector_by_variable(preds)
        targets = self.output_postprocesser.split_vector_by_variable(targets)

        # Everything we want to log to wandb should go in here
        train_log = dict()

        loss = 0

        # Compute loss for each variable
        for out_var in self.output_variables:

            loss_per_var = self.criterion(
                pred=preds[out_var],
                y=targets[out_var],
                lats=latitude
            )

            loss += loss_per_var
            train_log[f"train/{out_var}/loss"] = loss_per_var

        # Average Loss over variables
        loss = loss / len(self.output_variables)

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

        self.log_dict(
            {**train_log, "train/loss": loss, "n_zero_gradients": n_zero_gradients}
        )

        return {"loss": loss}

    def on_train_epoch_end(self):
        train_time = clocktime.time() - self._start_epoch_time
        self.log_dict({"epoch": self.current_epoch, "time/train": train_time})
        print("\n")

    def _get_temporal_batch_timescale(self):
        if "eval_timescale" in self.config and self.multi_scale:
            return self.config["eval_timescale"]
        if "single_timescale_target" in self.datamodule_config:
            return self.datamodule_config["single_timescale_target"]
        return "monthly"

    def _predict_step(self, batch):
        X = batch["X"]
        metadata = batch["metadata"]

        kwarg_names = ["pos_embed_q", "pos_embed_k", "latent_size", "seqlen_list_q", "seqlen_list_k", "timelen_list_q", "latent_time"]
        kwargs = {}
        for name in kwarg_names:
            if name in batch:
                kwargs[name] = batch[name]
            else:
                kwargs[name] = None

        dataset = self.datamodule_config["dataset"]

        if self.is_temporal:
            if "eval_timescale" in self.config:
                if not self.multi_scale:
                    timescale = self._get_temporal_batch_timescale()
                    eval_timescale = self.config["eval_timescale"]
                else:
                    timescale = self.config["eval_timescale"]
                    eval_timescale = timescale
            elif "single_timescale_target" in self.datamodule_config:
                timescale = self.datamodule_config["single_timescale_target"]
                eval_timescale = timescale
            else:
                timescale = "monthly"
                eval_timescale = timescale
            inputs = X[timescale]
            targets = batch["Y"][timescale]
            stats = metadata[f"{timescale}_stats"]
            in_stats = metadata[f"{timescale}_input_stats"]
            kwargs["timescale"] = timescale
        else:
            inputs = X
            targets = batch["Y"]
            stats = metadata["stats"]
            in_stats = metadata["input_stats"]
            if self.is_pyramidal:
                timescale = "monthly"
                eval_timescale = self.config.get("eval_timescale", "monthly")
            else:
                timescale = None
                eval_timescale = None

        if self.multi_scale and "eval_timescale" in self.config:
            kwargs["timescale"] = self.config["eval_timescale"]

        inputs = self.input_postprocessor.split_vector_by_variable(inputs)
        inputs = unnormalize_dict(inputs, stats=in_stats, batch_dim=True,
                                  log_vars=self.datamodule_config["log_vars"],
                                  min_max_norm=self.datamodule_config["min_max_norm"])

        targets = self.output_postprocesser.split_vector_by_variable(targets)
        targets = unnormalize_dict(targets, stats=stats, batch_dim=True,
                                   log_vars=self.datamodule_config["log_vars"],
                                   min_max_norm=self.datamodule_config["min_max_norm"])

        ensemble_preds = []
        for _ in range(self.config["num_eval_samples"]):
            preds = self.predict(X=X, metadata=metadata, **kwargs)
            preds = self.output_postprocesser.split_vector_by_variable(preds)
            preds = unnormalize_dict(preds, stats=stats, batch_dim=True, 
                                     log_vars=self.datamodule_config["log_vars"],  
                                     min_max_norm=self.datamodule_config["min_max_norm"])
            ensemble_preds.append(preds)

        # Reshape ensemble preds and targets
        ensemble_preds = {
            key: torch.stack(
                [pred[key] for pred in ensemble_preds]
            ).transpose(1, 0)# Swap B and N so B is first
            for key in ensemble_preds[0].keys()
        } # (B, N, T, V, H, W)

        if ((self.is_temporal or self.is_pyramidal) and (timescale != eval_timescale)):
            if self.is_temporal:
                # Assert eval timescale is only one "above" the training timescale
                assert self.stages.index(eval_timescale) + 1 == self.stages.index(timescale), \
                    f"eval_timescale {eval_timescale} should be one above training timescale {timescale}"
            else:
                assert eval_timescale == "yearly", f"For pyramidal model eval_timescale should be yearly but got {eval_timescale}"
            # Average both preds and targets over time dim
            ensemble_preds = {
                key: ensemble_preds[key].mean(dim=2, keepdim=True) 
                for key in ensemble_preds
            }
            targets = {
                key: targets[key].mean(dim=1, keepdim=True)
                for key in targets
            }

        return inputs, ensemble_preds, targets

    def _evaluation_step(
            self,
            batch: Any,
            batch_idx: int,
            outputs_collector: dict = None,
            evaluator: Evaluator = None,
            dataloader_idx: int = 0,
        ):
        if self.trainer.is_last_batch:
            # Slice the batch to remove any buffered samples
            if self.trainer.training:
                buffer_size = self.trainer.datamodule.last_batch_train_buffer
            elif self.trainer.validating:
                buffer_size = self.trainer.datamodule.last_batch_val_buffer
            elif self.trainer.testing:
                buffer_size = self.trainer.datamodule.last_batch_test_buffers[dataloader_idx]
            if buffer_size > 0:
                if self.is_temporal:
                    timescale = self._get_temporal_batch_timescale()
                    batch_size = batch["X"][timescale].shape[0]
                else:
                    batch_size = batch["X"].shape[0]
                slice_size = batch_size - buffer_size
                batch = slice_batch(batch, slice_size)

        metadata = batch["metadata"]
        inputs, ensemble_preds, targets = self._predict_step(batch)

        # Compute ensemble mean
        plot_sample_mean = self.config.get("plot_sample_mean", True)
        if plot_sample_mean:
            preds = {
                key: torch.mean(ensemble_preds[key], dim=1)
                for key in ensemble_preds
            }
        else:
            preds = ensemble_preds

        # Compute lat-weighted global means for each member and store them in outputs_collector
        latitude = metadata["latitude"]
        weights = torch.cos(latitude * np.pi / 180.0) # (B, W)
        B, N, T, V, H, W = ensemble_preds[self.output_variables[0]].shape
        weights = broadcast_latitude_weights(weights, targets[self.output_variables[0]]).expand(B, T, V, H, W)
        for out_var in self.output_variables:
            if not plot_sample_mean:
                _weights = weights.unsqueeze(1)
            else:
                _weights = weights
            global_mean_preds = torch.sum(
                preds[out_var] * _weights, dim=(-1, -2)
            ) / torch.sum(_weights, dim=(-1, -2)) # (B, T, V)
            global_mean_target = torch.sum(
                targets[out_var] * weights, dim=(-1, -2)
            ) / torch.sum(weights, dim=(-1, -2))
            outputs_collector[out_var]["preds"].append(global_mean_preds.cpu().numpy())
            outputs_collector[out_var]["targets"].append(global_mean_target.cpu().numpy())

        input_latitude = metadata["input_latitude"]
        input_weights = torch.cos(input_latitude * np.pi / 180.0)
        B, T, V, H, W = inputs[self.input_variables[0]].shape
        input_weights = broadcast_latitude_weights(input_weights, inputs[self.input_variables[0]]).expand(B, T, V, H, W)
        for in_var in self.input_variables:
            global_mean_inputs = torch.sum(
                inputs[in_var] * input_weights, dim=(-1, -2)
            ) / torch.sum(input_weights, dim=(-1, -2))
            outputs_collector[in_var]["inputs"].append(global_mean_inputs.cpu().numpy())

        # Store time, ssp, em, model
        if self.is_temporal:
            if "eval_timescale" in self.config:
                timescale = self._get_temporal_batch_timescale()
                eval_timescale = self.config["eval_timescale"]
            elif "single_timescale_target" in self.datamodule_config:
                eval_timescale = self.datamodule_config["single_timescale_target"]
                timescale = eval_timescale
            else:
                eval_timescale = "monthly"
                timescale = eval_timescale
            batch_size = batch["X"][timescale].shape[0]
            time = torch.stack(metadata[f"{timescale}_Y_time"], dim=1).cpu().numpy()
        else:
            batch_size = batch["X"].shape[0]
            time = torch.stack(metadata["Y_time"], dim=1).cpu().numpy()[:, 0]
            eval_timescale = None
        outputs_collector["metadata"]["time"].append(time)
        outputs_collector["metadata"]["scenario"].append(np.array(metadata["scenario"]))
        outputs_collector["metadata"]["member"].append(np.array(metadata["member"]))
        outputs_collector["metadata"]["climate_model"].append(np.array(metadata["climate_model"]))

        evaluator.update(
            preds=ensemble_preds,
            targets=targets,
            climate_models=metadata["climate_model"],
            scenarios=metadata["scenario"],
            members=metadata["member"],
            lat_weights=torch.cos(metadata["latitude"] * np.pi / 180.0),
            time=time,
            batch_idx=batch_idx
        )
    
        num_samples_viz = self.config.get("num_samples_viz", 1)
        viz_batch_indices = list(range(math.ceil(num_samples_viz / batch_size)))
        if self.logger is not None and batch_idx in viz_batch_indices:
            if batch_idx != viz_batch_indices[-1]:
                num_samples_viz_batch = batch_size
            else:
                num_samples_viz_batch = num_samples_viz - (len(viz_batch_indices) - 1) * batch_size
            if not plot_sample_mean:
                # If not plotting sample mean, take the first sample
                preds = {
                    key: preds[key][:, 0] 
                    for key in preds
                }
            preds = {
                key: preds[key].cpu().numpy()
                for key in preds
            }
            targets = {
                key: targets[key].cpu().numpy()
                for key in targets
            }
            # Visualize a validation prediction
            self._visualize_predictions(
                preds=preds,
                targets=targets,
                metadata={key: value.cpu().numpy() if isinstance(value, torch.Tensor) else value for key, value in metadata.items()},
                figure_name="Sample Validation Predictions",
                num_samples=num_samples_viz_batch,
                timescale=eval_timescale
            )

    def on_validation_epoch_start(self):
        self._start_validation_epoch_time = clocktime.time()
        self.val_step_outputs = defaultdict(lambda: defaultdict(list))
        run_name = os.path.basename(os.path.dirname(self.trainer.checkpoint_callback.dirpath))
        cache_dir = self._eval_cache_dir(os.path.join(run_name, "metric_cache"))
        self.evaluator = Evaluator(
            cache_dir=cache_dir,
            ensemble_metrics=[] if self.config["num_eval_samples"] <= 1 else None
        )

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = None):
        with torch.no_grad():
            self._evaluation_step(batch, batch_idx, self.val_step_outputs, self.evaluator, dataloader_idx)

    def on_validation_epoch_end(self) -> dict:
        val_time = clocktime.time() - self._start_validation_epoch_time
        self.log("time/validation", val_time)

        val_stats = self.evaluator.compute()
        self.evaluator.reset()

        cmodel_scenarios = set(key.split("/")[0] for key in val_stats.keys() if "/" in key)
        val_metrics_table = []
        for cmodel_scenario in cmodel_scenarios:
            # Filter the validation stats for the current climate model / scenario
            cmodel_scenario_stats = {key: val_stats[key] for key in val_stats if key.startswith(cmodel_scenario)}
            # Log the validation stats for the current climate model / scenario
            val_metrics_table.append(cmodel_scenario_stats)
        val_metrics_table.insert(0, {key: val_stats[key] for key in val_stats if "/" not in key})

        target_val_metric = val_stats.pop(self.hparams.monitor)
        self.log_dict({**val_stats, "epoch": self.current_epoch}, prog_bar=False)

        # Show the main validation metric on the progress bar:
        self.log(self.hparams.monitor, target_val_metric, prog_bar=True)

        if self.is_temporal:
            # Time endpoints are the same for monthly timescale, which is used for temporal models during validation
            time = np.concatenate(self.val_step_outputs["metadata"]["time"], axis=0)[:, 0]
        else:
            time = np.concatenate(self.val_step_outputs["metadata"]["time"], axis=0)

        if "eval_timescale" in self.config:
            timescale = self.config["eval_timescale"]
        elif "single_timescale_target" in self.datamodule_config:
            timescale = self.datamodule_config["single_timescale_target"]
        elif "monthly" in self.datamodule_config.timescales:
            timescale = "monthly"
        else:
            timescale = None

        self._visualize_global_mean_predictions(
            inputs={var: np.concatenate(self.val_step_outputs[var]["inputs"], axis=0) for var in self.input_variables},
            preds={var: np.concatenate(self.val_step_outputs[var]["preds"], axis=0) for var in self.output_variables},
            targets={var: np.concatenate(self.val_step_outputs[var]["targets"], axis=0) for var in self.output_variables},
            time=time,
            ssp=np.concatenate(self.val_step_outputs["metadata"]["scenario"], axis=0),
            em=np.concatenate(self.val_step_outputs["metadata"]["member"], axis=0),
            split_name="Validation",
            model=np.concatenate(self.val_step_outputs["metadata"]["climate_model"], axis=0),
            # Latitude and longitude are not needed for global mean when is_means=True
            latitude=None,
            longitude=None,
            is_means=True,
            timescale=timescale
        )

        if self.logger is not None:
            # Generate the result summary table
            columns, rows = _format_results_table(val_metrics_table)

            self.logger.log_table(key="Validation Results Summary", columns=columns, data=rows)

        self.val_step_outputs.clear()

        return val_stats

    def on_test_epoch_start(self) -> None:
        self._start_test_epoch_time = clocktime.time()
        self.test_step_outputs = {}
        self.evaluator = {}
        for i, _ in enumerate(self.trainer.datamodule.test_ds.climate_model_datasets):
            self.test_step_outputs[i] = defaultdict(lambda: defaultdict(list))
            run_name = os.path.basename(os.path.dirname(self.trainer.checkpoint_callback.dirpath))
            cache_dir = self._eval_cache_dir(os.path.join(run_name, f"metric_cache{i}"))
            self.evaluator[i] = Evaluator(
                cache_dir=cache_dir,
                ensemble_metrics=[] if self.config["num_eval_samples"] <= 1 else None
            )

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        with torch.no_grad():
            self._evaluation_step(
                batch,
                batch_idx,
                self.test_step_outputs[dataloader_idx],
                self.evaluator[dataloader_idx],
                dataloader_idx
            )

    def on_test_epoch_end(self) -> dict:
        test_time = clocktime.time() - self._start_test_epoch_time
        self.log("time/test", test_time)
        print("\n")

        test_metrics_table = []

        # Compute test statistisc per testing dataset
        datasets = self.trainer.datamodule.test_ds.climate_model_datasets

        if "eval_timescale" in self.config:
            timescale = self.config["eval_timescale"]
            if self.is_temporal or self.is_pyramidal:
                train_timescale = "monthly"
                if timescale != train_timescale:
                    plot_timescale = None
                else:
                    plot_timescale = timescale
            else:
                plot_timescale = timescale
        elif "single_timescale_target" in self.datamodule_config:
            timescale = self.datamodule_config["single_timescale_target"]
            plot_timescale = timescale
        elif "monthly" in self.datamodule_config.timescales:
            timescale = "monthly"
            plot_timescale = timescale
        else:
            timescale = None
            plot_timescale = timescale

        for i, dataset in enumerate(datasets):
            split_name = f"{dataset.climate_model}_{dataset.scenarios[0]}"

            if self.is_temporal and timescale == "monthly":
                # Time endpoints are the same for monthly timescale
                time = np.concatenate(self.test_step_outputs[i]["metadata"]["time"], axis=0)[:, 0]
            elif not self.is_temporal and timescale == "monthly":
                # For non-temporal models, times do not need to be sliced
                time = np.concatenate(self.test_step_outputs[i]["metadata"]["time"], axis=0)
            elif timescale in ["yearly", "decadal"] and self.is_temporal and not self.is_pyramidal:
                # Time is intervals, just take the start and the per-time means are plotted
                time = np.concatenate(self.test_step_outputs[i]["metadata"]["time"], axis=0)[:, 0]
            else:
                time = np.concatenate(self.test_step_outputs[i]["metadata"]["time"], axis=0)

            if self.trainer.datamodule.last_batch_test_buffers[i] > 0:
                # Remove buffered samples from time
                # buffer_size = self.trainer.datamodule.last_batch_test_buffers[i]
                # time = time[:-buffer_size]
                pass
            else:
                self._visualize_global_mean_predictions(
                    inputs={var: np.concatenate(self.test_step_outputs[i][var]["inputs"], axis=0) for var in self.input_variables},
                    preds={var: np.concatenate(self.test_step_outputs[i][var]["preds"], axis=0) for var in self.output_variables},
                    targets={var: np.concatenate(self.test_step_outputs[i][var]["targets"], axis=0) for var in self.output_variables},
                    time=time,
                    ssp=np.concatenate(self.test_step_outputs[i]["metadata"]["scenario"], axis=0),
                    em=np.concatenate(self.test_step_outputs[i]["metadata"]["member"], axis=0),
                    split_name=split_name,
                    model=np.concatenate(self.test_step_outputs[i]["metadata"]["climate_model"], axis=0),
                    # Latitude and longitude are not needed for global mean when is_means=True
                    latitude=None,
                    longitude=None,
                    is_means=True,
                    timescale=plot_timescale
                )

            test_stats = self.evaluator[i].compute()
            self.evaluator[i].reset()

            # Remove average metrics because each dataset is evaluated separately
            test_stats = {key: value for key, value in test_stats.items() if "/" in key}

            self.log_dict({**test_stats, "epoch": self.current_epoch}, prog_bar=False)

            self.test_step_outputs[i].clear()

            test_metrics_table.append(test_stats)

        if self.logger is not None:
            # Generate the result summary table
            columns, rows = _format_results_table(test_metrics_table)

            self.logger.log_table(key="Test Results Summary", columns=columns, data=rows)

    def configure_optimizers(self):

        optimizer_config = self.config["model"]["optimizer"]
        optimizer = hydra.utils.instantiate(optimizer_config, params=self.parameters())

        # Configure learning rate scheduler
        if "scheduler" in self.config["model"]:
            scheduler_config = self.config["model"]["scheduler"]
            scheduler = hydra.utils.instantiate(scheduler_config, optimizer=optimizer)

            lr_dict = {
                "scheduler": scheduler,
                "monitor": self.config["model"]["monitor"],
                "mode": "min",
            }

            return {"optimizer": optimizer, "lr_scheduler": lr_dict}
        else:
            return optimizer

    def _visualize_predictions(self, preds, targets, metadata, figure_name, num_samples=1, timescale=None):
        if self.logger is None:
            return

        fig = create_predictions_fig(
            preds=preds,
            targets=targets,
            metadata=metadata,
            n=list(range(num_samples)),
            t=0,
        )

        self.logger.log_image(key=figure_name, images=[fig])

    def _visualize_global_mean_predictions(
        self, inputs, preds, targets, time, ssp, em, split_name, model, latitude, longitude, is_means, timescale="monthly"
    ):
        if self.logger is None:
            return

        if timescale is not None:
            output_seq_len = self.trainer.datamodule.output_seq_len
            out_seq_len = output_seq_len[timescale]

            if timescale == "monthly":
                if out_seq_len > 12:
                    period_labels = [f"Month{i + 1}" for i in range(out_seq_len)]
                else:
                    period_labels = [calendar.month_abbr[i + 1] for i in range(out_seq_len)]
            elif timescale == "yearly":
                period_labels = [f"Year{i + 1}" for i in range(out_seq_len)]
            elif timescale == "decadal":
                period_labels = [f"Decade{i + 1}" for i in range(out_seq_len)]

            per_time_columns = out_seq_len > 1
            if not per_time_columns:
                period_labels = None
        else:
            out_seq_len = 1
            period_labels = None
            per_time_columns = False

        plot_sample_mean = self.config.get("plot_sample_mean", True)
        if not plot_sample_mean:
            # Flatten the predictions and expand the time, model, ssp, and em to match the predictions
            B, N, T, _ = preds[self.output_variables[0]].shape
            inputs = {key: np.repeat(inputs[key], repeats=N, axis=0) for key in inputs}
            preds = {key: preds[key].reshape(B*N, T, *preds[key].shape[3:]) for key in preds}
            targets = {key: np.repeat(targets[key], repeats=N, axis=0) for key in targets}
            time = np.repeat(time, repeats=N)
            model = np.repeat(model, repeats=N)
            ssp   = np.repeat(ssp, repeats=N)
            em    = np.array([f"{e}_{i+1}" for e in em for i in range(N)])
             
            np.savez_compressed(
                f"figures/{timescale}_global_means.npz",
                **inputs,
                **{f"preds_{k}": v for k, v in preds.items()},
                **{f"targets_{k}": v for k, v in targets.items()},
                time=time,
                model=model,
                ssp=ssp,
                em=em,
            )

        fig = global_mean_plot(
            inputs=inputs,
            preds=preds,
            targets=targets,
            latitude=latitude,
            longitude=longitude,
            time=time,
            model=model,
            ssp=ssp,
            em=em,
            split=None,
            is_means=is_means,
            is_sample_mean=plot_sample_mean,
            per_time_columns=per_time_columns,
            period_labels=period_labels,
        )

        self.logger.log_image(
            key=f"Visualization/{timescale}_global_mean", images=[fig]
        )
