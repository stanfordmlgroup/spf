import wandb
from hydra.utils import instantiate as hydra_instantiate
from omegaconf import DictConfig
import torch
import torch.distributed as dist

import pytorch_lightning as pl
from pytorch_lightning import seed_everything

import emulator.utils.config_utils as cfg_utils
from emulator.utils.interface import get_model_and_data
from emulator.utils.utils import get_logger
from pytorch_lightning.profilers import PyTorchProfiler
from datetime import datetime
from torch.profiler import (
    ProfilerActivity,
    tensorboard_trace_handler,
    schedule,
)


def run_model(config: DictConfig):
    seed_everything(config.seed, workers=True)
    log = get_logger(__name__)

    log.info("In run model")
    cfg_utils.extras(config)

    log.info("Running model")
    if config.get("print_config"):
        cfg_utils.print_config(config, fields="all")

    emulator_model, data_module = get_model_and_data(config)
    log.info(f"Got model - {config.name}")
    c = datetime.now()
    # Displays Time
    current_time = c.strftime("%H:%M:%S")

    profiler = None
    checkpointing = True
    if config.get("pyprofile"):
        checkpointing = False
        profiler = PyTorchProfiler(
            dirpath="logs/profiles",
            filename=f"Pyprofile-{config.name}-Basetest-{current_time}",
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True,
            record_shapes=True,
            on_trace_ready=tensorboard_trace_handler("logs/profiles"),
            schedule=schedule(wait=1, warmup=1, active=3, repeat=2),
        )

    log.info(config.name)

    # Init Lightning callbacks and loggers
    callbacks = cfg_utils.get_all_instantiable_hydra_modules(config, "callbacks")
    loggers = cfg_utils.get_all_instantiable_hydra_modules(config, "logger")

    # Init Lightning trainer
    trainer: pl.Trainer = hydra_instantiate(
        config.trainer,
        profiler=profiler,
        callbacks=callbacks,
        logger=loggers,  # , deterministic=True
        enable_checkpointing=checkpointing,
        log_every_n_steps=1,
    )

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters to the PyTorch Lightning loggers.")
    cfg_utils.log_hyperparameters(
        config=config,
        model=emulator_model,
        data_module=data_module,
        trainer=trainer,
        callbacks=callbacks,
    )

    if config.get("only_test_model") is False:
        trainer.fit(model=emulator_model, datamodule=data_module, ckpt_path=config.get("resume_from_checkpoint", None))
        previous_ckpt_path = trainer.checkpoint_callback.best_model_path
    else:
        log.info("Only testing the model.")

    if not config.logger.get("name") == "none":
        cfg_utils.save_hydra_config_to_wandb(config)

    # Testing:
    if config.get("test_after_training"):
        if config.get("only_test_model"):
            trainer: pl.Trainer = hydra_instantiate(
                config.trainer,
                profiler=profiler,
                callbacks=callbacks,
                logger=loggers,  # , deterministic=True
                enable_checkpointing=checkpointing,
                log_every_n_steps=1,
                devices=1,
            )
            trainer.test(model=emulator_model, datamodule=data_module)
        else:
            if dist.is_initialized():
                dist.destroy_process_group()
                proceed = trainer.is_global_zero
            else:
                proceed = True
            if proceed:
                trainer: pl.Trainer = hydra_instantiate(
                    config.trainer,
                    profiler=profiler,
                    callbacks=callbacks,
                    logger=loggers,  # , deterministic=True
                    enable_checkpointing=checkpointing,
                    log_every_n_steps=1,
                    devices=1,
                )
                # testner.test(datamodule=data_module, ckpt_path="best")
                trainer.test(
                    model=emulator_model,
                    datamodule=data_module,
                    ckpt_path=previous_ckpt_path,
                )

    if not config.logger.get("name") == "none":
        if config.get("logger"):
            wandb.finish()

    print("Finished")
