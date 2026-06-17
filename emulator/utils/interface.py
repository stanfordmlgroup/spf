import os
import ast
import hydra
import torch
import wandb

from omegaconf import DictConfig

from emulator.utils.utils import get_logger
from emulator.data.loading.datamodule import ClimateDataModule


"""
In this file you can find helper functions to avoid model/data loading and reloading boilerplate code
"""
log = get_logger()


def get_model(config: DictConfig, **kwargs):
    """
    Args:
        config (DictConfig): A OmegaConf config (e.g. produced by hydra yaml config file parsing)
    Returns:
        The model that you can directly use to train with pytorch-lightning
    """


    model = hydra.utils.instantiate(
        config.model,
        _recursive_=False,
        configs=config,
        **kwargs,
    )

    return model


def get_datamodule(config: DictConfig) -> ClimateDataModule:
    """
    Args:
        config (DictConfig): A OmegaConf config (e.g. produced by hydra yaml config file parsing)
    Returns:
        A datamodule that you can directly use to train pytorch-lightning models
    """

    data_module: ClimateDataModule = hydra.utils.instantiate(
        config.datamodule,
        _recursive_=False,
    )

    return data_module


def get_model_and_data(config: DictConfig,
                       group: str = "aicc",
                       project: str = "sai"):
    """
    Args:
        config (DictConfig): A OmegaConf config (e.g. produced by hydra yaml config file parsing)
    Returns:
        A tuple of (model, datamodule), that you can directly use to train with pytorch-lightning
        (e.g., checkpointing the best model w.r.t. a small validation set with the ModelCheckpoint callback),
        with:
            trainer.fit(model=model, datamodule=datamodule)
    """

    pretrained_run_id = config.model.get("pretrained_run_id")
    pretrained_ckpt_dir = config.model.get("pretrained_ckpt_dir")

    # If we are loading model weights, we need to get the correct model config from Wandb.
    if pretrained_run_id is not None and config.model.get("load_config_from_wandb", True):

        run = wandb.Api().run(f"{group}/{project}/{pretrained_run_id}")

        model_config = run.config["model"]

        config["model"].update(model_config)
        # Overwrite pretrained_run_id and pretrained_ckpt_dir 
        config["model"]["pretrained_run_id"] = pretrained_run_id
        config["model"]["pretrained_ckpt_dir"] = pretrained_ckpt_dir

        # Nested dicts from Wandb appear as str for some reason.
        if "unet3d_config" in config["model"]:
            config["model"]["unet3d_config"] = ast.literal_eval(config["model"]["unet3d_config"])

        if "unet2d_config" in config["model"]:
            config["model"]["unet2d_config"] = ast.literal_eval(config["model"]["unet2d_config"])

        if "emulator_config" in config["model"]:
            config["model"]["emulator_config"] = ast.literal_eval(config["model"]["emulator_config"])

        if "peft_config" in config["model"]:
            config["model"]["peft_config"] = ast.literal_eval(config["model"]["peft_config"])

        if "transformer_config" in config["model"]:
            config["model"]["transformer_config"] = ast.literal_eval(config["model"]["transformer_config"])

        if "encoder_config" in config["model"]:
            config["model"]["encoder_config"] = ast.literal_eval(config["model"]["encoder_config"])

        if "noise_scheduler_config" in config["model"]:
            config["model"]["noise_scheduler_config"] = ast.literal_eval(config["model"]["noise_scheduler_config"])

    model = get_model(config)
    data_module = get_datamodule(config)
    if hasattr(model, "noise_scheduler") and getattr(data_module, "flow", False):
        data_module.set_noise_scheduler(model.noise_scheduler)

    # If a run_id is specified, load the model weights.
    if pretrained_run_id is not None:
        # If an on-disk path is specified, use that to load the model weights.
        if pretrained_ckpt_dir is not None:
            model_path = pretrained_ckpt_dir
        else:
            # Otherwise, load the model weights from Wandb.
            try:
                artifact = wandb.Api().artifact(f"{group}/{project}/model-{pretrained_run_id}:best")
                model_path = artifact.download()
                model_path = os.path.join(model_path, "model.ckpt")
            except Exception as e:
                log.warn("Model weights were not found on Wandb.")
                return model, data_module

        model_state = torch.load(model_path)["state_dict"]
        if config.model.get("load_in_4bit", False) or config.model.get("load_in_8bit", False):
            model_state = {
                k: v for k, v in model_state.items()
                if not any(x in k for x in ["absmax", "quant_map", "quant_state"])
            }
        model.load_state_dict(model_state)

        log.info(f"Model weights loaded from: {model_path}")

    return model, data_module
