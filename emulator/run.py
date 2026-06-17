import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

@hydra.main(config_path="configs/", config_name="main_config.yaml", version_base=None)
def main(config: DictConfig):
    
    from emulator.train import run_model

    return run_model(config)


if __name__ == "__main__":
    print("cuda available: ",torch.cuda.is_available())
    main()
