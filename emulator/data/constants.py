import os
from pathlib import Path


DATA_DIR = Path(os.environ.get("SPF_DATA_DIR", "data"))
STATS_DIR = DATA_DIR / "stats"

OPENBURNING_MODEL_MAPPING = {
    "other": ("anthro-fires", "anthro-fires"),
    "ACCESS-CM2": ("all-fires", "all-fires"),
    "ACCESS-ESM1-5": ("all-fires", "all-fires"),
    "BCC-CSM2-MR": ("all-fires", "all-fires"),
    "CAMS-CSM1-0": ("all-fires", "all-fires"),
    "CESM2": ("anthro-fires", "all-fires"),
    "CESM2-WACCM": ("no-fires", "no-fires"),
    "CNRM-ESM2-1": ("anthro-fires", "anthro-fires"),
    "CMCC-ESM2": ("no-fires", "no-fires"),
    "CMCC-CM2-SR5": ("all-fires", "all-fires"),
    "EC-Earth3-Veg": ("anthro-fires", "anthro-fires"),
    "EC-Earth3-Veg-LR": ("anthro-fires", "anthro-fires"),
    "FGOALS-f3-L": ("all-fires", "all-fires"),
    "GFDL-ESM4": ("no-fires", "no-fires"),
    "INM-CM4-8": ("all-fires", "all-fires"),
    "INM-CM5-0": ("all-fires", "all-fires"),
    "IPSL-CM6A-LR": ("all-fires", "all-fires"),
    "KACE-1-0-G": ("all-fires", "all-fires"),
    "MCM-UA-1-0": ("all-fires", "all-fires"),
    "MPI-ESM1-2-LR": ("anthro-fires", "anthro-fires"),
    "MRI-ESM-2.0": ("anthro-fires", "all-fires"),
    "MRI-ESM2-0": ("anthro-fires", "all-fires"),
    "NorESM2-LM": ("no-fires", "no-fires"),
    "NorESM2-MM": ("no-fires", "no-fires"),
    "TaiESM1": ("anthro-fires", "all-fires"),
    "UKESM1-0-LL": ("all-fires", "all-fires"),
}
