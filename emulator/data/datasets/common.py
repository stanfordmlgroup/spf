import dask
import torch
import pickle
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple, Callable, Literal, Iterable

from torch.utils.data import ConcatDataset

from emulator.utils.utils import get_logger
from emulator.data.constants import OPENBURNING_MODEL_MAPPING
from emulator.data.utils import (
    normalize,
    _get_or_make_stats,
    _align_and_prune,
    _to_month_start
)


log = get_logger()

# NOTE: This is needed to avoid deadlock when using multiple workers for data loading
dask.config.set(scheduler='synchronous')

SPATIAL_LAYOUTS = ("swap_lat_lon", "lat_lon")

__all__ = [
    "dask",
    "torch",
    "pickle",
    "np",
    "pd",
    "xr",
    "tqdm",
    "Path",
    "defaultdict",
    "Dict",
    "List",
    "Sequence",
    "Tuple",
    "Callable",
    "Literal",
    "Iterable",
    "ConcatDataset",
    "log",
    "OPENBURNING_MODEL_MAPPING",
    "normalize",
    "_get_or_make_stats",
    "_align_and_prune",
    "_to_month_start",
    "SPATIAL_LAYOUTS",
    "ClimateSuiteDataStore",
    "filter_member_variant_labels",
]

class ClimateSuiteDataStore:
    """Path helper for the extracted ClimateSuite data layout."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    @property
    def grid_path(self) -> Path:
        return self.root / "grid.nc"

    def input_store(self, timescale: str, forcing_model: str, scenario: str, historical_key: str, ib: bool) -> Path:
        if scenario == historical_key:
            filename = f"{forcing_model}_historical.zarr"
        elif ib:
            filename = f"{forcing_model}_IB_{scenario}.zarr"
        else:
            filename = f"{forcing_model}_{scenario}.zarr"
        return self.root / "inputs" / timescale / filename

    def output_store(self, climate_model: str, timescale: str, scenario: str) -> Path:
        if climate_model == "UKESM1-0-LL" and "historical" in scenario.lower():
            scenario = "historical"
        return self.root / "outputs" / climate_model / timescale / f"{scenario}.zarr"


def _variant_label(member: str) -> str:
    """Return the CMIP-style variant label after the realization id, e.g. i1p1f1."""
    start = member.find("i")
    return member[start:] if start >= 0 else member


def filter_member_variant_labels(members: Sequence[str], variant_labels: Sequence[str]) -> list[str]:
    """Keep ensemble members whose CMIP variant label is explicitly requested."""
    allowed_variants = set(variant_labels)
    return [member for member in members if _variant_label(str(member)) in allowed_variants]
