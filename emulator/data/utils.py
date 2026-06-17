import torch
import pickle
import xarray as xr
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List
from emulator.data.constants import STATS_DIR


def _stats_path(model: str, IB: bool, ts: str, kind: str, min_max_norm: bool, log_vars: list[str] = []) -> Path:
    """
    kind = "inputs" | "outputs"
    """
    if IB:
        model = f"{model}_IB"
    if min_max_norm:
        model = f"{model}_minmax"
    if len(log_vars) > 0:
        model = f"{model}_log{'_'.join(log_vars)}"
    name = f"{kind}_{model}_{ts}.pkl" if kind == "inputs" else f"{model}_{ts}.pkl"
    return STATS_DIR / name


def _add_dim(ds, name, default):
    """
    Make *name* a length-1 dimension coordinate.

    Returns the dataset where *name* is guaranteed to be
    a dim of size 1, carrying a single string label.
    """
    if name in ds.coords and name not in ds.dims:
        ds = ds.drop_vars(name)
        label = default
        ds = ds.expand_dims({name: [label]})

    elif name in ds.dims:
        label = default
        ds[name]  = [label]

    if name not in ds.coords and name not in ds.dims:
        # coord missing entirely
        label = default
        ds = ds.expand_dims({name: [label]})

    return ds


def _fix_dtype(ds):
    # materialise object-dtype coords *before* we turn them into new dims
    for c in ("scenario", "experiment", "member"):
        if c in ds.coords and ds[c].dtype == object:
            ds = ds.assign_coords({c: ds[c].astype(str).compute()})
    return ds


def _compute_stats_across(ds_list: List[xr.Dataset], min_max_norm=False, log_vars: list[str] = []) -> dict:
    print("Computing stats across datasets...")
    # concat along time, letting xarray align coords lazily
    aligned = []
    running_sum  = None
    running_sum2 = None        # for std
    running_n    = 0
    for ds in tqdm(ds_list):
        ds = (_fix_dtype(ds)
            .pipe(_add_dim, "member", "nomem")
            .pipe(_add_dim, "experiment", "noexp")
            .pipe(_add_dim, "scenario", "nosce")
            .drop_vars(("month", "year"), errors="ignore")
            )
        for var in log_vars:
            if var in ds.data_vars:
                # Take ln(x + 1)
                ds[var] = xr.ufuncs.log(ds[var] + 1)
        if not min_max_norm:
            this_sum  = ds.sum(dim=("time", "lat", "lon"), skipna=True).compute()
            this_sum2 = (ds**2).sum(dim=("time", "lat", "lon"), skipna=True).compute()
            if running_sum is None:
                running_sum  = this_sum
                running_sum2 = this_sum2
            else:
                running_sum  = running_sum  + this_sum
                running_sum2 = running_sum2 + this_sum2

            running_n += ds.sizes["time"] * ds.sizes["lat"] * ds.sizes["lon"]
        else:
            aligned.append(ds)
    if not min_max_norm:
        mean = (running_sum  / running_n)
        std  = np.sqrt((running_sum2 / running_n) - mean ** 2)
        return {
            "mean": {v: mean[v].item() for v in mean.data_vars},
            "std":  {v: std [v].item() for v in std .data_vars},
        }
    else:
        ds_big = xr.concat(aligned,  dim="time")
        percentile_1p = ds_big.quantile(0.01,).compute()
        percentile_99p = ds_big.quantile(0.99).compute()
        return {
            "p1": {v: percentile_1p[v].item() for v in percentile_1p.data_vars},
            "p99": {v: percentile_99p[v].item() for v in percentile_99p.data_vars},
        }


def _get_or_make_stats(
    datasets: List[xr.Dataset],
    model: str,
    IB: bool,
    timescale: str,
    kind: str,  # "inputs" | "outputs",
    min_max_norm: bool = False,
    log_vars: list[str] = [],
) -> dict:
    pth = _stats_path(model, IB, timescale, kind, min_max_norm, log_vars)
    if pth.exists():
        with open(pth, "rb") as f:
            return pickle.load(f)
    stats = _compute_stats_across(datasets, min_max_norm, log_vars)
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    with open(pth, "wb") as f:
        pickle.dump(stats, f)
    return stats


def _align_and_prune(inp: xr.Dataset, out: xr.Dataset) -> tuple[xr.Dataset, xr.Dataset]:
    """
    1. Remove any time step where *all* variables are NaN in the inputs OR outputs.
    2. Return the two datasets re-indexed to the common, valid time coordinate.
    """
    keep_in  = inp.dropna(dim="time", how="all").time
    keep_out = out.dropna(dim="time", how="all").time

    # intersection keeps only the indices that survive in both
    common = np.intersect1d(keep_in.values, keep_out.values)

    inp  = inp.sel(time=common)
    out  = out.sel(time=common)
    return inp, out


def _to_month_start(ds: xr.Dataset) -> xr.Dataset:
    """
    Replace each timestamp with the first day of its month
    (e.g. 2015-01-16 → 2015-01-01).  Duplicates that arise after rounding
    are collapsed by taking the mean over the 'time' axis.
    """
    time = ds.time

    # 1. robust "floor to month-start" that works on old xarray versions
    vals = time.values.astype("datetime64[M]").astype("datetime64[ns]")
    rounded = xr.DataArray(vals, dims="time")

    # 2. assign new coordinate
    ds = ds.assign_coords(time=rounded)

    return ds


def normalize(ds: xr.Dataset, stats: dict, log_vars: list, min_max_norm: bool = False) -> xr.Dataset:
    for var in log_vars:
        # Take ln(x + 1)
        if var in ds:
            ds[var] = xr.ufuncs.log(ds[var] + 1)
    if not min_max_norm:
        ds = (ds - stats["mean"]) / stats["std"]
    else:
        # Normalize using min-max normalization
        diff = {var: stats["p99"][var] - stats["p1"][var] for var in ds.data_vars}
        ds = (ds - stats["p1"]) / diff
        # Clip values to [0, 1] (commenting as this prevents the model from learning extremes)
        # ds = ds.clip(min=0, max=1)
        # Shift to [-1, 1]
        ds = ds * 2 - 1  # Same as Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    return ds


def unnormalize(ds: xr.Dataset, stats: dict, log_vars: list, min_max_norm: bool = False) -> xr.Dataset:
    """Unnormalize an xarray dataset using the provided stats."""
    if not min_max_norm:
        ds = ds * stats["std"] + stats["mean"]
    else:
        # Unnormalize using min-max normalization
        # Shift back to [0, 1]
        ds = (ds + 1) / 2
        # Clip values to [0, 1] (commenting as this prevents the model from learning extremes)
        # ds = ds.clip(min=0, max=1)
        diff = {var: stats["p99"][var] - stats["p1"][var] for var in ds.data_vars}
        ds = ds * diff + stats["p1"]
    for var in log_vars:
        # Take exp(x) - 1
        if var in ds:
            ds[var] = xr.ufuncs.exp(ds[var]) - 1
    return ds


def unnormalize_dict(ds: dict, stats: dict, batch_dim: bool = False, log_vars: list = [], min_max_norm: bool = False) -> dict:
    """Unnormalize a dictionary mapping variable names to torch tensors.
    
    If batch_dim is True, the first dimension of each tensor in the ds dict is
    assumed to be the batch dimension.
    """
    if not min_max_norm:
        for var in ds:
            if batch_dim:
                std_stats = stats["std"][var]
                mean_stats = stats["mean"][var]
                # Expand last 4 dimensions
                std_stats = std_stats.view(*std_stats.shape, 1, 1, 1, 1)
                mean_stats = mean_stats.view(*mean_stats.shape, 1, 1, 1, 1)
                ds[var] = ds[var] * std_stats + mean_stats
            else:
                ds[var] = ds[var] * stats["std"][var] + stats["mean"][var]
    else:
        for var in ds:
            # Shift back to [0, 1]
            ds[var] = (ds[var] + 1) / 2
            # Clip values to [0, 1] (commenting as this prevents the model from learning extremes)
            # ds[var] = ds[var].clip(min=0, max=1)
            diff = stats["p99"][var] - stats["p1"][var]
            if batch_dim:
                # Expand last 4 dimensions
                diff = diff.view(*diff.shape, 1, 1, 1, 1)
                p1_stats = stats["p1"][var]
                ds[var] = ds[var] * diff + p1_stats.view(*p1_stats.shape, 1, 1, 1, 1)
            else:
                ds[var] = ds[var] * diff + stats["p1"][var]
    for var in log_vars:
        # Take exp(x) - 1
        if var in ds:
            ds[var] = torch.exp(ds[var]) - 1
    return ds
