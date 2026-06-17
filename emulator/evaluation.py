import math
import torch
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
from torch import Tensor
from tqdm import tqdm
from pathlib import Path
from einops import rearrange
from typing_extensions import TypeAlias
from typing import Dict, DefaultDict, List, Iterable, Literal, Optional, Protocol, Union, Tuple
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

Dimension: TypeAlias = Union[int, Iterable[int]]
Array: TypeAlias = Union[np.ndarray, torch.Tensor]


def weighted_mean(
    tensor: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    dim: Dimension = (),
    keepdim: bool = False,
) -> torch.Tensor:
    """Computes the weighted mean across the specified list of dimensions.

    Args:
        tensor: torch.Tensor
        weights: Weights to apply to the mean.
        dim: Dimensions to compute the mean over.
        keepdim: Whether the output tensor has `dim` retained or not.

    Returns:
        a tensor of the weighted mean averaged over the specified dimensions `dim`.
    """
    if weights is None:
        return tensor.mean(dim=dim, keepdim=keepdim)
    try:
        return (tensor * weights).sum(dim=dim, keepdim=keepdim) / weights.expand(tensor.shape).sum(
            dim=dim, keepdim=keepdim
        )
    except RuntimeError as e:
        raise RuntimeError(
            f"Error computing weighted mean. tensor.shape={tensor.shape}, weights.shape={weights.shape}, dim={dim}"
        ) from e


def weighted_mean_bias(
    truth: Tensor,
    predicted: Tensor,
    weights: Optional[Tensor] = None,
    dim: Dimension = (),
) -> Tensor:
    """Computes the mean bias across the specified list of dimensions assuming
    that the weights are applied to the last dimensions, e.g. the spatial dimensions.

    Args:
        truth: Tensor
        predicted: Tensor
        dim: Dimensions to compute the mean over.
        weights: Weights to apply to the mean.

    Returns a tensor of the mean biases averaged over the specified dimensions `dim`.
    """
    assert truth.shape == predicted.shape, "Truth and predicted should have the same shape."
    bias = predicted - truth
    return weighted_mean(bias, weights=weights, dim=dim)


def mean_squared_error(
    truth: Tensor,
    predicted: Tensor,
    weights: Optional[Tensor] = None,
    dim: Dimension = (),
) -> Tensor:
    """
    Computes the weighted global MSE over all variables. Namely, for each variable:

        sqrt((weights * ((xhat - x) ** 2)).mean(dims))

    If you want to compute the MSE over the time dimension, then pass in
    `truth.mean(time_dim)` and `predicted.mean(time_dim)` and specify `dims=space_dims`.

    Args:
        truth: Tensor whose last dimensions are to be weighted
        predicted: Tensor whose last dimensions are to be weighted
        weights: Tensor to apply to the squared bias.
        dim: Dimensions to average over.

    Returns a tensor of shape (variable,) of weighted RMSEs.
    """
    assert (
        truth.shape == predicted.shape
    ), f"Truth and predicted should have the same shape. But got {truth.shape} and {predicted.shape}."
    sq_bias = torch.square(predicted - truth)
    return weighted_mean(sq_bias, weights=weights, dim=dim)


def root_mean_squared_error(
    truth: Tensor,
    predicted: Tensor,
    weights: Optional[Tensor] = None,
    dim: Dimension = (),
) -> Tensor:
    """
    Computes the weighted global RMSE over all variables. Namely, for each variable:

        sqrt((weights * ((xhat - x) ** 2)).mean(dims))

    If you want to compute the RMSE over the time dimension, then pass in
    `truth.mean(time_dim)` and `predicted.mean(time_dim)` and specify `dims=space_dims`.

    Args:
        truth: Tensor whose last dimensions are to be weighted
        predicted: Tensor whose last dimensions are to be weighted
        weights: Tensor to apply to the squared bias.
        dim: Dimensions to average over.

    Returns a tensor of shape (variable,) of weighted RMSEs.
    """
    mse = mean_squared_error(truth, predicted, weights=weights, dim=dim)
    return torch.sqrt(mse)


def ensemble_spread(predicted: Tensor, weights: Optional[Tensor] = None, dim: Dimension = ()) -> Tensor:
    """Compute the spread of the ensemble members.
    This is calculated as the square root of the average ensemble variance,
    which is different from the standard deviation of the ensemble.
    See Fortuin et al. 2013 for more details why the square root of the average ensemble variance is adequate.
    Args:
        predicted (torch.Tensor): The predictions of the ensemble, of shape (n_member, n_samples, *)
    """
    mean_ensemble_variance = weighted_mean(predicted.var(dim=0), weights=weights, dim=dim)
    return torch.sqrt(mean_ensemble_variance)


def spread_skill_ratio(
    truth: Tensor, predicted: Tensor, weights: Optional[Tensor] = None, dim: Dimension = ()
) -> Tensor:
    """Compute the spread-skill ratio (SSR) of an ensemble of predictions.
    The SSR is defined as the ratio of the ensemble spread to the ensemble-mean RMSE.
    Args:
        predicted (torch.Tensor): The predictions of the ensemble, of shape (n_member, n_samples, *)
        truth (torch.Tensor): The targets, of shape (n_samples, *)
        weights (torch.Tensor, optional): The weights to apply to the spread. Defaults to None.
        dim (Dimension, optional): The dimensions over which to compute the spread. Defaults to ().
    """
    assert len(truth.shape) == len(predicted.shape) - 1, f"{truth.shape=} and {predicted.shape=}"
    n_mems = predicted.shape[0]
    spread = ensemble_spread(predicted, weights=weights, dim=dim)
    # calculate skill as ensemble_mean RMSE
    rmse = root_mean_squared_error(truth, predicted.mean(dim=0), weights=weights, dim=dim)
    # Add correction factor sqrt((M+1)/M); see https://doi.org/10.1175/JHM-D-14-0008.1), important for small ensemble sizes
    spread *= ((n_mems + 1) / n_mems) ** 0.5
    return spread / rmse


def crps_ensemble(
    truth: Tensor,  # TRUTH
    predicted: Tensor,  # FORECAST
    weights: Tensor = None,
    dim: Union[int, Iterable[int]] = (),
    reduction="mean",
) -> Tensor:
    """
    .. Author: Salva Rühling Cachay

    pytorch adaptation of https://github.com/TheClimateCorporation/properscoring/blob/master/properscoring/_crps.py#L187
    but implementing the fair, unbiased CRPS as in Zamo & Naveau (2018; https://doi.org/10.1007/s11004-017-9709-7)

    This implementation is based on the identity:
    .. math::
        CRPS(F, x) = E_F|X - x| - 1/2 * E_F|X - X'|
    where X and X' denote independent random variables drawn from the forecast
    distribution F, and E_F denotes the expectation value under F.

    We use the fair, unbiased formulation of the ensemble CRPS, which is particularly important for small ensembles.
    Anecdotically, the unbiased CRPS leads to slightly smaller (i.e. "better") values than the biased version.
    Basically, we use n_members * (n_members - 1) instead of n_members**2 to average over the ensemble spread.
    See Zamo & Naveau (2018; https://doi.org/10.1007/s11004-017-9709-7) for details.

    Alternative implementation: https://github.com/NVIDIA/modulus/pull/577/files
    """
    assert truth.ndim == predicted.ndim - 1, f"{truth.shape=}, {predicted.shape=}"
    assert truth.shape == predicted.shape[1:]  # ensemble ~ first axis
    n_members = predicted.shape[0]
    skill = (predicted - truth).abs().mean(dim=0)
    # insert new axes so forecasts_diff expands with the array broadcasting
    # torch.unsqueeze(predictions, 0) has shape (1, E, ...)
    # torch.unsqueeze(predictions, 1) has shape (E, 1, ...)
    forecasts_diff = torch.unsqueeze(predicted, 0) - torch.unsqueeze(predicted, 1)
    # Forecasts_diff has shape (E, E, ...)
    # Old version: score += - 0.5 * forecasts_diff.abs().mean(dim=(0, 1))
    # Using n_members * (n_members - 1) instead of n_members**2 is the fair, unbiased CRPS. Better for small ensembles.
    spread = forecasts_diff.abs().sum(dim=(0, 1)) / (n_members * (n_members - 1))
    crps = skill - 0.5 * spread
    # score has shape (...)  (same as observations)
    if reduction == "none":
        return crps
    assert reduction == "mean", f"Unknown reduction {reduction}"
    if weights is not None:  # weighted mean
        crps = (crps * weights).sum(dim=dim) / weights.expand(crps.shape).sum(dim=dim)
    else:
        crps = crps.mean(dim=dim)
    return crps


def _pairwise_abs_mean(x: torch.Tensor) -> torch.Tensor:
    """
    x : (E, S, H, W)  – ensemble members first
    returns a tensor of shape (S, H, W) with the unbiased mean
            |x_i - x_j|  over all i<j
    Memory: O(E)  instead of  O(E²) .
    """
    E = x.size(0)
    if E < 2:
        # No spread with a single member
        return torch.zeros_like(x[0])
    acc = 0.                          # will hold the running sum
    for i in range(E - 1):
        acc += (x[i] - x[i + 1:]).abs().sum(0)   # (S,H,W)

    return acc / (E * (E - 1) / 2)               # mean over pairs


def crps_ensemble_memory_efficient(truth, predicted, weights=None, dim=(), reduction="mean"):
    """
    predicted : (E,S,H,W)
    truth     : (S,H,W)
    """
    # skill term (same as before)
    skill = (predicted - truth).abs().mean(0)

    # unbiased spread without the E×E blow-up
    spread = _pairwise_abs_mean(predicted)

    crps = skill - 0.5 * spread                      # (S,H,W)

    # ---------------------------------------------------
    if reduction == "none":
        return crps

    if weights is not None:
        crps = (crps * weights).sum(dim=dim) / weights.expand(crps.shape).sum(dim=dim)
    else:
        crps = crps.mean(dim=dim)

    return crps


def spread_ratio(
        truth: Tensor,
        predicted: Tensor,
        weights: Optional[Tensor] = None,
        dim: Dimension = (),
    ) -> Tensor:
    """
    Ratio  sigma_pred / sigma_truth using latitude-weighted spatial variance and then
    averaging over time. `predicted` is (Kp,T,H,W); `truth` is (Kt,T,H,W).
    """
    # var over members, then weighted mean over space
    pred_var  = predicted.var(dim=0, unbiased=False)             # (T,H,W)
    truth_var = truth.var(dim=0, unbiased=False)            # (T,H,W)
    sigma_pred  = weighted_mean(pred_var,  weights, dim=dim).sqrt().mean()
    sigma_truth = weighted_mean(truth_var, weights, dim=dim).sqrt().mean()
    return (sigma_pred / sigma_truth).item() if sigma_truth > 0 else float("nan")


def energy_score(
    truth: Tensor,                   #    (K_truth, …)
    predicted: Tensor,               #    (K_pred , …)
    weights: Optional[Tensor] = None,
    dim: Dimension = (),             # axes to aggregate over
) -> Tensor:
    """
    Multivariate Energy Score
        ES = E‖X-Y‖ - 0.5 E‖X-X'‖ - 0.5 E‖Y-Y'‖
    • `truth`      : ensemble of reference fields   (member dim = 0)
    • `predicted`  : ensemble of forecast fields    (member dim = 0)
    • `weights`    : optional positive weights, broadcast-compatible
    • `dim`        : dimensions to reduce (default: *all* non-member dims)
    """
    assert truth.ndim == predicted.ndim, "truth/predicted rank mismatch"
    assert truth.shape[1:] == predicted.shape[1:], "field shapes differ"

    # 1) canonicalise `dim` into a sorted tuple of positive axes
    if dim == () or dim is None:
        reduce_axes: Tuple[int, ...] = tuple(range(1, truth.ndim))  # all non-member axes
    elif isinstance(dim, (list, tuple)):
        reduce_axes = tuple(sorted(d if d >= 0 else truth.ndim + d for d in dim))
    else:
        reduce_axes = (dim if dim >= 0 else truth.ndim + dim,)

    # 2) flatten the reduction axes into one trailing dimension D
    keep_axes = [ax for ax in range(1, truth.ndim) if ax not in reduce_axes]
    perm    = [0] + keep_axes + list(reduce_axes)            # member dim first
    Kp, Kt  = predicted.shape[0], truth.shape[0]
    D       = int(torch.prod(torch.tensor([truth.shape[d] for d in reduce_axes])))
    pred_f  = predicted.permute(*perm).reshape(Kp, -1, D)    # (Kp, S, D)  S=∏ keep_axes (often 1)
    truth_f = truth.permute(*perm).reshape(Kt, -1, D)        # (Kt, S, D)

    # 3) weights → (S,D) broadcast like the fields after flattening
    if weights is not None:
        w_f = weights.permute(*([ax - 1 for ax in keep_axes] + [ax - 1 for ax in reduce_axes]))
        w_f = w_f.reshape(-1, D)                              # (S,D)
        w_f = w_f / w_f.sum()                                 # normalise
    else:
        w_f = None

    # ---- helper: weighted L2 distance along last dim ------------------
    def _w_l2(a: Tensor, b: Tensor, w: Optional[Tensor]) -> Tensor:  # returns (S,)
        diff2 = (a - b).pow(2)
        if w is not None:
            diff2 = diff2 * w
        return diff2.sum(-1).sqrt()                             # (S,)

    # term1: E‖X−Y‖
    term1 = _w_l2(
        pred_f.unsqueeze(1),                                   # (Kp,1,S,D)
        truth_f.unsqueeze(0),                                  # (1,Kt,S,D)
        w_f,
    ).mean((0, 1)).mean()                                      # scalar

    # term2: E‖X−X'‖
    term2 = _w_l2(
        pred_f.unsqueeze(1), pred_f.unsqueeze(0), w_f
    ).mean((0, 1)).mean()

    # term3: E‖Y−Y'‖
    term3 = _w_l2(
        truth_f.unsqueeze(1), truth_f.unsqueeze(0), w_f
    ).mean((0, 1)).mean()

    return term1 - 0.5 * term2 - 0.5 * term3


def _lat_full(lat_vec: Tensor, hw: Tuple[int, int]) -> Tensor:
    """Expand latitude weights to the matching spatial axis."""
    H, W = hw
    if lat_vec.shape[-1] == H:
        return lat_vec.view(H, 1).repeat(1, W)
    if lat_vec.shape[-1] == W:
        return lat_vec.view(1, W).repeat(H, 1)
    raise ValueError(f"Latitude length {lat_vec.shape[-1]} does not match spatial shape {(H, W)}.")


def _torchmetrics_obj(cls, cache, device="cpu"):
    """Cache torchmetrics instances per device."""
    key = str(device)
    if key not in cache:
        cache[key] = cls().to(device)
    return cache[key]


class Evaluator:
    # --------------------------- defaults --------------------------------
    _DEF_METRICS = [
        "mse",
        "rmse",
        "abs_error",
        "llmse_climax",
        "llrmse_climax",
        "abs_error_climax",
        "bias",
        # "ssim",
        # "psnr",
    ]
    _DEF_ENSEMBLE = [
        "crps",
        "spread_skill_ratio",
        "spread_ratio",
        # "energy_score"
    ]
    _TIME_DIM = (-3, )
    _SPACE_DIMS = (-2, -1)  # spatial dimensions are always the last two (H, W)

    # --------------------------- constructor -----------------------------
    def __init__(
        self,
        *,
        cache_dir: str | Path,
        metrics: Optional[List[str]] = None,
        ensemble_metrics: Optional[List[str]] = None,
        no_overwrite: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if not no_overwrite:
            # start fresh if directory existed
            for file in self.cache_dir.iterdir():
                file.unlink()
        print("Saving evaluation cache at:", self.cache_dir)
        self.metrics = metrics or self._DEF_METRICS.copy()
        self.ensemble_metrics = ensemble_metrics if ensemble_metrics is not None else self._DEF_ENSEMBLE.copy()

        self._pred_files: DefaultDict[str,Dict[str,Dict[str,List[str]]]] = \
            defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self._truth_files: DefaultDict[str,Dict[str,Dict[str,List[str]]]] = \
            defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self._lat_w: DefaultDict[str, Dict[str, Tensor]] = \
            defaultdict(dict)

        # torchmetrics caches
        self._ssim: dict[str, StructuralSimilarityIndexMeasure] = {}
        self._psnr: dict[str, PeakSignalNoiseRatio] = {}

        self.metric2_fn: Dict[str, callable] = {
            "mse": mean_squared_error,
            "rmse": root_mean_squared_error,
            "abs_error": lambda y, p: (p - y).abs().mean(),
            "llmse_climax": self._llmse,
            "llrmse_climax": self._llrmse,
            "abs_error_climax": lambda y, p, weights, dim: weighted_mean((p - y).abs(), weights=weights, dim=dim).mean(),
            "bias": weighted_mean_bias,
            "ssim": lambda y, p: _torchmetrics_obj(StructuralSimilarityIndexMeasure, self._ssim, p.device)(y, p),
            "psnr": lambda y, p: _torchmetrics_obj(PeakSignalNoiseRatio, self._psnr, p.device)(y, p),
            "crps": crps_ensemble_memory_efficient,
            "spread_skill_ratio": spread_skill_ratio,
            "spread_ratio": spread_ratio,
            "energy_score": energy_score,
        }

        for metric in self.metrics:
            if metric not in self.metric2_fn:
                raise ValueError(f"Unknown metric: {metric}")

    # --------------------------- update ----------------------------------
    @torch.no_grad()
    def update(
        self,
        preds: Dict[str, Tensor],          # var → (B, M_sample, T, C, H, W)
        targets: Dict[str, Tensor],        # var → (B,           T, C, H, W)
        climate_models: List[str],
        scenarios: List[str],
        members: List[str],
        lat_weights: Tensor,               # (B, H)
        time, # (B, T)
        batch_idx: int,
    ) -> None:
        # Save time to cache dir
        for b in range(len(time)):
            scen_key = f"{climate_models[b]}_{scenarios[b]}"
            mem_key  = members[b]
            f_time = self.cache_dir / f"time_{scen_key}_{mem_key}_batch_{batch_idx}_{b}.pt"
            torch.save(time[b], f_time)

        for var, p in preds.items():
            y = targets[var]
            p = p.squeeze(3) # Squeeze out the channel dimension
            y = y.squeeze(2) # Squeeze out the channel dimension
            B, _, _, H, W = p.shape
            for b in range(B):
                scen_key = f"{climate_models[b]}_{scenarios[b]}"   # no member
                mem_key  = members[b]

                f_pred = self.cache_dir / f"{var}_{scen_key}_{mem_key}_pred_{batch_idx}_{b}.pt"
                torch.save(p[b].cpu().unsqueeze(1), f_pred)          # (M,N,T,H,W)
                self._pred_files[var][scen_key][mem_key].append(f_pred)

                f_truth= self.cache_dir / f"{var}_{scen_key}_{mem_key}_truth_{batch_idx}_{b}.pt"
                torch.save(y[b].cpu().unsqueeze(0), f_truth)         # (N,T,H,W)
                self._truth_files[var][scen_key][mem_key].append(f_truth)

                if scen_key not in self._lat_w[var]:
                    self._lat_w[var][scen_key] = _lat_full(lat_weights[b].cpu(), (H, W))

    # ------------------------ metric helpers -----------------------------
    @staticmethod
    def _llmse(pred: Tensor, truth: Tensor, weights: Tensor, dim: Dimension) -> float:
        return mean_squared_error(truth, pred, weights=weights, dim=dim).mean().item()

    @staticmethod
    def _llrmse(pred: Tensor, truth: Tensor, weights: Tensor, dim: Dimension) -> float:
        return root_mean_squared_error(truth, pred, weights=weights, dim=dim).mean().item()

    def _load_cat(self, paths: List[str], cat_dim: int) -> Tensor:
        """Load a list of .pt files and concatenate along cat_dim."""
        tensors = [torch.load(f, map_location="cpu") for f in paths]
        return torch.cat(tensors, dim=cat_dim)

    # --------------------------- compute ---------------------------------
    def compute(self) -> Dict[str, float]:
        out: Dict[str, float] = {}

        # --- helpers for later global averages ---------------------------
        scen_metric_sum: Dict[str, Dict[str, float]]   = defaultdict(lambda: defaultdict(float))
        scen_metric_cnt: Dict[str, Dict[str, int]]     = defaultdict(lambda: defaultdict(int))

        # loop over variable and scenario-member keys
        for var, scen_dict in self._pred_files.items():
            print(f"Computing {var} metrics for {len(scen_dict)} scenarios…")
            for scen_key, mem_dict in tqdm(scen_dict.items(), total=len(scen_dict)):
                pred_full = []
                truth_full= []
                for mem, pfiles in mem_dict.items():
                    pred_full.append(self._load_cat(pfiles, cat_dim=1))   # (M,T,H,W) cat on N-axis later
                    truth_full.append(self._load_cat(self._truth_files[var][scen_key][mem], cat_dim=0))  # (N,T,H,W)

                if self.ensemble_metrics:
                    pred_full  = torch.cat(pred_full, dim=0)           # (ΣM, N, T, H, W)
                    truth_full = torch.stack(truth_full, dim=0)        # (K_truth, N, T, H, W)

                    # # collapse N and T dims for metric code reuse
                    pred_full  = rearrange(pred_full , 'm n t h w -> m (n t) h w')
                    truth_full = rearrange(truth_full, 'k n t h w -> k (n t) h w')

                    # ensemble means
                    p_mean = pred_full.mean(dim=0) # (N*T, H, W)
                    y_mean = truth_full.mean(dim=0) # (N*T, H, W)

                else:
                    pred_mean = torch.cat(pred_full, dim=1)  # (1, N, T, H, W)
                    pred_mean = pred_mean.squeeze(0)  # (N, T, H, W)
                    truth_mean = torch.cat(truth_full, dim=0)  # (N, T, H, W)

                    # collapse N and T dims for metric code reuse
                    p_mean = rearrange(pred_mean, 'n t h w -> (n t) h w')  # (N*T, H, W)
                    y_mean = rearrange(truth_mean, 'n t h w -> (n t) h w')  # (N*T, H, W)

                lat_w      = self._lat_w[var][scen_key]               # (H,W)
                w3d        = lat_w.unsqueeze(0).repeat(p_mean.size(0), 1, 1)

                base = f"{scen_key}/{var}"

                for metric in self.metrics + self.ensemble_metrics:
                    calc_fn = self.metric2_fn[metric]
                    if metric in {"llmse_climax", "llrmse_climax", "bias", "abs_error_climax"}:
                        val = calc_fn(y_mean, p_mean, weights=w3d, dim=self._TIME_DIM + self._SPACE_DIMS)
                    elif metric in {"crps", "spread_skill_ratio"}:
                        val = calc_fn(
                            y_mean, pred_full, weights=w3d, dim=self._TIME_DIM + self._SPACE_DIMS
                        )
                    elif metric in {"spread_ratio", "energy_score"}:
                        val = calc_fn(
                            truth_full, pred_full, weights=w3d, dim=self._TIME_DIM + self._SPACE_DIMS
                        )
                    elif metric in {"ssim", "psnr"}:
                        img_p, img_y = p_mean.unsqueeze(1), y_mean.unsqueeze(1)
                        val = calc_fn(img_y, img_p)
                    else:
                        val = calc_fn(y_mean, p_mean)
                    val = val.item() if isinstance(val, Tensor) else val

                    out[f"{base}/{metric}"] = val
                    scen_metric_sum[metric][scen_key] += val
                    scen_metric_cnt[metric][scen_key] += 1

        # ------------- overall averages ------------------------
        for m in self.metrics + self.ensemble_metrics:
            if m not in scen_metric_sum:
                continue
            for sc, tot in scen_metric_sum[m].items():
                avg_val = tot / scen_metric_cnt[m][sc]
                out[f"{sc}/avg/{m}"] = avg_val

            # ---------- global average across scenarios & variables -----
            global_vals = [
                out[f"{sc}/avg/{m}"] for sc in scen_metric_sum[m].keys()
            ]
            if global_vals:
                out[m] = sum(global_vals) / len(global_vals)

        return out

    # --------------------------- reset -----------------------------------
    def reset(self) -> None:
        self._pred_files.clear()
        self._truth_files.clear()
        self._lat_w.clear()
        self._ssim.clear()
        self._psnr.clear()
