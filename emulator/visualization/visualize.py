import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
from itertools import cycle
import calendar


def create_predictions_fig(preds: dict, targets: dict, metadata: dict, n: int = 0, t: int = 0):
    """Generates a figure object for a plot that compares the predictions,
    ground truth, and their difference. The figure is a subplot with two
    rows and three columns. The first row shows the "tas" results, while
    the second row shows the "pr" results. The figure only shows a single
    example for a particular time step, as specified by the arguments.

    Args:
        preds (dict): Dict that contains "tas" and "pr" keys,
        each with an array of shape (N, 1, T, lon, lat) that
        contains the model predictions. N is the number of examples,
        and T in the time step.

        targets (dict): Dict that contains "tas" and "pr" keys,
        each with an array of shape (N, 1, T, lon, lat) that
        contains the ground truth. N is the number of examples,
        and T in the time step.

        metadata (dict): Dict that contains the metadata of the inputs.
        It is used to obtain the longitude and latitude arrays.

        n (int or list, optional): Index or indices of the example(s) to be plotted.
        Defaults to 0.
        t (int, optional): Time step index to be plotted.
        Defaults to 0.

    Returns:
        matplotlib.figure: Matplotlib figure object with the plot.
    """
    output_vars = list(preds.keys())

    if isinstance(n, int):
        n = [n]

    num_vars = len(output_vars)
    fig, axes = plt.subplots(
        num_vars, 3*len(n), figsize=(3*4*len(n), 8), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    for i in n:

        lon_array = metadata["longitude"][i]
        lat_array = metadata["latitude"][i]

        for var_index, var in enumerate(output_vars):

            if var == "tas":
                cmap = "coolwarm"
            elif var == "pr":
                cmap = "viridis"

            # Transpose to make sure the image is horizontal
            sample_target = targets[var][i][0, t, :, :].T
            sample_pred = preds[var][i][0, t, :, :].T
            sample_diff = sample_pred - sample_target

            # Match the colorbar scale of target and predictions for easier comparison
            vmin = sample_target.min()
            vmax = sample_target.max()

            data = {"target": sample_target, "pred": sample_pred, "diff": sample_diff}

            for fig_index, fig_name in enumerate(data):

                da = xr.DataArray(
                    data[fig_name], coords=[lat_array, lon_array], dims=["lat", "lon"]
                )
                # var1 target, var1 prediction, var1 difference, var1 target, ...]
                _ax = axes[var_index, 3*i + fig_index]

                if fig_name == "diff":
                    p = da.plot(
                        ax=_ax, add_colorbar=False, cmap="coolwarm"
                    )

                    _ax.coastlines()
                    _ax.set_title(fig_name)
                else:
                    p = da.plot(
                        ax=_ax,
                        add_colorbar=False,
                        vmin=vmin,
                        vmax=vmax,
                        cmap=cmap,
                    )

                    _ax.coastlines()
                    _ax.set_title(fig_name)

                # Create colorbar and match its size to the image
                fig.colorbar(p, ax=_ax, fraction=0.03, pad=0.04)

                # Disable latitude and longitude ticks
                _ax.set_xticks([])
                _ax.set_yticks([])
            axes[var_index, 0].set_ylabel(var.upper())

    plt.tight_layout()

    return fig


def create_temporal_predictions_figs(
    preds: dict,
    targets: dict,
    metadata: dict,
    n: int = 0,
    timescale="monthly"  # "monthly", "yearly", "decadal"
):
    """
    Produce one multi-panel figure **per variable**.
    Each figure has 12 rows (months) × 3 columns
    (target, prediction, target-prediction).

    Parameters
    ----------
    preds, targets : dict
        Dicts keyed by variable name ("tas", "pr", …) whose values
        have shape (N, 12, 1, lon, lat).
    metadata : dict
        Contains "longitude" and "latitude" arrays with shape (N, lon/lat).
    n : int or list, default 0
        Example index or indices in the batch to visualise.
    timescale : str, default "monthly"
        The timescale of the predictions, e.g. "monthly", "yearly", "

    Returns
    -------
    dict
        {variable_name: matplotlib.figure.Figure}
    """
    num_vars = len(preds)

    t = preds[next(iter(preds))].shape[1]  # time dimension, e.g. 12 for monthly

    if timescale == "monthly":
        # Only plot first 12 months
        t = min(t, 12)
        names = [calendar.month_abbr[i + 1] for i in range(t)]
    elif timescale == "yearly":
        names = [f"Year {i + 1}" for i in range(t)]
    elif timescale == "decadal":
        names = [f"Decade {i + 1}" for i in range(t)]
    else:
        raise ValueError(f"Unsupported timescale: {timescale}. "
                         f"Supported values are 'monthly', 'yearly', 'decadal'.")
    
    if isinstance(n, int):
        n = [n]

    fig, ax = plt.subplots(
        t,
        3*num_vars*len(n),
        figsize=(9*num_vars*len(n), 2.1 * t),
        subplot_kw={"projection": ccrs.PlateCarree()},
        squeeze=False,
    )

    for i in n:

        lon = metadata["longitude"][i]
        lat = metadata["latitude"][i]

        for j, var in enumerate(preds):
            # choose suitable colormap for target/pred
            cmap = {"tas": "coolwarm", "pr": "viridis"}.get(var, "viridis")

            # pull the full year and transpose (T, lat, lon) for convenience
            targ = targets[var][i][:, 0].transpose(0, 2, 1)
            pred = preds[var][i][:, 0].transpose(0, 2, 1)
            diff = targ - pred

            # common limits so colours match across months
            vmin, vmax = targ.min(), targ.max()

            for m in range(t):
                for c, (field, data) in enumerate(
                    zip(("target", "pred", "diff"), (targ[m], pred[m], diff[m]))
                ):
                    _ax = ax[m, 3*(i*len(preds) + j) + c]
                    da = xr.DataArray(data, coords=[lat, lon], dims=["lat", "lon"])
                    p = da.plot(
                        ax=_ax,
                        add_colorbar=False,
                        cmap="coolwarm" if field == "diff" else cmap,
                        vmin=None if field == "diff" else vmin,
                        vmax=None if field == "diff" else vmax,
                    )
                    _ax.coastlines()
                    _ax.set_title(field.capitalize() if m == 0 else "")
                    _ax.set_xticks([])
                    _ax.set_yticks([])
                    if c == 0:
                        _ax.set_ylabel("")

                    # colour-bar per panel keeps every plot self-contained
                    fig.colorbar(p, ax=_ax, fraction=0.035, pad=0.03)

                # month label on the left of each row
                ax[m, 0].text(
                    -0.07,
                    0.5,
                    names[m],
                    va="center",
                    ha="right",
                    transform=ax[m, 0].transAxes,
                    fontsize=10,
                )

    plt.tight_layout(h_pad=0.7, w_pad=0.3)

    return fig


def global_mean_plot(
    inputs,
    preds,
    targets,
    time,
    model,
    ssp,
    em,
    latitude=None,
    longitude=None,
    n: int = 0,
    ax=None,
    split=None,
    is_means: bool = False,
    is_sample_mean: bool = False,
    line_styles=None,
    *,
    per_time_columns: bool = False,        # NEW
    period_labels: list[str] | None = None # NEW
):
    """
    Plot area-weighted global-mean time-series.

    ───── Layout options ─────
    • Default:  (# SSPs  rows) × (# variables  columns); x-axis = time.
    • If ``per_time_columns`` is True:
        (# SSPs rows) × (T * #variables columns) where
        T = number of time slices (e.g. 12 months).  
        For var * tas * month * January you now get its
        own column whose x-axis is still “Time”.
    • If ``is_sample_mean`` is False, assumes that ``em`` is a list of ensemble member names
        where each member has a sample suffix like "_1", "_2", etc. and plots each sample and member separately.

      You may pass ``period_labels`` (len == T) to rename those
      columns (e.g. ["Jan", "Feb", … "Dec"]).
    """
    # ───────── basic bookkeeping ─────────
    input_vars  = list(inputs.keys()) if inputs else []
    output_vars   = list(preds.keys())
    model_ssp     = np.array([f"{m} {s}" for m, s in zip(model, ssp)])
    unique_ssps   = np.unique(model_ssp)
    unique_ems    = np.unique(em)
    time         = np.asarray(time).flatten()

    T = preds[list(preds.keys())[0]].shape[1]
    if per_time_columns and period_labels:
        assert len(period_labels) == T, "period_labels length must equal T"
    labels = period_labels

    # ───────── figure & axes ─────────
    all_vars = input_vars + output_vars
    n_rows = len(all_vars) * len(unique_ssps)
    n_cols = T
    if ax is None:
        fig, ax = plt.subplots(
            n_rows, n_cols,
            figsize=(4 * n_cols, 3.5 * n_rows),
            sharex='col', squeeze=False
        )
    else:
        fig = ax[0].figure if isinstance(ax, np.ndarray) else ax.figure
        ax  = np.atleast_2d(ax)

    def _row(var_idx: int, ssp_idx: int) -> int:
        """Row index in the new layout."""
        return var_idx * len(unique_ssps) + ssp_idx

    def _col(p: int) -> int:
        """Column index (only period‑slice if requested)."""
        return p if per_time_columns else 0

    var2title = {"tas": "Temperature", "pr": "Precipitation"}

    # ───────── helper for annual mean ─────────
    def annual_series(arr, times):
        uniq = np.unique(times)
        return uniq, np.asarray([arr[times == y].mean() for y in uniq])

    style_cycle_default = ["-", "--", ":", "-."]
    for var_idx, var in enumerate(all_vars):
        is_input = var in input_vars

        for ssp_idx, ssp_val in enumerate(unique_ssps):
            ssp_mask = (model_ssp == ssp_val)

            for p in range(T):                                   # p == 0 if annual
                r, c  = _row(var_idx, ssp_idx), _col(p)
                axis  = ax[r, c]
                axis.set_xlabel("Time")
                axis.set_ylabel(f"Global Mean {var2title.get(var, var)}")

                title_var = var2title.get(var, var)
                slice_lbl = labels[p] if (per_time_columns and labels) else ""
                axis.set_title(
                    f"{split or ''} {ssp_val} {title_var} {slice_lbl}".strip()
                )

                style_cycle = cycle(line_styles or style_cycle_default)

                # -------- plot per ensemble member --------
                for em_val in unique_ems:
                    mask = ssp_mask & (em == em_val)
                    em_val_label = em_val if is_sample_mean else em_val.split("_")[0]
                    if not mask.any():
                        continue

                    if is_input:              # <<< NEW branch
                        arr_raw = np.asarray(inputs[var]).squeeze()[mask]

                        if not is_means:      # latitude-weighted mean
                            lat_array = latitude[n]
                            lon_array = longitude[n]
                            da = xr.DataArray(
                                arr_raw, coords=[time[mask], lon_array, lat_array],
                                dims=["time", "lon", "lat"]
                            )
                            w = np.cos(np.deg2rad(da.lat))
                            inp = da.weighted(w).mean(("lat", "lon")).values
                        else:
                            inp = arr_raw

                        _time = time[mask]
                        order = np.argsort(_time)
                        _time, inp = _time[order], inp[order]

                        _time, inp = annual_series(inp, _time)

                        ls = next(style_cycle)
                        axis.plot(_time, inp, linestyle=ls, color="k",
                                label=f"Input EM{em_val_label}")

                    else:                     # original preds / targets logic
                        # ---- slice + global mean weighting ----
                        if is_means:
                            tgt = np.asarray(targets[var]).squeeze()[mask]
                            pred = np.asarray(preds[var]).squeeze()[mask]
                        else:
                            lon_array = longitude[n]
                            lat_array = latitude[n]
                            t_raw = np.asarray(targets[var]).squeeze()[mask]
                            p_raw = np.asarray(preds[var]).squeeze()[mask]

                            t_da = xr.DataArray(
                                t_raw, coords=[time[mask], lon_array, lat_array],
                                dims=["time", "lon", "lat"]
                            )
                            p_da = xr.DataArray(
                                p_raw, coords=[time[mask], lon_array, lat_array],
                                dims=["time", "lon", "lat"]
                            )
                            w = np.cos(np.deg2rad(t_da.lat))
                            tgt = t_da.weighted(w).mean(("lat", "lon")).values
                            pred = p_da.weighted(w).mean(("lat", "lon")).values

                        # ---- build x-axis + plot ----
                        _time = time[mask]
                        order = np.argsort(_time)
                        _time, tgt, pred = _time[order], tgt[order], pred[order]

                        if not per_time_columns:          # collapse to annual means
                            _time, tgt = annual_series(tgt, _time)
                            _, pred = annual_series(pred, _time)
                        else:
                            tgt = tgt[:, p]
                            pred = pred[:, p]

                        ls = next(style_cycle)
                        axis.plot(_time, tgt,  color="b", linestyle=ls,
                                label=f"Actual EM{em_val_label}")
                        axis.plot(_time, pred, color="r", linestyle=ls,
                                label=f"Pred EM{em_val_label}")

                # unique legend per axis
                h, l = axis.get_legend_handles_labels()
                uniq = {lab: hdl for lab, hdl in zip(l, h)}
                axis.legend(uniq.values(), uniq.keys(), fontsize=7)

    fig.tight_layout()
    return fig

