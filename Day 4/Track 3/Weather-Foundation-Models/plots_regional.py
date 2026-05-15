# =================== IMPORTS, FILTER WARNINGS =====================
import sys
import time
import warnings
from pathlib import Path
from typing import List
from datetime import datetime

import numpy as np
from scipy import stats
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

warnings.filterwarnings(
    "ignore", message=".*Mean of empty slice.*", category=RuntimeWarning
)
warnings.filterwarnings(
    "ignore", message=".*Precision loss occurred.*", category=RuntimeWarning
)

# ======================= PLOT CONFIGURATION =======================

# Colormap choice (uncomment choice)
c_cmap = plt.cm.bwr(np.linspace(0, 1, 21))

# Plot colors for each model (in level plots)
colors = [
    "black",  # black
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#F0E442",  # yellow
    "#D55E00",  # vermillion
    "#56B4E9",  # sky blue
    "#8B0000",  # dark red
    "#228B22",
]  # forest green

vars_title_map = {
    "H": "hght",
    "U": "uwnd",
    "V": "vwnd",
    "T": "tmpu",
    "Q": "sphu",
    "Z": "geop",
    "P": "slp",
    "PS": "sfcp",
    "Q2M": "sphu",
    "T2M": "tmpu",
    "U10M": "uwnd",
    "V10M": "vwnd",
    "D2M": "dwpt",
    # ← ADD LOWERCASE VERSIONS
    "Q2m": "sphu",
    "T2m": "tmpu",
    "U10m": "uwnd",
    "V10m": "vwnd",
    "D2m": "dwpt",
    "AOD": "aod",
    "LOGAOD": "lnaod",
    "PM25": "pm25",
}

vars_long_map = {
    "H": "Heights",
    "U": "U-Wind",
    "V": "V-Wind",
    "T": "Temperature",
    "Q": "Specific Humidity",
    "Z": "Geopotential",
    "P": "Sea-Level Pressure",
    "PS": "Surface Pressure",
    "Q2M": "2m Specific Humidity",
    "T2M": "2m Temperature",
    "U10M": "10m U-Wind",
    "V10M": "10m V-Wind",
    "D2M": "2m Dew Point",
    # ← ADD LOWERCASE VERSIONS
    "Q2m": "2m Specific Humidity",
    "T2m": "2m Temperature",
    "U10m": "10m U-Wind",
    "V10m": "10m V-Wind",
    "D2m": "2m Dew Point",
    "AOD": "Total Aerosol Extinction AOT [550 nm]",
    "LOGAOD": "log(AOD+0.01)",
    "PM25": "PM2.5 Total Mass",
}

vars_unit_map = {
    "H": "m",
    "U": "m/s",
    "V": "m/s",
    "T": "K",
    "Q": "g/kg",
    "Z": "m²/s²",
    "P": "hPa",
    "PS": "hPa",
    "Q2M": "g/kg",
    "T2M": "K",
    "U10M": "m/s",
    "V10M": "m/s",
    "D2M": "K",
    # ← ADD LOWERCASE VERSIONS
    "Q2m": "g/kg",
    "T2m": "K",
    "U10m": "m/s",
    "V10m": "m/s",
    "D2m": "K",
    "AOD": "",
    "LOGAOD": "",
    "PM25": "µg/m3",
}

# Statistics metadata
stat_label_map = {
    "acorr": "Anomaly Correlation",
    "rms": "Root Mean Square Error",
}
stat_output_map = {
    "acorr": "corcmp",
    "rms": "rmscmp"
}
stat_shortname_map = {
    "acorr": "ACORR",
    "rms": "RMSE",
}

# 3d variables min/max magnitude for zonal RMS plots
vars_range_map = {
    "H": 4,
    "U": 0.4,
    "V": 0.4,
    "T": 0.2,
    "Q": 0.1,
    "Z": 40,  # ← ADD THIS (adjust based on your data)
    "Q2m": 0.1,  # or "Q2m": 0.1
    "T2m": 0.2,  # or "T2m": 0.2
    "U10m": 0.4,  # or "U10m": 0.4
    "V10m": 0.4,  # or "V10m": 0.4
    "D2m": 0.2,  # or "D2m": 0.2
}

# Regions metadata
region_short_map = {
    "GLO": "global",
    "NHE": "n.hem",
    "TRO": "tropics",
    "SHE": "s.hem",
    "NWQ": "nw.quad",
    "NEQ": "ne.quad",
    "SWQ": "sw.quad",
    "SEQ": "se.quad",
    "NAM": "america",
    "EUR": "europe",
    "NPO": "npolar",
    "SPO": "spolar",
    "XPO": "xpolar",
    "CUS": "conus",
    "LND": "glob_l",
    "NHL": "n.hem_l",
    "TRL": "trop_l",
    "SHL": "s.hem_l",
}
region_long_map = {
    "GLO": "Global",
    "NHE": "N.Hem. ExtraTropics (Lats: 20,80)",
    "TRO": "Tropics (Lats: -20,20)",
    "SHE": "S.Hem. ExtraTropics (Lats: -20,-80)",
    "NWQ": "N.W. Quadrant (Lons:-180,0  Lats: 0, 90)",
    "NEQ": "N.E. Quadrant (Lons: 0,180  Lats: 0, 90)",
    "SWQ": "S.W. Quadrant (Lons:-180,0  Lats: 0,-90)",
    "SEQ": "S.E. Quadrant (Lons: 0,180  Lats: 0,-90)",
    "NAM": "North America (Lons:-140,-60  Lats: 20,60)",
    "EUR": "Europe (Lons:-10,30  Lats: 30,60)",
    "NPO": "N.Polar (Lats: 60,90)",
    "SPO": "S.Polar (Lats:-90,-60)",
    "XPO": "X.Polar (Lats: -60,60)",
    "CUS": "Continental United States",
    "LND": "Global Land-only",
    "NHL": "N.Hem. ExtraTropics (Lats: 20,80) Land-only",
    "TRL": "Tropics (Lats: -20,20) Land-only",
    "SHL": "S.Hem. ExtraTropics (Lats: -20,-80) Land-only",
}

# ================== REGIONAL FUNCTIONS ==================


def discover_experiments(data):
    """
    Discover available experiment indices from data structure.

    Returns:
    - sorted list of experiment indices (e.g., [0, 1, 3, 5])
    """
    # Look at first collection's raw data to get experiment keys
    first_coll = list(data["raw"].keys())[0]
    exp_indices = list(data["raw"][first_coll].keys())
    return sorted(exp_indices)


def plot_rectangle(ax, i, x_vals, ci_l, ci_u, color, width):
    """
    Plot confidence interval rectangle; x_vals are leads values,
    ci_l/ci_u are lower/upper conf. interval levels
    """
    rect = Rectangle(
        (x_vals[i] - width / 2, ci_l[i]),
        width,
        abs(ci_l[i]) + ci_u[i],
        fill=False,
        edgecolor=color,
        linestyle="-",
        linewidth=0.8,
    )
    ax.add_patch(rect)


def plot_level(
    nstats,
    reg,
    coll,
    v,
    var,
    lvl_idx,
    lev,
    data_dict,
    nfcsts,
    fcst_interval,
    fcst_length,
    models,
    available_exps,
    plotting_exps,
    colors,
    stat_lbls,
    stat_outnms,
    vars_unit_map,
    vars_long_map,
    region_long_map,
    is_3d,
    x_vals,
    xloc,
    seas_yr,
    season,
    ci_l,
    ci_u,
    ci_syn_l,
    ci_syn_u,
    conf_levels,
    nleads,
    plots_dir,
    dpi,
    title,
    long,
    lead_indices,
):
    """
    Create a level plot with ACC on upper axis and RMSE on lower axis.

    Upper panel (axu): ACC for all experiments with synoptic CIs
    Lower panel (axl): RMSE for all experiments with synoptic CIs
    """

    # Create directory structure: plots_dir/variable/level/
    var_dir = Path(plots_dir) / var
    var_dir.mkdir(parents=True, exist_ok=True)

    # Create figure with upper and lower axes
    fig = plt.figure(figsize=(8, 6))
    axu = fig.add_axes([0.205, 0.417, 0.678, 0.466])
    axl = fig.add_axes([0.205, 0.117, 0.678, 0.300])
    axu.set_ylabel("Anomaly Correlation")
    axl.set_ylabel("Root Mean Square Error")
    textsu, textsl = [], []

    # ========== UPPER PANEL: Plot ACC (stat_idx=0) ==========

    acc_idx = 0
    # Plot acc for all plotting experiments
    for idx, exp_idx in enumerate(plotting_exps):
        model = models[exp_idx]
        color_idx = available_exps.index(exp_idx)

        # Extract ACC data and apply lead filtering
        if is_3d[coll]:
            y_vals_acc = data_dict["avg"][coll][exp_idx][reg][:, acc_idx, v, lvl_idx][lead_indices]
        else:
            y_vals_acc = data_dict["avg"][coll][exp_idx][reg][:, acc_idx, v][lead_indices]

        # Check for non-finite values
        if not np.all(np.isfinite(y_vals_acc)):
            print(f"  WARNING: Non-finite ACC in {var}, {model}, region {reg}")
            print(f"    NaN count: {np.sum(np.isnan(y_vals_acc))}")

        # Format label: control gets second part, others get first part
        parts = model.split("_")
        if exp_idx <= 1:  # Control
            label = parts[1] if len(parts) == 2 else model
        else:  # Experiments
            label = parts[0] + " vs " + parts[1] if len(parts) >= 1 else model

        # Plot ACC line
        axu.plot(
            x_vals,
            y_vals_acc,
            label=label,
            color=colors[color_idx],
            linestyle="-",
            linewidth=1.5,
        )

        # Add final value text with vertical offset
        final = y_vals_acc[-1]
        if np.isfinite(final):
            # For ACC (0-1 range), use fixed offset
            offset_increment = 0.03  # Adjust this value if needed
            y_offset = (idx - len(plotting_exps)/2) * offset_increment

            txt = axu.text(
                xloc,
                final + y_offset,
                f"{final:.4f}",
                color=colors[color_idx],
                fontsize=8,
                ha="left",
                va="center",
            )
            textsu.append(txt)

    # Plot synoptic confidence intervals for ACC (control only, dashed)
    if is_3d[coll]:
        ci_syn_l_vals = ci_syn_l[coll][:, acc_idx, v, lvl_idx][lead_indices]
        ci_syn_u_vals = ci_syn_u[coll][:, acc_idx, v, lvl_idx][lead_indices]
    else:  # 2D
        ci_syn_l_vals = ci_syn_l[coll][:, acc_idx, v][lead_indices]  # No lvl_idx
        ci_syn_u_vals = ci_syn_u[coll][:, acc_idx, v][lead_indices]  # No lvl_idx

    control_color_idx = available_exps.index(acc_idx)
    ymin, ymax = axu.get_ylim()  # Save auto limits before adding CIs

    axu.plot(
        x_vals,
        ci_syn_l_vals,
        color=colors[control_color_idx],
        linestyle="--",
        linewidth=1.0,
    )
    axu.plot(
        x_vals,
        ci_syn_u_vals,
        color=colors[control_color_idx],
        linestyle="--",
        linewidth=1.0,
    )

    # Set ylim with CI max at 1.0 for ACC
    axu.set_ylim(ymin, 1.0)

    # ========== LOWER PANEL: Plot RMSE (stat_idx=1) ==========
    rmse_idx = 1

    # Plot RMSE for all experiments
    for idx, exp_idx in enumerate(plotting_exps):
        model = models[exp_idx]
        color_idx = available_exps.index(exp_idx)

        # Extract RMSE data and apply lead filtering
        if is_3d[coll]:
            y_vals_rmse = data_dict["avg"][coll][exp_idx][reg][:, rmse_idx, v, lvl_idx][lead_indices]
        else:
            y_vals_rmse = data_dict["avg"][coll][exp_idx][reg][:, rmse_idx, v][lead_indices]

        # Check for non-finite values
        if not np.all(np.isfinite(y_vals_rmse)):
            print(f"  WARNING: Non-finite RMSE in {var}, {model}, region {reg}")
            print(f"    NaN count: {np.sum(np.isnan(y_vals_rmse))}")

        # Format label: control gets second part, others get first part
        parts = model.split("_")
        if exp_idx <= 1:  # Control
            label = parts[1] if len(parts) == 2 else model
        else:  # Experiments
            label = parts[0] + " vs " + parts[1] if len(parts) >= 1 else model

        # Plot RMSE line
        axl.plot(
            x_vals,
            y_vals_rmse,
            label=label,
            color=colors[color_idx],
            linestyle="-",
            linewidth=1.5,
        )

        # Add final value text with vertical offset
        final = y_vals_rmse[-1]
        if np.isfinite(final):
            # For RMSE (variable range), calculate offset based on y-axis range
            ymin_rmse, ymax_rmse = axl.get_ylim()
            y_range = ymax_rmse - ymin_rmse
            offset_increment = y_range * 0.03  # 2% of y-axis range
            y_offset = (idx - len(plotting_exps)/2) * offset_increment

            txt = axl.text(
                xloc,
                final + y_offset,
                f"{final:.4f}",
                color=colors[color_idx],
                fontsize=8,
                ha="left",
                va="center",
            )
            textsl.append(txt)

    # Plot synoptic confidence intervals for RMSE (control only, dashed)
    if is_3d[coll]:
        ci_syn_l_vals_rmse = ci_syn_l[coll][:, rmse_idx, v, lvl_idx][lead_indices]
        ci_syn_u_vals_rmse = ci_syn_u[coll][:, rmse_idx, v, lvl_idx][lead_indices]
    else:  # 2D
        ci_syn_l_vals_rmse = ci_syn_l[coll][:, rmse_idx, v][lead_indices]
        ci_syn_u_vals_rmse = ci_syn_u[coll][:, rmse_idx, v][lead_indices]

    control_color_idx = available_exps.index(0)
    ymin_rmse, ymax_rmse = axl.get_ylim()  # Save auto limits before adding CIs

    axl.plot(
        x_vals,
        ci_syn_l_vals_rmse,
        color=colors[control_color_idx],
        linestyle="--",
        linewidth=1.0,
    )
    axl.plot(
        x_vals,
        ci_syn_u_vals_rmse,
        color=colors[control_color_idx],
        linestyle="--",
        linewidth=1.0,
    )

    # Restore auto limits for RMSE (don't cap at specific value)
    axl.set_ylim(ymin_rmse, ymax_rmse)

    # ========== Format upper panel ==========
    # Use actual filtered lead times for x-axis
    x_min = 0
    x_max = max(x_vals) if x_vals else fcst_length

    # Set ticks at the actual filtered lead times
    axu.set_xticks(x_vals)
    axu.set_xlim(x_min, x_max)
    axu.tick_params(axis="x", labelbottom=False)
    axu.tick_params(axis="y", labelsize=8, pad=2, length=2)
    axu.spines["bottom"].set_visible(False)
    formatter = mticker.FormatStrFormatter("%.1f")
    axu.xaxis.set_major_formatter(formatter)
    axu.grid(
        True,
        which="both",
        axis="both",
        color="lightgrey",
        linestyle="--",
        linewidth=0.5,
    )

    # ========== Format lower panel ==========
    # Use actual filtered lead times for x-axis
    axl.set_xticks(x_vals)
    axl.set_xlim(x_min, x_max)
    axl.set_xlabel("Forecast Day", fontsize=10, labelpad=3)
    axl.tick_params(axis="both", labelsize=8, pad=2, length=2)
    axl.grid(
        True,
        which="both",
        axis="both",
        color="lightgrey",
        linestyle="--",
        linewidth=0.5,
    )

    # ========== Add title and region/variable/season texts ==========
    u = vars_unit_map.get(var.upper())
    region = region_long_map.get(reg)
    if coll[:2] == "de" and is_3d[coll]:
        plot_title = (
            f"Forecast Statistics \n{lev}-mb "
            f"{long[coll][v]} ({u}) {region}"
        )
    else:  # 2D variables (sl2d, ae2d, de2d)
        plot_title = f"Forecast Statistics\n{long[coll][v]} ({u}) {region}"
    axu.set_title(plot_title, fontsize=12)

    handles, leg_labels = axu.get_legend_handles_labels()

    # Add custom entry for synoptic CI (95%)
    control_color_idx = available_exps.index(0)
    base_model = models[0].split("_")[1]
    synoptic_ci_line = Line2D(
        [0], [0],
        color=colors[control_color_idx],
        linestyle="--",
        linewidth=1.0,
        label=f"95% Synoptic CI ({base_model})"
    )
    handles.append(synoptic_ci_line)
    leg_labels.append(f"95% Synoptic CI ({base_model})")

    legend = axl.legend(
        handles,
        leg_labels,
        fontsize=8,
        labelspacing=0.4,
        handlelength=2,
        handletextpad=0.5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=4,
        frameon=False,
    )

    # ========== Enforce non-scientific notation ==========
    for ax in fig.get_axes():
        ax.ticklabel_format(style="plain", axis="y")

    # ========== Save plot ==========
    # Use hardcoded stat names since we always plot ACC and RMSE
    plotnm = f"stats_{title[coll][v]}_ACC-RMSE_{reg}_{lev}_{season}.png"
    plt.savefig(var_dir / plotnm, dpi=dpi)
    plt.close(fig)


def plot_all_collection_levels(
    coll,
    reg,
    fvars,
    vars_range_map,
    nstats,
    levs,
    available_exps,
    comparison_exps,  # Changed: added these parameters
    plotting_exps,
    data,
    models,
    colors,
    stat_lbls,
    stat_nms,
    stat_outnms,
    vars_unit_map,
    vars_long_map,
    region_long_map,
    is_3d,
    fcst_interval,
    fcst_length,
    nfcsts,
    nleads,
    nlevs,
    x_vals,
    xloc,
    seas_yr,
    season,
    ci_l,
    ci_u,
    ci_syn_l,
    ci_syn_u,
    conf_levels,
    cmap,
    plots_dir,
    dpi,
    title,
    long,
    levels_to_plot,
    stats_to_plot,
    lead_indices,
):
    """Create all plots (zonal and level) for a 3d collection."""

    if stats_to_plot is None:
        stats_to_plot = list(range(nstats))

    # Create plots for each variable/stat combination
    for v, var in enumerate(fvars[coll]):
        print(f"\nMaking level plots for {var} (3D) {reg}:")

        # Set scale min/max for RMS plots
        var_range = vars_range_map.get(var.upper())
        vmin, vmax = zip(
            *[
                (-0.02, 0.02) if i == 0 else (-var_range, var_range)
                for i in range(nstats)
            ]
        )

        # Create level plots
        if levels_to_plot is None:
            levels_to_iterate = enumerate(levs)
        else:
            levels_to_iterate = [
                (lead_idx, lev) for lead_idx, lev in enumerate(levs) if lev in levels_to_plot
            ]

        # Create all stats in one image (should only be 2)
        for lead_idx, lev in levels_to_iterate:
            plot_level(
                # n,
                stats_to_plot,
                reg,
                coll,
                v,
                var,
                lead_idx,
                lev,
                data,
                nfcsts,
                fcst_interval,
                fcst_length,
                models,
                available_exps,
                plotting_exps,
                colors,
                stat_lbls,
                stat_outnms,
                vars_unit_map,
                vars_long_map,
                region_long_map,
                is_3d,
                x_vals,
                xloc,
                seas_yr,
                season,
                ci_l,
                ci_u,
                ci_syn_l,
                ci_syn_u,
                conf_levels,
                nleads,
                plots_dir,
                dpi,
                title,
                long,
                lead_indices,
            )
            plt.close("all")


# ================== CONFIDENCE INTERVAL FUNCTIONS ==================


def calc_reg_diff_ci(
    reg,
    data_dict,
    comparison_exps,
    collections,
    fvars,
    is_3d,
    nleads,
    nstats,
    nvars,
    nlevs,
    nfcsts,
    conf_levels=[0.68, 0.90, 0.95, 0.99, 0.9999],
):
    ci_l, ci_u = {}, {}
    for conf in conf_levels:
        ci_l[conf] = {}
        ci_u[conf] = {}
        for coll in collections:
            ci_l[conf][coll] = {}
            ci_u[conf][coll] = {}

    # Calculate difference CIs for each comparison experiment
    for exp_idx in comparison_exps:
        for conf in conf_levels:
            for coll in collections:
                if len(fvars[coll]) > 0:
                    cil_coll, ciu_coll = calc_ci(
                        data_dict["raw"][coll][0][reg],
                        data_dict["raw"][coll][exp_idx][reg],
                        coll,
                        conf,
                        is_3d,
                        nleads,
                        nstats,
                        nvars,
                        nlevs,
                        nfcsts,
                    )
                    ci_l[conf][coll][exp_idx], ci_u[conf][coll][exp_idx] = (
                        cil_coll,
                        ciu_coll,
                    )

    return ci_l, ci_u


def calc_ci(
    ctl_data,
    model_data,
    coll,
    sig,
    is_3d,
    nleads,
    nstats,
    nvars,
    nlevs,
    nfcsts,
):
    """
    Calculate asymmetric confidence intervals for model-control differences.

    Uses Fisher z-transformation for correlation coefficients and power
    transformation for RMS values to approximate normality before calculating
    confidence intervals.

    Parameters:
    - ctl_data: Control experiment data array
                Shape: (nfcsts, nleads, nstats, nvars[, nlevs])
    - model_data: Comparison experiment data array
                  Shape: (nfcsts, nleads, nstats, nvars[, nlevs])
    - coll: Collection name (e.g., 'de3d', 'de2d')
    - sig: Significance level (e.g., 0.95 for 95% confidence)
    - is_3d: Dictionary indicating if collection is 3D
    - nleads: Number of lead times
    - nstats: Number of statistics
    - nvars: Dictionary of number of variables per collection
    - nlevs: Number of vertical levels (for 3D collections)
    - nfcsts: Number of forecasts (sample size)

    Returns:
    - ci_lower: Lower confidence interval bounds (centered on zero)
                Shape: (nleads, nstats, nvars[, nlevs])
    - ci_upper: Upper confidence interval bounds (centered on zero)
                Shape: (nleads, nstats, nvars[, nlevs])
    """

    # Initialize confidence interval arrays
    if is_3d[coll]:
        ci_lower = np.zeros((nleads, nstats, nvars[coll], nlevs))
        ci_upper = np.zeros((nleads, nstats, nvars[coll], nlevs))
    else:
        ci_lower = np.zeros((nleads, nstats, nvars[coll]))
        ci_upper = np.zeros((nleads, nstats, nvars[coll]))

    # Create CIs, looping through leads and stats
    for lead_idx in range(nleads):
        for s in range(nstats):
            if s == 0:  # ACORR (Anomaly Correlation)
                # Apply Fisher z-transformation to control and model
                # Clip to avoid log(0) or log(negative)
                ctl_clipped = np.clip(
                    ctl_data[:, lead_idx, s, :], -1 + 5.0e-6, 1 - 5.0e-6
                )
                model_clipped = np.clip(
                    model_data[:, lead_idx, s, :], -1 + 5.0e-6, 1 - 5.0e-6
                )

                # Fisher z-transformation: z = 0.5 * ln((1+r)/(1-r))
                ctl_tr = 0.5 * np.log((1 + ctl_clipped) / (1 - ctl_clipped))
                model_tr = 0.5 * np.log(
                    (1 + model_clipped) / (1 - model_clipped)
                )

                # Calculate differences and control mean in transformed space
                diffs_tr = model_tr - ctl_tr
                ctl_tr_mean = np.mean(ctl_tr, axis=0)

                # Calculate variance and standard error of transformed diffs
                vard_tr = stats.tvar(diffs_tr, axis=0)
                se_tr = (vard_tr / nfcsts) ** (1 / 2)

                # Calculate t-critical value (two-sided)
                t_crit = stats.t.ppf((1 + sig) / 2, nfcsts - 1)
                dx = se_tr * t_crit

                # Back-transform CIs around control, then subtract control
                # Inverse Fisher transform: r = (e^(2z) - 1) / (e^(2z) + 1)
                ci_lower[lead_idx, s] = (
                    (np.exp(2 * (ctl_tr_mean - dx)) - 1)
                    / (np.exp(2 * (ctl_tr_mean - dx)) + 1)
                ) - (
                    (np.exp(2 * ctl_tr_mean) - 1)
                    / (np.exp(2 * ctl_tr_mean) + 1)
                )
                ci_upper[lead_idx, s] = (
                    (np.exp(2 * (ctl_tr_mean + dx)) - 1)
                    / (np.exp(2 * (ctl_tr_mean + dx)) + 1)
                ) - (
                    (np.exp(2 * ctl_tr_mean) - 1)
                    / (np.exp(2 * ctl_tr_mean) + 1)
                )

            else:  # RMS (Root Mean Square Error)
                # Apply power transformation (square) to control and model
                # This helps normalize the distribution of RMS values
                ctl_tr = (ctl_data[:, lead_idx, s, :]) ** 2
                model_tr = (model_data[:, lead_idx, s, :]) ** 2

                # Calculate differences and control mean in transformed space
                diffs_tr = model_tr - ctl_tr
                ctl_tr_mean = np.mean(ctl_tr, axis=0)

                # Calculate variance and standard error of transformed diffs
                vard_tr = stats.tvar(diffs_tr, axis=0)

                se_tr = (vard_tr / nfcsts) ** (1 / 2)

                # Calculate t-critical value (two-sided)
                t_crit = stats.t.ppf((1 + sig) / 2, nfcsts - 1)
                dx = se_tr * t_crit

                # Back-transform CIs around control, then subtract control
                # Square root to get back to RMS scale, ensure non-negative
                ci_lower[lead_idx, s] = (np.maximum(0, ctl_tr_mean - dx)) ** (
                    1 / 2
                ) - (np.maximum(0, ctl_tr_mean)) ** (1 / 2)
                ci_upper[lead_idx, s] = (np.maximum(0, ctl_tr_mean + dx)) ** (
                    1 / 2
                ) - (np.maximum(0, ctl_tr_mean)) ** (1 / 2)

    return ci_lower, ci_upper


def calc_reg_syn_ci(
    reg,
    data_dict,
    comparison_exps,
    collections,
    is_3d,
    nleads,
    nstats,
    nvars,
    nlevs,
    nfcsts,
    conf_level=0.95,
):
    ci_syn_l, ci_syn_u = {}, {}
    for coll in collections:
        comparison_data = [
            data_dict["raw"][coll][comp_exp][reg]
            for comp_exp in comparison_exps
        ]

        ci_syn_l[coll], ci_syn_u[coll] = calc_syn_ci(
            data_dict["raw"][coll][0][reg],
            comparison_data,
            coll,
            conf_level,
            is_3d,
            nleads,
            nstats,
            nvars,
            nlevs,
            nfcsts,
        )

    return ci_syn_l, ci_syn_u


def calc_syn_ci(
    ctl_data,
    model_data_all,
    coll,
    sig,
    is_3d,
    nleads,
    nstats,
    nvars,
    nlevs,
    nfcsts,
):
    """
    Calculate synoptic confidence intervals for full experiment list.

    These are "synoptic" CIs that represent the expected variability of the
    control experiment mean across the ensemble of all experiments. They're
    plotted as dashed lines around the control in the upper panel.

    Parameters:
    - ctl_data: Control experiment data array
                Shape: (nfcsts, nleads, nstats, nvars[, nlevs])
    - model_data_all: List of non-control model data arrays
                      Each array shape: (nfcsts, nleads, nstats, nvars[, nlevs])
    - coll: Collection name (e.g., 'de3d', 'de2d')
    - sig: Significance level (e.g., 0.95 for 95% confidence)
    - is_3d: Dictionary indicating if collection is 3D
    - nleads: Number of lead times
    - nstats: Number of statistics
    - nvars: Dictionary of number of variables per collection
    - nlevs: Number of vertical levels (for 3D collections)
    - nfcsts: Number of forecasts (sample size)

    Returns:
    - ci_lower: Lower confidence interval bounds (centered on control mean)
                Shape: (nleads, nstats, nvars[, nlevs])
    - ci_upper: Upper confidence interval bounds (centered on control mean)
                Shape: (nleads, nstats, nvars[, nlevs])
    """

    # Get number of experiments (control + comparison models)
    nexps = len(model_data_all) + 1

    # Initialize confidence interval arrays
    if is_3d[coll]:
        ci_lower = np.zeros((nleads, nstats, nvars[coll], nlevs))
        ci_upper = np.zeros((nleads, nstats, nvars[coll], nlevs))
    else:
        ci_lower = np.zeros((nleads, nstats, nvars[coll]))
        ci_upper = np.zeros((nleads, nstats, nvars[coll]))

    if np.any(~np.isfinite(ctl_data)):
        print(
            f"  WARNING (synoptic): Non-finite values in ctl_data for {coll}"
        )

    # Create CIs, looping through leads and stats
    for lead_idx in range(nleads):
        for s in range(nstats):
            if s == 0:  # ACORR (Anomaly Correlation)
                # Apply Fisher z-transformation to control
                ctl_clipped = np.clip(
                    ctl_data[:, lead_idx, s, :], -1 + 5.0e-6, 1 - 5.0e-6
                )
                ctl_tr = 0.5 * np.log((1 + ctl_clipped) / (1 - ctl_clipped))

                # Calculate control mean in transformed space
                ctl_tr_mean = np.mean(ctl_tr, axis=0)

                # Apply transform to non-control models and calculate variance
                if is_3d[coll]:  # 3d
                    varm_tr = np.zeros((nexps - 1, nvars[coll], nlevs))
                else:  # 2d
                    varm_tr = np.zeros((nexps - 1, nvars[coll]))

                for m in range(nexps - 1):
                    model_data = np.asarray(model_data_all[m][:, lead_idx, s, :])
                    # Clip and transform model data
                    model_clipped = np.clip(
                        model_data, -1 + 5.0e-6, 1 - 5.0e-6
                    )
                    model_tr = 0.5 * np.log(
                        (1 + model_clipped) / (1 - model_clipped)
                    )
                    # Calculate variance for this model
                    varm_tr[m] = stats.tvar(model_tr)

                # Calculate control variance and mean of other model variances
                varctl_tr = stats.tvar(ctl_tr)
                var_tr = np.mean(varm_tr, axis=0)

                # Calculate standard error from average of control/other variances
                se_tr = ((varctl_tr + var_tr) / 2 / nfcsts) ** (1 / 2)

                # Calculate t-critical value (two-sided)
                t_crit = stats.t.ppf((1 + sig) / 2, nfcsts - 1)
                dx = se_tr * t_crit

                # Back-transform CIs around control, then subtract control
                ci_lower[lead_idx, s] = (
                    (np.exp(2 * (ctl_tr_mean - dx)) - 1)
                    / (np.exp(2 * (ctl_tr_mean - dx)) + 1)
                ) - (
                    (np.exp(2 * ctl_tr_mean) - 1)
                    / (np.exp(2 * ctl_tr_mean) + 1)
                )
                ci_upper[lead_idx, s] = (
                    (np.exp(2 * (ctl_tr_mean + dx)) - 1)
                    / (np.exp(2 * (ctl_tr_mean + dx)) + 1)
                ) - (
                    (np.exp(2 * ctl_tr_mean) - 1)
                    / (np.exp(2 * ctl_tr_mean) + 1)
                )

            else:  # RMS (Root Mean Square Error)
                # Apply power transform to control
                ctl_tr = (ctl_data[:, lead_idx, s, :]) ** 2

                # Calculate control mean in transformed space
                ctl_tr_mean = np.mean(ctl_tr, axis=0)

                # Apply transform to non-control models and calculate variance
                if is_3d[coll]:  # 3d
                    varm_tr = np.zeros((nexps - 1, nvars[coll], nlevs))
                else:  # 2d
                    varm_tr = np.zeros((nexps - 1, nvars[coll]))

                for m in range(nexps - 1):
                    model_data = np.asarray(model_data_all[m][:, lead_idx, s, :])
                    model_tr = (model_data) ** 2
                    varm_tr[m] = stats.tvar(model_tr)

                # Calculate control variance and mean of other model variances
                varctl_tr = stats.tvar(ctl_tr)
                var_tr = (varctl_tr + np.mean(varm_tr, axis=0)) / 2

                # Calculate standard error
                se_tr = (var_tr / nfcsts) ** (1 / 2)

                # Calculate t-critical value (two-sided)
                t_crit = stats.t.ppf((1 + sig) / 2, nfcsts - 1)
                dx = se_tr * t_crit

                # Back-transform CIs around control, then subtract control
                ci_lower[lead_idx, s] = (np.maximum(0, ctl_tr_mean - dx)) ** (
                    1 / 2
                ) - (np.maximum(0, ctl_tr_mean)) ** (1 / 2)
                ci_upper[lead_idx, s] = (np.maximum(0, ctl_tr_mean + dx)) ** (
                    1 / 2
                ) - (np.maximum(0, ctl_tr_mean)) ** (1 / 2)

            # Add CIs to control mean in original space
            ctl_mean = np.mean(ctl_data[:, lead_idx, s, :], axis=0)
            ci_lower[lead_idx, s] = ci_lower[lead_idx, s] + ctl_mean
            ci_upper[lead_idx, s] = ci_upper[lead_idx, s] + ctl_mean

    return ci_lower, ci_upper


# ================== MAIN PLOTTING DRIVERS ==================
def get_season(data_dict):
    date_names = data_dict["date_nms"]

    # Parse dates (assuming format YYYYMMDD_HH or YYYYMMDD)
    min_date = datetime.strptime(date_names[0][:8], "%Y%m%d")
    max_date = datetime.strptime(date_names[-1][:8], "%Y%m%d")

    # Format season string
    if min_date.month == max_date.month and min_date.year == max_date.year:
        return max_date.strftime("%b_%Y")
    else:
        return f"{min_date.strftime('%b_%Y')}-{max_date.strftime('%b_%Y')}"


def create_regional_plots(
    data_dict,
    vars_to_plot: List[str] = None,
    levels_to_plot: List[int] = None,
    leads_to_plot: List[int] = None,
    regions_to_plot: List[str] = ["NHE", "SHE", "TRO"],
    exps_to_comp: List[List[int]] = [[1, 0]],
    limit_percentiles: List[int] = [75, 75, 75],
    stats_to_plot: List[int] = [0, 1],
    plots_dir="outputs",
) -> None:
    """
    Main driver for plot mode: loops over data_dict dict and creates plots for
    2D and 3D variables.

    Parameters:
    - data_dict: Dictionary returned from load_stats_data() containing all
            statistics, metadata, and configuration
    - regions_to_plot: plot only the specified regions in level plots
    - levels_to_plot: plot only the specified levels in level plots
    - vars_to_plot: plot only the specified variables in level plots
    - plots_dir: Base plots directory (should already exist, created in notebook)
    """

    start_time = time.perf_counter()

    # Discover available experiments from data_dict structure
    available_exps = discover_experiments(data_dict)

    # Extract metadata from data_dict dict
    date_nms = data_dict["date_nms"]
    nfcsts = len(date_nms)
    leads = data_dict["leads"]
    nleads_orig = len(leads)
    fcst_length = data_dict["fcst_length"]  # days
    fcst_interval = data_dict["fcst_interval"]  # hours

    # Extract plot configuration
    fvars = data_dict["fvars"]
    collections = data_dict["collections"]
    is_3d = data_dict["is_3d"]
    nvars = {coll: len(fvars[coll]) for coll in collections}
    levs = data_dict["levels"]
    nlevs = len(levs)
    regions = data_dict["regions"]
    # plot_RMS_decomp = data_dict.get("plot_RMS_decomp", True)
    nstats = len(data_dict["plot_stats"])
    dpi = data_dict["plot_dpi"]

    # Create model names dict (keyed by experiment index)
    models = {}
    for exp_idx in available_exps:
        models[exp_idx] = (
            f'{data_dict["fcst_names"][exp_idx]}_{data_dict["ana_names"][exp_idx]}'
        )

    # We only want to plot GEOS-FP and FM
    plotting_exps = [0, 3]

    # Get comparison experiments (all except control)
    comparison_exps = [exp for exp in plotting_exps if exp != 0]

    # Apply filtering to regions, variables, leads, stats so plots are accurate
    if regions_to_plot is not None:
        regions = [reg for reg in regions if reg in regions_to_plot]

    if vars_to_plot is not None:  # Filter vars, collections, nvars
        fvars = {
            coll: [v for v in fvars[coll] if v in vars_to_plot]
            for coll in collections
        }
        # Filter out collections with no variables left
        collections = [coll for coll in collections if len(fvars[coll]) > 0]

        # Update nvars for remaining collections
        nvars = {coll: len(fvars[coll]) for coll in collections}

    if leads_to_plot is not None:
        lead_indices = [
            idx
            for idx, lead in enumerate(data_dict["leads"])
            if lead in leads_to_plot
        ]
        leads = [lead for lead in data_dict["leads"] if lead in leads_to_plot]
        nleads_filtered = len(leads)
    else:
        # No filtering - use all lead indices
        lead_indices = list(range(len(data_dict["leads"])))
        nleads_filtered = nleads_orig

    if stats_to_plot is not None:
        nstats = len(stats_to_plot)
    else:
        stats_to_plot = list(range(nstats))  # Use all stats

    # Setup plot labels
    x_vals = [x / 24 for x in leads]
    xloc = x_vals[-1] + (x_vals[-1] * 0.05)
    season = data_dict["season"]
    seas_yr = get_season(data_dict)

    # Collect variable metadata
    title, long = {}, {}
    for coll in collections:
        title[coll] = [vars_title_map.get(var.upper()) for var in fvars[coll]]
        long[coll] = [vars_long_map.get(var.upper()) for var in fvars[coll]]

    # Build lists from actual data_dict stats
    stat_lbls = [
        stat_label_map.get(s.lower(), s.upper())
        for s in data_dict["plot_stats"]
    ]
    stat_outnms = [
        stat_output_map.get(s.lower(), s.lower())
        for s in data_dict["plot_stats"]
    ]
    stat_nms = [
        stat_shortname_map.get(s.lower(), s.upper())
        for s in data_dict["plot_stats"]
    ]

    # Define colormap
    cmap = mcolors.ListedColormap(c_cmap)

    # Verify plots directory exists
    plots_dir = Path(plots_dir)
    regional_subdir = plots_dir / "regional"
    regional_subdir.mkdir(exist_ok=True, parents=True)

    # ========== MAIN PLOTTING LOOP: Loop over regions ==========
    all_region_times = []
    for reg in regions:
        region_start_time = time.perf_counter()

        region_dir = regional_subdir / reg
        region_dir.mkdir(exist_ok=True)

        # Initialize confidence interval dictionaries
        ci_l, ci_u = calc_reg_diff_ci(
            reg,
            data_dict,
            comparison_exps,
            collections,
            fvars,
            is_3d,
            nleads_orig,
            nstats,
            nvars,
            nlevs,
            nfcsts,
            conf_levels=[0.68, 0.90, 0.95, 0.99, 0.9999],
        )

        # Calculate synoptic CIs for full experiment list
        # Synoptic CI is stored per collection, per experiment
        ci_syn_l, ci_syn_u = calc_reg_syn_ci(
            reg,
            data_dict,
            comparison_exps,
            collections,
            is_3d,
            nleads_orig,
            nstats,
            nvars,
            nlevs,
            nfcsts,
            conf_level=0.95,
        )
        # Now that we have our diff and synoptic CI values, plot levels
        for coll in collections:
            if is_3d[coll]:
                plot_all_collection_levels(
                    coll,
                    reg,
                    fvars,
                    vars_range_map,
                    nstats,
                    levs,
                    available_exps,  
                    comparison_exps, 
                    plotting_exps,
                    data_dict,
                    models,
                    colors,
                    stat_lbls,
                    stat_nms,
                    stat_outnms,
                    vars_unit_map,
                    vars_long_map,
                    region_long_map,
                    is_3d,
                    fcst_interval,
                    fcst_length,
                    nfcsts,
                    nleads_filtered,
                    nlevs,
                    x_vals,
                    xloc,
                    seas_yr,
                    season,
                    ci_l,
                    ci_u,
                    ci_syn_l,
                    ci_syn_u,
                    [0.68, 0.90, 0.95, 0.99, 0.9999],  # confidence levels
                    cmap,
                    region_dir,
                    dpi,
                    title,
                    long,
                    levels_to_plot,
                    stats_to_plot,
                    lead_indices,
                )
            else:
                # Plot 2D variables (no level iteration needed)
                for v, var in enumerate(fvars[coll]):
                    if vars_to_plot and var not in vars_to_plot:
                        continue

                    print(f"\nMaking plot for {var} (2D) {reg}")

                    # Call plot_level once (no level loop for 2D)
                    plot_level(
                        stats_to_plot,     # nstats parameter (list of idx)
                        reg,
                        coll,
                        v,
                        var,
                        0,                 # lvl_idx (dummy for 2D)
                        0,                 # lev (dummy for 2D)
                        data_dict,
                        nfcsts,
                        fcst_interval,
                        fcst_length,
                        models,
                        available_exps,
                        plotting_exps,
                        colors,
                        stat_lbls,
                        stat_outnms,
                        vars_unit_map,
                        vars_long_map,
                        region_long_map,
                        is_3d,
                        x_vals,
                        xloc,
                        seas_yr,
                        season,
                        ci_l,
                        ci_u,
                        ci_syn_l,
                        ci_syn_u,
                        [0.68, 0.90, 0.95, 0.99, 0.9999],
                        nleads_filtered,
                        region_dir,
                        dpi,
                        title,
                        long,
                        lead_indices,
                    )
                    plt.close("all")

    print(f'\n{"="*60}')
    print(f"PLOTTING COMPLETE!")
    print(f'{"="*60}')
    print(f"Plots saved to: {regional_subdir}")
    print(f"\nAll finished! :)\n")
