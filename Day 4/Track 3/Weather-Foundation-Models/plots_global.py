# =================== IMPORTS, FILTER WARNINGS =====================
# Global plots
# import glob
import io
import math
import os
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
from scipy import stats

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize

warnings.filterwarnings(
    "ignore", message=".*Mean of empty slice.*", category=RuntimeWarning
)
warnings.filterwarnings(
    "ignore", message=".*Precision loss occurred.*", category=RuntimeWarning
)

# ======================= PLOT CONFIGURATION =======================

# Colormap choice (uncomment choice)
# c_cmap = plt.cm.bwr(np.linspace(0, 1, 21))
# c_cmap = [  # Custom spectral-like colormap
#     (0.40, 0.00, 0.60),  # 0 vivid deep purple
#     (0.30, 0.05, 0.78),  # 1 purple
#     (0.17, 0.15, 0.95),  # 2 bright purple-blue
#     (0.05, 0.40, 1.00),  # 3 bright blue
#     (0.05, 0.65, 1.00),  # 4 cyan-blue
#     (0.10, 0.80, 1.00),  # 5 cyan
#     (0.30, 0.90, 1.00),  # 6 light cyan
#     (0.60, 0.95, 1.00),  # 7 very light cyan
#     (0.85, 0.98, 1.00),  # 8 near white cyan
#     (0.95, 0.99, 1.00),  # 9 almost white cyan
#     (1.00, 1.00, 1.00),  # 10 white (center)
#     (1.00, 1.00, 0.85),  # 11 pale yellow
#     (1.00, 0.95, 0.60),  # 12 light yellow
#     (1.00, 0.85, 0.30),  # 13 golden yellow
#     (1.00, 0.70, 0.10),  # 14 bright orange
#     (1.00, 0.55, 0.00),  # 15 orange
#     (1.00, 0.40, 0.00),  # 16 red-orange
#     (0.95, 0.25, 0.00),  # 17 strong red
#     (0.80, 0.15, 0.00),  # 18 deep red
#     (0.65, 0.05, 0.00),  # 19 very deep red
#     (0.45, 0.00, 0.00),  # 20 darkest red (bold maroon)
# ]
c_cmap = plt.cm.RdYlBu_r(np.linspace(0, 1, 21))

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

# Variables metadata
vars_title_map = {
    "H": "hght",
    "U": "uwnd",
    "V": "vwnd",
    "T": "tmpu",
    "Q": "sphu",
    "P": "slp",
    "PS": "sfcp",
    "Q2M": "sphu",
    "T2M": "tmpu",
    "U10M": "uwnd",
    "V10M": "vwnd",
    "D2M": "dwpt",
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
    "P": "Sea-Level Pressure",
    "PS": "Surface Pressure",
    "Q2M": "2m Specific Humidity",
    "T2M": "2m Temperature",
    "U10M": "10m U-Wind",
    "V10M": "10m V-Wind",
    "D2M": "2m Dew Point",
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
    "P": "hPa",
    "PS": "hPa",
    "Q2M": "g/kg",
    "T2M": "K",
    "U10M": "m/s",
    "V10M": "m/s",
    "D2M": "K",
    "AOD": "",
    "LOGAOD": "",
    "PM25": "µg/m3",
}

# 3d variables min/max magnitude for zonal RMS plots
vars_range_map = {"H": 4, "U": 0.4, "V": 0.4, "T": 0.2, "Q": 0.1}

# Collection level names (for plotting labels)
coll_lev_nms = {"de2d": 1000, "sl2d": 2000, "ae2d": 1000}

sig_levs = [0.90]
styles = ["solid"]



'''
[72, 71, 68, 63, 56, 53, 51, 48, 45, 44, 43, 41, 39, 34]

These levels, based on MERRA-2 data, are specifically at:
985, 970, 925, 850, 700, 600, 525, 412, 288, 245, 208, 150, 109, and 48 hPa
'''

PRITHVI_PRESSURE_LVLS = {
    72: 985,
    71: 970,
    68: 925,
    63: 850,
    56: 700,
    53: 600,
    51: 525,
    48: 412,
    45: 288,
    44: 245,
    43: 208,
    41: 150,
    39: 109,
    34: 48
}

# ================== GLOBAL FUNCTIONS ====================


def check_coastline_data():
    """Check if cartopy coastline data is available by trying to use it"""
    try:
        # Create a minimal test figure
        fig = plt.figure(figsize=(1, 1))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE)
        # Force the download/data access by saving to a buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", dpi=10)  # Very low DPI for speed
        buffer.close()
        plt.close(fig)
        return True
    except Exception as e:
        print(f"ERROR: CARTOPY MAP DATA NOT FOUND\nTechnical error: {e}")
        command = (
            "python -c 'import matplotlib.pyplot as plt; "
            "import cartopy.crs as ccrs; "
            "fig = plt.figure(); "
            "ax = plt.axes(projection=ccrs.PlateCarree()); "
            "ax.coastlines(); "
            'plt.savefig("test.png"); '
            "import os; "
            'os.remove("test.png"); '
            'print("Map data downloaded successfully")\''
        )

        print(
            f"""
    This script requires coastline data that must be downloaded first.
    To fix this, please run the following command ON A LOGIN NODE:

        {command}

    This downloads ~2MB of coastline data
    to ~/.local/share/cartopy/
    You only need to do this once, then you can
    run this script.

    Script execution halted."""
        )
        sys.exit(1)


# ================== PLACEHOLDER HELPER FUNCTIONS ====================


def round_down_max(arr, percentile=98):
    """
    Return the max absolute value rounded down to the nearest 10, with a
    minimum threshold of 10 (smaller values use specific "nice" values).

    Parameters:
    -----------
    arr : ndarray
        Array of data values

    Returns:
    --------
    float or int
        Rounded maximum value appropriate for colorbar scales
    """

    with np.errstate(invalid="ignore"):
        # max_abs = np.nanmax(np.abs(arr))
        max_abs = np.nanpercentile(np.abs(arr), percentile)

    # Define thresholds and their corresponding return values
    # thresholds = [
    #     (0.1, 0.1),
    #     (0.5, 0.5),
    #     (1, 1),
    #     (2.5, 2.5),
    #     (5, 5),
    #     (10, 10),
    # ]

    # Check if max_abs falls within predefined thresholds
    # for threshold, value in thresholds:
    #     if 0 < max_abs <= threshold:
    #         return value

    if max_abs < 10 and max_abs > 0:
        # return math.ceil(max_abs)
        return 1.0
    elif max_abs > 10:  # For values > 10, round down to nearest 10
        return int(max_abs // 10 * 10)
    else:
        return 0


def nice_levels(data_min, data_max, n_levels, n_ticks, use_step=False):
    """
    Return contour levels with nice tick label values based on data range.
    """

    # Calculate raw step size and order of magnitude
    data_range = data_max - data_min
    raw_step = data_range / (n_levels - 1)
    mag = 10 ** np.floor(np.log10(raw_step))
    norm_step = raw_step / mag

    # Define nice step thresholds and their corresponding values
    nice_steps = [(1, 1), (2, 2), (2.5, 2.5), (5, 5), (np.inf, 10)]

    # Select appropriate nice step
    nice_step = next(
        value for threshold, value in nice_steps if norm_step <= threshold
    )
    step = nice_step * mag

    # Compute bounds and generate levels based on step
    lower = np.floor(data_min / step) * step
    upper = np.ceil(data_max / step) * step

    # Calculate actual number of levels based on step size
    n_levels_actual = int(np.round((upper - lower) / step)) + 1
    levels_to_use = n_levels_actual if use_step else n_levels
    levels = np.linspace(lower, upper, levels_to_use)

    return levels


def setup_plot_metadata(
    coll,
    v,
    lev,
    lead,
    title_map,
    long_map,
    is_3d_map,
    lev_levs_map,
    season,
    fvars,
):
    """
    Setup plot metadata including naming components and figure with gridspec.

    Creates the filename prefix, plot title ending text, figure object, and
    gridspec layout.

    Parameters:
    -----------
    coll : str
        Collection name (e.g., 'de3d', 'de2d', 'sl2d', 'ae2d')
    v : int
        Variable index within collection
    lev : int or float
        Level value (pressure in mb or nominal value)
    zonal : int
        Zonal plot type: 0 for level plots, or top level (100, 10, 1) for zonal
    lead : int
        Lead time in hours
    title_map : dict
        Maps collection to list of variable title strings
    long_map : dict
        Maps collection to list of variable long names
    is_3d_map : dict
        Maps collection to 3D boolean
    lev_levs_map : dict
        Maps collection to list of level strings (for sl2d variables)
    season : str
        Season string (e.g., 'DJF')
    fvars : dict
        Maps collection to variable lists

    Returns:
    --------
    tuple of (str, str, GridSpec, Figure)
        pre : Filename prefix (without lead time suffix)
        ending : Plot title ending text
        gs : GridSpec object for subplot layout
        fig : Figure object
    """

    # Determine suffix and number of columns based on mode
    # suffix = "comparison"
    ncols = 3

    # Determine plot type and create filename prefix
    pre = f"/stats_{title_map[coll][v]}_GLO_{lev}_{season}"

    # Create title ending text
    if is_3d_map[coll]:
        ending = f"Var: {long_map[coll][v]}, Level: {lev} mb, Hour: {lead}"
    elif coll[:2] == "de":
        ending = f"{long_map[coll][v]}  Level: 1000 mb  Hour: {lead}"
    elif coll[:2] == "sl":
        ending = f"{long_map[coll][v]}  Level: {lev_levs_map[coll][v]}m  Hour: {lead}"
    else:  # ae2d
        ending = f"{long_map[coll][v]}  Level: surface  Hour: {lead}"

    # Create figure and gridspec
    fig_width = 15
    fig = plt.figure(figsize=(fig_width, 7))

    gs = gridspec.GridSpec(
        5,
        ncols,
        figure=fig,
        hspace=0.45,  # Add spacing between all rows
        height_ratios=[1, 0.04, 0.20, 1, 0.04],  # Increase middle spacer
    )

    return pre, ending, gs, fig


def setup_level_subplot(ax, data, lons, lats, masks=None):
    """
    Assemble input components and format coastlines and ticks for an
    individual subplot for level plots.

    Adds cyclic points to data and masks (to close map at dateline), adds
    coastlines, and formats lat/lon tick labels for cartopy map plots.

    Parameters:
    -----------
    ax : cartopy.mpl.geoaxes.GeoAxes
        The subplot axes with cartopy projection
    data : ndarray
        2D array of data to plot (lat x lon)
    lons : ndarray
        Longitude values
    lats : ndarray
        Latitude values
    masks : list or None, optional
        List of significance masks (each is 2D array: lat x lon)
        If provided, cyclic points are added to each mask

    Returns:
    --------
    tuple of (x, y, data, masks)
        x : Longitude values with cyclic point added
        y : Latitude values (unchanged)
        data : Data array with cyclic point added
        masks : Masks with cyclic points added (or None if input was None)
    """

    # Add cyclic point to data and longitudes
    data_cyclic, lons_cyclic = add_cyclic_point(data, coord=lons)

    # Add cyclic point to masks if provided
    masks_cyclic = None
    if masks is not None:
        masks_cyclic = [
            add_cyclic_point(mask, coord=lons)[0] for mask in masks
        ]

    # Add coastlines
    ax.coastlines(linewidth=0.8, color="black")

    # Set longitude ticks and labels
    ax.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
    ax.set_xticklabels(
        ["0", "60E", "120E", "180", "120W", "60W", "0"], fontsize=8
    )

    # Set latitude ticks and labels
    ax.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
    ax.set_yticklabels(
        ["90S", "60S", "30S", "0", "30N", "60N", "90N"], fontsize=8
    )

    # Return coordinate arrays and cyclic data/masks
    return lons_cyclic, lats, data_cyclic, masks_cyclic


def generate_masks(base_raw, comp_raw, v, sig_levs, dim, nfcsts, nlats, nlons):
    """
    Generate significance mask(s) for comp plots from model/comp raw data.

    Uses paired t-tests to determine where differences between models are
    statistically significant at the specified confidence levels. Returns
    boolean masks indicating significance for each statistic and level.

    Parameters:
    -----------
    base_raw : ndarray
        Base model raw data with shape:
        - 3D: (n_init_dates, n_stats, n_vars, n_levs, n_lats, n_lons)
        - 2D: (n_init_dates, n_stats, n_vars, n_lats, n_lons)
    comp_raw : ndarray
        Comparison model raw data (same shape as base_raw)
    v : int
        Variable index to extract from the data
    sig_levs : list
        List of significance levels (e.g., [0.90, 0.95] for 90% and 95%)
    dim : int
        Dimension indicator: 3 for 3D data, 2 for 2D data
    nfcsts : int
        Number of forecast initialization dates (sample size for t-test)
    nlats : int
        Number of latitude points
    nlons : int
        Number of longitude points

    Returns:
    --------
    list of lists
        Nested structure: [stat_idx][sig_level_idx] -> boolean mask array
        - For 3D: mask shape is (n_levs, n_lats, n_lons)
        - For 2D: mask shape is (n_lats, n_lons)
        - True where difference is statistically significant

    Notes:
    ------
    - Uses two-tailed t-test with degrees of freedom = nfcsts - 1
    - Handles division by zero gracefully (sets to NaN)
    - Differences calculated as comp - base (positive = comp larger)
    """

    # Process all stats at once
    t_critical = np.array(
        [stats.t.ppf((1 + sig) / 2, nfcsts - 1) for sig in sig_levs]
    )

    if dim == 3:
        all_diff_data = comp_raw[:, :, v, :, :, :] - base_raw[:, :, v, :, :, :]
        # Shape: (n_init, n_stats, n_levs, nlats, nlons)
    else:
        all_diff_data = comp_raw[:, :, v, :, :] - base_raw[:, :, v, :, :]
        # Shape: (n_init, n_stats, nlats, nlons)

    # Compute along init dimension (axis=0) for all stats at once
    mean_diff = np.mean(all_diff_data, axis=0)
    var_diff = np.var(
        all_diff_data, axis=0, ddof=1
    )  # ddof=1 for sample variance

    # Compute t-values for all stats
    t_values = np.divide(
        mean_diff * np.sqrt(nfcsts - 1),
        np.sqrt(var_diff),
        out=np.full_like(mean_diff, np.nan),
        where=var_diff > 0,
    )

    # t_values shape: (n_stats, n_levs, nlats, nlons) for 3D
    #                 (n_stats, nlats, nlons) for 2D

    # Create masks for each stat
    n_stats = base_raw.shape[1]
    masks = []

    for stat_idx in range(n_stats):
        # Extract t-values for this statistic
        stat_t_values = t_values[stat_idx]

        # Create mask for each significance level
        stat_masks = [np.abs(stat_t_values) >= t_crit for t_crit in t_critical]

        masks.append(stat_masks)

    return masks


def create_filled_colorbar(
    cs, fig, cax, bounds, vmax, ticks_vis, is_contour=False
):
    """Create colorbar for filled plots."""

    # Check if colormap is continuous (LinearSegmentedColormap) or discrete (ListedColormap)
    is_continuous = isinstance(cs.cmap, mcolors.LinearSegmentedColormap)

    # Create colorbar with different method for contour vs contourf
    if is_contour:
        if is_continuous:
            # For continuous colormaps, use Normalize for smooth gradients
            norm = mcolors.Normalize(vmin=min(bounds), vmax=max(bounds))
            sm = ScalarMappable(norm=norm, cmap=cs.cmap)
            sm.set_array(np.ones((1, 1)))
            cbar = fig.colorbar(
                sm,
                cax=cax,
                orientation="horizontal",
                drawedges=False,  # No edges for continuous colormaps
            )
        else:
            # For discrete colormaps, use BoundaryNorm for binned colors
            norm = BoundaryNorm(boundaries=bounds, ncolors=cs.cmap.N)
            sm = ScalarMappable(norm=norm, cmap=cs.cmap)
            sm.set_array(np.ones((1, 1)))
            cbar = fig.colorbar(
                sm,
                cax=cax,
                orientation="horizontal",
                boundaries=bounds,
                drawedges=True,
            )
    else:
        # For contourf, let it handle the colorbar automatically
        # but still respect continuous vs discrete
        cbar = fig.colorbar(
            cs,
            cax=cax,
            orientation="horizontal",
            drawedges=(not is_continuous)
        )

    # Fix floating point precision for values near zero
    ticks_clean = np.array([0.0 if abs(t) < 1e-10 else t for t in ticks_vis])

    # Determine precision based on vmax
    precision = 2 if vmax < 1 else (1 if vmax < 10 else 0)

    # Format tick labels with special handling for zero
    ticklabels = []
    for i, val in enumerate(ticks_clean):
        if i % 2 == 0:  # Every other tick gets a label
            if val == 0.0:
                # Special case: always format zero as "0"
                label = "0"
            else:
                # Format with appropriate precision
                label = f"{val:.{precision}f}"
                # Apply compact formatting only for non-zero values
                if vmax < 1:
                    label = label.replace("-0.", "-.").replace("0.", ".")
            ticklabels.append(label)
        else:
            ticklabels.append("")

    # Apply formatting to colorbar
    cbar.ax.tick_params(size=2, bottom=True, labelbottom=True, labelsize=8)
    cbar.set_ticks(ticks_clean)
    cbar.set_ticklabels(ticklabels)
    cbar.ax.xaxis.set_tick_params(pad=2)

    return cbar


def create_empty_colorbar(fig, cax):
    """
    Create white (empty) colorbar with only center 0 tick for empty plots.

    Creates a placeholder colorbar that maintains visual consistency when a
    subplot has no data to display. Shows only a "0" label at the center.

    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure object
    cax : matplotlib.axes.Axes
        Colorbar subplot axis

    Returns:
    --------
    matplotlib.colorbar.Colorbar
        The created empty colorbar object
    """

    # Create white colormap and scalar mappable
    norm = Normalize(vmin=-1, vmax=1)
    white_cmap = ListedColormap(["white"])
    sm = ScalarMappable(norm=norm, cmap=white_cmap)
    sm.set_array([])

    # Create colorbar with single tick at 0
    cbar = fig.colorbar(
        sm, cax=cax, orientation="horizontal", drawedges=True, ticks=[0]
    )

    # Format colorbar
    cbar.ax.tick_params(size=2, bottom=True, labelbottom=True, labelsize=8)
    cbar.set_ticklabels(["0"])
    cbar.ax.xaxis.set_tick_params(pad=2)

    return cbar


def extract_number(filename):
    """Extract numeric suffix from filename like image_0042.gif"""
    match = re.search(r"(\d+)(?=\.gif$)", filename)
    return int(match.group()) if match else -1


def get_limits_from_diff(diff_data, percentile):
    """Used for forecasts, gets percentile limits from diff."""
    return round_down_max(diff_data, percentile=percentile)


def get_limits_comp(
    base, comp, index_str, use_mean=False, axis=-1, percentiles=None
):
    """
    Get appropriate max limits for ACORR and RMS subplots using ALL lead times.
    """

    # Parse index string into slice tuple
    dims = tuple(
        slice(None) if part.strip() == ":" else int(part.strip())
        for part in index_str.split(",")
    )

    # Prepend slice(None) for lead time dimension
    dims_with_leads = (slice(None),) + dims

    # Helper to extract difference and optionally average for a given stat index
    def get_diff_data(idx):
        # Apply dims_with_leads to include lead dimension
        diff = base[:, idx][dims_with_leads] - comp[:, idx][dims_with_leads]
        return np.nanmean(diff, axis=axis) if use_mean else diff

    # Calculate rounded max values for ACORR and RMS
    Amax, Rmax = (
        round_down_max(get_diff_data(idx), percentile=percentiles[idx])
        for idx in [1, 2]
    )

    return Amax, Rmax


def plot_level_subplot(
    fig, ax, cax, i, title, zdata, cmap_key, mask, x, y,
    plot_data, colormaps, cmax=None, cmin=None,
    FCmax=None, MEmax=None, Rmax=None, Amax=None,
):
    """Create and render a single subplot for level (map) plots."""

    # Don't fill colors if max abs value is 0 (no diff)
    fill = False

    # ref data is the global avg diff
    ref_data = plot_data[2][1] if i == 2 else zdata

    # Extract max based on stat index
    vmax = [FCmax, Amax, Rmax][i]

    if np.nanmax(abs(ref_data)) > 0:
        fill = True
        vmin = -vmax
        offset = (vmax - vmin) / 20 / 2

        # More levels for continuous colormap
        if isinstance(colormaps[cmap_key], mcolors.LinearSegmentedColormap):
            n_levels = 256  # Smooth continuous
        else:
            n_levels = 22   # Keep discrete binning

        bounds = np.linspace(vmin - offset, vmax + offset, n_levels)

        # For ticks, always use ~20 positions regardless of n_levels
        if n_levels > 50:
            # For smooth colormaps, create tick positions separately
            ticks_vis = np.linspace(vmin, vmax, 21)
        else:
            # For discrete colormaps, use bin centers
            ticks_vis = (bounds[:-1] + bounds[1:]) / 2

        # Create a mask for NaN values (moved outside the if/else above)
        nan_mask = np.isnan(zdata)

        # Plot gray background for NaN regions FIRST
        if np.any(nan_mask):
            # Plot NaN locations as gray
            ax.contourf(
                x, y, nan_mask.astype(float),
                levels=[0.5, 1.5],
                colors=['gray'],
                alpha=0.5,
                extend='neither'
            )

        # Get the colormap (don't set_bad since we're handling NaNs separately)
        cmap = colormaps[cmap_key].copy()

        # Plot the data using contourf
        cs = ax.contourf(
            x,
            y,
            zdata,
            levels=bounds,
            cmap=cmap,
            extend="neither",
        )

    # Add significance contours for comp mode
    if mask is not None and fill and i > 0:
        for s, sig_mask in enumerate(mask):
            ax.contour(
                x,
                y,
                sig_mask.astype(int),
                levels=[0.5],
                colors="k",
                linestyles=styles[s],
                linewidths=0.2,
                alpha=0.5,
            )

    # Create colorbar
    if fill:
        create_filled_colorbar(
            cs,
            fig,
            cax,
            bounds,
            abs(vmax),
            ticks_vis,
            is_contour=(i == 0),
        )
    else:
        create_empty_colorbar(fig, cax)

    # Set title and map extent
    ax.set_title(title, fontsize=10)
    ax.set_global()


def create_level_plots(
    # mode,
    exps_to_comp,
    leads,
    v,
    var,
    coll,
    levs,
    data,
    models,
    nfcsts,
    season_year,
    is_3d_map,
    title_map,
    long_map,
    unit_map,
    lev_levs_map,
    lats,
    lons,
    nlats,
    nlons,
    fvars,
    season,
    output_dir,
    dpi,
    colormaps,
    sig_levs,
    fcst_interval,
    levels_to_plot=None,
    lead_indices=None,
    limit_percentiles=None,
    colormap_key=None,
    gif_frame_duration=1000,
):
    """
    Create all level plots (horizontal maps) for a variable.

    Level plots show horizontal maps at specific pressure levels (for 3D
    variables) or at a nominal surface level (for 2D variables). Each lead
    time gets plots for all levels, followed by animation per level.

    Parameters:
    -----------
    exps_to_comp: list, optional
        List of indices to compare, in the form [exp_idx, baseline idx];
        Diff will be calculated as exp_stat - base_stat
    leads : list
        List of lead times in hours
    v : int
        Variable index within collection
    var : str
        Variable name
    coll : str
        Collection name
    levs : list
        List of levels to plot (multiple for 3D, single for 2D)
    data : dict
        Data dictionary with 'raw', 'avg', 'glo' arrays
    models : list
        List of model names (needed for comp mode)
    nfcsts : int
        Number of forecast initializations
    season_year : str
        Season and year string for titles
    is_3d_map : dict
        Maps collections to 3D boolean
    title_map : dict
        Maps collections to variable title strings
    long_map : dict
        Maps collections to variable long names
    unit_map : dict
        Maps collections to variable units
    lev_levs_map : dict
        Maps collections to level strings (for sl2d)
    lats : array
        Latitude array
    lons : array
        Longitude array
    nlats : int
        Number of latitudes
    nlons : int
        Number of longitudes
    fvars : dict
        Maps collections to variable lists
    season : str
        Season string
    output_dir : str
        Output directory path
    dpi : int
        Plot resolution
    colormaps : dict
        Dictionary of colormaps
    sig_levs : list
        Significance levels for comp mode
    fcst_interval: int
        number of hours between forecasts (used for file naming)
    levels_to_plot: list, optional
        Levels that we want to plot; others are ignored

    Returns:
    --------
    None
        Plots saved to disk
    """

    # Keep reference to ORIGINAL full levels list
    original_levels = levs[:]  # Make a copy before filtering

    # Filter levels
    if levels_to_plot:
        filter_set = set(levels_to_plot)
        levs = [lev for lev in levs if lev in filter_set]

    if len(exps_to_comp) != 2:
        raise ValueError(
            "<2 or >2 experiment comparisons detected, this is not supported."
        )

    print(f"\nSetting up variable output directory in: {output_dir}")

    var_dir = output_dir / var
    var_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Created directory: {var_dir}")

    # Create subdirectories for each level after filtering
    level_subdirs = {}
    for lev in levs:
        level_subdirs[lev] = var_dir / str(lev)
        level_subdirs[lev].mkdir(parents=True, exist_ok=True)
    print(f"  Created level subdirs in {var_dir}; {levs}")

    # ═══════════════════════════════════════════════════════════════
    # Loop through each lead time (OUTER LOOP)
    # ═══════════════════════════════════════════════════════════════
    # Store limits per comparison -- reused across lead times
    # Each level has its own limit
    level_limits = {0: {}, 1: {}}
    print(f"Iterating over lead times: {leads}")
    for n, lead in enumerate(leads):

        # ═══════════════════════════════════════════════════════════
        # Generate significance masks once per lead (for all levels)
        # ═══════════════════════════════════════════════════════════
        print("  Creating 3d masks per experiment comparison...")
        masks_3d = []
        for exp_idx, base_idx in exps_to_comp:
            # Generate full 3D masks once per lead
            masks_3d.append(
                generate_masks(
                    data["raw"][coll][exp_idx][:, n, :],
                    data["raw"][coll][base_idx][:, n, :],
                    v,
                    sig_levs,
                    dim=3,
                    nfcsts=nfcsts,
                    nlats=nlats,
                    nlons=nlons,
                )
            )
        print("    Done.")

        # ═══════════════════════════════════════════════════════════
        # Loop through each level (INNER LOOP)
        # ══════════════════════════════════════════════════════════
        for l, lev in enumerate(levs):
            actual_lev_idx = original_levels.index(lev)
            # ═══════════════════════════════════════════════════════
            # Extract level-specific masks (comp mode only)
            # ═══════════════════════════════════════════════════════
            print(
                f"    Plotting level {lev} mb (filtered index {l}, "
                f"actual index {actual_lev_idx})"
            )
            # Extract this level's masks from 3D masks
            # Masks_3d is now a 4d array, first dim is comparison idx
            masks = []
            for comp_result in masks_3d:  # iterate over 2 comparisons
                comp_level_masks = []
                for stat_masks in comp_result:  # iterate over 3 stats
                    # Extract level l from each significance level mask
                    level_masks = [
                        sig_mask[l, :, :] for sig_mask in stat_masks
                    ]
                    comp_level_masks.append(level_masks)
                masks.append(comp_level_masks)
            print("    Done.")

            # ═══════════════════════════════════════════════════════
            # Extract and prepare data for this lead/level
            # ═══════════════════════════════════════════════════════

            # Setup plot metadata
            print(f"   Creating plot metadata/gridspec for level: {lev}")
            u = unit_map[coll][v]
            pre, ending, gs, fig = setup_plot_metadata(
                coll,
                v,
                lev,
                lead,
                # mode,
                title_map,
                long_map,
                is_3d_map,
                lev_levs_map,
                season,
                fvars,
            )

            print("      Iterating over comparisons...")
            top_title = [""]
            for comp_idx, (exp_idx, base_idx) in enumerate(exps_to_comp):
                base_avg = data["avg"][coll][exp_idx][n]
                base_glo = data["glo"][coll][exp_idx][n]
                comp_avg = data["avg"][coll][base_idx][n]
                comp_glo = data["glo"][coll][base_idx][n]

                exp_model, base_model = models[exp_idx], models[base_idx]
                print(
                    f"        Comparison {comp_idx}: {exp_model} vs {base_model}"
                )

                # ═══════════════════════════════════════════════════════
                # Calculate limits once per comparison per level (on first lead only)
                # ═══════════════════════════════════════════════════════
                limit_percentiles = [75, 50, 50]
                if n == 0:  # First lead
                    # Extract fcst diff across all leads
                    # 1. Filter to leads we are using
                    lead_slice = lead_indices if lead_indices else slice(None)

                    # Indexing: all leads, 0 idx for forecast, v for var, l for lvl,
                    #           all lat, all lon
                    fcst_idx_slice = np.s_[lead_slice, 0, v, actual_lev_idx, :, :]
                    fcst_diff_lvl = (
                        data["avg"][coll][exp_idx][fcst_idx_slice]
                        - data["avg"][coll][base_idx][fcst_idx_slice]
                    )
                    fcst_diff_limit = get_limits_from_diff(
                        np.nanmean(fcst_diff_lvl, axis=-1),  # Zonal mean
                        percentile=limit_percentiles[0],
                    )
                    # Use ALL lead times to calculate ACC/RMSE max
                    Amax_lvl, Rmax_lvl = get_limits_comp(
                        data["avg"][coll][exp_idx][lead_slice, :],  # ALL leads
                        data["avg"][coll][base_idx][lead_slice, :],
                        f"{v},{l},:,:",
                        use_mean=True,
                        axis=-1,
                        percentiles=limit_percentiles,
                    )
                    # Assign level limits based on all lead times
                    level_limits[comp_idx][l] = (
                        fcst_diff_limit,
                        Amax_lvl,
                        Rmax_lvl,
                    )

                # Extract limits for this level, set unused to None
                FCmax, Amax, Rmax = level_limits[comp_idx][l]
                MEmax, cmax, cmin = None, None, None

                pct_as_str = ", ".join(f"{p}%" for p in limit_percentiles)
                print(
                    f"      Found limits (at {pct_as_str} percentiles): "
                    f"FC={FCmax}, ACC={Amax}, RMS={Rmax}"
                )

                def diff_slicer(i):
                    base_data = base_avg[i][v, actual_lev_idx, :, :]
                    comp_data = comp_avg[i][v, actual_lev_idx, :, :]
                    diff_data = base_data - comp_data

                    # DIAGNOSTIC
                    # if i == 0 and l == 0:  # Only print once per comparison
                    #     print(f"\n      DIAGNOSTIC for stat {i}, level {l}:")
                    #     print(f"        base_avg shape: {base_avg[i].shape}")
                    #     print(f"        base_data shape: {base_data.shape}")
                    #     print(f"        base_data NaN%: {np.sum(np.isnan(base_data))/base_data.size*100:.1f}%")
                    #     print(f"        comp_data NaN%: {np.sum(np.isnan(comp_data))/comp_data.size*100:.1f}%")
                    #     print(f"        diff_data NaN%: {np.sum(np.isnan(diff_data))/diff_data.size*100:.1f}%")
                    #     print(f"        base_glo value: {base_glo[i][v, l]}")
                    #     print(f"        comp_glo value: {comp_glo[i][v, l]}")

                    return [
                        diff_data,
                        base_glo[i][v, actual_lev_idx] - comp_glo[i][v, actual_lev_idx],
                        masks[comp_idx][i],
                    ]

                # 3 stats for comp mode (differences only)
                z0, z1, z2 = (diff_slicer(i) for i in range(3))
                # ncols = 3
                exp_model_short = exp_model.split("_")[0]
                base_model_short = base_model.split("_")[0]
                diff_string = f"({exp_model_short} - {base_model_short})"

                def min_max_diff(z):
                    min_val = float(np.nanmin(z[0]))
                    max_val = float(np.nanmax(z[0]))
                    return (f"{min_val:.3f}", f"{max_val:.3f}")

                def count_bad(z):
                    raw_bad = np.sum(np.isnan(z[0]))
                    total = z[0].size
                    percent_bad = (raw_bad / total) * 100
                    return raw_bad, percent_bad


                min_max_diffs = [min_max_diff(z) for z in [z0, z1, z2]]
                nan_counts = [count_bad(z) for z in [z0, z1, z2]]

                titles = [
                    (
                        f"Mean Forecast Diff {diff_string}: {z0[1]:.2f} {u}"
                        "\nMin/Max Diff: "
                        f"{min_max_diffs[0][0]}, {min_max_diffs[0][1]}"
                        # f"\nBad values (GRAY): {nan_counts[0][0]}, {nan_counts[0][1]:.1f}%"
                    ),
                    (
                        f"Mean ACC Diff (F-A) {diff_string}: {z1[1]:.2f}"
                        "\nMin/Max Diff: "
                        f"{min_max_diffs[1][0]}, {min_max_diffs[1][1]}"
                        # f"\nBad values (GRAY): {nan_counts[1][0]}, {nan_counts[1][1]:.1f}%"
                    ),
                    (
                        f"Global Mean RMSE Diff (F-A) {diff_string}: {z2[1]:.2f}"
                        "\nMin/Max Diff: "
                        f"{min_max_diffs[2][0]}, {min_max_diffs[2][1]}"
                        # f"\nBad values (GRAY): {nan_counts[2][0]}, {nan_counts[2][1]:.1f}%"
                    ),
                ]
                # colormap_key = "c" if colormap_key is none else colormap_key
                plot_data = [
                    (titles[0], z0[0], colormap_key, z0[2]),
                    (titles[1], z1[0], colormap_key, z1[2]),
                    (titles[2], z2[0], colormap_key, z2[2]),
                ]

                # ═══════════════════════════════════════════════════════
                # Create each subplot
                # ═══════════════════════════════════════════════════════
                print(f"          Creating comp plots for row {comp_idx}...")
                for i, (title, zdata, cmap_key, mask) in enumerate(plot_data):

                    # Setup subplot axis
                    prow = comp_idx * 3
                    col = i
                    ax = fig.add_subplot(
                        gs[prow, col], projection=ccrs.PlateCarree()
                    )
                    cax = fig.add_subplot(gs[prow + 1, col])

                    # Setup level subplot (add cyclic point, format map)
                    x, y, zdata, mask = setup_level_subplot(
                        ax, zdata, lons, lats, mask
                    )

                    # Create the subplot plot and colorbar
                    plot_level_subplot(
                        fig,
                        ax,
                        cax,
                        i,
                        title,
                        zdata,
                        cmap_key,
                        mask,
                        x,
                        y,
                        plot_data,
                        colormaps,
                        cmax,
                        cmin,
                        FCmax,
                        MEmax,
                        Rmax,
                        Amax,
                    )

                # ═══════════════════════════════════════════════════════
                # Add parts to main title
                # ═══════════════════════════════════════════════════════
                # top_title.append(
                #     f"Row {comp_idx}: {exp_model} vs {base_model}"
                # )

            # ═══════════════════════════════════════════════════════
            # Add main title to figure, save to disk
            # ═══════════════════════════════════════════════════════

            print("        Saving plot for var/lead/level...")

            # MODE IS ALWAYS COMP NOW
            top_title += [
                f"Forecasts Statistics ({nfcsts})  {season_year}",
                f"{ending}",
                "(contours: >90% confidence)"
            ]
            top_title = "\n".join(top_title)

            plt.suptitle(top_title, fontsize=12, y=0.98)  # Smaller font
            plt.subplots_adjust(
                top=0.76
            )  # More space at top (lower value = more space)

            # Save plot
            plotnm = f"{pre}_lead_{lead}_hrs.png"
            save_path = level_subdirs[lev] / plotnm.lstrip("/")
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
            plt.close("all")
            print(f"        Done.")


    # ═══════════════════════════════════════════════════════════════
    # Save all plots to .gif across all lead times
    # ═══════════════════════════════════════════════════════════════
    print(f"Generating .gifs of all plots...")
    for lev, level_subdir in level_subdirs.items():
        print(f"  Processing level subdir: {level_subdir}...")
        png_filenames = list(level_subdir.glob("*.png"))
        png_images = [Image.open(fn) for fn in png_filenames]
        base_output_name = str(png_filenames[0]).split("_lead")[0]
        png_images[0].save(
            f"{base_output_name}_all_lead.gif",
            save_all=True,
            append_images=png_images[1:],
            duration=gif_frame_duration,
            loop=0,
        )
        print(f"  Saved to: {base_output_name}_all_lead.gif!")

# ================== SMALL DRIVER HELPERS ====================


def validate_data_dict(data_dict: Dict[str, Any]):
    # Validate required keys in data_dict
    required_keys = [
        "avg",
        "glo",
        "date_nms",
        "leads",
        "fcst_names",
        "ana_names",
        "clim_names",
        "fcst_length",
        "fcst_interval",
        "fvars",
        "levels",
        "lats",
        "lons",
        "collections",
        "is_3d",
        "plot_stats",
        "season",
        "year",
        "plot_dpi",
        "out_suffix",
    ]

    missing_keys = [key for key in required_keys if key not in data_dict]
    if missing_keys:
        raise ValueError(f"Missing required keys in data_dict: {missing_keys}")

    # For comp mode, also need raw data
    if "raw" not in data_dict:
        raise ValueError(
            "'raw' data required in data_dict to create COMP significance masks."
        )

    return True


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


# ================== MAIN DRIVER FUNCTION ====================


def create_global_plots(
    data_dict: Dict[str, Any],
    vars_to_plot: List[str] = None,
    levels_to_plot: List[int] = None,
    leads_to_plot: List[int] = None,
    exps_to_comp: List[List[int]] = [[1, 0]],
    limit_percentiles: List[int] = [75, 75, 75],
    colormap_key: str = "ryb",
    gif_frame_duration: int = 1000,
    output_dir: str = "output",
) -> None:
    """
    Main driver function for creating global statistics plots.

    Parameters:
    -----------
    data_dict : dict
        Dictionary containing loaded statistics data with keys:
        - 'raw', 'avg', 'glo': data arrays (raw only required for comp mode)
          Each nested by experiment index: data['avg'][coll][exp_idx][...]
        - 'date_nms', 'leads': time information
        - 'fcst_names', 'ana_names', 'clim_names': model names
        - 'fcst_length', 'fcst_interval': forecast timing
        - 'fvars', 'levels', 'lats', 'lons': spatial/variable info
        - 'collections', 'is_3d': data organization
        - 'plot_stats': statistics to plot
        - 'season', 'year': temporal metadata
        - 'plot_dpi': output quality
        - 'out_suffix': output suffix

    vars_to_plot : list, optional
        List of variables to plot (str)

    levels_to_plot : list, optional
        List of levels to plot (int)

    leads_to_plot: list, optional
        List of indices of leads to plot (1 day after init is 1)

    exps_to_comp: list
        List of indices to compare, in the form [exp_idx, baseline idx];
        Diff will be calculated as exp_stat - base_stat

    output_dir : str
        Directory path where plots will be saved

    gif_frame_duration: int
        Number of milliseconds to show each frame in the .gif.

    Returns:
    --------
    None
        Plots are saved to disk in the specified output directory
    """

    # ========== VALIDATION ==========

    print(f'\n{"="*60}')
    print(f"GLOBAL STATS PLOTTING DRIVER - COMP MODE")
    print(f'{"="*60}\n')

    valid = validate_data_dict(data_dict)

    # Check coastline data availability
    check_coastline_data()

    # ========== EXTRACT DATA FROM DICT ==========

    print("Extracting data from dictionary...")

    # Date/time info
    date_nms = data_dict["date_nms"]
    nfcsts = len(date_nms)
    leads = data_dict["leads"]
    nleads = len(leads)
    fcst_length = data_dict["fcst_length"]
    fcst_interval = data_dict["fcst_interval"]

    # Plot input info
    fvars = data_dict["fvars"]
    collections = data_dict["collections"]
    is_3d = data_dict["is_3d"]
    nvars = {coll: len(fvars[coll]) for coll in collections}
    levs = data_dict["levels"]
    nlevs = len(levs)
    lats = data_dict["lats"]
    nlats = len(lats)
    lons = data_dict["lons"]
    nlons = len(lons)
    nstats = len(data_dict["plot_stats"])

    # ========== DISCOVER AVAILABLE EXPERIMENTS ==========

    # Extract which experiments actually exist in the data
    available_exps = sorted(list(data_dict["avg"][collections[0]].keys()))
    print(f"Available experiments in data: {available_exps}")

    # Determine which experiments to plot
    exps_to_plot = available_exps
    print(
        f"Comparison mode: will compare {len(exps_to_plot)} experiments to baseline (exp 0)"
    )

    # Plot label info
    season_year = get_season(data_dict)

    # Build models dict keyed by experiment index
    models = {}
    for exp_idx in available_exps:
        if data_dict["fcst_names"][exp_idx] == "f5295fp":
            data_dict["fcst_names"][
                exp_idx
            ] = "GEOSFP"  # replace f5295fp with GEOS
        models[exp_idx] = (
            f'{data_dict["fcst_names"][exp_idx]}_{data_dict["ana_names"][exp_idx]}'
        )

    print(f"\nModel mapping:")
    for exp_idx in available_exps:
        print(f"  Experiment {exp_idx}: {models[exp_idx]}")

    if exps_to_comp is not None:
        print("Received a list of experiments to compare.")
        for idx, elem in enumerate(exps_to_comp):
            exp_idx, base_idx = elem
            model_exp, model_base = models[exp_idx], models[base_idx]
            print(
                f"For comparison {elem}, will compare {model_exp} to {model_base}"
            )

    # Filter out variables
    fvars_filtered = fvars  # Keep original for metadata

    if vars_to_plot is not None:
        print(f"\nFiltering variables to: {vars_to_plot}")
        fvars_filtered = {
            coll: [v for v in fvars[coll] if v in vars_to_plot]
            for coll in collections
        }
        print(f"Filtered fvars: {fvars_filtered}")

    if leads_to_plot is not None:
        print(f"Filtering lead times to: {leads_to_plot}")
        lead_indices = [
            idx for idx, lead in enumerate(leads) if lead in leads_to_plot
        ]
        leads = [lead for lead in leads if lead in leads_to_plot]

    # Variable metadata (use original unfiltered fvars)
    title_map, long_map, unit_map, lev_levs_map = {}, {}, {}, {}
    for coll in collections:
        title_map[coll] = [
            vars_title_map.get(var.upper()) for var in fvars[coll]
        ]
        long_map[coll] = [
            vars_long_map.get(var.upper()) for var in fvars[coll]
        ]
        unit_map[coll] = [
            vars_unit_map.get(var.upper()) for var in fvars[coll]
        ]
        if coll[:2] == "sl":
            lev_levs_map[coll] = [
                re.search(r"\d+", s).group() for s in fvars[coll]
            ]

    # Define colormaps
    # Create red-to-blue colormap without white
    n_colors = 21
    rdbu_colors = [plt.cm.RdBu_r(i / (n_colors - 1)) for i in range(n_colors)]
    rdbu_discrete = mcolors.ListedColormap(rdbu_colors)

    # Create continuous colormap with white center and many colors (256 by default)
    c_cmap_continuous = mcolors.LinearSegmentedColormap.from_list(
        'custom_continuous', c_cmap, N=256
    )

    colormaps = {
        "c": c_cmap_continuous,                    # continuous colormap with white (high detail)
        "rb": rdbu_discrete,                       # red to blue without white
        "ryb": plt.cm.RdYlBu_r(np.linspace(0, 1, 21)),  # red/yellow/blue
        "b": mcolors.ListedColormap(c_cmap[:11]),  # blue half of colormap
        "o": mcolors.ListedColormap(c_cmap),       # full colormap for contours
    }

    # Plot saving info
    dpi = data_dict["plot_dpi"]

    # ========== GENERATE PLOTS ==========

    print(f'\n{"="*60}')
    print(f"GENERATING COMP PLOTS")
    print(f'{"="*60}\n')

    global_subdir = Path(f"{output_dir}/global")

    # Common parameters to pass to plotting functions (using dicts)
    plot_params = {
        "data": data_dict,
        "models": models,
        "nfcsts": nfcsts,
        "season_year": season_year,
        "is_3d_map": is_3d,
        "title_map": title_map,
        "long_map": long_map,
        "unit_map": unit_map,
        "lev_levs_map": lev_levs_map,
        "lats": lats,
        "lons": lons,
        "nlats": nlats,
        "nlons": nlons,
        "fvars": fvars,
        "season": season_year,
        "output_dir": global_subdir,
        "dpi": dpi,
        "colormaps": colormaps,
        "sig_levs": sig_levs,
        "fcst_interval": fcst_interval,
        "levels_to_plot": levels_to_plot,  # KEEP - used for filtering
        "lead_indices": lead_indices,  # 4/1: added to fix cmap limits
        "limit_percentiles": limit_percentiles,  # 4/1: added as func arg
        "colormap_key": colormap_key,
        "gif_frame_duration": gif_frame_duration,
    }

    # Unified plotting loop -- one comparison per experiment to compare
    for coll in collections:
        print(f"  Collection: {coll}")
        if not is_3d[coll]:  # Filter out 2D behavior
            continue
        dim_label = "3D"

        for var in fvars_filtered[coll]:
            v = fvars[coll].index(var)

            print(
                f"    Making comp plots for {var} (index {v}, {dim_label})..."
            )
            # Pass levs as 7th positional, levels_to_plot in kwargs
            create_level_plots(
                # m, models[m],
                exps_to_comp,
                leads,
                v,
                var,
                coll,
                levs,
                **plot_params,
            )
    # ========== COMPLETION ==========

    print(f'\n{"="*60}')
    print(f"PLOTTING COMPLETE!")
    print(f'{"="*60}')
    print(f"Plots saved to: {global_subdir}")
    print(f"\nAll finished! :)\n")