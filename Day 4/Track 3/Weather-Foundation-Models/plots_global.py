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

red_yel_blue = [
    (0.00, 0.00, 0.55),  # 0  dark blue
    (0.00, 0.10, 0.65),  # 1  
    (0.00, 0.20, 0.75),  # 2  
    (0.00, 0.35, 0.85),  # 3  
    (0.00, 0.50, 0.95),  # 4  bright blue
    (0.10, 0.60, 1.00),  # 5  
    (0.20, 0.70, 1.00),  # 6  light blue
    (0.30, 0.80, 1.00),  # 7  
    (0.45, 0.85, 0.95),  # 8  cyan-blue
    (0.60, 0.90, 0.85),  # 9  light cyan
    (1.00, 1.00, 0.00),  # 10 pure yellow (center)
    (1.00, 0.95, 0.00),  # 11 
    (1.00, 0.85, 0.00),  # 12 golden yellow
    (1.00, 0.75, 0.00),  # 13 
    (1.00, 0.65, 0.00),  # 14 orange-yellow
    (1.00, 0.50, 0.00),  # 15 orange
    (1.00, 0.35, 0.00),  # 16 
    (0.95, 0.20, 0.00),  # 17 red-orange
    (0.85, 0.10, 0.00),  # 18 bright red
    (0.75, 0.00, 0.00),  # 19 
    (0.60, 0.00, 0.00),  # 20 dark red
]

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


# ================== HELPER FUNCTIONS ====================


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

    # ============= DEBUGGING =============
    stat_names = ['Forecast', 'ACC', 'RMSE']
    print(f"\n            PLOT_SUBPLOT DEBUG for {stat_names[i]}:")
    print(f"              zdata shape: {zdata.shape}")
    print(f"              zdata NaN: {np.sum(np.isnan(zdata))}/{zdata.size} ({np.sum(np.isnan(zdata))/zdata.size*100:.1f}%)")
    print(f"              ref_data type: {type(ref_data)}")
    if isinstance(ref_data, np.ndarray):
        print(f"              ref_data shape: {ref_data.shape}")
        print(f"              ref_data NaN: {np.sum(np.isnan(ref_data))}/{ref_data.size}")
    else:
        print(f"              ref_data value: {ref_data}")
    print(f"              vmax: {vmax}")
    print(f"              cmap_key: {cmap_key}")
    
    # Check if ref_data has any valid values before checking max
    if isinstance(ref_data, np.ndarray):
        has_valid = not np.all(np.isnan(ref_data))
    else:
        has_valid = not np.isnan(ref_data)
    
    print(f"              has_valid: {has_valid}")
    
    if has_valid:
        if isinstance(ref_data, np.ndarray):
            max_abs_ref = np.nanmax(np.abs(ref_data))
        else:
            max_abs_ref = abs(ref_data)
        print(f"              max_abs(ref_data): {max_abs_ref}")
    # =====================================

    # Check if ref_data has any valid values before checking max
    if not np.all(np.isnan(ref_data)) and np.nanmax(abs(ref_data)) > 0:
        fill = True

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
                extend='both'
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
            extend="both",
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

def plot_level_subplot_v2(
    fig, ax, cax, i, title, zdata, cmap_key, mask, x, y,
    colormaps, is_difference, vmax
):
    """
    Create and render a single subplot for level (map) plots.
    
    Handles both difference plots (centered at 0) and raw value plots (0 to max)
    using the same colormap with different bounds.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure object
    ax : cartopy.mpl.geoaxes.GeoAxes
        Map axes
    cax : matplotlib.axes.Axes
        Colorbar axes
    i : int
        Subplot index (0=forecast, 1=ACC, 2=RMSE)
    title : str
        Subplot title
    zdata : ndarray
        Data to plot (with cyclic point added)
    cmap_key : str
        Colormap key
    mask : list or None
        Significance masks (only used for difference plots)
    x : ndarray
        Longitude coordinates (with cyclic point)
    y : ndarray
        Latitude coordinates
    colormaps : dict
        Dictionary of colormaps
    is_difference : bool
        True for difference plots (centered at 0), False for raw values (0 to max)
    vmax : float
        Maximum value for colormap
    """
    
    # Check if we have valid data to plot
    has_valid = not np.all(np.isnan(zdata))
    
    if not has_valid:
        # Create empty colorbar for all-NaN data
        create_empty_colorbar(fig, cax)
        ax.set_title(title, fontsize=10)
        ax.set_global()
        return
    
    # Check if all values are effectively zero
    if np.nanmax(np.abs(zdata)) < 1e-10:
        create_empty_colorbar(fig, cax)
        ax.set_title(title, fontsize=10)
        ax.set_global()
        return
    
    # Set up colormap bounds based on whether this is a difference or raw value
    if is_difference:
        # Centered at 0 for differences: -vmax to +vmax
        vmin = -vmax
        offset = (vmax - vmin) / 20 / 2
        n_levels = 22
        bounds = np.linspace(vmin - offset, vmax + offset, n_levels)
        ticks_vis = np.linspace(vmin, vmax, 21)
    else:
        # From 0 to vmax for raw values (ACC, RMSE)
        vmin = 0
        offset = vmax / 20 / 2
        n_levels = 22
        bounds = np.linspace(vmin - offset, vmax + offset, n_levels)
        ticks_vis = np.linspace(vmin, vmax, 21)
    
    # Create NaN mask and plot gray background
    nan_mask = np.isnan(zdata)
    if np.any(nan_mask):
        nan_display = np.where(nan_mask, 1.0, np.nan)
        ax.pcolormesh(
            x, y, nan_display,
            cmap=mcolors.ListedColormap(['lightgray']),
            vmin=0, vmax=2,
            alpha=0.7,
            shading='auto',
            transform=ccrs.PlateCarree()
        )
    
    # Get colormap
    cmap = colormaps[cmap_key].copy()
    
    # Plot the data
    cs = ax.contourf(
        x, y, zdata,
        levels=bounds,
        cmap=cmap,
        extend="both",
    )
    
    # Add significance contours (only for difference plots with masks)
    if mask is not None and is_difference:
        for s, sig_mask in enumerate(mask):
            ax.contour(
                x, y, sig_mask.astype(int),
                levels=[0.5],
                colors="k",
                linestyles='solid',
                linewidths=0.2,
                alpha=0.5,
            )
    
    # Create colorbar
    create_filled_colorbar(cs, fig, cax, bounds, abs(vmax), ticks_vis, is_contour=(i == 0))
    
    # Set title and map extent
    ax.set_title(title, fontsize=10)
    ax.set_global()


def create_level_plots(
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
    stat_limits=None,
    colormap_key=None,
    gif_frame_duration=1000,
):
    """
    Create all level plots (horizontal maps) for a variable.

    Plots show:
    - Column 1: Forecast difference (exp - base)
    - Column 2: Raw ACC from exp
    - Column 3: Raw RMSE from exp

    Parameters:
    -----------
    exps_to_comp: list
        List of [exp_idx, baseline_idx] pairs for each row
        Example: [[2, 1], [3, 0]] creates 2 rows
    ... (other parameters same as before)
    """

    # Keep reference to ORIGINAL full levels list
    original_levels = levs[:]

    # Filter levels
    if levels_to_plot:
        filter_set = set(levels_to_plot)
        levs = [lev for lev in levs if lev in filter_set]

    if len(exps_to_comp) != 2:
        raise ValueError(
            "Expected exactly 2 comparisons for 2-row layout."
        )

    print(f"\nSetting up variable output directory in: {output_dir}")

    var_dir = output_dir / var
    var_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Created directory: {var_dir}")

    # Create subdirectories for each level
    level_subdirs = {}
    for lev in levs:
        level_subdirs[lev] = var_dir / str(lev)
        level_subdirs[lev].mkdir(parents=True, exist_ok=True)
    print(f"  Created level subdirs in {var_dir}; {levs}")

    # ═══════════════════════════════════════════════════════════════
    # Set up hard-coded colormap limits
    # ═══════════════════════════════════════════════════════════════
    var_upper = var.upper()
    
    if stat_limits is not None and var_upper in stat_limits:
        # Use provided hard-coded limits
        FCmax_diff = stat_limits[var_upper]['forecast_diff']
        ACCmax_raw = stat_limits[var_upper]['acc_raw']
        RMSmax_raw = stat_limits[var_upper]['rmse_raw']
        print(f"  Using hard-coded limits: FC_diff=±{FCmax_diff}, ACC_raw=[0,{ACCmax_raw}], RMS_raw=[0,{RMSmax_raw}]")
    else:
        # Fallback to default values
        FCmax_diff = 5.0
        ACCmax_raw = 1.0
        RMSmax_raw = 10.0
        print(f"  WARNING: No hard-coded limits for {var_upper}, using defaults")

    # Store limits for all levels (same for each level)
    level_limits = {}
    for l, lev in enumerate(levs):
        level_limits[l] = (FCmax_diff, ACCmax_raw, RMSmax_raw)

    # ═══════════════════════════════════════════════════════════════
    # Loop through each lead time (OUTER LOOP)
    # ═══════════════════════════════════════════════════════════════
    print(f"Iterating over lead times: {leads}")
    for n, lead in enumerate(leads):

        # ═══════════════════════════════════════════════════════════
        # Generate significance masks once per lead (for forecast diff only)
        # ═══════════════════════════════════════════════════════════
        print("  Creating 3d masks for forecast differences...")
        masks_3d = []
        for exp_idx, base_idx in exps_to_comp:
            # Generate masks for forecast differences only (stat_idx=0)
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
            
            print(f"    Plotting level {lev} mb (filtered index {l}, actual index {actual_lev_idx})")
            
            # ═══════════════════════════════════════════════════════
            # Extract level-specific masks for forecast differences
            # ═══════════════════════════════════════════════════════
            fcst_masks = []
            for comp_result in masks_3d:
                # Extract forecast mask (stat_idx=0)
                stat_masks = comp_result[0]  # First stat is forecast
                level_masks = [sig_mask[l, :, :] for sig_mask in stat_masks]
                fcst_masks.append(level_masks)
            print("    Done.")

            # ═══════════════════════════════════════════════════════
            # Setup plot metadata
            # ═══════════════════════════════════════════════════════
            print(f"   Creating plot metadata/gridspec for level: {lev}")
            u = unit_map[coll][v]
            pre, ending, gs, fig = setup_plot_metadata(
                coll, v, lev, lead, title_map, long_map, is_3d_map, lev_levs_map, season, fvars,
            )

            print("      Iterating over comparisons...")
            top_title = [""]

            first_exp_idx = exps_to_comp[0][0]
            first_model = models[first_exp_idx].split("_")[0]  # Get short name (e.g., "AIFS")
            
            for comp_idx, (exp_idx, base_idx) in enumerate(exps_to_comp):
                # Get data for both experiments
                exp_avg = data["avg"][coll][exp_idx][n]      # Comparison experiment
                exp_glo = data["glo"][coll][exp_idx][n]
                base_avg = data["avg"][coll][base_idx][n]    # Baseline experiment
                base_glo = data["glo"][coll][base_idx][n]

                exp_model, base_model = models[exp_idx], models[base_idx]
                print(f"        Comparison {comp_idx}: {exp_model} vs {base_model}")

                # ═══════════════════════════════════════════════════════
                # Extract limits
                # ═══════════════════════════════════════════════════════
                FCmax_diff, ACCmax_raw, RMSmax_raw = level_limits[l]

                if n == 0 and comp_idx == 0:
                    print(f"      Using limits: FC_diff=±{FCmax_diff}, ACC_raw=[0,{ACCmax_raw}], RMS_raw=[0,{RMSmax_raw}]")

                # ═══════════════════════════════════════════════════════
                # Extract data for each subplot
                # Column 1: Forecast DIFFERENCE (exp - base)
                # Column 2: Raw ACC from exp
                # Column 3: Raw RMSE from exp
                # ═══════════════════════════════════════════════════════
                
                # Column 1: Forecast difference
                fcst_exp = exp_avg[0][v, actual_lev_idx, :, :]
                fcst_base = base_avg[0][v, actual_lev_idx, :, :]
                fcst_diff = fcst_exp - fcst_base
                fcst_glo_diff = exp_glo[0][v, actual_lev_idx] - base_glo[0][v, actual_lev_idx]
                
                # Column 2: Raw ACC from comparison experiment
                acc_raw = exp_avg[1][v, actual_lev_idx, :, :]
                acc_glo_raw = exp_glo[1][v, actual_lev_idx]
                
                # Column 3: Raw RMSE from comparison experiment
                rmse_raw = exp_avg[2][v, actual_lev_idx, :, :]
                rmse_glo_raw = exp_glo[2][v, actual_lev_idx]

                # DEBUG - Only print for first lead/level/comparison
                if n == 0 and l == 0 and comp_idx == 0:
                    print(f"\n      DEBUG DATA EXTRACTION:")
                    print(f"        exp_avg shape: {np.array(exp_avg).shape}")
                    print(f"        exp_avg[0] (fcst) shape: {exp_avg[0].shape}")
                    print(f"        exp_avg[1] (acc) shape: {exp_avg[1].shape}")
                    print(f"        exp_avg[2] (rmse) shape: {exp_avg[2].shape}")
                    print(f"        acc_raw shape: {acc_raw.shape}")
                    print(f"        acc_raw NaN%: {np.sum(np.isnan(acc_raw))/acc_raw.size*100:.1f}%")
                    print(f"        acc_raw range: [{np.nanmin(acc_raw):.3f}, {np.nanmax(acc_raw):.3f}]")
                    print(f"        rmse_raw NaN%: {np.sum(np.isnan(rmse_raw))/rmse_raw.size*100:.1f}%")
                    print(f"        rmse_raw range: [{np.nanmin(rmse_raw):.3f}, {np.nanmax(rmse_raw):.3f}]")

                # ═══════════════════════════════════════════════════════
                # Create titles
                # ═══════════════════════════════════════════════════════
                exp_model_short = exp_model.split("_")[0]
                base_model_short = base_model.split("_")[0]

                def min_max_stat(data):
                    if np.all(np.isnan(data)):
                        return ("NaN", "NaN")
                    return (f"{float(np.nanmin(data)):.3f}", f"{float(np.nanmax(data)):.3f}")

                fcst_minmax = min_max_stat(fcst_diff)
                acc_minmax = min_max_stat(acc_raw)
                rmse_minmax = min_max_stat(rmse_raw)

                titles = [
                    (
                        f"Forecast Diff ({exp_model_short} - {base_model_short}): {fcst_glo_diff:.2f} {u}"
                        f"\nMin/Max: {fcst_minmax[0]}, {fcst_minmax[1]}"
                        f"\nMean Diff: {np.nanmean(fcst_diff):.2f} {u}"  # ADD THIS
                    ),
                    (
                        f"ACC (F-A) - {exp_model_short}: {acc_glo_raw:.3f}"
                        f"\nMin/Max: {acc_minmax[0]}, {acc_minmax[1]}"
                        f"\nMean ACC: {np.nanmean(acc_raw):.3f}"  # ADD THIS
                    ),
                    (
                        f"RMSE (F-A) - {exp_model_short}: {rmse_glo_raw:.2f} {u}"
                        f"\nMin/Max: {rmse_minmax[0]}, {rmse_minmax[1]}"
                        f"\nMean RMSE: {np.nanmean(rmse_raw):.2f} {u}"  # ADD THIS
                    ),
                ]
                
                # Data structure: (title, data, colormap_key, mask, is_difference, vmax)
                plot_data = [
                    (titles[0], fcst_diff, colormap_key, fcst_masks[comp_idx], True, FCmax_diff),   # Use passed colormap
                    (titles[1], acc_raw, 'acc', None, False, ACCmax_raw),                           # Use ACC colormap
                    (titles[2], rmse_raw, 'rmse', None, False, RMSmax_raw),                         # Use RMSE colormap
                ]

                # ═══════════════════════════════════════════════════════
                # Create each subplot
                # ═══════════════════════════════════════════════════════
                print(f"          Creating plots for row {comp_idx}...")
                for i, (title, zdata, cmap_key, mask, is_diff, vmax) in enumerate(plot_data):

                    # Setup subplot axis
                    prow = comp_idx * 3
                    col = i
                    ax = fig.add_subplot(gs[prow, col], projection=ccrs.PlateCarree())
                    cax = fig.add_subplot(gs[prow + 1, col])

                    # Setup level subplot (add cyclic point, format map)
                    x, y, zdata_cyclic, mask_cyclic = setup_level_subplot(ax, zdata, lons, lats, mask)

                    # Create the subplot plot and colorbar
                    plot_level_subplot_v2(
                        fig, ax, cax, i, title, zdata_cyclic, cmap_key, mask_cyclic, 
                        x, y, colormaps, is_diff, vmax
                    )

            # ═══════════════════════════════════════════════════════
            # Add main title to figure, save to disk
            # ═══════════════════════════════════════════════════════
            print("        Saving plot for var/lead/level...")
            
            top_title += [
                f"{first_model} Statistics - {season_year}",  # Changed: use model name, removed (nfcsts)
                f"{ending}",
                "Contours on forecast diff: >90% confidence"
            ]
            top_title = "\n".join(top_title)

            plt.suptitle(top_title, fontsize=12, y=0.98)
            plt.subplots_adjust(top=0.76)

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
        png_filenames = sorted(list(level_subdir.glob("*.png")))
        if png_filenames:
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
    limit_percentiles: List[int] = [90, 95, 95],  # Only used if stat_limits=None
    stat_limits: Dict[str, Dict[str, float]] = None,  # NEW: Hard-coded limits
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

    # ========== DEFAULT HARD-CODED LIMITS ==========
    # Define default limits if not provided
    if stat_limits is None:
        stat_limits = {
            'T': {'forecast_diff': 5.0, 'acc_raw': 1.0, 'rmse_raw': 10.0},
            'U': {'forecast_diff': 2.0, 'acc_raw': 1.0, 'rmse_raw': 5.0},
            'V': {'forecast_diff': 2.0, 'acc_raw': 1.0, 'rmse_raw': 5.0},
            'Z': {'forecast_diff': 50.0, 'acc_raw': 1.0, 'rmse_raw': 100.0},
            'Q': {'forecast_diff': 1.0, 'acc_raw': 1.0, 'rmse_raw': 2.0},
        }
    
    print(f"Using hard-coded colormap limits:")
    for var, limits in stat_limits.items():
        print(f"  {var}: Forecast_diff=±{limits['forecast_diff']}, ACC_raw=[0,{limits['acc_raw']}], RMSE_raw=[0,{limits['rmse_raw']}]")
    print()

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

    # rd_yl_bl = mcolors.ListedColormap(plt.cm.RdYlBu_r(np.linspace(0, 1, 21)))

    # colormaps = {
    #     "c": c_cmap_continuous,
    #     "rb": rdbu_discrete,
    #     "ryb": mcolors.ListedColormap(red_yel_blue),
    #     "b": mcolors.ListedColormap(c_cmap[:11]),
    #     "o": mcolors.ListedColormap(c_cmap),
    #     # New colormaps for ACC and RMSE
    #     "acc": mcolors.ListedColormap(plt.cm.RdYlGn(np.linspace(0, 1, 21))),    # Red (bad) to Green (good)
    #     "rmse": mcolors.ListedColormap(plt.cm.YlOrRd(np.linspace(0, 1, 21))),   # Yellow (good) to Red (bad)
    # }
    colormaps = {
        "c": c_cmap_continuous,
        "rb": rdbu_discrete,
        "ryb": plt.cm.viridis,      # Continuous - just use directly
        "b": plt.cm.viridis,
        "o": plt.cm.viridis,
        "acc": plt.cm.viridis,
        "rmse": plt.cm.viridis,
    }
    # colormaps = {
    #     "c": c_cmap_continuous,
    #     "rb": rdbu_discrete,
    #     "ryb": mcolors.ListedColormap(plt.cm.coolwarm(np.linspace(0, 1, 21))),  # Sample it
    #     "b": mcolors.ListedColormap(plt.cm.coolwarm(np.linspace(0, 1, 21))),
    #     "o": mcolors.ListedColormap(plt.cm.coolwarm(np.linspace(0, 1, 21))),
    #     "acc": mcolors.ListedColormap(plt.cm.coolwarm(np.linspace(0, 1, 21))),
    #     "rmse": mcolors.ListedColormap(plt.cm.coolwarm(np.linspace(0, 1, 21))),
    # }

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
        "stat_limits": stat_limits,  # 5/13: Add hard-coded limits
        # "limit_percentiles": limit_percentiles,  # 4/1: added as func arg
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