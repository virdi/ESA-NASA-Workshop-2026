import os
import re
from datetime import datetime
from typing import Dict, Any, List, Union

import numpy as np
import xarray as xr
import yaml
import sys

# ========================================================================
# CONSTANTS
# ========================================================================

COLLECTION_VARIABLES = {
    "de3d": ["H", "Q", "T", "U", "V"],
    "de2d": ["P", "PS"],
    "sl2d": ["Q2M", "T2M", "U10M", "V10M", "D2M"],
    "ae2d": ["AOD", "LOGAOD", "PM25"],
}

COLLECTION_DIMENSIONALITY = {
    "de3d": True,
    "de2d": False,
    "sl2d": False,
    "ae2d": False,
}


# ========================================================================
# MERGING OF DATASETS (02/26/26)
# ========================================================================
def _apply_excl(datasets, basename, excl_match, ds, experiment_excluded_dates):
    # If we found excluded regex match, but no metadata, throw err
    if "excluded_dates" not in ds.attrs:
        for ds_cleanup in datasets:
            ds_cleanup.close()
        sys.exit(
            f"ERROR: File {basename} contains "
            f'"excl{excl_match.group(1)}" pattern but '
            f'missing "excluded_dates" attribute in metadata'
        )

    # Parse excluded dates (handle both string and list formats)
    excluded_dates_str = ds.attrs["excluded_dates"]
    if isinstance(excluded_dates_str, (list, np.ndarray)):
        # Already a list/array, convert to strings and strip
        excluded_list = [str(item).strip() for item in excluded_dates_str]
    else:
        # String format - parse comma-separated values
        if "," in excluded_dates_str:
            excluded_list = [d.strip() for d in excluded_dates_str.split(",")]
        else:
            excluded_list = [excluded_dates_str.strip()]

    # Convert to standard format (add _00 if no hour specified)
    for date_str in excluded_list:
        if len(date_str) == 8:  # YYYYMMDD format
            formatted_date = f"{date_str}_00"
        else:
            formatted_date = date_str
        experiment_excluded_dates.add(formatted_date)
    return experiment_excluded_dates


def _open_datasets_apply_excl(filename_list):
    datasets = []
    files_with_exclusions = 0
    experiment_excluded_dates = set()
    for filename in filename_list:
        ds = xr.open_dataset(filename)
        datasets.append(ds)
        basename = os.path.basename(filename)
        excl_match = re.search(r"excl(\d+)", basename)
        if excl_match:
            experiment_excluded_dates = _apply_excl(
                datasets, basename, excl_match, ds, experiment_excluded_dates
            )
            files_with_exclusions += 1
    return datasets, files_with_exclusions, experiment_excluded_dates


def _find_date_overlap(datasets, filename_list, exp_name):
    all_dates = []
    date_counts = []
    for i, ds in enumerate(datasets):
        file_dates = set(ds.init_date.values)
        date_counts.append(len(file_dates))

        # Check for overlap with previously processed files
        overlap = set(all_dates) & file_dates
        if overlap:
            overlap_strs = [
                d.astype("datetime64[us]")
                .astype(datetime)
                .strftime("%Y%m%d_%H")
                for d in overlap
            ]
            # Clean up
            for ds_cleanup in datasets:
                ds_cleanup.close()
            sys.exit(
                f"ERROR: Duplicate dates found in {exp_name} files:\n"
                f"  File {i}: {os.path.basename(filename_list[i])}\n"
                f"  Overlapping dates: {overlap_strs}"
            )
        all_dates.extend(file_dates)
    return all_dates, date_counts


def merge_experiment_files(
    filename_list, exp_name="test", merge_mode="regional"
):
    """Merge multiple files for a single experiment

    Parameters
    ----------
    filename_list : list
        List of file paths to merge
    exp_name : str
        Experiment name for reporting
    merge_mode : str, optional
        'regional' - skip _avg variables, no weighted averaging
        'global' - weighted averaging for _avg and _glo variables
    """
    if len(filename_list) == 1:
        print(f"Loading single file for {exp_name}...")
    else:
        print(f"Merging {len(filename_list)} files for {exp_name}...")
        print(f"Filename list: \n{filename_list}")

    if merge_mode not in ["regional", "global"]:
        raise ValueError(
            'Error during merge: merge mode must be "regional" or "global".'
        )

    # Open all datasets and collect exclusion metadata as needed
    datasets, files_with_exclusions, experiment_excluded_dates = (
        _open_datasets_apply_excl(filename_list)
    )

    if len(datasets) > 1:  # Multiple files
        all_dates, date_counts = _find_date_overlap(
            datasets, filename_list, exp_name
        )

        # Separate variables by type
        first_ds = datasets[0]
        vars_with_init_date = []
        vars_without_init_date = []
        vars_avg = []
        vars_glo = []

        for var_name in first_ds.data_vars:
            if var_name.endswith("_avg"):
                if merge_mode == "global":
                    vars_avg.append(var_name)
                # else: skip entirely for regional mode
            elif var_name.endswith("_glo"):
                if merge_mode == "global":
                    vars_glo.append(var_name)
                else:
                    vars_without_init_date.append(var_name)
            elif "init_date" in first_ds[var_name].dims:
                vars_with_init_date.append(var_name)
            else:
                vars_without_init_date.append(var_name)

        # Merge variables with init_date dimension (regular concatenation)
        ds_with_init = xr.concat(
            [ds[vars_with_init_date] for ds in datasets], dim="init_date"
        ).sortby("init_date")

        # For non-avg/glo variables without init_date, take from first dataset
        ds_without_init = first_ds[vars_without_init_date]

        # Weighted averaging for global mode
        if merge_mode == "global":
            total_dates = sum(date_counts)

            # Weight _avg variables
            if vars_avg:
                ds_avg_weighted = datasets[0][vars_avg] * (
                    date_counts[0] / total_dates
                )
                for i in range(1, len(datasets)):
                    weight = date_counts[i] / total_dates
                    ds_avg_weighted = (
                        ds_avg_weighted + datasets[i][vars_avg] * weight
                    )
            else:
                ds_avg_weighted = xr.Dataset()

            # Weight _glo variables
            if vars_glo:
                ds_glo_weighted = datasets[0][vars_glo] * (
                    date_counts[0] / total_dates
                )
                for i in range(1, len(datasets)):
                    weight = date_counts[i] / total_dates
                    ds_glo_weighted = (
                        ds_glo_weighted + datasets[i][vars_glo] * weight
                    )
            else:
                ds_glo_weighted = xr.Dataset()

            merged_ds = xr.merge(
                [
                    ds_with_init,
                    ds_avg_weighted,
                    ds_glo_weighted,
                    ds_without_init,
                ]
            )
        else:  # regional mode
            merged_ds = xr.merge([ds_with_init, ds_without_init])

        # Close individual datasets
        for ds in datasets:
            ds.close()

    else:  # Single file case - skip merge operations
        merged_ds = datasets[0]

    # Report result
    final_dates = merged_ds.init_date.values
    first_date = (
        final_dates[0]
        .astype("datetime64[us]")
        .astype(datetime)
        .strftime("%Y%m%d_%H")
    )
    last_date = (
        final_dates[-1]
        .astype("datetime64[us]")
        .astype(datetime)
        .strftime("%Y%m%d_%H")
    )
    if len(datasets) > 1:
        print(
            f"Merged dataset: {len(final_dates)} init dates from "
            f"{first_date} to {last_date}"
        )
        if merge_mode == "global":
            print(f"Averaged variables weighted by date counts: {date_counts}")
    else:
        print(
            f"Dataset contains {len(final_dates)} init dates from "
            f"{first_date} to {last_date}"
        )

    return merged_ds, experiment_excluded_dates, files_with_exclusions


# ========================================================================
# Formatting, data extraction
# ========================================================================


def _format_variable_names(
    fvars: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    formatted_fvars = {}

    for key, var_list in fvars.items():
        if key.startswith("de") or key.startswith("ae"):
            # For collections beginning with 'de' or 'ae', make all uppercase
            formatted_fvars[key] = [var.upper() for var in var_list]

        elif key.startswith("sl"):
            # For 'sl', format as uppercase before numbers, lowercase after
            formatted_vars = []
            for var in var_list:
                # Find the first digit in the variable name
                match = re.search(r"\d", var)
                if match:
                    digit_pos = match.start()
                    # Uppercase before the digit, lowercase after
                    new_var = var[:digit_pos].upper() + var[digit_pos:].lower()
                    formatted_vars.append(new_var)
                else:
                    # If no digit found, raise error
                    raise ValueError(
                        f'Variable "{var}" in collection "{key}" must '
                        f"contain at least one digit."
                    )
            formatted_fvars[key] = formatted_vars
        else:
            # Unknown prefix - keep as-is
            formatted_fvars[key] = var_list

    return formatted_fvars


def _extract_timing_from_filename(
    filename: str, leads: List[int] = None
) -> Dict[str, int]:
    basename = os.path.basename(filename)

    # Extract from filename patterns
    len_match = re.search(r"len(\d+)", basename)
    int_match = re.search(r"int(\d+)", basename)

    fcst_length = int(len_match.group(1)) if len_match else None
    fcst_interval = int(int_match.group(1)) if int_match else None

    # Fallback to leads if provided and patterns not found
    if leads is not None:
        if fcst_length is None and len(leads) > 0:
            fcst_length = max(leads) // 24
        if fcst_interval is None and len(leads) > 1:
            fcst_interval = leads[1] - leads[0]

    return {"fcst_length": fcst_length, "fcst_interval": fcst_interval}


def _extract_model_names_from_filename(filename: str) -> Dict[str, str]:
    """
    Extract model names from stats filename.

    Expected format: stats_{type}_FCST_ANA_CLIM_{dates}_{suffix}
    Where type is 'regional' or 'global'

    Example:
        stats_global_GenCast_ERA5_ERA5_20241201-20241231_len10d_int12h_spc1d_91x144.nc4
        → fcst=GenCast, ana=ERA5, clim=ERA5
    """
    basename = os.path.basename(filename)
    parts = basename.split("_")

    # Validate basic structure
    if len(parts) < 6:
        print(f"WARNING: Filename has too few parts: {basename}")
        return {
            "fcst_name": "unknown",
            "ana_name": "unknown",
            "clim_name": "unknown",
        }

    if parts[0] != "stats":
        print(f"WARNING: Filename doesn't start with 'stats': {basename}")
        return {
            "fcst_name": "unknown",
            "ana_name": "unknown",
            "clim_name": "unknown",
        }

    # parts[1] should be file type (regional/global), but we don't need to validate it
    # parts[2:5] are the model names we want
    file_type = parts[1]  # 'regional' or 'global'

    # Extract model names (positions 2, 3, 4)
    fcst_name = parts[2]
    ana_name = parts[3]
    clim_name = parts[4]

    # parts[5] should be date range (validate format loosely)
    date_part = parts[5]
    if not re.match(r"\d{8}-\d{8}", date_part):
        print(
            f"WARNING: Unexpected date format in position 5: {date_part} in {basename}"
        )

    return {
        "fcst_name": fcst_name,
        "ana_name": ana_name,
        "clim_name": clim_name,
        "file_type": file_type,  # Optional: include for reference
    }


def _format_date_strings(date_values: np.ndarray) -> List[str]:
    return [
        d.astype("datetime64[us]").astype(datetime).strftime("%Y%m%d_%H")
        for d in date_values
    ]


# ============================================================================
# NEW SHARED HELPER FUNCTIONS FOR MULTI-EXPERIMENT SUPPORT
# ============================================================================


def _infer_collection_from_varname(var):
    """Map variable name to collection using known variable lists"""
    var_upper = var.upper()
    for coll, var_list in COLLECTION_VARIABLES.items():
        if var_upper in [v.upper() for v in var_list]:
            return coll
    return None


def _find_common_variables(datasets, collections):
    """
    Find variables and stats that exist in ALL datasets (intersection)

    Parameters:
    - datasets: List of xarray datasets (one per experiment)
    - collections: List of collection names like ['de3d', 'de2d', 'sl2d', 'ae2d']

    Returns:
    - common_vars: Dict[collection -> List[var]] - only vars present in all exps
    - common_stats: List[str] - only stats present in all exps
    """

    # Scan each dataset independently
    all_exp_vars = []  # List of {coll: [vars]} dicts, one per experiment
    all_exp_stats = []  # List of sets, one per experiment

    for exp_idx, ds in enumerate(datasets):
        # Initialize for this experiment
        exp_vars = {coll: [] for coll in collections}
        exp_stats = set()

        # Scan all variables in this dataset
        for var_name in ds.variables:
            # Skip coordinates and pre-computed averages
            if var_name in ds.coords or var_name.endswith("_avg"):
                continue

            # Parse variable name: VAR_stat (e.g., "H_acorr", "T2M_rms")
            parts = var_name.split("_")
            if len(parts) < 2:
                continue

            var = parts[0]  # e.g., "H", "T2M"
            stat = "_".join(parts[1:])  # e.g., "acorr", "rms_ran"

            # Determine which collection this variable belongs to
            collection = _infer_collection_from_varname(var)
            if collection is None:
                continue  # Variable doesn't match any known collection

            # Add to this experiment's lists
            if var not in exp_vars[collection]:
                exp_vars[collection].append(var)
            exp_stats.add(stat)

        all_exp_vars.append(exp_vars)
        all_exp_stats.append(exp_stats)

    # Compute intersection across all experiments
    common_vars = {}
    for coll in collections:
        # Get set of vars for this collection from each experiment
        vars_per_exp = [set(exp_vars[coll]) for exp_vars in all_exp_vars]

        # Intersection
        if vars_per_exp:  # Make sure list isn't empty
            common_vars[coll] = sorted(list(set.intersection(*vars_per_exp)))
        else:
            common_vars[coll] = []

    # Intersection of stats
    common_stats = sorted(list(set.intersection(*all_exp_stats)))

    return common_vars, common_stats


def _load_experiment_data(
    ds,
    collections,
    fvars,
    plot_stats,
    regions,
    is_3d_map,
    levels,
    common_date_indices,
    lead_indices,  # ← NEW PARAMETER
):
    """Load data from xarray dataset into structured numpy arrays"""

    data = {}
    n_init_dates = len(common_date_indices)
    n_leads = len(lead_indices)  # ← Use length of lead_indices

    for coll in collections:
        data[coll] = {}
        n_vars = len(fvars[coll])
        n_stats = len(plot_stats)
        n_levels = len(levels) if is_3d_map[coll] else None

        for region in regions:
            reg_idx = regions.index(region)

            # Create array
            if is_3d_map[coll]:
                raw_array = np.zeros(
                    (n_init_dates, n_leads, n_stats, n_vars, n_levels)
                )
            else:
                raw_array = np.zeros((n_init_dates, n_leads, n_stats, n_vars))

            # Fill array
            for var_idx, var in enumerate(fvars[coll]):
                for stat_idx, stat in enumerate(plot_stats):
                    var_name = f"{var}_{stat}"

                    if var_name not in ds:
                        print(
                            f"WARNING: Variable {var_name} not found, skipping"
                        )
                        continue

                    raw_data = ds[var_name].values

                    if is_3d_map[coll]:
                        # Filter to common dates AND common leads
                        raw_array[:, :, stat_idx, var_idx, :] = raw_data[
                            common_date_indices, :, :, reg_idx
                        ][:, lead_indices, :]
                    else:
                        # Filter to common dates AND common leads
                        raw_array[:, :, stat_idx, var_idx] = raw_data[
                            common_date_indices, :, reg_idx
                        ][:, lead_indices]

            data[coll][region] = raw_array

    return data


def _filter_arrays_to_common_dates(
    data,  # Dict with ['raw'][coll][exp_idx][region]
    collections,  # List of collection names
    regions,  # List of region names
    experiment_valid_dates,  # List of date lists per experiment
    common_dates,  # List of common dates
):
    """Filter raw arrays if experiments have different valid dates

    After loading, checks if all experiments ended up with same dates.
    If not (due to different exclusions), filters arrays to only common dates.

    Returns: (data, final_common_dates)
    """

    # Check if all experiments have identical dates
    all_same = all(
        set(exp_dates) == set(experiment_valid_dates[0])
        for exp_dates in experiment_valid_dates
    )

    if all_same:
        print("✓ All experiments have identical dates - no filtering needed")
        return data, common_dates

    # Need to filter - experiments have different dates
    print(f"⚠ Filtering arrays to {len(common_dates)} common dates...")

    for coll in collections:
        n_experiments = len(experiment_valid_dates)

        for exp_idx in range(n_experiments):
            exp_dates = experiment_valid_dates[exp_idx]

            # Find indices of common dates in this experiment's date list
            date_indices = [exp_dates.index(d) for d in common_dates]

            for region in regions:
                raw_array = data["raw"][coll][exp_idx][region]
                # Filter first dimension (init_date)
                data["raw"][coll][exp_idx][region] = raw_array[
                    date_indices, ...
                ]

    # Report filtering
    for exp_idx, exp_dates in enumerate(experiment_valid_dates):
        n_filtered = len(exp_dates) - len(common_dates)
        if n_filtered > 0:
            print(f"  exp{exp_idx}: filtered out {n_filtered} dates")

    return data, common_dates


def _normalize_filename_input(filenames: Union[str, List[str]]) -> List[str]:
    # Case 1: Single string → single experiment with one file
    if isinstance(filenames, str):
        return [[filenames]]

    # Case 2: List (need to check what kind)
    if isinstance(filenames, list):
        # Empty list
        if len(filenames) == 0:
            raise ValueError("Empty filename list provided")

        # Check first element to determine format
        first_elem = filenames[0]

        # Case 2a: List of lists (already correct format)
        if isinstance(first_elem, list):
            return filenames

        # Case 2b: List of strings (single experiment with multiple files)
        elif isinstance(first_elem, str):
            return [filenames]  # Wrap in another list

        else:
            raise TypeError(
                f"Invalid filename format. Expected str or list, "
                f"got {type(first_elem)}"
            )

    else:
        raise TypeError(
            f"Invalid filename type. Expected str, List[str], or List[List[str]], "
            f"got {type(filenames)}"
        )


def _validate_identical_levels(datasets):
    """Validate that all experiments have identical pressure levels"""
    if len(datasets) == 0:
        return []

    ref_levels = [int(x) for x in datasets[0].lev.values.tolist()]

    for exp_idx, ds in enumerate(datasets[1:], 1):
        curr_levels = [int(x) for x in ds.lev.values.tolist()]

        if curr_levels != ref_levels:
            raise ValueError(
                f"Experiment {exp_idx} has different pressure levels than exp0.\n"
                f"  exp0: {ref_levels}\n"
                f"  exp{exp_idx}: {curr_levels}\n"
                f"Pressure levels must be identical across experiments."
            )

    return ref_levels


def _find_common_leads(datasets, verbose=True):
    """Find lead times present in ALL datasets (intersection)

    Returns:
        common_leads: List of integer lead times (hours)
        lead_indices_per_exp: List of index arrays for each experiment
    """
    if len(datasets) == 0:
        return [], []

    # Extract leads from each dataset as sets
    all_exp_leads = []
    all_exp_lead_lists = []  # Preserve original order

    for ds in datasets:
        leads = [int(x) for x in ds.lead.values.tolist()]
        all_exp_leads.append(set(leads))
        all_exp_lead_lists.append(leads)

    # Find intersection
    common_leads_set = all_exp_leads[0]
    for lead_set in all_exp_leads[1:]:
        common_leads_set = common_leads_set.intersection(lead_set)

    if not common_leads_set:
        raise ValueError("No common lead times found across all experiments!")

    # Sort for consistent ordering
    common_leads = sorted(list(common_leads_set))

    # Find indices of common leads in each experiment
    lead_indices_per_exp = []
    for exp_leads in all_exp_lead_lists:
        indices = [exp_leads.index(lead) for lead in common_leads]
        lead_indices_per_exp.append(indices)

    # Report filtering
    if verbose:
        print(f"Common lead times: {len(common_leads)} steps")
        print(f"  Lead range: {common_leads[0]}h to {common_leads[-1]}h")
        print(f"  Interval: {common_leads[1] - common_leads[0]}h")

        for exp_idx, exp_leads in enumerate(all_exp_lead_lists):
            n_filtered = len(exp_leads) - len(common_leads)
            if n_filtered > 0:
                print(
                    f"  exp{exp_idx}: {n_filtered} lead times filtered out "
                    f"({len(exp_leads)} → {len(common_leads)})"
                )

    return common_leads, lead_indices_per_exp


def _compile_model_names(filenames: List[str]) -> Dict[str, List[str]]:
    """
    Extract model names from multiple filenames.

    Note: Removed file_type parameter - auto-detected now
    """
    fcst_names = []
    ana_names = []
    clim_names = []

    for filename in filenames:
        model_names = _extract_model_names_from_filename(
            filename
        )  # ✅ No file_type param
        fcst_names.append(model_names["fcst_name"])
        ana_names.append(model_names["ana_name"])
        clim_names.append(model_names["clim_name"])

    return {
        "fcst_names": fcst_names,
        "ana_names": ana_names,
        "clim_names": clim_names,
    }


def _validate_timing_compatibility(
    timing_params_list: List[Dict[str, int]],
    mode: str = "compatible",
    verbose: bool = True,
) -> Dict[str, int]:
    if len(timing_params_list) == 0:
        return {}

    ref_params = timing_params_list[0]
    ref_length = ref_params["fcst_length"]
    ref_interval = ref_params["fcst_interval"]

    for exp_idx, params in enumerate(timing_params_list[1:], 1):
        length = params["fcst_length"]
        interval = params["fcst_interval"]

        if mode == "identical":
            if length != ref_length:
                raise ValueError(
                    f"Experiment {exp_idx} has different forecast length than exp0.\n"
                    f"  exp0: {ref_length}d\n"
                    f"  exp{exp_idx}: {length}d"
                )
            if interval != ref_interval:
                raise ValueError(
                    f"Experiment {exp_idx} has different forecast interval than exp0.\n"
                    f"  exp0: {ref_interval}h\n"
                    f"  exp{exp_idx}: {interval}h"
                )

        elif mode == "compatible":
            if length and length < ref_length:
                if verbose:
                    print(
                        f"WARNING: exp{exp_idx} has shorter forecast length "
                        f"({length}d < {ref_length}d)"
                    )
            if interval and interval > ref_interval:
                if verbose:
                    print(
                        f"WARNING: exp{exp_idx} has coarser forecast interval "
                        f"({interval}h > {ref_interval}h)"
                    )

    return ref_params


# ============================================================================
# REGIONAL-SPECIFIC HELPER FUNCTIONS
# ============================================================================


def _validate_regional_spatial_coords(datasets: List[xr.Dataset]) -> List[str]:
    if len(datasets) == 0:
        return []

    if "region" not in datasets[0].coords:
        raise ValueError("'region' coordinate not found in experiment 0")

    ref_regions = datasets[0].region.values.tolist()

    for exp_idx, ds in enumerate(datasets[1:], 1):
        if "region" not in ds.coords:
            raise ValueError(
                f"'region' coordinate not found in experiment {exp_idx}"
            )

        curr_regions = ds.region.values.tolist()
        if curr_regions != ref_regions:
            raise ValueError(
                f"Experiment {exp_idx} has different regions than exp0.\n"
                f"  exp0: {ref_regions}\n"
                f"  exp{exp_idx}: {curr_regions}"
            )

    return ref_regions


def _find_common_dates_regional(datasets: List[xr.Dataset]) -> tuple:
    if len(datasets) == 0:
        return ([], [])

    # Extract date sets for each experiment
    exp_date_sets = []
    exp_date_lists = []

    for ds in datasets:
        exp_dates = _format_date_strings(ds.init_date.values)
        exp_date_sets.append(set(exp_dates))
        exp_date_lists.append(exp_dates)

    # Find intersection of all date sets
    common_dates_set = exp_date_sets[0]
    for date_set in exp_date_sets[1:]:
        common_dates_set = common_dates_set.intersection(date_set)

    common_dates = sorted(list(common_dates_set))

    # Find indices of common dates in each experiment's date array
    date_indices_per_exp = []
    for exp_dates in exp_date_lists:
        indices = [exp_dates.index(date) for date in common_dates]
        date_indices_per_exp.append(indices)

    return (common_dates, date_indices_per_exp)


# ============================================================================
# GLOBAL-SPECIFIC HELPER FUNCTIONS
# ============================================================================


def _validate_global_spatial_coords(
    datasets: List[xr.Dataset],
) -> Dict[str, np.ndarray]:
    if len(datasets) == 0:
        return {}

    if "lat" not in datasets[0].coords or "lon" not in datasets[0].coords:
        raise ValueError("'lat' or 'lon' coordinate not found in experiment 0")

    ref_lats = datasets[0].lat.values
    ref_lons = datasets[0].lon.values

    for exp_idx, ds in enumerate(datasets[1:], 1):
        if "lat" not in ds.coords or "lon" not in ds.coords:
            raise ValueError(
                f"'lat' or 'lon' coordinate not found in experiment {exp_idx}"
            )

        curr_lats = ds.lat.values
        curr_lons = ds.lon.values

        if not np.array_equal(curr_lats, ref_lats):
            raise ValueError(
                f"Experiment {exp_idx} has different latitudes than exp0.\n"
                f"  exp0: shape={ref_lats.shape}, range=[{ref_lats[0]}, {ref_lats[-1]}]\n"
                f"  exp{exp_idx}: shape={curr_lats.shape}, range=[{curr_lats[0]}, {curr_lats[-1]}]"
            )

        if not np.array_equal(curr_lons, ref_lons):
            raise ValueError(
                f"Experiment {exp_idx} has different longitudes than exp0.\n"
                f"  exp0: shape={ref_lons.shape}, range=[{ref_lons[0]}, {ref_lons[-1]}]\n"
                f"  exp{exp_idx}: shape={curr_lons.shape}, range=[{curr_lons[0]}, {curr_lons[-1]}]"
            )

    return {"lat": ref_lats, "lon": ref_lons}


def _validate_identical_dates_global(datasets: List[xr.Dataset]) -> List[str]:
    if len(datasets) == 0:
        return []

    # Extract dates from first experiment
    ref_dates = _format_date_strings(datasets[0].init_date.values)
    ref_dates_set = set(ref_dates)

    # Validate all subsequent experiments have EXACTLY the same dates
    for exp_idx, ds in enumerate(datasets[1:], 1):
        curr_dates = _format_date_strings(ds.init_date.values)
        curr_dates_set = set(curr_dates)

        # Check for missing dates
        missing_dates = ref_dates_set - curr_dates_set
        if missing_dates:
            missing_strs = sorted(list(missing_dates))[:10]
            raise ValueError(
                f"Experiment {exp_idx} is missing dates present in exp0:\n"
                f"  Missing: {missing_strs}{'...' if len(missing_dates) > 10 else ''}\n"
                f"  Total missing: {len(missing_dates)}/{len(ref_dates)}\n"
                f"Global data requires identical dates across all experiments."
            )

        # Check for extra dates
        extra_dates = curr_dates_set - ref_dates_set
        if extra_dates:
            extra_strs = sorted(list(extra_dates))[:10]
            raise ValueError(
                f"Experiment {exp_idx} has extra dates not in exp0:\n"
                f"  Extra: {extra_strs}{'...' if len(extra_dates) > 10 else ''}\n"
                f"  Total extra: {len(extra_dates)}\n"
                f"Global data requires identical dates across all experiments."
            )

        # Check order is the same
        if curr_dates != ref_dates:
            raise ValueError(
                f"Experiment {exp_idx} has dates in different order than exp0.\n"
                f"Global data requires identical date order across all experiments."
            )

    return ref_dates


def _load_global_avg_variables(
    ds,
    coll,
    var_list,
    stat_list,
    lead_indices,
    levels,
    lats,
    lons,
    is_3d,
    verbose=False,
):
    """
    Load time-averaged variables (VAR_STAT_avg) from global dataset.

    Parameters:
    - ds: xarray Dataset
    - coll: Collection name
    - var_list: List of variable names
    - stat_list: List of statistic names
    - lead_indices: Indices of common leads in this dataset
    - levels: List of pressure levels
    - lats, lons: Coordinate arrays
    - is_3d: Boolean indicating if collection is 3D

    Returns:
    - avg_array: numpy array with shape:
        3D: [n_leads, n_stats, n_vars, n_levels, n_lats, n_lons]
        2D: [n_leads, n_stats, n_vars, n_lats, n_lons]
    """
    n_leads = len(lead_indices)
    n_stats = len(stat_list)
    n_vars = len(var_list)
    n_lats = len(lats)
    n_lons = len(lons)

    # Create array
    if is_3d:
        n_levels = len(levels)
        avg_array = np.full(
            (n_leads, n_stats, n_vars, n_levels, n_lats, n_lons),
            np.nan,
            dtype=np.float32,
        )
    else:
        avg_array = np.full(
            (n_leads, n_stats, n_vars, n_lats, n_lons),
            np.nan,
            dtype=np.float32,
        )

    # Fill array
    for var_idx, var in enumerate(var_list):
        for stat_idx, stat in enumerate(stat_list):
            var_name = f"{var}_{stat}_avg"

            if var_name not in ds:
                if verbose:
                    print(f"        WARNING: {var_name} not found")
                continue

            # Load data and apply lead filtering
            data = ds[var_name].values

            if is_3d:
                # Expected shape: (lead, lev, lat, lon)
                # Filter to common leads
                avg_array[:, stat_idx, var_idx, :, :, :] = data[
                    lead_indices, :, :, :
                ]
            else:
                # Expected shape: (lead, lat, lon)
                # Filter to common leads
                avg_array[:, stat_idx, var_idx, :, :] = data[
                    lead_indices, :, :
                ]

    return avg_array


def _load_global_glo_variables(
    ds, coll, var_list, stat_list, lead_indices, levels, is_3d, verbose=False
):
    """
    Load spatially-averaged variables (VAR_STAT_glo) from global dataset.

    Parameters:
    - ds: xarray Dataset
    - coll: Collection name
    - var_list: List of variable names
    - stat_list: List of statistic names
    - lead_indices: Indices of common leads in this dataset
    - levels: List of pressure levels
    - is_3d: Boolean indicating if collection is 3D

    Returns:
    - glo_array: numpy array with shape:
        3D: [n_leads, n_stats, n_vars, n_levels]
        2D: [n_leads, n_stats, n_vars]
    """
    n_leads = len(lead_indices)
    n_stats = len(stat_list)
    n_vars = len(var_list)

    # Create array
    if is_3d:
        n_levels = len(levels)
        glo_array = np.full(
            (n_leads, n_stats, n_vars, n_levels), np.nan, dtype=np.float32
        )
    else:
        glo_array = np.full(
            (n_leads, n_stats, n_vars), np.nan, dtype=np.float32
        )

    # Fill array
    for var_idx, var in enumerate(var_list):
        for stat_idx, stat in enumerate(stat_list):
            var_name = f"{var}_{stat}_glo"

            if var_name not in ds:
                if verbose:
                    print(f"        WARNING: {var_name} not found")
                continue

            # Load data and apply lead filtering
            data = ds[var_name].values

            if is_3d:
                # Expected shape: (lead, lev)
                # Filter to common leads
                glo_array[:, stat_idx, var_idx, :] = data[lead_indices, :]
            else:
                # Expected shape: (lead,)
                # Filter to common leads
                glo_array[:, stat_idx, var_idx] = data[lead_indices]

    return glo_array


def _load_global_raw_variables(
    ds,
    coll,
    var_list,
    stat_list,
    lead_indices,
    levels,
    lats,
    lons,
    is_3d,
    verbose=False,
):
    """
    Load raw variables (VAR_STAT) from global dataset.

    Parameters:
    - ds: xarray Dataset
    - coll: Collection name
    - var_list: List of variable names
    - stat_list: List of statistic names
    - lead_indices: Indices of common leads in this dataset
    - levels: List of pressure levels
    - lats, lons: Coordinate arrays
    - is_3d: Boolean indicating if collection is 3D

    Returns:
    - raw_array: numpy array with shape:
        3D: [n_init_dates, n_leads, n_stats, n_vars, n_levels, n_lats, n_lons]
        2D: [n_init_dates, n_leads, n_stats, n_vars, n_lats, n_lons]
    """
    n_init_dates = len(ds.init_date.values)
    n_leads = len(lead_indices)
    n_stats = len(stat_list)
    n_vars = len(var_list)
    n_lats = len(lats)
    n_lons = len(lons)

    # Create array
    if is_3d:
        n_levels = len(levels)
        raw_array = np.full(
            (n_init_dates, n_leads, n_stats, n_vars, n_levels, n_lats, n_lons),
            np.nan,
            dtype=np.float32,
        )
    else:
        raw_array = np.full(
            (n_init_dates, n_leads, n_stats, n_vars, n_lats, n_lons),
            np.nan,
            dtype=np.float32,
        )

    # Fill array
    for var_idx, var in enumerate(var_list):
        for stat_idx, stat in enumerate(stat_list):
            var_name = f"{var}_{stat}"

            if var_name not in ds:
                if verbose:
                    print(f"        WARNING: {var_name} not found")
                continue

            # Load data and apply lead filtering
            data = ds[var_name].values

            if is_3d:
                # Expected shape: (init_date, lead, lev, lat, lon)
                # Filter to common leads (no date filtering for global)
                raw_array[:, :, stat_idx, var_idx, :, :, :] = data[
                    :, lead_indices, :, :, :
                ]
            else:
                # Expected shape: (init_date, lead, lat, lon)
                # Filter to common leads
                raw_array[:, :, stat_idx, var_idx, :, :] = data[
                    :, lead_indices, :, :
                ]

    return raw_array


# ============================================================================
# DATA LOADING DRIVER FUNCTIONS
# ============================================================================


def load_regional_stats_data(
    stats_nc_filenames: Union[str, List[str], List[List[str]]],
    plot_RMS_decomp: bool = True,
    verbose: bool = True,
    # Optional metadata for plotting
    season: str = "",
    year: int = 0,
    plot_dpi: int = 300,
    out_suffix: str = "",
    vars_to_fetch: List[str] = None,
) -> Dict[str, Any]:
    """
    Load regional statistics data from NetCDF file(s).

    Parameters:
    - stats_nc_filenames: Single file, list of files, or list of lists
    - plot_RMS_decomp: If True, includes decomposed RMS stats
    - verbose: If True, print detailed loading progress
    - season, year, plot_dpi, out_suffix: Optional metadata for plotting

    Returns dictionary containing statistics data and configuration.
    """
    # DEBUG: print stats filenames
    print("Starting loading of regional stat data from netcdf...")
    if isinstance(stats_nc_filenames, list):
        print(f"Found {len(stats_nc_filenames)} experiment files")
        print(stats_nc_filenames)
    else:
        print(f"Loading from single experiment file.")
        print(stats_nc_filenames)

    # Define collections
    collections = list(COLLECTION_VARIABLES.keys())

    # ========================================================================
    # STEP 1: Normalize input to list-of-lists format
    # ========================================================================
    exp_file_lists = _normalize_filename_input(stats_nc_filenames)
    n_experiments = len(exp_file_lists)

    if verbose:
        print(f"Loading {n_experiments} experiment(s)...")

    # ========================================================================
    # STEP 2: Merge files within each experiment
    # ========================================================================
    merged_datasets = []
    exp_excluded_dates = []
    exp_valid_dates = []

    for exp_idx, file_list in enumerate(exp_file_lists):
        merged_ds, excluded_dates, n_excl = merge_experiment_files(
            file_list, f"exp{exp_idx}"
        )
        merged_datasets.append(merged_ds)
        exp_excluded_dates.append(excluded_dates)

        date_strs = _format_date_strings(merged_ds.init_date.values)
        exp_valid_dates.append(date_strs)

    # ========================================================================
    # STEP 2.1: Extract model names
    # ========================================================================
    model_names_dict = _compile_model_names(
        [file_list[0] for file_list in exp_file_lists]
    )
    fcst_names = model_names_dict["fcst_names"]
    ana_names = model_names_dict["ana_names"]
    clim_names = model_names_dict["clim_names"]

    if verbose:
        for exp_idx, (fcst, ana, clim) in enumerate(
            zip(fcst_names, ana_names, clim_names)
        ):  # ✅ Fixed
            print(f"  Experiment {exp_idx}: {fcst}(F) / {ana}(A) / {clim}(C)")

    # ========================================================================
    # STEP 3: Find common dates
    # ========================================================================
    if verbose:
        print("\nFinding common dates across experiments...")

    common_dates, date_indices_per_exp = _find_common_dates_regional(
        merged_datasets
    )

    if len(common_dates) == 0:
        for ds in merged_datasets:
            ds.close()
        raise ValueError("No common dates found across all experiments!")

    if verbose:
        print(
            f"  Common dates: {len(common_dates)} "
            f"({common_dates[0]} to {common_dates[-1]})"
        )

    # ========================================================================
    # STEP 4: Find common variables/stats
    # ========================================================================
    if verbose:
        print("\n=== Finding Common Variables ===")

    common_vars, common_stats = _find_common_variables(
        merged_datasets, collections
    )

    if verbose:
        print(f"  Before filtering, common vars:")
        for coll in collections:
            if common_vars[coll]:
                print(f"  {coll}: {common_vars[coll]}")
        print(f"  Stats: {common_stats}")

    # Filter to only requested stats
    # common_stats = [s for s in plot_stats if s in common_stats]

    # Filter to only requested variables
    if vars_to_fetch:
        for coll in collections:
            # For each collection, filter the common vars
            # This will create some empty collections but this is fine!
            common_vars[coll] = [
                var for var in common_vars[coll] if var in vars_to_fetch
            ]

    if verbose:
        print(f"  After filtering, common vars:")
        for coll in collections:
            if common_vars[coll]:
                print(f"  {coll}: {common_vars[coll]}")
        print(f"  Stats: {common_stats}")

    # ========================================================================
    # STEP 5: Validate coordinates match
    # ========================================================================
    # Levels must be identical (strict validation)
    levels = _validate_identical_levels(merged_datasets)

    # Leads can differ - find intersection
    leads, lead_indices_per_exp = _find_common_leads(
        merged_datasets, verbose=verbose
    )
    regions = _validate_regional_spatial_coords(merged_datasets)
    # # ✅ Extract leads and levels here
    # leads = [int(x) for x in common_coords["lead"]]
    # levels = [int(x) for x in common_coords["lev"]]

    # ========================================================================
    # STEP 6: Load data into arrays
    # ========================================================================
    data = {"raw": {}, "avg": {}}
    for coll in collections:
        data["raw"][coll] = {}

    for exp_idx, ds in enumerate(merged_datasets):
        exp_data = _load_experiment_data(
            ds,
            collections,
            common_vars,
            common_stats,
            regions,
            COLLECTION_DIMENSIONALITY,
            levels,
            date_indices_per_exp[exp_idx],
            lead_indices_per_exp[exp_idx],
        )

        for coll in collections:
            data["raw"][coll][exp_idx] = exp_data[coll]

    # ========================================================================
    # STEP 7: Filter arrays if needed
    # ========================================================================
    data, final_common_dates = _filter_arrays_to_common_dates(
        data, collections, regions, exp_valid_dates, common_dates
    )

    # ========================================================================
    # STEP 8: Compute averages
    # ========================================================================
    for coll in collections:
        data["avg"][coll] = {}
        for exp_idx in range(n_experiments):  # ✅ Fixed
            data["avg"][coll][exp_idx] = {}
            for region in regions:
                raw_array = data["raw"][coll][exp_idx][region]
                data["avg"][coll][exp_idx][region] = np.mean(raw_array, axis=0)

    # ========================================================================
    # STEP 8.5: Extract timing parameters (BEFORE closing datasets!)
    # ========================================================================
    timing_params_list = []
    for exp_idx, file_list in enumerate(exp_file_lists):
        timing_params = _extract_timing_from_filename(
            file_list[0], leads  # ✅ leads is now defined
        )
        timing_params_list.append(timing_params)

    ref_timing = _validate_timing_compatibility(
        timing_params_list, mode="compatible", verbose=verbose
    )
    fcst_length = ref_timing.get("fcst_length")
    fcst_interval = ref_timing.get("fcst_interval")

    # ========================================================================
    # STEP 9: Close datasets
    # ========================================================================
    for ds in merged_datasets:
        ds.close()

    # ========================================================================
    # RETURN: All variables now properly defined
    # ========================================================================
    return {
        "raw": data["raw"],  # Whole point of the function!
        "avg": data["avg"],  # Whole point of the function!
        "date_nms": final_common_dates,  # From Step 7
        "leads": leads,  # From Step 5
        "fcst_names": fcst_names,  # From Step 2.1
        "ana_names": ana_names,  # From Step 2.1
        "clim_names": clim_names,  # From Step 2.1
        "fcst_length": fcst_length,  # From Step 8.5
        "fcst_interval": fcst_interval,  # From Step 8.5
        "fvars": common_vars,  # From Step 4
        "levels": levels,  # From Step 5
        "regions": regions,  # From Step 5
        "collections": collections,  # From top of function
        "is_3d": COLLECTION_DIMENSIONALITY,  # ✅ Constant
        "plot_stats": common_stats,  # From Step 4 (or use plot_stats if you prefer)
        "season": season,  # Parameter
        "year": year,  # Parameter
        "plot_dpi": plot_dpi,  # Parameter
        "out_suffix": out_suffix,  # Parameter
        "plot_RMS_decomp": plot_RMS_decomp,  # Parameter
    }


def load_global_stats_data(
    stats_nc_filenames: Union[str, List[str], List[List[str]]],
    process: str = "indiv",
    load_raw: bool = False,
    verbose: bool = True,
    # Optional metadata (no longer from YAML)
    season: str = "",
    year: int = 0,
    plot_dpi: int = 300,
    out_suffix: str = "",
    vars_to_fetch: List[str] = None,
) -> Dict[str, Any]:
    """
    Load global statistics data from NetCDF file(s).

    Parameters:
    - stats_nc_filenames: Single file, list of files, or list of lists
        - Single file: "stats_global_model.nc4"
        - Single exp (multi-file): ["file1.nc4", "file2.nc4"]
        - Multi exp: [["exp0_file1.nc4"], ["exp1_file1.nc4", "exp1_file2.nc4"]]
    - process: 'indiv' (individual plots) or 'comp' (comparison plots)
               Determines which statistics are loaded
    - load_raw: If True, loads raw (per-forecast) data in addition to avg/glo
                Only meaningful for 'comp' process
    - verbose: If True, print detailed loading progress
    - season, year, plot_dpi, out_suffix: Optional metadata for plotting

    Returns dictionary containing statistics data and configuration:
        - 'avg': Time-averaged data [coll][exp] -> array[lead, stat, var, lev?, lat, lon]
        - 'glo': Spatially-averaged data [coll][exp] -> array[lead, stat, var, lev?]
        - 'raw': Optional raw data [coll][exp] -> array[init_date, lead, stat, var, lev?, lat, lon]
        - Plus metadata (dates, leads, model names, etc.)
    """

    # Define statistics based on process type
    if process == "indiv":
        plot_stats = [
            "f",
            "f_c",
            "me",
            "acorr",
            "rms",
            "rms_bar",
            "rms_amp",
            "rms_phz",
        ]
    elif process == "comp":
        plot_stats = ["f", "acorr", "rms", "rms_bar", "rms_amp", "rms_phz"]
    else:
        raise ValueError(f"process must be 'indiv' or 'comp', got: {process}")

    # Define collections
    collections = list(COLLECTION_VARIABLES.keys())

    # ========================================================================
    # STEP 1: Normalize input to list-of-lists format and merge
    # ========================================================================
    exp_file_lists = _normalize_filename_input(stats_nc_filenames)
    n_experiments = len(exp_file_lists)

    if verbose:
        print(
            f"Loading {n_experiments} experiment(s) for {process} process..."
        )

    # ========================================================================
    # STEP 2: Merge files within each experiment
    # ========================================================================
    merged_datasets = []
    exp_excluded_dates = []
    exp_valid_dates = []

    for exp_idx, file_list in enumerate(exp_file_lists):
        if verbose:
            print(f"\n--- Experiment {exp_idx} ---")

        # Merge files for this experiment (global mode)
        merged_ds, excluded_dates, n_excl = merge_experiment_files(
            file_list, f"exp{exp_idx}", merge_mode="global"
        )
        merged_datasets.append(merged_ds)
        exp_excluded_dates.append(excluded_dates)

        # Track what dates this experiment has after merging
        date_strs = _format_date_strings(merged_ds.init_date.values)
        exp_valid_dates.append(date_strs)

    # ========================================================================
    # STEP 2.1: Extract and print model names
    # ========================================================================
    model_names_dict = _compile_model_names(
        [file_list[0] for file_list in exp_file_lists]
    )
    fcst_names = model_names_dict["fcst_names"]
    ana_names = model_names_dict["ana_names"]
    clim_names = model_names_dict["clim_names"]

    if verbose:
        print("\n=== Experiment Summary ===")
        for exp_idx, (fcst, ana, clim) in enumerate(
            zip(fcst_names, ana_names, clim_names)
        ):
            print(f"  exp{exp_idx}: {fcst}(F) / {ana}(A) / {clim}(C)")

    # ========================================================================
    # STEP 3: Validate dates are IDENTICAL (strict for global)
    # ========================================================================
    if verbose:
        print("\n=== Validating Dates (Strict) ===")

    try:
        date_nms = _validate_identical_dates_global(merged_datasets)
    except ValueError as e:
        for ds in merged_datasets:
            ds.close()
        raise ValueError(f"Date validation failed: {e}")

    if verbose:
        print(
            f"  All experiments have identical {len(date_nms)} dates "
            f"({date_nms[0]} to {date_nms[-1]}) ✓"
        )

    # ========================================================================
    # STEP 4: Find common variables and stats (intersection)
    # ========================================================================
    if verbose:
        print("\n=== Finding Common Variables ===")

    common_vars, common_stats = _find_common_variables(
        merged_datasets, collections
    )

    if verbose:
        print(f"  Before filtering, common vars:")
        for coll in collections:
            if common_vars[coll]:
                print(f"  {coll}: {common_vars[coll]}")
        print(f"  Stats: {common_stats}")

    # Filter to only requested stats
    common_stats = [s for s in plot_stats if s in common_stats]

    # Filter to only requested variables
    if vars_to_fetch:
        for coll in collections:
            # For each collection, filter the common vars
            # This will create some empty collections but this is fine!
            common_vars[coll] = [
                var for var in common_vars[coll] if var in vars_to_fetch
            ]

    if verbose:
        print(f"  After filtering, common vars:")
        for coll in collections:
            if common_vars[coll]:
                print(f"  {coll}: {common_vars[coll]}")
        print(f"  Stats: {common_stats}")

    # ========================================================================
    # STEP 5: Find common leads (intersection) and validate other coords
    # ========================================================================
    if verbose:
        print("\n=== Validating Coordinates ===")

    # Levels must be identical (strict)
    levels = _validate_identical_levels(merged_datasets)

    # Leads can differ - find intersection
    leads, lead_indices_per_exp = _find_common_leads(
        merged_datasets, verbose=verbose
    )

    # Spatial coordinates must be identical (strict)
    try:
        spatial_coords = _validate_global_spatial_coords(merged_datasets)
        lats = spatial_coords["lat"]
        lons = spatial_coords["lon"]
    except ValueError as e:
        for ds in merged_datasets:
            ds.close()
        raise ValueError(f"Spatial coordinate validation failed: {e}")

    if verbose:
        print(f"  Levels: {levels}")
        print(f"  Grid: {len(lats)} lats x {len(lons)} lons")

    # ========================================================================
    # STEP 6: Extract timing parameters
    # ========================================================================
    timing_params_list = []
    for file_list in exp_file_lists:
        timing_params = _extract_timing_from_filename(file_list[0], leads)
        timing_params_list.append(timing_params)

    # For global, validate timing is identical (strict)
    try:
        ref_timing = _validate_timing_compatibility(
            timing_params_list, mode="identical", verbose=verbose
        )
        fcst_length = ref_timing.get("fcst_length")
        fcst_interval = ref_timing.get("fcst_interval")
    except ValueError as e:
        # If strict fails, fall back to compatible mode
        if verbose:
            print(
                "  WARNING: Experiments have different timing parameters, "
                "using compatible mode"
            )
        ref_timing = _validate_timing_compatibility(
            timing_params_list, mode="compatible", verbose=verbose
        )
        fcst_length = ref_timing.get("fcst_length")
        fcst_interval = ref_timing.get("fcst_interval")

    # ========================================================================
    # STEP 7: Initialize data structures
    # ========================================================================
    data = {"avg": {}, "glo": {}}
    if load_raw and process == "comp":
        data["raw"] = {}

    for coll in collections:
        data["avg"][coll] = {}
        data["glo"][coll] = {}
        if load_raw and process == "comp":
            data["raw"][coll] = {}

    # ========================================================================
    # STEP 8: Load data for each experiment
    # ========================================================================
    if verbose:
        print("\n=== Loading Data ===")

    n_init_dates = len(date_nms)
    n_leads = len(leads)
    n_lats = len(lats)
    n_lons = len(lons)

    for exp_idx, ds in enumerate(merged_datasets):
        if verbose:
            print(f"  Loading exp{exp_idx}...")

        lead_indices = lead_indices_per_exp[exp_idx]

        for coll in collections:
            if not common_vars[coll]:
                continue  # Skip empty collections

            if verbose:
                print(f"    Collection {coll}: {common_vars[coll]}")

            # Load _avg variables (time-averaged)
            avg_array = _load_global_avg_variables(
                ds,
                coll,
                common_vars[coll],
                common_stats,
                lead_indices,
                levels,
                lats,
                lons,
                COLLECTION_DIMENSIONALITY[coll],
                verbose=verbose,
            )
            data["avg"][coll][exp_idx] = avg_array

            # Load _glo variables (spatially-averaged)
            glo_array = _load_global_glo_variables(
                ds,
                coll,
                common_vars[coll],
                common_stats,
                lead_indices,
                levels,
                COLLECTION_DIMENSIONALITY[coll],
                verbose=verbose,
            )
            data["glo"][coll][exp_idx] = glo_array

            # Optionally load raw data
            if load_raw and process == "comp":
                raw_array = _load_global_raw_variables(
                    ds,
                    coll,
                    common_vars[coll],
                    common_stats,
                    lead_indices,
                    levels,
                    lats,
                    lons,
                    COLLECTION_DIMENSIONALITY[coll],
                    verbose=verbose,
                )
                data["raw"][coll][exp_idx] = raw_array

            # Print shapes
            if verbose:
                print(f"      glo shape: {glo_array.shape}")
                print(f"      avg shape: {avg_array.shape}")
                if load_raw and process == "comp":
                    print(f"      raw shape: {raw_array.shape}")

    # ========================================================================
    # STEP 9: Close datasets
    # ========================================================================
    for ds in merged_datasets:
        ds.close()

    if verbose:
        print(
            f"\n✓ Loading complete: {n_experiments} experiments, "
            f"{len(date_nms)} dates, {process} process"
        )

    # ========================================================================
    # RETURN: Data dictionary
    # ========================================================================
    return {
        "avg": data["avg"],
        "glo": data["glo"],
        "raw": data.get("raw", {}),
        "date_nms": date_nms,
        "leads": leads,
        "fcst_names": fcst_names,
        "ana_names": ana_names,
        "clim_names": clim_names,
        "fcst_length": fcst_length,
        "fcst_interval": fcst_interval,
        "fvars": common_vars,
        "levels": levels,
        "lats": lats,
        "lons": lons,
        "collections": collections,
        "is_3d": COLLECTION_DIMENSIONALITY,
        "plot_stats": common_stats,
        "season": season,
        "year": year,
        "plot_dpi": plot_dpi,
        "out_suffix": out_suffix,
        "process": process,
    }
