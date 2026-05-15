from datetime import datetime
from glob import glob
import os
import boto3
import fnmatch

def create_base_patterns(forecast_model):
    """
    Create 4 base patterns for forecast model comparison.

    Returns a list of 4 patterns:
        1. Baseline (GEOSFP)
        2. Input data baseline (ERA5/MERRA2)
        3. Forecast model vs baseline (GEOSFP)
        4. Forecast model vs input (ERA5/MERRA2)

    Args:
        forecast_model: Name of the forecast model ("Prithvi" or other)

    Returns:
        list: Four base pattern strings
    """
    # Baseline pattern using GEOSFP and MERRA2
    exp0_base_pattern = "f5295fp_GEOSFP_MERRA2"

    # Determine input baseline based on forecast model
    input_baseline = "MERRA2" if forecast_model == "Prithvi" else "ERA5"

    # Input baseline pattern (e.g., "ERA5_ERA5_ERA5" or "MERRA2_MERRA2_MERRA2")
    exp1_base_pattern = "_".join([input_baseline] * 3)

    # Forecast model vs GEOSFP baseline
    exp2_base_pattern = f"{forecast_model}_GEOSFP_MERRA2"

    # Forecast model vs input baseline
    exp3_base_pattern = f"{forecast_model}_{input_baseline}_{input_baseline}"

    return [
        exp0_base_pattern,
        exp1_base_pattern,
        exp2_base_pattern,
        exp3_base_pattern
    ]

def get_stat_dataset_filenames(base_glob_patterns, month):
    filename_dict = {"regional": [], "global": []}
    
    S3_BUCKET = "enw-04241552-kx1nks-shared"
    BASE_PREFIX = "data/stats"
    
    # Direct construction
    bucket_dir = f"{month.lower()}_2024"
    prefix = f"{BASE_PREFIX}/{bucket_dir}/"

    # Initialize S3 client
    s3_client = boto3.client('s3')
    
    # List all objects in the directory
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
    
    if 'Contents' not in response:
        raise ValueError(f"No files found in s3://{S3_BUCKET}/{prefix}")
    
    # Get all filenames
    all_files = [obj['Key'] for obj in response['Contents']]

    # Build patterns and filter
    for base_pattern in base_glob_patterns:
        pattern = f"stats*{base_pattern}*.nc4"
        
        # Filter files matching the pattern
        stat_filenames = [
            f"s3://{S3_BUCKET}/{f}" for f in all_files 
            if fnmatch.fnmatch(f.split('/')[-1], pattern)
        ]

        # Find regional and global files
        regional_files = [f for f in stat_filenames if "regional" in f]
        global_files = [f for f in stat_filenames if "global" in f]

        # Validate both exist
        if not regional_files or not global_files:
            missing = "regional" if not regional_files else "global"
            raise ValueError(f"No {missing} filename found for pattern: {pattern}")

        filename_dict["regional"].append(regional_files)
        filename_dict["global"].append(global_files)

    return filename_dict