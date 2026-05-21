# Tool: get_prithvi_results

## Purpose
Retrieve outputs for a completed Prithvi inference job.

## When it should be used
- When `get_prithvi_job_status` indicates `finished`.

## Inputs
- `job_id` (string, required)

## Outputs (minimal)
- `task_type` (string): `flood | burn | crop`
- `result_urls` (list[string]): URLs to output files (GeoTIFFs, CSVs, figures, etc.)
- `result_tiles` (object, burn only): mapping `tile_id` → GeoTIFF URL/path
- `summary` (object): area statistics
  - Flood: `area_hectares` (number)
  - Burn: `per_tile_burn_pct` (object): mapping `tile_id` → percent burned
  - Crop: `area_hectares` (number) + `per_class_hectares` (object)
- `message` (string, optional)

## Expected failure modes
- Results not ready
- Missing output files
- Permission/IO errors
