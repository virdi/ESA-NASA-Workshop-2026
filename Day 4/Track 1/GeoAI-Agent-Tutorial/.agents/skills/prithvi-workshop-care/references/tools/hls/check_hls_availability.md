# Tool: check_hls_availability

## Purpose
Check whether usable HLS imagery exists for a given AOI and task-specific temporal/quality constraints, and return selected acquisition date(s) plus quality metrics.

## When it should be used
- After AOI bbox is finalized and date(s) need to be validated/selected, before running Prithvi inference.

## Inputs
- `bbox` (list[float], length=4, required): `[west, south, east, north]`
- `date` (string, required): target date in `YYYY-MM-DD`
- `task_type` (string, required): `flood | burn | crop`
- Crop-only (required when `task_type="crop"`):
  - `date_range` (object):
    - `start_date` (string): `YYYY-MM-DD`
    - `end_date` (string): `YYYY-MM-DD`

## Outputs (minimal)
- `available` (boolean)
- `selected_date` (string)
- `collection` (string): `HLSS30 | HLSL30`
- `clear_pct` (float)
- `offset_days` (integer)
- `crop_dates` (list[string], length=3, crop only)
- `alternatives` (list, optional; when `available=false`): each item includes:
  - `date` (string)
  - `clear_pct` (float)
  - `collection` (string)
  - `offset_days` (integer)
- `message` (string)

## Validation & business rules
- Validate bbox ordering and date formats.
- Flood:
  - Search exact date first; if missing, fallback within ±3 days.
  - Choose imagery with best `clear_pct`.
- Burn:
  - Search from requested date through +30 days for clear post-fire imagery.
- Crop:
  - Find 3 clean dates within the date range with ≥70-day gaps and ≥70% clear pixels.
  - If strict fails, relax to ≥50-day gaps and ≥50% clear pixels.

## Expected failure modes
- No overpass / no imagery meeting constraints
- Imagery too cloudy
- Catalog/query backend errors (timeouts, throttling, auth)
