# Tool: run_prithvi_inference

## Purpose
Run synchronous Prithvi-EO-2.0 inference for a given AOI and date(s). The task type is fixed by the server's `USECASE` environment variable (e.g. `flood`).

## When it should be used
- After feasibility is confirmed, AOI and date(s) are confirmed by the user, and HLS availability/quality checks pass.

## Inputs
- `bounding_box` (list[float], length=4, required): `[west, south, east, north]`
- `date` (string, required for flood/burn tasks): `YYYY-MM-DD`
- Crop-specific (required for crop tasks):
  - `date_range` (object):
    - `start_date` (string): `YYYY-MM-DD`
    - `end_date` (string): `YYYY-MM-DD`
  - `dates` (list[string], length=3): three selected dates (`YYYY-MM-DD`), each ≥70 days apart

## Outputs (on success)
- `<usecase>` (object, e.g. `flood`): result keyed by the server's usecase name, containing:
  - `s3_link` (string): S3 path to the output GeoTIFF
  - `predictions` (object): GeoJSON FeatureCollection of detected features

## Outputs (on failure)
- `status` (string): `failed`
- `message` (string): error description

## Security / permissions
- Runs on the GPU server environment; Earthdata access handled server-side (credential model TBD).

## Expected failure modes
- Invalid inputs (bbox, dates)
- Inference server unreachable or timeout
- Runtime environment missing dependencies
