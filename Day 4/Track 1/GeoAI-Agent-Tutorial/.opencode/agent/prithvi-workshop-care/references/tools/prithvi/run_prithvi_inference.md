# Tool: run_prithvi_inference

## Purpose
Submit an asynchronous Prithvi-EO-2.0 inference job for one of the supported tasks for a given AOI and date(s).

## When it should be used
- After feasibility is confirmed, AOI and date(s) are confirmed by the user, and HLS availability/quality checks pass.

## Inputs
- `task_type` (string, required): `flood | burn | crop`
- `bbox` (list[float], length=4, required): `[west, south, east, north]`
- `date` (string, required for `task_type` in `{flood,burn}`): `YYYY-MM-DD`
- Crop-specific (required for `task_type="crop"`):
  - `date_range` (object):
    - `start_date` (string): `YYYY-MM-DD`
    - `end_date` (string): `YYYY-MM-DD`
  - `dates` (list[string], length=3): three selected dates (`YYYY-MM-DD`), each ≥70 days apart

## Outputs (minimal)
- `job_id` (string)
- `status` (string): `submitted` on success
- `message` (string)

## Validation & business rules
- Enforce supported `task_type` values only.
- Validate required dates are present for the selected task type.

## Security / permissions
- Runs on the GPU server environment; Earthdata access handled server-side (credential model TBD).

## Expected failure modes
- Invalid inputs (bbox, dates)
- Job submission failure / scheduler errors
- Runtime environment missing dependencies
