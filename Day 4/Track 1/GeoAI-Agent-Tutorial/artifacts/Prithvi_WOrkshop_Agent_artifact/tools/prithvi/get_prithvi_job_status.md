# Tool: get_prithvi_job_status

## Purpose
Check status of a previously submitted Prithvi inference job.

## When it should be used
- After `run_prithvi_inference` returns a `job_id`.

## Inputs
- `job_id` (string, required)

## Outputs (minimal)
- `status` (string): `running | finished | failed`
- `message` (string)

## Expected failure modes
- Unknown `job_id`
- Backend timeout
