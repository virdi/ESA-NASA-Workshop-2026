# Tool: query_active_fires

## Purpose
Query FIRMS to detect recent/ongoing fire activity signals within a bounding box and date range.

## When it should be used
- When the user does not specify hazard type and the agent needs evidence of fire activity.

## Inputs
- `bbox` (list[float], length=4, required): `[west, south, east, north]`
- `start_date` (string, required): `YYYY-MM-DD`
- `end_date` (string, required): `YYYY-MM-DD`

## Outputs (minimal)
- Summary fields indicating whether detections exist, count, and detection date range (exact schema TBD)
- Optional: links/IDs to underlying records
- `message` (string)

## Validation & business rules
- Validate bbox ordering and date range.

## Expected failure modes
- No detections in range
- Dataset/API unavailable
- Rate limiting
