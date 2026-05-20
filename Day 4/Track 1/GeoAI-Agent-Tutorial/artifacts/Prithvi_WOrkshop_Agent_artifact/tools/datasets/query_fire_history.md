# Tool: query_fire_history

## Purpose
Query MTBS to find historical fire perimeters intersecting a bounding box within a date range.

## When it should be used
- When the user does not specify hazard type and the agent needs evidence of burns.

## Inputs
- `bbox` (list[float], length=4, required): `[west, south, east, north]`
- `start_date` (string, required): `YYYY-MM-DD`
- `end_date` (string, required): `YYYY-MM-DD`

## Outputs (minimal)
- Intersections summary (e.g., number of perimeters, names/IDs, date(s); exact schema TBD)
- Optional: links/IDs to perimeter geometries
- `message` (string)

## Validation & business rules
- Validate bbox ordering and date range.

## Expected failure modes
- No intersecting perimeters
- Dataset/API unavailable
