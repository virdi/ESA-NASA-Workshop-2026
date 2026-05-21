# Tool: query_surface_water

## Purpose
Query OPERA DSWx (Dynamic Surface Water Extent) to detect surface-water / potential flooding signals within a bounding box and date range.

## When it should be used
- When the user does not specify hazard type and the agent needs evidence of water/flooding.

## Inputs
- `bbox` (list[float], length=4, required): `[west, south, east, north]`
- `start_date` (string, required): `YYYY-MM-DD`
- `end_date` (string, required): `YYYY-MM-DD`

## Outputs (minimal)
- Signal summary indicating whether products exist, dates available, and any simple area metric if available (exact schema TBD)
- Optional: links/IDs to underlying products
- `message` (string)

## Validation & business rules
- Validate bbox ordering and date range.

## Expected failure modes
- No products for range/AOI
- Dataset/API unavailable
- Rate limiting
