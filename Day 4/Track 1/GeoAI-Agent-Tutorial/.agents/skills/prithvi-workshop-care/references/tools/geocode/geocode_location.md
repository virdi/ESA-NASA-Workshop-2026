# Tool: geocode_location

## Purpose
Convert a user-provided place description into a bounding box `[west, south, east, north]` suitable for downstream HLS search and inference.

## When it should be used
- When the user provides a place name / region description instead of an explicit bounding box.

## Inputs
- `query` (string, required): place name, event/location description, **or bbox coordinates** (as user-provided text).

## Outputs (minimal)
- `bbox` (list[float], length=4): `[west, south, east, north]` (present when a single best match exists)
- `display_name` (string): resolved location name for user confirmation (present when `bbox` present)
- `candidates` (list, optional): when ambiguous, return up to top 3 candidates, each with:
  - `display_name` (string)
  - `bbox` (list[float], length=4)
- `message` (string)

## Validation & business rules
- If **multiple matches** are found (ambiguous): return `candidates` and do not auto-select.
- If **zero matches**: return an error via `message` suggesting rephrase or provide coordinates.

## Expected failure modes
- Ambiguous place name
- No results
- Backend timeout / rate limit
