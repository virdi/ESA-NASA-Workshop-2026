# Tool: query_crop_landcover

## Purpose
Query the USDA Cropland Data Layer (CDL) to summarize dominant crop/landcover classes within a bounding box.

## When it should be used
- When the user request is about crop classification, or when the agent needs to gauge whether an AOI is agricultural.

## Inputs
- `bbox` (list[float], length=4, required): `[west, south, east, north]`
- `year` (integer, optional): CDL year to query

## Outputs (minimal)
- Summary distribution (e.g., top classes and their area fractions; exact schema TBD)
- Optional: links/IDs to underlying products
- `message` (string)

## Validation & business rules
- Validate bbox ordering.

## Expected failure modes
- No coverage for AOI/year
- Dataset/API unavailable
