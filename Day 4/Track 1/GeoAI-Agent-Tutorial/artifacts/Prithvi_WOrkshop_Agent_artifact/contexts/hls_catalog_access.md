# Existing system: HLS catalog search and access

## Tool / API inventory
- **Name**: NASA HLS catalog search (HLSS30 + HLSL30)
- **Owner**: TBD
- **Purpose**: Verify imagery availability for a given AOI and date/range; estimate/assess cloud cover.
- **When currently used**: Not currently implemented (access tools to be built).
- **Access method**: TBD (e.g., STAC, CMR, LP DAAC search, direct tile listing).

## Inputs / outputs
### Inputs
- AOI bbox: `[west, south, east, north]`
- Date or date range
- Product types: `HLSS30`, `HLSL30`
- Task-specific constraints:
  - Flood/burn: allow nearest date within ±3 days if exact missing
  - Crop: require 3 clean dates with ≥70-day gaps and ≥70% clear pixels

### Outputs
- Candidate HLS scenes/tiles and acquisition dates
- Cloud/clear metrics (exact definition TBD)
- Notes on partial coverage / tiling behavior (TBD)

## Schemas, access patterns, and documentation
- Endpoint(s) and query syntax: TBD
- Response schema (scene vs tile granularity): TBD
- Cloud metric source: TBD (e.g., derived from Fmask QA or precomputed metadata)

## Permissions, limits, and operational constraints
- **Credentials**: NASA Earthdata credentials are required for authenticated download (see `contexts/earthdata_access.md`).
- Quotas/rate limits: TBD

## Known error patterns / failure modes
- No overpass on the exact date (common for floods)
- Imagery too cloudy to satisfy crop screening
- Backend/catalog outages or throttling (TBD specifics)

## Open questions / unknowns
- Which catalog/search interface will be used (STAC/CMR/etc.)
- How “better AOI coverage” is measured
- Spatial granularity (scene vs tile) and how mosaicking is handled
- How clear% is computed and returned
