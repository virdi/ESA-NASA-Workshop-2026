# Existing system: Prithvi-EO-2.0 inference runtime / tool interface

## Tool / API inventory
- **Name**: Prithvi-EO-2.0 inference (tool interface)
- **Owner**: TBD
- **Purpose**: Run inference for supported geospatial tasks:
  - flood detection
  - burn-scar detection
  - crop classification
- **When currently used**: Not currently implemented (tools to be built).
- **Access method**: Tool call into GPU server/HPC environment (details TBD).

## Inputs / outputs (as currently understood)
### Inputs
- AOI bounding box: `[west, south, east, north]`
- Task selector: `flood | burn | crop`
- Date inputs:
  - Flood/burn: single date
  - Crop: date range + 3 selected dates meeting temporal-gap and cloud constraints
- HLS source selection/band stacking/inference: expected to be handled server-side.

### Outputs
- Flood/burn: GeoTIFF binary mask(s)
- Crop: GeoTIFF 13-class crop map
- Summary statistics (e.g., area in hectares; crop breakdown; burn per-tile stats)

## Schemas, access patterns, and documentation
- Tool names, request/response JSON, error codes: TBD
- Sync vs async job control model: TBD
- Result delivery mechanism (URLs vs paths vs blobs): TBD
- CRS requirements and bbox ordering validation rules: TBD

## Permissions, limits, and operational constraints
- Authentication/authorization model: TBD
- Expected latency/quotas: TBD

## Known error patterns / failure modes
- Job submission failures / scheduler errors: TBD
- Runtime failures (dependency/data access): TBD
