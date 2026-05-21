# Contexts manifest

This directory contains reference documents used to describe the operational environment, external dependencies, and reusable reference context.

## Existing systems and prerequisites
- `hls_catalog_access.md` — HLS search and access expectations (HLSS30/HLSL30), inputs/outputs, and unknown implementation details.
- `earthdata_access.md` — Earthdata Login / LP DAAC authentication prerequisites (e.g., `.netrc` for `urs.earthdata.nasa.gov`).
- `event_catalogs.md` — Candidate event catalogs for date resolution (NOAA Storm Events, MTBS, FIRMS), with open endpoint/schema questions.
- `compute_and_storage.md` — Runtime assumptions for GPU compute and server-side GeoTIFF storage.
- `prithvi_inference_mcp.md` — Prithvi inference runtime/tool-interface placeholder: intended inputs/outputs and key unknowns.

## Reusable reference context
- `prithvi_task_capabilities.md` — What Prithvi tasks are available and what inputs/outputs they require.
- `hls_conventions.md` — Acceptable HLS products, clear-pixel definition, and tiling/mosaicking conventions.
- `user_confirmations.md` — What must be confirmed before inference and what degradations must be disclosed.
