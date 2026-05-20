# Context: Prithvi task capability & input/output requirements

## What this context is used for
Reusable reference for:
- Validating whether a user request is in-scope.
- Knowing required inputs per supported task.
- Knowing expected outputs and class legend.

## When this becomes relevant
- On every new user query (scope/feasibility check)

## Authority
- Authoritative (SME-confirmed)

## Supported tasks (3 only)
- **Flood detection**: binary segmentation (flood / not-flood)
- **Burn scar detection**: binary segmentation (burned / not-burned)
- **Crop classification**: multi-class segmentation (13 classes)

## Required inputs per task
- **Flood**: bounding box + exactly 1 date (`YYYY-MM-DD`)
- **Burn**: bounding box + exactly 1 date (`YYYY-MM-DD`)
- **Crop**: bounding box + date range + exactly 3 dates (within the range), each with:
  - ≥70-day gaps between the dates
  - ≥70% clear pixels each

## Common inputs (all tasks)
- HLS 6-band imagery at 30m:
  - Blue, Green, Red, NIR, SWIR-1, SWIR-2

## Outputs per task
- **Flood**:
  - GeoTIFF binary mask (1=flood, 0=not flood)
  - Flood area (hectares)
- **Burn**:
  - Multiple per-tile GeoTIFF binary masks (tile_id → GeoTIFF)
  - Per-tile burn percentages
- **Crop**:
  - GeoTIFF 13-class map
  - Area per class

## Crop class legend (13)
1. Natural Vegetation
2. Forest
3. Corn
4. Soybeans
5. Wetlands
6. Developed/Barren
7. Open Water
8. Winter Wheat
9. Alfalfa
10. Fallow/Idle
11. Cotton
12. Sorghum
13. Other

## Explicitly not supported (reject immediately)
- drought
- deforestation
- soil moisture
- urban change
- snow/ice
- ocean color
- atmospheric composition
- building damage
- landslide mapping
