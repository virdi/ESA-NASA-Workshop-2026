# Existing system: Event catalogs (for date resolution)

## Tool / API inventory
- **Name**: Event catalogs / databases for hazard date resolution
- **Owner**: Multiple (TBD)
- **Purpose**: Infer likely event dates when a user provides vague time descriptions.
- **When currently used**: Not currently implemented (query tools to be built).
- **Access method**: TBD

## Candidate flood-related sources
- **NOAA Storm Events**
  - Purpose: identify flood event dates near a location.
  - Interface/auth/limits: TBD

## Candidate burn-related sources
- **MTBS**
  - Purpose: identify burn events/perimeters and dates.
  - Interface/auth/limits: TBD
- **FIRMS**
  - Purpose: active fire detections to anchor timing.
  - Interface/auth/limits: TBD

## Dataset / knowledge source inventory
- NOAA Storm Events (fields/coverage TBD)
- MTBS products (specific products/fields TBD)
- FIRMS feeds (product types/fields TBD)
- MODIS/VIIRS products (specific products TBD)

## Known error patterns / failure modes
- Date inference can be ambiguous.

## Open questions / unknowns
- Which specific endpoints/exports are used
- Which fields are used to translate event record → date window
- Whether these sources are approved/available in the target runtime
