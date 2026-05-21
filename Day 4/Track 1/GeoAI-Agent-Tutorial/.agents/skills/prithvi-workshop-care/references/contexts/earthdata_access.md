# Existing prerequisite: NASA Earthdata / LP DAAC access

## Dataset / knowledge source inventory
- **NASA Earthdata / LP DAAC** public data holdings used by the system:
  - HLS imagery (HLSS30, HLSL30)
  - MODIS/VIIRS products (specific products TBD)

## Schemas, access patterns, and documentation
- Access pattern: authenticated download with Earthdata Login.
- Credential mechanism: `.netrc` entry for `urs.earthdata.nasa.gov`.
- Exact endpoints, file formats, and catalog interfaces: TBD

## Permissions, limits, and operational constraints
- Requires user/service account with Earthdata Login credentials.
- Quotas/rate limits: TBD

## Known error patterns / failure modes
- Credential misconfiguration (missing/incorrect `.netrc`)
- Credential expiration / throttling behavior: TBD

## Open questions / unknowns
- Whether downloads happen server-side only or can return signed URLs
- Credential model (shared service account vs per-user): TBD
