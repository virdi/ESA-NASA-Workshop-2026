# Context: HLS data requirements & conventions

## How this is used in practice
- Used as a reference for what HLS products are acceptable and how to screen for usable imagery.
- Used internally by the agent (not typically re-stated to the user) unless needed to justify a caveat.

## When this becomes relevant
- Before checking HLS imagery availability / selecting imagery for a run.

## What task(s) this supports
- Flood detection (and any future burn/crop if enabled later).

## Usage type
- Lookup + validation reference (acceptable products; clear-pixel definition; tiling/mosaicking expectations).

## Authority
- Intended to be **authoritative** for what counts as usable HLS inputs (unless superseded by implementation constraints).

## Mandatory vs optional
- Mandatory whenever the system is about to check/select HLS imagery.

## Acceptable products
- **HLSS30 (Sentinel-2)** and **HLSL30 (Landsat)** are both acceptable.
- For a given date, search both and use the one with better AOI coverage.

## “Clear pixels” definition (cloud screening)
- Based on **Fmask QA band**:
  - Treat pixels flagged as **cloud**, **cloud shadow**, or **snow/ice** as not-clear.

## Thresholds
- Standard requirement: **≥70%** clear pixels.
- If strict threshold fails: may relax to **≥50%** clear pixels.

## AOI / tiling conventions
- HLS delivered in **MGRS tiles**.
- Multi-tile AOIs are **mosaiced automatically**.
- No hard AOI size limit specified.

## Open questions / unknowns
- Exact bit definitions/encoding for the Fmask QA band used in the pipeline.
- How “better AOI coverage” is measured (percent AOI covered, number of tiles, etc.).
