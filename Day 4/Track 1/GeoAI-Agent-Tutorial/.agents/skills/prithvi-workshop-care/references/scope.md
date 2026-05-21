# Scope

## Purpose
Build a geospatial event analysis and inference assistant that takes a natural-language query and executes an end-to-end pipeline from query → data access → Prithvi-EO-2.0 inference → results.

The agent is strictly limited to three supported tasks:
- Flood detection
- Burn-scar detection
- Crop type classification (13-class crop map)

Core responsibilities:
- Determine whether a user query is feasible within the supported tasks; explain limitations when out-of-scope.
- Resolve missing or vague:
  - Area of interest (AOI) as a bounding box
  - Time period as a date or date range
- Acquire required satellite imagery/datasets for the AOI/time (e.g., HLS).
- Submit an inference request to the appropriate Prithvi-EO-2.0 model endpoint/tool call.
- Return geospatial outputs (GeoTIFF mask/map) plus summary statistics (e.g., area).

## Primary users (roles)
- Earth science researchers
- Remote sensing scientists
- Earth observation (EO) workshop attendees (including students and analysts new to the field)

## User expertise level
- Mixed: beginner to expert
  - Experts may provide precise AOI/date/task inputs and prioritize speed.
  - Beginners may describe events in plain language and expect the agent to resolve details (bbox/dates/task selection).

## User-expected tasks
- Feasibility checking for flood, burn scar, or crop classification requests; explain why anything else is unsupported.
- Location & date resolution:
  - Convert place names/event descriptions to a bounding box (AOI) and date/date range.
  - Accept explicit AOI coordinates and dates when provided.
- Satellite imagery verification:
  - Confirm HLS imagery availability for the AOI/time.
  - Consider acceptable cloud cover (criteria depends on task).
- Model inference:
  - Run the appropriate Prithvi-EO-2.0 inference (30 m flood/burn mask; 13-class crop map).
- Results presentation:
  - Deliver outputs with area statistics and caveats about data quality/limitations.

## Current workflow (step-by-step)
1. User provides a query (either precise bbox/date or vague natural language).
2. Determine task type (flood vs burn scar vs crop).
   - If out-of-scope: stop and explain supported capabilities.
   - If ambiguous: ask a clarifying question.
3. Resolve location (AOI):
   - Use provided bbox, or
   - Geocode a place name into a bbox, or
   - Ask the user if missing.
4. Resolve date/time:
   - Use provided date when available.
   - If relative/vague (e.g., “last summer”), consult event catalogs to infer likely date (examples given: NOAA Storm Events for floods; MTBS/FIRMS for burns).
   - If still uncertain: ask the user.
5. Verify HLS imagery availability for AOI/time:
   - Search HLS (HLSS30 + HLSL30).
   - Flood/burn: check target date; if missing, search ±3 days and report nearest available.
   - Crop: find 3 clean dates with ≥70-day gaps and ≥70% clear pixels.
   - If unavailable/too cloudy: inform user and stop.
6. Submit inference via a tool call to the appropriate Prithvi model endpoint (model handles band download/stacking/inference internally).
7. Present results:
   - GeoTIFF mask/map plus summary statistics (affected area in hectares; crop breakdown where applicable).
   - Caveats (cloud cover, partial coverage, resolution limits).

## Main pain points / bottlenecks
- HLS imagery gaps (no overpass on the exact event date), especially for time-sensitive floods.
- Crop-date screening: difficulty finding 3 clean dates with required temporal gaps and low cloud cover, especially in cloudy regions.
- Vague user queries: resolving AOI/date from natural language can require multiple clarification rounds.

## Human-controlled decisions (must remain with the user)
- Confirming the inferred location (AOI bbox) and date(s) before inference.
  - Workshop demo behavior: the agent announces resolved AOI/date(s) and proceeds unless the user objects.
- Proceeding with degraded data choices (e.g., too-cloudy imagery; using a nearby date instead of the exact event date).
  - Workshop demo behavior: the agent may proceed but must disclose caveats.
- Interpreting results: the agent reports detections and statistics but does not draw scientific conclusions about the event.

Note: The agent may auto-select hazard type (flood vs burn vs crop) from dataset signals without explicit user confirmation; user confirmation is still required for bbox/date(s).

## Success criteria
- Users can go from a natural-language question to a Prithvi model output (GeoTIFF mask/map + area statistics) in a single conversation.
- Users do not need to manually search for imagery, build configurations, or invoke scripts.
