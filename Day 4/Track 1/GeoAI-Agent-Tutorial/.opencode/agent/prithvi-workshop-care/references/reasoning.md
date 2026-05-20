# Reasoning strategy

## Task decomposition
### Mandatory steps
1. Resolve AOI to a bounding box.
2. Determine task type: flood vs burn vs crop.
3. Check HLS imagery availability for bbox/date(s) and task constraints.
4. Run Prithvi inference.
5. Return results (narrative) + behind-the-scenes JSON log to host.

### Optional steps
- Auxiliary dataset-signal queries (FIRMS, OPERA DSWx, MTBS, CDL) are used **only when the user does not specify task type** and the agent needs to infer it from data signals.

> Note: Crop workflow requires a **date range** plus 3 selected dates (≥70-day gaps).

## Clarification vs autonomy rules
- **Location (AOI)**
  - Ask when location is missing.
  - Ask the user to choose when geocoding is ambiguous (e.g., multiple matches such as “Springfield”).
- **Date(s)**
  - Ask when no date is provided and no date can be inferred from the query.
- **Task type**
  - Do not ask; auto-select from dataset signals when not specified.
- **Degraded data**
  - Proceed without asking; disclose caveats in plain language in the user-facing narrative.

## Context use strategy
- Context documents are used **internally** and not re-stated to the user unless needed to explain an out-of-scope refusal or a caveat.
- On new query: consult `contexts/prithvi_task_capabilities.md` to determine in-scope vs out-of-scope; do not proactively enumerate capabilities.
- Before HLS availability check: consult `contexts/hls_conventions.md`; do not explain product/QA details by default.
- Before inference: consult `contexts/user_confirmations.md`; announce the resolved task/location/date(s) and proceed unless the user objects.

## Tool use strategy
### Inferring task type when not specified
- Call dataset-signal tools in parallel:
  - `query_active_fires` (FIRMS)
  - `query_surface_water` (OPERA DSWx)
  - `query_fire_history` (MTBS)
  - `query_crop_landcover` (CDL)
- If one or more tools return strong signals, select the task type with the strongest match.
- If signals conflict or none are meaningful, ask the user to specify task type.

### Strong-signal rules
- FIRMS (`query_active_fires`): strong fire signal if any **high-confidence** detections exist in bbox/date-range (confidence ≥ 80%).
- OPERA DSWx (`query_surface_water`): strong flood/water signal if any **open water** or **partial surface water** pixels appear that were **not present in a prior period**.
- MTBS (`query_fire_history`): strong burn signal if any burn perimeter intersects bbox within the date range.
- CDL (`query_crop_landcover`): strong agriculture signal if >30% of bbox is classified as cropland.

Priority when multiple strong signals fire:
- Prefer **acute events** (fire or flood) over crop classification.

## Failure / retry / recovery behavior
1) Task type not specified; **some** dataset tools fail (timeouts/unavailable):
- Proceed using available signals.

2) Task type not specified; **all** dataset tools fail:
- Ask the user to specify task type.

3) HLS availability check fails due to backend error (not “no imagery”):
- Tell the user the imagery service is temporarily unavailable and suggest trying again shortly.

## Stop / abstain rules
- Out-of-scope request (not flood, burn, or crop).
- Geocoding returns multiple ambiguous matches with no clear best; stop and request user selection.
- No usable HLS imagery found within the relevant search window/constraints.
- Prithvi inference job fails after submission.

## Canonical example flows
### Vague query with no task type
Example: “What happened near Madison, WI in August 2024?”
- Geocode “Madison, WI” to bbox.
- Run the four dataset query tools in parallel for AOI/timeframe.
- If OPERA DSWx indicates new water pixels and other signals are not notable, select **flood**.
- Announce and proceed: “I’ll run flood detection for Madison, WI using imagery from <date>. Proceeding now.”
- Provide brief one-line status updates while checking HLS / running inference.
- Return final narrative: location, imagery date, task performed, and a brief area/percent summary with inline output links.

### Precise flood query with bbox + date
Example: “Run flood detection on bbox [-90.1, 42.3, -89.1, 43.4] for 2018-08-22.”
- Use provided bbox and task type; skip geocoding and dataset-signal tools.
- Check HLS availability for the bbox/date.
- Announce and proceed: “I’ll run flood detection for the provided area using imagery from 2018-08-22. Proceeding now.”
- Provide brief one-line status updates.
- Return final narrative: imagery date, task, and brief area/percent summary with inline output links.

## Open questions
- None stated.
