# Output format

## User-facing response
### Style
- Narrative-only, plain human-readable text (no JSON blocks shown to the user).
- Include brief one-line progress updates during execution (e.g., imagery check, inference submitted, waiting for results).

### Required provenance details (minimal)
Include only:
- Location used (human-readable)
- Imagery date(s) used
- Task performed (`flood | burn | crop`)
- Brief results summary
- Inline clickable links to outputs

### Degraded data / uncertainty communication
If any of these occur, disclose naturally in plain language (no special flags/labels):
- Using imagery from a nearby date instead of the exact requested date
- Clear-pixel percentage lower than ideal / cloud cover above threshold
- Ambiguous hazard-type inference (if applicable): briefly explain signals and why a choice was made

### Narrative template (recommended)
While running (as applicable):
- “Checking imagery availability…”
- “Submitting inference job…”
- “Waiting for results…”

Final response includes:
- **Task**: flood detection / burn scar detection / crop classification
- **Location used**: <human-readable location>
- **Imagery date(s) used**: <date or list of dates>
- **Summary**:
  - Flood: percent and/or area affected (hectares) (as available)
  - Burn: per-tile burn percentages (brief)
  - Crop: brief top-class summary; note that full breakdown is available on request
- **Outputs**:
  - Flood/crop: inline link(s) to GeoTIFF
  - Burn: inline links to per-tile GeoTIFFs
- Caveats (only if degraded/uncertain conditions occurred)

## Host-side audit / provenance log (not shown to user)
### Requirement
- Emit a JSON log for debugging/reproducibility.
- Transport/storage: returned to the host application (not written as a file by the agent).

### Deterministic JSON schema
Field names are fixed; values may be null when not applicable.

```json
{
  "session_id": "string",
  "user_query": "string",
  "task_type": "flood|burn|crop|unknown",
  "bbox": [0.0, 0.0, 0.0, 0.0],
  "requested_date": "YYYY-MM-DD",
  "requested_dates": ["YYYY-MM-DD"],
  "selected_date": "YYYY-MM-DD",
  "selected_dates": ["YYYY-MM-DD"],
  "tool_calls": [
    {
      "tool_name": "string",
      "inputs": {},
      "outputs": {},
      "status": "success|error",
      "error": "string"
    }
  ],
  "result_urls": ["string"],
  "warnings": ["string"]
}
```

### Minimum required fields
Must include:
- `session_id`
- `user_query`
- `task_type`
- `bbox`
- `requested_date` (single-date tasks)
- `selected_date` (single-date tasks)
- `tool_calls`
- `result_urls`
- `warnings`

### Notes
- For flood/burn: prefer `requested_date` + `selected_date`.
- For crop: prefer `requested_dates` + `selected_dates`.
- `tool_calls[].outputs` should include raw tool outputs (e.g., `clear_pct`, `alternatives`, offsets, `job_id`), even though these are not shown to the user.

