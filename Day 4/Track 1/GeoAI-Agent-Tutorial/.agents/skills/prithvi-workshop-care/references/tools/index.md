# Tools

This directory defines the runtime tools the agent may call.

## Grouping
Tools are grouped by workflow step:
- `geocode/` — location → bbox resolution
- `hls/` — imagery availability and quality screening
- `datasets/` — auxiliary dataset signals for interpreting vague requests
- `prithvi/` — asynchronous inference job submission + polling + results

## Minimal response policy
Unless explicitly extended later, tools should return minimal fields only plus a `message`. The agent is responsible for interpretation and user-facing narrative.
