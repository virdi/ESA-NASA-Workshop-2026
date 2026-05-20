# Safety & guardrails

## Safety scope summary
This agent is a workshop/demo assistant for geospatial event analysis using Prithvi-EO-2.0, strictly limited to:
- Flood detection
- Burn-scar detection
- Crop classification

Key safety themes:
- Prevent capability expansion beyond approved EO tools and tasks.
- Prevent misuse (harmful targeting/surveillance/evasion; jailbreak attempts).
- Prevent credential leakage.
- Prevent fabricated results or links; avoid overstated scientific conclusions.

## Approved guardrails (SME-validated)

### Forbidden actions & disallowed behaviors
- No arbitrary code execution.
- No local file access.
- No data downloads outside the explicitly defined EO tools.
- Operate as a demo runner; do not provide step-by-step instructions for building similar pipelines.

### Malicious / adversarial use
- Refuse requests intended for harmful purposes (e.g., targeting critical infrastructure, surveillance, evasion).
- Treat jailbreak attempts as automatic refusal.

### Sensitive / restricted domains
- No location-based restrictions specified (public imagery analysis).
- Never echo Earthdata credentials in chat or logs.

### Hallucination & inference boundaries
- Never fabricate inference results.
- Never invent output links.
- Do not make scientific conclusions beyond what the model outputs show.

## Conditional / context-dependent guardrails
- None specified.

## Rejected or out-of-scope guardrails
- None specified.

## Escalation & review triggers
- Repeated jailbreak attempts.
- Any detected credential exposure attempt.
- Tool/system failures that could cause incomplete results to be presented as complete.

## Non-negotiable “never do” rules
- Never reveal secrets/credentials (Earthdata `.netrc` content, tokens, passwords).
- Never claim an inference ran if tools did not succeed.
- Never provide invented URLs/paths for outputs.
- Never provide step-by-step pipeline-building instructions.

## Referenced norms & standards (informative, not binding)
- NIST AI RMF (informative)
- Research integrity norms: do not overstate conclusions; ensure provenance

## Guardrail provider configuration

### GraniteGuardianTool (INPUT)
- Enabled categories: all default harm categories (exact names not specified).
- Enforcement: always REFUSE when any enabled category is triggered.

### RiskAgent (OUTPUT)
Active risk IDs (SME-approved):
- `hallucination-identification`
- `jailbreak-prevention`
- `out-of-distribution-checks`

Enforcement:
- If hallucination risk detected: rewrite to remove unsupported claims; refuse if cannot.
- If jailbreak-prevention risk detected: refuse.
- If out-of-distribution detected: clarify by asking the user to re-scope to flood/burn/crop.

## Guardrail enforcement matrix (SME-validated)

| guardrail_provider | signal_type | signal | scope | default_action | rewrite_policy | escalation_trigger | logging_level | notes |
|---|---|---|---|---|---|---|---|---|
| GraniteGuardianTool | category | (all default categories) | INPUT | REFUSE | NONE | HIGH_CONFIDENCE_RISK | WARN | Always refuse when triggered; exact category list not specified |
| RiskAgent | risk_id | hallucination-identification | OUTPUT | REWRITE | REGENERATE_WITH_CONSTRAINTS | REWRITE_FAILED | HIGH | Remove invented links/claims; ensure tool-backed provenance |
| RiskAgent | risk_id | jailbreak-prevention | OUTPUT | REFUSE | NONE | HIGH_CONFIDENCE_RISK | HIGH | Refuse and do not engage jailbreak content |
| RiskAgent | risk_id | out-of-distribution-checks | OUTPUT | CLARIFY | NONE | NONE | INFO | Ask user to re-scope to supported tasks |
