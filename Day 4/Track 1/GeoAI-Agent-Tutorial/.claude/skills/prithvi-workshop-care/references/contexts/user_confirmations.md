# Context: User confirmations & degraded-data disclosure

## How this is used in practice
- Reference for what must be explicitly confirmed by the user before running inference.
- Reference for what data-quality degradations must be disclosed in the user-facing narrative.

## When this becomes relevant
- Immediately before submitting inference (after AOI/date(s) and imagery choice/quality are resolved).

## What task(s) this supports
- Flood detection (and any future burn/crop if enabled later).

## Usage type
- Validation + communication reference.

## Authority
- Intended to be **authoritative** for workshop/demo interaction rules.

## Mandatory vs optional
- Mandatory before every inference submission.

## AOI/date(s) confirmation behavior (workshop)
- Announce the resolved AOI (human-readable) and date(s) to be used, then proceed immediately unless the user objects.
- Applies especially when:
  - The agent geocoded a place name into a bbox.
  - The agent inferred a date from event search / vague time descriptions.

## Degraded data conditions (must disclose in narrative)
- Cloud cover above the threshold.
- Using imagery from a nearby date instead of the exact requested date.
