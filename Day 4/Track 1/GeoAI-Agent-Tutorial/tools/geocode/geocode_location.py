import asyncio
import time

import requests
from akd._base import InputSchema, OutputSchema
from akd.tools import BaseTool
from akd_ext.mcp import mcp_tool
from pydantic import ConfigDict, Field

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
HEADERS = {"User-Agent": "prithvi-workshop-agent/1.0"}


class GeoCodeLocationInput(InputSchema):
    """Place name or description to geocode."""
    query: str = Field(
        ...,
        description="Place name, event/location description, or bbox coordinates as user-provided text",
    )


class GeoCodeLocationOutput(OutputSchema):
    """Bounding box result or candidate list from geocoding."""
    model_config = ConfigDict(extra="ignore")
    bbox: list[float] | None = None
    display_name: str | None = None
    candidates: list[dict] | None = None
    message: str = Field(default="")


def _geocode_location(query: str) -> dict:
    params = {"q": query, "format": "json", "limit": 5}
    try:
        resp = requests.get(NOMINATIM_URL, params=params, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        results = resp.json()
    except requests.RequestException as e:
        return {"message": f"Geocoding service unavailable: {e}"}
    finally:
        time.sleep(1)  # Nominatim enforces 1 req/sec

    if not results:
        return {
            "message": f"No results for '{query}'. Try rephrasing or provide coordinates."
        }

    def _parse_bbox(r: dict) -> list:
        bb = r["boundingbox"]
        return [float(bb[2]), float(bb[0]), float(bb[3]), float(bb[1])]

    if len(results) == 1:
        return {
            "bbox": _parse_bbox(results[0]),
            "display_name": results[0]["display_name"],
            "message": "ok",
        }

    candidates = [
        {"display_name": r["display_name"], "bbox": _parse_bbox(r)}
        for r in results[:3]
    ]
    return {"candidates": candidates, "message": "Multiple matches found. Please choose one."}


@mcp_tool
class GeoCodeLocationTool(BaseTool[GeoCodeLocationInput, GeoCodeLocationOutput]):
    """Convert a place name or description to a bounding box [west, south, east, north].

    Returns a single bbox when unambiguous, or a candidates list when multiple
    matches are found. Nominatim rate limit: 1 req/sec enforced by sleep.
    """

    input_schema = GeoCodeLocationInput
    output_schema = GeoCodeLocationOutput

    async def _arun(self, params: GeoCodeLocationInput) -> GeoCodeLocationOutput:
        result = await asyncio.to_thread(_geocode_location, params.query)
        return GeoCodeLocationOutput.model_validate(result)
