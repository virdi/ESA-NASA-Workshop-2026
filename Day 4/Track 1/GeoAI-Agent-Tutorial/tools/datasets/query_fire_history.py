import asyncio

import requests
from akd._base import InputSchema, OutputSchema
from akd.tools import BaseTool
from akd_ext.mcp import mcp_tool
from pydantic import ConfigDict, Field

MTBS_SERVICE_URL = (
    "https://apps.fs.usda.gov/arcx/rest/services/EDW/EDW_MTBS_01/MapServer/0/query"
)


class QueryFireHistoryInput(InputSchema):
    """Parameters for MTBS historical-burn query."""
    bbox: list[float] = Field(..., description="[west, south, east, north]")
    start_date: str = Field(..., description="YYYY-MM-DD")
    end_date: str = Field(..., description="YYYY-MM-DD")


class QueryFireHistoryOutput(OutputSchema):
    """MTBS fire-perimeter intersection results."""
    model_config = ConfigDict(extra="ignore")
    intersections: int = Field(default=0)
    fires: list[dict] | None = None
    message: str = Field(default="")


def _query_fire_history(bbox: list[float], start_date: str, end_date: str) -> dict:
    """Query MTBS fire perimeters intersecting a bbox within a date range.

    Signal rule: strong burn signal if any perimeter intersects bbox within the date range.
    MTBS data typically lags ~2 years; use query_active_fires for recent events.
    """
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])

    params = {
        "where": f"Year >= {start_year} AND Year <= {end_year}",
        "geometry": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "geometryType": "esriGeometryEnvelope",
        "spatialRel": "esriSpatialRelIntersects",
        "inSR": "4326",
        "outFields": "Incid_Name,Year,BurnBndAc,Ig_Date",
        "returnGeometry": "false",
        "f": "json",
    }

    try:
        resp = requests.get(MTBS_SERVICE_URL, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        return {"intersections": 0, "message": f"MTBS service unavailable: {e}"}

    features = data.get("features", [])
    if not features:
        return {
            "intersections": 0,
            "message": f"No MTBS burn perimeters found for {start_year}–{end_year} in this area.",
        }

    fires = [f["attributes"] for f in features]
    return {
        "intersections": len(fires),
        "fires": fires,
        "message": f"{len(fires)} MTBS burn perimeter(s) intersect this area.",
    }


@mcp_tool
class QueryFireHistoryTool(BaseTool[QueryFireHistoryInput, QueryFireHistoryOutput]):
    """Query MTBS to find historical fire perimeters intersecting a bbox within a date range.

    Use when the user does not specify hazard type and the agent needs evidence of burns.
    MTBS data lags ~2 years; for recent activity use query_active_fires instead.
    """

    input_schema = QueryFireHistoryInput
    output_schema = QueryFireHistoryOutput

    async def _arun(self, params: QueryFireHistoryInput) -> QueryFireHistoryOutput:
        result = await asyncio.to_thread(
            _query_fire_history, params.bbox, params.start_date, params.end_date
        )
        return QueryFireHistoryOutput.model_validate(result)
