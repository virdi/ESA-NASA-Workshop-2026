import asyncio

import earthaccess
from akd._base import InputSchema, OutputSchema
from akd.tools import BaseTool
from akd_ext.mcp import mcp_tool
from pydantic import ConfigDict, Field
from pystac_client import Client

CMR_STAC_URL = "https://cmr.earthdata.nasa.gov/stac/POCLOUD"
COLLECTION = "OPERA_L3_DSWx-HLS_V1"


class QuerySurfaceWaterInput(InputSchema):
    """Parameters for OPERA DSWx surface-water query."""
    bbox: list[float] = Field(..., description="[west, south, east, north]")
    start_date: str = Field(..., description="YYYY-MM-DD")
    end_date: str = Field(..., description="YYYY-MM-DD")


class QuerySurfaceWaterOutput(OutputSchema):
    """DSWx surface-water / flooding signal summary."""
    model_config = ConfigDict(extra="ignore")
    signal: bool = Field(default=False)
    product_count: int | None = None
    dates_available: list[str] | None = None
    message: str = Field(default="")


def _query_surface_water(bbox: list[float], start_date: str, end_date: str) -> dict:
    """Query OPERA DSWx-HLS for surface water / flooding signals.

    Signal rule: product existence indicates potential water extent.
    Requires Earthdata credentials in env (EARTHDATA_LOGIN / EARTHDATA_PASSWORD).
    """
    try:
        earthaccess.login(strategy="environment")
    except Exception as e:
        return {"signal": False, "message": f"Earthdata auth failed: {e}"}

    try:
        catalog = Client.open(CMR_STAC_URL)
        search = catalog.search(
            collections=[COLLECTION],
            bbox=bbox,
            datetime=f"{start_date}/{end_date}",
            max_items=100,
        )
        items = list(search.items())
    except Exception as e:
        return {"signal": False, "message": f"CMR STAC query failed: {e}"}

    if not items:
        return {
            "signal": False,
            "product_count": 0,
            "message": "No DSWx products found for this area and date range.",
        }

    dates = sorted(
        set(item.datetime.strftime("%Y-%m-%d") for item in items if item.datetime)
    )
    return {
        "signal": True,
        "product_count": len(items),
        "dates_available": dates,
        "message": f"Found {len(items)} DSWx product(s) across {len(dates)} date(s).",
    }


@mcp_tool
class QuerySurfaceWaterTool(BaseTool[QuerySurfaceWaterInput, QuerySurfaceWaterOutput]):
    """Query OPERA DSWx-HLS to detect surface-water / potential flooding signals.

    Use when the user does not specify hazard type and the agent needs evidence of
    water/flooding. Requires Earthdata credentials.
    """

    input_schema = QuerySurfaceWaterInput
    output_schema = QuerySurfaceWaterOutput

    async def _arun(self, params: QuerySurfaceWaterInput) -> QuerySurfaceWaterOutput:
        result = await asyncio.to_thread(
            _query_surface_water, params.bbox, params.start_date, params.end_date
        )
        return QuerySurfaceWaterOutput.model_validate(result)
