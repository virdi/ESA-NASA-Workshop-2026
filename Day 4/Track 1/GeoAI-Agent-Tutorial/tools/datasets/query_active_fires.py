import asyncio
import os
from io import StringIO

import pandas as pd
import requests
from akd._base import InputSchema, OutputSchema
from akd.tools import BaseTool
from akd_ext.mcp import mcp_tool
from pydantic import ConfigDict, Field

FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"


class QueryActiveFiresInput(InputSchema):
    """Parameters for FIRMS active-fire query."""
    bbox: list[float] = Field(..., description="[west, south, east, north]")
    start_date: str = Field(..., description="YYYY-MM-DD")
    end_date: str = Field(..., description="YYYY-MM-DD")


class QueryActiveFiresOutput(OutputSchema):
    """Active-fire detection summary from FIRMS VIIRS."""
    model_config = ConfigDict(extra="ignore")
    detections: bool = Field(default=False)
    count: int = Field(default=0)
    total_detections: int | None = None
    date_range: list | None = None
    message: str = Field(default="")


def _query_active_fires(bbox: list[float], start_date: str, end_date: str) -> dict:
    """Query FIRMS VIIRS_SNPP_NRT for high-confidence fire detections.

    Signal rule: confidence == 'high' (VIIRS string field).
    FIRMS NRT covers ~3 months. Requires FIRMS_MAP_KEY env var.
    """
    map_key = os.environ.get("FIRMS_MAP_KEY")
    if not map_key:
        return {"detections": False, "count": 0, "message": "FIRMS_MAP_KEY env var not set."}

    bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    url = f"{FIRMS_BASE}/{map_key}/VIIRS_SNPP_NRT/{bbox_str}/10/{start_date}"

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
    except requests.RequestException as e:
        return {"detections": False, "count": 0, "message": f"FIRMS API error: {e}"}
    except Exception as e:
        return {"detections": False, "count": 0, "message": f"Response parse error: {e}"}

    if df.empty or "confidence" not in df.columns:
        return {"detections": False, "count": 0, "message": "No fire detections found."}

    df["acq_date"] = pd.to_datetime(df["acq_date"])
    df = df[(df["acq_date"] >= start_date) & (df["acq_date"] <= end_date)]
    high_conf = df[df["confidence"] == "high"]

    return {
        "detections": len(high_conf) > 0,
        "count": int(len(high_conf)),
        "total_detections": int(len(df)),
        "date_range": [
            df["acq_date"].min().strftime("%Y-%m-%d") if not df.empty else None,
            df["acq_date"].max().strftime("%Y-%m-%d") if not df.empty else None,
        ],
        "message": (
            f"{len(high_conf)} high-confidence VIIRS fire detection(s)."
            if len(high_conf) > 0
            else "No high-confidence fire detections."
        ),
    }


@mcp_tool
class QueryActiveFiresTool(BaseTool[QueryActiveFiresInput, QueryActiveFiresOutput]):
    """Query FIRMS VIIRS_SNPP_NRT for high-confidence fire detections within a bbox and date range.

    Use when the user request is about fire/burn and the agent needs evidence of
    recent fire activity. FIRMS NRT covers ~3 months; for older events use query_fire_history.
    """

    input_schema = QueryActiveFiresInput
    output_schema = QueryActiveFiresOutput

    async def _arun(self, params: QueryActiveFiresInput) -> QueryActiveFiresOutput:
        result = await asyncio.to_thread(
            _query_active_fires, params.bbox, params.start_date, params.end_date
        )
        return QueryActiveFiresOutput.model_validate(result)
