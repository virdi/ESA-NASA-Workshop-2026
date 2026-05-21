import asyncio
import os

import requests
from akd._base import InputSchema, OutputSchema
from akd.tools import BaseTool
from akd_ext.mcp import mcp_tool
from pydantic import ConfigDict, Field, BaseModel

PRITHVI_SERVER_URL = os.environ.get("PRITHVI_SERVER_URL", "http://localhost:8080")
MODEL_SERVER_TIMEOUT = int(os.environ.get("MODEL_SERVER_TIMEOUT", "300"))


class RunPrithviInferenceInput(InputSchema):
    """Parameters for Prithvi-EO inference."""
    bounding_box: list[float] = Field(..., description="[west, south, east, north]")
    date: str | None = Field(default=None, description="YYYY-MM-DD (required for flood/burn)")
    date_range: dict | None = Field(
        default=None,
        description="{'start_date': 'YYYY-MM-DD', 'end_date': 'YYYY-MM-DD'} (crop only)",
    )
    dates: list[str] | None = Field(
        default=None,
        description="List of 3 YYYY-MM-DD strings with >=70-day gaps (crop only)",
    )

class TaskResult(BaseModel):
    cog_s3_link: str
    geojson_s3_link: str
   
class RunPrithviInferenceOutput(OutputSchema):
    """Prithvi-EO inference result.

    On success, contains the result under the usecase key (e.g. 'flood'), with
    s3_link (GeoTIFF path) and predictions (GeoJSON FeatureCollection).
    On failure, status='failed' and message describes the error.
    """
    model_config = ConfigDict(extra="allow")
    status: str = Field(default="")
    message: str = Field(default="")
    flood: TaskResult | None = None
    burn: TaskResult | None = None
    crop: TaskResult | None = None


def _run_prithvi_inference(
    bounding_box: list[float],
    date: str | None,
    date_range: dict | None,
    dates: list[str] | None,
) -> dict:
    payload = {
        "bounding_box": bounding_box,
        "date": date,
        "date_range": date_range,
        "dates": dates,
    }

    try:
        resp = requests.post(
            f"{PRITHVI_SERVER_URL}/invocations",
            json=payload,
            timeout=MODEL_SERVER_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        return {"status": "failed", "message": f"Inference failed: {e}"}


@mcp_tool
class RunPrithviInferenceTool(BaseTool[RunPrithviInferenceInput, RunPrithviInferenceOutput]):
    """Run Prithvi-EO inference for flood detection, burn-scar mapping, or crop classification.

    Returns the result directly under the usecase key (e.g. 'flood'), containing
    s3_link (GeoTIFF on S3) and predictions (GeoJSON FeatureCollection).
    """

    input_schema = RunPrithviInferenceInput
    output_schema = RunPrithviInferenceOutput

    async def _arun(self, params: RunPrithviInferenceInput) -> RunPrithviInferenceOutput:
        result = await asyncio.to_thread(
            _run_prithvi_inference,
            params.bounding_box,
            params.date,
            params.date_range,
            params.dates,
        )
        return RunPrithviInferenceOutput.model_validate(result)
