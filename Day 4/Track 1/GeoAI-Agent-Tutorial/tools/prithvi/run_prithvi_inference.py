import asyncio
import os

import requests
from akd._base import InputSchema, OutputSchema
from akd.tools import BaseTool
from akd_ext.mcp import mcp_tool
from pydantic import ConfigDict, Field

PRITHVI_SERVER_URL = os.environ.get("PRITHVI_SERVER_URL", "http://localhost:8000")


class RunPrithviInferenceInput(InputSchema):
    """Parameters for Prithvi-EO inference job submission."""
    task_type: str = Field(..., description="'flood' | 'burn' | 'crop'")
    bbox: list[float] = Field(..., description="[west, south, east, north]")
    date: str | None = Field(default=None, description="YYYY-MM-DD (required for flood/burn)")
    date_range: dict | None = Field(
        default=None,
        description="{'start_date': 'YYYY-MM-DD', 'end_date': 'YYYY-MM-DD'} (crop only)",
    )
    dates: list[str] | None = Field(
        default=None,
        description="List of 3 YYYY-MM-DD strings with >=70-day gaps (crop only)",
    )


class RunPrithviInferenceOutput(OutputSchema):
    """Inference job submission result with job_id."""
    model_config = ConfigDict(extra="ignore")
    job_id: str | None = None
    status: str = Field(default="")
    message: str = Field(default="")


def _run_prithvi_inference(
    task_type: str,
    bbox: list[float],
    date: str | None,
    date_range: dict | None,
    dates: list[str] | None,
) -> dict:
    """Submit an async Prithvi-EO inference job to the local inference server."""
    if task_type not in ("flood", "burn", "crop"):
        return {
            "message": f"Unsupported task_type '{task_type}'. Must be flood, burn, or crop."
        }
    if task_type in ("flood", "burn") and not date:
        return {"message": f"'date' is required for {task_type} task."}
    if task_type == "crop" and (not date_range or not dates or len(dates) != 3):
        return {"message": "Crop task requires date_range and exactly 3 dates."}

    payload = {
        "task_type": task_type,
        "bbox": bbox,
        "date": date,
        "date_range": date_range,
        "dates": dates,
    }

    try:
        resp = requests.post(
            f"{PRITHVI_SERVER_URL}/infer",
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        return {"status": "failed", "message": f"Job submission failed: {e}"}


@mcp_tool
class RunPrithviInferenceTool(BaseTool[RunPrithviInferenceInput, RunPrithviInferenceOutput]):
    """Submit an async Prithvi-EO inference job for flood detection, burn-scar mapping, or crop classification.

    Returns a job_id. Poll status with get_prithvi_job_status, then retrieve
    results with get_prithvi_results.
    """

    input_schema = RunPrithviInferenceInput
    output_schema = RunPrithviInferenceOutput

    async def _arun(self, params: RunPrithviInferenceInput) -> RunPrithviInferenceOutput:
        result = await asyncio.to_thread(
            _run_prithvi_inference,
            params.task_type,
            params.bbox,
            params.date,
            params.date_range,
            params.dates,
        )
        return RunPrithviInferenceOutput.model_validate(result)
