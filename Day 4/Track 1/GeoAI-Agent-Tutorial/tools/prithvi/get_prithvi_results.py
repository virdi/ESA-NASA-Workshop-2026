import asyncio
import os

import requests
from akd._base import InputSchema, OutputSchema
from akd.tools import BaseTool
from akd_ext.mcp import mcp_tool
from pydantic import ConfigDict, Field

PRITHVI_SERVER_URL = os.environ.get("PRITHVI_SERVER_URL", "http://localhost:8000")


class GetPrithviResultsInput(InputSchema):
    """Parameters for Prithvi job results retrieval."""
    job_id: str = Field(..., description="Job ID returned by run_prithvi_inference")


class GetPrithviResultsOutput(OutputSchema):
    """Prithvi inference outputs: GeoTIFF URLs and area statistics."""
    model_config = ConfigDict(extra="ignore")
    task_type: str | None = None
    result_urls: list[str] | None = None
    result_tiles: dict | None = None
    summary: dict | None = None
    message: str | None = None


def _get_prithvi_results(job_id: str) -> dict:
    try:
        resp = requests.get(
            f"{PRITHVI_SERVER_URL}/results/{job_id}",
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        return {"message": f"Results retrieval failed: {e}"}


@mcp_tool
class GetPrithviResultsTool(BaseTool[GetPrithviResultsInput, GetPrithviResultsOutput]):
    """Retrieve outputs for a finished Prithvi inference job.

    Returns result_urls (GeoTIFFs) and summary statistics.
    Call only after get_prithvi_job_status returns 'finished'.
    """

    input_schema = GetPrithviResultsInput
    output_schema = GetPrithviResultsOutput

    async def _arun(self, params: GetPrithviResultsInput) -> GetPrithviResultsOutput:
        result = await asyncio.to_thread(_get_prithvi_results, params.job_id)
        return GetPrithviResultsOutput.model_validate(result)
