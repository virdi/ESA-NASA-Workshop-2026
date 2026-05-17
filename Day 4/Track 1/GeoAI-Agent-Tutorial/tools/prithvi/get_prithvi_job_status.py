import asyncio
import os

import requests
from akd._base import InputSchema, OutputSchema
from akd.tools import BaseTool
from akd_ext.mcp import mcp_tool
from pydantic import ConfigDict, Field

PRITHVI_SERVER_URL = os.environ.get("PRITHVI_SERVER_URL", "http://localhost:8080")


class GetPrithviJobStatusInput(InputSchema):
    """Parameters for Prithvi job status poll."""
    job_id: str = Field(..., description="Job ID returned by run_prithvi_inference")


class GetPrithviJobStatusOutput(OutputSchema):
    """Current status of a Prithvi inference job."""
    model_config = ConfigDict(extra="ignore")
    status: str = Field(default="", description="'running' | 'finished' | 'failed'")
    message: str = Field(default="")


def _get_prithvi_job_status(job_id: str) -> dict:
    try:
        resp = requests.get(
            f"{PRITHVI_SERVER_URL}/status/{job_id}",
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        return {"status": "failed", "message": f"Status check failed: {e}"}


@mcp_tool
class GetPrithviJobStatusTool(BaseTool[GetPrithviJobStatusInput, GetPrithviJobStatusOutput]):
    """Check the status of a submitted Prithvi inference job.

    Returns status: 'running' | 'finished' | 'failed'.
    Call after run_prithvi_inference; poll until finished before calling get_prithvi_results.
    """

    input_schema = GetPrithviJobStatusInput
    output_schema = GetPrithviJobStatusOutput

    async def _arun(self, params: GetPrithviJobStatusInput) -> GetPrithviJobStatusOutput:
        result = await asyncio.to_thread(_get_prithvi_job_status, params.job_id)
        return GetPrithviJobStatusOutput.model_validate(result)
