from akd.tools import BaseToolConfig

from .datasets.query_active_fires import QueryActiveFiresTool
from .datasets.query_crop_landcover import QueryCropLandcoverTool
from .datasets.query_fire_history import QueryFireHistoryTool
from .datasets.query_surface_water import QuerySurfaceWaterTool
from .geocode.geocode_location import GeoCodeLocationTool
from .hls.check_hls_availability import CheckHLSAvailabilityTool
from .prithvi.get_prithvi_job_status import GetPrithviJobStatusTool
from .prithvi.get_prithvi_results import GetPrithviResultsTool
from .prithvi.run_prithvi_inference import RunPrithviInferenceTool

# Tool names must match what the artifact documents (agents.md / tools/*.md).
# BaseToolConfig(name=...) overrides the default class-name-derived name.
TOOLS = [
    GeoCodeLocationTool(config=BaseToolConfig(name="geocode_location")),
    CheckHLSAvailabilityTool(config=BaseToolConfig(name="check_hls_availability")),
    RunPrithviInferenceTool(config=BaseToolConfig(name="run_prithvi_inference")),
    GetPrithviJobStatusTool(config=BaseToolConfig(name="get_prithvi_job_status")),
    GetPrithviResultsTool(config=BaseToolConfig(name="get_prithvi_results")),
    QueryActiveFiresTool(config=BaseToolConfig(name="query_active_fires")),
    QueryFireHistoryTool(config=BaseToolConfig(name="query_fire_history")),
    QueryCropLandcoverTool(config=BaseToolConfig(name="query_crop_landcover")),
    QuerySurfaceWaterTool(config=BaseToolConfig(name="query_surface_water")),
]

__all__ = ["TOOLS"]
