"""Unit tests for GeoAI-Agent-Tutorial tools.

Each tool's pure helper function is tested directly (synchronous, no I/O).
The async _arun methods are tested via the tool class to confirm schema
validation and threading behaviour.

External dependencies (requests, earthaccess, pystac_client, time.sleep)
are patched throughout; no network calls are made.
"""
from __future__ import annotations

import os
from datetime import date, datetime
from unittest.mock import MagicMock, patch

import pytest
import requests

from akd.tools import BaseToolConfig

from tools.datasets.query_active_fires import (
    QueryActiveFiresInput,
    QueryActiveFiresTool,
    _query_active_fires,
)
from tools.datasets.query_crop_landcover import (
    QueryCropLandcoverInput,
    QueryCropLandcoverTool,
    _query_crop_landcover,
)
from tools.datasets.query_fire_history import (
    QueryFireHistoryInput,
    QueryFireHistoryTool,
    _query_fire_history,
)
from tools.datasets.query_surface_water import (
    QuerySurfaceWaterInput,
    QuerySurfaceWaterTool,
    _query_surface_water,
)
from tools.geocode.geocode_location import (
    GeoCodeLocationInput,
    GeoCodeLocationTool,
    _geocode_location,
)
from tools.hls.check_hls_availability import (
    CheckHLSAvailabilityInput,
    CheckHLSAvailabilityTool,
    _check_hls_availability,
)
from tools.prithvi.get_prithvi_job_status import (
    GetPrithviJobStatusInput,
    GetPrithviJobStatusTool,
    _get_prithvi_job_status,
)
from tools.prithvi.get_prithvi_results import (
    GetPrithviResultsInput,
    GetPrithviResultsTool,
    _get_prithvi_results,
)
from tools.prithvi.run_prithvi_inference import (
    RunPrithviInferenceInput,
    RunPrithviInferenceTool,
    _run_prithvi_inference,
)

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

BBOX = [-122.5, 37.5, -122.0, 38.0]


def _mock_response(status: int = 200, json_data=None, text: str = "") -> MagicMock:
    """Return a minimal mock requests.Response."""
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = json_data if json_data is not None else {}
    resp.text = text
    if status >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(response=resp)
    else:
        resp.raise_for_status.return_value = None
    return resp


def _tool(cls, name: str):
    return cls(config=BaseToolConfig(name=name))


# ---------------------------------------------------------------------------
# QueryActiveFires
# ---------------------------------------------------------------------------

class TestQueryActiveFiresFunction:
    def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("FIRMS_MAP_KEY", raising=False)
        result = _query_active_fires(BBOX, "2024-01-01", "2024-01-10")
        assert result["detections"] is False
        assert "FIRMS_MAP_KEY" in result["message"]

    def test_http_error(self, monkeypatch):
        monkeypatch.setenv("FIRMS_MAP_KEY", "testkey")
        with patch("tools.datasets.query_active_fires.requests.get",
                   return_value=_mock_response(500)):
            result = _query_active_fires(BBOX, "2024-01-01", "2024-01-10")
        assert result["detections"] is False
        assert "FIRMS API error" in result["message"]

    def test_empty_response_no_detections(self, monkeypatch):
        monkeypatch.setenv("FIRMS_MAP_KEY", "testkey")
        with patch("tools.datasets.query_active_fires.requests.get",
                   return_value=_mock_response(200, text="latitude,longitude,confidence\n")):
            result = _query_active_fires(BBOX, "2024-01-01", "2024-01-10")
        assert result["detections"] is False
        assert result["count"] == 0

    def test_high_confidence_detections_counted(self, monkeypatch):
        monkeypatch.setenv("FIRMS_MAP_KEY", "testkey")
        csv = (
            "latitude,longitude,acq_date,confidence\n"
            "38.0,-122.0,2024-01-05,high\n"
            "38.1,-122.1,2024-01-06,nominal\n"
        )
        with patch("tools.datasets.query_active_fires.requests.get",
                   return_value=_mock_response(200, text=csv)):
            result = _query_active_fires(BBOX, "2024-01-01", "2024-01-10")
        assert result["detections"] is True
        assert result["count"] == 1
        assert result["total_detections"] == 2

    def test_only_nominal_confidence_no_signal(self, monkeypatch):
        monkeypatch.setenv("FIRMS_MAP_KEY", "testkey")
        csv = "latitude,longitude,acq_date,confidence\n38.0,-122.0,2024-01-05,nominal\n"
        with patch("tools.datasets.query_active_fires.requests.get",
                   return_value=_mock_response(200, text=csv)):
            result = _query_active_fires(BBOX, "2024-01-01", "2024-01-10")
        assert result["detections"] is False
        assert result["count"] == 0

    def test_detections_outside_date_range_excluded(self, monkeypatch):
        monkeypatch.setenv("FIRMS_MAP_KEY", "testkey")
        csv = (
            "latitude,longitude,acq_date,confidence\n"
            "38.0,-122.0,2023-12-31,high\n"  # before range
            "38.0,-122.0,2024-01-05,high\n"  # inside range
        )
        with patch("tools.datasets.query_active_fires.requests.get",
                   return_value=_mock_response(200, text=csv)):
            result = _query_active_fires(BBOX, "2024-01-01", "2024-01-10")
        assert result["count"] == 1


@pytest.mark.asyncio
class TestQueryActiveFiresTool:
    async def test_arun_returns_validated_output(self, monkeypatch):
        monkeypatch.setenv("FIRMS_MAP_KEY", "testkey")
        csv = "latitude,longitude,acq_date,confidence\n38.0,-122.0,2024-01-05,high\n"
        tool = _tool(QueryActiveFiresTool, "query_active_fires")
        with patch("tools.datasets.query_active_fires.requests.get",
                   return_value=_mock_response(200, text=csv)):
            out = await tool._arun(
                QueryActiveFiresInput(bbox=BBOX, start_date="2024-01-01", end_date="2024-01-10")
            )
        assert out.detections is True
        assert out.count == 1

    async def test_arun_handles_missing_key(self, monkeypatch):
        monkeypatch.delenv("FIRMS_MAP_KEY", raising=False)
        tool = _tool(QueryActiveFiresTool, "query_active_fires")
        out = await tool._arun(
            QueryActiveFiresInput(bbox=BBOX, start_date="2024-01-01", end_date="2024-01-10")
        )
        assert out.detections is False


# ---------------------------------------------------------------------------
# QueryCropLandcover
# ---------------------------------------------------------------------------

class TestQueryCropLandcoverFunction:
    def test_api_error(self):
        with patch("tools.datasets.query_crop_landcover.requests.get",
                   return_value=_mock_response(500)):
            result = _query_crop_landcover(BBOX, 2023)
        assert "CropScape API error" in result["message"]

    def test_empty_class_statistics(self):
        with patch("tools.datasets.query_crop_landcover.requests.get",
                   return_value=_mock_response(200, json_data={})):
            result = _query_crop_landcover(BBOX, 2023)
        assert "No CDL data" in result["message"]

    def test_zero_total_area(self):
        data = {"categoricalStatistics": {"classStatistics": [
            {"classCode": "1", "className": "Corn", "area": "0"}
        ]}}
        with patch("tools.datasets.query_crop_landcover.requests.get",
                   return_value=_mock_response(200, json_data=data)):
            result = _query_crop_landcover(BBOX, 2023)
        assert "zero total area" in result["message"]

    def test_strong_agriculture_signal(self):
        # 80 % cropland → strong signal
        data = {"categoricalStatistics": {"classStatistics": [
            {"classCode": "1", "className": "Corn", "area": "800"},
            {"classCode": "111", "className": "Open Water", "area": "200"},
        ]}}
        with patch("tools.datasets.query_crop_landcover.requests.get",
                   return_value=_mock_response(200, json_data=data)):
            result = _query_crop_landcover(BBOX, 2023)
        assert result["strong_agriculture_signal"] is True
        assert result["crop_fraction"] == pytest.approx(0.8, abs=0.01)

    def test_weak_agriculture_signal(self):
        # 10 % cropland → no strong signal
        data = {"categoricalStatistics": {"classStatistics": [
            {"classCode": "1", "className": "Corn", "area": "100"},
            {"classCode": "111", "className": "Open Water", "area": "900"},
        ]}}
        with patch("tools.datasets.query_crop_landcover.requests.get",
                   return_value=_mock_response(200, json_data=data)):
            result = _query_crop_landcover(BBOX, 2023)
        assert result["strong_agriculture_signal"] is False

    def test_year_none_defaults_to_last_year(self):
        with patch("tools.datasets.query_crop_landcover.requests.get",
                   return_value=_mock_response(200, json_data={})) as mock_get:
            _query_crop_landcover(BBOX, None)
        call_params = mock_get.call_args[1]["params"]
        assert call_params["year"] == datetime.now().year - 1

    def test_top_classes_capped_at_five(self):
        classes = [
            {"classCode": str(i), "className": f"Class{i}", "area": str(100 - i)}
            for i in range(1, 10)
        ]
        data = {"categoricalStatistics": {"classStatistics": classes}}
        with patch("tools.datasets.query_crop_landcover.requests.get",
                   return_value=_mock_response(200, json_data=data)):
            result = _query_crop_landcover(BBOX, 2023)
        assert len(result["top_classes"]) == 5


# ---------------------------------------------------------------------------
# QueryFireHistory
# ---------------------------------------------------------------------------

class TestQueryFireHistoryFunction:
    def test_api_error(self):
        with patch("tools.datasets.query_fire_history.requests.get",
                   return_value=_mock_response(500)):
            result = _query_fire_history(BBOX, "2020-01-01", "2021-12-31")
        assert result["intersections"] == 0
        assert "MTBS service unavailable" in result["message"]

    def test_no_features(self):
        with patch("tools.datasets.query_fire_history.requests.get",
                   return_value=_mock_response(200, json_data={"features": []})):
            result = _query_fire_history(BBOX, "2020-01-01", "2021-12-31")
        assert result["intersections"] == 0
        assert "No MTBS" in result["message"]

    def test_fires_found(self):
        features = [
            {"attributes": {"Incid_Name": "CAMP", "Year": 2020, "BurnBndAc": 1000}},
            {"attributes": {"Incid_Name": "DIXIE", "Year": 2021, "BurnBndAc": 2000}},
        ]
        with patch("tools.datasets.query_fire_history.requests.get",
                   return_value=_mock_response(200, json_data={"features": features})):
            result = _query_fire_history(BBOX, "2020-01-01", "2021-12-31")
        assert result["intersections"] == 2
        assert result["fires"][0]["Incid_Name"] == "CAMP"

    def test_year_range_sent_correctly(self):
        with patch("tools.datasets.query_fire_history.requests.get",
                   return_value=_mock_response(200, json_data={"features": []})) as mock_get:
            _query_fire_history(BBOX, "2019-06-01", "2021-03-15")
        where_clause = mock_get.call_args[1]["params"]["where"]
        assert "2019" in where_clause
        assert "2021" in where_clause


# ---------------------------------------------------------------------------
# GeoCodeLocation
# ---------------------------------------------------------------------------

class TestGeoCodeLocationFunction:
    def test_api_error(self):
        with patch("tools.geocode.geocode_location.requests.get",
                   return_value=_mock_response(500)), \
             patch("tools.geocode.geocode_location.time.sleep"):
            result = _geocode_location("some place")
        assert "Geocoding service unavailable" in result["message"]

    def test_no_results(self):
        with patch("tools.geocode.geocode_location.requests.get",
                   return_value=_mock_response(200, json_data=[])), \
             patch("tools.geocode.geocode_location.time.sleep"):
            result = _geocode_location("nonexistent place xyz")
        assert "No results" in result["message"]

    def test_single_result_returns_bbox_and_name(self):
        nominatim_results = [
            {"display_name": "Paris, France", "boundingbox": ["48.8", "49.0", "2.2", "2.5"]}
        ]
        with patch("tools.geocode.geocode_location.requests.get",
                   return_value=_mock_response(200, json_data=nominatim_results)), \
             patch("tools.geocode.geocode_location.time.sleep"):
            result = _geocode_location("Paris")
        assert result["display_name"] == "Paris, France"
        assert len(result["bbox"]) == 4
        assert result["message"] == "ok"

    def test_bbox_order_is_west_south_east_north(self):
        # Nominatim boundingbox: [south, north, west, east]
        nominatim_results = [
            {"display_name": "Test", "boundingbox": ["10.0", "20.0", "30.0", "40.0"]}
        ]
        with patch("tools.geocode.geocode_location.requests.get",
                   return_value=_mock_response(200, json_data=nominatim_results)), \
             patch("tools.geocode.geocode_location.time.sleep"):
            result = _geocode_location("Test")
        # expected: [west=30, south=10, east=40, north=20]
        assert result["bbox"] == [30.0, 10.0, 40.0, 20.0]

    def test_multiple_results_returns_candidates(self):
        nominatim_results = [
            {"display_name": f"Springfield {s}", "boundingbox": ["39.0", "40.0", "-90.0", "-89.0"]}
            for s in ("IL", "MO", "OH")
        ]
        with patch("tools.geocode.geocode_location.requests.get",
                   return_value=_mock_response(200, json_data=nominatim_results)), \
             patch("tools.geocode.geocode_location.time.sleep"):
            result = _geocode_location("Springfield")
        assert "candidates" in result
        assert len(result["candidates"]) == 3

    def test_sleep_always_called(self):
        with patch("tools.geocode.geocode_location.requests.get",
                   return_value=_mock_response(200, json_data=[])), \
             patch("tools.geocode.geocode_location.time.sleep") as mock_sleep:
            _geocode_location("anywhere")
        mock_sleep.assert_called_once_with(1)


# ---------------------------------------------------------------------------
# CheckHLSAvailability
# ---------------------------------------------------------------------------

class TestCheckHLSAvailabilityFunction:
    def test_earthdata_auth_failure(self):
        with patch("tools.hls.check_hls_availability.earthaccess.login",
                   side_effect=Exception("bad credentials")):
            result = _check_hls_availability(BBOX, "2024-01-01", "flood", None)
        assert result["available"] is False
        assert "Earthdata auth failed" in result["message"]

    def test_stac_catalog_unavailable(self):
        with patch("tools.hls.check_hls_availability.earthaccess.login"), \
             patch("tools.hls.check_hls_availability.Client.open",
                   side_effect=Exception("connection refused")):
            result = _check_hls_availability(BBOX, "2024-01-01", "flood", None)
        assert result["available"] is False
        assert "LP DAAC STAC unavailable" in result["message"]

    def test_no_imagery_found(self):
        mock_catalog = MagicMock()
        mock_catalog.search.return_value.items.return_value = []
        with patch("tools.hls.check_hls_availability.earthaccess.login"), \
             patch("tools.hls.check_hls_availability.Client.open",
                   return_value=mock_catalog):
            result = _check_hls_availability(BBOX, "2024-01-01", "flood", None)
        assert result["available"] is False
        assert "No HLS imagery" in result["message"]

    def test_imagery_found_returns_best_date(self):
        item = MagicMock()
        item.properties = {"eo:cloud_cover": 15}
        item.datetime.date.return_value = date(2024, 1, 5)
        mock_catalog = MagicMock()
        mock_catalog.search.return_value.items.return_value = [item]
        with patch("tools.hls.check_hls_availability.earthaccess.login"), \
             patch("tools.hls.check_hls_availability.Client.open",
                   return_value=mock_catalog):
            result = _check_hls_availability(BBOX, "2024-01-05", "flood", None)
        assert result["available"] is True
        assert result["selected_date"] == "2024-01-05"
        assert result["clear_pct"] == pytest.approx(85.0)

    def test_best_scene_is_clearest(self):
        def _item(cloud_cover, acq_date):
            m = MagicMock()
            m.properties = {"eo:cloud_cover": cloud_cover}
            m.datetime.date.return_value = acq_date
            return m

        items = [_item(50, date(2024, 1, 3)), _item(10, date(2024, 1, 5))]
        mock_catalog = MagicMock()
        mock_catalog.search.return_value.items.return_value = items
        with patch("tools.hls.check_hls_availability.earthaccess.login"), \
             patch("tools.hls.check_hls_availability.Client.open",
                   return_value=mock_catalog):
            result = _check_hls_availability(BBOX, "2024-01-05", "flood", None)
        assert result["clear_pct"] == pytest.approx(90.0)  # 100 - 10

    def test_crop_task_requires_date_range(self):
        with patch("tools.hls.check_hls_availability.earthaccess.login"), \
             patch("tools.hls.check_hls_availability.Client.open"):
            result = _check_hls_availability(BBOX, None, "crop", None)
        assert result["available"] is False
        assert "date_range required" in result["message"]


# ---------------------------------------------------------------------------
# QuerySurfaceWater
# ---------------------------------------------------------------------------

class TestQuerySurfaceWaterFunction:
    def test_earthdata_auth_failure(self):
        with patch("tools.datasets.query_surface_water.earthaccess.login",
                   side_effect=Exception("bad creds")):
            result = _query_surface_water(BBOX, "2024-01-01", "2024-01-31")
        assert result["signal"] is False
        assert "Earthdata auth failed" in result["message"]

    def test_stac_query_failure(self):
        with patch("tools.datasets.query_surface_water.earthaccess.login"), \
             patch("tools.datasets.query_surface_water.Client.open",
                   side_effect=Exception("STAC down")):
            result = _query_surface_water(BBOX, "2024-01-01", "2024-01-31")
        assert result["signal"] is False
        assert "CMR STAC query failed" in result["message"]

    def test_no_products_found(self):
        mock_catalog = MagicMock()
        mock_catalog.search.return_value.items.return_value = []
        with patch("tools.datasets.query_surface_water.earthaccess.login"), \
             patch("tools.datasets.query_surface_water.Client.open",
                   return_value=mock_catalog):
            result = _query_surface_water(BBOX, "2024-01-01", "2024-01-31")
        assert result["signal"] is False
        assert result["product_count"] == 0

    def test_products_found_returns_signal_and_dates(self):
        item1 = MagicMock()
        item1.datetime = datetime(2024, 1, 10)
        item2 = MagicMock()
        item2.datetime = datetime(2024, 1, 15)
        mock_catalog = MagicMock()
        mock_catalog.search.return_value.items.return_value = [item1, item2]
        with patch("tools.datasets.query_surface_water.earthaccess.login"), \
             patch("tools.datasets.query_surface_water.Client.open",
                   return_value=mock_catalog):
            result = _query_surface_water(BBOX, "2024-01-01", "2024-01-31")
        assert result["signal"] is True
        assert result["product_count"] == 2
        assert "2024-01-10" in result["dates_available"]
        assert "2024-01-15" in result["dates_available"]

    def test_duplicate_dates_deduplicated(self):
        item1 = MagicMock()
        item1.datetime = datetime(2024, 1, 10)
        item2 = MagicMock()
        item2.datetime = datetime(2024, 1, 10)  # same date, different tile
        mock_catalog = MagicMock()
        mock_catalog.search.return_value.items.return_value = [item1, item2]
        with patch("tools.datasets.query_surface_water.earthaccess.login"), \
             patch("tools.datasets.query_surface_water.Client.open",
                   return_value=mock_catalog):
            result = _query_surface_water(BBOX, "2024-01-01", "2024-01-31")
        assert result["dates_available"] == ["2024-01-10"]


# ---------------------------------------------------------------------------
# RunPrithviInference
# ---------------------------------------------------------------------------

class TestRunPrithviInferenceFunction:
    def test_invalid_task_type(self):
        result = _run_prithvi_inference("blizzard", BBOX, "2024-01-01", None, None)
        assert "Unsupported task_type" in result["message"]

    def test_flood_requires_date(self):
        result = _run_prithvi_inference("flood", BBOX, None, None, None)
        assert "'date' is required" in result["message"]

    def test_burn_requires_date(self):
        result = _run_prithvi_inference("burn", BBOX, None, None, None)
        assert "'date' is required" in result["message"]

    def test_crop_requires_date_range_and_dates(self):
        result = _run_prithvi_inference("crop", BBOX, None, None, None)
        assert "Crop task requires" in result["message"]

    def test_crop_requires_exactly_three_dates(self):
        date_range = {"start_date": "2024-01-01", "end_date": "2024-12-31"}
        result = _run_prithvi_inference("crop", BBOX, None, date_range, ["2024-03-01", "2024-06-01"])
        assert "Crop task requires" in result["message"]

    def test_flood_successful_submission(self):
        with patch("tools.prithvi.run_prithvi_inference.requests.post",
                   return_value=_mock_response(200, json_data={"job_id": "abc123", "status": "submitted", "message": "Job queued."})):
            result = _run_prithvi_inference("flood", BBOX, "2024-01-01", None, None)
        assert result["job_id"] == "abc123"
        assert result["status"] == "submitted"

    def test_crop_successful_submission(self):
        dates = ["2024-02-01", "2024-05-01", "2024-08-01"]
        date_range = {"start_date": "2024-01-01", "end_date": "2024-12-31"}
        with patch("tools.prithvi.run_prithvi_inference.requests.post",
                   return_value=_mock_response(200, json_data={"job_id": "xyz", "status": "submitted", "message": ""})):
            result = _run_prithvi_inference("crop", BBOX, None, date_range, dates)
        assert result["job_id"] == "xyz"

    def test_http_error(self):
        with patch("tools.prithvi.run_prithvi_inference.requests.post",
                   return_value=_mock_response(500)):
            result = _run_prithvi_inference("flood", BBOX, "2024-01-01", None, None)
        assert "Job submission failed" in result["message"]


# ---------------------------------------------------------------------------
# GetPrithviJobStatus
# ---------------------------------------------------------------------------

class TestGetPrithviJobStatusFunction:
    def test_http_error(self):
        with patch("tools.prithvi.get_prithvi_job_status.requests.get",
                   return_value=_mock_response(500)):
            result = _get_prithvi_job_status("job-123")
        assert result["status"] == "failed"
        assert "Status check failed" in result["message"]

    def test_running_status(self):
        with patch("tools.prithvi.get_prithvi_job_status.requests.get",
                   return_value=_mock_response(200, json_data={"status": "running", "message": ""})):
            result = _get_prithvi_job_status("job-123")
        assert result["status"] == "running"

    def test_finished_status(self):
        with patch("tools.prithvi.get_prithvi_job_status.requests.get",
                   return_value=_mock_response(200, json_data={"status": "finished", "message": ""})):
            result = _get_prithvi_job_status("job-123")
        assert result["status"] == "finished"

    def test_job_id_used_in_url(self):
        with patch("tools.prithvi.get_prithvi_job_status.requests.get",
                   return_value=_mock_response(200, json_data={"status": "running", "message": ""})) as mock_get:
            _get_prithvi_job_status("unique-job-id-99")
        called_url = mock_get.call_args[0][0]
        assert "unique-job-id-99" in called_url


# ---------------------------------------------------------------------------
# GetPrithviResults
# ---------------------------------------------------------------------------

class TestGetPrithviResultsFunction:
    def test_http_error(self):
        with patch("tools.prithvi.get_prithvi_results.requests.get",
                   return_value=_mock_response(500)):
            result = _get_prithvi_results("job-123")
        assert "Results retrieval failed" in result["message"]

    def test_flood_results(self):
        payload = {
            "task_type": "flood",
            "result_urls": ["/outputs/flood_abc.tif"],
            "summary": {"area_hectares": 42.5},
            "message": "Flood detection complete.",
        }
        with patch("tools.prithvi.get_prithvi_results.requests.get",
                   return_value=_mock_response(200, json_data=payload)):
            result = _get_prithvi_results("job-123")
        assert result["task_type"] == "flood"
        assert result["summary"]["area_hectares"] == 42.5
        assert len(result["result_urls"]) == 1

    def test_crop_results(self):
        payload = {
            "task_type": "crop",
            "result_urls": ["/outputs/crop_abc.tif"],
            "summary": {"area_hectares": 100.0, "per_class_hectares": {"Corn": 60.0}},
            "message": "Crop classification complete.",
        }
        with patch("tools.prithvi.get_prithvi_results.requests.get",
                   return_value=_mock_response(200, json_data=payload)):
            result = _get_prithvi_results("job-456")
        assert result["task_type"] == "crop"
        assert result["summary"]["per_class_hectares"]["Corn"] == 60.0

    def test_job_id_used_in_url(self):
        with patch("tools.prithvi.get_prithvi_results.requests.get",
                   return_value=_mock_response(200, json_data={})) as mock_get:
            _get_prithvi_results("special-job-77")
        called_url = mock_get.call_args[0][0]
        assert "special-job-77" in called_url
