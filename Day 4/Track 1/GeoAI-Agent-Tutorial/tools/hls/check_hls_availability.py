from datetime import datetime, timedelta

import earthaccess
from pystac_client import Client

LPCLOUD_URL = "https://cmr.earthdata.nasa.gov/stac/LPCLOUD"
HLS_COLLECTIONS = ["HLSS30.v2.0", "HLSL30.v2.0"]


def check_hls_availability(
    bbox: list[float],
    date: str,
    task_type: str,
    date_range: dict | None = None,
) -> dict:
    """Check HLS imagery availability for a bbox and task.

    bbox: [west, south, east, north]
    task_type: 'flood' | 'burn' | 'crop'
    date_range: {'start_date': 'YYYY-MM-DD', 'end_date': 'YYYY-MM-DD'} (crop only)

    Search windows per reasoning.md:
      flood: exact date ± 3 days, best clear_pct wins
      burn:  requested date through +30 days
      crop:  3 clean dates with ≥70-day gaps inside date_range (relaxes to 50/50 if needed)

    clear_pct derived from eo:cloud_cover metadata (proxy for Fmask; see hls_conventions.md).
    Requires Earthdata credentials in ~/.netrc.
    """
    try:
        earthaccess.login(strategy="netrc")
    except Exception as e:
        return {"available": False, "message": f"Earthdata auth failed: {e}"}

    try:
        catalog = Client.open(LPCLOUD_URL)
    except Exception as e:
        return {"available": False, "message": f"LP DAAC STAC unavailable: {e}"}

    if task_type == "crop":
        if not date_range:
            return {"available": False, "message": "date_range required for crop task."}
        return _check_crop_dates(catalog, bbox, date_range)

    return _check_single_date(catalog, bbox, date, task_type)


def _check_single_date(catalog, bbox: list, date: str, task_type: str) -> dict:
    dt = datetime.strptime(date, "%Y-%m-%d")
    before_days = 3 if task_type == "flood" else 0
    after_days = 3 if task_type == "flood" else 30

    search_start = (dt - timedelta(days=before_days)).strftime("%Y-%m-%d")
    search_end = (dt + timedelta(days=after_days)).strftime("%Y-%m-%d")

    best = None
    alternatives = []

    for collection in HLS_COLLECTIONS:
        try:
            items = list(
                catalog.search(
                    collections=[collection],
                    bbox=bbox,
                    datetime=f"{search_start}/{search_end}",
                    max_items=50,
                ).items()
            )
        except Exception:
            continue

        for item in items:
            cloud_cover = item.properties.get("eo:cloud_cover", 100)
            clear_pct = round(100.0 - float(cloud_cover), 1)
            item_date = item.datetime.date()
            offset = abs((item_date - dt.date()).days)
            entry = {
                "date": item_date.strftime("%Y-%m-%d"),
                "collection": collection.split(".")[0],
                "clear_pct": clear_pct,
                "offset_days": offset,
            }
            if best is None or clear_pct > best["clear_pct"]:
                if best is not None:
                    alternatives.append(best)
                best = entry
            else:
                alternatives.append(entry)

    if not best:
        return {
            "available": False,
            "message": "No HLS imagery found in the search window.",
        }

    return {
        "available": True,
        **best,
        "alternatives": alternatives[:3],
        "message": "ok",
    }


def _check_crop_dates(catalog, bbox: list, date_range: dict) -> dict:
    start = date_range["start_date"]
    end = date_range["end_date"]

    all_scenes = []
    for collection in HLS_COLLECTIONS:
        try:
            items = list(
                catalog.search(
                    collections=[collection],
                    bbox=bbox,
                    datetime=f"{start}/{end}",
                    max_items=300,
                ).items()
            )
        except Exception:
            continue

        for item in items:
            if item.datetime is None:
                continue
            cloud_cover = item.properties.get("eo:cloud_cover", 100)
            clear_pct = round(100.0 - float(cloud_cover), 1)
            all_scenes.append(
                {
                    "date": item.datetime.date(),
                    "clear_pct": clear_pct,
                    "collection": collection.split(".")[0],
                }
            )

    all_scenes.sort(key=lambda x: x["date"])

    # Try strict thresholds first (70% clear, 70-day gap), then relax to 50/50
    for min_clear, min_gap in [(70, 70), (50, 50)]:
        clean = [s for s in all_scenes if s["clear_pct"] >= min_clear]
        selected = _pick_three_with_gaps(clean, min_gap)
        if selected:
            return {
                "available": True,
                "crop_dates": [s["date"].strftime("%Y-%m-%d") for s in selected],
                "clear_pcts": [s["clear_pct"] for s in selected],
                "collections": [s["collection"] for s in selected],
                "relaxed_thresholds": min_clear < 70,
                "message": "ok",
            }

    return {
        "available": False,
        "message": "Cannot find 3 clean dates with required temporal gaps in date range.",
    }


def _pick_three_with_gaps(scenes: list, min_gap_days: int) -> list:
    for i, s1 in enumerate(scenes):
        for j, s2 in enumerate(scenes[i + 1 :], i + 1):
            if (s2["date"] - s1["date"]).days < min_gap_days:
                continue
            for s3 in scenes[j + 1 :]:
                if (s3["date"] - s2["date"]).days >= min_gap_days:
                    return [s1, s2, s3]
    return None
