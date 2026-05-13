import os
from io import StringIO

import pandas as pd
import requests

FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"


def query_active_fires(bbox: list[float], start_date: str, end_date: str) -> dict:
    """Query FIRMS VIIRS_SNPP_NRT for high-confidence fire detections.

    bbox: [west, south, east, north]
    Signal rule (from reasoning.md): confidence == 'high' (VIIRS string field).
    FIRMS NRT covers ~3 months. For older events use the FIRMS archive API.
    Requires FIRMS_MAP_KEY env var (free from firms.modaps.eosdis.nasa.gov/api/).
    """
    map_key = os.environ.get("FIRMS_MAP_KEY")
    if not map_key:
        return {"detections": False, "count": 0, "message": "FIRMS_MAP_KEY env var not set."}

    bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    # FIRMS area API: max 10-day window per request; use start_date as anchor
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

    # VIIRS confidence is a string: 'low' | 'nominal' | 'high'
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
