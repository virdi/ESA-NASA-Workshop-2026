from datetime import datetime

import requests

CROPSCAPE_URL = "https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLStat"

# USDA CDL crop class codes (major cropland categories; see NASS CDL documentation)
CROPLAND_CLASS_CODES = {
    1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 21, 22, 23, 24, 25, 26, 27,
    28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 66,
    67, 68, 69, 72, 74, 75, 76, 77,
}


def query_crop_landcover(bbox: list[float], year: int | None = None) -> dict:
    """Query USDA CDL for crop/landcover class distribution in a bbox.

    bbox: [west, south, east, north]
    Signal rule (from reasoning.md): strong agriculture signal if crop_fraction > 0.30.
    CDL covers CONUS only and typically lags ~1 year.
    """
    if year is None:
        year = datetime.now().year - 1

    params = {
        "year": year,
        "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "format": "json",
    }

    try:
        resp = requests.get(CROPSCAPE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        return {"message": f"CropScape API error: {e}"}
    except ValueError as e:
        return {"message": f"CropScape response parse error: {e}"}

    # Response schema varies slightly by endpoint version; try both paths
    classes = (
        data.get("categoricalStatistics", {}).get("classStatistics", [])
        or data.get("data", {}).get("classStatistics", [])
    )

    if not classes:
        return {"message": "No CDL data returned. CDL covers CONUS only."}

    total_area = sum(float(c.get("area", 0)) for c in classes)
    if total_area == 0:
        return {"message": "CDL returned zero total area for this bbox."}

    crop_area = sum(
        float(c.get("area", 0))
        for c in classes
        if int(c.get("classCode", 0)) in CROPLAND_CLASS_CODES
    )
    crop_fraction = crop_area / total_area

    top_classes = sorted(classes, key=lambda x: -float(x.get("area", 0)))[:5]

    return {
        "year": year,
        "crop_fraction": round(crop_fraction, 3),
        "strong_agriculture_signal": crop_fraction > 0.30,
        "top_classes": [
            {
                "class_code": c.get("classCode"),
                "class_name": c.get("className"),
                "area": float(c.get("area", 0)),
            }
            for c in top_classes
        ],
        "message": f"{crop_fraction:.0%} of bbox is cropland (CDL {year}).",
    }
