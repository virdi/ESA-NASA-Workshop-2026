import earthaccess
from pystac_client import Client

CMR_STAC_URL = "https://cmr.earthdata.nasa.gov/stac/POCLOUD"
COLLECTION = "OPERA_L3_DSWx-HLS_V1"


def query_surface_water(bbox: list[float], start_date: str, end_date: str) -> dict:
    """Query OPERA DSWx-HLS for surface water / flooding signals.

    bbox: [west, south, east, north]
    Signal rule (from reasoning.md): product existence indicates potential water
    extent. Pixel-level new-water comparison requires downloading WTR layers
    and is out of scope for this signal tool.
    Requires Earthdata credentials in ~/.netrc.
    """
    try:
        earthaccess.login(strategy="netrc")
    except Exception as e:
        return {"signal": False, "message": f"Earthdata auth failed: {e}"}

    try:
        catalog = Client.open(CMR_STAC_URL)
        search = catalog.search(
            collections=[COLLECTION],
            bbox=bbox,
            datetime=f"{start_date}/{end_date}",
            max_items=100,
        )
        items = list(search.items())
    except Exception as e:
        return {"signal": False, "message": f"CMR STAC query failed: {e}"}

    if not items:
        return {
            "signal": False,
            "product_count": 0,
            "message": "No DSWx products found for this area and date range.",
        }

    dates = sorted(
        set(item.datetime.strftime("%Y-%m-%d") for item in items if item.datetime)
    )
    return {
        "signal": True,
        "product_count": len(items),
        "dates_available": dates,
        "message": f"Found {len(items)} DSWx product(s) across {len(dates)} date(s).",
    }
