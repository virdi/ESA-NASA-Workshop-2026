import time
import requests

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
HEADERS = {"User-Agent": "prithvi-workshop-agent/1.0"}


def geocode_location(query: str) -> dict:
    """Convert a place name or description to a bounding box [west, south, east, north].

    Returns a single bbox when unambiguous, or a candidates list when multiple
    matches are found. Nominatim rate limit: 1 req/sec enforced by sleep.
    """
    params = {"q": query, "format": "json", "limit": 5}
    try:
        resp = requests.get(NOMINATIM_URL, params=params, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        results = resp.json()
    except requests.RequestException as e:
        return {"message": f"Geocoding service unavailable: {e}"}
    finally:
        time.sleep(1)  # Nominatim enforces 1 req/sec

    if not results:
        return {
            "message": f"No results for '{query}'. Try rephrasing or provide coordinates."
        }

    def _parse_bbox(r: dict) -> list:
        # Nominatim boundingbox: [south, north, west, east] → reorder to [west, south, east, north]
        bb = r["boundingbox"]
        return [float(bb[2]), float(bb[0]), float(bb[3]), float(bb[1])]

    if len(results) == 1:
        return {
            "bbox": _parse_bbox(results[0]),
            "display_name": results[0]["display_name"],
            "message": "ok",
        }

    candidates = [
        {"display_name": r["display_name"], "bbox": _parse_bbox(r)}
        for r in results[:3]
    ]
    return {
        "candidates": candidates,
        "message": "Multiple matches found. Please choose one.",
    }
