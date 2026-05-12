"""Standalone visual test for the flip processor — no vLLM required.

Loads a GeoTIFF, applies the same `_apply_flip` function the processor uses
in production, and writes flipped copies next to the input. Open the outputs
in QGIS / Preview / any GeoTIFF viewer to visually confirm that the flip
directions are correct.

Usage:
    python test_flip.py <path/to/image.tif> [--out-dir DIR]

If no path is given, falls back to a sample image shipped with the workshop.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio

from terratorch_flip_processor.flip_processor import _apply_flip
from terratorch_flip_processor.types import FlipRequestData

DEFAULT_SAMPLE = Path(__file__).resolve().parents[1] / "3_vllm_inference" / "Spain_7370579_S2Hand.tif"


def flip_geotiff(src_path: Path, dst_path: Path, flip_h: bool, flip_v: bool) -> None:
    """Read src_path, apply the flip, write to dst_path preserving georef metadata."""
    with rasterio.open(src_path) as src:
        data = src.read()  # shape: (bands, H, W)
        meta = src.meta.copy()

    flipped = _apply_flip(data, flip_h=flip_h, flip_v=flip_v)

    with rasterio.open(dst_path, "w", **meta) as dst:
        dst.write(flipped)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        default=DEFAULT_SAMPLE,
        help=f"Input GeoTIFF (default: {DEFAULT_SAMPLE})",
    )
    parser.add_argument("--out-dir", type=Path, default=Path.cwd(), help="Where to write flipped copies (default: cwd)")
    args = parser.parse_args()

    if not args.image.exists():
        parser.error(f"Input file not found: {args.image}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Schema sanity check — the same pydantic model vLLM would build from the request body
    request = FlipRequestData(
        data=str(args.image),
        data_format="path",
        out_data_format="path",
        flip_horizontal=True,
        flip_vertical=True,
    )
    print(f"Parsed request OK: flip_horizontal={request.flip_horizontal}, flip_vertical={request.flip_vertical}")

    stem = args.image.stem
    cases = [
        ("flip_h", True, False),
        ("flip_v", False, True),
        ("flip_hv", True, True),
    ]

    print(f"\nInput:  {args.image}")
    for suffix, flip_h, flip_v in cases:
        out = args.out_dir / f"{stem}__{suffix}.tif"
        flip_geotiff(args.image, out, flip_h=flip_h, flip_v=flip_v)
        print(f"Wrote:  {out}  (flip_horizontal={flip_h}, flip_vertical={flip_v})")

    # Numeric sanity check: flipping twice should round-trip to the original
    with rasterio.open(args.image) as src:
        original = src.read()
    round_trip = _apply_flip(_apply_flip(original, True, True), True, True)
    assert np.array_equal(original, round_trip), "Double flip did not round-trip — check axis logic"
    print("\nDouble-flip round-trip: OK")


if __name__ == "__main__":
    main()
