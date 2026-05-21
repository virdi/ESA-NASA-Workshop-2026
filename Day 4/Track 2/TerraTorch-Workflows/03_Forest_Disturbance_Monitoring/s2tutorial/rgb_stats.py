"""Dataset-wide RGB normalisation stats (B02 / B03 / B04 percentiles).

The default ``ts.rgb()`` stretch reads these stats from
``audit/rgb_stats.json`` to make figures consistent across frames: cloudy
/ snowy frames don't blow out the gammut, dim winter frames don't get
crushed.

Run once::

    uv run python -m s2tutorial.rgb_stats data --out audit/rgb_stats.json

By default the scan samples ~5 frames per sample, masks to SCL classes 4
(vegetation) and 5 (bare-soil) — the strict-land subset — and reports
the 2nd / 98th percentile per band over the ~196k retained 10 m pixels.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import zarr

from . import _paths as _p

# SCL classes considered "strict land" — drops clouds, snow, shadow,
# water, and the catch-all unclassified bucket.
LAND_SCL_CLASSES: tuple[int, ...] = (4, 5)

# Crop that matches the TerraMind ViT input on the 10 m (252, 252) grid.
# Stats are computed over this crop only.
_CROP_10M = (slice(6, 230), slice(6, 230))

# SCL is at 20 m → half the 10 m index range.
_CROP_20M = (slice(3, 115), slice(3, 115))

_RGB_BANDS = ("B02", "B03", "B04")
_RGB_BAND_IDX = {"B02": 0, "B03": 1, "B04": 2}


def _upsample_scl_to_10m(scl: np.ndarray) -> np.ndarray:
    """SCL is 126×126 (20 m); RGB is 252×252 (10 m). Repeat each SCL
    pixel 2× to match the 10 m grid, then crop the same window."""
    if scl.ndim != 2:
        raise ValueError(f"scl expected (H, W), got {scl.shape}")
    return scl.repeat(2, axis=0).repeat(2, axis=1)


def compute_rgb_stats(
    data_root: str | Path,
    *,
    n_frames_per_sid: int = 5,
    percentiles: tuple[float, float] = (2.0, 98.0),
    rng_seed: int = 0,
    verbose: bool = True,
) -> dict[str, tuple[float, float]]:
    """Stratified land-only percentile scan over the dataset.

    Returns ``{"B02": (p_lo, p_hi), "B03": ..., "B04": ...}``. Per-band
    physical reflectance (BOA_ADD_OFFSET handled). The crop is the same
    ``[6:230, 6:230]`` window the FM ingests, so the stats match the
    region notebooks display via ``rgb_centered_224``.
    """
    root = Path(data_root)
    sid_dirs = sorted(
        int(p.name.removesuffix(".zarr"))
        for p in (root / "patches").iterdir() if p.suffix == ".zarr"
    )
    rng = np.random.default_rng(rng_seed)
    t0 = time.time()
    per_band_values: dict[str, list[np.ndarray]] = {b: [] for b in _RGB_BANDS}

    for k, sid in enumerate(sid_dirs):
        g = zarr.open_group(str(root / "patches" / f"{sid}.zarr"), mode="r")
        T = g["s2_10m"].shape[0]
        n = min(n_frames_per_sid, T)
        frame_idx = rng.choice(T, size=n, replace=False)
        for fi in sorted(int(i) for i in frame_idx):
            s10 = np.asarray(g["s2_10m"][fi]).astype(np.float32)  # (4, 252, 252)
            valid_dn = s10 > 0
            s10[valid_dn] = (s10[valid_dn] - 1000.0) / 10000.0
            s10[~valid_dn] = np.nan  # NODATA out of percentile inputs

            scl = np.asarray(g["s2_scl"][fi]).astype(np.uint8)  # (126, 126)
            scl_10m = _upsample_scl_to_10m(scl)  # (252, 252)

            crop_band0 = s10[0][_CROP_10M]  # any band, just to get shape
            land = np.isin(scl_10m[_CROP_10M], LAND_SCL_CLASSES)
            land &= np.isfinite(crop_band0)
            if not land.any():
                continue
            for band in _RGB_BANDS:
                arr = s10[_RGB_BAND_IDX[band]][_CROP_10M]
                per_band_values[band].append(arr[land])

        if verbose and (k + 1) % 40 == 0:
            print(
                f"  scanned {k + 1}/{len(sid_dirs)} sids "
                f"({time.time() - t0:.1f}s)"
            )

    out: dict[str, tuple[float, float]] = {}
    for band, chunks in per_band_values.items():
        if not chunks:
            raise RuntimeError(f"no land pixels found for band {band}")
        all_vals = np.concatenate(chunks)
        lo = float(np.nanpercentile(all_vals, percentiles[0]))
        hi = float(np.nanpercentile(all_vals, percentiles[1]))
        out[band] = (lo, hi)
        if verbose:
            print(
                f"  {band}: n_pixels={all_vals.size}, "
                f"p{percentiles[0]:g}={lo:.4f}, p{percentiles[1]:g}={hi:.4f}"
            )

    if verbose:
        print(f"compute_rgb_stats: {time.time() - t0:.1f}s total")
    return out


def save_rgb_stats(
    stats: dict[str, tuple[float, float]],
    out_path: str | Path,
    *,
    extra_meta: dict | None = None,
) -> None:
    """Write per-band stats to JSON. ``extra_meta`` is merged at the
    top level — useful to record percentile choice, SCL classes, etc."""
    payload: dict[str, object] = {b: [float(lo), float(hi)]
                                  for b, (lo, hi) in stats.items()}
    if extra_meta:
        payload["_meta"] = extra_meta
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload, indent=2) + "\n")


def load_rgb_stats(audit_dir: str | Path) -> dict[str, tuple[float, float]]:
    """Load per-band stats from ``audit/rgb_stats.json``.

    Accepts a local path or an ``s3://`` URI.
    """
    path = _p.child(audit_dir, "rgb_stats.json")
    if not _p.exists(path):
        raise FileNotFoundError(path)
    payload = _p.load_json(path)
    return {b: tuple(payload[b]) for b in _RGB_BANDS}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("root", type=Path, help="data root (contains patches/)")
    p.add_argument("--out", type=Path,
                   default=Path("audit/rgb_stats.json"))
    p.add_argument("--n-frames-per-sid", type=int, default=5)
    p.add_argument("--p-lo", type=float, default=2.0)
    p.add_argument("--p-hi", type=float, default=98.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    stats = compute_rgb_stats(
        args.root,
        n_frames_per_sid=args.n_frames_per_sid,
        percentiles=(args.p_lo, args.p_hi),
        rng_seed=args.seed,
    )
    save_rgb_stats(stats, args.out, extra_meta={
        "source": str(args.root),
        "n_frames_per_sid": args.n_frames_per_sid,
        "percentiles": [args.p_lo, args.p_hi],
        "scl_land_classes": list(LAND_SCL_CLASSES),
        "crop": "10m[6:230, 6:230]",
        "seed": args.seed,
    })
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
