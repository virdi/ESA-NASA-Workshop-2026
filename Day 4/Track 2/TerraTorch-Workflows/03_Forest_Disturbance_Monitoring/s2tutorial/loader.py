"""Standalone loader for the workshop S2 time-series subset.

Layout (one tree, no per-tile splits):

    {root}/
      samples.parquet     sample_id, original_sample_id, window_start, window_end
      labels.parquet      sample_id, period_idx, label, start,
                          end_evidence, end_validity, is_event
      frames.parquet      sample_id, date  (one row per stored frame)
      classes.json        int code -> human-readable name
      patches/{sample_id}.zarr/
        group attrs       {"sample_id": int}
        s2_10m            (T, 4, 252, 252) uint16  B02 B03 B04 B08
        s2_20m            (T, 6, 126, 126) uint16  B05 B06 B07 B8A B11 B12
        s2_60m            (T, 2,  42,  42) uint16  B01 B09
        s2_scl            (T,    126, 126) uint8   Scene Classification
        each sensor array attrs: {"dates": [ISO str, ...],
                                  "band_order": [str, ...]   (omitted on s2_scl)}

Invariants the loader trusts (asserted on first open of each store):

1. all four sensor arrays share the same `attrs["dates"]` list;
2. that list is calendar-sorted ascending and free of duplicates;
3. `len(dates) == arr.shape[0]` for every sensor;
4. that list equals `frames.parquet[sample_id == sid]["date"]`;
5. `group.attrs["sample_id"]` matches the integer in the zarr filename.

Reflectance is stored as the **raw uint16 from Sentinel-2 Processing
Baseline 04.00+ products**, i.e. `DN = reflectance * 10000 + 1000`
(ESA added the `BOA_ADD_OFFSET = -1000` convention in Jan 2022 to
allow negative reflectance to survive as a positive integer; all dates
in this subset are post-baseline-04.00 reprocessed). SCL is `uint8`
with the standard Sen2Cor codes (unaffected by the offset).

To recover physical reflectance the loader does
`reflectance = (DN - 1000) / 10000` whenever you call
`as_reflectance=True`. NODATA (DN == 0) stays NODATA (we clamp before
the offset).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import polars as pl
import zarr


_S2_BANDS = {
    "s2_10m": ("B02", "B03", "B04", "B08"),
    "s2_20m": ("B05", "B06", "B07", "B8A", "B11", "B12"),
    "s2_60m": ("B01", "B09"),
}


def load_metadata(root: str | Path) -> dict:
    """Load the four metadata artefacts as a single dict."""
    root = Path(root)
    classes = {
        int(k): v for k, v in json.loads((root / "classes.json").read_text()).items()
    }
    return {
        "samples": pl.read_parquet(root / "samples.parquet"),
        "labels": pl.read_parquet(root / "labels.parquet"),
        "frames": pl.read_parquet(root / "frames.parquet"),
        "classes": classes,
        "root": root,
    }


def decode_label(code: int, classes: dict[int, str]) -> str:
    """Return the human-readable name for a label code."""
    return classes[int(code)]


@dataclass
class TimeSeries:
    """One sample's full time series of cloud-free S2 patches and labels.

    Created by `get_sample(root, sample_id)` or yielded by
    `iter_samples(root)`. Lazy: opens the zarr store on construction
    but reads pixel data only on indexing.
    """

    sample_id: int
    dates: list[str]              # ISO strings, calendar-sorted
    window_start: str
    window_end: str
    labels: pl.DataFrame          # full period chain for this sid
    _grp: zarr.Group
    _root: Path

    def __len__(self) -> int:
        return len(self.dates)

    @property
    def n_frames(self) -> int:
        return len(self.dates)

    @staticmethod
    def _to_reflectance(arr: np.ndarray) -> np.ndarray:
        """Apply the BOA_ADD_OFFSET = -1000 convention then divide by 10000.

        Only valid pixels get shifted; NODATA (DN == 0) stays 0.
        """
        f = arr.astype(np.float32)
        valid = f > 0
        f[valid] = (f[valid] - 1000.0) / 10000.0
        return f

    def s2_10m(self, i: int, as_reflectance: bool = False) -> np.ndarray:
        """B02/B03/B04/B08 at 10 m. Shape (4, 252, 252)."""
        arr = np.asarray(self._grp["s2_10m"][i])
        return self._to_reflectance(arr) if as_reflectance else arr

    def s2_20m(self, i: int, as_reflectance: bool = False) -> np.ndarray:
        """B05/B06/B07/B8A/B11/B12 at 20 m. Shape (6, 126, 126)."""
        arr = np.asarray(self._grp["s2_20m"][i])
        return self._to_reflectance(arr) if as_reflectance else arr

    def s2_60m(self, i: int, as_reflectance: bool = False) -> np.ndarray:
        """B01/B09 at 60 m. Shape (2, 42, 42)."""
        arr = np.asarray(self._grp["s2_60m"][i])
        return self._to_reflectance(arr) if as_reflectance else arr

    def s2_scl(self, i: int) -> np.ndarray:
        """Scene Classification mask at 20 m. Shape (126, 126)."""
        return np.asarray(self._grp["s2_scl"][i])

    def rgb(self, i: int, gamma: float = 1.0,
            stretch: tuple[float, float] = (0.0, 0.3)) -> np.ndarray:
        """Convenience: HxWx3 RGB in [0,1] from B04/B03/B02.

        Uses physical reflectance under the hood (BOA_ADD_OFFSET handled).
        Default `stretch=(0.0, 0.3)` works for vegetation surfaces.
        """
        s10 = self.s2_10m(i, as_reflectance=True)
        rgb = np.stack([s10[2], s10[1], s10[0]], axis=-1)
        lo, hi = stretch
        rgb = np.clip((rgb - lo) / (hi - lo), 0.0, 1.0)
        if gamma != 1.0:
            rgb = rgb ** (1.0 / gamma)
        return rgb

    def label_at(self, date: str) -> int | None:
        """Return the label code active on `date` (YYYY-MM-DD).

        Selects the most recent period whose `start <= date`. Returns
        None if `date` precedes every period start (shouldn't happen
        for stored frames).
        """
        rows = self.labels.filter(pl.col("start") <= pl.lit(date).str.to_date())
        if rows.is_empty():
            return None
        return int(rows.sort("start").tail(1)["label"].item())

    def event_period(self) -> dict | None:
        """Return the single `is_event=True` period as a dict, or None."""
        ev = self.labels.filter(pl.col("is_event"))
        if ev.is_empty():
            return None
        if len(ev) > 1:
            raise ValueError(
                f"sample_id={self.sample_id} has {len(ev)} event rows; "
                f"this subset is curated for exactly one"
            )
        return ev.row(0, named=True)


def _open_and_validate(root: Path, sample_id: int,
                       frames: pl.DataFrame) -> tuple[zarr.Group, list[str]]:
    """Open the zarr store and check the date-mapping invariants."""
    store = root / "patches" / f"{sample_id}.zarr"
    if not store.exists():
        raise FileNotFoundError(store)
    grp = zarr.open_group(str(store), mode="r")

    grp_sid = int(grp.attrs.get("sample_id", -1))
    if grp_sid != sample_id:
        raise ValueError(
            f"{store}: group attrs sample_id={grp_sid} != filename {sample_id}"
        )

    dates_per_sensor = {
        s: list(grp[s].attrs.get("dates") or []) for s in _S2_BANDS
    }
    dates_per_sensor["s2_scl"] = list(grp["s2_scl"].attrs.get("dates") or [])
    ref = dates_per_sensor["s2_10m"]
    for sensor, dates in dates_per_sensor.items():
        if dates != ref:
            raise ValueError(
                f"{store}: {sensor}.attrs['dates'] disagrees with s2_10m"
            )
        if grp[sensor].shape[0] != len(dates):
            raise ValueError(
                f"{store}: {sensor}.shape[0]={grp[sensor].shape[0]} "
                f"!= len(dates)={len(dates)}"
            )
    if sorted(ref) != ref:
        raise ValueError(f"{store}: dates not calendar-sorted")
    if len(set(ref)) != len(ref):
        raise ValueError(f"{store}: duplicate dates")

    pq_dates = (
        frames.filter(pl.col("sample_id") == sample_id)
        .sort("date")["date"]
        .to_list()
    )
    pq_dates = [d.isoformat() for d in pq_dates]
    if pq_dates != ref:
        raise ValueError(
            f"{store}: frames.parquet dates ({len(pq_dates)}) disagree with "
            f"zarr dates ({len(ref)}) — first diff index "
            f"{next((i for i in range(min(len(pq_dates), len(ref))) if pq_dates[i] != ref[i]), 'tail')}"
        )

    return grp, ref


def get_sample(root: str | Path, sample_id: int,
               metadata: dict | None = None) -> TimeSeries:
    """Open one sample's time series."""
    root = Path(root)
    md = metadata or load_metadata(root)
    sample_row = md["samples"].filter(pl.col("sample_id") == sample_id)
    if sample_row.is_empty():
        raise KeyError(f"sample_id={sample_id} not in samples.parquet")
    sample_row = sample_row.row(0, named=True)

    grp, dates = _open_and_validate(root, sample_id, md["frames"])
    labels = (
        md["labels"]
        .filter(pl.col("sample_id") == sample_id)
        .sort("period_idx")
    )
    return TimeSeries(
        sample_id=sample_id,
        dates=dates,
        window_start=sample_row["window_start"].isoformat(),
        window_end=sample_row["window_end"].isoformat(),
        labels=labels,
        _grp=grp,
        _root=root,
    )


def iter_samples(root: str | Path) -> Iterator[TimeSeries]:
    """Yield all samples in `samples.parquet` order."""
    md = load_metadata(root)
    for sid in md["samples"]["sample_id"].to_list():
        yield get_sample(root, int(sid), metadata=md)
