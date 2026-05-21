"""Standalone loader for the workshop S2 time-series subset.

Layout (one tree, no per-tile splits):

    {root}/
      samples.parquet     sample_id, original_sample_id, window_start, window_end
      labels.parquet      sample_id, period_idx, label, start,
                          end_evidence, end_validity, is_event
      frames.parquet      sample_id, date  (one row per stored frame)
      splits.parquet      (optional) sample_id, split — train/val/test
      classes.json        int code -> human-readable name
      patches/{sample_id}.zarr/
        group attrs       {"sample_id": int}
        s2_10m            (T, 4, 252, 252) uint16  B02 B03 B04 B08
        s2_20m            (T, 6, 126, 126) uint16  B05 B06 B07 B8A B11 B12
        s2_60m            (T, 2,  42,  42) uint16  B01 B09
        s2_scl            (T,    126, 126) uint8   Scene Classification
        each sensor array attrs: {"dates": [ISO str, ...],
                                  "band_order": [str, ...]   (omitted on s2_scl)}

Frozen-FM embeddings live in a *separate* tree alongside `{root}/`, with one
zarr per sample. Two variants: a dense token grid and the single labelled
centre token. Default layout (env-overridable, matches the colleague's
TerraMind generation pipeline):

    {root}/../embeddings/terramind_v1_small/layer_00/{sample_id}.zarr/
      embedding           (T, 196, 384) float32 — 14×14 row-major tokens
      group attrs         {"sample_id": int}
      embedding attrs     {"dates": [ISO str, ...]} matching frames.parquet

    {root}/../embeddings_center_patch/terramind_v1_small/layer_00/{sample_id}.zarr/
      embedding           (T, 384)      float32 — the centre token only
      same attrs

Override via the env vars `S2T_EMBEDDING_ROOT` (dense) and
`S2T_EMBEDDING_PATCH_CENTER_ROOT` (centre patch).

Invariants the loader trusts (asserted on first open of each patches/
store, plus lazily on first embedding access):

1. all sensor arrays in patches/ share the same `attrs["dates"]` list;
2. that list is calendar-sorted ascending and free of duplicates;
3. `len(dates) == arr.shape[0]` for every array;
4. that list equals `frames.parquet[sample_id == sid]["date"]`;
5. `group.attrs["sample_id"]` matches the integer in the zarr filename;
6. **embedding stores**: when an embedding accessor is first called for
   a sample, its `attrs["dates"]` is checked against patches' dates and
   its shape's last-dim is verified.

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

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import polars as pl
import zarr

from . import _paths as _p


_S2_BANDS = {
    "s2_10m": ("B02", "B03", "B04", "B08"),
    "s2_20m": ("B05", "B06", "B07", "B8A", "B11", "B12"),
    "s2_60m": ("B01", "B09"),
}

# Frozen-FM embedding contract. Real TerraMind-small embeddings live in
# external trees alongside `data/`. See module docstring for layout.
_EMB_NAME = "embedding"           # array name inside each embedding zarr
_EMB_TOKEN_GRID = (14, 14)
_EMB_DIM = 384                    # TerraMind-small; TerraMind-base would be 768

_DEFAULT_EMB_REL = Path("embeddings") / "terramind_v1_small" / "layer_00"
_DEFAULT_EMB_CENTER_REL = (
    Path("embeddings_center_patch") / "terramind_v1_small" / "layer_00"
)

# Crop that matches the TerraMind ViT input. Applied to (252, 252) reflectance
# arrays. With this crop, 10 m pixel (x, y) aligns with ViT token (x // 16,
# y // 16) of the 14×14 grid.
RGB_CROP_224 = (slice(6, 230), slice(6, 230))

# Cache for dataset-wide RGB stats — loaded once per `data_root`.
_RGB_STATS_CACHE: dict[str, dict | None] = {}


def _maybe_load_rgb_stats(data_root: str | Path) -> dict | None:
    """Return per-band (B02..) → (lo, hi) percentile dict from
    ``audit/rgb_stats.json`` next to ``data_root``, or None if absent.

    Accepts a local path or an ``s3://`` URI.
    """
    audit = _p.child(_p.parent(data_root), "audit", "rgb_stats.json")
    key = _p.as_str(audit)
    if key in _RGB_STATS_CACHE:
        return _RGB_STATS_CACHE[key]
    if not _p.exists(audit):
        _RGB_STATS_CACHE[key] = None
        return None
    payload = _p.load_json(audit)
    stats = {b: tuple(payload[b]) for b in ("B02", "B03", "B04")}
    _RGB_STATS_CACHE[key] = stats
    return stats


def default_emb_root(data_root: str | Path) -> str | Path:
    """Default dense-embedding root for a given `data_root`.

    Env var `S2T_EMBEDDING_ROOT` wins if set. Accepts local paths and
    ``s3://`` URIs (returns whichever flavour the input is).
    """
    env = os.environ.get("S2T_EMBEDDING_ROOT")
    if env:
        return env if _p.is_uri(env) else Path(env)
    return _p.child(_p.parent(data_root), *_DEFAULT_EMB_REL.parts)


def default_emb_center_root(data_root: str | Path) -> str | Path:
    """Default centre-patch embedding root for a given `data_root`.

    Env var `S2T_EMBEDDING_PATCH_CENTER_ROOT` wins if set. Accepts
    local paths and ``s3://`` URIs.
    """
    env = os.environ.get("S2T_EMBEDDING_PATCH_CENTER_ROOT")
    if env:
        return env if _p.is_uri(env) else Path(env)
    return _p.child(_p.parent(data_root), *_DEFAULT_EMB_CENTER_REL.parts)


def _rewrite_segment_dates(
    labels: pl.DataFrame,
    frames: pl.DataFrame,
) -> pl.DataFrame:
    """Anchor undisturbed/revegetation periods to the observable TS.

    For every sample the chain is exactly 3 periods:
      * period_idx=0 (Undisturbed) → start := first frame date,
                                     end := event date
      * period_idx=1 (Event)       → unchanged
      * period_idx=2 (Revegetation)→ start := event date,
                                     end := last frame date

    This is the rendering the workshop notebook wants: the swimlane
    segments span the full observable history rather than an
    interpreter-defined "evidence window" that can leave gaps before
    the first observation or after the last.
    """
    fr_bounds = frames.group_by("sample_id").agg(
        pl.col("date").min().alias("_first_frame"),
        pl.col("date").max().alias("_last_frame"),
    )
    event_starts = (
        labels.filter(pl.col("is_event"))
        .select(
            pl.col("sample_id"),
            pl.col("start").alias("_event_start"),
        )
    )
    joined = labels.join(fr_bounds, on="sample_id", how="left").join(
        event_starts, on="sample_id", how="left"
    )
    new_start = (
        pl.when(pl.col("period_idx") == 0)
        .then(pl.col("_first_frame"))
        .when(pl.col("period_idx") == 2)
        .then(pl.col("_event_start"))
        .otherwise(pl.col("start"))
    )
    new_end_evidence = (
        pl.when(pl.col("period_idx") == 0)
        .then(pl.col("_event_start"))
        .when(pl.col("period_idx") == 2)
        .then(pl.col("_last_frame"))
        .otherwise(pl.col("end_evidence"))
    )
    new_end_validity = (
        pl.when(pl.col("period_idx") == 0)
        .then(pl.col("_event_start"))
        .when(pl.col("period_idx") == 2)
        .then(pl.col("_last_frame"))
        .otherwise(pl.col("end_validity"))
    )
    rewritten = joined.with_columns(
        new_start.alias("start"),
        new_end_evidence.alias("end_evidence"),
        new_end_validity.alias("end_validity"),
    ).drop(["_first_frame", "_last_frame", "_event_start"])
    return rewritten.select(labels.columns)


def load_metadata(root: str | Path) -> dict:
    """Load the four metadata artefacts as a single dict.

    Accepts a local path or an ``s3://`` URI — polars reads parquet
    directly from S3 when ``s3fs`` is installed.

    Undisturbed/revegetation segment dates in ``labels`` are rewritten
    to span the observable time series (see ``_rewrite_segment_dates``)
    so the swimlane in ``sample_timeline`` aligns cleanly with the
    frames in the zarr store.
    """
    if not _p.is_uri(root):
        root = Path(root)
    classes_raw = _p.load_json(_p.child(root, "classes.json"))
    classes = {int(k): v for k, v in classes_raw.items()}
    samples = pl.read_parquet(_p.as_str(_p.child(root, "samples.parquet")))
    labels_raw = pl.read_parquet(_p.as_str(_p.child(root, "labels.parquet")))
    frames = pl.read_parquet(_p.as_str(_p.child(root, "frames.parquet")))
    labels = _rewrite_segment_dates(labels_raw, frames)
    return {
        "samples": samples,
        "labels": labels,
        "frames": frames,
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
    `iter_samples(root)`. Lazy: opens the patches/ zarr store on
    construction but defers pixel and embedding reads until indexing.
    """

    sample_id: int
    dates: list[str]              # ISO strings, calendar-sorted
    window_start: str
    window_end: str
    labels: pl.DataFrame          # full period chain for this sid
    _grp: zarr.Group
    _root: str | Path
    _emb_root: str | Path | None = None
    _emb_center_root: str | Path | None = None
    _emb_grp: zarr.Group | None = field(default=None, repr=False)
    _emb_center_grp: zarr.Group | None = field(default=None, repr=False)

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

    def _open_emb(self) -> zarr.Group:
        """Lazy-open the dense embedding store, validating dates."""
        if self._emb_grp is not None:
            return self._emb_grp
        if self._emb_root is None:
            raise RuntimeError(
                "Dense embeddings root not configured. Set the "
                "S2T_EMBEDDING_ROOT env var or pass embedding_root= to "
                "get_sample()."
            )
        store = _p.child(self._emb_root, f"{self.sample_id}.zarr")
        if not _p.exists(store):
            raise FileNotFoundError(
                f"Dense embedding store not found: {store}"
            )
        grp = zarr.open_group(_p.as_str(store), mode="r")
        _validate_emb_store(grp, store, self.sample_id, self.dates,
                            expected_last_dim=_EMB_DIM, ndim_expected=3)
        self._emb_grp = grp
        return grp

    def _open_emb_centre(self) -> zarr.Group:
        """Lazy-open the centre-patch embedding store, validating dates."""
        if self._emb_center_grp is not None:
            return self._emb_center_grp
        if self._emb_center_root is None:
            raise RuntimeError(
                "Centre-patch embeddings root not configured. Set "
                "S2T_EMBEDDING_PATCH_CENTER_ROOT or pass "
                "embedding_center_root= to get_sample()."
            )
        store = _p.child(self._emb_center_root, f"{self.sample_id}.zarr")
        if not _p.exists(store):
            raise FileNotFoundError(
                f"Centre-patch embedding store not found: {store}"
            )
        grp = zarr.open_group(_p.as_str(store), mode="r")
        _validate_emb_store(grp, store, self.sample_id, self.dates,
                            expected_last_dim=_EMB_DIM, ndim_expected=2)
        self._emb_center_grp = grp
        return grp

    def tm_emb(self, i: int) -> np.ndarray:
        """Dense FM token embedding for frame `i`.

        Returns a `(14, 14, 384)` float32 array reshaped from the
        on-disk `(196, 384)` row-major token layout.
        """
        grp = self._open_emb()
        flat = np.asarray(grp[_EMB_NAME][i], dtype=np.float32)  # (196, 384)
        h, w = _EMB_TOKEN_GRID
        return flat.reshape(h, w, _EMB_DIM)

    def tm_emb_centre(self, i: int) -> np.ndarray:
        """Centre-patch FM embedding for frame `i`. Shape `(384,)` float32."""
        grp = self._open_emb_centre()
        return np.asarray(grp[_EMB_NAME][i], dtype=np.float32)

    @property
    def emb_dim(self) -> int:
        """Embedding dimensionality of the model in the store."""
        return _EMB_DIM

    def rgb(self, i: int, gamma: float = 1.0,
            stretch: tuple[float, float] | dict[str, tuple[float, float]] | None = None,
            crop_224: bool = False) -> np.ndarray:
        """Convenience: HxWx3 RGB in [0,1] from B04/B03/B02.

        Uses physical reflectance under the hood (BOA_ADD_OFFSET handled).

        ``stretch`` accepts:

        - ``None`` (default) → load dataset-wide per-band percentiles from
          ``audit/rgb_stats.json`` (computed by ``compute_rgb_stats()``).
          Falls back to a hard-coded ``(0.0, 0.3)`` if the audit file isn't
          present.
        - ``(lo, hi)`` → single shared stretch applied to all 3 channels.
        - ``{'B02': (lo, hi), 'B03': ..., 'B04': ...}`` → explicit per-band.

        ``crop_224=True`` returns the [6:230, 6:230] crop that matches the
        TerraMind ViT input — useful when displaying RGB beside
        ``pca_token_image`` for pixel-aligned overlays.
        """
        s10 = self.s2_10m(i, as_reflectance=True)
        rgb = np.stack([s10[2], s10[1], s10[0]], axis=-1)  # (H, W, 3) = R, G, B
        if crop_224:
            rgb = rgb[RGB_CROP_224]

        # Resolve per-band stretch.
        if stretch is None:
            stats = _maybe_load_rgb_stats(self._root)
            if stats is not None:
                lohi = np.asarray([
                    stats["B04"], stats["B03"], stats["B02"],
                ], dtype=np.float32)   # (3, 2)
            else:
                lohi = np.asarray([[0.0, 0.3]] * 3, dtype=np.float32)
        elif isinstance(stretch, dict):
            lohi = np.asarray([
                stretch["B04"], stretch["B03"], stretch["B02"],
            ], dtype=np.float32)
        else:
            lo, hi = stretch
            lohi = np.asarray([[lo, hi]] * 3, dtype=np.float32)

        lo = lohi[:, 0][None, None, :]
        hi = lohi[:, 1][None, None, :]
        rgb = np.clip((rgb - lo) / np.maximum(hi - lo, 1e-9), 0.0, 1.0)
        if gamma != 1.0:
            rgb = rgb ** (1.0 / gamma)
        return rgb

    def rgb_centered_224(self, i: int, **kwargs) -> np.ndarray:
        """``rgb(i, crop_224=True, ...)`` — the crop the FM ViT ingests.

        With this crop, RGB pixel (x, y) aligns with PCA-token pixel
        (x // 16, y // 16) — useful for side-by-side display of RGB and
        ``pca_token_image``.
        """
        return self.rgb(i, crop_224=True, **kwargs)

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


def _open_and_validate(root: str | Path, sample_id: int,
                       frames: pl.DataFrame) -> tuple[zarr.Group, list[str]]:
    """Open the zarr store and check the date-mapping invariants."""
    store = _p.child(root, "patches", f"{sample_id}.zarr")
    if not _p.exists(store):
        raise FileNotFoundError(store)
    grp = zarr.open_group(_p.as_str(store), mode="r")

    grp_sid = int(grp.attrs.get("sample_id", -1))
    if grp_sid != sample_id:
        raise ValueError(
            f"{store}: group attrs sample_id={grp_sid} != filename {sample_id}"
        )

    array_keys = {*_S2_BANDS, "s2_scl"}
    dates_per_array = {
        k: list(grp[k].attrs.get("dates") or []) for k in array_keys
    }
    ref = dates_per_array["s2_10m"]
    for arr_name, dates in dates_per_array.items():
        if dates != ref:
            raise ValueError(
                f"{store}: {arr_name}.attrs['dates'] disagrees with s2_10m"
            )
        if grp[arr_name].shape[0] != len(dates):
            raise ValueError(
                f"{store}: {arr_name}.shape[0]={grp[arr_name].shape[0]} "
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


def _validate_emb_store(
    grp: zarr.Group,
    store: str | Path,
    sample_id: int,
    expected_dates: list[str],
    *,
    expected_last_dim: int,
    ndim_expected: int,
) -> None:
    """Validate that an external embedding store matches the patches' dates.

    Used by `TimeSeries._open_emb` / `_open_emb_centre` on first access.
    `ndim_expected=3` for the dense `(T, 196, D)` store; `=2` for the
    centre-patch `(T, D)` store.
    """
    grp_sid = grp.attrs.get("sample_id")
    if grp_sid is not None and int(grp_sid) != sample_id:
        raise ValueError(
            f"{store}: attrs sample_id={grp_sid} != filename {sample_id}"
        )
    if _EMB_NAME not in list(grp.array_keys()):
        raise ValueError(f"{store}: missing array {_EMB_NAME!r}")
    arr = grp[_EMB_NAME]
    if arr.ndim != ndim_expected:
        raise ValueError(
            f"{store}: {_EMB_NAME}.ndim={arr.ndim} != expected {ndim_expected}"
        )
    if arr.shape[-1] != expected_last_dim:
        raise ValueError(
            f"{store}: {_EMB_NAME}.shape[-1]={arr.shape[-1]} != "
            f"expected embed_dim {expected_last_dim}"
        )
    # dates lives on the array (colleague's convention) but allow group-level
    # fallback for forward compatibility.
    dates = list(arr.attrs.get("dates") or grp.attrs.get("dates") or [])
    if not dates:
        raise ValueError(f"{store}: missing dates attr on {_EMB_NAME}")
    if dates != expected_dates:
        raise ValueError(
            f"{store}: embedding dates disagree with patches/{sample_id}.zarr "
            f"({len(dates)} vs {len(expected_dates)})"
        )
    if arr.shape[0] != len(dates):
        raise ValueError(
            f"{store}: {_EMB_NAME}.shape[0]={arr.shape[0]} != len(dates)={len(dates)}"
        )


def load_npz(path: str | Path, *, allow_pickle: bool = False) -> dict:
    """Load a numpy ``.npz`` file from a local path **or** ``s3://`` URI.

    URI paths require ``s3fs`` to be installed; the bytes are buffered
    in memory before ``numpy.load`` (since np.load needs a seekable
    stream). For the workshop the audit ``.npz`` files are < 100 MB
    each, so the in-memory buffer is fine.
    """
    if _p.is_uri(path):
        with np.load(_p.open_buffered(path), allow_pickle=allow_pickle) as f:
            return {k: np.asarray(f[k]) for k in f.files}
    with np.load(str(path), allow_pickle=allow_pickle) as f:
        return {k: np.asarray(f[k]) for k in f.files}


def get_sample(
    root: str | Path,
    sample_id: int,
    metadata: dict | None = None,
    *,
    embedding_root: str | Path | None = None,
    embedding_center_root: str | Path | None = None,
) -> TimeSeries:
    """Open one sample's time series.

    `embedding_root` / `embedding_center_root` override the auto-detected
    paths (see `default_emb_root` / `default_emb_center_root`). Local
    filesystem paths and ``s3://`` URIs are both accepted.
    """
    if not _p.is_uri(root):
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

    def _coerce(p):
        return p if _p.is_uri(p) else Path(p)

    emb_root = _coerce(embedding_root) if embedding_root is not None else default_emb_root(root)
    emb_center_root = (
        _coerce(embedding_center_root) if embedding_center_root is not None
        else default_emb_center_root(root)
    )
    return TimeSeries(
        sample_id=sample_id,
        dates=dates,
        window_start=sample_row["window_start"].isoformat(),
        window_end=sample_row["window_end"].isoformat(),
        labels=labels,
        _grp=grp,
        _root=root,
        _emb_root=emb_root if _p.exists(emb_root) else None,
        _emb_center_root=emb_center_root if _p.exists(emb_center_root) else None,
    )


def iter_samples(
    root: str | Path,
    *,
    embedding_root: str | Path | None = None,
    embedding_center_root: str | Path | None = None,
) -> Iterator[TimeSeries]:
    """Yield all samples in `samples.parquet` order."""
    md = load_metadata(root)
    for sid in md["samples"]["sample_id"].to_list():
        yield get_sample(
            root, int(sid),
            metadata=md,
            embedding_root=embedding_root,
            embedding_center_root=embedding_center_root,
        )
