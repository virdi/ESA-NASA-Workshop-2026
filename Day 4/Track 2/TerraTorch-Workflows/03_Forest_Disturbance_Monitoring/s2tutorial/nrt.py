"""Phenological-anomaly framework for NRT detection + event attribution.

The unifying idea — used by operational disturbance-detection algorithms
like CCDC, BFAST, and COLD — is to compare an observation at the *current*
date against past observations of the same sample at the same calendar
time. Phenology stays roughly constant year-on-year, so the deviation
between current and past is what carries disturbance signal.

We expose the framework as small, composable helpers; the workshop
notebook (`03_nrt_and_attribution.ipynb`) wires them together.

No spatial pooling on either side:

- **FM path** — single token at the labelled centre of the 14×14 grid:
  `tm_emb[i, 7, 7, :]` → `(384,)` per frame.
- **Baseline path** — single labelled centre pixel of the 252×252
  footprint. By default 12 S2 spectral bands (B01..B12 except B10
  which is not stored — cirrus) + 3 indices (NDVI, NDWI, NDMI), but the
  band/index subset is configurable.

Per-query feature is determined by one of six named **recipes** (see
`RECIPE_REGISTRY`). The reference set R is observations within
±`REF_WINDOW_DAYS` of the query's calendar (month, day) over the past
`N_REF_YEARS` years strictly before the query.

Reference frames may include event-period or revegetation observations —
the model is asked to learn the "is-it-disturbed-now" pattern from the
data, not from a hand-curated clean baseline.
"""

from __future__ import annotations

import datetime as dt
from typing import Callable, Iterator

import numpy as np

from .loader import TimeSeries


# ----------------------------------------------------------------------
# User-locked defaults. The notebook can override per call but the helpers
# themselves stick to these unless told otherwise.
# ----------------------------------------------------------------------
N_REF_YEARS = 2
REF_WINDOW_DAYS = 45
MIN_REF_FRAMES = 3
ALERT_WINDOW_MONTHS = 12
POST_EVENT_WINDOW_MONTHS = 3
CENTER_TOKEN_RC = (7, 7)        # 14×14 grid; labelled centre falls in (7, 7)
CENTER_PIXEL_RC = (126, 126)    # 252×252 (10 m) footprint; labelled centre

_DAYS_PER_YEAR = 365.25
_DAYS_PER_MONTH = 30.4375

# All 12 S2 bands stored in the zarr (B10 cirrus is omitted upstream).
ALL_S2_BANDS: tuple[str, ...] = (
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B09", "B11", "B12",
)
# Indices computable from the 12 stored bands at the centre pixel.
ALL_INDICES: tuple[str, ...] = ("NDVI", "NDWI", "NDMI")

# Default baseline = all 12 spectral bands + 3 indices = 15 dims.
DEFAULT_BANDS = ALL_S2_BANDS
DEFAULT_INDICES = ALL_INDICES


# ----------------------------------------------------------------------
# Per-frame feature extraction
# ----------------------------------------------------------------------

def center_token(emb_frame: np.ndarray) -> np.ndarray:
    """Single (384,) centre token from a (14, 14, 384) embedding frame."""
    if emb_frame.ndim != 3:
        raise ValueError(
            f"center_token expects (H, W, D), got {emb_frame.shape}"
        )
    r, c = CENTER_TOKEN_RC
    return np.asarray(emb_frame[r, c, :], dtype=np.float32)


def center_pixel_features(
    s10: np.ndarray,
    s20: np.ndarray,
    s60: np.ndarray | None = None,
    *,
    bands: tuple[str, ...] = DEFAULT_BANDS,
    indices: tuple[str, ...] = DEFAULT_INDICES,
) -> np.ndarray:
    """Per-pixel feature vector at the labelled centre of one frame.

    All inputs in physical reflectance — pass the output of
    `TimeSeries.s2_10m(i, as_reflectance=True)`,
    `TimeSeries.s2_20m(i, as_reflectance=True)`, and (optional)
    `TimeSeries.s2_60m(i, as_reflectance=True)`.

    `bands` is any subset of `ALL_S2_BANDS`. If any of `B01` or `B09`
    are requested, `s60` must be provided (they're stored at 60 m).
    `indices` is any subset of `ALL_INDICES`.

    Returns a `(len(bands) + len(indices),)` float32 vector.

    Indices are computed at the centre pixel from the underlying bands
    in their native resolution (no spatial pooling). If a band needed
    for a requested index is not in `bands`, it's still read from the
    raster — the band list only controls what ends up in the output.
        NDVI = (B08 - B04) / (B08 + B04)
        NDWI = (B03 - B08) / (B03 + B08)         (McFeeters, water)
        NDMI = (B08 - B11) / (B08 + B11)         (forest moisture)
    """
    if s10.shape[0] != 4:
        raise ValueError(f"s10 expected (4, H, W), got {s10.shape}")
    if s20.shape[0] != 6:
        raise ValueError(f"s20 expected (6, H, W), got {s20.shape}")
    cy_10, cx_10 = CENTER_PIXEL_RC
    cy_20, cx_20 = cy_10 // 2, cx_10 // 2
    cy_60, cx_60 = cy_10 // 6, cx_10 // 6

    # Read every band we might need (used for both `bands` and any
    # index whose ingredients aren't already in `bands`).
    raw: dict[str, float] = {
        "B02": float(s10[0, cy_10, cx_10]),
        "B03": float(s10[1, cy_10, cx_10]),
        "B04": float(s10[2, cy_10, cx_10]),
        "B08": float(s10[3, cy_10, cx_10]),
        "B05": float(s20[0, cy_20, cx_20]),
        "B06": float(s20[1, cy_20, cx_20]),
        "B07": float(s20[2, cy_20, cx_20]),
        "B8A": float(s20[3, cy_20, cx_20]),
        "B11": float(s20[4, cy_20, cx_20]),
        "B12": float(s20[5, cy_20, cx_20]),
    }
    needs_60 = bool({"B01", "B09"} & set(bands))
    if needs_60:
        if s60 is None:
            raise ValueError(
                "s60 (60 m bands) is required when bands include B01 or B09"
            )
        if s60.shape[0] != 2:
            raise ValueError(f"s60 expected (2, H, W), got {s60.shape}")
        raw["B01"] = float(s60[0, cy_60, cx_60])
        raw["B09"] = float(s60[1, cy_60, cx_60])

    eps = 1e-6
    idx_vals: dict[str, float] = {}
    if "NDVI" in indices:
        idx_vals["NDVI"] = (raw["B08"] - raw["B04"]) / max(raw["B08"] + raw["B04"], eps)
    if "NDWI" in indices:
        idx_vals["NDWI"] = (raw["B03"] - raw["B08"]) / max(raw["B03"] + raw["B08"], eps)
    if "NDMI" in indices:
        idx_vals["NDMI"] = (raw["B08"] - raw["B11"]) / max(raw["B08"] + raw["B11"], eps)

    out: list[float] = []
    for b in bands:
        if b not in raw:
            raise ValueError(f"Unknown band {b!r}; must be in {ALL_S2_BANDS}")
        out.append(raw[b])
    for ix in indices:
        if ix not in idx_vals:
            raise ValueError(f"Unknown index {ix!r}; must be in {ALL_INDICES}")
        out.append(idx_vals[ix])
    return np.asarray(out, dtype=np.float32)


# Back-compat alias for callers that expect a fixed list of bands.
BASELINE_BANDS = ALL_S2_BANDS + ALL_INDICES


# ----------------------------------------------------------------------
# Calendar / reference set
# ----------------------------------------------------------------------

def _calendar_distance_days(d1: dt.date, d2: dt.date) -> int:
    """Minimum |day-of-year| distance between two dates, ignoring year.

    Wraps around the year boundary, e.g. Dec 31 ↔ Jan 1 = 1 day.
    """
    doy1 = d1.timetuple().tm_yday
    doy2 = d2.timetuple().tm_yday
    diff = abs(doy1 - doy2)
    return min(diff, 365 - diff)


def reference_indices(
    ts: TimeSeries,
    query_idx: int,
    *,
    n_years: int = N_REF_YEARS,
    window_days: int = REF_WINDOW_DAYS,
    min_frames: int = MIN_REF_FRAMES,
) -> list[int]:
    """`ts.dates` indices of reference frames for the query at `query_idx`.

    A reference frame is one whose date is:

    1. **Strictly before** the query date (no future leakage).
    2. **Within `n_years` years** of the query (with `window_days` slack).
    3. **Within ±`window_days` of the query's day-of-year** (calendar match).

    Returns `[]` if fewer than `min_frames` such frames exist.
    """
    query_date = dt.date.fromisoformat(ts.dates[query_idx])
    horizon_days = n_years * _DAYS_PER_YEAR + window_days
    out: list[int] = []
    for i, d_str in enumerate(ts.dates):
        if i == query_idx:
            continue
        obs = dt.date.fromisoformat(d_str)
        if obs >= query_date:
            continue
        days_back = (query_date - obs).days
        if days_back > horizon_days:
            continue
        if _calendar_distance_days(obs, query_date) <= window_days:
            out.append(i)
    return out if len(out) >= min_frames else []


def query_label(
    ts: TimeSeries,
    query_idx: int,
    *,
    alert_months: int = ALERT_WINDOW_MONTHS,
) -> int | None:
    """Binary detection label for the query at `query_idx`.

    Returns `1` iff `query_date ∈ [event_start, event_start + alert_months]`,
    else `0`. Returns `None` if `ts` has no event period.
    """
    event = ts.event_period()
    if event is None:
        return None
    event_start = event["start"]
    if isinstance(event_start, str):
        event_start = dt.date.fromisoformat(event_start)
    alert_end = event_start + dt.timedelta(
        days=round(alert_months * _DAYS_PER_MONTH)
    )
    query_date = dt.date.fromisoformat(ts.dates[query_idx])
    return int(event_start <= query_date <= alert_end)


def post_event_indices(
    ts: TimeSeries,
    *,
    months: int = POST_EVENT_WINDOW_MONTHS,
) -> list[int]:
    """Indices of cloud-free frames in `[event_start, event_start + months]`."""
    event = ts.event_period()
    if event is None:
        return []
    event_start = event["start"]
    if isinstance(event_start, str):
        event_start = dt.date.fromisoformat(event_start)
    end = event_start + dt.timedelta(days=round(months * _DAYS_PER_MONTH))
    return [
        i for i, d_str in enumerate(ts.dates)
        if event_start <= dt.date.fromisoformat(d_str) <= end
    ]


# ----------------------------------------------------------------------
# Feature recipes
# ----------------------------------------------------------------------
#
# Standard convention: cosine *distance* = 1 − cosine similarity.
# Identical vectors give 0, orthogonal give 1, opposite give 2.

def cosine_distance(a: np.ndarray, b: np.ndarray, *, eps: float = 1e-12) -> float:
    """1 - (a · b) / (‖a‖ ‖b‖). Returns a Python float."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < eps or nb < eps:
        return 1.0
    return float(1.0 - (a @ b) / (na * nb))


def _recipe_diff(query: np.ndarray, ref_vecs: np.ndarray) -> np.ndarray:
    """query − mean(R). Returns (D,)."""
    return (query - ref_vecs.mean(axis=0)).astype(np.float32)


def _recipe_cd_scalar(query: np.ndarray, ref_vecs: np.ndarray) -> np.ndarray:
    """[CD(query, mean(R))]. Returns (1,)."""
    return np.asarray(
        [cosine_distance(query, ref_vecs.mean(axis=0))], dtype=np.float32
    )


def _recipe_cd_pseudo_std(query: np.ndarray, ref_vecs: np.ndarray) -> np.ndarray:
    """[CD(query, mean(R)), mean(CD(R, mean(R)))]. Returns (2,)."""
    mean_r = ref_vecs.mean(axis=0)
    cd_q = cosine_distance(query, mean_r)
    cd_refs = float(np.mean([cosine_distance(r, mean_r) for r in ref_vecs]))
    return np.asarray([cd_q, cd_refs], dtype=np.float32)


def _recipe_query_cd(query: np.ndarray, ref_vecs: np.ndarray) -> np.ndarray:
    """concat[query, CD(query, mean(R)), mean(CD(R, mean(R)))]. Returns (D+2,)."""
    mean_r = ref_vecs.mean(axis=0)
    cd_q = cosine_distance(query, mean_r)
    cd_refs = float(np.mean([cosine_distance(r, mean_r) for r in ref_vecs]))
    return np.concatenate(
        [query, np.asarray([cd_q, cd_refs], dtype=np.float32)]
    ).astype(np.float32)


def _recipe_query_mean_cd(query: np.ndarray, ref_vecs: np.ndarray) -> np.ndarray:
    """concat[query, mean(R), CD(query, mean(R)), mean(CD(R, mean(R)))]. Returns (2D+2,)."""
    mean_r = ref_vecs.mean(axis=0)
    cd_q = cosine_distance(query, mean_r)
    cd_refs = float(np.mean([cosine_distance(r, mean_r) for r in ref_vecs]))
    return np.concatenate(
        [query, mean_r, np.asarray([cd_q, cd_refs], dtype=np.float32)]
    ).astype(np.float32)


def _recipe_full(query: np.ndarray, ref_vecs: np.ndarray) -> np.ndarray:
    """concat[query, mean(R), query − mean(R), std(R)]. Returns (4D,)."""
    mean_r = ref_vecs.mean(axis=0)
    return np.concatenate(
        [query, mean_r, query - mean_r, ref_vecs.std(axis=0)]
    ).astype(np.float32)


RECIPE_REGISTRY: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "diff":           _recipe_diff,
    "cd_scalar":      _recipe_cd_scalar,
    "cd_pseudo_std":  _recipe_cd_pseudo_std,
    "query_cd":       _recipe_query_cd,
    "query_mean_cd":  _recipe_query_mean_cd,
    "full":           _recipe_full,
}


def recipe_feature_dim(recipe: str, base_dim: int) -> int:
    """Number of features the recipe produces for a `base_dim`-dim input."""
    return {
        "diff":          base_dim,
        "cd_scalar":     1,
        "cd_pseudo_std": 2,
        "query_cd":      base_dim + 2,
        "query_mean_cd": 2 * base_dim + 2,
        "full":          4 * base_dim,
    }[recipe]


def compute_recipe(
    query_vec: np.ndarray,
    ref_vecs: np.ndarray,
    *,
    recipe: str = "full",
) -> np.ndarray:
    """Apply one named recipe; convenience wrapper around `RECIPE_REGISTRY`."""
    if recipe not in RECIPE_REGISTRY:
        raise ValueError(
            f"Unknown recipe {recipe!r}; must be one of {list(RECIPE_REGISTRY)}"
        )
    if ref_vecs.ndim != 2 or ref_vecs.shape[1] != query_vec.shape[0]:
        raise ValueError(
            f"shape mismatch: query {query_vec.shape}, refs {ref_vecs.shape}"
        )
    return RECIPE_REGISTRY[recipe](query_vec, ref_vecs)


# Back-compat: anomaly_features is the "full" recipe.
def anomaly_features(query_vec: np.ndarray, ref_vecs: np.ndarray) -> np.ndarray:
    """concat[query, mean(R), query - mean(R), std(R)] (the `full` recipe)."""
    return compute_recipe(query_vec, ref_vecs, recipe="full")


# ----------------------------------------------------------------------
# Classifier factory
# ----------------------------------------------------------------------

def make_classifier(
    *,
    penalty: str = "l2",
    C: float = 1.0,
    l1_ratio: float | None = None,
    class_weight: str | dict | None = "balanced",
    max_iter: int = 5000,
    random_state: int = 42,
):
    """`sklearn.linear_model.LogisticRegression` with the right solver per penalty.

    - `penalty='l2'` → `lbfgs` (default sklearn).
    - `penalty='l1'` → `liblinear` (binary) or `saga` (multinomial).
    - `penalty='elasticnet'` → `saga`, `l1_ratio` required (defaults to 0.5).
    - `penalty='none'` → no regularisation, `lbfgs`.
    """
    from sklearn.linear_model import LogisticRegression

    kwargs = dict(
        C=C, max_iter=max_iter, class_weight=class_weight,
        random_state=random_state,
    )
    if penalty == "l2":
        return LogisticRegression(penalty="l2", solver="lbfgs", **kwargs)
    if penalty == "l1":
        # liblinear is faster than saga for binary, and supports multi-class
        # via one-vs-rest. saga would be needed for L1 + true multinomial.
        return LogisticRegression(penalty="l1", solver="liblinear", **kwargs)
    if penalty == "elasticnet":
        return LogisticRegression(
            penalty="elasticnet", solver="saga",
            l1_ratio=l1_ratio if l1_ratio is not None else 0.5, **kwargs,
        )
    if penalty in ("none", None):
        # newer sklearn uses penalty=None; pop C since it's irrelevant
        kwargs.pop("C")
        return LogisticRegression(penalty=None, solver="lbfgs", **kwargs)
    raise ValueError(f"Unknown penalty {penalty!r}")


# ----------------------------------------------------------------------
# Detection-query iterator
# ----------------------------------------------------------------------

def iter_detection_queries(
    ts: TimeSeries,
    *,
    n_ref_years: int = N_REF_YEARS,
    ref_window_days: int = REF_WINDOW_DAYS,
    min_ref_frames: int = MIN_REF_FRAMES,
    alert_months: int = ALERT_WINDOW_MONTHS,
) -> Iterator[tuple[int, int, list[int]]]:
    """Yield `(query_idx, label, ref_indices)` for every eligible frame."""
    for i in range(len(ts)):
        ref = reference_indices(
            ts, i,
            n_years=n_ref_years,
            window_days=ref_window_days,
            min_frames=min_ref_frames,
        )
        if not ref:
            continue
        label = query_label(ts, i, alert_months=alert_months)
        if label is None:
            continue
        yield i, label, ref


def references_at_event(
    ts: TimeSeries,
    *,
    n_years: int = N_REF_YEARS,
    window_days: int = REF_WINDOW_DAYS,
    min_frames: int = MIN_REF_FRAMES,
) -> list[int] | None:
    """Reference frame indices centred on the sample's event date.

    Treats the first cloud-free frame at or after ``event_period().start``
    as the query date, then calls :func:`reference_indices`. Returns
    ``None`` if the sample has no event or no eligible references.
    """
    ev = ts.event_period()
    if ev is None:
        return None
    ev_start = ev["start"]
    event_idx = next(
        (i for i, d in enumerate(ts.dates)
         if dt.date.fromisoformat(d) >= ev_start),
        None,
    )
    if event_idx is None:
        return None
    refs = reference_indices(
        ts, event_idx,
        n_years=n_years,
        window_days=window_days,
        min_frames=min_frames,
    )
    return refs or None


# ----------------------------------------------------------------------
# Dataset builders — used by Section 3 (detection) and Section 4
# (attribution) of the workshop notebook.
# ----------------------------------------------------------------------

# Default 4-class attribution palette: the workshop's curated disturbance
# codes plus their human-readable names.
DEFAULT_ATTR_CODES = (211, 212, 242, 243)
DEFAULT_ATTR_NAMES = ("Clear-Cut", "Thinning", "Wildfire", "Wind")


def build_detection_dataset(
    meta: dict,
    *,
    root,
    emb_centre_for: Callable[[int], np.ndarray],
    pix_centre_for: Callable[[int], np.ndarray],
    recipe: str = "full",
    n_ref_years: int = N_REF_YEARS,
    ref_window_days: int = REF_WINDOW_DAYS,
    alert_months: int = ALERT_WINDOW_MONTHS,
    verbose: bool = True,
) -> dict:
    """Assemble the binary detection training set across all samples.

    For every sample in ``meta['samples']`` and every eligible
    ``(query, label, reference_set)`` tuple from
    :func:`iter_detection_queries`, build the FM + baseline recipe
    feature vector and stack them.

    ``emb_centre_for(sid) -> (T, D_fm)`` and
    ``pix_centre_for(sid) -> (T, D_pix)`` are the per-sample per-frame
    feature accessors. They typically read from a pre-computed cache so
    the loop stays fast (~30 s on CPU for 200 samples).

    Returns a dict with keys
        ``X_emb`` (N, F_fm), ``X_pix`` (N, F_pix),
        ``y`` (N,) int, ``groups`` (N,) int = sample_id,
        ``elapsed_s`` (float wall-clock seconds).
    """
    import time
    from .loader import get_sample

    t0 = time.time()
    X_emb: list[np.ndarray] = []
    X_pix: list[np.ndarray] = []
    y: list[int] = []
    groups: list[int] = []
    for sid in meta["samples"]["sample_id"].to_list():
        sid = int(sid)
        ts_i = get_sample(root, sid, metadata=meta)
        emb_i = emb_centre_for(sid)
        pix_i = pix_centre_for(sid)
        for qi, lbl, ref in iter_detection_queries(
            ts_i,
            n_ref_years=n_ref_years,
            ref_window_days=ref_window_days,
            alert_months=alert_months,
        ):
            X_emb.append(compute_recipe(emb_i[qi], emb_i[ref], recipe=recipe))
            X_pix.append(compute_recipe(pix_i[qi], pix_i[ref], recipe=recipe))
            y.append(lbl)
            groups.append(sid)
    out = {
        "X_emb": np.stack(X_emb),
        "X_pix": np.stack(X_pix),
        "y": np.asarray(y),
        "groups": np.asarray(groups),
        "elapsed_s": time.time() - t0,
    }
    if verbose:
        n_pos = int(out["y"].sum())
        print(
            f"Detection dataset: {len(out['y'])} queries from "
            f"{len(set(out['groups'].tolist()))} samples; "
            f"{n_pos} positive ({100 * out['y'].mean():.1f}%)"
        )
        print(
            f"Feature dims:      FM {out['X_emb'].shape[1]}, "
            f"baseline {out['X_pix'].shape[1]}"
        )
        print(f"Total time:        {out['elapsed_s']:.1f}s")
    return out


def attribution_feature_vector(
    arr: np.ndarray,
    ref_idx: list[int],
    post_idx: list[int],
) -> np.ndarray:
    """5-block per-sample attribution feature.

    Concatenates ``[pre_mean, pre_std, post_mean, post_std,
    post_mean - pre_mean]`` along the feature axis. ``arr`` is the
    per-frame feature matrix ``(T, D)``; the result is ``(5*D,)``.

    The same block layout is used at training (per-sample, centre token
    or centre pixel) and at dense inference (per-spatial-location, via
    :func:`s2tutorial.dense.dense_attribution_token_features`).
    """
    pre = arr[ref_idx]
    post = arr[post_idx]
    return np.concatenate([
        pre.mean(0),
        pre.std(0),
        post.mean(0),
        post.std(0),
        post.mean(0) - pre.mean(0),
    ])


def build_attribution_dataset(
    meta: dict,
    *,
    root,
    emb_centre_for: Callable[[int], np.ndarray],
    pix_centre_for: Callable[[int], np.ndarray],
    attr_codes: tuple[int, ...] = DEFAULT_ATTR_CODES,
    attr_names: tuple[str, ...] = DEFAULT_ATTR_NAMES,
    n_ref_years: int = N_REF_YEARS,
    ref_window_days: int = REF_WINDOW_DAYS,
    verbose: bool = True,
) -> dict:
    """Assemble the 4-class attribution dataset across all samples.

    One vector per sample whose event label falls in ``attr_codes``.
    Reference frames come from :func:`references_at_event` (calendar-
    matched past-year same-period set anchored on event_start),
    post-event frames from :func:`post_event_indices`.

    Returns a dict with keys
        ``X_emb`` (N, 5*D_fm), ``X_pix`` (N, 5*D_pix),
        ``y`` (N,) int = index into ``attr_codes``,
        ``sids`` (N,) int,
        ``attr_codes`` / ``attr_names`` / ``attr_to_idx`` for plotting,
        ``elapsed_s`` (float wall-clock seconds).
    """
    import time
    from .loader import get_sample

    attr_to_idx = {int(c): i for i, c in enumerate(attr_codes)}
    t0 = time.time()
    Xa_emb: list[np.ndarray] = []
    Xa_pix: list[np.ndarray] = []
    y_attr: list[int] = []
    sids_attr: list[int] = []
    for sid in meta["samples"]["sample_id"].to_list():
        sid = int(sid)
        ts_i = get_sample(root, sid, metadata=meta)
        label = int(ts_i.event_period()["label"])
        if label not in attr_to_idx:
            continue
        ref_idx = references_at_event(
            ts_i, n_years=n_ref_years, window_days=ref_window_days,
        )
        post_idx = post_event_indices(ts_i)
        if not ref_idx or len(post_idx) < 1:
            continue
        emb_i = emb_centre_for(sid)
        pix_i = pix_centre_for(sid)
        Xa_emb.append(attribution_feature_vector(emb_i, ref_idx, post_idx))
        Xa_pix.append(attribution_feature_vector(pix_i, ref_idx, post_idx))
        y_attr.append(attr_to_idx[label])
        sids_attr.append(sid)
    out = {
        "X_emb": np.stack(Xa_emb),
        "X_pix": np.stack(Xa_pix),
        "y": np.asarray(y_attr),
        "sids": np.asarray(sids_attr),
        "attr_codes": np.asarray(attr_codes),
        "attr_names": list(attr_names),
        "attr_to_idx": attr_to_idx,
        "elapsed_s": time.time() - t0,
    }
    if verbose:
        print(
            f"Attribution dataset: {len(out['y'])} samples in "
            f"{out['elapsed_s']:.1f}s"
        )
        print(
            f"  feature dims: FM {out['X_emb'].shape[1]}, "
            f"baseline {out['X_pix'].shape[1]}"
        )
        for k, code in enumerate(attr_codes):
            n = int((out["y"] == k).sum())
            print(f"  {int(code)} {attr_names[k]:12s}  n={n}")
    return out
