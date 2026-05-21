"""Dense-inference helpers for the unified workshop notebook.

The NRT-detection / attribution training uses *single-pixel* (baseline) or
*single-token* (FM) features at the labelled centre of each patch. For the
workshop's dense-prediction figures we want to *apply* the same trained
classifier to every spatial location of one frame, producing a 2-D prediction
map.

This module vectorises the per-location feature extraction so the workshop
notebook can call:

    feats = dense_token_features(ts, query_idx, ref_indices)        # (14, 14, 4*384)
    proba = dense_predict(clf_fm, feats)                            # (14, 14)

    feats_pix = dense_pixel_features(ts, query_idx, ref_indices)    # (H, W, 4*13)
    proba_pix = dense_predict(clf_baseline, feats_pix)              # (H, W)

The same six recipes available at the centre (``s2t.RECIPE_REGISTRY``)
are honoured: the implementation just lifts each recipe across the spatial
axes via numpy broadcasting.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from .loader import TimeSeries
from .models import TorchClassifier
from .nrt import (
    ALL_INDICES,
    ALL_S2_BANDS,
    DEFAULT_BANDS,
    DEFAULT_INDICES,
    RECIPE_REGISTRY,
    iter_detection_queries,
    post_event_indices,
    reference_indices,
    references_at_event,
)


# ---------------------------------------------------------------------------
# Dense feature builders — FM (token) and baseline (pixel) paths
# ---------------------------------------------------------------------------

def _apply_recipe_spatial(
    query: np.ndarray,   # (H, W, D)
    refs: np.ndarray,    # (N_ref, H, W, D)
    *,
    recipe: str = "full",
) -> np.ndarray:
    """Vectorised version of ``RECIPE_REGISTRY[recipe]`` over a spatial grid.

    For each ``(r, c)`` location the per-recipe function from ``nrt.py`` is
    applied to ``query[r, c]`` and ``refs[:, r, c]``. The implementation
    avoids the per-location Python loop by computing the reductions
    (``mean``, ``std``) once along ``axis=0`` and broadcasting.

    Returns ``(H, W, F)`` where ``F`` is the recipe's feature dimension.
    """
    if recipe not in RECIPE_REGISTRY:
        raise ValueError(
            f"Unknown recipe {recipe!r}; must be one of {list(RECIPE_REGISTRY)}"
        )
    if refs.ndim != 4 or query.ndim != 3:
        raise ValueError(
            f"shapes: query expected (H, W, D), got {query.shape}; "
            f"refs expected (N, H, W, D), got {refs.shape}"
        )
    mean_r = refs.mean(axis=0)           # (H, W, D)

    if recipe == "diff":
        out = (query - mean_r).astype(np.float32)
    elif recipe == "full":
        std_r = refs.std(axis=0)         # (H, W, D)
        out = np.concatenate(
            [query, mean_r, query - mean_r, std_r], axis=-1,
        ).astype(np.float32)
    elif recipe in {"cd_scalar", "cd_pseudo_std", "query_cd", "query_mean_cd"}:
        # cosine-distance recipes: compute per-location cosine.
        eps = 1e-12
        q_n = np.linalg.norm(query, axis=-1, keepdims=True)
        m_n = np.linalg.norm(mean_r, axis=-1, keepdims=True)
        cd_q = 1.0 - np.sum(query * mean_r, axis=-1, keepdims=True) / np.clip(q_n * m_n, eps, None)
        # cd_refs: mean over refs of CD(refs[i], mean_r)
        r_n = np.linalg.norm(refs, axis=-1, keepdims=True)              # (N, H, W, 1)
        cd_each = 1.0 - np.sum(refs * mean_r[None], axis=-1, keepdims=True) / np.clip(r_n * m_n[None], eps, None)
        cd_refs = cd_each.mean(axis=0)                                  # (H, W, 1)
        if recipe == "cd_scalar":
            out = cd_q.astype(np.float32)
        elif recipe == "cd_pseudo_std":
            out = np.concatenate([cd_q, cd_refs], axis=-1).astype(np.float32)
        elif recipe == "query_cd":
            out = np.concatenate([query, cd_q, cd_refs], axis=-1).astype(np.float32)
        else:  # query_mean_cd
            out = np.concatenate([query, mean_r, cd_q, cd_refs], axis=-1).astype(np.float32)
    else:
        raise AssertionError(f"unhandled recipe {recipe!r}")
    return out


def dense_token_features(
    ts: TimeSeries,
    query_idx: int,
    ref_indices: Sequence[int],
    *,
    recipe: str = "full",
) -> np.ndarray:
    """Per-token recipe features for one frame.

    Returns a ``(14, 14, F)`` float32 array where ``F`` depends on the recipe
    (``4*384`` for ``"full"``, ``384`` for ``"diff"``, etc.). The recipe is
    applied independently at each of the 196 spatial locations.
    """
    query = ts.tm_emb(query_idx).astype(np.float32)              # (14, 14, 384)
    refs = np.stack([ts.tm_emb(i) for i in ref_indices], axis=0) # (N, 14, 14, 384)
    return _apply_recipe_spatial(query, refs, recipe=recipe)


# ---------------------------------------------------------------------------
# Dense pixel-baseline features
# ---------------------------------------------------------------------------

def _band_grid(
    s10: np.ndarray, s20: np.ndarray, s60: np.ndarray | None,
    *, bands: tuple[str, ...],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Stack the requested bands into a (H, W, len(bands)) grid at 10 m.

    Native-resolution bands are nearest-neighbour upsampled.
    """
    # 10 m bands directly.
    name_to_10m = {"B02": 0, "B03": 1, "B04": 2, "B08": 3}
    # 20 m bands need 2x upsampling.
    name_to_20m = {"B05": 0, "B06": 1, "B07": 2, "B8A": 3, "B11": 4, "B12": 5}
    # 60 m bands need 6x upsampling.
    name_to_60m = {"B01": 0, "B09": 1}
    H, W = s10.shape[1], s10.shape[2]
    upsampled: dict[str, np.ndarray] = {}
    for b in bands:
        if b in name_to_10m:
            upsampled[b] = s10[name_to_10m[b]]
        elif b in name_to_20m:
            arr = s20[name_to_20m[b]]
            upsampled[b] = np.repeat(np.repeat(arr, 2, axis=0), 2, axis=1)[:H, :W]
        elif b in name_to_60m:
            if s60 is None:
                raise ValueError(f"s60 required for band {b!r}")
            arr = s60[name_to_60m[b]]
            upsampled[b] = np.repeat(np.repeat(arr, 6, axis=0), 6, axis=1)[:H, :W]
        else:
            raise ValueError(f"Unknown band {b!r}; must be in {ALL_S2_BANDS}")
    stacked = np.stack([upsampled[b] for b in bands], axis=-1).astype(np.float32)
    return stacked, upsampled


def index_maps(
    ts: TimeSeries,
    frame_idx: int,
    *,
    indices: tuple[str, ...] = ("NDVI", "NDMI", "NDWI"),
) -> np.ndarray:
    """Dense per-pixel maps of normalized vegetation indices on one frame.

    Same formulas as :func:`s2tutorial.nrt.center_pixel_features`, applied
    densely over the 10 m grid instead of only the labelled centre pixel.

    - NDVI = (B08 − B04) / (B08 + B04)
    - NDMI = (B08 − B11) / (B08 + B11)   (B11 nearest-neighbour upsampled from 20 m)
    - NDWI = (B03 − B08) / (B03 + B08)   (McFeeters)

    Returns ``(H, W, len(indices))`` float32 in physical units (typically
    within [-1, 1]). Order of the last axis follows ``indices``.
    """
    s10 = ts.s2_10m(frame_idx, as_reflectance=True)   # (4, H, W)  B02 B03 B04 B08
    s20 = ts.s2_20m(frame_idx, as_reflectance=True)   # (6, H/2, W/2)  …B11…
    H, W = s10.shape[1], s10.shape[2]
    B03 = s10[1]
    B04 = s10[2]
    B08 = s10[3]
    B11_20m = s20[4]
    B11 = np.repeat(np.repeat(B11_20m, 2, axis=0), 2, axis=1)[:H, :W]
    eps = 1e-6
    out: list[np.ndarray] = []
    for name in indices:
        if name == "NDVI":
            out.append((B08 - B04) / np.clip(B08 + B04, eps, None))
        elif name == "NDMI":
            out.append((B08 - B11) / np.clip(B08 + B11, eps, None))
        elif name == "NDWI":
            out.append((B03 - B08) / np.clip(B03 + B08, eps, None))
        else:
            raise ValueError(
                f"Unknown index {name!r}; expected one of NDVI, NDMI, NDWI"
            )
    return np.stack(out, axis=-1).astype(np.float32)


def _per_pixel_baseline_features(
    s10: np.ndarray, s20: np.ndarray, s60: np.ndarray | None,
    *,
    bands: tuple[str, ...] = DEFAULT_BANDS,
    indices: tuple[str, ...] = DEFAULT_INDICES,
) -> np.ndarray:
    """Per-pixel S2 baseline features over the whole 10 m grid.

    Returns ``(H, W, len(bands) + len(indices))`` float32. Indices follow the
    same definitions as ``s2tutorial.nrt.center_pixel_features``.
    """
    stacked, raw = _band_grid(s10, s20, s60, bands=bands)
    H, W = stacked.shape[:2]
    eps = 1e-6
    extra: list[np.ndarray] = []
    if "NDVI" in indices:
        extra.append(((raw["B08"] - raw["B04"]) / np.clip(raw["B08"] + raw["B04"], eps, None))[..., None])
    if "NDWI" in indices:
        extra.append(((raw["B03"] - raw["B08"]) / np.clip(raw["B03"] + raw["B08"], eps, None))[..., None])
    if "NDMI" in indices:
        extra.append(((raw["B08"] - raw["B11"]) / np.clip(raw["B08"] + raw["B11"], eps, None))[..., None])
    if extra:
        stacked = np.concatenate([stacked, *extra], axis=-1).astype(np.float32)
    return stacked   # (H, W, D_baseline)


def dense_pixel_features(
    ts: TimeSeries,
    query_idx: int,
    ref_indices: Sequence[int],
    *,
    downsample: int | None = 16,
    bands: tuple[str, ...] = DEFAULT_BANDS,
    indices: tuple[str, ...] = DEFAULT_INDICES,
    recipe: str = "full",
) -> np.ndarray:
    """Per-pixel baseline recipe features for one frame.

    The 12 S2 bands + 3 vegetation indices are computed at every 10 m pixel
    (with nearest-neighbour upsampling of 20 m / 60 m bands), then the recipe
    is applied per-location against ``ref_indices`` past frames at the same
    locations.

    Parameters
    ----------
    downsample : int | None
        If an int, average-pool the resulting feature grid by this stride
        (e.g. ``16`` aligns 1:1 with the 14×14 FM token grid). ``None``
        keeps native 10 m resolution (252×252).

    Returns ``(H, W, F)`` float32.
    """
    def _frame_features(i: int) -> np.ndarray:
        s10 = ts.s2_10m(i, as_reflectance=True)
        s20 = ts.s2_20m(i, as_reflectance=True)
        try:
            s60 = ts.s2_60m(i, as_reflectance=True)
        except Exception:
            s60 = None
        return _per_pixel_baseline_features(
            s10, s20, s60, bands=bands, indices=indices,
        )

    q = _frame_features(query_idx)                         # (H, W, D_b)
    refs = np.stack([_frame_features(i) for i in ref_indices], axis=0)
    feats = _apply_recipe_spatial(q, refs, recipe=recipe)  # (H, W, F)

    if downsample is None or downsample == 1:
        return feats
    stride = int(downsample)
    H, W, F = feats.shape
    nh, nw = H // stride, W // stride
    crop = feats[: nh * stride, : nw * stride, :]
    return crop.reshape(nh, stride, nw, stride, F).mean(axis=(1, 3)).astype(np.float32)


# ---------------------------------------------------------------------------
# Dense classifier inference
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Attribution: dense pre/post-event feature builders
# ---------------------------------------------------------------------------

def dense_attribution_token_features(
    ts: TimeSeries,
    ref_indices: Sequence[int],
    post_indices: Sequence[int],
) -> np.ndarray:
    """Per-token attribution features for one sample's event.

    Mirrors the centre-token attribution recipe used during training
    (see :func:`s2tutorial.nrt.attribution_feature_vector`):
    ``concat[pre_mean, pre_std, post_mean, post_std, post_mean − pre_mean]``
    but applied to the full ``(14, 14, 384)`` token grid instead of the
    centre token. Returns ``(14, 14, 5 * 384) = (14, 14, 1920)`` float32.

    Use it to apply a trained 4-class attribution head at every spatial
    location on one frame, producing a dense argmax map for visualisation.
    """
    if len(ref_indices) == 0 or len(post_indices) == 0:
        raise ValueError("ref_indices and post_indices must be non-empty")
    refs = np.stack([ts.tm_emb(i) for i in ref_indices], axis=0)   # (N_ref, 14, 14, 384)
    posts = np.stack([ts.tm_emb(i) for i in post_indices], axis=0)  # (N_post, 14, 14, 384)
    pre_mean = refs.mean(axis=0)
    pre_std = refs.std(axis=0)
    post_mean = posts.mean(axis=0)
    post_std = posts.std(axis=0)
    return np.concatenate(
        [pre_mean, pre_std, post_mean, post_std, post_mean - pre_mean],
        axis=-1,
    ).astype(np.float32)


def dense_attribution_pixel_features(
    ts: TimeSeries,
    ref_indices: Sequence[int],
    post_indices: Sequence[int],
    *,
    downsample: int | None = 16,
    bands: tuple[str, ...] = DEFAULT_BANDS,
    indices: tuple[str, ...] = DEFAULT_INDICES,
) -> np.ndarray:
    """Per-pixel baseline attribution features for one sample's event.

    Same temporal aggregation as :func:`dense_attribution_token_features`,
    but per-pixel on the 12 S2 bands + NDVI/NDWI/NDMI. With
    ``downsample=16`` (default) the output aligns 1:1 with the FM
    14×14 token grid; pass ``None`` for native 10 m resolution.
    """
    if len(ref_indices) == 0 or len(post_indices) == 0:
        raise ValueError("ref_indices and post_indices must be non-empty")

    def _per_frame(i: int) -> np.ndarray:
        s10 = ts.s2_10m(i, as_reflectance=True)
        s20 = ts.s2_20m(i, as_reflectance=True)
        try:
            s60 = ts.s2_60m(i, as_reflectance=True)
        except Exception:
            s60 = None
        return _per_pixel_baseline_features(
            s10, s20, s60, bands=bands, indices=indices,
        )

    refs = np.stack([_per_frame(i) for i in ref_indices], axis=0)
    posts = np.stack([_per_frame(i) for i in post_indices], axis=0)
    pre_mean = refs.mean(axis=0)
    pre_std = refs.std(axis=0)
    post_mean = posts.mean(axis=0)
    post_std = posts.std(axis=0)
    feats = np.concatenate(
        [pre_mean, pre_std, post_mean, post_std, post_mean - pre_mean],
        axis=-1,
    ).astype(np.float32)
    if downsample is None or downsample == 1:
        return feats
    stride = int(downsample)
    H, W, F = feats.shape
    nh, nw = H // stride, W // stride
    crop = feats[: nh * stride, : nw * stride, :]
    return crop.reshape(nh, stride, nw, stride, F).mean(axis=(1, 3)).astype(np.float32)


@torch.no_grad()
def dense_predict(
    clf: TorchClassifier,
    feats: np.ndarray,
    *,
    return_proba: bool = False,
) -> np.ndarray:
    """Apply ``clf`` to every spatial location of ``feats``.

    ``feats`` is ``(H, W, F)``; returns ``(H, W)`` argmax (default) or
    ``(H, W, K)`` softmax if ``return_proba=True``. Single forward pass on
    ``H*W`` samples — the model is invariant to spatial layout.
    """
    if clf.model is None:
        raise RuntimeError("Classifier has not been fitted yet.")
    H, W, F = feats.shape
    flat = feats.reshape(H * W, F)
    Xt = torch.as_tensor(flat, dtype=torch.float32, device=clf.device)
    clf.model.eval()
    logits = clf.model(Xt)
    if return_proba:
        proba = torch.softmax(logits, dim=-1).cpu().numpy()
        return proba.reshape(H, W, -1)
    return logits.argmax(dim=-1).cpu().numpy().reshape(H, W)


def dense_attribution_gated(
    ts: TimeSeries,
    *,
    clf_det: TorchClassifier,
    clf_attr: TorchClassifier,
    det_query_idx: int,
    det_ref_indices: Sequence[int],
    attr_ref_indices: Sequence[int],
    attr_post_indices: Sequence[int],
    mode: str = "token",          # 'token' for FM, 'pixel' for baseline
    det_recipe: str = "full",
    downsample: int = 16,
    threshold: float = 0.7,
) -> dict:
    """Run detection + attribution at every spatial location, then gate.

    1. Build dense detection features at ``det_query_idx`` against
       ``det_ref_indices`` (past-year same-calendar references).
    2. Predict detection probability for the event class.
    3. Threshold to a binary mask (``proba >= threshold``).
    4. Build dense attribution features from
       (``attr_ref_indices``, ``attr_post_indices``).
    5. Predict 4-class argmax with ``clf_attr``.
    6. Multiply by the mask — pixels below threshold are tagged ``-1``
       (no-detection / no-attribution).

    ``mode`` selects which feature builder to use:

    * ``'token'`` → FM 14×14 token grid (``dense_token_features`` /
      ``dense_attribution_token_features``).
    * ``'pixel'`` → baseline avg-pooled to the same 14×14 grid via
      ``downsample=16`` on ``dense_pixel_features`` /
      ``dense_attribution_pixel_features``.

    Returns
    -------
    dict with keys
        ``detection_proba`` (H, W) float32,
        ``detection_mask``  (H, W) uint8,
        ``attribution_argmax`` (H, W) int32  — raw, ungated,
        ``gated_argmax``    (H, W) int8     — class index where mask==1, -1 elsewhere.
    """
    if mode not in ("token", "pixel"):
        raise ValueError(f"mode must be 'token' or 'pixel', got {mode!r}")

    # 1. Detection features + proba.
    if mode == "token":
        det_feats = dense_token_features(
            ts, det_query_idx, det_ref_indices, recipe=det_recipe,
        )
    else:
        det_feats = dense_pixel_features(
            ts, det_query_idx, det_ref_indices,
            downsample=downsample, recipe=det_recipe,
        )
    det_proba_all = dense_predict(clf_det, det_feats, return_proba=True)
    if det_proba_all.shape[-1] < 2:
        raise ValueError(
            "Detection classifier must produce >= 2 classes; got "
            f"{det_proba_all.shape[-1]}."
        )
    det_proba = det_proba_all[..., 1].astype(np.float32)
    det_mask = (det_proba >= float(threshold)).astype(np.uint8)

    # 2. Attribution features + argmax.
    if mode == "token":
        attr_feats = dense_attribution_token_features(
            ts, attr_ref_indices, attr_post_indices,
        )
    else:
        attr_feats = dense_attribution_pixel_features(
            ts, attr_ref_indices, attr_post_indices,
            downsample=downsample,
        )
    attr_argmax = dense_predict(clf_attr, attr_feats).astype(np.int32)

    if det_mask.shape != attr_argmax.shape:
        raise ValueError(
            f"Spatial shape mismatch: detection {det_mask.shape} vs "
            f"attribution {attr_argmax.shape}. Use the same `downsample`."
        )

    gated = np.where(det_mask == 1, attr_argmax, -1).astype(np.int8)

    return {
        "detection_proba": det_proba,
        "detection_mask": det_mask,
        "attribution_argmax": attr_argmax,
        "gated_argmax": gated,
    }


# ---------------------------------------------------------------------------
# Notebook-facing orchestration: dense detection sweep + dense attribution map
# ---------------------------------------------------------------------------

def _evenly_spaced(seq: Sequence[int], n: int) -> list[int]:
    """Return up to ``n`` evenly-spaced items from ``seq``."""
    seq = list(seq)
    if len(seq) <= n:
        return seq
    step = (len(seq) - 1) / (n - 1)
    return [seq[round(i * step)] for i in range(n)]


def dense_detection_sweep(
    ts: TimeSeries,
    *,
    clf_fm: TorchClassifier,
    clf_bl: TorchClassifier,
    recipe: str = "full",
    n_frames: int = 6,
    fig_width: float = 14.0,
    show_swim_bubbles: bool | None = None,
    title: str | None = None,
):
    """Render the temporal-sweep figure for one sample.

    Selects ``n_frames`` evenly spaced across the eligible-query frames
    (split half pre-event / half post-event), runs the trained FM and
    baseline heads densely at each spatial location of every selected
    frame, and renders the result via :func:`sample_timeline` with two
    extra rows (heatmaps on the absolute (0, 1) probability scale).

    Returns ``(fig, sweep_idx)`` so the caller can re-use ``sweep_idx``
    downstream.
    """
    import datetime as _dt
    from .viz import dense_sweep_grid as _dense_sweep_grid

    ref_by_frame = {
        qi: ref for qi, _, ref in iter_detection_queries(ts)
    }
    ev_start = ts.event_period()["start"]
    dates_dt = [_dt.date.fromisoformat(d) for d in ts.dates]  # noqa: E501
    eligible_pre = [i for i in ref_by_frame if dates_dt[i] < ev_start]
    eligible_post = [i for i in ref_by_frame if dates_dt[i] >= ev_start]

    half = n_frames // 2
    sweep_idx = (
        _evenly_spaced(eligible_pre, half)
        + _evenly_spaced(eligible_post, n_frames - half)
    )

    def _fm_map(_ts: TimeSeries, i: int) -> np.ndarray:
        feats = dense_token_features(
            _ts, i, ref_by_frame[i], recipe=recipe,
        )
        return dense_predict(clf_fm, feats, return_proba=True)[..., 1]

    def _bl_map(_ts: TimeSeries, i: int) -> np.ndarray:
        feats = dense_pixel_features(
            _ts, i, ref_by_frame[i],
            downsample=16, recipe=recipe,
        )
        return dense_predict(clf_bl, feats, return_proba=True)[..., 1]

    fig = _dense_sweep_grid(
        ts,
        panels={
            "FM P(event)":       _fm_map,
            "Baseline P(event)": _bl_map,
        },
        frame_indices=sweep_idx,
        cmaps={"FM P(event)": "magma", "Baseline P(event)": "magma"},
        vmin=0.0,
        vmax=1.0,
        fig_width=fig_width,
        show_swim_bubbles=show_swim_bubbles,
        title=title or (
            f"sid={ts.sample_id} — dense detection sweep "
            f"(RGB row + FM / baseline P(event) heatmaps)"
        ),
    )
    return fig, sweep_idx


def dense_attribution_map(
    ts: TimeSeries,
    *,
    clf_fm_det: TorchClassifier,
    clf_bl_det: TorchClassifier,
    clf_fm_attr: TorchClassifier,
    clf_bl_attr: TorchClassifier,
    attr_codes: Sequence[int],
    attr_names: Sequence[str],
    attr_palette: Sequence[str],
    attr_to_idx: dict[int, int] | None = None,
    threshold: float = 0.7,
    det_recipe: str = "full",
    fig_width: float = 14.0,
    fig_height: float = 3.6,
    title: str | None = None,
    verbose: bool = True,
):
    """Render the detection-gated attribution figure for one sample.

    Picks a representative pre-event RGB (middle of the reference set)
    and the first post-event frame, runs both modalities through
    :func:`dense_attribution_gated` (each model gated by its own
    detector), and renders the four-panel layout via
    :func:`s2tutorial.viz.dense_attribution_figure`.

    Returns the figure.
    """
    from .viz import dense_attribution_figure as _dense_attribution_figure

    if attr_to_idx is None:
        attr_to_idx = {int(c): i for i, c in enumerate(attr_codes)}

    attr_ref_idx = references_at_event(ts)
    post_idx_seq = post_event_indices(ts)
    if not attr_ref_idx or not post_idx_seq:
        raise RuntimeError(
            f"sid={ts.sample_id}: insufficient frames for attribution"
        )
    pre_frame_idx = attr_ref_idx[len(attr_ref_idx) // 2]
    post_frame_idx = post_idx_seq[0]
    det_ref_idx = reference_indices(ts, post_frame_idx)

    true_code = int(ts.event_period()["label"])
    true_class_idx = attr_to_idx[true_code]

    gated_fm = dense_attribution_gated(
        ts,
        clf_det=clf_fm_det,
        clf_attr=clf_fm_attr,
        det_query_idx=post_frame_idx,
        det_ref_indices=det_ref_idx,
        attr_ref_indices=attr_ref_idx,
        attr_post_indices=post_idx_seq,
        mode="token",
        det_recipe=det_recipe,
        threshold=threshold,
    )
    gated_bl = dense_attribution_gated(
        ts,
        clf_det=clf_bl_det,
        clf_attr=clf_bl_attr,
        det_query_idx=post_frame_idx,
        det_ref_indices=det_ref_idx,
        attr_ref_indices=attr_ref_idx,
        attr_post_indices=post_idx_seq,
        mode="pixel",
        det_recipe=det_recipe,
        downsample=16,
        threshold=threshold,
    )
    if verbose:
        print(
            f"sid={ts.sample_id}  true class = {true_code} "
            f"({attr_names[true_class_idx]}); "
            f"FM mask {int(gated_fm['detection_mask'].sum())}/196 "
            f"above threshold, baseline mask "
            f"{int(gated_bl['detection_mask'].sum())}/196"
        )

    return _dense_attribution_figure(
        ts,
        pre_idx=pre_frame_idx,
        post_idx=post_frame_idx,
        fm_gated=gated_fm["gated_argmax"],
        baseline_gated=gated_bl["gated_argmax"],
        class_codes=list(attr_codes),
        class_names=list(attr_names),
        class_palette=list(attr_palette),
        true_class_idx=true_class_idx,
        fig_width=fig_width,
        fig_height=fig_height,
        title=title or (
            f"sid={ts.sample_id}, true class = "
            f"{true_code} {attr_names[true_class_idx]}, "
            f"detection threshold = {threshold:.2f}"
        ),
    )
