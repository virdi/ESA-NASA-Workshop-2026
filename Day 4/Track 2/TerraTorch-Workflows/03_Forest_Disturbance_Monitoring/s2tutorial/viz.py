"""Visualisation helpers for the workshop subset.

Top-level entry points:

- `sample_timeline(ts, …)` — the only time-series viz. Always renders a
  swim-lane (label periods as colored ribbons, the single disturbance
  event as a marker, cloud-free S2 acquisitions along the bottom) and
  optionally a thumbnail strip below it. The same call covers a
  sampled view (`n_patches=N`), every cloud-free frame
  (`n_patches=-1`), and a swim-lane only (`n_patches=0`); thumbnails
  can be RGB, SCL, or both via `mode`.
- `class_breakdown(meta, level)` — horizontal bar chart of period or
  frame counts at hierarchy level 1 / 2 / 3.
- `pca_token_image(emb_frame, n_components=3)` — turn one frame's token
  embedding map ``(H, W, D)`` into a min-max stretched false-colour
  image ``(H, W, 3)`` in [0, 1]. Per-frame PCA on tokens.

Stays minimal on purpose; copy / adapt freely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from .loader import TimeSeries


@dataclass
class ExtraRow:
    """One extra panel row stacked below the RGB strip in ``sample_timeline``.

    ``panel_fn(ts, frame_idx)`` must return either:
      * a ``(H, W)`` 2D array → rendered with ``cmap`` + ``vmin/vmax``
      * a ``(H, W, 3)`` RGB array in [0, 1] → rendered as-is.

    ``vmin`` / ``vmax`` apply globally (shared across all frames in the
    figure) so heatmaps are NOT per-frame normalised. Leave both ``None``
    to let imshow auto-scale per panel (matches the default RGB behaviour).
    """

    name: str
    panel_fn: Callable[[TimeSeries, int], np.ndarray]
    cmap: str | None = None
    vmin: float | None = None
    vmax: float | None = None
    overlay_center_marker: bool = False


# Short, figure-friendly class names. Derived from the workshop's
# label hierarchy; the loader carries the full names in classes.json.
SHORT_NAMES = {
    100: "Healthy", 110: "Undisturbed", 120: "Revegetation",
    121: "Re-veg w/ trees", 122: "Re-veg canopy closing",
    123: "Re-veg w/o trees",
    200: "Disturbed", 210: "Planned", 211: "Clear-Cut", 212: "Thinning",
    213: "Mulching",
    240: "Abiotic", 241: "Drought", 242: "Wildfire", 243: "Wind",
    244: "Avalanche", 245: "Flood",
}

# Stable colors per L2 group.
L2_PALETTE = {
    110: "#2e8b57", 120: "#a3c45c",
    210: "#1f77b4", 240: "#dc2626",
    100: "#bbbbbb", 200: "#737373",
}

# Distinct dingbats per event subtype (or concentric circle for
# `event_markers="circle"`).
EVENT_GLYPH = {
    211: "▼", 212: "✂", 213: "❀",
    241: "☀", 242: "▲", 243: "➤", 244: "❄", 245: "≈",
}
CIRCLE_GLYPH = "◉"

# SCL classes considered "land" — used to drive the RGB stretch so a
# snow / cloud panel can't squash vegetation into near-black.
_SCL_LAND_CLASSES = (4, 5, 6, 7)
_SCL_NODATA = 0
_PATCH_GRAY = 0.65

# Sen2Cor SCL palette (close to the official one). Used to render the
# scene-classification mask alongside the RGB strip so we can see
# *why* a frame looks bright (snow / bare) or dark (shadow).
SCL_COLORS = {
    0:  "#000000",  # no data
    1:  "#ff0000",  # saturated / defective
    2:  "#2f2f2f",  # dark area pixels
    3:  "#643200",  # cloud shadow
    4:  "#1aaf3d",  # vegetation
    5:  "#d6c54f",  # bare soil
    6:  "#0000ff",  # water
    7:  "#7e7e7e",  # unclassified
    8:  "#b6b6b6",  # cloud medium probability
    9:  "#ffffff",  # cloud high probability
    10: "#a4cce8",  # thin cirrus
    11: "#52f6f6",  # snow / ice
}
SCL_NAMES = {
    0: "no data", 1: "saturated", 2: "dark", 3: "shadow",
    4: "veg", 5: "bare", 6: "water", 7: "unclass.",
    8: "cloud (med)", 9: "cloud (hi)", 10: "cirrus", 11: "snow",
}


def _scl_to_rgb(scl: np.ndarray) -> np.ndarray:
    """Map SCL class indices → an HxWx3 RGB image in [0, 1]."""
    h, w = scl.shape
    out = np.zeros((h, w, 3), dtype=np.float32)
    for code, hex_color in SCL_COLORS.items():
        mask = scl == code
        if not mask.any():
            continue
        rgb = np.array([int(hex_color[i:i+2], 16) / 255.0 for i in (1, 3, 5)])
        out[mask] = rgb
    return out


def _crop_window(h: int, w: int, size: int,
                 offset: tuple[int, int] = (0, 0)) -> tuple[int, int, int, int]:
    """Compute (y0, y1, x0, x1) for a centered crop with optional offset.

    Crop window edges are clamped to the array bounds — passing an
    offset that would push the window past the edge silently snaps
    back so we still return a `size × size` slice.
    """
    half = size // 2
    cy = h // 2 + int(offset[0])
    cx = w // 2 + int(offset[1])
    y0 = max(0, min(h - size, cy - half))
    x0 = max(0, min(w - size, cx - half))
    return y0, y0 + size, x0, x0 + size


def _apply_crop(arr: np.ndarray, size: int | None,
                offset: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int]]:
    """Crop the last two axes to (size, size); return (crop, marker_yx).

    `marker_yx` is the (row, col) of the original patch-center pixel
    inside the cropped frame. If the crop is the full array
    (`size is None` or `size >= min(h,w)`), no slicing happens.
    """
    h, w = arr.shape[-2:]
    if size is None or size >= min(h, w):
        return arr, (h // 2, w // 2)
    y0, y1, x0, x1 = _crop_window(h, w, int(size), offset)
    cropped = arr[..., y0:y1, x0:x1]
    marker_y = h // 2 - y0
    marker_x = w // 2 - x0
    return cropped, (marker_y, marker_x)


def _l2(code: int) -> int:
    return (int(code) // 10) * 10


def _color(code: int) -> str:
    return L2_PALETTE.get(_l2(code), "#888888")


def _name(code: int) -> str:
    return SHORT_NAMES.get(int(code), str(code))


# ───────────────────────── timeline ───────────────────────────────────


def sample_timeline(
    ts: TimeSeries,
    *,
    n_patches: int = 6,
    ncols: int | None = None,
    highlight: Iterable[int] | None = None,
    draw_patches: bool = True,
    mode: str = "rgb",
    rgb: str = "true",
    clip_pct: tuple[float, float] = (2.0, 98.0),
    crop_size: int | None = None,
    crop_offset: tuple[int, int] = (0, 0),
    center_marker_radius: float = 6.0,
    figsize: tuple[float, float] | None = None,
    fig_width: float = 14.0,
    dpi: int = 130,
    background: str = "white",
    swim_height: float = 2.6,
    patch_axis_height: float = 2.0,
    patch_wspace: float = 0.10,
    patch_hspace: float = 0.30,
    s2_color: str = "#e07a5f",
    s2_marker: str = "|",
    s2_size: float = 600,
    s2_alpha: float = 1.0,
    s2_linewidth: float = 1.3,
    segment_thickness: float = 9,
    segment_alpha: float = 0.45,
    event_markers: str = "circle",
    event_marker_size: float = 16,
    title: str | None = None,
    show_swim_bubbles: bool | None = None,
    extra_rows: Sequence[ExtraRow] | None = None,
    target_marker_fn: Callable[[TimeSeries, int], dict | None] | None = None,
) -> plt.Figure:
    """Swimlane + (optional) thumbnail grid for one sample.

    `n_patches`:
      - positive int : evenly sample that many frames across the time series.
      - `-1`         : show every frame.
      - `0`          : skip thumbnails (swimlane only — same as
                       `draw_patches=False`).
      - overridden by `highlight` if given.

    `ncols` panels per row. `None` (default) auto-picks
    `min(n_patches_resolved, 10)` so a small `n_patches` stays on one row.

    `mode`:
      - `'rgb'` (default): one RGB row per batch of `ncols` frames.
      - `'scl'` : one SCL row per batch.
      - `'both'`: alternating RGB / SCL rows.

    `crop_size` / `crop_offset` (10 m pixel units): zoom each panel to
    a square crop centered on the annotated pixel (with optional shift).
    Default keeps the full 252 × 252 patch.

    `rgb`: `'true'` (B04/B03/B02) or `'false'` (B08/B04/B03 false-color).
    `clip_pct`: `(lo, hi)` percentiles for the shared brightness stretch.

    `event_markers`:
      - `'circle'` (default): ◉ coloured per L2 group (Planned / Abiotic),
                              legend names the specific L3 agent
                              (Clear-Cut / Thinning / Wildfire / Wind).
      - `'glyph'` : per-event-subtype glyphs (▼ ✂ ❀ for Planned;
                    ☀ ▲ ➤ ❄ ≈ for Abiotic).

    `show_swim_bubbles`: numbered bubbles linking obs ↔ thumbnails.
    `None` (default) auto-disables when more than 30 frames are shown
    (otherwise they overlap unreadably).
    """
    if event_markers not in ("circle", "glyph"):
        raise ValueError(f"event_markers must be 'circle' or 'glyph', got {event_markers!r}")
    if mode not in ("rgb", "scl", "both"):
        raise ValueError(f"mode must be 'rgb', 'scl', or 'both', got {mode!r}")

    obs_dates = [date.fromisoformat(d) for d in ts.dates]
    n_obs = len(obs_dates)

    # 1. Resolve which indices to show.
    if highlight is not None:
        indices = [int(i) for i in highlight]
    elif n_patches == -1:
        indices = list(range(n_obs))
    elif n_patches == 0:
        indices = []
    else:
        k = max(0, min(int(n_patches), n_obs))
        indices = list(np.linspace(0, n_obs - 1, k, dtype=int)) if k else []

    # `draw_patches=False` is equivalent to `n_patches=0`.
    if not draw_patches:
        indices = []

    # 2. Layout: rows of `ncols` panels each; `mode='both'` doubles rows;
    #    each ExtraRow adds one more row per batch.
    if ncols is None:
        ncols = min(len(indices), 10) if indices else 1
    ncols = max(1, int(ncols))
    n_batches = (len(indices) + ncols - 1) // ncols if indices else 0
    base_rows = 2 if mode == "both" else 1
    extras = list(extra_rows) if extra_rows else []
    rows_per_batch = (base_rows + len(extras)) if indices else 0
    n_thumb_rows = n_batches * rows_per_batch

    # 3. Bubble policy.
    if show_swim_bubbles is None:
        show_swim_bubbles = 0 < len(indices) <= 30

    # 4. Figure & gridspec.
    if figsize is None:
        figsize = (fig_width, swim_height + patch_axis_height * n_thumb_rows)
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=background)
    if n_thumb_rows:
        ratios = [swim_height] + [patch_axis_height] * n_thumb_rows
        gs = fig.add_gridspec(
            1 + n_thumb_rows, 1, height_ratios=ratios, hspace=patch_hspace,
        )
        ax_swim = fig.add_subplot(gs[0])
        thumb_specs = [gs[i + 1] for i in range(n_thumb_rows)]
    else:
        ax_swim = fig.add_subplot(1, 1, 1)
        thumb_specs = []

    _draw_swimlane(
        ax_swim, ts, obs_dates,
        indices if show_swim_bubbles else None,
        s2_color=s2_color, s2_marker=s2_marker, s2_size=s2_size,
        s2_alpha=s2_alpha, s2_linewidth=s2_linewidth,
        segment_thickness=segment_thickness,
        segment_alpha=segment_alpha,
        event_markers=event_markers,
        event_marker_size=event_marker_size,
    )

    if title is None:
        ev = ts.event_period()
        head = f"sample {ts.sample_id}  ·  {n_obs} cloud-free frames"
        if ev is not None:
            title = (f"{head}  ·  {_name(ev['label'])} event "
                     f"on {ev['start'].isoformat()}")
        else:
            title = head
    ax_swim.set_title(title, loc="left", fontsize=11, color="#111111", pad=8)

    # 5. Per-batch thumbnail rows.
    for batch in range(n_batches):
        i0, i1 = batch * ncols, min(batch * ncols + ncols, len(indices))
        batch_idx = indices[i0:i1]
        badge_start = i0 + 1
        if mode in ("rgb", "both"):
            row_offset = batch * rows_per_batch
            sub = thumb_specs[row_offset].subgridspec(1, ncols, wspace=patch_wspace)
            axes_row = [fig.add_subplot(sub[0, j]) for j in range(len(batch_idx))]
            for j in range(len(batch_idx), ncols):
                fig.add_subplot(sub[0, j]).axis("off")
            _draw_s2_thumbnails(
                axes_row, ts, batch_idx,
                rgb=rgb, clip_pct=clip_pct,
                badge_color=s2_color,
                center_marker_radius=center_marker_radius,
                crop_size=crop_size, crop_offset=crop_offset,
                badge_start=badge_start,
                target_marker_fn=target_marker_fn,
            )
        if mode in ("scl", "both"):
            scl_row_in_batch = base_rows - 1
            row_offset = batch * rows_per_batch + scl_row_in_batch
            sub = thumb_specs[row_offset].subgridspec(1, ncols, wspace=patch_wspace)
            axes_row = [fig.add_subplot(sub[0, j]) for j in range(len(batch_idx))]
            for j in range(len(batch_idx), ncols):
                fig.add_subplot(sub[0, j]).axis("off")
            _draw_scl_thumbnails(
                axes_row, ts, batch_idx,
                center_marker_radius=center_marker_radius,
                crop_size=crop_size, crop_offset=crop_offset,
                badge_start=badge_start,
                show_title=(mode == "scl"),
            )
        for ei, er in enumerate(extras):
            row_offset = batch * rows_per_batch + base_rows + ei
            sub = thumb_specs[row_offset].subgridspec(1, ncols, wspace=patch_wspace)
            axes_row = [fig.add_subplot(sub[0, j]) for j in range(len(batch_idx))]
            for j in range(len(batch_idx), ncols):
                fig.add_subplot(sub[0, j]).axis("off")
            _draw_extra_thumbnails(
                axes_row, ts, batch_idx, er,
                center_marker_radius=center_marker_radius,
            )

    # Some axes (the swim-lane has manual transforms) can't be tight-laid
    # out; the resulting matplotlib UserWarning is cosmetic noise here.
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()
    return fig


def _draw_swimlane(
    ax, ts: TimeSeries, obs_dates: list[date],
    highlight: list[int] | None,
    *,
    s2_color, s2_marker, s2_size, s2_alpha, s2_linewidth,
    segment_thickness, segment_alpha,
    event_markers: str, event_marker_size,
) -> None:
    Y_OBS, Y_SEG, Y_EVT, Y_TOP = 0.0, 0.85, 1.55, 2.10

    # Observations.
    if obs_dates:
        scatter_kw = dict(
            marker=s2_marker, s=s2_size, color=s2_color,
            alpha=s2_alpha, linewidths=s2_linewidth, zorder=2,
        )
        # `|` and `_` are stroke-only markers — passing `edgecolors`
        # triggers a matplotlib warning. Skip it for those.
        if s2_marker not in ("|", "_"):
            scatter_kw["edgecolors"] = s2_color
        ax.scatter(obs_dates, [Y_OBS] * len(obs_dates), **scatter_kw)

    # Periods + event.
    t_min = date.fromisoformat(ts.window_start)
    t_max = date.fromisoformat(ts.window_end)
    seen_seg: dict[int, str] = {}
    seen_evt: dict[int, str] = {}
    for r in ts.labels.iter_rows(named=True):
        c = _color(r["label"])
        is_event = bool(r["is_event"])
        start = r["start"]
        end = r["end_evidence"] or r["end_validity"] or t_max
        if is_event:
            if event_markers == "circle":
                glyph = CIRCLE_GLYPH
            else:
                glyph = EVENT_GLYPH.get(int(r["label"]), "●")
            ax.text(start, Y_EVT, glyph,
                    ha="center", va="center",
                    fontsize=event_marker_size, color=c, zorder=5)
            ax.plot([start, start], [Y_OBS + 0.10, Y_EVT - 0.18],
                    color=c, alpha=0.20, linewidth=0.7, zorder=1,
                    solid_capstyle="round")
            seen_evt.setdefault(int(r["label"]), c)
        else:
            ax.plot([start, end], [Y_SEG, Y_SEG],
                    color=c, alpha=segment_alpha,
                    linewidth=segment_thickness, solid_capstyle="round",
                    zorder=3)
            ax.plot([start, end], [Y_SEG, Y_SEG],
                    color=c, alpha=0.95,
                    linewidth=max(1.2, segment_thickness * 0.18),
                    solid_capstyle="round", zorder=4)
            seen_seg.setdefault(_l2(r["label"]), c)

    # Numbered highlight bubbles linking obs ↔ thumbnails.
    if highlight is not None:
        for n, idx in enumerate(highlight, start=1):
            d = obs_dates[idx]
            ax.scatter([d], [Y_OBS], marker="o", s=180,
                       color="white", edgecolors="#222222", linewidths=0.9,
                       zorder=4)
            ax.text(d, Y_OBS, str(n), ha="center", va="center",
                    fontsize=8, color="#222222", zorder=5)

    # X axis.
    ax.set_xlim(t_min - timedelta(days=30), t_max + timedelta(days=30))
    ax.set_ylim(-0.55, Y_TOP)
    ax.set_yticks([])
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", length=3, width=0.5, color="#666666", labelsize=9)
    for side, sp in ax.spines.items():
        sp.set_visible(side == "bottom")
        if side == "bottom":
            sp.set_color("#999999")
            sp.set_linewidth(0.5)

    from matplotlib.transforms import blended_transform_factory
    bt = blended_transform_factory(ax.transAxes, ax.transData)
    for y, txt in [(Y_OBS, "Obs"), (Y_SEG, "Periods"), (Y_EVT, "Event")]:
        ax.text(-0.005, y, txt, transform=bt,
                ha="right", va="center", color="#555555", fontsize=8)

    handles = []
    for l2, c in sorted(seen_seg.items()):
        handles.append(plt.Line2D([], [], linestyle="-", color=c,
                                  linewidth=4, alpha=0.75,
                                  solid_capstyle="round",
                                  label=_name(l2)))
    for code, c in sorted(seen_evt.items()):
        handles.append(plt.Line2D([], [], marker="o", linestyle="None",
                                  markerfacecolor=c, markeredgecolor="none",
                                  markersize=8, label=f"{_name(code)} (event)"))
    handles.append(plt.Line2D([], [], marker=s2_marker, linestyle="None",
                              markerfacecolor=s2_color,
                              markeredgecolor=s2_color,
                              markersize=8, label="cloud-free S2"))
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=8,
              handlelength=1.4, ncol=len(handles))


# ───────────────────────── thumbnails ─────────────────────────────────


def _compute_shared_stretch(
    cubes: list[np.ndarray | None],
    scl_arrays: list[np.ndarray | None],
    clip_pct: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel (lo, hi) percentiles over land-class pixels across
    all displayed cubes — keeps stretch consistent panel-to-panel.
    Falls back to all-pixel percentiles if no land pixels exist.
    """
    per_ch: list[list[np.ndarray]] = [[], [], []]
    for c, sc in zip(cubes, scl_arrays):
        if c is None:
            continue
        if sc is not None:
            mask = np.isin(sc, _SCL_LAND_CLASSES)
        else:
            mask = c[0] > 0
        if mask.sum() == 0:
            continue
        for i in range(3):
            per_ch[i].append(c[i][mask])
    if all(len(v) > 0 for v in per_ch):
        lo = np.array([np.percentile(np.concatenate(v), clip_pct[0]) for v in per_ch])
        hi = np.array([np.percentile(np.concatenate(v), clip_pct[1]) for v in per_ch])
        return lo, np.maximum(hi, lo + 1e-6)
    valid = [c for c in cubes if c is not None]
    if valid:
        stack = np.concatenate([c.reshape(3, -1) for c in valid], axis=1)
        lo = np.percentile(stack, clip_pct[0], axis=1)
        hi = np.percentile(stack, clip_pct[1], axis=1)
        return lo, np.maximum(hi, lo + 1e-6)
    return np.zeros(3), np.ones(3)


def _draw_s2_thumbnails(
    axes: list, ts: TimeSeries, indices: list[int],
    *,
    rgb: str = "true",
    clip_pct: tuple[float, float] = (2.0, 98.0),
    badge_color: str = "#e07a5f",
    center_marker_radius: float = 6.0,
    crop_size: int | None = None,
    crop_offset: tuple[int, int] = (0, 0),
    badge_start: int = 1,
    target_marker_fn: Callable[[TimeSeries, int], dict | None] | None = None,
) -> None:
    """Render an RGB strip on the given list of axes.

    Bands in s2_10m are stored [B02, B03, B04, B08]. For natural color
    the channel order is [B04, B03, B02] = indices [2, 1, 0]. For the
    classic NIR-R-G false-color it is [B08, B04, B03] = [3, 2, 1].

    Stretch is computed once across all displayed frames so panel
    brightness is comparable; SCL NODATA pixels are filled with neutral
    gray; an open white circle marks the annotated pixel.

    `crop_size` (in 10 m pixel units) crops each panel to a square of
    that size centered on the annotated pixel. Default `None` keeps
    the full 252×252 patch. `crop_offset` shifts the crop window by
    `(dy, dx)` 10 m pixels — also default `(0, 0)`.
    """
    if rgb == "true":
        ch = (2, 1, 0)
    elif rgb == "false":
        ch = (3, 2, 1)
    else:
        raise ValueError(f"s2_rgb must be 'true' or 'false', got {rgb!r}")

    cubes: list[np.ndarray | None] = []
    scls: list[np.ndarray | None] = []
    marker_xy: list[tuple[int, int] | None] = []
    for i in indices:
        arr = ts.s2_10m(i)  # (4, H, W) raw uint16
        if arr.shape[0] < 4 or arr.max() == 0:
            cubes.append(None); scls.append(None); marker_xy.append(None)
            continue
        sub = arr[list(ch), ...].astype(np.float32)
        sub_cropped, (my, mx) = _apply_crop(sub, crop_size, crop_offset)
        cubes.append(sub_cropped)
        marker_xy.append((mx, my))
        # SCL is at half resolution → halve crop_size and offset for masking.
        scl = ts.s2_scl(i)
        if crop_size is not None:
            scl_cropped, _ = _apply_crop(
                scl, max(1, crop_size // 2),
                (crop_offset[0] // 2, crop_offset[1] // 2),
            )
            scl_up = np.repeat(np.repeat(scl_cropped, 2, axis=0), 2, axis=1)
        else:
            scl_up = np.repeat(np.repeat(scl, 2, axis=0), 2, axis=1)
        scls.append(scl_up)

    lo, hi = _compute_shared_stretch(cubes, scls, clip_pct)
    denom = hi - lo

    for badge_n, (ax, c, sc, i, mxy) in enumerate(
            zip(axes, cubes, scls, indices, marker_xy), start=badge_start):
        if c is None:
            ax.text(0.5, 0.5, "no\nclear\nS2", ha="center", va="center",
                    transform=ax.transAxes, color="#888888", fontsize=8)
            ax.axis("off")
            continue
        rgb_arr = (c - lo[:, None, None]) / denom[:, None, None]
        rgb_arr = np.clip(rgb_arr, 0, 1).transpose(1, 2, 0)
        if sc is not None and sc.shape == rgb_arr.shape[:2]:
            rgb_arr = np.where((sc == _SCL_NODATA)[..., None], _PATCH_GRAY, rgb_arr)
        ax.imshow(rgb_arr, interpolation="nearest")
        ax.set_title(ts.dates[i], fontsize=9, color="#222222", pad=3)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#cccccc")
            sp.set_linewidth(0.5)
        # White circle at the annotated pixel (may be off-center after crop).
        if mxy is not None:
            mx, my = mxy
            h, w = rgb_arr.shape[:2]
            if 0 <= my < h and 0 <= mx < w:
                ax.add_patch(mpatches.Circle(
                    (mx, my), radius=center_marker_radius,
                    fill=False, edgecolor="white", linewidth=1.2, zorder=5,
                ))
        # Numbered badge anchored to the panel top-left in axes-fraction
        # coords — stays put regardless of crop_size or panel zoom.
        ax.scatter([0.12], [0.86], marker="o", s=200,
                   color="white", edgecolors=badge_color, linewidths=1.4,
                   zorder=6, transform=ax.transAxes, clip_on=False)
        ax.text(0.12, 0.86, str(badge_n), ha="center", va="center",
                fontsize=9, color="#222222", zorder=7,
                transform=ax.transAxes, clip_on=False)
        # Optional target-label marker (e.g. for dense attribution maps:
        # a colored ring at the annotated pixel showing the true class).
        if target_marker_fn is not None and mxy is not None:
            spec = target_marker_fn(ts, i)
            if spec is not None:
                mx, my = mxy
                ax.add_patch(mpatches.Circle(
                    (mx, my),
                    radius=spec.get("radius", center_marker_radius * 1.7),
                    fill=False,
                    edgecolor=spec.get("color", "#dc2626"),
                    linewidth=spec.get("linewidth", 2.2),
                    zorder=6,
                ))


def _draw_extra_thumbnails(
    axes: list,
    ts: TimeSeries,
    indices: list[int],
    extra: ExtraRow,
    *,
    center_marker_radius: float = 6.0,
) -> None:
    """Render one ExtraRow as a strip of panels, one per frame.

    Shared (vmin, vmax) — if both set on the ExtraRow — gives an absolute
    colour scale across all frames in the figure (no per-frame
    renormalisation). ``overlay_center_marker=True`` draws the same
    white circle at the annotated pixel that the RGB row uses, which is
    useful for token-PCA / detection-probability maps.
    """
    if not axes:
        return
    imshow_kwargs: dict[str, Any] = {}
    if extra.cmap is not None:
        imshow_kwargs["cmap"] = extra.cmap
    if extra.vmin is not None:
        imshow_kwargs["vmin"] = extra.vmin
    if extra.vmax is not None:
        imshow_kwargs["vmax"] = extra.vmax
    for ax, i in zip(axes, indices):
        img = extra.panel_fn(ts, i)
        kw = {} if (img.ndim == 3) else imshow_kwargs
        ax.imshow(img, interpolation="nearest", **kw)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#cccccc")
            sp.set_linewidth(0.5)
        if extra.overlay_center_marker:
            h, w = img.shape[:2]
            ax.add_patch(mpatches.Circle(
                (w // 2, h // 2),
                radius=max(1.0, center_marker_radius * (w / 252.0)),
                fill=False,
                edgecolor="white",
                linewidth=1.0,
                zorder=5,
            ))
    # Row label on the leftmost panel, vertical text outside the axes.
    axes[0].text(
        -0.08, 0.5, extra.name,
        transform=axes[0].transAxes,
        ha="right", va="center",
        rotation=90, fontsize=9, color="#444444",
    )


def _draw_scl_thumbnails(
    axes: list, ts: TimeSeries, indices: list[int],
    *,
    center_marker_radius: float = 6.0,
    show_title: bool = False,
    crop_size: int | None = None,
    crop_offset: tuple[int, int] = (0, 0),
    badge_start: int = 1,
) -> None:
    """Render an SCL-coloured strip on the given axes.

    `crop_size` and `crop_offset` are in 10 m units (matching the RGB
    path); SCL itself is at 20 m so they are halved internally.
    """
    scl_size = None if crop_size is None else max(1, int(crop_size) // 2)
    scl_offset = (int(crop_offset[0]) // 2, int(crop_offset[1]) // 2)
    radius = center_marker_radius / 2  # SCL is half-resolution
    for badge_n, (ax, i) in enumerate(zip(axes, indices), start=1):
        scl = ts.s2_scl(i)
        scl_cropped, (my, mx) = _apply_crop(scl, scl_size, scl_offset)
        rgb_arr = _scl_to_rgb(scl_cropped)
        ax.imshow(rgb_arr, interpolation="nearest")
        if show_title:
            ax.set_title(ts.dates[i], fontsize=9, color="#222222", pad=3)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#cccccc")
            sp.set_linewidth(0.5)
        h, w = rgb_arr.shape[:2]
        if 0 <= my < h and 0 <= mx < w:
            ax.add_patch(mpatches.Circle(
                (mx, my), radius=radius, fill=False,
                edgecolor="white", linewidth=1.0, zorder=5,
            ))


# ───────────────────────── class breakdown ────────────────────────────


def class_breakdown(
    meta: dict,
    *,
    level: int = 2,
    weight_by: str = "periods",
    figsize: tuple[float, float] = (8, 3.5),
    dpi: int = 130,
) -> plt.Figure:
    """Horizontal bar chart of class proportions in `labels.parquet`.

    Parameters
    ----------
    meta : dict from `load_metadata(root)`.
    level : 1, 2, or 3 — hierarchy collapse level.
    weight_by :
        - 'periods' : count one per labels.parquet row (one period);
        - 'frames'  : count one per (sample_id, date) frame, after
                      a date-aware join against labels.parquet.
    """
    if level not in (1, 2, 3):
        raise ValueError("level must be 1, 2, or 3")

    labels: pl.DataFrame = meta["labels"]
    classes: dict[int, str] = meta["classes"]

    if weight_by == "periods":
        df = labels.select("label")
    elif weight_by == "frames":
        frames: pl.DataFrame = meta["frames"]
        labels_sorted = labels.select("sample_id", "label", "start").sort("start")
        joined = frames.sort("date").join_asof(
            labels_sorted, left_on="date", right_on="start",
            by="sample_id", strategy="backward",
        ).filter(pl.col("label").is_not_null())
        df = joined.select("label")
    else:
        raise ValueError("weight_by must be 'periods' or 'frames'")

    if level == 1:
        codes = df.with_columns((pl.col("label") // 100 * 100).alias("code"))
    elif level == 2:
        codes = df.with_columns((pl.col("label") // 10 * 10).alias("code"))
    else:
        codes = df.with_columns(pl.col("label").alias("code"))

    counts = (
        codes.group_by("code").len()
        .rename({"len": "count"})
        .sort("code")
    )

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    code_arr = counts["code"].to_list()
    count_arr = counts["count"].to_list()
    names = [SHORT_NAMES.get(c, classes.get(c, str(c))) for c in code_arr]
    colors = [_color(c) for c in code_arr]
    y = np.arange(len(code_arr))
    ax.barh(y, count_arr, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(f"# {weight_by}", fontsize=9)
    ax.set_title(f"Level-{level} breakdown ({weight_by})",
                 loc="left", fontsize=10, pad=6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    return fig


def fit_dataset_pca(
    emb_frames: np.ndarray,
    *,
    n_components: int = 3,
    stretch_percentiles: tuple[float, float] = (2.0, 98.0),
) -> dict:
    """Fit a single PCA on tokens pooled across many frames.

    ``emb_frames`` is ``(N, H, W, D)`` or ``(N, H*W, D)`` — N frames
    concatenated along the leading axis. Returns a serialisable dict::

        {
          "mean":  (D,),
          "components": (n_components, D),
          "proj_lo": (n_components,),
          "proj_hi": (n_components,),
        }

    The ``proj_lo`` / ``proj_hi`` are dataset-wide percentile clip
    points on the projected values, so ``pca_token_image`` can apply a
    *consistent* stretch across all frames in the dataset (rather than
    per-frame min/max, which was the source of the inter-frame
    colour-scheme drift the user reported).
    """
    from sklearn.decomposition import PCA

    if emb_frames.ndim == 4:
        N, H, W, D = emb_frames.shape
        flat = emb_frames.reshape(N * H * W, D)
    elif emb_frames.ndim == 3:
        N, NT, D = emb_frames.shape
        flat = emb_frames.reshape(N * NT, D)
    else:
        raise ValueError(
            f"fit_dataset_pca expects (N, H, W, D) or (N, T, D); "
            f"got {emb_frames.shape}"
        )
    flat = np.ascontiguousarray(flat, dtype=np.float32)
    pca = PCA(n_components=n_components)
    proj_all = pca.fit_transform(flat)
    p_lo, p_hi = stretch_percentiles
    return {
        "mean": pca.mean_.astype(np.float32),
        "components": pca.components_.astype(np.float32),
        "proj_lo": np.percentile(proj_all, p_lo, axis=0).astype(np.float32),
        "proj_hi": np.percentile(proj_all, p_hi, axis=0).astype(np.float32),
        "stretch_percentiles": np.asarray(stretch_percentiles,
                                          dtype=np.float32),
        "explained_variance_ratio":
            pca.explained_variance_ratio_.astype(np.float32),
    }


def save_dataset_pca(state: dict, path: str | Path) -> None:
    """Save a fitted dataset-PCA state to .npz."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(path), **{k: np.asarray(v) for k, v in state.items()})


def load_dataset_pca(path: str | Path) -> dict:
    """Load a fitted dataset-PCA state from .npz.

    Accepts a local path or an ``s3://`` URI.
    """
    from . import _paths as _p
    if _p.is_uri(path):
        with np.load(_p.open_buffered(path)) as f:
            return {k: np.asarray(f[k]) for k in f.files}
    with np.load(str(path)) as f:
        return {k: np.asarray(f[k]) for k in f.files}


def pca_token_image(
    emb_frame: np.ndarray,
    pca_state: dict | None = None,
    *,
    n_components: int = 3,
) -> np.ndarray:
    """False-colour PCA image of one frame's tokens.

    Two modes:

    1. **Dataset-wide** (``pca_state`` provided, recommended): use the
       precomputed components + dataset percentile clip points. Colours
       are *consistent across frames* — a healthy region in two
       different frames carries the same false-colour. This is the
       behaviour you want for the "PCA of tokens reveals hidden
       structure" climax.

    2. **Per-frame** (``pca_state=None``, back-compat): fit PCA on this
       frame's tokens alone, then per-component min-max stretch.
       Convenient for one-off inspection but colour basis is
       arbitrary frame-to-frame.

    Returns ``(H, W, n_components)`` float32 in ``[0, 1]``.
    """
    if emb_frame.ndim != 3:
        raise ValueError(
            f"pca_token_image expects (H, W, D), got {emb_frame.shape}"
        )
    H, W, D = emb_frame.shape
    flat = emb_frame.reshape(H * W, D).astype(np.float32)

    if pca_state is None:
        from sklearn.decomposition import PCA
        proj = PCA(n_components=n_components).fit_transform(flat)
        img = proj.reshape(H, W, n_components)
        lo = img.min(axis=(0, 1), keepdims=True)
        hi = img.max(axis=(0, 1), keepdims=True)
        span = np.maximum(hi - lo, 1e-9)
        return ((img - lo) / span).astype(np.float32)

    # Dataset-wide projection + clip.
    mean = np.asarray(pca_state["mean"], dtype=np.float32)
    comps = np.asarray(pca_state["components"], dtype=np.float32)
    proj = (flat - mean) @ comps[:n_components].T  # (H*W, n_components)
    img = proj.reshape(H, W, n_components)
    lo = np.asarray(pca_state["proj_lo"], dtype=np.float32)[:n_components]
    hi = np.asarray(pca_state["proj_hi"], dtype=np.float32)[:n_components]
    span = np.maximum(hi - lo, 1e-9)
    img = (img - lo) / span
    return np.clip(img, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Workshop-tailored figures (Section 1 / 2 of notebooks/workshop.ipynb)
# ---------------------------------------------------------------------------

# Codes shown in the event-class breakdown — matches the 4 disturbance agents
# that label every sample in the subset (one event per sample).
DISTURBANCE_CODES: tuple[int, ...] = (211, 212, 242, 243)


def event_class_breakdown(
    meta: dict,
    *,
    codes: tuple[int, ...] = DISTURBANCE_CODES,
    figsize: tuple[float, float] = (7, 3.0),
    dpi: int = 130,
    cmap: str = "tab10",
    title: str | None = "Disturbance events in the subset",
) -> plt.Figure:
    """Bar chart of *event* counts per disturbance code.

    Counts the number of samples whose single event has each ``code`` —
    not the level-2 hierarchy that ``class_breakdown`` shows. Used in
    ``notebooks/workshop.ipynb`` Section 1 to display only the four
    disturbance agents (no Undisturbed / Revegetation placeholders).
    """
    labels: pl.DataFrame = meta["labels"]
    classes: dict[int, str] = meta["classes"]

    counts = (
        labels.filter(pl.col("label").is_in(list(codes)))
        .group_by("label")
        .agg(pl.len().alias("n"))
    )
    have = {int(r["label"]): int(r["n"]) for r in counts.iter_rows(named=True)}
    n_per_code = [have.get(int(c), 0) for c in codes]
    names = [classes.get(int(c), str(c)).replace("Forestry ", "") for c in codes]
    cm = plt.get_cmap(cmap, max(4, len(codes)))
    colors = [cm(i) for i in range(len(codes))]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    bars = ax.barh(range(len(codes)), n_per_code,
                   color=colors, edgecolor="white", linewidth=0.6)
    ax.invert_yaxis()
    ax.set_yticks(range(len(codes)),
                  [f"{int(c)}  {n}" for c, n in zip(codes, names)],
                  fontsize=9)
    ax.set_xlabel("# samples", fontsize=9)
    if title:
        ax.set_title(title, loc="left", fontsize=10, pad=6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)
    for bar, v in zip(bars, n_per_code):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                str(v), va="center", ha="left", fontsize=9, color="#444")
    ax.set_xlim(0, max(n_per_code) * 1.15)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Δt(clear S2) evolution — workshop port of disfor.tools.viz.delta_t_evolution.
# Workshop's meta["frames"] is already cloud-free S2 only and has no
# ``sensor`` / ``cloud_frac`` columns, so we drop the filter logic.
# ---------------------------------------------------------------------------

def _smooth_1d_workshop(arr: np.ndarray, window: int, kind: str = "gaussian") -> np.ndarray:
    """Edge-aware 1-D smoother. Mirrors disfor.tools.viz._smooth_1d."""
    if window <= 0:
        return arr
    a = np.asarray(arr, dtype=float)
    finite = np.isfinite(a)
    if kind == "gaussian":
        radius = max(1, int(np.ceil(3.0 * float(window))))
        xs = np.arange(-radius, radius + 1, dtype=float)
        kernel = np.exp(-(xs ** 2) / (2.0 * float(window) ** 2))
    elif kind == "rolling":
        kernel = np.ones(int(window), dtype=float)
    else:
        raise ValueError("kind must be 'gaussian' or 'rolling'")
    n = a.size
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        if not finite[i]:
            continue
        lo = max(0, i - kernel.size // 2)
        hi = min(n, i + kernel.size // 2 + (kernel.size % 2))
        kk = kernel[
            (kernel.size // 2 - (i - lo)) : (kernel.size // 2 + (hi - i))
        ]
        vv = a[lo:hi]
        mm = finite[lo:hi]
        if mm.any():
            out[i] = np.average(vv[mm], weights=kk[mm])
    return out


def _format_workshop_date_axis(ax: plt.Axes, n_years: int) -> None:
    if n_years <= 2:
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    else:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", labelsize=8)


def delta_t_evolution(
    frames: "pl.DataFrame",
    *,
    freq: str = "1w",
    primary_stat: str = "median",
    show_iqr: bool = True,
    show_mean: bool = False,
    show_std: bool = False,
    smooth: int = 2,
    smooth_kind: str = "gaussian",
    trim_edges: int | str | None = 5,
    cmap: str = "coolwarm",
    band_alpha: float = 0.5,
    reference_lines: list[float] | None = None,
    figsize: tuple[float, float] = (12, 4.0),
    dpi: int = 130,
    title: str | None = "Δt(clear S2)",
    ymax: float | None = None,
) -> plt.Figure:
    """Per-bin distribution of per-sample Δt(clear S2) over time.

    Workshop variant of ``disfor.tools.viz.delta_t_evolution``: the input
    ``frames`` has only ``{sample_id, date}`` (already cloud-free S2),
    so the original sensor / cloud-fraction filtering is dropped.

    For each ``(sample_id)``, dates are sorted + de-duplicated and
    ``Δt[k] = date[k] − date[k−1]`` (days). Δt is attributed to the
    midpoint between the two acquisitions, binned in time at ``freq``,
    and aggregated across samples.

    The primary curve is colour-encoded by its own y-value (so colder
    cells = denser revisits) and overlaid on a IQR band by default.
    """
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    if primary_stat not in ("median", "mean"):
        raise ValueError("primary_stat must be 'median' or 'mean'")
    refs = reference_lines if reference_lines is not None else [5.0, 10.0]

    sub = (
        frames.select(["sample_id", "date"])
        .unique(maintain_order=False)
        .sort(["sample_id", "date"])
    )
    sub = sub.with_columns(pl.col("date").shift(1).over("sample_id").alias("_prev"))
    sub = sub.with_columns(
        (pl.col("date") - pl.col("_prev")).dt.total_days().alias("dt"),
    ).drop_nulls("dt")
    sub = sub.with_columns(
        (pl.col("_prev") + (pl.col("date") - pl.col("_prev")) / 2).alias("stamp_date"),
    )

    binned = (
        sub.sort("stamp_date")
        .group_by_dynamic("stamp_date", every=freq)
        .agg(
            pl.col("dt").median().alias("median"),
            pl.col("dt").mean().alias("mean"),
            pl.col("dt").std().alias("std"),
            pl.col("dt").quantile(0.25).alias("q25"),
            pl.col("dt").quantile(0.75).alias("q75"),
            pl.len().alias("n"),
        )
        .sort("stamp_date")
    )

    xs = binned["stamp_date"].to_list()
    median = np.asarray(binned["median"].to_list(), dtype=float)
    mean = np.asarray(binned["mean"].to_list(), dtype=float)
    std = np.asarray(binned["std"].to_list(), dtype=float)
    q25 = np.asarray(binned["q25"].to_list(), dtype=float)
    q75 = np.asarray(binned["q75"].to_list(), dtype=float)

    if smooth and smooth > 0:
        median = _smooth_1d_workshop(median, smooth, smooth_kind)
        mean = _smooth_1d_workshop(mean, smooth, smooth_kind)
        std = _smooth_1d_workshop(std, smooth, smooth_kind)
        q25 = _smooth_1d_workshop(q25, smooth, smooth_kind)
        q75 = _smooth_1d_workshop(q75, smooth, smooth_kind)

    if trim_edges == "auto":
        n_trim = (int(np.ceil(2.0 * float(smooth))) if smooth_kind == "gaussian"
                  else max(0, int(smooth) // 2)) if smooth else 0
    elif trim_edges is None:
        n_trim = 0
    else:
        n_trim = int(trim_edges)
    if n_trim > 0 and len(xs) > 2 * n_trim:
        xs = xs[n_trim:-n_trim]
        median = median[n_trim:-n_trim]
        mean = mean[n_trim:-n_trim]
        std = std[n_trim:-n_trim]
        q25 = q25[n_trim:-n_trim]
        q75 = q75[n_trim:-n_trim]

    primary_y = median if primary_stat == "median" else mean
    secondary_y = mean if primary_stat == "median" else median
    secondary_label = "mean" if primary_stat == "median" else "median"

    cm = plt.get_cmap(cmap)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    for r in refs:
        ax.axhline(r, color="#cccccc", linestyle=(0, (2, 3)), linewidth=0.7, zorder=1)
        ax.text(xs[-1], r, f"  {r:g}d", color="#888888",
                fontsize=7.5, va="center", ha="left", zorder=1)

    if show_iqr:
        ax.fill_between(xs, q25, q75,
                        color=cm(0.55), alpha=band_alpha, linewidth=0, zorder=2)
    if show_std and show_mean:
        with np.errstate(invalid="ignore"):
            lo_s = np.where(np.isfinite(std), mean - std, np.nan)
            hi_s = np.where(np.isfinite(std), mean + std, np.nan)
        ax.fill_between(xs, lo_s, hi_s,
                        color=cm(0.4), alpha=0.18, linewidth=0, zorder=2)

    if show_mean:
        ax.plot(xs, secondary_y, color="#888888", linewidth=0.9,
                linestyle=(0, (4, 2)), alpha=0.85, zorder=3)

    # Primary curve: colored continuously by its own y-value via LineCollection.
    if len(xs) >= 2:
        mdates_x = mdates.date2num(xs)
        n_orig = len(mdates_x)
        sub_factor = max(2, int(np.ceil(800 / (n_orig - 1))))
        orig_idx = np.arange(n_orig, dtype=float)
        dense_idx = np.linspace(0.0, n_orig - 1, sub_factor * (n_orig - 1) + 1)
        dense_x = np.interp(dense_idx, orig_idx, mdates_x)
        dense_y = np.interp(dense_idx, orig_idx, primary_y)
        pts = np.column_stack([dense_x, dense_y])
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        seg_vals = (dense_y[:-1] + dense_y[1:]) / 2.0
        vmax = ymax if ymax is not None else float(np.nanpercentile(primary_y, 98))
        vmin = max(0.0, float(np.nanmin(primary_y)) * 0.9)
        norm = Normalize(vmin=vmin, vmax=vmax)
        lc = LineCollection(
            segs, cmap=cm, norm=norm,
            linewidths=2.4, capstyle="round", joinstyle="round",
            zorder=4,
        )
        lc.set_array(seg_vals)
        ax.add_collection(lc)
        cbar = fig.colorbar(lc, ax=ax, shrink=0.7, pad=0.015, aspect=30)
        cbar.set_label(f"{primary_stat} Δt (days)", fontsize=9, color="#444444")
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=8, length=0, which="both")

    if len(xs):
        ax.text(xs[-1], primary_y[-1], f" {primary_stat}",
                color="#222222", fontsize=9, fontweight="semibold",
                va="center", ha="left", zorder=5)
        if show_mean:
            ax.text(xs[-1], secondary_y[-1], f" {secondary_label}",
                    color="#888888", fontsize=8.5, va="center", ha="left", zorder=5)

    if ymax is not None:
        ax.set_ylim(0, ymax)
    else:
        ymax_auto = float(np.nanpercentile(q75, 99)) * 1.1
        ax.set_ylim(0, ymax_auto)

    n_years = max(1, (max(xs) - min(xs)).days // 365)
    _format_workshop_date_axis(ax, n_years)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax.set_ylabel("Δt (days)", fontsize=9, color="#444444")
    if title:
        ax.set_title(title, loc="left", fontsize=11, pad=8, color="#111111")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Token-PCA grid + embedding trajectory (Sections 1 & 2)
# ---------------------------------------------------------------------------

def token_pca_grid(
    ts: TimeSeries,
    pca_state: dict,
    *,
    n_patches: int = 6,
    frame_indices: Iterable[int] | None = None,
    fig_width: float = 14.0,
    title: str | None = None,
    show_swim_bubbles: bool | None = None,
) -> plt.Figure:
    """RGB strip + dataset-PCA token strip below it, sharing one swimlane.

    Thin wrapper around ``sample_timeline``: adds a single ExtraRow that
    renders ``pca_token_image(ts.tm_emb(i), pca_state=pca_state)`` per
    frame. Pass ``frame_indices`` to choose exact frames, or rely on the
    evenly-spaced ``n_patches`` default.
    """
    return sample_timeline(
        ts,
        n_patches=n_patches,
        highlight=frame_indices,
        mode="rgb",
        fig_width=fig_width,
        title=title,
        show_swim_bubbles=show_swim_bubbles,
        extra_rows=[ExtraRow(
            name="Token PCA",
            panel_fn=lambda t, i: pca_token_image(t.tm_emb(i), pca_state=pca_state),
            overlay_center_marker=True,
        )],
    )


def _seasonal_cmap():
    """Cyclic cold-blue → flower-pink → leaf-green → pumpkin-orange palette.

    Anchored at the four mid-season months (Feb=blue, May=pink, Aug=green,
    Nov=orange), with a transitional brown bridging Nov ↔ Feb across
    Dec/Jan so the palette is continuous along the year wheel.
    """
    import matplotlib.colors as mcolors
    return mcolors.LinearSegmentedColormap.from_list(
        "seasons",
        [(0.0,    "#8a76a0"),   # Dec/Jan transition
         (1 / 12, "#3A6FB0"),   # Feb — cold blue
         (4 / 12, "#E8A2C4"),   # May — flower pink
         (7 / 12, "#3FA34D"),   # Aug — leaf green
         (10 / 12, "#E47C2C"),  # Nov — pumpkin orange
         (1.0,    "#8a76a0")],  # Dec/Jan transition (closes the cycle)
    )


def embedding_trajectory(
    ts: TimeSeries,
    pca_state: dict,
    *,
    figsize: tuple[float, float] = (13, 5.3),
    dpi: int = 130,
    marker_size: float = 110,
    period_colors: tuple[str, str] = ("#1b5e20", "#d96666"),
    title: str | None = None,
) -> None:
    """Two-panel embedding-space trajectory with an interactive time slider.

    Projects spatial-mean-pooled ``ts.tm_emb(i)`` through the **dataset-wide**
    PCA basis from ``pca_state`` (same basis the token-PCA grid uses).

    Layout:
      * Left panel: markers fill-coloured by month (seasonal palette —
        cold blue / pink / green / orange across winter / spring / summer / fall).
      * Right panel: markers fill-coloured by period
        (forest green before / red after event).

    Both panels share an ``ipywidgets`` slider that controls how many of the
    ``T`` frames are visible — drag right to "play forward" through the
    time-series. In a static notebook (nbconvert) the slider is captured but
    the rendered output shows the full trajectory at ``t = T``.

    Returns ``None`` — the function calls ``display(...)`` directly because
    the widget output and the figure need to share one cell.
    """
    import matplotlib.colors as mcolors
    import ipywidgets as widgets
    from IPython.display import display

    mean_vec = np.asarray(pca_state["mean"], dtype=np.float32)
    comps = np.asarray(pca_state["components"], dtype=np.float32)[:2]   # (2, 384)
    emb_mean = np.stack([ts.tm_emb(i).mean(axis=(0, 1)) for i in range(len(ts))])
    proj = (emb_mean - mean_vec) @ comps.T          # (T, 2)

    ev = ts.event_period()
    event_date = ev["start"] if ev is not None else None
    dates_iso = list(ts.dates)
    dates_dt = [date.fromisoformat(d) for d in dates_iso]
    months = np.array([d.month for d in dates_dt])
    if event_date is not None:
        is_after = np.array([d >= event_date for d in dates_dt])
    else:
        is_after = np.zeros(len(ts), dtype=bool)

    # Seasonal palette: cold blue / pink / green / orange across the year.
    month_cmap = _seasonal_cmap()
    month_rgba = np.stack([month_cmap((m - 1) / 12) for m in months])
    period_rgba = np.where(is_after[:, None], 1, 0) * np.array(mcolors.to_rgba(period_colors[1])) \
                + (1 - np.where(is_after[:, None], 1, 0)) * np.array(mcolors.to_rgba(period_colors[0]))

    T = len(ts)
    pad = 0.07 * (proj.max(axis=0) - proj.min(axis=0))
    xlim = (proj[:, 0].min() - pad[0], proj[:, 0].max() + pad[0])
    ylim = (proj[:, 1].min() - pad[1], proj[:, 1].max() + pad[1])

    # Build the figure once. Subsequent slider updates only mutate the
    # data of a few persistent artists (scatter + line + suptitle), which
    # is fast enough to respond to a continuous-drag slider.
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=figsize, dpi=dpi, constrained_layout=True,
    )
    for ax in (axL, axR):
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xlabel("PC1", fontsize=9)
        ax.set_ylabel("PC2", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(False)
        for side in ax.spines:
            ax.spines[side].set_visible(side in ("left", "bottom"))
            if side in ("left", "bottom"):
                ax.spines[side].set_color("#bbbbbb")
                ax.spines[side].set_linewidth(0.6)

    line_L, = axL.plot([], [], "-", color="#888", alpha=0.45, lw=0.9, zorder=2)
    line_R, = axR.plot([], [], "-", color="#888", alpha=0.45, lw=0.9, zorder=2)
    scatter_L = axL.scatter([], [], s=marker_size,
                            facecolors="none", edgecolors="none", zorder=3)
    scatter_R = axR.scatter([], [], s=marker_size,
                            facecolors="none", edgecolors="none", zorder=3)
    cursor_L = axL.scatter([], [], s=marker_size * 1.8,
                           facecolors="none", edgecolors="#111",
                           linewidths=1.6, zorder=4)
    cursor_R = axR.scatter([], [], s=marker_size * 1.8,
                           facecolors="none", edgecolors="#111",
                           linewidths=1.6, zorder=4)

    # Static per-panel legends — seasonal anchors on the left, period on the right.
    season_handles = [
        plt.scatter([], [], s=marker_size, c=[month_cmap((m - 1) / 12)],
                    edgecolors="none", label=lab)
        for m, lab in [(2, "winter"), (5, "spring"), (8, "summer"), (11, "fall")]
    ]
    axL.legend(handles=season_handles, loc="best",
               frameon=False, fontsize=8, ncol=4)
    period_handles = [
        plt.scatter([], [], s=marker_size, color=period_colors[0],
                    edgecolors="none", label="before event"),
        plt.scatter([], [], s=marker_size, color=period_colors[1],
                    edgecolors="none", label="after event"),
    ]
    axR.legend(handles=period_handles, loc="best",
               frameon=False, fontsize=9)

    base_title = title or (f"sid={ts.sample_id} — embedding trajectory "
                           f"in the dataset-wide PCA basis")
    sup = fig.suptitle(base_title, fontsize=10)
    plt.close(fig)                                     # render via the Output below

    out = widgets.Output()

    def _redraw(t: int) -> None:
        # Mutate data in place — no figure recreation.
        line_L.set_data(proj[:t, 0], proj[:t, 1])
        line_R.set_data(proj[:t, 0], proj[:t, 1])
        scatter_L.set_offsets(proj[:t])
        scatter_R.set_offsets(proj[:t])
        scatter_L.set_facecolors(month_rgba[:t])
        scatter_R.set_facecolors(period_rgba[:t])
        if t >= 1:
            cursor_L.set_offsets(proj[t - 1:t])
            cursor_R.set_offsets(proj[t - 1:t])
        else:
            cursor_L.set_offsets(np.empty((0, 2)))
            cursor_R.set_offsets(np.empty((0, 2)))
        current_date = dates_iso[t - 1] if t >= 1 else "—"
        sup.set_text(f"{base_title}\nframes shown: {t} / {T}  |  "
                     f"current date: {current_date}")
        out.clear_output(wait=True)
        with out:
            from IPython.display import display as _disp
            _disp(fig)

    slider = widgets.IntSlider(
        value=T, min=1, max=T, step=1, description="# frames",
        continuous_update=True,                # update while dragging
        layout=widgets.Layout(width="55%"),
    )
    slider.observe(lambda change: _redraw(change["new"]), names="value")
    _redraw(T)
    # Slider below the figure, horizontally centred via HBox + flex layout.
    centered_slider = widgets.HBox(
        [slider],
        layout=widgets.Layout(justify_content="center", width="100%"),
    )
    display(widgets.VBox([out, centered_slider]))


# ---------------------------------------------------------------------------
# Section 3 / 4 figures — anomaly score over time + dense prediction sweep
# ---------------------------------------------------------------------------

def anomaly_score_timeseries(
    ts: TimeSeries,
    scores: dict[str, np.ndarray],
    *,
    query_dates: list[date],
    labels: np.ndarray | None = None,
    event_start: date | None = None,
    alert_months: int = 12,
    figsize: tuple[float, float] = (11, 5.0),
    dpi: int = 130,
    title: str | None = None,
) -> plt.Figure:
    """Plot one anomaly-score time-series per mode, stacked vertically.

    Parameters
    ----------
    scores
        Dict ``{panel_label -> (n_queries,) float array}``.
    query_dates
        ``len(n_queries)`` list of query dates (``datetime.date``).
    labels
        Optional ``(n_queries,)`` int array; non-zero entries are scattered
        as red highlights ("positive query").
    event_start, alert_months
        Together define the alert window shaded pink.
    """
    n = len(scores)
    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True,
                             constrained_layout=True, dpi=dpi)
    if n == 1:
        axes = [axes]
    qd_arr = np.asarray(query_dates)
    for ax, (panel, vals) in zip(axes, scores.items()):
        if event_start is not None:
            alert_end = event_start + timedelta(days=round(alert_months * 30.4375))
            ax.axvspan(event_start, alert_end, color="#fde2e2", alpha=0.7,
                       label=f"alert window ({alert_months} mo)")
            ax.axvline(event_start, color="black", ls="--", lw=0.8)
        ax.plot(qd_arr, vals, "-o", color="#444", ms=3, lw=0.8)
        if labels is not None:
            pos = np.asarray(labels) == 1
            if pos.any():
                ax.scatter(qd_arr[pos], np.asarray(vals)[pos],
                           c="#dc2626", s=22, zorder=3,
                           label="positive query")
        ax.set_ylabel("||query - mean(R)||", fontsize=8)
        ax.set_title(panel, fontsize=9, loc="left")
        ax.legend(loc="upper left", fontsize=8, frameon=False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[-1].set_xlabel("query date")
    if title:
        fig.suptitle(title, fontsize=10)
    return fig


def dense_sweep_grid(
    ts: TimeSeries,
    panels: dict[str, Callable[[TimeSeries, int], np.ndarray]],
    *,
    frame_indices: Iterable[int] | None = None,
    n_patches: int = 6,
    cmaps: dict[str, str] | None = None,
    vmin: float | None = 0.0,
    vmax: float | None = 1.0,
    overlay_center_marker: bool = True,
    fig_width: float = 14.0,
    title: str | None = None,
    show_swim_bubbles: bool | None = None,
) -> plt.Figure:
    """Dense-prediction sweep — columns = frames, rows = named panels.

    Thin wrapper around ``sample_timeline``: each entry in ``panels``
    becomes one ExtraRow rendered below the RGB strip. The target value
    per frame is shown on the swimlane (existing event-marker rendering),
    not as a separate image column.

    Heatmaps share an absolute ``(vmin, vmax) = (0.0, 1.0)`` so colour
    is consistent across frames (no per-frame renormalisation). Pass
    ``vmin=None`` / ``vmax=None`` to fall back to imshow's auto-scale.
    """
    cmaps = cmaps or {}
    extras = [
        ExtraRow(
            name=name,
            panel_fn=fn,
            cmap=cmaps.get(name, "magma"),
            vmin=vmin,
            vmax=vmax,
            overlay_center_marker=overlay_center_marker,
        )
        for name, fn in panels.items()
    ]
    return sample_timeline(
        ts,
        n_patches=n_patches if frame_indices is None else len(list(frame_indices)),
        highlight=frame_indices,
        mode="rgb",
        fig_width=fig_width,
        title=title,
        show_swim_bubbles=show_swim_bubbles,
        extra_rows=extras,
    )


def confusion_matrices_figure(
    cms: Sequence[np.ndarray],
    *,
    names: Sequence[str],
    f1_arrays: Sequence[np.ndarray] | None,
    class_names: Sequence[str],
    figsize: tuple[float, float] = (10, 4),
    cmap: str = "Blues",
    title: str | None = None,
) -> plt.Figure:
    """Side-by-side row-normalised confusion-matrix panels.

    ``cms`` is a list of ``(K, K)`` integer count matrices; ``names`` is
    a label per panel (e.g. ``["Baseline LR", "FM LR"]``). If ``f1_arrays``
    is given (one ``(n_folds,)`` macro-F1 array per panel), each panel's
    title shows ``mean ± std``. ``class_names`` labels the K rows / cols.

    Each cell prints both the row-normalised proportion and the raw count.
    """
    n = len(cms)
    if names is not None and len(names) != n:
        raise ValueError("len(names) must match len(cms)")
    if f1_arrays is not None and len(f1_arrays) != n:
        raise ValueError("len(f1_arrays) must match len(cms)")

    fig, axes = plt.subplots(1, n, figsize=figsize, constrained_layout=True)
    if n == 1:
        axes = [axes]

    K = len(class_names)
    for k, (ax, cm) in enumerate(zip(axes, cms)):
        cm = np.asarray(cm)
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
        for i in range(K):
            for j in range(K):
                ax.text(
                    j, i,
                    f"{norm[i, j]:.2f}\n({int(cm[i, j])})",
                    ha="center", va="center",
                    fontsize=8,
                    color="white" if norm[i, j] > 0.5 else "black",
                )
        ax.set_xticks(range(K), class_names, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(K), class_names, fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")

        head = names[k] if names is not None else f"panel {k+1}"
        if f1_arrays is not None:
            f1 = np.asarray(f1_arrays[k])
            head = f"{head}\nmacro-F1 = {f1.mean():.3f} ± {f1.std():.3f}"
        ax.set_title(head, fontsize=9)

    if title:
        fig.suptitle(title, fontsize=10)
    return fig


def _argmax_to_rgb(
    argmax: np.ndarray,
    palette: Sequence[str],
    *,
    background: str = "#000000",
) -> np.ndarray:
    """Map a (H, W) int argmax to (H, W, 3) RGB using ``palette``.

    Cells equal to ``-1`` (or any negative index) render as ``background`` —
    the convention used by ``dense_attribution_gated`` for pixels masked
    out by the detection threshold.
    """
    import matplotlib.colors as mcolors
    h, w = argmax.shape
    img = np.zeros((h, w, 3), dtype=np.float32)
    bg = np.array(mcolors.to_rgb(background), dtype=np.float32)
    img[:, :, :] = bg
    for k, c in enumerate(palette):
        img[argmax == k] = np.array(mcolors.to_rgb(c), dtype=np.float32)
    return img


def dense_attribution_figure(
    ts: TimeSeries,
    *,
    pre_idx: int,
    post_idx: int,
    fm_gated: np.ndarray,
    baseline_gated: np.ndarray,
    class_codes: Sequence[int],
    class_names: Sequence[str],
    class_palette: Sequence[str],
    true_class_idx: int,
    fig_width: float = 14.0,
    fig_height: float = 3.6,
    background: str = "#000000",
    target_circle_radius: float = 12.0,
    target_circle_linewidth: float = 3.0,
    title: str | None = None,
) -> plt.Figure:
    """Single-row attribution figure for one example sample.

    Columns:
      1. Reference (pre-event) RGB.
      2. Post-event RGB with a thick coloured circle over the annotated
         pixel — colour = the example's true disturbance class.
      3. FM gated attribution argmax (4-class palette × detection mask).
      4. Baseline gated attribution argmax (same palette × baseline mask).

    Pixels below the detection threshold (``gated_argmax == -1``) render
    as ``background``, so panels 3 and 4 show colour only where the
    detection model fired.

    ``fm_gated`` / ``baseline_gated`` are ``(H, W)`` int arrays from
    ``s2tutorial.dense.dense_attribution_gated`` — value ∈ {-1, 0, …, K-1}.
    """
    fig, axes = plt.subplots(1, 4, figsize=(fig_width, fig_height),
                             constrained_layout=True)

    # --- Panel 1: pre-event RGB ---
    pre_rgb = ts.rgb_centered_224(pre_idx)
    axes[0].imshow(pre_rgb, interpolation="nearest")
    axes[0].set_title(f"reference (pre)\n{ts.dates[pre_idx]}", fontsize=9)

    # --- Panel 2: post-event RGB with thick true-class circle ---
    post_rgb = ts.rgb_centered_224(post_idx)
    axes[1].imshow(post_rgb, interpolation="nearest")
    axes[1].set_title(f"query (post)\n{ts.dates[post_idx]}", fontsize=9)
    # rgb_centered_224 uses crop slice [6:230]; original patch centre
    # (126, 126) maps to crop pixel (120, 120).
    crop_h, crop_w = post_rgb.shape[:2]
    circle_xy = (crop_w // 2 - 4, crop_h // 2 - 4)
    target_color = class_palette[int(true_class_idx)]
    axes[1].add_patch(mpatches.Circle(
        circle_xy,
        radius=target_circle_radius,
        fill=False,
        edgecolor=target_color,
        linewidth=target_circle_linewidth,
        zorder=5,
    ))

    # --- Panel 3: FM gated argmax ---
    fm_rgb = _argmax_to_rgb(fm_gated, class_palette, background=background)
    axes[2].imshow(fm_rgb, interpolation="nearest")
    axes[2].set_title("FM attribution\n(detection-gated)", fontsize=9)

    # --- Panel 4: baseline gated argmax ---
    bl_rgb = _argmax_to_rgb(baseline_gated, class_palette, background=background)
    axes[3].imshow(bl_rgb, interpolation="nearest")
    axes[3].set_title("Baseline attribution\n(detection-gated)", fontsize=9)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#cccccc")
            sp.set_linewidth(0.5)

    # Shared legend under the figure: 4 classes + "no detection" patch.
    handles = [
        mpatches.Patch(facecolor=class_palette[k],
                       edgecolor="none",
                       label=f"{int(class_codes[k])} {class_names[k]}")
        for k in range(len(class_codes))
    ]
    handles.append(mpatches.Patch(
        facecolor=background, edgecolor="#999999", linewidth=0.6,
        label="no detection (below threshold)",
    ))
    handles.append(plt.Line2D(
        [], [], marker="o", linestyle="None",
        markerfacecolor="none", markeredgecolor="#444444",
        markersize=10, markeredgewidth=2.0,
        label="annotated pixel (true class)",
    ))
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=len(handles),
        frameon=False,
        fontsize=8,
    )

    if title:
        fig.suptitle(title, fontsize=10)
    return fig

