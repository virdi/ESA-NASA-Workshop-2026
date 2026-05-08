"""Visualisation helpers for the workshop subset.

Three top-level entry points:

- `sample_timeline(ts, …)` — single-axis time line. Label periods are
  drawn as colored ribbons; the single disturbance event is drawn as a
  marker; cloud-free S2 acquisitions are drawn along the bottom. Pass
  `draw_patches=True` to render a strip of RGB patch thumbnails inline
  below the swimlane (numbered bubbles on the obs axis link 1↔1 to the
  numbered badges on the thumbnails).
- `patch_thumbnails(ts, indices, …)` — same RGB strip, standalone. Use
  this when you want the thumbnails on their own axes / page.
- `class_breakdown(meta, level)` — horizontal bar chart of period or
  frame counts at hierarchy level 1 / 2 / 3.

Stays minimal on purpose; copy / adapt freely.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from .loader import TimeSeries


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
      - `'circle'` (default): ◉ colored per L2 palette.
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

    # 2. Layout: rows of `ncols` panels each; `mode='both'` doubles rows.
    if ncols is None:
        ncols = min(len(indices), 10) if indices else 1
    ncols = max(1, int(ncols))
    n_batches = (len(indices) + ncols - 1) // ncols if indices else 0
    rows_per_batch = 2 if (mode == "both" and indices) else 1
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
            )
        if mode in ("scl", "both"):
            row_offset = batch * rows_per_batch + (rows_per_batch - 1)
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

    fig.tight_layout()
    return fig


# Back-compat shim: full_timeseries == sample_timeline with n_patches=-1.
def full_timeseries(ts: TimeSeries, *,
                    mode: str = "rgb",
                    ncols: int = 10,
                    panel_size: float | None = None,
                    **kwargs) -> plt.Figure:
    """Deprecated alias — use ``sample_timeline(ts, n_patches=-1, ncols=…)``."""
    if panel_size is not None:
        kwargs.setdefault("fig_width", panel_size * ncols)
    return sample_timeline(ts, n_patches=-1, ncols=ncols, mode=mode, **kwargs)


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
            seen_evt.setdefault(_l2(r["label"]), c)
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
    for l2, c in sorted(seen_evt.items()):
        handles.append(plt.Line2D([], [], marker="o", linestyle="None",
                                  markerfacecolor=c, markeredgecolor="none",
                                  markersize=8, label=f"{_name(l2)} (event)"))
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


def patch_thumbnails(
    ts: TimeSeries,
    indices: Iterable[int],
    *,
    mode: str = "rgb",
    rgb: str = "true",
    clip_pct: tuple[float, float] = (2.0, 98.0),
    crop_size: int | None = None,
    crop_offset: tuple[int, int] = (0, 0),
    center_marker_radius: float = 6.0,
    figsize_per_panel: tuple[float, float] = (1.7, 1.9),
    dpi: int = 130,
    badge_color: str = "#e07a5f",
) -> plt.Figure:
    """Standalone strip of thumbnails for the requested frames.

    `indices` are positions in `ts.dates`. A numbered badge in the
    top-left of each panel matches the bubbles drawn by
    `sample_timeline(..., highlight=indices)`.

    `mode`, `rgb`, `clip_pct`, `crop_size`, `crop_offset` mean the
    same thing as in `sample_timeline` and `full_timeseries`.
    """
    if mode not in ("rgb", "scl", "both"):
        raise ValueError(f"mode must be 'rgb', 'scl', or 'both', got {mode!r}")
    indices = list(indices)
    n = len(indices)
    if n == 0:
        raise ValueError("indices must be non-empty")
    nrows = 2 if mode == "both" else 1
    fig, axes = plt.subplots(
        nrows, n,
        figsize=(figsize_per_panel[0] * n,
                 figsize_per_panel[1] * nrows),
        dpi=dpi,
        squeeze=False,
    )
    row = 0
    if mode in ("rgb", "both"):
        _draw_s2_thumbnails(list(axes[row]), ts, indices,
                            rgb=rgb, clip_pct=clip_pct,
                            badge_color=badge_color,
                            center_marker_radius=center_marker_radius,
                            crop_size=crop_size, crop_offset=crop_offset)
        row += 1
    if mode in ("scl", "both"):
        _draw_scl_thumbnails(list(axes[row]), ts, indices,
                             center_marker_radius=center_marker_radius,
                             crop_size=crop_size, crop_offset=crop_offset,
                             show_title=(mode == "scl"))
    fig.tight_layout()
    return fig




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
