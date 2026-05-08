from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import torch
import zarr
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.datasets import NonGeoDataset
from terratorch.datamodules.generic_pixel_wise_data_module import Normalize
from terratorch.datasets.utils import (
    resize_hwc,
    to_pca_rgb,
    to_rgb,
)

import s2tutorial as s2t

SampleMode = Literal["frames", "sequences", "event_centered"]
LabelMode = Literal["binary", "grouped", "none"]
EventWindowMode = Literal["mean", "variable"]
SplitName = Literal["train", "val", "test"]
PredictSplit = Literal["train", "val", "test", "all"]
Sample = dict[str, object]
IndexEntry = tuple[int, int, int, int, str]

DEFAULT_SPLIT_FILENAME = "splits.parquet"
SPLIT_NAMES: tuple[SplitName, ...] = ("train", "val", "test")

S2_ALL_BANDS = (
    "B02",
    "B03",
    "B04",
    "B08",
    "B05",
    "B06",
    "B07",
    "B8A",
    "B11",
    "B12",
    "B01",
    "B09",
)

S2L2A_MEANS = {
    "B01": 1390.458,
    "B02": 1503.317,
    "B03": 1718.197,
    "B04": 1853.910,
    "B05": 2199.100,
    "B06": 2779.975,
    "B07": 2987.011,
    "B08": 3083.234,
    "B8A": 3132.220,
    "B09": 3162.988,
    "B11": 2424.884,
    "B12": 1857.648,
}

S2L2A_STDS = {
    "B01": 2106.761,
    "B02": 2141.107,
    "B03": 2038.973,
    "B04": 2134.138,
    "B05": 2085.321,
    "B06": 1889.926,
    "B07": 1820.257,
    "B08": 1871.918,
    "B8A": 1753.829,
    "B09": 1797.379,
    "B11": 1434.261,
    "B12": 1334.311,
}

EVENT_CODE_TO_GROUP_INDEX = {
    211: 1,  # Clear-Cut -> Planned
    212: 1,  # Thinning -> Planned
    213: 1,  # Forestry Mulching -> Planned
    242: 2,  # Wildfire
    243: 3,  # Wind
}

NO_EVENT_LABEL = 0
EVENT_LABEL = 1
BINARY_CLASSES = ("No Event", "Event")
GROUPED_CLASSES = ("No Event", "Planned", "Wildfire", "Wind")


def _check_sample_mode(sample_mode: str) -> None:
    if sample_mode not in ("frames", "sequences", "event_centered"):
        raise ValueError(
            "sample_mode must be 'frames', 'sequences', or 'event_centered'"
        )


def _check_label_mode(label_mode: str) -> None:
    if label_mode not in ("binary", "grouped", "none"):
        raise ValueError("label_mode must be 'binary', 'grouped', or 'none'")


def _check_event_window_mode(event_window_mode: str) -> None:
    if event_window_mode not in ("mean", "variable"):
        raise ValueError("event_window_mode must be 'mean' or 'variable'")


def _classes_for_label_mode(label_mode: LabelMode) -> tuple[str, ...]:
    if label_mode == "binary":
        return BINARY_CLASSES
    if label_mode == "grouped":
        return GROUPED_CLASSES
    return ()


def _default_split_file(root: Path | str) -> Path:
    return Path(root) / DEFAULT_SPLIT_FILENAME


def _validate_split_ids(split_ids: dict[str, list[int]]) -> None:
    for name, ids in split_ids.items():
        duplicates = sorted(sid for sid, count in Counter(ids).items() if count > 1)
        if duplicates:
            raise ValueError(
                f"{name} split contains duplicate sample ids: {duplicates[:10]}"
            )

    if split_ids.get("val") != split_ids.get("test"):
        raise ValueError("val and test splits must be the same sample ids")

    seen =  {}
    overlaps = []
    for name, ids in split_ids.items():
        for sid in ids:
            if name == "test" and seen.get(sid) == "val":
                continue
        if sid in seen:
            if not (name == "test" and seen.get(sid) == "val"):
                overlaps.append((sid, seen[sid], name))
        else:
            seen[sid] = name

    if overlaps:
        examples = ", ".join(
            f"{sid} in {left}/{right}" for sid, left, right in overlaps[:10]
        )
        raise ValueError(f"Split files must be disjoint; found overlaps: {examples}")


def _read_split_file(path: Path | str) -> dict[SplitName, list[int]]:
    path = Path(path)
    if path.suffix != ".parquet":
        raise ValueError(f"Split file must be a parquet file, got: {path}")

    df = pl.read_parquet(path)
    required = {"sample_id", "split"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Split file {path} is missing columns: {missing}")

    allowed = set(SPLIT_NAMES)
    values = {str(value) for value in df["split"].unique().to_list()}
    unknown = sorted(values - allowed)
    if unknown:
        raise ValueError(f"Split file {path} contains unknown splits: {unknown}")

    missing_splits = sorted(allowed - values)
    if missing_splits:
        raise ValueError(f"Split file {path} is missing splits: {missing_splits}")

    split_ids = {
        name: [
            int(sid)
            for sid in df.filter(pl.col("split") == name)["sample_id"].to_list()
        ]
        for name in SPLIT_NAMES
    }
    split_ids["test"] = list(split_ids["val"])
    _validate_split_ids(split_ids)
    return split_ids


def generate_random_split_file(
    root: Path | str = "../data",
    *,
    split_file: Path | str | None = None,
    eval_split_pct: float = 0.20,
    seed: int = 0,
    overwrite: bool = False,
) -> Path:
    """Generate deterministic train/val/test split parquet."""
    if not 0 < eval_split_pct < 1:
        raise ValueError(f"eval_split_pct must be in (0, 1), got {eval_split_pct}")

    root = Path(root)
    path = Path(split_file) if split_file is not None else _default_split_file(root)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing split file: {path}")

    samples = s2t.load_metadata(root)["samples"].select("sample_id")
    sample_ids = [int(sid) for sid in samples["sample_id"].to_list()]
    if len(sample_ids) < 3:
        raise ValueError(f"Need at least 3 samples, got {len(sample_ids)}")

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(sample_ids), generator=generator).tolist()
    shuffled = [sample_ids[i] for i in order]

    n_total = len(shuffled)
    n_eval = int(round(n_total * eval_split_pct))
    n_train = n_total - n_eval
    if n_train <= 0 or n_eval <= 0:
        raise ValueError(
            "Split percentages must leave at least one sample in each split: "
            f"train={n_train}, val={n_eval}, test={n_eval}"
        )

    split_ids = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:],
        "test": shuffled[n_train:],
    }
    _validate_split_ids(split_ids)

    rows = [
        {"sample_id": sample_id, "split": split}
        for split in SPLIT_NAMES
        for sample_id in split_ids[split]
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(path)
    return path


def _load_or_generate_split_ids(
    *,
    root: Path | str,
    split_file: Path,
    auto_generate_splits: bool,
    eval_split_pct: float,
    seed: int,
) -> dict[SplitName, list[int]]:
    if split_file.exists():
        return _read_split_file(split_file)

    if not auto_generate_splits:
        raise FileNotFoundError(
            f"Missing split parquet file: {split_file}. "
            "Create it with generate_random_split_file(...) or set "
            "auto_generate_splits=True."
        )

    warnings.warn(
        f"Missing split parquet file: {split_file}. Generating deterministic splits.",
        stacklevel=2,
    )
    generate_random_split_file(
        root,
        split_file=split_file,
        eval_split_pct=eval_split_pct,
        seed=seed,
        overwrite=False,
    )
    return _read_split_file(split_file)


def _validate_sample_ids(sample_ids: Sequence[int], all_ids: Sequence[int]) -> None:
    missing = sorted(set(sample_ids) - set(all_ids))
    if missing:
        raise ValueError(f"Split references unknown sample ids: {missing[:10]}")


def _event_period(metadata: dict[str, object], sample_id: int) -> dict:
    labels = metadata["labels"]
    if not isinstance(labels, pl.DataFrame):
        raise TypeError("metadata['labels'] must be a Polars DataFrame")

    events = labels.filter((pl.col("sample_id") == sample_id) & pl.col("is_event"))
    if len(events) != 1:
        raise ValueError(f"sample_id={sample_id} has {len(events)} event rows")
    return events.row(0, named=True)


def _event_label_for_mode(event_code: int, label_mode: LabelMode) -> int | None:
    if label_mode == "none":
        return None
    if label_mode == "binary":
        return EVENT_LABEL
    if event_code not in EVENT_CODE_TO_GROUP_INDEX:
        raise ValueError(f"Unsupported event code for grouped labels: {event_code}")
    return EVENT_CODE_TO_GROUP_INDEX[event_code]


def _event_in_window(event: dict, dates: Sequence[str]) -> bool:
    event_start = event["start"].isoformat()
    return bool(dates) and dates[0] <= event_start <= dates[-1]


def _as_date(value: object) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    if hasattr(value, "item"):
        return _as_date(value.item())
    raise TypeError(f"Expected date-like value, got {type(value)!r}")


def _closest_date_index(dates: Sequence[str], anchor: date) -> int:
    if not dates:
        raise ValueError("Cannot select an anchor frame from an empty time series")
    parsed = [_as_date(value) for value in dates]
    return min(
        range(len(parsed)),
        key=lambda i: (abs((parsed[i] - anchor).days), parsed[i]),
    )


def _fake_event_date(
    dates: Sequence[str],
    event_date: date,
    *,
    context_days: int,
    seed: int,
    sample_id: int,
) -> date:
    parsed = [_as_date(value) for value in dates]
    candidates: list[date] = []
    margin = timedelta(days=context_days)

    for candidate in parsed:
        if abs(candidate - event_date) <= margin:
            continue

        pre_start = candidate - margin
        post_end = candidate + margin
        has_pre = any(pre_start <= frame_date < candidate for frame_date in parsed)
        has_post = any(candidate < frame_date <= post_end for frame_date in parsed)
        if has_pre and has_post:
            candidates.append(candidate)

    if not candidates:
        raise ValueError(
            f"sample_id={sample_id} has no fake event date more than "
            f"{context_days} days from the true event with pre/post context"
        )

    rng = np.random.default_rng(seed + sample_id)
    return candidates[int(rng.integers(0, len(candidates)))]


def _add_label(
    sample: Sample,
    *,
    event: dict,
    dates: Sequence[str] | None,
    label_mode: LabelMode,
) -> None:
    """Attach a simple benchmark label, if requested."""
    if label_mode == "none":
        return

    has_event = True if dates is None else _event_in_window(event, dates)
    label = NO_EVENT_LABEL
    if has_event:
        maybe_label = _event_label_for_mode(int(event["label"]), label_mode)
        if maybe_label is None:
            return
        label = maybe_label

    sample["label"] = torch.tensor(label, dtype=torch.long)
    sample["contains_event"] = torch.tensor(has_event, dtype=torch.bool)
    sample["event_code"] = torch.tensor(int(event["label"]), dtype=torch.long)
    sample["event_start"] = event["start"].isoformat()


def _frame_dates(metadata: dict[str, object], sample_id: int) -> list[str]:
    frames = metadata["frames"]
    if not isinstance(frames, pl.DataFrame):
        raise TypeError("metadata['frames'] must be a Polars DataFrame")

    dates = (
        frames.filter(pl.col("sample_id") == sample_id)
        .sort("date")["date"]
        .to_list()
    )
    return [date.isoformat() for date in dates]


def _build_index(
    metadata: dict[str, object],
    sample_ids: Sequence[int],
    *,
    sample_mode: SampleMode,
    sequence_length: int,
    sequence_stride: int,
    event_context_days: int = 365,
    fake_event_seed: int = 0,
) -> list[IndexEntry]:
    """Return (sample_id, start, stop, target, anchor_date) entries."""
    _check_sample_mode(sample_mode)
    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be positive, got {sequence_length}")
    if sequence_stride <= 0:
        raise ValueError(f"sequence_stride must be positive, got {sequence_stride}")
    if event_context_days <= 0:
        raise ValueError(
            f"event_context_days must be positive, got {event_context_days}"
        )

    entries: list[IndexEntry] = []
    for sample_id in sample_ids:
        dates = _frame_dates(metadata, sample_id)
        n_frames = len(dates)

        if sample_mode == "frames":
            entries.extend((sample_id, i, i + 1, -1, "") for i in range(n_frames))
            continue

        if sample_mode == "event_centered":
            event = _event_period(metadata, sample_id)
            event_date = _as_date(event["start"])
            fake_date = _fake_event_date(
                dates,
                event_date,
                context_days=event_context_days,
                seed=fake_event_seed,
                sample_id=sample_id,
            )
            entries.append(
                (sample_id, 0, n_frames, EVENT_LABEL, event_date.isoformat())
            )
            entries.append(
                (sample_id, 0, n_frames, NO_EVENT_LABEL, fake_date.isoformat())
            )
            continue

        if n_frames < sequence_length:
            continue

        starts = range(0, n_frames - sequence_length + 1, sequence_stride)
        entries.extend(
            (sample_id, start, start + sequence_length, -1, "") for start in starts
        )

    return entries


def _upsample_to(reference: Tensor, image: Tensor) -> Tensor:
    return F.interpolate(
        image.unsqueeze(0),
        size=reference.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def _load_s2_all_frame(ts: s2t.TimeSeries, frame_idx: int) -> Tensor:
    s10 = torch.from_numpy(ts.s2_10m(frame_idx, as_reflectance=False)).float()
    s20 = torch.from_numpy(ts.s2_20m(frame_idx, as_reflectance=False)).float()
    s60 = torch.from_numpy(ts.s2_60m(frame_idx, as_reflectance=False)).float()
    return torch.cat([s10, _upsample_to(s10, s20), _upsample_to(s10, s60)], dim=0)



def _select_bands(image: Tensor, bands: Sequence[str]) -> Tensor:
    indices = [S2_ALL_BANDS.index(band) for band in bands]
    return image[indices]


def _event_period_indices(
    dates: Sequence[str],
    anchor_date: str,
    *,
    context_days: int,
) -> tuple[list[int], list[int], list[str], Tensor]:
    parsed = [_as_date(value) for value in dates]
    anchor = _as_date(anchor_date)
    margin = timedelta(days=context_days)
    anchor_idx = _closest_date_index(dates, anchor)

    pre = [
        i
        for i, frame_date in enumerate(parsed)
        if i != anchor_idx and anchor - margin <= frame_date < anchor
    ]
    post = [
        i
        for i, frame_date in enumerate(parsed)
        if i != anchor_idx and anchor < frame_date <= anchor + margin
    ]
    post = [anchor_idx] + post

    indices = pre + post
    window_dates = [dates[i] for i in indices]
    relative_days = torch.tensor(
        [(parsed[i] - anchor).days for i in indices],
        dtype=torch.long,
    )
    return pre, post, window_dates, relative_days


def _period_values(values: Tensor, indices: Sequence[int]) -> Tensor:
    if not indices:
        raise ValueError("Cannot build an event-centered period with no frames")
    return values[list(indices)]


def collate_fixed_samples(batch: list[Sample]) -> Sample:
    """Collate frame or fixed-length sequence samples."""
    collated: Sample = {}

    for key in batch[0]:
        values = [sample[key] for sample in batch]
        if isinstance(values[0], Tensor):
            shapes = {tuple(value.shape) for value in values}
            collated[key] = torch.stack(values) if len(shapes) == 1 else values
        else:
            collated[key] = values

    return collated


def _as_numpy(value: object) -> np.ndarray:
    if isinstance(value, Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _tokens_to_chw(tokens: np.ndarray) -> np.ndarray | None:
    num_tokens, channels = tokens.shape
    size = int(np.sqrt(num_tokens))
    if size * size != num_tokens:
        return None
    return tokens.T.reshape(channels, size, size)


def _last_chw_image(image: object, *, embedding_input: bool) -> np.ndarray | None:
    if isinstance(image, (list, tuple)):
        if not image:
            return None
        return _last_chw_image(image[-1], embedding_input=embedding_input)

    image_np = _as_numpy(image)
    if image_np.ndim == 2:
        return _tokens_to_chw(image_np) if embedding_input else None

    if image_np.ndim == 3:
        if embedding_input and image_np.shape[-1] > 3:
            tokens = _tokens_to_chw(image_np[-1])
            if tokens is not None:
                return tokens
        return image_np
    if image_np.ndim != 4:
        return None

    if embedding_input and image_np.shape[0] <= image_np.shape[1]:
        return image_np[-1]
    return image_np[:, -1]


def _embedding_sequence_to_pca_rgb(
    image: object,
    *,
    pca_step: int,
    temporal_center: bool = False,
) -> np.ndarray:
    if isinstance(image, Tensor):
        embedding = image.detach().cpu().float()
    else:
        embedding = torch.as_tensor(image).float()

    if embedding.ndim == 2:
        embedding = embedding.unsqueeze(0)
    if embedding.ndim != 3:
        raise ValueError(
            "Expected embedding shape (T, N, C) or (N, C), "
            f"got {tuple(embedding.shape)}"
        )

    time, tokens, channels = embedding.shape
    side = int(np.sqrt(tokens))
    if side * side != tokens:
        raise ValueError(f"Cannot reshape {tokens} tokens into a square grid")

    if temporal_center and time > 1:
        embedding = embedding - embedding.mean(dim=0, keepdim=True)

    flat = embedding.reshape(time * tokens, channels)
    fit = flat[::pca_step]
    mean = fit.mean(dim=0, keepdim=True)
    _, _, components = torch.pca_lowrank(fit - mean, q=3, center=False)
    projected = (flat - mean) @ components[:, :3]
    projected = projected.reshape(time, tokens, 3)

    lower = torch.quantile(projected.reshape(-1, 3), 0.02, dim=0).view(1, 1, 3)
    upper = torch.quantile(projected.reshape(-1, 3), 0.98, dim=0).view(1, 1, 3)
    rgb = (projected - lower) / (upper - lower).clamp_min(1e-6)
    rgb = rgb.clamp(0, 1)
    return rgb.reshape(time, side, side, 3).numpy()


class _BaseForestDataset(NonGeoDataset):
    """Shared metadata, split, index and label handling."""

    def __init__(
        self,
        root: Path | str = "../data",
        split: Sequence[int] | None = None,
        sample_mode: SampleMode = "frames",
        sequence_length: int = 10,
        sequence_stride: int = 1,
        event_context_days: int = 365,
        fake_event_seed: int = 0,
        event_window_mode: EventWindowMode = "mean",
        label_mode: LabelMode = "none",
        rgb_indices: Sequence[int] | None = None,
        embedding_input: bool = False,
        pca_step: int = 4,
        transforms: Callable[[Sample], Sample] | None = None,
    ) -> None:
        _check_sample_mode(sample_mode)
        _check_label_mode(label_mode)
        _check_event_window_mode(event_window_mode)

        self.root = Path(root)
        self.sample_mode = sample_mode
        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride
        self.event_context_days = event_context_days
        self.fake_event_seed = fake_event_seed
        self.event_window_mode = event_window_mode
        self.label_mode = label_mode
        self.rgb_indices = list(rgb_indices or (2, 1, 0))
        self.embedding_input = embedding_input
        self.pca_step = pca_step
        self.transforms = transforms
        self.metadata = s2t.load_metadata(self.root)

        samples = self.metadata["samples"]
        all_ids = [int(sid) for sid in samples["sample_id"].to_list()]
        
        self.sample_ids = [int(sid) for sid in split] if split is not None else all_ids
        _validate_sample_ids(self.sample_ids, all_ids)

        self.index = _build_index(
            self.metadata,
            self.sample_ids,
            sample_mode=sample_mode,
            sequence_length=sequence_length,
            sequence_stride=sequence_stride,
            event_context_days=event_context_days,
            fake_event_seed=fake_event_seed,
        )
        self.classes = _classes_for_label_mode(label_mode)

    def __len__(self) -> int:
        return len(self.index)

    def _base_sample(
        self,
        *,
        sample_id: int,
        start: int,
        stop: int,
        dates: list[str],
        source_path: Path,
        target: int = -1,
        anchor_date: str = "",
        relative_days: Tensor | None = None,
    ) -> Sample:
        if self.sample_mode == "frames":
            return {
                "sample_id": torch.tensor(sample_id, dtype=torch.long),
                "frame_index": torch.tensor(start, dtype=torch.long),
                "date": dates[0],
                "filename": f"{sample_id}_{dates[0]}",
                "zarr_path": str(source_path),
            }

        if self.sample_mode == "event_centered":
            return {
                "sample_id": torch.tensor(sample_id, dtype=torch.long),
                "anchor_date": anchor_date,
                "dates": dates,
                "relative_days": relative_days,
                "filename": f"{sample_id}_{target}_{anchor_date}",
                "zarr_path": str(source_path),
                "is_fake_event": torch.tensor(target == NO_EVENT_LABEL),
            }

        return {
            "sample_id": torch.tensor(sample_id, dtype=torch.long),
            "start_index": torch.tensor(start, dtype=torch.long),
            "end_index": torch.tensor(stop - 1, dtype=torch.long),
            "dates": dates,
            "filename": f"{sample_id}_{dates[0]}_{dates[-1]}",
            "zarr_path": str(source_path),
        }

    def _finish_sample(
        self,
        sample: Sample,
        *,
        sample_id: int,
        dates: list[str],
        target: int = -1,
    ) -> Sample:
        event = _event_period(self.metadata, sample_id)

        if self.sample_mode == "event_centered":
            if self.label_mode != "none":
                label = target
                if target == EVENT_LABEL and self.label_mode == "grouped":
                    maybe_label = _event_label_for_mode(
                        int(event["label"]), self.label_mode
                    )
                    if maybe_label is None:
                        return sample
                    label = maybe_label
                sample["label"] = torch.tensor(label, dtype=torch.long)
                sample["contains_event"] = torch.tensor(
                    target == EVENT_LABEL, dtype=torch.bool
                )
                sample["event_code"] = torch.tensor(
                    int(event["label"]), dtype=torch.long
                )
                sample["event_start"] = event["start"].isoformat()
            if self.transforms is not None:
                sample = self.transforms(sample)
            return sample

        # Frame mode is for embedding extraction, so best used with label_mode="none"
        label_dates = dates if self.sample_mode == "sequences" else None
        _add_label(sample, event=event, dates=label_dates, label_mode=self.label_mode)

        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def plot(
        self,
        sample: Sample,
        suptitle: str | None = None,
        show_axes: bool = False,
        embedding_input: bool | None = None,
        pca_step: int | None = None,
    ) -> Figure | None:
        embedding_input = (
            self.embedding_input if embedding_input is None else embedding_input
        )
        pca_step = self.pca_step if pca_step is None else pca_step

        image = _last_chw_image(sample["image"], embedding_input=embedding_input)
        if image is None:
            warnings.warn(
                "Plotting is only supported for spatial images with shape "
                "(C, H, W), (C, T, H, W), or (T, C, H, W)."
            )
            return None

        if embedding_input:
            if image.ndim == 2:
                warnings.warn(
                    "Embedding plotting is only supported for spatially arranged embeddings."
                )
                return None
            image_for_plot, _, _ = to_pca_rgb(image_chw=image, step=pca_step)
        else:
            image_for_plot = to_rgb(image_chw=image, rgb_indices=self.rgb_indices)

        has_prediction = "prediction" in sample
        num_images = 2 if has_prediction else 1
        fig, ax = plt.subplots(
            1,
            num_images,
            figsize=(5 * num_images, 5),
            layout="compressed",
        )
        axes = np.atleast_1d(ax)
        axes_visibility = "on" if show_axes else "off"

        axes[0].axis(axes_visibility)
        axes[0].set_title("Embedding PCA" if embedding_input else "RGB Image")
        axes[0].imshow(image_for_plot)

        label_text = None
        if "label" in sample:
            label = sample["label"]
            if isinstance(label, Tensor):
                label = label.detach().cpu().item()
            label_index = int(label)
            label_name = self.classes[label_index] if self.classes else str(label_index)
            label_text = f"Label: {label_name}"

        if has_prediction:
            prediction = sample["prediction"]
            if isinstance(prediction, Tensor):
                prediction = prediction.detach().cpu().item()
            prediction_index = int(prediction)
            prediction_name = (
                self.classes[prediction_index]
                if self.classes
                else str(prediction_index)
            )
            axes[1].axis(axes_visibility)
            axes[1].set_title(f"Prediction: {prediction_name}")
            axes[1].imshow(image_for_plot)

        if suptitle is not None:
            fig.suptitle(suptitle)
        elif label_text is not None:
            fig.suptitle(label_text)
        return fig

    def plot_pca_timeseries(
        self,
        samples: Sample | Sequence[Sample],
        *,
        pca_step: int | None = None,
        suptitle: str | None = None,
        show_axes: bool = False,
        temporal_center: bool = False,
    ) -> Figure:
        if isinstance(samples, dict):
            sample_list = [samples]
        else:
            sample_list = list(samples)
        if not sample_list:
            raise ValueError("Please provide at least one sample to plot")

        pca_step = self.pca_step if pca_step is None else pca_step
        sequences = [
            _embedding_sequence_to_pca_rgb(
                sample["image"],
                pca_step=pca_step,
                temporal_center=temporal_center,
            )
            for sample in sample_list
        ]
        num_cols = max(sequence.shape[0] for sequence in sequences)

        fig, ax = plt.subplots(
            len(sample_list),
            num_cols,
            figsize=(1.6 * num_cols, 1.8 * len(sample_list)),
            squeeze=False,
            layout="compressed",
        )
        axes_visibility = "on" if show_axes else "off"

        for row, (sample, rgb_sequence) in enumerate(zip(sample_list, sequences)):
            label_text = ""
            if "label" in sample:
                label = sample["label"]
                if isinstance(label, Tensor):
                    label = label.detach().cpu().item()
                label_index = int(label)
                label_text = (
                    self.classes[label_index] if self.classes else str(label_index)
                )
            sample_id = sample.get("sample_id")
            if isinstance(sample_id, Tensor):
                sample_id = sample_id.detach().cpu().item()
            if sample_id is not None:
                label_text = f"{label_text}\nID {int(sample_id)}".strip()

            dates = sample.get("dates", [""] * len(rgb_sequence))
            for col in range(num_cols):
                ax[row, col].axis(axes_visibility)
                if col >= len(rgb_sequence):
                    ax[row, col].set_visible(False)
                    continue

                ax[row, col].imshow(rgb_sequence[col], interpolation="nearest")
                ax[row, col].set_title(str(dates[col])[:10], fontsize=7)
                if col == 0 and label_text:
                    ax[row, col].set_ylabel(
                        label_text,
                        rotation=0,
                        ha="right",
                        va="center",
                        fontsize=9,
                    )

        if suptitle is not None:
            fig.suptitle(suptitle)
        return fig


class ForestDisturbanceDataset(_BaseForestDataset):
    """Raw Sentinel-2 dataset.

    Use ``sample_mode="frames", label_mode="none"`` to compute embeddings from
    individual frames.
    """

    all_sensors = ("s2_all",)
    s2_all_bands = S2_ALL_BANDS

    def __init__(
        self,
        root: Path | str = "../data",
        split: Sequence[int] | None = None,
        sample_mode: SampleMode = "frames",
        sequence_length: int = 10,
        sequence_stride: int = 1,
        event_context_days: int = 365,
        fake_event_seed: int = 0,
        event_window_mode: EventWindowMode = "mean",
        sensor: Literal["s2_all"] = "s2_all",
        bands: Sequence[str] | None = None,
        label_mode: LabelMode = "none",
        rgb_indices: Sequence[int] | None = None,
        embedding_input: bool = False,
        pca_step: int = 4,
        transforms: Callable[[Sample], Sample] | None = None,
    ) -> None:
        if sensor != "s2_all":
            raise ValueError("This datamodule only exposes sensor='s2_all'")

        self.sensor = sensor
        self.bands = tuple(bands or S2_ALL_BANDS)

        unknown = sorted(set(self.bands) - set(S2_ALL_BANDS))
        if unknown:
            raise ValueError(f"Unknown s2_all bands: {unknown}")

        super().__init__(
            root=root,
            split=split,
            sample_mode=sample_mode,
            sequence_length=sequence_length,
            sequence_stride=sequence_stride,
            event_context_days=event_context_days,
            fake_event_seed=fake_event_seed,
            event_window_mode=event_window_mode,
            label_mode=label_mode,
            rgb_indices=rgb_indices,
            embedding_input=embedding_input,
            pca_step=pca_step,
            transforms=transforms,
        )

    def __getitem__(self, index: int) -> Sample:
        sample_id, start, stop, target, anchor_date = self.index[index]
        ts = s2t.get_sample(self.root, sample_id, metadata=self.metadata)
        zarr_path = self.root / "patches" / f"{sample_id}.zarr"

        if self.sample_mode == "event_centered":
            pre_indices, post_indices, dates, relative_days = _event_period_indices(
                ts.dates,
                anchor_date,
                context_days=self.event_context_days,
            )
            frames = [
                _select_bands(_load_s2_all_frame(ts, i), self.bands)
                for i in range(len(ts.dates))
            ]
            values = torch.stack(frames)
            pre = _period_values(values, pre_indices)
            post = _period_values(values, post_indices)
            sample = self._base_sample(
                sample_id=sample_id,
                start=start,
                stop=stop,
                dates=dates,
                source_path=zarr_path,
                target=target,
                anchor_date=anchor_date,
                relative_days=relative_days,
            )
            if self.event_window_mode == "mean":
                sample["pre_image"] = pre.mean(dim=0)
                sample["post_image"] = post.mean(dim=0)
                sample["image"] = torch.stack(
                    [sample["pre_image"], sample["post_image"]],
                    dim=1,
                )
            else:
                sample["pre_image"] = pre.movedim(0, 1)
                sample["post_image"] = post.movedim(0, 1)
                sample["image"] = [sample["pre_image"], sample["post_image"]]
            return self._finish_sample(
                sample, sample_id=sample_id, dates=dates, target=target
            )

        dates = ts.dates[start:stop]
        sample = self._base_sample(
            sample_id=sample_id,
            start=start,
            stop=stop,
            dates=dates,
            source_path=zarr_path,
        )

        frames = [
            _select_bands(_load_s2_all_frame(ts, i), self.bands)
            for i in range(start, stop)
        ]
        sample["image"] = frames[0] if self.sample_mode == "frames" else torch.stack(
            frames,
            dim=1,
        )

        return self._finish_sample(sample, sample_id=sample_id, dates=dates)


class ForestDisturbanceEmbeddingDataset(_BaseForestDataset):
    """embedding dataset
    """

    def __init__(
        self,
        root: Path | str = "../data",
        embedding_root: Path | str = "../embeddings",
        split: Sequence[int] | None = None,
        sample_mode: SampleMode = "frames",
        sequence_length: int = 10,
        sequence_stride: int = 1,
        event_context_days: int = 365,
        fake_event_seed: int = 0,
        event_window_mode: EventWindowMode = "mean",
        embedding_key: str = "embedding",
        label_mode: LabelMode = "none",
        rgb_indices: Sequence[int] | None = None,
        embedding_input: bool = True,
        pca_step: int = 4,
        transforms: Callable[[Sample], Sample] | None = None,
    ) -> None:
        self.embedding_root = Path(embedding_root)
        self.embedding_key = embedding_key

        super().__init__(
            root=root,
            split=split,
            sample_mode=sample_mode,
            sequence_length=sequence_length,
            sequence_stride=sequence_stride,
            event_context_days=event_context_days,
            fake_event_seed=fake_event_seed,
            event_window_mode=event_window_mode,
            label_mode=label_mode,
            rgb_indices=rgb_indices,
            embedding_input=embedding_input,
            pca_step=pca_step,
            transforms=transforms,
        )

    def __getitem__(self, index: int) -> Sample:
        sample_id, start, stop, target, anchor_date = self.index[index]
        embedding, dates, zarr_path = self._read_embedding_store(sample_id)

        if self.sample_mode == "event_centered":
            pre_indices, post_indices, window_dates, relative_days = (
                _event_period_indices(
                    dates,
                    anchor_date,
                    context_days=self.event_context_days,
                )
            )
            pre = _period_values(embedding, pre_indices)
            post = _period_values(embedding, post_indices)
            sample = self._base_sample(
                sample_id=sample_id,
                start=start,
                stop=stop,
                dates=window_dates,
                source_path=zarr_path,
                target=target,
                anchor_date=anchor_date,
                relative_days=relative_days,
            )
            if self.event_window_mode == "mean":
                sample["pre_embedding"] = pre.mean(dim=0)
                sample["post_embedding"] = post.mean(dim=0)
                sample["embedding"] = torch.stack(
                    [sample["pre_embedding"], sample["post_embedding"]]
                )
                sample["image"] = sample["embedding"]
            else:
                sample["pre_embedding"] = pre
                sample["post_embedding"] = post
                sample["embedding"] = [pre, post]
                sample["image"] = sample["embedding"]
            return self._finish_sample(
                sample, sample_id=sample_id, dates=window_dates, target=target
            )

        window_dates = dates[start:stop]
        sample = self._base_sample(
            sample_id=sample_id,
            start=start,
            stop=stop,
            dates=window_dates,
            source_path=zarr_path,
        )

        window = embedding[start:stop]
        sample["embedding"] = window[0] if self.sample_mode == "frames" else window
        sample["image"] = sample["embedding"]

        return self._finish_sample(sample, sample_id=sample_id, dates=window_dates)

    def _read_embedding_store(self, sample_id: int) -> tuple[Tensor, list[str], Path]:
        store = self.embedding_root / f"{sample_id}.zarr"
        if not store.exists():
            raise FileNotFoundError(store)

        group = zarr.open_group(str(store), mode="r")
        if self.embedding_key not in group:
            raise KeyError(f"{store} does not contain array {self.embedding_key!r}")

        group_sample_id = group.attrs.get("sample_id")
        if group_sample_id is not None and int(group_sample_id) != sample_id:
            raise ValueError(
                f"{store}: attrs sample_id={group_sample_id} != {sample_id}"
            )

        array = group[self.embedding_key]
        dates = list(array.attrs.get("dates") or group.attrs.get("dates") or [])
        if not dates:
            raise ValueError(f"{store}: missing dates attrs")
        if array.shape[0] != len(dates):
            raise ValueError(
                f"{store}: {self.embedding_key}.shape[0]={array.shape[0]} "
                f"!= len(dates)={len(dates)}"
            )

        expected = _frame_dates(self.metadata, sample_id)
        if dates != expected:
            raise ValueError(f"{store}: embedding dates disagree with frames.parquet")

        return torch.from_numpy(np.asarray(array)).float(), dates, store


class BaseForestDataModule(NonGeoDataModule):
    dataset_class: type[NonGeoDataset]
    collate: Callable[[list[Sample]], Sample]

    def __init__(
        self,
        root: Path | str = "../data",
        batch_size: int = 8,
        num_workers: int = 0,
        split_file: Path | str | None = None,
        predict_split: PredictSplit = "test",
        eval_split_pct: float = 0.15,
        auto_generate_splits: bool = True,
        seed: int = 0,
        means: Sequence[float] | str | None = None,                  
        stds: Sequence[float] | str | None = None,  
        **kwargs: object,
    ) -> None:
        super().__init__(
            self.dataset_class,
            batch_size=batch_size,
            num_workers=num_workers,
            **kwargs,
        )
        self.root = root
        self.split_file = (
            Path(split_file) if split_file is not None else _default_split_file(root)
        )
        if predict_split not in ("train", "val", "test", "all"):
            raise ValueError(f"Unknown predict_split: {predict_split!r}")

        self.predict_split = predict_split
        self.eval_split_pct = eval_split_pct
        self.auto_generate_splits = auto_generate_splits
        self.seed = seed
        self._split_ids: dict[SplitName, list[int]] | None = None
        self.collate_fn = self.collate

    def setup(self, stage: str | None = None) -> None:
        split_ids = self._ensure_split_file()

        if stage in (None, "fit"):
            self.train_dataset = self._make_dataset(split_ids["train"])
            self.val_dataset = self._make_dataset(split_ids["val"])

        if stage in (None, "validate"):
            self.val_dataset = self._make_dataset(split_ids["val"])

        if stage in (None, "test"):
            self.test_dataset = self._make_dataset(split_ids["test"])

        if stage in (None, "predict"):
            predict_ids = (
                None if self.predict_split == "all" else split_ids[self.predict_split]
            )
            self.predict_dataset = self._make_dataset(predict_ids)

    def _ensure_split_file(self) -> dict[SplitName, list[int]]:
        if self._split_ids is None:
            self._split_ids = _load_or_generate_split_ids(
                root=self.root,
                split_file=self.split_file,
                auto_generate_splits=self.auto_generate_splits,
                eval_split_pct=self.eval_split_pct,
                seed=self.seed,
            )
        return self._split_ids

    def _make_dataset(self, split: Sequence[int] | None) -> NonGeoDataset:
        return self.dataset_class(root=self.root, split=split, **self.kwargs)

    def _dataloader_factory(self, split: str) -> DataLoader[Sample]:
        dataset = self._valid_attribute(f"{split}_dataset", "dataset")
        batch_size = self._valid_attribute(f"{split}_batch_size", "batch_size")

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=getattr(self, "pin_memory", False),
        )

class ForestDisturbanceDataModule(BaseForestDataModule):
    dataset_class = ForestDisturbanceDataset
    collate = staticmethod(collate_fixed_samples)

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        bands = tuple(self.kwargs.get("bands") or S2_ALL_BANDS)
        self.means = [S2L2A_MEANS[band] for band in bands]
        self.stds = [S2L2A_STDS[band] for band in bands]
        self.aug = Normalize(self.means, self.stds)


class ForestDisturbanceEmbeddingDataModule(BaseForestDataModule):
    dataset_class = ForestDisturbanceEmbeddingDataset
    collate = staticmethod(collate_fixed_samples)

    def __init__(
        self,
        root: Path | str = "../data",
        embedding_root: Path | str = "../embeddings",
        batch_size: int = 8,
        num_workers: int = 0,
        split_file: Path | str | None = None,
        predict_split: PredictSplit = "test",
        eval_split_pct: float = 0.20,
        auto_generate_splits: bool = True,
        seed: int = 0,
        **kwargs: object,
    ) -> None:
        self.embedding_root = embedding_root

        super().__init__(
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
            split_file=split_file,
            predict_split=predict_split,
            eval_split_pct=eval_split_pct,
            auto_generate_splits=auto_generate_splits,
            seed=seed,
            **kwargs,
        )

    def _make_dataset(self, split: Sequence[int] | None) -> NonGeoDataset:
        return self.dataset_class(
            root=self.root,
            embedding_root=self.embedding_root,
            split=split,
            **self.kwargs,
        )