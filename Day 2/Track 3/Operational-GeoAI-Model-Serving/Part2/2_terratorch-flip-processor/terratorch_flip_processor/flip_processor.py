# Copyright contributors to the Terratorch project

"""FlipAugmentationIOProcessor.

A minimal example of how to extend :class:`SegmentationIOProcessor` with a
single piece of custom pre-processing: flipping the input image.

The flip is intentionally **not** reversed on the output — the resulting
segmentation mask comes back flipped, so a user comparing the input GeoTIFF
to the output mask can visually confirm that the custom processor ran.

Everything else (normalization, tiling, padding, vLLM batching, de-tiling,
saving) is inherited unchanged from the parent class.
"""

from __future__ import annotations

import contextvars
import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
from vllm.entrypoints.pooling.pooling.protocol import IOProcessorRequest
from vllm.inputs import PromptType
from vllm.plugins.io_processors.interface import IOProcessorInput

from terratorch.vllm.plugins.segmentation.segmentation_io_processor import SegmentationIOProcessor

from .types import FlipRequestData

logger = logging.getLogger(__name__)


# ContextVar so that load_image (called inside the parent's pre_process_async)
# can pick up the flip flags for the current request without breaking under
# concurrent asyncio tasks. ContextVars are per-task in asyncio.
_flip_ctx: contextvars.ContextVar[tuple[bool, bool]] = contextvars.ContextVar(
    "flip_ctx", default=(False, False)
)


def _apply_flip(arr: np.ndarray, flip_h: bool, flip_v: bool) -> np.ndarray:
    """Flip an array along its width (-1) and/or height (-2) axes."""
    if flip_h:
        arr = np.flip(arr, axis=-1)
    if flip_v:
        arr = np.flip(arr, axis=-2)
    return arr.copy()  # avoid negative strides downstream


class FlipAugmentationIOProcessor(SegmentationIOProcessor):
    """Segmentation IO processor that flips the input image before inference.

    The flip is not undone on the output, so the returned mask is also flipped
    relative to the input — this makes the effect of the custom processor
    visible end-to-end.
    """

    def parse_data(self, data: Any) -> IOProcessorInput:
        if isinstance(data, dict):
            return FlipRequestData(**data)
        if isinstance(data, IOProcessorRequest):
            if not hasattr(data, "data") or not isinstance(data.data, dict):
                raise ValueError("Unable to parse the request data")
            return FlipRequestData(**data.data)
        raise ValueError("Unable to parse request")

    async def load_image(self, *args, **kwargs):
        imgs, temporal_coords, location_coords, metas = await super().load_image(*args, **kwargs)
        flip_h, flip_v = _flip_ctx.get()
        if flip_h or flip_v:
            imgs = _apply_flip(imgs, flip_h, flip_v)
        return imgs, temporal_coords, location_coords, metas

    async def pre_process_async(
        self,
        prompt: FlipRequestData,
        request_id: str | None = None,
        **kwargs,
    ) -> PromptType | Sequence[PromptType]:
        token = _flip_ctx.set((prompt.flip_horizontal, prompt.flip_vertical))
        try:
            return await super().pre_process_async(prompt, request_id, **kwargs)
        finally:
            _flip_ctx.reset(token)
