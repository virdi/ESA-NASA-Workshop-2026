# Copyright contributors to the Terratorch project

"""Type definitions for the flip-augmentation IO processor."""

from terratorch.vllm.plugins.segmentation.types import RequestData


class FlipRequestData(RequestData):
    """Standard segmentation request plus optional flip flags."""

    flip_horizontal: bool = False
    """Flip the input image left-right before inference."""

    flip_vertical: bool = False
    """Flip the input image top-bottom before inference."""
