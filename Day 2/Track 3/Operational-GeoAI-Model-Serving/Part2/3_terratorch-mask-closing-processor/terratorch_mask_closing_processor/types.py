# Copyright contributors to the Terratorch project

"""Type definitions for the mask-closing IO processor."""

from terratorch.vllm.plugins.segmentation.types import RequestData


class MaskClosingRequestData(RequestData):
    """Standard segmentation request plus optional morphological-closing flags."""

    morph_close_enabled: bool = True
    """Run cv2.morphologyEx(MORPH_CLOSE) on the predicted mask before saving."""

    morph_close_kernel_size: int = 110
    """Side length (pixels) of the square structuring element. Must be > 0."""