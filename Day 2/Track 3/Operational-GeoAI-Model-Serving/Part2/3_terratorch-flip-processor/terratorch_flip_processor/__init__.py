# Copyright contributors to the Terratorch project

"""TerraTorch Flip Processor — a custom vLLM IOProcessor demo.

Extends :class:`SegmentationIOProcessor` with a single flip operation on the
input image. The flip is intentionally not reversed on the output so the
transformation is visible in the returned segmentation mask.
"""

from .flip_processor import FlipAugmentationIOProcessor
from .types import FlipRequestData

__version__ = "0.1.0"
__all__ = ["FlipAugmentationIOProcessor", "FlipRequestData", "register_flip_processor"]


def register_flip_processor():
    """Register the FlipAugmentationIOProcessor with vLLM.
    
    This function is called by vLLM's plugin system via the entry point
    defined in pyproject.toml. It returns the fully qualified class name
    as a string that vLLM will use to load the IOProcessor class.
    
    Returns:
        str: The fully qualified class name (module.path.ClassName)
    """
    return "terratorch_flip_processor.flip_processor.FlipAugmentationIOProcessor"
