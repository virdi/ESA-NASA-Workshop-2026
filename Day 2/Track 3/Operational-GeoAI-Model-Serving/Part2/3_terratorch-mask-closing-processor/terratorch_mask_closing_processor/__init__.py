# Copyright contributors to the Terratorch project

"""TerraTorch Mask-Closing Processor — a custom vLLM IOProcessor demo.

Extends :class:`SegmentationIOProcessor` with a morphological-closing step
applied to the predicted mask before it is written out. The closing fills
small gaps and removes the grid-shaped seams that appear at tile borders
during stitched inference.
"""

from .mask_closing_processor import MaskClosingIOProcessor

__version__ = "0.1.0"
__all__ = [
    "MaskClosingIOProcessor",
    "register_mask_closing_processor",
]


def register_mask_closing_processor():
    """Register the MaskClosingIOProcessor with vLLM.

    Called by vLLM's plugin system via the entry point defined in
    ``pyproject.toml``. Returns the fully qualified class name as a string
    that vLLM uses to load the IOProcessor class.

    Returns:
        str: The fully qualified class name (module.path.ClassName)
    """
    return "terratorch_mask_closing_processor.mask_closing_processor.MaskClosingIOProcessor"
