# Copyright contributors to the Terratorch project

"""MaskClosingIOProcessor.

A minimal example of how to extend :class:`SegmentationIOProcessor` with a
single piece of custom post-processing: morphological closing of the
predicted mask.

Stitched segmentation masks frequently show a grid of thin seams at the
boundaries between tiles. A morphological close
(``cv2.morphologyEx(mask, MORPH_CLOSE, kernel)``) with a kernel slightly
larger than the seam width fills those gaps without otherwise altering
class boundaries.

Everything else (normalization, tiling, padding, vLLM batching, de-tiling,
saving) is inherited unchanged from the parent class.
"""

from __future__ import annotations
import logging
import cv2
import numpy as np

from terratorch.vllm.plugins.segmentation.segmentation_io_processor import SegmentationIOProcessor

logger = logging.getLogger(__name__)


class MaskClosingIOProcessor(SegmentationIOProcessor):
    """Segmentation IO processor that runs morphological closing on the mask.

    The close is applied to the final stitched mask, immediately before it is
    written to disk / encoded as base64 by the parent's ``save_geotiff``.
    """

    def stich_mask(self,mask,kernel_size=110):
        mask_dtype = mask.dtype
        closed_mask = cv2.morphologyEx(
            mask.squeeze().astype("uint8"), 
            cv2.MORPH_CLOSE, 
            np.ones((kernel_size, kernel_size), np.uint8))
        return np.expand_dims(closed_mask, axis=0).astype(mask_dtype)


    def save_geotiff(self, mask, meta,out_format,*args, **kwargs):
      

        return super().save_geotiff(self.stich_mask(mask), 
                                    meta, out_format, *args, **kwargs)