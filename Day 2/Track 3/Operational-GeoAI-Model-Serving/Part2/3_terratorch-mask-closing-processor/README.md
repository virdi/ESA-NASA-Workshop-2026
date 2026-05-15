# TerraTorch Mask-Closing Processor

A small custom IOProcessor for TerraTorch segmentation models served by vLLM. It runs `cv2.morphologyEx(mask, MORPH_CLOSE, kernel)` on the predicted mask before it is returned, which fills the thin grid-shaped seams that appear at tile boundaries during stitched inference.

## What you'll do in this step

1. Install the plugin into your environment.
2. Restart vLLM so it picks up the plugin via Python entry points.
3. Send a request from the notebook and confirm the grid artifacts are gone from the returned mask.

## 1. Install

From this directory:

```bash
cd ~/ESA-NASA-Workshop-2026/Day\ 2/Track\ 3/Operational-GeoAI-Model-Serving/Part2/3_terratorch-mask-closing-processor/
uv pip install -e .
cd ..
```

This registers `mask_closing` as a vLLM I/O processor plugin. No code changes to vLLM are needed — entry points do the wiring.

## 2. Download sample and start vLLM with the plugin

Stop any running vLLM server, then restart it pointing at the new plugin.

From the `Part2` directorty run:

```bash

hf download mgazz/prithvi-eo-burnscars park_fire_scaled.tif --local-dir ./samples/
```

Open [`inference.ipynb`](inference.ipynb) and follow each step

## What to take away

This package is a teaching example. The three patterns worth recognising:

- Subclassing `SegmentationIOProcessor` to add behaviour without re-implementing the pipeline.
- Hooking the **output** side of the pipeline by overriding a single inherited method (`save_geotiff`) — the mirror of the flip example, which hooks the input side via `load_image`.
- Registering a plugin with vLLM through Python entry points (see `pyproject.toml`).

The code lives in [`terratorch_mask_closing_processor/mask_closing_processor.py`](terratorch_mask_closing_processor/mask_closing_processor.py) — under 50 lines, worth a read.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
