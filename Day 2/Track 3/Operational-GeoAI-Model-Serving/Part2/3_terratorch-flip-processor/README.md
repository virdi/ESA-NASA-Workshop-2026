# TerraTorch Flip Processor

A small custom IOProcessor for TerraTorch segmentation models served by vLLM. It flips the input image before inference and returns the mask still flipped, so you can see at a glance that the plugin ran.

## What you'll do in this step

1. Install the plugin into your environment.
2. Restart vLLM so it picks up the plugin via Python entry points.
3. Send a request from the notebook and confirm the output mask comes back flipped.

## 1. Install

From this directory:

```bash
cd 3_terratorch-flip-processor
uv pip install -e .
```

This registers `flip_augmentation` as a vLLM I/O processor plugin. No code changes to vLLM are needed — entry points do the wiring.

## 2. Start vLLM with the plugin

Stop any running vLLM server from Step 1, then restart it pointing at the new plugin:

```bash
cd ..
vllm serve 1_run_model_in_vllm \
    --skip-tokenizer-init \
    --enable-mm-embeds \
    --io-processor-plugin flip_augmentation \
    --max-num-seqs 32 \
    --enforce-eager
```

The only difference from Step 1 is `--io-processor-plugin flip_augmentation`.

## 3. Run the notebook

Open [`inference.ipynb`](inference.ipynb) and run it top to bottom. The request payload includes `"flip_horizontal": true`, so the mask you get back will be horizontally flipped relative to the input. That visible flip is the signal the plugin is running.

Try toggling `flip_horizontal` / `flip_vertical` in the payload and rerunning the request cell.

## Request parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `data` | str | Yes | — | Path, URL, or base64 string of input GeoTIFF |
| `data_format` | str | Yes | — | `"path"`, `"url"`, or `"b64_json"` |
| `out_data_format` | str | Yes | — | `"path"` or `"b64_json"` |
| `flip_horizontal` | bool | No | `False` | Flip input left-right (mask comes back flipped) |
| `flip_vertical` | bool | No | `False` | Flip input top-bottom (mask comes back flipped) |
| `indices` | list[int] | No | `[0,1,2,3,4,5]` | Band indices to use |
| `out_path` | str | No | `None` | Custom output directory (for `"path"` format) |

## What to take away

This package is a teaching example. The three patterns worth recognising:

- Subclassing `SegmentationIOProcessor` to add behaviour without re-implementing the pipeline.
- Registering a plugin with vLLM through Python entry points (see `pyproject.toml`).
- Threading per-request state through the parent's async pipeline.

The code lives in [`terratorch_flip_processor/flip_processor.py`](terratorch_flip_processor/flip_processor.py) — under 200 lines, worth a read.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
