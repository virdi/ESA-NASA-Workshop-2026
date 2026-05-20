# Model Preparation and Execution with vLLM

This step prepares the checkpoint to be served via vLLM: produce a `config.json` and a `.bin` weights file from your TerraTorch artifacts, then start the server.

## Step 0: Get the model checkpoint and config

Open a terminal and source the virtualenv

```bash
cd ~/ESA-NASA-Workshop-2026/Day\ 2/Track\ 3/Operational-GeoAI-Model-Serving/Part2/
source .venv/bin/activate
```

If you don't already have the Prithvi flood-segmentation checkpoint and its TerraTorch config on disk, grab them from the pre-created HuggingFace model repository:

```bash
cd ~/ESA-NASA-Workshop-2026/Day\ 2/Track\ 3/Operational-GeoAI-Model-Serving/Part2/1_run_model_in_vllm/
hf download mgazz/prithvi-eo-flood config_deploy.yaml --local-dir .
hf download mgazz/prithvi-eo-flood state_dict.ckpt --local-dir .
```

The rest of this README assumes both files (`state_dict.ckpt` and `config_deploy.yaml`) are in the current directory.

## Step 1: Generate the vLLM config

Run [`vllm_config_generator.py`](vllm_config_generator.py) to turn the TerraTorch YAML into the `config.json` that vLLM expects. The script reads the YAML, adds the vLLM-specific fields (architecture name and related metadata), embeds the input tensor shape and dtype, and writes `config.json` next to the input file. 

**Note:** Input details are necessary to correctly warm-up the vLLM server.

```bash
python vllm_config_generator.py \
    --ttconfig ./config_deploy.yaml \
    -i '{"target":"pixel_values","data":{"pixel_values":{"type": "torch.Tensor","shape": [6, 512, 512]}}}'
```

> Paste the whole block at once (it spans several lines), not line by line.

Flags:
- `--ttconfig` — path to the TerraTorch model config (YAML, here `config_deploy.yaml`)
- `-i, --input` — JSON string, or path to a JSON file, describing the model's input: target field, tensor shape, and dtype

**You should see:** a new file `config.json` in this folder, next to your input YAML.

## Step 2: Convert the checkpoint to a binary

vLLM wants a plain PyTorch state dict in `.bin` form, so [`convert_ckpt_to_bin.py`](convert_ckpt_to_bin.py) pulls out the `state_dict`, drops training-only entries, and saves the result as a standard PyTorch binary.

```bash
python convert_ckpt_to_bin.py state_dict.ckpt -o state_dict.bin
```

Flags:
- `ckpt_path` — the input Lightning checkpoint (here `state_dict.ckpt`)
- `-o, --output` — output path. Defaults to the input name with `.ckpt` swapped for `.bin`
- `-v, --verbose` — print parameter details as the script runs

**You should see:** a new file `state_dict.bin` in this folder. That is the weights file vLLM can load.

## Step 3: Run the model with vLLM

Move back to the Part2 directory
```
cd ..
```

With `config.json` and the `.bin` file in place, start the server.

> **This command does not finish.** It starts a server and then keeps running, printing logs. That is expected — do not close this terminal and do not press Enter again. When you see a line like `Application startup complete` or `Uvicorn running on http://0.0.0.0:8000`, the server is ready. To stop the server later, come back to this terminal and press `Ctrl+C`.

From the `Part2` directory run: 

```bash
vllm serve 1_run_model_in_vllm \
   --skip-tokenizer-init \
   --enable-mm-embeds \
   --max-num-seqs 32 \
   --io-processor-plugin terratorch_segmentation \
   --enforce-eager
```

**Note:** the path set after `vllm serve` is used as model name in the vLLM API.

What each flag is doing:
- `1_run_model_in_vllm` — the directory holding `config.json` and the `.bin`
- `--skip-tokenizer-init` — there's no text tokenizer for a vision model, so don't try to load one
- `--enable-mm-embeds` — turn on multi-modal embeddings support
- `--io-processor-plugin` — which I/O processor vLLM should route requests through (here, `terratorch_segmentation`)
- `--max-num-seqs` — max sequences processed in parallel
- `--enforce-eager` — use eager execution rather than CUDA graphs

vLLM loads the model from the directory you point it at and starts listening on `http://localhost:8000`. From there you can send inference requests via the OpenAI-compatible API or use the notebook.

### Testing the server

Open [`inference.ipynb`](inference.ipynb) to send a sample request and visualize the segmentation mask. The notebook loads a geospatial image, posts it to the vLLM server, and renders the returned mask.

## Step 4: Publish the model to the Hugging Face Hub (optional)

> We skip this in the live session for time. It matters once you move past a single machine. Kubernetes pods, for example, need to pull the model from somewhere reachable at startup.

With `config.json` and `.bin` file produced, the natural next move for a production deployment is to publish them as a model repository on the [Hugging Face Hub](https://huggingface.co/). vLLM can then load by repo ID (e.g. `your-org/prithvi-eo-flood`) instead of a local path.

### Why this matters in Kubernetes

Pods are ephemeral and rarely have your local checkpoints on disk. A Hugging Face repo gives every replica one network-accessible place to pull from at startup, with token-based access for private repos.

### Workflow

1. Install the Hugging Face CLI and authenticate:
   ```bash
   pip install -U "huggingface_hub[cli]"
   hf auth login
   ```

2. Create a new model repository (public or private):
   ```bash
   hf repo create your-org/prithvi-eo-flood --type model
   ```

3. Upload the artifacts produced by Steps 1 and 2:
   ```bash
   cd 1_run_model_in_vllm
   hf upload your-org/prithvi-eo-flood ./config.json
   hf upload your-org/prithvi-eo-flood ./state_dict.bin
   ```

4. Add a `README.md` to the repo describing the model, the expected inputs/outputs, and the IOProcessor it should be served with.

### Loading from the Hub in vLLM

Once published, point vLLM at the repo ID instead of a local path:
```bash
vllm serve your-org/prithvi-eo-flood \
   --skip-tokenizer-init \
   --enable-mm-embeds \
   --io-processor-plugin terratorch_segmentation \
   --max-num-seqs 32 \
   --enforce-eager
```

In Kubernetes, you'd inject an `HF_TOKEN` secret into the pod environment so private repos can be pulled at startup.

## Summary

You should now have:

1. `config.json` — vLLM's view of the model architecture and input specs
2. `<model-name>.bin` — the model weights in PyTorch binary format
3. A running vLLM server, ready to accept inference requests

From here, use [`inference.ipynb`](inference.ipynb) or post directly to the OpenAI-compatible API. For real deployments (Kubernetes especially), publish the two artifacts to the Hugging Face Hub as in Step 4 and let vLLM load by repo ID.
