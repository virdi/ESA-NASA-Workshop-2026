# Geospatial Model Deployment with vLLM

This project walks through serving TerraTorch geospatial foundation models with vLLM, end to end: from raw training artifacts on disk to a running server you can benchmark. 

## Session Structure

Four steps.

### [Step 1: Run a Model in vLLM](./1_run_model_in_vllm/README.md)

Take a TerraTorch model, make it loadable by vLLM, and serve it:
- Convert a TerraTorch YAML configuration into a vLLM-compatible `config.json`
- Prepare the checkpoint (`.ckpt`) for vLLM by creating a minimal PyTorch binary (`.bin`)
- Start a vLLM server with the model and run inference from a notebook

You come out the other side with a `config.json`, a `.bin` weights file, and a running vLLM server returning segmentation masks.

### [Step 2: Benchmarking the vLLM Server](./2_vllm_benchmarking/README.md)

Now that there's a server, push some load at it. Two tools, two different questions:
- `vllm bench` — vLLM's built-in client. Good for "what's the latency at *this* rate?"
- `guidellm` — sweeps across request rates to find where the server starts to fall over
- A notebook reads the resulting `results.json` and pulls out the practical capacity of the deployment

The point is to learn the different tools available to benchmark EO models.

### [Step 3: (Bonus) Custom IOProcessor](./3_terratorch-mask-closing-processor/README.md)

vLLM speaks tensors-in / tensors-out. Real geospatial users send GeoTIFFs and expect GeoTIFFs back. The IOProcessor is the adapter that sits between the two. This step walks through a small custom processor that fills the gaps in the inference mask:
- How a custom IOProcessor is structured
- How to register it with vLLM via Python entry points
- How to thread per-request state through the parent's pipeline without re-implementing it

You come out with an installable `mask_closing` package, auto-registered with vLLM.

### [Step 4: Serving vLLM on Kubernetes with KServe](./4_serving_vllm_on_k8s_with_kserve/README.md)

Take the same vLLM serving setup off your local machine and onto a cluster:
- Declare the model endpoint with a KServe `InferenceService`, running vLLM (Prithvi + TerraTorch IO processor) as the serving runtime
- Wire up the storage, monitoring, networking, and RBAC prerequisites
- Autoscale with KEDA driven by Prometheus metrics, then drive load with `vllm bench` and watch replicas scale

You come out with a Kubernetes-deployed, autoscaling vLLM InferenceService.


## Prerequisites

> **If you are joining the live workshop session:** everything below is already installed on your sandbox. Skip "Prerequisites" and "Setup" and go straight to [Getting Started](#getting-started). The rest of this section only matters if you later run this on your own machine.

To run this on your own machine you need:

- Linux system with NVIDIA GPU
- Python 3.12
- CUDA and NVIDIA drivers installed
- `uv` package manager ([installation guide](https://github.com/astral-sh/uv))

## Setup

> These steps are run in a **terminal**. Each block below is one command (or a small set of them) — copy the whole block and paste it into the terminal, then press Enter. Lines that start with `cd` change which folder you are working in; run them exactly as written. The `\ ` inside the path is an escaped space — keep it as-is. Where a command spans several lines ending in `\`, paste the whole block at once, not line by line.

Create a virtual environment with `uv` and install the dependencies:

```bash
cd ~/ESA-NASA-Workshop-2026/Day\ 2/Track\ 3/Operational-GeoAI-Model-Serving/Part2/
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Register the environment as a Jupyter kernel so the notebooks can pick it up:

```bash
python3 -m ipykernel install --user --name=.venv
```

### Selecting the kernel in a notebook

Open one of the notebooks and look at the top-right of the toolbar. The current kernel is shown next to a small circle:

![Kernel selector button](imgs/button_to_change_py_kernel.png)

Click it to open the kernel picker, then choose `.venv` from the list:

![Kernel picker](imgs/select_kernel.png)

Once selected, the notebook will run against the environment you just set up.

## Getting Started

Work through the directories in order. Each has its own README with the commands and context for that step.

1. [`1_run_model_in_vllm/`](./1_run_model_in_vllm/README.md) — prepare the model artifacts and serve them with vLLM
2. [`2_vllm_benchmarking/`](./2_vllm_benchmarking/README.md) — benchmark the running server with `vllm bench` and `guidellm`
3. [`3_terratorch-mask-closing-processor/`](./3_terratorch-mask-closing-processor/README.md) — (bonus) add a custom IOProcessor and serve it
4. [`4_serving_vllm_on_k8s_with_kserve/`](./4_serving_vllm_on_k8s_with_kserve/README.md) — deploy the vLLM server on Kubernetes with KServe and KEDA autoscaling
