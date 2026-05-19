# Step 2: Benchmarking the vLLM Server

The model is loaded, so how does the server actually hold up under load? Two tools for measuring throughput and latency:

- **`vllm bench`** — the load generator built into vLLM itself
- **`guidellm`** — a more flexible benchmarking harness from the broader LLM-serving ecosystem

Both drive the running vLLM server with the datasets shipped in this directory (`dataset_path_input_india.jsonl` for `vllm bench` and `dataset_path_input_india_guidellm.jsonl` for `guidellm`). They hit the same `/pooling` endpoint exposed by the Prithvi model, but they answer different questions.

> The vLLM server from Step 1 must still be running in its own terminal for any of the commands below to work. Run these benchmarking commands in a separate terminal.

```bash
cd ~/ESA-NASA-Workshop-2026/Day\ 2/Track\ 3/Operational-GeoAI-Model-Serving/Part2/2_vllm_benchmarking
```
## `vllm bench`

`vllm bench` is the benchmarking client that ships with vLLM. It's the most direct way to ask "given a fixed request rate, what latency does my server produce?" — the same harness vLLM developers use for regression tests.

From the `2_vllm_benchmarking` directory run: 

```bash
vllm bench serve \
  --base-url http://localhost:8000 \
  --dataset-name=custom \
  --model 1_run_model_in_vllm \
  --skip-tokenizer-init \
  --endpoint /pooling \
  --backend vllm-pooling \
  --percentile-metrics e2el \
  --metric-percentiles 25,75,99 \
  --num-prompts 100 \
  --request-rate 10 \
  --dataset-path ./dataset_path_input_india.jsonl
```

Key flags:

- `--backend vllm-pooling` and `--endpoint /pooling` — target the pooling endpoint used by encoder-style geospatial models (rather than the chat/completion endpoints used for LLMs).
- `--skip-tokenizer-init` — Prithvi consumes raw image data; there is no text tokenizer to load.
- `--num-prompts 100 --request-rate 10` — fire 100 requests at a steady 10 req/s.
- `--percentile-metrics e2el --metric-percentiles 25,75,99` — report end-to-end latency at the 25th/75th/99th percentiles.

Strengths: nothing extra to install, and the metric definitions match what the vLLM project uses internally for its own regression numbers.

## `guidellm`

[`guidellm`](https://github.com/vllm-project/guidellm) is a standalone benchmarking tool that focuses on **load-curve sweeps** — running the same workload at progressively higher rates to find the saturation point of the server, not just measure a single rate.

From the `2_vllm_benchmarking` directory run: 

```bash
guidellm benchmark \
  --target http://localhost:8000 \
  --backend openai_http \
  --model 1_run_model_in_vllm \
  --data dataset_path_input_india_guidellm.jsonl \
  --request-format /pooling \
  --data-column-mapper pooling_column_mapper \
  --max-requests 100 \
  --output-path results.json
```

A single `guidellm benchmark` invocation produces multiple runs by default:

1. A **synchronous** run (one request at a time) to establish a latency floor.
2. A **throughput** run (no rate cap) to find the server's ceiling.
3. A series of **constant-rate** runs sweeping between those two bounds.

Results land in `results.json` for offline analysis. The repeated dataset (`dataset_path_input_india_guidellm.jsonl`) is long enough that the harness can keep dispatching requests across every sweep stage without recycling input.

## Differences at a glance

| | `vllm bench` | `guidellm` |
|---|---|---|
| Origin | Built into vLLM | Standalone tool |
| Question it answers | "What does latency look like at *this* rate?" | "How does latency scale across a *range* of rates?" |
| Output | Console summary | Structured `results.json` for plotting |
| Pooling endpoint | Native (`vllm-pooling` backend) | Via `openai_http` + custom `--data-column-mapper` |
| Best for | Quick regression checks, single-point measurements | Capacity planning, finding saturation, generating load curves |

In practice you use both. `vllm bench` is what you reach for after a configuration change to spot-check that nothing regressed at one rate. `guidellm` is what you run when you actually want to know where the server falls over.

## Analyzing `guidellm` results

Once `results.json` is written, the notebook in this directory (`benchmark.ipynb`) loads it into pandas and summarizes each run.

A typical sweep looks like this (10 stages, 94 successful requests each):

```
================================================================================
GUIDELLM BENCHMARK SUMMARY
================================================================================

Total Benchmark Runs: 10

Benchmark Results:
 Run    Strategy      Rate  Mean Latency (s)  Median Latency (s)  P99 Latency (s)  Requests/Second  Total Requests  Duration (s)
   1 synchronous       NaN          0.297399            0.290825         0.577812         3.359372              94     27.981417
   2  throughput       NaN          6.200474            6.831711         9.029911        10.399577              94      9.038829
   3    constant  4.239398          0.328654            0.327678         0.400833         4.221043              94     22.269378
   4    constant  5.119423          0.341600            0.332051         0.585692         5.081402              94     18.498833
   5    constant  5.999449          0.353626            0.344351         0.551517         5.934567              94     15.839405
   6    constant  6.879475          0.381170            0.373507         0.670659         6.752028              94     13.921744
   7    constant  7.759500          0.382672            0.376983         0.546489         7.586402              94     12.390591
   8    constant  8.639526          0.397605            0.391105         0.569833         8.414102              94     11.171721
   9    constant  9.519551          0.381145            0.386149         0.463534         9.253932              94     10.157844
  10    constant 10.399577          0.444858            0.428710         1.013053        10.027209              94      9.374493
```

Reading the table:

- The **synchronous** baseline shows ~0.29 s of unavoidable single-request latency (network + preprocessing + inference + postprocessing).
- The **throughput** run pushes ~10.4 req/s but with a P99 above 9 s — requests are queueing.
- The **constant-rate** sweep traces the curve between those two extremes. Latency stays well-behaved up to ~9 req/s, then P99 jumps from 0.46 s (Run 9) to 1.01 s (Run 10). That knee is the practical capacity of this server configuration for this workload.

This is the answer `guidellm` is built to give you in one invocation, and the one `vllm bench` won't.


## References

- **vllm bench**: [vLLM Benchmarking Documentation](https://docs.vllm.ai/en/latest/serving/benchmarking.html)
- **guidellm**: [GuideLLM GitHub Repository](https://github.com/vllm-project/guidellm)
