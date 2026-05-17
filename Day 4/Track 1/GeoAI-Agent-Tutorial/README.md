# GeoAI Agent Tutorial

A two-part workshop notebook that walks through (1) **designing** an agent interactively via the **CARE** process and (2) **wiring up** a pre-built agent artifact (Prithvi-EO-2.0) into a runnable agent using `pydantic-ai` + `pydantic-ai-backends`.

## What's in it

- **Agent Design (CARE)** — chat with an interviewer agent that walks an SME through the four CARE phases (Scope & Decompose → Key Info → Reasoning & Guardrails → Prompt Architecture) and authors the resulting artifact files (`scope.md`, `contexts/*.md`, `tools/...`, `agents.md`, …) directly into a workspace directory. Lifted from [github.com/NASA-IMPACT/akd-labs](https://github.com/NASA-IMPACT/akd-labs); the per-phase prompt library comes from [github.com/NASA-IMPACT/AKD-CARE](https://github.com/NASA-IMPACT/AKD-CARE) and is auto-cloned on first run.
- **FM Agent (Prithvi)** — wires the pre-built Prithvi artifact into a runnable agent with a `LocalBackend` in read-only mode; demonstrates async runs, full-trace streaming (tool calls + reasoning + text deltas), and an inline Gradio chat.

## Setup

Install [uv](https://docs.astral.sh/uv/) if you don't have it, then from this directory:

```bash
uv sync
```

Create a `.env` file alongside the notebook with your OpenAI key (a `.env.example` is provided as a template):

```
OPENAI_API_KEY=sk-...
```

The notebook calls `load_dotenv()` at the top, so the key is picked up automatically.

## Run the notebook

```bash
uv run --with jupyterlab jupyter lab
```

Open `geoai_agent_tutorial.ipynb` and run the cells top to bottom. The first time you reach the CARE section, the notebook will `git clone` the AKD-CARE phase-prompts repo into `./AKD-CARE/` (gitignored).

## Files

- `geoai_agent_tutorial.ipynb` — the workshop notebook.
- `artifacts/Prithvi_WOrkshop_Agent_artifact/` — the pre-built Prithvi agent artifact (scope, contexts, tools, guardrails, reasoning, output, and the assembled `agents.md` entry point).
- `artifacts/care_workspace/` — *(gitignored)* where the CARE interviewer writes the artifact it authors during the session.
- `AKD-CARE/` — *(gitignored)* the auto-cloned phase-prompts library used by the CARE interviewer's `read_prompt` tool.
- `pyproject.toml` / `uv.lock` — project dependencies pinned for reproducibility.
