# GeoAI Agent Tutorial

Workshop notebook.

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

Open `geoai_agent_tutorial.ipynb` and run the cells top to bottom.

## Files

- `geoai_agent_tutorial.ipynb` — the workshop notebook.
- `artifacts/Prithvi_WOrkshop_Agent_artifact/` — the agent artifact (scope, contexts, tools, guardrails, reasoning, output, and the assembled `agents.md` entry point).
- `pyproject.toml` / `uv.lock` — project dependencies pinned for reproducibility.
