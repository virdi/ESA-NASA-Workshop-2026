# GeoAI Agent Tutorial

A two-part workshop notebook that walks through (1) **designing** an agent interactively via the **CARE** process and (2) **wiring up** a pre-built agent artifact (Prithvi-EO-2.0) into a runnable agent using `pydantic-ai` + `pydantic-ai-backends`.

## What's in it

- **Agent Design (CARE)** — chat with an interviewer agent that walks an SME through the four CARE phases (Scope & Decompose → Key Info → Reasoning & Guardrails → Prompt Architecture) and authors the resulting artifact files (`scope.md`, `contexts/*.md`, `tools/...`, `agents.md`, …) directly into a workspace directory. Lifted from [github.com/NASA-IMPACT/akd-labs](https://github.com/NASA-IMPACT/akd-labs); the per-phase prompt library comes from [github.com/NASA-IMPACT/AKD-CARE](https://github.com/NASA-IMPACT/AKD-CARE) and is auto-cloned on first run.
- **FM Agent (Prithvi)** — wires the pre-built Prithvi artifact into a runnable agent with a `LocalBackend` in read-only mode; the runtime tools (`geocode_location`, `check_hls_availability`, `run_prithvi_inference`, plus auxiliary dataset signals) live in `tools/` and are exposed both to the in-notebook agent and to external MCP clients via `mcp_server.py`. Demonstrates async runs, full-trace streaming (tool calls + reasoning + text deltas), and an inline Gradio chat.

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

Open `geoai_agent_tutorial.ipynb` and run the cells top to bottom. The first time you reach the CARE section, the notebook will `git clone` the AKD-CARE phase-prompts repo into `./AKD-CARE/` (gitignored). The CARE interviewer authors its workspace into `./artifacts/care_workspace/` (also gitignored).

## Local MCP server

`mcp_server.py` exposes all 9 tutorial tools as a local MCP server — useful as a
fallback when a remote MCP endpoint isn't available, or to connect external clients
(Claude Code, Claude Desktop, another agent) directly to the tools.

**Start the server** (SSE/HTTP, default):

```bash
uv run python mcp_server.py          # http://127.0.0.1:8080/sse
uv run python mcp_server.py --port 9000
```

**From inside a notebook cell** (background process):

```python
import subprocess, sys
proc = subprocess.Popen([sys.executable, "mcp_server.py", "--port", "8080"])
# proc.terminate() to shut it down
```

### Connecting Claude Code

Claude Code reads MCP server config from `.claude/settings.json` (project-local) or
`~/.claude/settings.json` (global). Add this block to either file, then restart
Claude Code:

```json
{
  "mcpServers": {
    "geoai-tools": {
      "type": "sse",
      "url": "http://127.0.0.1:8080/sse"
    }
  }
}
```

With the server running and the config in place, Claude Code discovers the tools
automatically — you can ask it to call `geocode_location`, `run_prithvi_inference`,
etc. directly in a conversation.

**Tips:**
- The server must be running before Claude Code starts (or use `/mcp` to reload).
- Env vars (`FIRMS_MAP_KEY`, `EARTHDATA_LOGIN`, etc.) must be set in the shell that
  launches `mcp_server.py` — the server loads them from `.env` via `python-dotenv`.
- For a stdio-based connection (Claude Code manages the process lifetime itself), use:

```json
{
  "mcpServers": {
    "geoai-tools": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "python", "mcp_server.py", "--transport", "stdio"],
      "cwd": "/path/to/GeoAI-Agent-Tutorial"
    }
  }
}
```

  With `stdio`, Claude Code spawns and owns the server process — no need to start it
  manually, and env vars are inherited from Claude Code's shell environment.

## Files

- `geoai_agent_tutorial.ipynb` — the workshop notebook.
- `mcp_server.py` — local MCP server for all tutorial tools.
- `artifacts/Prithvi_WOrkshop_Agent_artifact/` — the pre-built Prithvi agent artifact (scope, contexts, tools, guardrails, reasoning, output, and the assembled `agents.md` entry point).
- `artifacts/care_workspace/` — *(gitignored)* where the CARE interviewer writes the artifact it authors during the session.
- `AKD-CARE/` — *(gitignored)* the auto-cloned phase-prompts library used by the CARE interviewer's `read_prompt` tool.
- `pyproject.toml` / `uv.lock` — project dependencies pinned for reproducibility.
