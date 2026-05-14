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
- `artifacts/Prithvi_WOrkshop_Agent_artifact/` — the agent artifact (scope, contexts, tools, guardrails, reasoning, output, and the assembled `agents.md` entry point).
- `pyproject.toml` / `uv.lock` — project dependencies pinned for reproducibility.
