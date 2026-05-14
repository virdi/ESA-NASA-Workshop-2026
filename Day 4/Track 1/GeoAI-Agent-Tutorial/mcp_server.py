"""Local MCP server exposing the GeoAI tutorial tools.

Serves all 9 tools from tools/ over SSE (HTTP) or stdio so any MCP-capable
client (Claude Code, Claude Desktop, a remote agent) can call them without
importing the Python code directly.

Usage
-----
SSE (HTTP) — default, suits notebooks and Claude Code:
    python mcp_server.py                        # localhost:8080
    python mcp_server.py --port 9000            # custom port
    MCP_PORT=9000 python mcp_server.py          # via env var

stdio — for clients that launch the server as a subprocess:
    python mcp_server.py --transport stdio

Environment variables (from .env or shell):
    FIRMS_MAP_KEY, EARTHDATA_LOGIN, EARTHDATA_PASSWORD, PRITHVI_SERVER_URL
    Load them before starting: `source .env` or use python-dotenv in the client.
"""

import argparse
import os

from dotenv import load_dotenv
from fastmcp import FastMCP

from akd_ext.mcp.converter import tool_converter, register_mcp_tool
from tools import TOOLS

load_dotenv()

mcp = FastMCP(
    "geoai-tutorial-tools",
    instructions=(
        "Geospatial tools for the ESA/NASA workshop GeoAI agent: geocoding, "
        "HLS imagery availability, Prithvi-EO inference (flood / burn-scar / crop), "
        "and auxiliary dataset queries (FIRMS, DSWx, MTBS, USDA CDL)."
    ),
)

for tool in TOOLS:
    register_mcp_tool(tool_converter(tool), mcp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GeoAI tutorial MCP server")
    parser.add_argument(
        "--transport",
        choices=["sse", "stdio"],
        default=os.getenv("MCP_TRANSPORT", "sse"),
    )
    parser.add_argument("--host", default=os.getenv("MCP_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MCP_PORT", "8080")))
    args = parser.parse_args()

    if args.transport == "sse":
        print(f"Starting GeoAI MCP server at http://{args.host}:{args.port}/sse")
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        mcp.run()
