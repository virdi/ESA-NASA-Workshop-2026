# EVE Platform Tour

In this folder we find the material for **"Building Agentic Earth Intelligence: A Hands-On Tour of the EVE Platform and Tool Ecosystem"**.

The main resource is `Agentic_EVE.ipynb`, and through it we will explore how to move from a plain LLM workflow to an agentic Earth Observation workflow. Along the way, we build practical tooling and test it end to end.

## What You Learn

- How to design tool functions an LLM can reliably call
- How MCP standardizes tool discovery and invocation
- How EVE integrates public MCP servers into an agentic runtime
- How to inspect intermediate reasoning/tool steps using execution traces

## Folder Contents

- `Agentic_EVE.ipynb`: end-to-end tutorial notebook.

## Prerequisites

- Python 3.10+ (recommended)
- Jupyter environment (JupyterLab, VS Code notebooks, or similar)
- Access to an EVE platform account/API endpoint for login-based sections

## Environment Setup

Create a local `.env` file (same directory as the notebook) with:

```env
BASE_URL=https://<your-eve-platform-base-url>
EMAIL=<your-account-email>
PASSWORD=<your-account-password>
```

> Keep credentials private. Do not commit `.env` files.

The notebook installs these packages in-kernel:

- `python-dotenv`
- `httpx`
- `geopy`
- `requests`
- `fastmcp`

## Run the Notebook

1. Open `Agentic_EVE.ipynb`.
2. Run the **Setup** cells first (dependency install + `.env` loading).
3. Execute sections in order:
  - **Part I**: From services to tools (Nominatim geocode/reverse-geocode tools)
  - **Part II**: EVE Tool Registry (discover and inspect MCP servers/tools)
  - **Part III**: EVE agentic framework (ReAct loop via LangChain/LangGraph + endpoint calls)
4. Review the trace visualizations/output to understand model-to-tool orchestration.

## Troubleshooting

- **Missing dependencies**: re-run the install cell and restart kernel if needed.
- **Auth errors**: verify `BASE_URL`, `EMAIL`, `PASSWORD` in `.env`.
- **No tool results**: confirm login/access token generation succeeded before registry/agentic calls.
- **Kernel state issues**: restart kernel and run cells from top in order.

## Funding

This project is supported by the European Space Agency (ESA) Φ-lab through the Large Language Model for Earth Observation and Earth Science project, as part of the Foresight Element within FutureEO Block 4 programme.

## Citation
If you use this project in academic or research settings, please cite:
```
@misc{atrio2026evedomainspecificllmframework,
      title={{EVE}: A Domain-Specific {LLM} Framework for Earth Intelligence}, 
      author={Àlex R. Atrio and Antonio Lopez and Jino Rohit and Yassine El Ouahidi and Marcello Politi and Vijayasri Iyer and Umar Jamil and Sébastien Bratières and Nicolas Longépé},
      year={2026},
      eprint={2604.13071},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.13071}, 
}
```