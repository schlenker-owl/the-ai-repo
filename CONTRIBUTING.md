# Contributing to the-ai-repo

Thanks for helping build a world-class learning repo for classical ML → DL → LLMs (and soon RL)!
This guide keeps local runs deterministic and contributions consistent.

## Quickstart (uv)

```bash
# 1) Create env
uv venv --python 3.11
source .venv/bin/activate

# 2) Install project + dev tools
uv sync -g dev

# 3) Install pre-commit hooks
uv run pre-commit install

# 4) Run the full suite locally (auto-fixes code)
uv run pre-commit run -a
uv run pytest -q
```
