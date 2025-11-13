FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/var/cache/huggingface

# System deps (curl for uv installer; build-essential for any wheels that need compiling)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install uv (we'll use the project's uv.lock)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# Copy lock + metadata first (allows caching of dependency layer when code changes)
COPY pyproject.toml uv.lock ./

# Copy the minimal project files needed to build/install the local package
# (src-layout and README referenced by setuptools)
COPY README.md ./README.md
COPY src/ ./src/

# Create the project venv exactly as pinned in uv.lock
# --frozen: use the lockfile; --no-dev: skip dev deps (keeps image slim)
# If you want dev deps inside the image, remove --no-dev.
RUN uv sync --frozen --no-dev --python 3.11

# Activate the venv for subsequent RUN/CMD layers
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:${PATH}"

# Copy the FastAPI app (if you keep it outside src/)
COPY servers/fastapi_lora/app.py ./servers/fastapi_lora/app.py

# HF cache directory
RUN mkdir -p ${HF_HOME}

EXPOSE 8000
CMD ["uvicorn", "servers.fastapi_lora.app:app", "--host", "0.0.0.0", "--port", "8000"]
