# Use the exact Python major/minor version your project targets.
# Why: avoids "works locally, breaks in container" version drift.
FROM python:3.12-slim

# Prevent Python from writing .pyc files and force unbuffered logs.
# Why:
# - .pyc files are unnecessary in containers
# - unbuffered stdout/stderr makes logs appear immediately in Docker/Kubernetes
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_HOME=/app \
    PYTHONPATH=/app

WORKDIR ${APP_HOME}

# Install only minimal OS-level runtime dependencies.
# Why:
# - libgomp1 is commonly needed by LightGBM/XGBoost runtime
# - we avoid installing a large toolchain unless actually required
# - keeping the image small improves pull/deploy speed
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file first to maximize Docker layer caching.
# Why:
# - if application code changes but requirements do not,
#   Docker can reuse the dependency-install layer
COPY requirements-serve.txt .

# Upgrade pip and install Python dependencies.
# Why:
# - predictable dependency install step
# - no cache keeps the final image smaller
# - using requirements-serve.txt excludes heavy training/dev dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements-serve.txt

# Copy only the directories/files needed by the serving app.
# Why:
# - reduces accidental bloat
# - avoids pulling in notebooks, tests, local junk, git history, etc.
COPY common ./common
COPY src ./src
COPY configs ./configs
COPY pyproject.toml ./

# Create a non-root user and drop privileges.
# Why:
# - running as root in containers is a common avoidable security mistake
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser ${APP_HOME}

USER appuser

# Document the API port.
# Why:
# - makes container intent explicit
EXPOSE 8000

# Optional but useful container health check.
# Why:
# - lets container platforms detect startup/readiness problems
# - this checks the FastAPI /health endpoint exposed by your app
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

# Start the FastAPI app with uvicorn.
# Why:
# - this matches your actual project entrypoint: src.deployment.app:app
# - host 0.0.0.0 is required inside containers so traffic can reach the app
CMD ["uvicorn", "src.deployment.app:app", "--host", "0.0.0.0", "--port", "8000"]
