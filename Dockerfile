# ─────────────────────────────────────────────────────────────────────────────
# PolySignal Dockerfile
#
# Why multi-stage?
#   Stage 1 ("builder") installs all build-time deps (gcc, libpq-dev for
#   psycopg2 if you ever switch from the binary wheel) without carrying them
#   into the final runtime image.  Keeps the runtime image ~300 MB smaller.
#
# Why python:3.11-slim?
#   - 3.11 is the sweet spot: supported by torch 2.3, econml 0.15, and
#     torch_geometric 2.5; Debian-slim gives us a ~45 MB base vs ~900 MB full.
#
# What would break at scale:
#   - The CPU torch wheel is ~750 MB.  On a GPU cluster you'd swap the
#     --extra-index-url to the cu121 wheel and add CUDA base image.
#   - dask[complete] pins bokeh/distributed; if you add more graph libs later
#     you may hit solver conflicts — use `pip check` after every install.
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# System deps needed to compile any C extensions (psycopg2 source, scipy, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only the requirements first so Docker caches the pip layer until
# requirements.txt changes — saves minutes on every rebuild.
COPY requirements.txt .

RUN pip install --upgrade pip \
 && pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Runtime system libs (libpq for psycopg2-binary at runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy project source
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Default: start the API. Override with `docker run ... pytest` for tests.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
