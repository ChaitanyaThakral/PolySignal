"""
PolySignal — api/main.py
========================
Skeleton FastAPI application.  Routes will be filled in on Day 20 (serving layer).
Included now so:
  1. The Docker CMD can reference a real entry point.
  2. The smoke test confirms the import chain works end-to-end.
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(
    title="PolySignal",
    description=(
        "Pharmacovigilance signal detection via dual-method triangulation: "
        "classical disproportionality statistics + graph neural network link prediction."
    ),
    version="0.1.0",
)


@app.get("/health", tags=["ops"])
async def health_check() -> JSONResponse:
    """Liveness probe — used by Docker Compose and k8s health checks."""
    return JSONResponse({"status": "ok", "service": "polysignal-api"})


@app.get("/", tags=["ops"])
async def root() -> JSONResponse:
    return JSONResponse({
        "message": "PolySignal API — see /docs for the interactive Swagger UI."
    })
