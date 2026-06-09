"""
tests/test_structure.py
=======================
Day 1 structural tests — confirm the package layout and stubs are importable.
These are deliberately minimal: they don't test any domain logic yet.
Domain logic tests are added alongside the implementation on each day.
"""

import importlib
import pytest


@pytest.mark.parametrize("module_path", [
    "api.main",
    "signal_detection.disproportionality",
    "graph.hetero_graph",
    "causal.doubly_robust",
])
def test_module_importable(module_path: str) -> None:
    """All stub modules must import without error from Day 1."""
    mod = importlib.import_module(module_path)
    assert mod is not None


def test_stub_raises_not_implemented() -> None:
    """Stubs must raise NotImplementedError (not silently pass or return None)."""
    from signal_detection.disproportionality import compute_prr
    with pytest.raises(NotImplementedError):
        compute_prr(1, 10, 100, 1000)


def test_fastapi_app_exists() -> None:
    """The FastAPI app object must be importable and have the expected title."""
    from api.main import app
    assert app.title == "PolySignal"


def test_health_endpoint() -> None:
    """Health check endpoint must return 200 with status=ok."""
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
