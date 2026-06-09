"""
PolySignal — smoke_test.py
==========================
Imports every library used in the project and prints a version report.
Run this immediately after `pip install -r requirements.txt` to confirm
the environment is wired correctly before spending any time writing domain code.

Usage:
    python smoke_test.py

Expected output: a table of library versions with no ImportErrors.
If anything fails, fix it before Day 2 — downstream code assumes all of these.
"""

import sys
import importlib
from typing import List, Tuple

# ── (library_import_name, display_name, version_attr) ──────────────────────
REQUIRED_LIBS: List[Tuple[str, str, str]] = [
    # Core scientific
    ("numpy",           "NumPy",                "__version__"),
    ("pandas",          "Pandas",               "__version__"),
    ("scipy",           "SciPy",                "__version__"),
    ("statsmodels",     "Statsmodels",          "__version__"),
    # Dask
    ("dask",            "Dask",                 "__version__"),
    ("distributed",     "Dask Distributed",     "__version__"),
    # PyTorch
    ("torch",           "PyTorch",              "__version__"),
    ("torch_geometric", "PyTorch Geometric",    "__version__"),
    # Causal inference
    ("econml",          "EconML",               "__version__"),
    ("causalml",        "CausalML",             "__version__"),
    # API
    ("fastapi",         "FastAPI",              "__version__"),
    ("uvicorn",         "Uvicorn",              "__version__"),
    ("pydantic",        "Pydantic",             "__version__"),
    # Database
    ("psycopg2",        "psycopg2",             "__version__"),
    ("neo4j",           "Neo4j driver",         "__version__"),
    ("sqlalchemy",      "SQLAlchemy",           "__version__"),
    # Testing
    ("pytest",          "pytest",               "__version__"),
    ("httpx",           "HTTPX",                "__version__"),
    # Notebooks / viz
    ("jupyterlab",      "JupyterLab",           "__version__"),
    ("matplotlib",      "Matplotlib",           "__version__"),
    ("seaborn",         "Seaborn",              "__version__"),
    ("plotly",          "Plotly",               "__version__"),
]


def check_imports() -> None:
    col_w = 24
    print(f"\n{'Library':<{col_w}} {'Version':<20} {'Status'}")
    print("─" * 60)

    failures: List[str] = []

    for import_name, display_name, ver_attr in REQUIRED_LIBS:
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, ver_attr, "unknown")
            print(f"{display_name:<{col_w}} {str(version):<20} ✓")
        except ImportError as exc:
            print(f"{display_name:<{col_w}} {'—':<20} ✗  {exc}")
            failures.append(display_name)

    print("─" * 60)

    # ── PyTorch-specific checks ───────────────────────────────────────────────
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"\nPyTorch CUDA available : {cuda_available}")
        if not cuda_available:
            print("  → Running on CPU (expected for local dev). "
                  "Swap to cu121 wheel on GPU cluster.")
        print(f"PyTorch device count   : {torch.cuda.device_count()}")
    except ImportError:
        pass

    # ── PyG heterogeneous graph smoke test ───────────────────────────────────
    try:
        from torch_geometric.data import HeteroData
        data = HeteroData()
        import torch
        data["drug"].x = torch.zeros(10, 8)        # 10 drug nodes, 8 features
        data["event"].x = torch.zeros(20, 8)       # 20 event nodes
        data["drug", "causes", "event"].edge_index = torch.zeros(2, 5, dtype=torch.long)
        print(f"\nPyG HeteroData smoke   : ✓  {data}")
    except Exception as exc:
        print(f"\nPyG HeteroData smoke   : ✗  {exc}")
        failures.append("PyG HeteroData")

    # ── EconML doubly-robust smoke test ─────────────────────────────────────
    try:
        import numpy as np
        from econml.dr import LinearDRLearner
        from sklearn.linear_model import LogisticRegression, LinearRegression
        dr = LinearDRLearner(
            model_propensity=LogisticRegression(),
            model_regression=LinearRegression(),
        )
        print("EconML DR smoke        : ✓  LinearDRLearner instantiated")
    except Exception as exc:
        print(f"EconML DR smoke        : ✗  {exc}")
        failures.append("EconML DR")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    if failures:
        print(f"❌  {len(failures)} import(s) failed: {', '.join(failures)}")
        print("   Fix these before proceeding to Day 2.")
        sys.exit(1)
    else:
        print("✅  All imports successful — environment is ready.")
        print("   Proceed to Day 2: ETL pipeline for FAERS quarterly zips.")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    check_imports()
