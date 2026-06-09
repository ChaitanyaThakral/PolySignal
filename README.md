# PolySignal

> Pharmacovigilance signal detection via **dual-method triangulation** — classical disproportionality statistics (PRR / ROR / EBGM) cross-validated against a heterogeneous graph neural network, with doubly-robust causal effect estimation for confirmed signals.

---

## Architecture overview

```
FAERS raw data
     │
     ▼
  /etl ─── Dask-based ETL, FAERS quarterly zip ingestion → Postgres
     │
     ├──► /signal_detection ── Classical: PRR, ROR, EBGM (statsmodels / scipy)
     │
     ├──► /graph ─────────────  GNN: PyG HeteroData + link prediction (torch_geometric)
     │
     ├──► Cross-validation ──── Only pairs both methods flag proceed ↓
     │
     ▼
  /causal ── Doubly-robust ATE estimation (EconML LinearDRLearner)
     │
     ▼
  /api ──── FastAPI serving layer  →  /signals, /causal-effects endpoints
```

**Why two independent methods?**  
This mirrors how regulatory agencies (FDA, EMA) actually work: disproportionality analysis is the workhorse, but it has well-known failure modes (masking, competing-risk confounding). A GNN learns structural co-occurrence patterns the statistics miss. Signals that both methods agree on are far more likely to be real — and the causal estimation only runs on that filtered set, keeping compute costs manageable.

---

## Project structure

| Directory | Purpose |
|---|---|
| `/etl` | FAERS zip → Postgres ingestion; drug/event normalization |
| `/signal_detection` | PRR, ROR, EBGM disproportionality statistics |
| `/graph` | PyG HeteroData graph construction + GNN model |
| `/causal` | Doubly-robust ATE estimation (EconML) |
| `/api` | FastAPI endpoints for signal serving |
| `/tests` | pytest suite |
| `/notebooks` | EDA, concept primers, result visualization |

---

## Day-by-day build plan

| Day | Deliverable |
|---|---|
| **1** | Repo structure, Docker environment, smoke test ← *you are here* |
| **7** | PRR / ROR / EBGM implementation + unit tests |
| **10** | Decide Neo4j vs in-memory PyG for graph storage |
| **12** | HeteroData graph builder + GNN link prediction model |
| **19** | Doubly-robust causal effect estimation |
| **20** | FastAPI serving layer + Docker Compose full-stack |

---

## Quick start

```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify environment
python smoke_test.py

# 4. Start Postgres
docker compose up postgres -d

# 5. Run tests
pytest tests/
```

---

## Datasets

| Dataset | Purpose |
|---|---|
| FAERS quarterly files | Raw adverse event reports (primary signal source) |
| TWOSIDES / nSIDES | Pre-mined drug-pair → side-effect associations (ground truth) |
| DrugBank (free tier) | Known drug-drug interactions (ground truth) |
| RxNorm (NLM) | Canonical drug name normalization |

---

## Key concepts (Day 1 primer)

### Disproportionality analysis

**PRR (Proportional Reporting Ratio):** Ratio of the proportion of reports for drug D mentioning event E, to the proportion of *all other* reports mentioning E. PRR > 2 with ≥3 reports is a common threshold. Problem: sparse pairs (1-2 reports) produce PRR = ∞ or wildly inflated values.

**ROR (Reporting Odds Ratio):** Same idea, framed as odds. Slightly more statistically principled than PRR; 95% CI lower bound > 1 is the threshold.

**EBGM (Empirical Bayes Geometric Mean):** DuMouchel's shrinkage estimator. Fits a mixture of Gamma-Poisson distributions to the *entire* drug-event contingency table, then shrinks each cell's PRR toward the prior mean. EB05 (5th percentile) > 2 is the FDA MGPS threshold. **This is what solves the sparse-pair inflation problem** — a drug-event pair with 1 report gets shrunk toward the population average rather than reporting PRR = 40.

### Doubly robust estimation

Combining **propensity score weighting** (models who *received* the treatment drug) with **outcome regression** (models the ATE directly from covariates). The key insight: the estimate is *consistent* if **either** model is correctly specified — you don't need both to be right. This robustness is why it's the standard for observational causal inference in drug safety.
