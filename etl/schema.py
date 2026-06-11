"""
etl/schema.py
=============
Column definitions, dtypes, and rename maps for every FAERS table.

Centralising this here means the Dask ingestion and Pandas exploration
both import from a single source of truth — no column-name drift.

Design: dataclasses rather than plain dicts so IDEs can autocomplete
column names and catch typos at development time, not at 2 AM on a
production run.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List

# ── FAERS table schemas ────────────────────────────────────────────────────────

@dataclass
class TableSchema:
    name: str
    keep_cols: List[str]          # columns we import (all others dropped)
    rename: Dict[str, str]        # source_col → target_col
    dtypes: Dict[str, str]        # target_col → pandas dtype string


DEMO_SCHEMA = TableSchema(
    name="DEMO",
    keep_cols=[
        "primaryid", "caseid", "fda_dt", "rept_cod",
        "gndr_cod", "age", "age_cod", "wt", "wt_cod",
        "occp_cod", "reporter_country",
    ],
    rename={},   # already lowercase in ASCII files since 2014
    dtypes={
        "primaryid":        "Int64",
        "caseid":           "Int64",
        "fda_dt":           "object",    # parse to date in ETL step
        "rept_cod":         "category",
        "gndr_cod":         "category",
        "age":              "Float64",
        "age_cod":          "category",
        "wt":               "Float64",
        "wt_cod":           "category",
        "occp_cod":         "category",
        "reporter_country": "category",
    },
)

DRUG_SCHEMA = TableSchema(
    name="DRUG",
    keep_cols=[
        "primaryid", "caseid", "drug_seq", "role_cod",
        "drugname", "prod_ai", "route",
    ],
    rename={},
    dtypes={
        "primaryid":  "Int64",
        "caseid":     "Int64",
        "drug_seq":   "Int64",
        "role_cod":   "category",
        "drugname":   "object",
        "prod_ai":    "object",
        "route":      "category",
    },
)

REAC_SCHEMA = TableSchema(
    name="REAC",
    keep_cols=["primaryid", "caseid", "pt", "drug_rec_act"],
    rename={},
    dtypes={
        "primaryid":    "Int64",
        "caseid":       "Int64",
        "pt":           "object",
        "drug_rec_act": "category",
    },
)

OUTC_SCHEMA = TableSchema(
    name="OUTC",
    keep_cols=["primaryid", "caseid", "outc_cod"],
    rename={},
    dtypes={
        "primaryid": "Int64",
        "caseid":    "Int64",
        "outc_cod":  "category",
    },
)

ALL_SCHEMAS = {
    "DEMO": DEMO_SCHEMA,
    "DRUG": DRUG_SCHEMA,
    "REAC": REAC_SCHEMA,
    "OUTC": OUTC_SCHEMA,
}

# ── Age normalization ──────────────────────────────────────────────────────────
# Multipliers to convert each AGE_COD to decimal years.
AGE_TO_YEARS = {
    "YR":  1.0,
    "DEC": 10.0,
    "MON": 1 / 12,
    "WK":  1 / 52.18,
    "DY":  1 / 365.25,
    "HR":  1 / 8766.0,
}

# ── Weight normalization ───────────────────────────────────────────────────────
WEIGHT_TO_KG = {
    "KG":  1.0,
    "LBS": 0.453592,
    "LB":  0.453592,
}

# ── ROLE_COD filter ───────────────────────────────────────────────────────────
PRIMARY_SUSPECT_ROLES = {"PS"}   # expand to {"PS", "SS"} if you want suspects too

# ── Outcome severity ordering (for aggregation) ───────────────────────────────
OUTC_SEVERITY = {
    "DE": 6,   # Death (highest)
    "LT": 5,
    "HO": 4,
    "DS": 3,
    "CA": 3,
    "RI": 2,
    "OT": 1,
}
