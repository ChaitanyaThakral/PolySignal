"""
etl/faers_ingest.py
====================
Dask-based ingestion of FAERS quarterly ASCII zip files.

Architecture decisions:
  WHY DASK?
    Each FAERS quarter unzips to ~300-500 MB across 6 files. 24 quarters
    = ~8 GB before filtering. Loading all into RAM with pandas would require
    ~24 GB. Dask reads lazily in partitions, applies the column filter and
    role filter before materialising, so peak RAM stays under ~2 GB.

  WHY NOT SPARK?
    Single-machine problem (<50 GB). Dask spins up with zero infra overhead.
    Spark would need a cluster setup that adds Day 1's worth of config work.

  WHAT WOULD BREAK AT SCALE:
    - Beyond ~200 GB (all FAERS history since 2004), switch to Dask
      distributed with a scheduler, or to Spark on a cluster.
    - The inner zipfile read is single-threaded because Python's zipfile
      module doesn't support concurrent reads from the same archive.
      We parallelise at the quarter level, not the file level.

Output: cleaned Dask DataFrames for DEMO, DRUG, REAC, OUTC — ready
for the Day 4 normalisation and Day 6 Postgres loader.
"""
from __future__ import annotations

import logging
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import dask.dataframe as dd
import pandas as pd

from etl.schema import (
    ALL_SCHEMAS, AGE_TO_YEARS, WEIGHT_TO_KG, PRIMARY_SUSPECT_ROLES,
    TableSchema, DEMO_SCHEMA, DRUG_SCHEMA, REAC_SCHEMA, OUTC_SCHEMA,
)

log = logging.getLogger(__name__)

DATA_RAW_FAERS = Path(__file__).parent.parent / "data" / "raw" / "faers"


# ── Low-level helpers ──────────────────────────────────────────────────────────

def _read_faers_table_from_zip(zip_path: Path, table_name: str) -> pd.DataFrame:
    """
    Read one FAERS table (DEMO/DRUG/REAC/OUTC) from a quarterly zip.
    Returns a raw pandas DataFrame with lowercase column names.

    FAERS quirks handled here:
      - Delimiter is $ (not comma)
      - Encoding is latin-1 (brand names have non-ASCII chars)
      - Some quarters have an 'ASCII/' subdirectory inside the zip
      - Column names vary in case across quarters
    """
    with zipfile.ZipFile(zip_path) as zf:
        candidates = [
            n for n in zf.namelist()
            if table_name.upper() in n.upper() and n.lower().endswith(".txt")
        ]
        if not candidates:
            log.warning("No %s table in %s", table_name, zip_path.name)
            return pd.DataFrame()

        fname = candidates[0]
        log.debug("Reading %s from %s", fname, zip_path.name)
        raw = zf.read(fname)

    df = pd.read_csv(
        BytesIO(raw),
        sep="$",
        encoding="latin-1",
        low_memory=False,
        on_bad_lines="warn",
    )
    df.columns = [c.lower().strip() for c in df.columns]
    return df


def _apply_schema(df: pd.DataFrame, schema: TableSchema, quarter: str) -> pd.DataFrame:
    """
    Select columns, cast dtypes, tag with quarter label.
    Missing columns (schema drift across quarters) are added as null.
    """
    # Add any columns the schema expects but this quarter lacks
    for col in schema.keep_cols:
        if col not in df.columns:
            log.warning("[%s] Missing column '%s' in %s — filling with NA",
                        quarter, col, schema.name)
            df[col] = pd.NA

    df = df[schema.keep_cols].copy()

    # Apply column renames
    if schema.rename:
        df = df.rename(columns=schema.rename)

    # Cast dtypes (best-effort, coerce on failure)
    for col, dtype in schema.dtypes.items():
        try:
            if dtype == "category":
                df[col] = df[col].astype("category")
            else:
                df[col] = pd.array(df[col], dtype=dtype)
        except Exception as exc:
            log.warning("[%s] Dtype cast failed for %s.%s (%s): %s",
                        quarter, schema.name, col, dtype, exc)

    df["quarter"] = quarter
    return df


# ── Normalisation helpers ──────────────────────────────────────────────────────

def normalise_age(df: pd.DataFrame) -> pd.DataFrame:
    """Convert age to decimal years based on AGE_COD."""
    multiplier = df["age_cod"].astype(str).str.upper().map(AGE_TO_YEARS)
    df["age_years"] = df["age"] * multiplier
    # Sanity bounds: 0–120 years
    df.loc[(df["age_years"] < 0) | (df["age_years"] > 120), "age_years"] = pd.NA
    return df


def normalise_weight(df: pd.DataFrame) -> pd.DataFrame:
    """Convert weight to kg based on WT_COD."""
    multiplier = df["wt_cod"].astype(str).str.upper().map(WEIGHT_TO_KG)
    df["weight_kg"] = df["wt"] * multiplier
    # Sanity bounds: 1–500 kg
    df.loc[(df["weight_kg"] < 1) | (df["weight_kg"] > 500), "weight_kg"] = pd.NA
    return df


def deduplicate_demo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the latest version of each case (max PRIMARYID per CASEID).

    Why: FAERS re-issues updated case reports with a new PRIMARYID but the
    same CASEID. If we don't deduplicate, join counts inflate ~15%.
    """
    before = len(df)
    df = df.sort_values("primaryid").groupby("caseid", as_index=False).last()
    after = len(df)
    log.info("Dedup DEMO: %d → %d rows (%.1f%% duplicates removed)",
             before, after, (before - after) / max(before, 1) * 100)
    return df


# ── Per-quarter ingestion ──────────────────────────────────────────────────────

def ingest_quarter(zip_path: Path) -> Dict[str, pd.DataFrame]:
    """
    Ingest one FAERS quarterly zip.
    Returns dict: {'DEMO': df, 'DRUG': df, 'REAC': df, 'OUTC': df}

    Data quality checkpoints embedded (⚠️):
      - Row count logged before and after ROLE_COD filter
      - Orphan DRUG rows (no matching DEMO) counted and reported
      - Missing PRIMARYID rows dropped and counted
    """
    quarter = zip_path.stem.replace("faers_ascii_", "").upper()
    log.info("Ingesting quarter: %s", quarter)

    results: Dict[str, pd.DataFrame] = {}

    # ── DEMO ──────────────────────────────────────────────────────────────────
    demo_raw = _read_faers_table_from_zip(zip_path, "DEMO")
    if demo_raw.empty:
        log.error("Empty DEMO for %s — skipping quarter", quarter)
        return {}

    demo = _apply_schema(demo_raw, DEMO_SCHEMA, quarter)
    demo = demo.dropna(subset=["primaryid"])
    demo = normalise_age(demo)
    demo = normalise_weight(demo)
    demo = deduplicate_demo(demo)
    results["DEMO"] = demo
    log.info("[%s] DEMO: %d cases after dedup", quarter, len(demo))

    # ── DRUG ──────────────────────────────────────────────────────────────────
    drug_raw = _read_faers_table_from_zip(zip_path, "DRUG")
    drug = _apply_schema(drug_raw, DRUG_SCHEMA, quarter)
    drug = drug.dropna(subset=["primaryid", "drugname"])

    # ⚠️ Role filter
    n_before = len(drug)
    drug = drug[drug["role_cod"].isin(PRIMARY_SUSPECT_ROLES)]
    n_after = len(drug)
    log.info("[%s] DRUG role filter: %d → %d rows (%.1f%% kept)",
             quarter, n_before, n_after, n_after / max(n_before, 1) * 100)

    # ⚠️ Orphan check
    valid_pids = set(demo["primaryid"].dropna())
    orphans = drug[~drug["primaryid"].isin(valid_pids)]
    if len(orphans) > 0:
        log.warning("[%s] %d DRUG rows have no matching DEMO primaryid (dropped)",
                    quarter, len(orphans))
        drug = drug[drug["primaryid"].isin(valid_pids)]

    # Normalise drug names for lookup
    drug["drugname_lower"] = drug["drugname"].str.lower().str.strip()
    drug["prod_ai_lower"] = drug["prod_ai"].str.lower().str.strip().fillna("")
    results["DRUG"] = drug

    # ── REAC ──────────────────────────────────────────────────────────────────
    reac_raw = _read_faers_table_from_zip(zip_path, "REAC")
    reac = _apply_schema(reac_raw, REAC_SCHEMA, quarter)
    reac = reac.dropna(subset=["primaryid", "pt"])
    reac = reac[reac["primaryid"].isin(valid_pids)]
    results["REAC"] = reac
    log.info("[%s] REAC: %d event rows", quarter, len(reac))

    # ── OUTC ──────────────────────────────────────────────────────────────────
    outc_raw = _read_faers_table_from_zip(zip_path, "OUTC")
    outc = _apply_schema(outc_raw, OUTC_SCHEMA, quarter)
    outc = outc.dropna(subset=["primaryid"])
    outc = outc[outc["primaryid"].isin(valid_pids)]
    results["OUTC"] = outc
    log.info("[%s] OUTC: %d outcome rows", quarter, len(outc))

    return results


# ── Multi-quarter Dask ingestion ───────────────────────────────────────────────

def ingest_all_quarters(
    faers_dir: Path = DATA_RAW_FAERS,
    quarters: Optional[List[str]] = None,
) -> Dict[str, dd.DataFrame]:
    """
    Ingest all FAERS quarters found in faers_dir (or a subset if `quarters`
    is specified, e.g. ['2024Q4', '2025Q1']).

    Returns dict of Dask DataFrames: {'DEMO': ddf, 'DRUG': ddf, ...}

    Why Dask here? Each per-quarter pandas DataFrame is ~30-60 MB.
    dask.dataframe.from_delayed creates a lazy graph over all quarters —
    no RAM overhead until you call .compute() or write to Parquet.
    """
    import dask

    zips = sorted(faers_dir.glob("faers_ascii_*.zip"))
    if quarters:
        quarter_set = {q.upper() for q in quarters}
        zips = [z for z in zips if any(q in z.stem.upper() for q in quarter_set)]

    if not zips:
        raise FileNotFoundError(
            f"No FAERS zip files found in {faers_dir}. "
            "Run `python etl/download_raw.py --source faers` first."
        )

    log.info("Ingesting %d FAERS quarters via Dask delayed...", len(zips))

    # Build delayed graph: one ingest_quarter call per zip
    delayed_results = [dask.delayed(ingest_quarter)(z) for z in zips]

    # Separate tables across quarters, then concatenate into Dask DFs
    table_delayed: Dict[str, list] = {t: [] for t in ALL_SCHEMAS}

    for dr in delayed_results:
        for table in ALL_SCHEMAS:
            table_delayed[table].append(dask.delayed(lambda r, t: r.get(t, pd.DataFrame()))(dr, table))

    dask_dfs: Dict[str, dd.DataFrame] = {}
    for table, delayed_list in table_delayed.items():
        schema = ALL_SCHEMAS[table]
        meta = pd.DataFrame(columns=schema.keep_cols + ["quarter"])
        dask_dfs[table] = dd.from_delayed(delayed_list, meta=meta)

    return dask_dfs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # Quick smoke test on a single quarter if available
    zips = sorted(DATA_RAW_FAERS.glob("*.zip"))
    if zips:
        result = ingest_quarter(zips[-1])  # most recent quarter
        for table, df in result.items():
            print(f"{table}: {len(df):,} rows")
    else:
        print("No FAERS zips found — run download_raw.py first")
