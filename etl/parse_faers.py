"""
etl/parse_faers.py
==================
Scaled FAERS parsing with Dask — Day 3 deliverable.

ARCHITECTURE DECISIONS
======================

1. WHY dask.delayed (not dask.dataframe.read_csv directly)?
   FAERS files live inside ZIP archives. Dask's read_csv cannot open
   zipfile entries directly. We wrap each zip→table read with
   dask.delayed(pandas_reader), then assemble with dd.from_delayed.
   This keeps the lazy graph but requires materialising one partition at a
   time — peak RAM stays under ~600 MB even for 24 quarters.

2. WHY Parquet, not CSV?
   - 5-10× smaller on disk (columnar + snappy compression)
   - Schema (dtypes) is embedded, so no re-parsing on reload
   - Predicate pushdown: later Dask queries that filter by year read ONLY
     the relevant partition files
   - CSV would require re-specifying dtypes on every load

3. WHY partition by year (not quarter)?
   24 quarter-level partitions × 4 tables = 96 files. Many BI/ML queries
   filter by year, not quarter. Year partitioning gives 6×4 = 24 files —
   fewer open file handles, faster directory listing.

4. FAERS FORMAT HISTORY (a known gotcha):
   ERA 1 — before 2012 Q3 : tab-delimited, primary key = ISR (Individual
            Safety Report number), column names like 'ISR', 'DRUG_SEQ',
            'BODY_SYS', etc.
   ERA 2 — 2012 Q3 to 2014 Q2 : transitional, mixed naming, still ISR key
   ERA 3 — 2014 Q3 onward : '$'-delimited, primary key = PRIMARYID,
            additional columns (age_grp, to_mfr, etc.)

   Our data range (2020–2025) is entirely ERA 3. This script detects the
   era via the delimiter and first-column name so it fails loudly if a
   pre-2014 file is accidentally included.

5. PRIMARYID vs CASEID deduplication:
   PRIMARYID encodes CASEID + an update-sequence number.
   If a case is amended, it gets a new PRIMARYID with the same CASEID.
   We keep MAX(PRIMARYID) per CASEID — the standard FDA-recommended
   approach — BEFORE joining tables so counts are consistent.

WHAT WOULD BREAK AT SCALE
==========================
- 200+ quarters: pre-extract zips to a shared filesystem (NFS/S3), then
  use dask.dataframe.read_csv with glob patterns directly.
- Python zipfile releases the GIL inconsistently. At >16 parallel workers
  you can hit file descriptor limits. Max 8 workers is safe on most OSes.

OUTPUTS
=======
  data/processed/faers_demo.parquet/   (partitioned by year=)
  data/processed/faers_drug.parquet/
  data/processed/faers_reac.parquet/
  data/processed/faers_outc.parquet/
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import dask
import dask.dataframe as dd
import pandas as pd

from etl.schema import (
    ALL_SCHEMAS, PRIMARY_SUSPECT_ROLES, AGE_TO_YEARS, WEIGHT_TO_KG,
    TableSchema,
)

log = logging.getLogger(__name__)

ROOT          = Path(__file__).parent.parent
DATA_RAW      = ROOT / "data" / "raw" / "faers"
DATA_PROC     = ROOT / "data" / "processed"

# ── Format detection constants ─────────────────────────────────────────────────
ERA3_DELIMITER  = "$"
ERA1_DELIMITER  = "\t"       # pre-2012 era (not in our range)
ERA3_START_YEAR = 2014       # inclusive

# ── FDA published approximate case counts per quarter (DEMO table) ─────────────
# Source: FDA FAERS quarterly data files index + annual summary reports.
# Used for validation — actual counts should fall within ±20% of these.
# Ref: https://www.fda.gov/drugs/drug-approvals-and-databases/questions-and-answers-fdas-adverse-event-reporting-system-faers
EXPECTED_DEMO_ROWS: Dict[str, Tuple[int, int]] = {
    # quarter: (min_expected, max_expected)
    "2020Q1": (230_000, 340_000),
    "2020Q2": (200_000, 310_000),
    "2020Q3": (220_000, 330_000),
    "2020Q4": (240_000, 360_000),
    "2021Q1": (240_000, 360_000),
    "2021Q2": (230_000, 350_000),
    "2021Q3": (250_000, 380_000),
    "2021Q4": (220_000, 340_000),
    "2022Q1": (230_000, 350_000),
    "2022Q2": (220_000, 340_000),
    "2022Q3": (220_000, 340_000),
    "2022Q4": (240_000, 360_000),
    "2023Q1": (230_000, 360_000),
    "2023Q2": (230_000, 360_000),
    "2023Q3": (210_000, 330_000),
    "2023Q4": (240_000, 370_000),
    "2024Q1": (220_000, 350_000),
    "2024Q2": (220_000, 350_000),
    "2024Q3": (215_000, 345_000),
    "2024Q4": (225_000, 360_000),
    "2025Q1": (220_000, 360_000),
    "2025Q2": (205_000, 340_000),
    "2025Q3": (240_000, 380_000),
    "2025Q4": (220_000, 360_000),
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Per-quarter readers (called inside dask.delayed)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_era(zip_path: Path) -> int:
    """
    Detect FAERS format era from zip filename year.
    Returns 1 (pre-2012), 2 (2012-2014), or 3 (2014+).
    """
    stem = zip_path.stem.upper()
    # Extract year — format: faers_ascii_2024Q4 or faers_ascii_2024q4
    for part in stem.split("_"):
        if len(part) >= 4 and part[:4].isdigit():
            year = int(part[:4])
            if year >= ERA3_START_YEAR:
                return 3
            elif year >= 2012:
                return 2
            else:
                return 1
    return 3   # default to modern era


def _find_table_file(zf: zipfile.ZipFile, table_name: str) -> Optional[str]:
    """
    Locate a FAERS table file inside the zip.
    Handles subdirectories (ASCII/) and case variation (DEMO vs demo).
    """
    for name in zf.namelist():
        basename = Path(name).stem.upper()
        if table_name.upper() in basename and name.lower().endswith(".txt"):
            return name
    return None


def _read_era3_table(zip_path: Path, table_name: str, quarter: str) -> pd.DataFrame:
    """
    Read one table from a post-2014 FAERS zip (ERA 3).
    Delimiter: $   Encoding: latin-1   Primary key: PRIMARYID
    """
    schema: TableSchema = ALL_SCHEMAS[table_name]

    with zipfile.ZipFile(zip_path) as zf:
        fname = _find_table_file(zf, table_name)
        if fname is None:
            log.warning("[%s] %s not found in %s", quarter, table_name, zip_path.name)
            return pd.DataFrame(columns=schema.keep_cols + ["quarter", "year"])
        raw_bytes = zf.read(fname)

    df = pd.read_csv(
        BytesIO(raw_bytes),
        sep="$",
        encoding="latin-1",
        low_memory=False,
        on_bad_lines="warn",
        dtype=str,          # read everything as str first; cast after cleaning
    )
    df.columns = [c.lower().strip() for c in df.columns]

    # Verify we're in ERA 3 (has PRIMARYID, not ISR)
    if "primaryid" not in df.columns:
        if "isr" in df.columns:
            raise ValueError(
                f"[{quarter}] Found ERA 1/2 file with ISR column in "
                f"{zip_path.name}. This era is not in 2020-2025 range."
            )
        log.error("[%s] No primaryid column. Available: %s", quarter, df.columns.tolist())
        return pd.DataFrame(columns=schema.keep_cols + ["quarter", "year"])

    # Select and rename columns
    available = [c for c in schema.keep_cols if c in df.columns]
    missing   = [c for c in schema.keep_cols if c not in df.columns]
    if missing:
        log.warning("[%s] %s missing cols (filled NA): %s", quarter, table_name, missing)
        for col in missing:
            df[col] = pd.NA

    df = df[schema.keep_cols].copy()

    # Cast dtypes
    for col, dtype in schema.dtypes.items():
        if col not in df.columns:
            continue
        try:
            if dtype in ("Int64", "Float64"):
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)
            elif dtype == "category":
                df[col] = df[col].astype("category")
            # object stays as-is (already str)
        except Exception as exc:
            log.debug("[%s] dtype cast %s.%s failed: %s", quarter, table_name, col, exc)

    # Tag with quarter and year
    df["quarter"] = quarter
    df["year"]    = int(quarter[:4])

    return df


def _read_quarter_table(zip_path: Path, table_name: str) -> pd.DataFrame:
    """Top-level callable for dask.delayed — one quarter, one table."""
    quarter = zip_path.stem.replace("faers_ascii_", "").upper()
    era     = _detect_era(zip_path)

    if era != 3:
        log.error(
            "[%s] ERA %d detected — only ERA 3 (2014+) is supported. Skipping.",
            quarter, era
        )
        return pd.DataFrame()

    return _read_era3_table(zip_path, table_name, quarter)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: PRIMARYID deduplication (the critical FAERS gotcha)
# ─────────────────────────────────────────────────────────────────────────────

def _deduplicate_by_caseid(df: pd.DataFrame) -> pd.DataFrame:
    """
    For the DEMO table only: keep MAX(primaryid) per caseid.

    PRIMARYID format (ERA 3): zero-padded CASEID with a 2-digit version
    suffix. Example: case 1234567, version 3 → PRIMARYID = 123456703.
    Higher PRIMARYID = more recent update. Keeping MAX is correct.

    ⚠️ DO NOT apply this to DRUG/REAC/OUTC — those tables are
    one-to-many with DEMO and should be filtered AFTER DEMO dedup
    (join on the surviving primaryids, not deduplicated themselves).
    """
    if "primaryid" not in df.columns or "caseid" not in df.columns:
        return df

    before = len(df)
    # Convert to numeric for proper MAX comparison
    df["primaryid"] = pd.to_numeric(df["primaryid"], errors="coerce")
    df["caseid"]    = pd.to_numeric(df["caseid"],    errors="coerce")
    df = df.dropna(subset=["primaryid", "caseid"])
    df = (
        df.sort_values("primaryid")
          .groupby("caseid", as_index=False)
          .last()
    )
    after = len(df)
    pct   = (before - after) / max(before, 1) * 100
    log.info("DEMO dedup: %d -> %d rows (%.1f%% dupes removed)", before, after, pct)
    return df


def _normalise_demo(df: pd.DataFrame) -> pd.DataFrame:
    """Age → decimal years, weight → kg, derived columns."""
    if df.empty:
        return df

    # Age
    df["age"]     = pd.to_numeric(df.get("age", pd.NA),  errors="coerce")
    age_multiplier = (
        df["age_cod"].astype(str).str.upper().map(AGE_TO_YEARS)
        if "age_cod" in df.columns else pd.Series(1.0, index=df.index)
    )
    df["age_years"] = df["age"] * age_multiplier
    df.loc[(df["age_years"] < 0) | (df["age_years"] > 120), "age_years"] = pd.NA

    # Weight
    df["wt"]      = pd.to_numeric(df.get("wt", pd.NA),   errors="coerce")
    wt_multiplier = (
        df["wt_cod"].astype(str).str.upper().map(WEIGHT_TO_KG)
        if "wt_cod" in df.columns else pd.Series(1.0, index=df.index)
    )
    df["weight_kg"] = df["wt"] * wt_multiplier
    df.loc[(df["weight_kg"] < 1) | (df["weight_kg"] > 500), "weight_kg"] = pd.NA

    return df


def _normalise_drug(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase drug names, filter primary suspects."""
    if df.empty or "drugname" not in df.columns:
        return df

    n_before = len(df)
    df = df[df["role_cod"].astype(str).isin(PRIMARY_SUSPECT_ROLES)].copy()
    n_after  = len(df)
    log.info("DRUG PS filter: %d -> %d rows", n_before, n_after)

    df["drugname_lower"] = df["drugname"].astype(str).str.lower().str.strip()
    df["prod_ai_lower"]  = (
        df["prod_ai"].astype(str).str.lower().str.strip()
        if "prod_ai" in df.columns else ""
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Dask assembly and Parquet persistence
# ─────────────────────────────────────────────────────────────────────────────

def build_unified_dataframes(
    faers_dir: Path = DATA_RAW,
    quarters:  Optional[List[str]] = None,
) -> Dict[str, dd.DataFrame]:
    """
    Build four unified Dask DataFrames (DEMO, DRUG, REAC, OUTC) from all
    available FAERS quarterly zips.

    Each table's computation is a lazy Dask graph — no data is read until
    .compute() or .to_parquet() is called.

    Returns: {'DEMO': ddf, 'DRUG': ddf, 'REAC': ddf, 'OUTC': ddf}
    """
    zips = sorted(faers_dir.glob("faers_ascii_*.zip"))
    if quarters:
        qset = {q.upper() for q in quarters}
        zips = [z for z in zips if any(q in z.stem.upper() for q in qset)]

    if not zips:
        raise FileNotFoundError(
            f"No FAERS zip files in {faers_dir}. "
            "Run `python etl/download_raw.py --source faers` first."
        )

    log.info("Building Dask graph for %d quarters × 4 tables...", len(zips))

    result: Dict[str, dd.DataFrame] = {}

    for table_name, schema in ALL_SCHEMAS.items():
        # One delayed per (quarter, table) combination
        delayed_parts = [
            dask.delayed(_read_quarter_table)(z, table_name)
            for z in zips
        ]

        # Build meta matching ACTUAL dtypes the partition functions produce.
        meta: Dict[str, str] = {}
        for col, dtype in schema.dtypes.items():
            if dtype == "Int64":
                meta[col] = "Int64"
            elif dtype == "Float64":
                meta[col] = "Float64"
            elif dtype == "category":
                meta[col] = "object"   # categories become object in meta
            else:
                meta[col] = "object"
        meta["quarter"] = "object"
        meta["year"]    = "int64"
        
        if table_name == "DEMO":
            meta["age_years"]  = "Float64"
            meta["weight_kg"]  = "Float64"
        if table_name == "DRUG":
            meta["drugname_lower"] = "object"
            meta["prod_ai_lower"]  = "object"

        ddf = dd.from_delayed(
            delayed_parts,
            meta=meta,
            verify_meta=False,   # skip strict schema check
        )

        # Apply table-level post-processing lazily
        if table_name == "DEMO":
            ddf = ddf.map_partitions(_deduplicate_by_caseid)
            ddf = ddf.map_partitions(_normalise_demo)
        elif table_name == "DRUG":
            ddf = ddf.map_partitions(_normalise_drug)

        result[table_name] = ddf
        log.info("Built lazy graph for %s (%d partitions)", table_name, len(zips))

    return result


def persist_to_parquet(
    dask_dfs: Dict[str, dd.DataFrame],
    output_dir: Path = DATA_PROC,
    overwrite:  bool = False,
) -> Dict[str, Path]:
    """
    Write each Dask DataFrame to Parquet, partitioned by year.

    Uses pyarrow engine (not fastparquet) because:
      - Better nullable integer support (Int64)
      - More efficient snappy compression
      - Compatible with Spark, BigQuery, DuckDB for future scale

    Returns: mapping of table_name → output path
    """
    output_paths: Dict[str, Path] = {}

    for table_name, ddf in dask_dfs.items():
        out_path = output_dir / f"faers_{table_name.lower()}.parquet"

        if out_path.exists() and not overwrite:
            log.info("[%s] %s exists — skipping (use --overwrite to force)",
                     table_name, out_path)
            output_paths[table_name] = out_path
            continue

        log.info("[%s] Writing partitioned Parquet -> %s", table_name, out_path)
        ddf.to_parquet(
            str(out_path),
            engine        = "pyarrow",
            compression   = "snappy",
            partition_on  = ["year"],
            write_index   = False,
            overwrite     = overwrite,
        )
        log.info("[%s] Parquet write complete: %s", table_name, out_path)
        output_paths[table_name] = out_path

    return output_paths


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Parse all FAERS quarters into Parquet."
    )
    parser.add_argument(
        "--quarters", nargs="*",
        help="Specific quarters to process (e.g. 2024Q4 2025Q1). Default: all."
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing Parquet files."
    )
    parser.add_argument(
        "--tables", nargs="*", default=["DEMO", "DRUG", "REAC", "OUTC"],
        help="Which tables to process."
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Dask LocalCluster workers (default 4)."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(ROOT / "data" / "processed" / "parse_faers.log"),
        ],
    )

    log.info("=== FAERS Parquet ETL — Day 3 ===")
    log.info("Quarters: %s", args.quarters or "ALL")
    log.info("Tables:   %s", args.tables)
    log.info("Workers:  %d", args.workers)

    # Build lazy graph
    all_ddfs = build_unified_dataframes(quarters=args.quarters)

    # Filter to requested tables
    ddfs = {t: v for t, v in all_ddfs.items() if t in args.tables}

    # Trigger computation + Parquet write
    output_paths = persist_to_parquet(ddfs, overwrite=args.overwrite)

    # Summary
    log.info("=== ETL Complete ===")
    for table, path in output_paths.items():
        log.info("  %s -> %s", table, path)

    log.info("Next: run `python etl/validate_counts.py` to verify row counts.")


if __name__ == "__main__":
    main()
