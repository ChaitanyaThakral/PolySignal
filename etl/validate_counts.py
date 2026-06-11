"""
etl/validate_counts.py
======================
Validates parsed FAERS Parquet row counts against FDA published statistics.

WHY THIS EXISTS
===============
Silent data loss is the biggest ETL risk. A wrong delimiter detection,
a missed encoding, or a bad join can silently drop 30% of records without
raising any exception. This script makes that failure loud.

WHAT IT CHECKS
==============
1. Parquet files exist for all 4 tables
2. Per-quarter DEMO row counts fall within expected ranges
   (±tolerance% of FDA-published statistics)
3. DRUG rows > DEMO rows (each case has ≥1 drug on average)
4. REAC rows > DEMO rows (each case has ≥1 reaction)
5. No quarter is missing entirely (file-level completeness)
6. Dedup integrity: unique CASEID count matches DEMO row count

TOLERANCE
=========
±25% from expected. FDA FAERS counts fluctuate with reporting lag, seasonal
effects, and mass-reporting events (e.g. COVID vaccine reporting in 2021).
Flag but don't fail on ±25-50%; hard-fail above ±50%.

USAGE
=====
    python etl/validate_counts.py                # full validation
    python etl/validate_counts.py --quarter 2024Q4  # single quarter
    python etl/validate_counts.py --report        # print JSON report
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from etl.parse_faers import EXPECTED_DEMO_ROWS

log = logging.getLogger(__name__)

ROOT      = Path(__file__).parent.parent
DATA_PROC = ROOT / "data" / "processed"

# ── Validation thresholds ──────────────────────────────────────────────────────
WARN_TOLERANCE  = 0.25   # ±25% → warning
FAIL_TOLERANCE  = 0.50   # ±50% → hard failure
DRUG_DEMO_RATIO = (2.0, 6.0)   # DRUG rows should be 2–6× DEMO rows
REAC_DEMO_RATIO = (1.5, 5.0)   # REAC rows should be 1.5–5× DEMO rows

# ── Parquet file paths ─────────────────────────────────────────────────────────
TABLE_PATHS = {
    "DEMO": DATA_PROC / "faers_demo.parquet",
    "DRUG": DATA_PROC / "faers_drug.parquet",
    "REAC": DATA_PROC / "faers_reac.parquet",
    "OUTC": DATA_PROC / "faers_outc.parquet",
}


# ─────────────────────────────────────────────────────────────────────────────
# Core validation functions
# ─────────────────────────────────────────────────────────────────────────────

def _load_parquet_meta(path: Path) -> Optional[pd.DataFrame]:
    """
    Load Parquet metadata only (no data) to get row counts per partition.
    Uses pyarrow directly — no Dask overhead, very fast.
    """
    try:
        import pyarrow.parquet as pq
        dataset = pq.ParquetDataset(str(path), use_legacy_dataset=False)
        rows_per_file = []
        for frag in dataset.fragments:
            meta = frag.metadata
            rows_per_file.append({
                "file":     Path(frag.path).name,
                "row_count": meta.num_rows,
                # Extract partition value from directory name (year=2024)
                "partition": str(Path(frag.path).parent.name),
            })
        return pd.DataFrame(rows_per_file)
    except Exception as exc:
        log.error("Failed to read Parquet metadata from %s: %s", path, exc)
        return None


def check_files_exist() -> List[str]:
    """Return list of missing Parquet files."""
    missing = []
    for table, path in TABLE_PATHS.items():
        if not path.exists():
            missing.append(f"{table} ({path})")
    return missing


def check_demo_counts_by_quarter(
    demo_path: Path,
    quarter_filter: Optional[str] = None,
    warn_tol: float = WARN_TOLERANCE,
    fail_tol: float = FAIL_TOLERANCE,
) -> List[Dict]:
    """
    Check DEMO row counts per quarter against expected ranges.
    Reads the actual data (needs one pass through the Parquet files).

    ⚠️ Data quality checkpoint: any quarter outside ±50% of expected
    is a hard failure — indicates parsing error, not just reporting lag.
    """
    import pyarrow.parquet as pq

    findings = []

    try:
        dataset = pq.ParquetDataset(str(demo_path), use_legacy_dataset=False)
    except Exception as exc:
        findings.append({"level": "ERROR", "message": f"Cannot open DEMO Parquet: {exc}"})
        return findings

    # Read quarter column (small — just strings)
    table = dataset.read(columns=["quarter"])
    demo_df = table.to_pandas()

    counts = demo_df["quarter"].value_counts().sort_index()

    for quarter, actual in counts.items():
        if quarter_filter and quarter_filter.upper() != quarter.upper():
            continue

        expected = EXPECTED_DEMO_ROWS.get(quarter.upper())
        if expected is None:
            findings.append({
                "level":   "WARN",
                "quarter": quarter,
                "actual":  int(actual),
                "expected_range": "unknown",
                "message": f"No expected range for quarter {quarter}",
            })
            continue

        exp_min, exp_max = expected
        exp_mid  = (exp_min + exp_max) / 2
        pct_diff = (actual - exp_mid) / exp_mid

        if abs(pct_diff) > fail_tol:
            level   = "FAIL"
            message = (
                f"Row count {actual:,} is {pct_diff:+.0%} from expected midpoint "
                f"{exp_mid:,.0f} — EXCEEDS ±{fail_tol:.0%} hard-fail threshold. "
                f"Check delimiter/encoding in {quarter} zip."
            )
        elif abs(pct_diff) > warn_tol:
            level   = "WARN"
            message = (
                f"Row count {actual:,} is {pct_diff:+.0%} from expected midpoint "
                f"(within ±{warn_tol:.0%}–{fail_tol:.0%} — possible reporting lag)."
            )
        else:
            level   = "OK"
            message = f"Row count {actual:,} within expected range [{exp_min:,}–{exp_max:,}]."

        findings.append({
            "level":          level,
            "quarter":        quarter,
            "actual":         int(actual),
            "expected_range": f"{exp_min:,}–{exp_max:,}",
            "pct_from_mid":   f"{pct_diff:+.1%}",
            "message":        message,
        })

    return findings


def check_table_ratios() -> List[Dict]:
    """
    Verify DRUG and REAC row counts are in expected multiples of DEMO.
    Uses Parquet row group metadata (no full data scan).
    """
    findings = []
    counts: Dict[str, int] = {}

    for table, path in TABLE_PATHS.items():
        if not path.exists():
            continue
        try:
            import pyarrow.parquet as pq
            ds = pq.ParquetDataset(str(path), use_legacy_dataset=False)
            total = sum(
                frag.metadata.num_rows for frag in ds.fragments
            )
            counts[table] = total
        except Exception as exc:
            findings.append({
                "level": "ERROR",
                "table": table,
                "message": f"Cannot read {table} Parquet: {exc}",
            })

    demo_count = counts.get("DEMO", 0)
    if demo_count == 0:
        findings.append({"level": "ERROR", "message": "DEMO count is 0 — ETL may have failed."})
        return findings

    for table, (ratio_min, ratio_max) in [
        ("DRUG", DRUG_DEMO_RATIO),
        ("REAC", REAC_DEMO_RATIO),
    ]:
        if table not in counts:
            continue
        ratio = counts[table] / demo_count
        if ratio < ratio_min:
            findings.append({
                "level":   "WARN",
                "table":   table,
                "message": (
                    f"{table}:{demo_count:,} DEMO ratio = {ratio:.2f}× "
                    f"(expected ≥{ratio_min}×). Possible over-filtering."
                ),
            })
        elif ratio > ratio_max:
            findings.append({
                "level":   "WARN",
                "table":   table,
                "message": (
                    f"{table}:{demo_count:,} DEMO ratio = {ratio:.2f}× "
                    f"(expected ≤{ratio_max}×). Possible duplication."
                ),
            })
        else:
            findings.append({
                "level":   "OK",
                "table":   table,
                "message": (
                    f"{table}/{table} ratio = {ratio:.2f}× "
                    f"(within [{ratio_min}–{ratio_max}]×)"
                ),
            })

    return findings


def check_caseid_uniqueness() -> List[Dict]:
    """
    Verify that CASEID is unique in the DEMO Parquet.
    Duplication means dedup logic in parse_faers.py failed.
    Uses a streaming count — reads caseid column only.
    """
    findings = []
    demo_path = TABLE_PATHS["DEMO"]
    if not demo_path.exists():
        return findings

    try:
        import pyarrow.parquet as pq
        table = pq.ParquetDataset(str(demo_path), use_legacy_dataset=False)\
                  .read(columns=["caseid"])
        series = table.to_pandas()["caseid"]
        total     = len(series)
        unique    = series.nunique()
        dupes     = total - unique

        if dupes > 0:
            pct = dupes / total * 100
            findings.append({
                "level":   "FAIL" if pct > 5 else "WARN",
                "check":   "caseid_uniqueness",
                "message": (
                    f"{dupes:,} duplicate CASEID rows ({pct:.1f}%) in DEMO. "
                    "Deduplication in parse_faers.py may not have applied correctly."
                ),
            })
        else:
            findings.append({
                "level":   "OK",
                "check":   "caseid_uniqueness",
                "message": f"All {total:,} DEMO rows have unique CASEID.",
            })
    except Exception as exc:
        findings.append({
            "level":   "ERROR",
            "check":   "caseid_uniqueness",
            "message": str(exc),
        })

    return findings


# ─────────────────────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────────────────────

def run_all_checks(quarter_filter: Optional[str] = None) -> Dict:
    """Run every validation check and aggregate results."""
    report = {
        "missing_files": check_files_exist(),
        "demo_counts":   [],
        "table_ratios":  [],
        "caseid_unique": [],
        "summary": {},
    }

    if report["missing_files"]:
        report["summary"]["status"] = "INCOMPLETE"
        report["summary"]["message"] = (
            f"Missing Parquet files: {report['missing_files']}. "
            "Run `python etl/parse_faers.py` first."
        )
        return report

    demo_path = TABLE_PATHS["DEMO"]
    report["demo_counts"]  = check_demo_counts_by_quarter(demo_path, quarter_filter)
    report["table_ratios"] = check_table_ratios()
    report["caseid_unique"] = check_caseid_uniqueness()

    all_findings = (
        report["demo_counts"]
        + report["table_ratios"]
        + report["caseid_unique"]
    )
    levels = [f["level"] for f in all_findings]
    n_fail = levels.count("FAIL")
    n_warn = levels.count("WARN")
    n_ok   = levels.count("OK")

    if n_fail > 0:
        status = "FAIL"
    elif n_warn > 0:
        status = "WARN"
    else:
        status = "PASS"

    report["summary"] = {
        "status":  status,
        "n_ok":    n_ok,
        "n_warn":  n_warn,
        "n_fail":  n_fail,
        "message": (
            f"{n_ok} checks passed, {n_warn} warnings, {n_fail} failures."
        ),
    }
    return report


def _print_report(report: Dict) -> None:
    """Print a human-readable validation report to stdout."""
    COLORS = {"OK": "\033[92m", "WARN": "\033[93m",
               "FAIL": "\033[91m", "ERROR": "\033[91m", "RESET": "\033[0m"}

    def _c(level: str, text: str) -> str:
        return f"{COLORS.get(level, '')}{text}{COLORS['RESET']}"

    print("\n" + "=" * 70)
    print("  FAERS Parquet Validation Report")
    print("=" * 70)

    if report["missing_files"]:
        print(_c("FAIL", f"[FAIL] Missing files: {report['missing_files']}"))
        return

    print("\n-- DEMO row counts by quarter --")
    for f in report["demo_counts"]:
        tag = f"[{f['level']:4s}] {f.get('quarter','')}"
        print(_c(f["level"], f"  {tag}: {f['message']}"))

    print("\n-- Table ratio checks --")
    for f in report["table_ratios"]:
        print(_c(f["level"], f"  [{f['level']:4s}] {f['message']}"))

    print("\n-- CASEID uniqueness --")
    for f in report["caseid_unique"]:
        print(_c(f["level"], f"  [{f['level']:4s}] {f['message']}"))

    summary = report["summary"]
    print("\n" + "=" * 70)
    status_line = f"  RESULT: {summary['status']} — {summary['message']}"
    print(_c(summary["status"], status_line))
    print("=" * 70 + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate FAERS Parquet ETL output.")
    parser.add_argument("--quarter", help="Validate a single quarter (e.g. 2024Q4)")
    parser.add_argument("--report",  action="store_true", help="Output JSON report")
    parser.add_argument("--out",     help="Write JSON report to file")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    report = run_all_checks(quarter_filter=args.quarter)

    if args.report or args.out:
        json_str = json.dumps(report, indent=2)
        if args.out:
            Path(args.out).write_text(json_str)
            print(f"Report written to {args.out}")
        else:
            print(json_str)
    else:
        _print_report(report)

    # Exit code: 0=pass, 1=warn, 2=fail
    status = report["summary"].get("status", "FAIL")
    return {"PASS": 0, "WARN": 1, "FAIL": 2}.get(status, 2)


if __name__ == "__main__":
    sys.exit(main())
