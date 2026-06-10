# -*- coding: utf-8 -*-
"""
etl/download_raw.py
===================
Downloads all raw data sources for PolySignal.

Architecture decision — why separate download script, not inline in ETL?
  Downloading is a one-time, network-bound operation. ETL (Day 4) is
  CPU/IO-bound transformation. Separating them means you can re-run ETL
  without re-downloading 1.5 GB of FAERS zips each time.

Concurrency model:
  ThreadPoolExecutor, not asyncio. Reason: urllib/requests release the GIL
  during socket I/O, so threads get true parallelism for network downloads
  without the complexity of aiohttp. Max 4 workers to avoid hammering FDA
  servers (they rate-limit around 5 concurrent connections).

What would break at scale:
  - FDA servers occasionally return 503 during batch downloads. The retry
    logic below uses exponential backoff up to 3 attempts.
  - TWOSIDES (~1.9 GB uncompressed) is stored on Zenodo with a redirect.
    urllib follows redirects automatically, but the Content-Length header
    may report 0 (chunked transfer) — so we use streaming download.
  - DrugBank requires a free academic account; its download URL contains a
    time-limited token. We do NOT automate it — see MANUAL_STEPS below.

Usage:
    python etl/download_raw.py                        # download all
    python etl/download_raw.py --source faers         # FAERS only
    python etl/download_raw.py --source rxnorm        # RxNorm only
    python etl/download_raw.py --source twosides      # TWOSIDES only
    python etl/download_raw.py --dry-run              # print URLs, don't download
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
import io
import urllib.request

# Fix Windows console encoding (cp1252 can't print box-drawing chars)
import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── Project root (etl/ is one level down from root) ───────────────────────────
ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"

# ── FAERS quarterly ASCII URLs ─────────────────────────────────────────────────
# Pattern confirmed from https://fis.fda.gov/extensions/FPD-QDE-FAERS/
# Note: FDA uses uppercase Q for 2020–2021 Q3/Q4 and 2023–2026 Q4; lowercase q
# for Q1/Q2/Q3 of most years. We use the exact casing from the page.
# 6 years: 2020Q1 → 2025Q4  (24 quarters = ~1.5 GB compressed)
FAERS_QUARTERS: list[tuple[str, str]] = [
    # (quarter_label, url)
    ("2020Q1", "https://fis.fda.gov/content/Exports/faers_ascii_2020Q1.zip"),
    ("2020Q2", "https://fis.fda.gov/content/Exports/faers_ascii_2020Q2.zip"),
    ("2020Q3", "https://fis.fda.gov/content/Exports/faers_ascii_2020Q3.zip"),
    ("2020Q4", "https://fis.fda.gov/content/Exports/faers_ascii_2020Q4.zip"),
    ("2021Q1", "https://fis.fda.gov/content/Exports/faers_ascii_2021Q1.zip"),
    ("2021Q2", "https://fis.fda.gov/content/Exports/faers_ascii_2021Q2.zip"),
    ("2021Q3", "https://fis.fda.gov/content/Exports/faers_ascii_2021Q3.zip"),
    ("2021Q4", "https://fis.fda.gov/content/Exports/faers_ascii_2021Q4.zip"),
    ("2022Q1", "https://fis.fda.gov/content/Exports/faers_ascii_2022q1.zip"),
    ("2022Q2", "https://fis.fda.gov/content/Exports/faers_ascii_2022q2.zip"),
    ("2022Q3", "https://fis.fda.gov/content/Exports/faers_ascii_2022Q3.zip"),
    ("2022Q4", "https://fis.fda.gov/content/Exports/faers_ascii_2022Q4.zip"),
    ("2023Q1", "https://fis.fda.gov/content/Exports/faers_ascii_2023q1.zip"),
    ("2023Q2", "https://fis.fda.gov/content/Exports/faers_ascii_2023q2.zip"),
    ("2023Q3", "https://fis.fda.gov/content/Exports/faers_ascii_2023Q3.zip"),
    ("2023Q4", "https://fis.fda.gov/content/Exports/faers_ascii_2023Q4.zip"),
    ("2024Q1", "https://fis.fda.gov/content/Exports/faers_ascii_2024q1.zip"),
    ("2024Q2", "https://fis.fda.gov/content/Exports/faers_ascii_2024q2.zip"),
    ("2024Q3", "https://fis.fda.gov/content/Exports/faers_ascii_2024q3.zip"),
    ("2024Q4", "https://fis.fda.gov/content/Exports/faers_ascii_2024Q4.zip"),
    ("2025Q1", "https://fis.fda.gov/content/Exports/faers_ascii_2025q1.zip"),
    ("2025Q2", "https://fis.fda.gov/content/Exports/faers_ascii_2025q2.zip"),
    ("2025Q3", "https://fis.fda.gov/content/Exports/faers_ascii_2025q3.zip"),
    ("2025Q4", "https://fis.fda.gov/content/Exports/faers_ascii_2025Q4.zip"),
]

# ── Reference dataset URLs ─────────────────────────────────────────────────────
# RxNorm Prescribable Content — no UMLS license required (confirmed June 2026)
RXNORM_URL = "https://download.nlm.nih.gov/rxnorm/RxNorm_full_prescribe_06012026.zip"

# TWOSIDES via Tatonetti Lab OSF/Zenodo mirror
# This is the drug-pair to side-effect CSV (~1.9 GB uncompressed).
# If this URL 404s, check: https://github.com/tatonetti-lab/nsides-release
TWOSIDES_URL = (
    "https://zenodo.org/records/10975016/files/"
    "TWOSIDES.csv.gz?download=1"
)

# ── MANUAL STEPS (cannot be automated) ─────────────────────────────────────────
MANUAL_STEPS = """
MANUAL DOWNLOAD REQUIRED — DrugBank
=====================================
DrugBank requires a free academic account. Steps:
  1. Go to: https://go.drugbank.com/releases/latest
  2. Create a free academic account (instant, no wait)
  3. Download "DrugBank All Drugs" → drugbank.xml  (~120 MB)
     OR "Approved Drug Links" CSV for a lighter option
  4. Place the file at:  data/raw/drugbank/drugbank_all_drugs.xml
     (or data/raw/drugbank/drugbank_approved.csv for the CSV version)

Why XML over CSV?
  The full XML has drug-drug interaction data we need for ground-truth
  validation. The CSV "Approved" export does NOT include DDI pairs.
"""


# ── Download utilities ─────────────────────────────────────────────────────────

def _stream_download(
    url: str,
    dest: Path,
    label: str,
    max_retries: int = 3,
    chunk_size: int = 1 << 20,  # 1 MB chunks
) -> Path:
    """
    Stream-download `url` to `dest`. Skips if file already exists.
    Uses exponential backoff on transient failures.
    Returns the destination path.
    """
    if dest.exists():
        size_mb = dest.stat().st_size / 1e6
        print(f"  ↳ [{label}] already exists ({size_mb:.1f} MB) — skipping")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")

    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "PolySignal/1.0 (academic research)"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp, \
                 open(tmp, "wb") as fh:
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                while chunk := resp.read(chunk_size):
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(
                            f"  ↳ [{label}] {downloaded/1e6:.1f} MB"
                            f" / {total/1e6:.1f} MB ({pct:.0f}%)",
                            end="\r",
                        )
            tmp.rename(dest)
            size_mb = dest.stat().st_size / 1e6
            print(f"\n  ✓ [{label}] downloaded → {dest.name} ({size_mb:.1f} MB)")
            return dest

        except Exception as exc:
            wait = 2 ** attempt
            print(f"\n  ✗ [{label}] attempt {attempt} failed: {exc}")
            if attempt < max_retries:
                print(f"    retrying in {wait}s …")
                time.sleep(wait)
            else:
                if tmp.exists():
                    tmp.unlink()
                raise RuntimeError(
                    f"Failed to download {url} after {max_retries} attempts"
                ) from exc

    return dest  # unreachable but satisfies type checker


def download_faers(dry_run: bool = False) -> None:
    """Download all 24 FAERS quarters in parallel (max 4 concurrent)."""
    print("\n-- FAERS quarterly ASCII files (2020Q1-2025Q4) -----------------")
    tasks = []
    for label, url in FAERS_QUARTERS:
        dest = DATA_RAW / "faers" / f"faers_ascii_{label}.zip"
        if dry_run:
            print(f"  [DRY RUN] {url} → {dest.relative_to(ROOT)}")
        else:
            tasks.append((label, url, dest))

    if dry_run:
        return

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(_stream_download, url, dest, label): label
            for label, url, dest in tasks
        }
        failed = []
        for future in as_completed(futures):
            label = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"\n  ERROR [{label}]: {exc}")
                failed.append(label)

    if failed:
        print(f"\n  WARNING: {len(failed)} quarters failed: {failed}")
        print("  Run again to retry failed quarters.")
    else:
        print(f"\n  ✓ All {len(FAERS_QUARTERS)} FAERS quarters downloaded.")


def download_rxnorm(dry_run: bool = False) -> None:
    """Download RxNorm Prescribable Content (no license required)."""
    print("\n-- RxNorm Prescribable Content (June 2026) ---------------------")
    dest = DATA_RAW / "rxnorm" / "RxNorm_full_prescribe_06012026.zip"
    if dry_run:
        print(f"  [DRY RUN] {RXNORM_URL} → {dest.relative_to(ROOT)}")
        return
    _stream_download(RXNORM_URL, dest, "RxNorm")


def download_twosides(dry_run: bool = False) -> None:
    """Download TWOSIDES drug-pair side-effect CSV from Zenodo."""
    print("\n-- TWOSIDES (Tatonetti Lab, Zenodo) ----------------------------")
    dest = DATA_RAW / "twosides" / "TWOSIDES.csv.gz"
    if dry_run:
        print(f"  [DRY RUN] {TWOSIDES_URL} → {dest.relative_to(ROOT)}")
        return
    _stream_download(TWOSIDES_URL, dest, "TWOSIDES")


def print_manual_steps() -> None:
    print("\n-- Manual steps required ---------------------------------------")
    print(MANUAL_STEPS)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download PolySignal raw data sources."
    )
    parser.add_argument(
        "--source",
        choices=["faers", "rxnorm", "twosides", "all"],
        default="all",
        help="Which source to download (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without fetching",
    )
    args = parser.parse_args()

    source = args.source
    dry_run = args.dry_run

    if source in ("faers", "all"):
        download_faers(dry_run)
    if source in ("rxnorm", "all"):
        download_rxnorm(dry_run)
    if source in ("twosides", "all"):
        download_twosides(dry_run)

    print_manual_steps()

    if not dry_run:
        print("\n-- Data quality checkpoint -------------------------------------")
        faers_zips = list((DATA_RAW / "faers").glob("*.zip"))
        print(f"  FAERS zips on disk     : {len(faers_zips)} / {len(FAERS_QUARTERS)}")
        rxnorm_zip = DATA_RAW / "rxnorm" / "RxNorm_full_prescribe_06012026.zip"
        print(f"  RxNorm zip on disk     : {'✓' if rxnorm_zip.exists() else '✗ MISSING'}")
        twosides_gz = DATA_RAW / "twosides" / "TWOSIDES.csv.gz"
        print(f"  TWOSIDES gz on disk    : {'✓' if twosides_gz.exists() else '✗ MISSING'}")
        drugbank_xml = DATA_RAW / "drugbank" / "drugbank_all_drugs.xml"
        drugbank_csv = DATA_RAW / "drugbank" / "drugbank_approved.csv"
        db_present = drugbank_xml.exists() or drugbank_csv.exists()
        print(f"  DrugBank file on disk  : {'✓' if db_present else '✗ MANUAL STEP NEEDED'}")


if __name__ == "__main__":
    main()
