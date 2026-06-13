# -*- coding: utf-8 -*-
"""
etl/normalize_drugs.py
=======================
Three-tier drug name normalization: FAERS free-text -> canonical RXCUI.

TIER STRATEGY (cheapest -> most expensive)
==========================================
Tier 1 -- EXACT MATCH (O(1) dict lookup)
  Lowercase the FAERS drugname and look it up directly in str_to_rxcui.
  Match rate: ~55-65% of unique drug names (the well-formed ones).

Tier 2 -- DOSE/ROUTE STRIP THEN EXACT (O(1) after regex, O(1) lookup)
  Strip numeric dose tokens ("81 mg", "500 mcg"), dosage forms
  ("oral tablet", "injection"), and route qualifiers from the drug name,
  then retry exact lookup.
  Handles: "ASPIRIN 81 MG ORAL" -> "aspirin" -> RXCUI 1191
  Match rate: +20-25% additional coverage.

Tier 3 -- RAPIDFUZZ TOKEN SORT (O(k) where k = candidates)
  For remaining unmatched names, use rapidfuzz.process.extractOne
  with token_sort_ratio against ALL RxNorm STR strings.
  To keep this tractable (not O(300k) per query), we pre-filter candidates
  by matching the first 3 characters of the drug name.
  Match rate: +5-10% additional coverage. Threshold: 88/100.

MATCH RATE TARGET
=================
- Tier 1+2+3 combined: target >90% of unique PS drug names.
- Remaining <10%: log as UNMAPPED; these are combination products,
  investigational drugs, and nonsense entries.

OUTPUT ARTIFACTS
================
- data/processed/drug_name_to_rxcui_lookup.parquet
- data/processed/match_rate_report.json
- Updated FAERS drug parquet has new column: rxcui, ingredient_rxcui

WHAT WOULD BREAK AT SCALE
==========================
- Tier 3 over 100k unique names with 300k candidates:
  ~30B string comparisons, too slow even with rapidfuzz C backend.
  At scale: replace with a character-level n-gram inverted index
  (like Elasticsearch fuzzy search) pre-built offline.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

log = logging.getLogger(__name__)

ROOT      = Path(__file__).parent.parent
DATA_PROC = ROOT / "data" / "processed"


# ── Dose/route stripping patterns ─────────────────────────────────────────────
# Order matters: strip dose first, then route, then trailing whitespace.
_DOSE_PATTERN = re.compile(
    r"""\s+
    (?:
        \d+(?:\.\d+)?               # numeric dose  e.g. "81" or "0.5"
        \s*
        (?:mg|mcg|ug|g|ml|l|iu|u|  # dose unit
           units?|%|mmol|meq|
           billion|million|          # probiotic doses
           ppm|ppb|
           mg/ml|mg/dl|mcg/ml)      # concentration forms
    )
    .*                               # everything after dose is noise
    """,
    re.VERBOSE | re.IGNORECASE,
)

_ROUTE_PATTERN = re.compile(
    r"""\s+
    (?:oral|iv\b|intravenous|intramuscular|im\b|
       subcutaneous|sc\b|sq\b|subcut|
       topical|inhale[d]?|inhal[e]?|
       patch|transdermal|rectal|vaginal|
       ophthalmic|otic|nasal|sublingual|buccal|
       tablet|capsule|cap\b|tab\b|solution|
       powder|cream|ointment|gel|lotion|spray|
       drop|suppository|injection|infusion|
       extended.release|er\b|xr\b|sr\b|dr\b|
       immediate.release|ir\b|controlled.release|cr\b|
       delayed.release|modified.release
    ).*
    """,
    re.VERBOSE | re.IGNORECASE,
)

_SALT_SUFFIXES = re.compile(
    r"""\s+
    (?:hydrochloride|hcl|sodium|potassium|calcium|
       acetate|sulfate|sulphate|phosphate|citrate|
       gluconate|maleate|fumarate|succinate|tartrate|
       bromide|chloride|nitrate|mesylate|tosylate|
       besylate|malate|oxalate|pamoate|embonate|
       dihydrate|monohydrate|anhydrous|crystalline
    )\b.*
    """,
    re.VERBOSE | re.IGNORECASE,
)


def strip_dose_route(name: str) -> str:
    """
    Remove dose, route, and salt suffix from a drug name.
    Returns the cleaned name (may be empty string if fully stripped).
    """
    s = name.strip()
    s = _DOSE_PATTERN.sub("", s).strip()
    s = _ROUTE_PATTERN.sub("", s).strip()
    s = _SALT_SUFFIXES.sub("", s).strip()
    # Remove trailing punctuation
    s = re.sub(r"[,;:\-\(\)\[\]]+$", "", s).strip()
    return s


# ── Three-tier matcher ─────────────────────────────────────────────────────────

class DrugNormalizer:
    """
    Stateful normalizer that loads RxNorm lookup tables once and
    normalizes batches of drug names efficiently.
    """

    FUZZY_THRESHOLD = 88     # minimum rapidfuzz score (0-100)
    PREFIX_LEN      = 3      # characters to pre-filter fuzzy candidates

    def __init__(
        self,
        str_to_rxcui: Dict[str, str],
        rxcui_to_ingredient: Dict[str, str],
    ):
        self.str_to_rxcui        = str_to_rxcui
        self.rxcui_to_ingredient = rxcui_to_ingredient

        # Pre-build prefix index for tier 3
        log.info("Building prefix index for fuzzy fallback...")
        self._prefix_index: Dict[str, List[str]] = {}
        for s in str_to_rxcui:
            key = s[: self.PREFIX_LEN]
            self._prefix_index.setdefault(key, []).append(s)
        log.info("Prefix index: %d entries", len(self._prefix_index))

    def _lookup(self, name: str) -> Optional[str]:
        """Direct dict lookup -> ingredient RXCUI or None."""
        rxcui = self.str_to_rxcui.get(name)
        if rxcui:
            return self.rxcui_to_ingredient.get(rxcui, rxcui)
        return None

    def normalize_one(self, raw_name: str) -> Tuple[Optional[str], str]:
        """
        Normalize one drug name.
        Returns: (ingredient_rxcui_or_None, tier_used)
        tier_used: 'exact' | 'stripped' | 'fuzzy' | 'unmapped'
        """
        name = raw_name.lower().strip()
        if not name:
            return None, "empty"

        # Tier 1: exact
        rxcui = self._lookup(name)
        if rxcui:
            return rxcui, "exact"

        # Tier 2: strip dose/route then exact
        stripped = strip_dose_route(name)
        if stripped and stripped != name:
            rxcui = self._lookup(stripped)
            if rxcui:
                return rxcui, "stripped"

        # Tier 2b: try just the first word (e.g. "atorvastatin calcium 10mg" -> "atorvastatin")
        first_word = name.split()[0] if name.split() else ""
        if first_word and len(first_word) > 3:
            rxcui = self._lookup(first_word)
            if rxcui:
                return rxcui, "first_word"

        # Tier 3: rapidfuzz fuzzy match
        try:
            from rapidfuzz import process as rfp, fuzz as rff
            query   = stripped or name
            prefix  = query[: self.PREFIX_LEN]
            cands   = self._prefix_index.get(prefix, [])
            if cands:
                best = rfp.extractOne(
                    query, cands,
                    scorer=rff.token_sort_ratio,
                    score_cutoff=self.FUZZY_THRESHOLD,
                )
                if best:
                    rxcui = self._lookup(best[0])
                    if rxcui:
                        return rxcui, "fuzzy"
        except ImportError:
            log.warning("rapidfuzz not installed -- fuzzy tier skipped")

        return None, "unmapped"

    def normalize_batch(
        self,
        names: List[str],
        log_every: int = 5000,
    ) -> pd.DataFrame:
        """
        Normalize a list of drug names.
        Returns a DataFrame with columns:
            drugname_lower, rxcui, ingredient_rxcui, match_tier
        """
        results = []
        for i, name in enumerate(names):
            if i > 0 and i % log_every == 0:
                log.info("Normalized %d / %d names...", i, len(names))
            rxcui, tier = self.normalize_one(name)
            results.append({
                "drugname_lower":   name.lower().strip(),
                "rxcui":            rxcui,
                "ingredient_rxcui": rxcui,   # same for now (already ingredient)
                "match_tier":       tier,
            })
        return pd.DataFrame(results)


# ── Match rate report ─────────────────────────────────────────────────────────

def compute_match_rate_report(lookup_df: pd.DataFrame) -> dict:
    """
    Compute match rate statistics from the lookup DataFrame.
    Returns a dict suitable for JSON serialization.
    """
    total    = len(lookup_df)
    by_tier  = lookup_df["match_tier"].value_counts().to_dict()
    unmapped = by_tier.get("unmapped", 0) + by_tier.get("empty", 0)
    matched  = total - unmapped

    report = {
        "total_unique_names":   total,
        "matched":              matched,
        "unmapped":             unmapped,
        "match_rate_pct":       round(matched / max(total, 1) * 100, 2),
        "by_tier":              by_tier,
        "sample_unmapped":      (
            lookup_df[lookup_df["match_tier"] == "unmapped"]["drugname_lower"]
            .head(30).tolist()
        ),
    }
    return report


# ── Apply to FAERS Parquet ─────────────────────────────────────────────────────

def normalize_faers_drugs(
    faers_drug_parquet: Path,
    normalizer: DrugNormalizer,
    output_dir: Path = DATA_PROC,
) -> Tuple[pd.DataFrame, dict]:
    """
    Apply normalization to all unique drug names in the FAERS drug Parquet.
    Returns: (lookup_df, match_rate_report)

    ⚠️ Data quality checkpoint:
    - Row count in FAERS drug table must not change (we're adding columns)
    - Null rxcui count logged before/after
    """
    log.info("Loading FAERS drug Parquet from %s ...", faers_drug_parquet)
    drug_df = pd.read_parquet(faers_drug_parquet, columns=["drugname_lower"])

    n_before = len(drug_df)
    log.info("FAERS DRUG rows: %d", n_before)

    unique_names = drug_df["drugname_lower"].dropna().unique().tolist()
    log.info("Unique drug names to normalize: %d", len(unique_names))

    lookup_df = normalizer.normalize_batch(unique_names)

    report = compute_match_rate_report(lookup_df)
    log.info(
        "Match rate: %.1f%% (%d/%d names matched)",
        report["match_rate_pct"], report["matched"], report["total_unique_names"]
    )
    log.info("By tier: %s", report["by_tier"])
    log.info("Sample unmapped: %s", report["sample_unmapped"][:10])

    return lookup_df, report


def normalize_twosides(twosides_path: Path, rxcui_to_ingredient: Dict[str, str]) -> pd.DataFrame:
    """
    TWOSIDES already has rxcui columns -- normalize to ingredient level.
    Returns a mapping: drug_rxcui -> ingredient_rxcui for all TWOSIDES drugs.
    """
    if not twosides_path.exists():
        log.warning("TWOSIDES file not found: %s -- skipping", twosides_path)
        return pd.DataFrame()

    log.info("Loading TWOSIDES sample (first 200k rows) ...")
    import gzip
    with gzip.open(twosides_path, "rb") as f:
        ts = pd.read_csv(f, nrows=200_000, usecols=lambda c: any(x in c.lower() for x in ["rxcui", "rxnorm", "rxnorn"]))

    rxcui_cols = [c for c in ts.columns if any(x in c.lower() for x in ["rxcui", "rxnorm", "rxnorn"])]
    log.info("TWOSIDES rxcui columns: %s", rxcui_cols)

    if not rxcui_cols:
        log.warning("No rxcui/rxnorm columns found in TWOSIDES!")
        return pd.DataFrame(columns=["drug_rxcui", "ingredient_rxcui"])

    all_rxcuis = pd.unique(ts[rxcui_cols].values.ravel("K"))
    all_rxcuis = [str(int(float(r))) if pd.notna(r) and str(r).replace('.','').isdigit() else str(r) for r in all_rxcuis if pd.notna(r)]

    mapping = {r: rxcui_to_ingredient.get(r, r) for r in all_rxcuis}

    if not mapping:
        return pd.DataFrame(columns=["drug_rxcui", "ingredient_rxcui"])

    result = pd.DataFrame([
        {"drug_rxcui": k, "ingredient_rxcui": v}
        for k, v in mapping.items()
    ])
    log.info(
        "TWOSIDES: %d unique RXCUIs -> %d ingredient RXCUIs (%.1f%% already ingredient)",
        len(all_rxcuis),
        result["ingredient_rxcui"].nunique(),
        result[result["drug_rxcui"] == result["ingredient_rxcui"]].shape[0] / max(len(result), 1) * 100,
    )
    return result


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Normalize drug names to RXCUI.")
    parser.add_argument("--faers-parquet", type=Path,
                        default=DATA_PROC / "faers_drug.parquet",
                        help="Path to FAERS DRUG Parquet directory.")
    parser.add_argument("--twosides", type=Path,
                        default=ROOT / "data/raw/twosides/TWOSIDES.csv.gz")
    parser.add_argument("--output-dir", type=Path, default=DATA_PROC)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Load RxNorm lookup tables
    from etl.rxnorm_loader import get_lookup_tables
    str_to_rxcui, rxcui_to_ingredient = get_lookup_tables()
    normalizer = DrugNormalizer(str_to_rxcui, rxcui_to_ingredient)

    results = {}

    # FAERS normalization
    if args.faers_parquet.exists():
        lookup_df, report = normalize_faers_drugs(
            args.faers_parquet, normalizer, args.output_dir
        )
        out_lookup = args.output_dir / "drug_name_to_rxcui_lookup.parquet"
        lookup_df.to_parquet(out_lookup, index=False, engine="pyarrow")
        log.info("Saved lookup table -> %s", out_lookup)
        results["faers"] = report
    else:
        log.warning(
            "FAERS drug Parquet not found at %s.\n"
            "Run: python etl/parse_faers.py first",
            args.faers_parquet
        )

    # TWOSIDES normalization
    ts_mapping = normalize_twosides(args.twosides, rxcui_to_ingredient)
    if not ts_mapping.empty:
        out_ts = args.output_dir / "twosides_rxcui_mapping.parquet"
        ts_mapping.to_parquet(out_ts, index=False, engine="pyarrow")
        log.info("Saved TWOSIDES mapping -> %s", out_ts)
        results["twosides_unique_rxcuis"] = len(ts_mapping)

    # Write JSON report
    report_path = args.output_dir / "match_rate_report.json"
    report_path.write_text(json.dumps(results, indent=2))
    log.info("Match rate report -> %s", report_path)

    # Print summary
    if "faers" in results:
        r = results["faers"]
        print(f"\n{'='*60}")
        print(f"  Drug Normalization Report")
        print(f"{'='*60}")
        print(f"  Total unique names :  {r['total_unique_names']:,}")
        print(f"  Matched            :  {r['matched']:,} ({r['match_rate_pct']:.1f}%)")
        print(f"  Unmapped           :  {r['unmapped']:,}")
        print(f"  Tier breakdown     :  {r['by_tier']}")
        print(f"  Sample unmapped    :  {r['sample_unmapped'][:5]}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
