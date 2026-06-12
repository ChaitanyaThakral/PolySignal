# -*- coding: utf-8 -*-
"""
etl/rxnorm_loader.py
=====================
Loads RxNorm Prescribable Content RRF files into in-memory lookup structures.

ARCHITECTURE DECISIONS
======================
1. WHY TWO SEPARATE DICTS (str_to_rxcui and rxcui_to_ingredient)?
   - str_to_rxcui: maps any drug name string -> RXCUI (for lookup)
   - rxcui_to_ingredient: maps any RXCUI -> canonical ingredient RXCUI
   Separating them lets us do: FAERS_name -> RXCUI -> ingredient_RXCUI
   in two O(1) lookups, without loading the full RxNorm graph.

2. WHY NOT LOAD ALL RRF FILES?
   We only need RXNCONSO.RRF (names/concepts). RXNREL.RRF (relationships)
   is 2 GB uncompressed and we don't need the full graph -- the ingredient
   hierarchy we care about is already captured by filtering to TTY='IN'.

3. WHY FILTER TO ENG + RXNORM + MMSL SOURCES?
   - RXNORM source has the canonical ingredient names (TTY=IN)
   - MMSL (Multum drug database) has brand names that match FAERS closely
   - Other sources (SNOMEDCT, MSH) add noise without matching improvement

4. MEMORY FOOTPRINT:
   Full RXNCONSO ~900k rows. After filtering ENG + useful SABs:
   ~400k rows. As two dicts of str->str: ~80 MB RAM. Acceptable.

5. WHAT WOULD BREAK AT SCALE:
   For >10M unique drug name lookups, replace dict with a Redis hash or
   DuckDB in-process DB for memory-mapped access.
"""
from __future__ import annotations

import logging
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

log = logging.getLogger(__name__)

ROOT      = Path(__file__).parent.parent
RXNORM_ZIP = ROOT / "data" / "raw" / "rxnorm" / "RxNorm_full_prescribe_06012026.zip"

# RRF column names for RXNCONSO (fixed format, no header in file)
RXNCONSO_COLS = [
    "RXCUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF",
    "RXAUI", "SAUI", "SCUI", "SDUI", "SAB", "TTY", "CODE",
    "STR", "SRL", "SUPPRESS", "CVF", "EMPTY"
]

# Term types we load (other TTYs add noise for our use case)
USEFUL_TTYS = {
    "IN",    # Ingredient -- canonical target we normalize TO
    "PIN",   # Precise Ingredient (salt/ester forms)
    "MIN",   # Multiple Ingredients
    "BN",    # Brand Name -- matches FAERS brand-name entries
    "SY",    # Synonym
    "TMSY",  # Tall Man lettering synonym
    "PSN",   # Prescribable Name
}

# Sources to include (others add rare academic names that waste RAM)
USEFUL_SABS = {
    "RXNORM",     # The canonical RxNorm source
    "MMSL",       # Multum -- matches many FAERS brand names
}


def load_rxnconso(zip_path: Path = RXNORM_ZIP) -> pd.DataFrame:
    """
    Load RXNCONSO.RRF from the RxNorm zip file.
    Returns a filtered DataFrame with columns: RXCUI, SAB, TTY, STR (lowercase).
    """
    if not zip_path.exists():
        raise FileNotFoundError(
            f"RxNorm zip not found: {zip_path}\n"
            "Run: python etl/download_raw.py --source rxnorm"
        )

    log.info("Loading RXNCONSO.RRF from %s ...", zip_path.name)

    with zipfile.ZipFile(zip_path) as zf:
        candidates = [n for n in zf.namelist()
                      if "RXNCONSO" in n.upper() and n.upper().endswith(".RRF")]
        if not candidates:
            raise FileNotFoundError(
                f"RXNCONSO.RRF not found inside {zip_path.name}. "
                f"Contents: {zf.namelist()[:10]}"
            )
        fname = candidates[0]
        log.info("Reading %s ...", fname)
        raw = zf.read(fname)

    df = pd.read_csv(
        BytesIO(raw),
        sep="|",
        header=None,
        names=RXNCONSO_COLS,
        encoding="utf-8",
        low_memory=False,
        on_bad_lines="warn",
        usecols=["RXCUI", "LAT", "SAB", "TTY", "STR", "SUPPRESS"],
    )

    log.info("RXNCONSO raw rows: %d", len(df))

    # Filter: English only, active (not suppressed), useful sources
    df = df[
        (df["LAT"] == "ENG") &
        (df["SUPPRESS"].isin(["N", "O"])) &  # O = obsolete but still matchable
        (df["SAB"].isin(USEFUL_SABS)) &
        (df["TTY"].isin(USEFUL_TTYS))
    ].copy()

    df["STR_lower"] = df["STR"].str.lower().str.strip()
    log.info("RXNCONSO filtered rows: %d", len(df))

    return df


def build_lookup_tables(
    rxnconso: Optional[pd.DataFrame] = None,
    zip_path: Path = RXNORM_ZIP,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Build two lookup dicts from RXNCONSO:

    str_to_rxcui:
        Maps any lowercase drug name string -> RXCUI.
        Covers brand names, generics, synonyms, tall-man names.
        If a name maps to multiple RXCUIs (rare), prefer IN > BN > others.

    rxcui_to_ingredient:
        Maps any RXCUI -> its ingredient-level RXCUI (TTY=IN).
        For RXCUIs that ARE already an ingredient, maps to itself.
        For brand-name RXCUIs, maps to the ingredient RXCUI.
        Used as the second step: name -> rxcui -> ingredient_rxcui.

    Returns: (str_to_rxcui, rxcui_to_ingredient)
    """
    if rxnconso is None:
        rxnconso = load_rxnconso(zip_path)

    # Build str_to_rxcui -- prefer IN terms over BN if collision
    TTY_PRIORITY = {"IN": 0, "PIN": 1, "MIN": 2, "PSN": 3, "SY": 4, "BN": 5, "TMSY": 6}

    # Sort so higher-priority TTYs appear last (we want last-write wins for IN)
    sorted_df = rxnconso.sort_values(
        "TTY",
        key=lambda s: s.map(lambda x: TTY_PRIORITY.get(x, 99)),
        ascending=False,  # low priority first -> high priority overwrites
    )
    str_to_rxcui: Dict[str, str] = dict(
        zip(sorted_df["STR_lower"], sorted_df["RXCUI"].astype(str))
    )
    log.info("str_to_rxcui: %d entries", len(str_to_rxcui))

    # Build rxcui_to_ingredient
    # Ingredients map to themselves; everything else maps to the ingredient
    # that shares the same STR (for MMSL brand names, look up ingredient)
    ingredient_df = rxnconso[rxnconso["TTY"] == "IN"][["RXCUI", "STR_lower"]].copy()
    ingredient_df = ingredient_df.drop_duplicates("RXCUI")

    # All RXCUIs that are ingredients -> map to themselves
    rxcui_to_ingredient: Dict[str, str] = {
        str(row.RXCUI): str(row.RXCUI)
        for row in ingredient_df.itertuples()
    }

    # For non-ingredient RXCUIs: find ingredient by matching the STR
    non_ingredient_df = rxnconso[rxnconso["TTY"] != "IN"][["RXCUI", "STR_lower"]].copy()
    ing_str_to_rxcui = dict(zip(ingredient_df["STR_lower"], ingredient_df["RXCUI"].astype(str)))

    for row in non_ingredient_df.itertuples():
        rxcui_str = str(row.RXCUI)
        if rxcui_str not in rxcui_to_ingredient:
            # Try to find an ingredient with same STR
            ing = ing_str_to_rxcui.get(row.STR_lower)
            if ing:
                rxcui_to_ingredient[rxcui_str] = ing
            else:
                rxcui_to_ingredient[rxcui_str] = rxcui_str  # fallback: self

    log.info("rxcui_to_ingredient: %d entries", len(rxcui_to_ingredient))

    return str_to_rxcui, rxcui_to_ingredient


# Singleton cache -- expensive to rebuild, so we cache after first load
_CACHE: Optional[Tuple[Dict[str, str], Dict[str, str]]] = None


def get_lookup_tables(
    zip_path: Path = RXNORM_ZIP,
    force_reload: bool = False,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return cached lookup tables, loading from disk if needed."""
    global _CACHE
    if _CACHE is None or force_reload:
        _CACHE = build_lookup_tables(zip_path=zip_path)
    return _CACHE
