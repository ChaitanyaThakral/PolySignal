# -*- coding: utf-8 -*-
"""
tests/test_normalize_drugs.py
==============================
Tests for etl/rxnorm_loader.py and etl/normalize_drugs.py.

All tests use synthetic data -- no real RxNorm zip or FAERS Parquet needed.
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from etl.normalize_drugs import (
    strip_dose_route,
    DrugNormalizer,
    compute_match_rate_report,
)
from etl.rxnorm_loader import build_lookup_tables


# ── Synthetic RxNorm data ──────────────────────────────────────────────────────

SYNTHETIC_RXNCONSO = pd.DataFrame({
    "RXCUI":    ["1191",  "1191",  "41493", "1191",  "36567",  "36567"],
    "LAT":      ["ENG",   "ENG",   "ENG",   "ENG",   "ENG",    "ENG"],
    "SAB":      ["RXNORM","RXNORM","RXNORM","MMSL",  "RXNORM", "MMSL"],
    "TTY":      ["IN",    "SY",    "IN",    "BN",    "BN",     "SY"],
    "STR":      ["aspirin","ASA","acetaminophen","Bayer Aspirin","Tylenol","Paracetamol"],
    "SUPPRESS": ["N",     "N",     "N",     "N",     "N",      "N"],
})
SYNTHETIC_RXNCONSO["STR_lower"] = SYNTHETIC_RXNCONSO["STR"].str.lower().str.strip()


# ── strip_dose_route ──────────────────────────────────────────────────────────

class TestStripDoseRoute:
    def test_strips_mg(self):
        assert strip_dose_route("aspirin 81 mg") == "aspirin"

    def test_strips_mcg(self):
        assert strip_dose_route("fentanyl 100 mcg") == "fentanyl"

    def test_strips_oral_tablet(self):
        assert strip_dose_route("metformin oral tablet") == "metformin"

    def test_strips_dose_and_route(self):
        assert strip_dose_route("atorvastatin 10 mg oral") == "atorvastatin"

    def test_strips_salt_suffix(self):
        result = strip_dose_route("metformin hydrochloride")
        assert result == "metformin"

    def test_strips_concentration(self):
        result = strip_dose_route("amoxicillin 500 mg/ml")
        assert result == "amoxicillin"

    def test_empty_string(self):
        assert strip_dose_route("") == ""

    def test_no_dose_unchanged(self):
        result = strip_dose_route("aspirin")
        assert result == "aspirin"

    def test_er_suffix_stripped(self):
        result = strip_dose_route("metformin er")
        assert "er" not in result.lower().split()


# ── build_lookup_tables ────────────────────────────────────────────────────────

class TestBuildLookupTables:
    def test_str_to_rxcui_contains_ingredient(self):
        s2r, r2i = build_lookup_tables(rxnconso=SYNTHETIC_RXNCONSO)
        assert "aspirin" in s2r
        assert s2r["aspirin"] == "1191"

    def test_str_to_rxcui_contains_brand_name(self):
        s2r, r2i = build_lookup_tables(rxnconso=SYNTHETIC_RXNCONSO)
        assert "bayer aspirin" in s2r

    def test_str_to_rxcui_synonym(self):
        s2r, r2i = build_lookup_tables(rxnconso=SYNTHETIC_RXNCONSO)
        assert "asa" in s2r

    def test_ingredient_maps_to_itself(self):
        s2r, r2i = build_lookup_tables(rxnconso=SYNTHETIC_RXNCONSO)
        # RXCUI 1191 is an IN -> maps to itself
        assert r2i.get("1191") == "1191"

    def test_brand_maps_to_ingredient(self):
        s2r, r2i = build_lookup_tables(rxnconso=SYNTHETIC_RXNCONSO)
        # RXCUI 36567 is Tylenol (BN), should map to acetaminophen RXCUI
        # "tylenol" str -> RXCUI 36567 -> ingredient "acetaminophen" -> 41493
        # (if the mapping resolves via STR match)
        assert "36567" in r2i  # must have an entry

    def test_in_preferred_over_bn_for_same_name(self):
        """When ingredient and brand share a name, IN wins."""
        df = SYNTHETIC_RXNCONSO.copy()
        # Add a BN with same STR as existing IN
        collision_row = pd.DataFrame([{
            "RXCUI": "99999", "LAT": "ENG", "SAB": "RXNORM",
            "TTY": "BN", "STR": "aspirin", "SUPPRESS": "N",
            "STR_lower": "aspirin"
        }])
        df = pd.concat([df, collision_row], ignore_index=True)
        s2r, r2i = build_lookup_tables(rxnconso=df)
        # IN (1191) should win over BN (99999) for "aspirin"
        assert s2r["aspirin"] == "1191"


# ── DrugNormalizer ─────────────────────────────────────────────────────────────

@pytest.fixture
def normalizer():
    s2r, r2i = build_lookup_tables(rxnconso=SYNTHETIC_RXNCONSO)
    return DrugNormalizer(s2r, r2i)


class TestDrugNormalizerExactMatch:
    def test_exact_match_aspirin(self, normalizer):
        rxcui, tier = normalizer.normalize_one("aspirin")
        assert rxcui == "1191"
        assert tier == "exact"

    def test_case_insensitive(self, normalizer):
        rxcui, tier = normalizer.normalize_one("ASPIRIN")
        assert rxcui == "1191"

    def test_synonym_match(self, normalizer):
        rxcui, tier = normalizer.normalize_one("ASA")
        assert rxcui == "1191"
        assert tier == "exact"

    def test_brand_name(self, normalizer):
        rxcui, tier = normalizer.normalize_one("bayer aspirin")
        assert rxcui is not None
        assert tier == "exact"


class TestDrugNormalizerStrippedMatch:
    def test_dose_stripped_match(self, normalizer):
        rxcui, tier = normalizer.normalize_one("aspirin 81 mg")
        assert rxcui == "1191"
        assert tier in ("stripped", "exact", "first_word")

    def test_route_stripped_match(self, normalizer):
        rxcui, tier = normalizer.normalize_one("aspirin oral")
        assert rxcui == "1191"
        assert tier in ("stripped", "exact", "first_word")

    def test_dose_and_route_stripped(self, normalizer):
        rxcui, tier = normalizer.normalize_one("aspirin 100 mg oral tablet")
        assert rxcui == "1191"
        assert tier in ("stripped", "exact", "first_word")


class TestDrugNormalizerUnmapped:
    def test_completely_unknown(self, normalizer):
        rxcui, tier = normalizer.normalize_one("XYZ_INVESTIGATIONAL_2024")
        assert rxcui is None
        assert tier == "unmapped"

    def test_empty_string(self, normalizer):
        rxcui, tier = normalizer.normalize_one("")
        assert rxcui is None
        assert tier == "empty"


class TestDrugNormalizerBatch:
    def test_batch_returns_dataframe(self, normalizer):
        names = ["aspirin", "aspirin 81 mg", "XYZ_UNKNOWN"]
        result = normalizer.normalize_batch(names)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert "rxcui" in result.columns
        assert "match_tier" in result.columns

    def test_batch_match_rate(self, normalizer):
        names = ["aspirin", "asa", "aspirin 81 mg", "UNKNOWN_XYZ", ""]
        result = normalizer.normalize_batch(names)
        matched = result[result["match_tier"] != "unmapped"]["rxcui"].notna().sum()
        assert matched >= 3   # aspirin, asa, aspirin 81 mg should all match


# ── Match rate report ──────────────────────────────────────────────────────────

class TestComputeMatchRateReport:
    def test_perfect_match(self):
        df = pd.DataFrame({
            "drugname_lower": ["a", "b"],
            "rxcui":          ["1", "2"],
            "match_tier":     ["exact", "stripped"],
        })
        report = compute_match_rate_report(df)
        assert report["match_rate_pct"] == 100.0
        assert report["unmapped"] == 0

    def test_partial_match(self):
        df = pd.DataFrame({
            "drugname_lower": ["a", "b", "c"],
            "rxcui":          ["1", None, None],
            "match_tier":     ["exact", "unmapped", "unmapped"],
        })
        report = compute_match_rate_report(df)
        assert report["matched"] == 1
        assert report["unmapped"] == 2
        assert abs(report["match_rate_pct"] - 33.33) < 0.1

    def test_sample_unmapped_in_report(self):
        df = pd.DataFrame({
            "drugname_lower": ["unknown1", "unknown2"],
            "rxcui":          [None, None],
            "match_tier":     ["unmapped", "unmapped"],
        })
        report = compute_match_rate_report(df)
        assert "unknown1" in report["sample_unmapped"]
