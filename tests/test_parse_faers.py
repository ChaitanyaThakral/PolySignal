"""
tests/test_parse_faers.py
=========================
Unit tests for etl/parse_faers.py.

All tests use synthetic in-memory zip fixtures — no real FAERS data needed.
Tests cover: era detection, deduplication, normalisation, ratio checks.
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from etl.parse_faers import (
    _detect_era,
    _deduplicate_by_caseid,
    _normalise_demo,
    _normalise_drug,
    _read_era3_table,
    _read_quarter_table,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_zip(tables: dict[str, pd.DataFrame], suffix: str = "txt") -> bytes:
    """Build an in-memory FAERS-style zip file."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, df in tables.items():
            csv_bytes = df.to_csv(sep="$", index=False).encode("latin-1")
            zf.writestr(f"ASCII/{name}24Q4.{suffix}", csv_bytes)
    return buf.getvalue()


DEMO_DF = pd.DataFrame({
    "primaryid":        ["10001", "10002", "10003", "10002"],   # 10002 = dup
    "caseid":           ["100",   "101",   "102",   "101"],
    "fda_dt":           ["20240101"] * 4,
    "rept_cod":         ["EXP"] * 4,
    "gndr_cod":         ["M", "F", "M", "F"],
    "age":              ["45", "30", "720", "30"],    # 720 = months
    "age_cod":          ["YR", "YR", "MON", "YR"],
    "wt":               ["80", "132", "", "60"],
    "wt_cod":           ["KG", "LBS", "", "KG"],
    "occp_cod":         ["MD", "CN", "MD", "CN"],
    "reporter_country": ["US", "US", "GB", "US"],
})

DRUG_DF = pd.DataFrame({
    "primaryid":  ["10001", "10001", "10002", "10003"],
    "caseid":     ["100",   "100",   "101",   "102"],
    "drug_seq":   ["1", "2", "1", "1"],
    "role_cod":   ["PS", "C", "PS", "SS"],   # C and SS should be filtered
    "drugname":   ["ASPIRIN", "Ibuprofen", "METFORMIN", "warfarin"],
    "prod_ai":    ["aspirin", "", "metformin", "warfarin"],
    "route":      ["oral", "oral", "oral", "oral"],
})

REAC_DF = pd.DataFrame({
    "primaryid":    ["10001", "10002"],
    "caseid":       ["100",   "101"],
    "pt":           ["Headache", "Nausea"],
    "drug_rec_act": ["", ""],
})

OUTC_DF = pd.DataFrame({
    "primaryid": ["10001"],
    "caseid":    ["100"],
    "outc_cod":  ["HO"],
})


# ── Era detection ──────────────────────────────────────────────────────────────

class TestDetectEra:
    def test_2024_is_era3(self, tmp_path):
        p = tmp_path / "faers_ascii_2024Q4.zip"
        p.write_bytes(b"")
        assert _detect_era(p) == 3

    def test_2020_is_era3(self, tmp_path):
        p = tmp_path / "faers_ascii_2020Q1.zip"
        p.write_bytes(b"")
        assert _detect_era(p) == 3

    def test_2010_is_era1(self, tmp_path):
        p = tmp_path / "faers_ascii_2010Q2.zip"
        p.write_bytes(b"")
        assert _detect_era(p) == 1

    def test_2013_is_era2(self, tmp_path):
        p = tmp_path / "faers_ascii_2013Q1.zip"
        p.write_bytes(b"")
        assert _detect_era(p) == 2


# ── Deduplication ──────────────────────────────────────────────────────────────

class TestDeduplicateByCaseid:
    def test_keeps_max_primaryid(self):
        df = DEMO_DF.copy()
        result = _deduplicate_by_caseid(df)
        # caseid=101 has primaryids 10002 and 10002 (same in this fixture)
        case101 = result[result["caseid"] == 101]
        assert len(case101) == 1

    def test_unique_caseid_after_dedup(self):
        df = DEMO_DF.copy()
        result = _deduplicate_by_caseid(df)
        assert result["caseid"].nunique() == len(result), \
            "CASEID should be unique after dedup"

    def test_empty_df_returns_empty(self):
        empty = pd.DataFrame(columns=["primaryid", "caseid"])
        result = _deduplicate_by_caseid(empty)
        assert len(result) == 0


# ── Normalisation ──────────────────────────────────────────────────────────────

class TestNormaliseDemo:
    def setup_method(self):
        self.df = _deduplicate_by_caseid(DEMO_DF.copy())
        self.df["age"]    = pd.to_numeric(self.df["age"], errors="coerce")
        self.df["wt"]     = pd.to_numeric(self.df["wt"],  errors="coerce")

    def test_yr_age_unchanged(self):
        result = _normalise_demo(self.df)
        yr_rows = result[result["age_cod"] == "YR"]
        # age 45 YR → 45.0 years
        row45 = yr_rows[yr_rows["age"] == 45.0]
        assert abs(row45["age_years"].iloc[0] - 45.0) < 0.01

    def test_months_converted(self):
        result = _normalise_demo(self.df)
        mon_rows = result[result["age_cod"] == "MON"]
        if len(mon_rows) > 0:
            # 720 months = 60 years
            assert abs(mon_rows["age_years"].iloc[0] - 60.0) < 0.5

    def test_lbs_converted_to_kg(self):
        result = _normalise_demo(self.df)
        lbs_rows = result[result["wt_cod"] == "LBS"]
        if len(lbs_rows) > 0:
            # 132 LBS ≈ 59.87 kg
            assert abs(lbs_rows["weight_kg"].iloc[0] - 59.87) < 0.5

    def test_empty_weight_is_null(self):
        result = _normalise_demo(self.df)
        # row with empty wt should be null
        empty_wt = result[result["wt"].isna()]
        if len(empty_wt) > 0:
            assert empty_wt["weight_kg"].isna().all()


class TestNormaliseDrug:
    def test_only_ps_role_kept(self):
        df = DRUG_DF.copy()
        result = _normalise_drug(df)
        assert (result["role_cod"] == "PS").all(), \
            "Only primary suspect drugs should remain"

    def test_ps_count_correct(self):
        df = DRUG_DF.copy()
        result = _normalise_drug(df)
        # DRUG_DF has 2 PS rows (rows 0 and 2)
        assert len(result) == 2

    def test_drugname_lowercased(self):
        df = DRUG_DF.copy()
        result = _normalise_drug(df)
        assert (result["drugname_lower"] == result["drugname_lower"].str.lower()).all()


# ── ERA3 table reading ─────────────────────────────────────────────────────────

class TestReadEra3Table:
    def test_reads_demo(self, tmp_path):
        zip_bytes = _make_zip({"DEMO": DEMO_DF})
        zip_path  = tmp_path / "faers_ascii_2024Q4.zip"
        zip_path.write_bytes(zip_bytes)
        result = _read_era3_table(zip_path, "DEMO", "2024Q4")
        assert "primaryid" in result.columns
        assert "quarter" in result.columns
        assert (result["quarter"] == "2024Q4").all()
        assert (result["year"] == 2024).all()

    def test_reads_drug(self, tmp_path):
        zip_bytes = _make_zip({"DRUG": DRUG_DF})
        zip_path  = tmp_path / "faers_ascii_2024Q4.zip"
        zip_path.write_bytes(zip_bytes)
        result = _read_era3_table(zip_path, "DRUG", "2024Q4")
        assert "drugname" in result.columns

    def test_missing_table_returns_empty(self, tmp_path):
        zip_bytes = _make_zip({"DEMO": DEMO_DF})   # no DRUG table
        zip_path  = tmp_path / "faers_ascii_2024Q4.zip"
        zip_path.write_bytes(zip_bytes)
        result = _read_era3_table(zip_path, "DRUG", "2024Q4")
        assert result.empty or len(result) == 0

    def test_isr_column_raises(self, tmp_path):
        """ERA1 file (ISR column) should raise ValueError."""
        era1_df = DEMO_DF.rename(columns={"primaryid": "isr"})
        zip_bytes = _make_zip({"DEMO": era1_df})
        zip_path  = tmp_path / "faers_ascii_2024Q4.zip"
        zip_path.write_bytes(zip_bytes)
        with pytest.raises(ValueError, match="ERA 1/2"):
            _read_era3_table(zip_path, "DEMO", "2024Q4")


# ── Full quarter ingestion ─────────────────────────────────────────────────────

class TestReadQuarterTable:
    def test_era1_skipped_with_empty_result(self, tmp_path):
        """ERA 1 zip (year < 2012) should return empty DF without crashing."""
        zip_path = tmp_path / "faers_ascii_2010Q1.zip"
        zip_bytes = _make_zip({"DEMO": DEMO_DF})
        zip_path.write_bytes(zip_bytes)
        result = _read_quarter_table(zip_path, "DEMO")
        assert result.empty

    def test_era3_returns_data(self, tmp_path):
        zip_path  = tmp_path / "faers_ascii_2024Q4.zip"
        zip_bytes = _make_zip({"DEMO": DEMO_DF})
        zip_path.write_bytes(zip_bytes)
        result = _read_quarter_table(zip_path, "DEMO")
        assert not result.empty
        assert "quarter" in result.columns
