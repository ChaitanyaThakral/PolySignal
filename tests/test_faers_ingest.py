"""
tests/test_faers_ingest.py
==========================
Unit tests for the FAERS ETL ingestion layer.
Tests run without real FAERS data by building minimal synthetic fixtures.
"""
from __future__ import annotations
import io
import zipfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from etl.schema import DEMO_SCHEMA, DRUG_SCHEMA, AGE_TO_YEARS, WEIGHT_TO_KG
from etl.faers_ingest import (
    _apply_schema,
    normalise_age,
    normalise_weight,
    deduplicate_demo,
    ingest_quarter,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_demo_df() -> pd.DataFrame:
    return pd.DataFrame({
        "primaryid":        [100, 101, 102, 101],   # 101 duplicated
        "caseid":           [10,  11,  12,  11],
        "fda_dt":           ["20240101"] * 4,
        "rept_cod":         ["EXP"] * 4,
        "gndr_cod":         ["M", "F", "M", "F"],
        "age":              [45.0, 30.0, 60.0, 30.0],
        "age_cod":          ["YR", "YR", "MON", "YR"],
        "wt":               [80.0, 60.0, None, 60.0],
        "wt_cod":           ["KG", "LBS", None, "KG"],
        "occp_cod":         ["MD", "CN", "MD", "CN"],
        "reporter_country": ["US", "US", "GB", "US"],
    })


def _make_drug_df() -> pd.DataFrame:
    return pd.DataFrame({
        "primaryid":  [100, 100, 101, 102],
        "caseid":     [10,  10,  11,  12],
        "drug_seq":   [1,   2,   1,   1],
        "role_cod":   ["PS", "C", "PS", "SS"],   # only PS should survive
        "drugname":   ["ASPIRIN", "Ibuprofen", "metformin", "warfarin"],
        "prod_ai":    ["aspirin", None, "metformin", "warfarin"],
        "route":      ["oral"] * 4,
    })


def _build_fake_zip(tables: dict[str, pd.DataFrame]) -> bytes:
    """Write DataFrames into an in-memory FAERS-style zip ($ delimiter)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, df in tables.items():
            csv_bytes = df.to_csv(sep="$", index=False).encode("latin-1")
            zf.writestr(f"ASCII/{name}24Q4.txt", csv_bytes)
    return buf.getvalue()


# ── Schema application ─────────────────────────────────────────────────────────

class TestApplySchema:
    def test_keeps_only_schema_columns(self):
        df = _make_demo_df()
        df["extra_col"] = "noise"
        result = _apply_schema(df, DEMO_SCHEMA, "2024Q4")
        assert "extra_col" not in result.columns
        for col in DEMO_SCHEMA.keep_cols:
            assert col in result.columns

    def test_adds_quarter_column(self):
        df = _make_demo_df()
        result = _apply_schema(df, DEMO_SCHEMA, "2024Q4")
        assert "quarter" in result.columns
        assert (result["quarter"] == "2024Q4").all()

    def test_missing_column_filled_with_na(self):
        df = _make_demo_df().drop(columns=["occp_cod"])
        result = _apply_schema(df, DEMO_SCHEMA, "2024Q4")
        assert "occp_cod" in result.columns
        assert result["occp_cod"].isna().all()


# ── Normalisation ──────────────────────────────────────────────────────────────

class TestNormaliseAge:
    def test_years_unchanged(self):
        df = pd.DataFrame({"age": [45.0], "age_cod": ["YR"]})
        result = normalise_age(df)
        assert abs(result["age_years"].iloc[0] - 45.0) < 0.01

    def test_months_to_years(self):
        df = pd.DataFrame({"age": [24.0], "age_cod": ["MON"]})
        result = normalise_age(df)
        assert abs(result["age_years"].iloc[0] - 2.0) < 0.1

    def test_out_of_bounds_set_null(self):
        df = pd.DataFrame({"age": [200.0, -5.0], "age_cod": ["YR", "YR"]})
        result = normalise_age(df)
        assert result["age_years"].isna().all()


class TestNormaliseWeight:
    def test_kg_unchanged(self):
        df = pd.DataFrame({"wt": [80.0], "wt_cod": ["KG"]})
        result = normalise_weight(df)
        assert abs(result["weight_kg"].iloc[0] - 80.0) < 0.01

    def test_lbs_to_kg(self):
        df = pd.DataFrame({"wt": [176.0], "wt_cod": ["LBS"]})
        result = normalise_weight(df)
        assert abs(result["weight_kg"].iloc[0] - 79.83) < 0.1


# ── Deduplication ─────────────────────────────────────────────────────────────

class TestDeduplicateDemo:
    def test_keeps_latest_primaryid(self):
        """CASEID 11 has primaryids 101 and 101 — higher wins."""
        df = _make_demo_df()
        result = deduplicate_demo(df)
        # caseid=11 should keep primaryid=101 (the max)
        case11 = result[result["caseid"] == 11]
        assert len(case11) == 1
        assert case11["primaryid"].iloc[0] == 101

    def test_unique_caseid_after_dedup(self):
        df = _make_demo_df()
        result = deduplicate_demo(df)
        assert result["caseid"].nunique() == len(result)


# ── Role filter in ingest_quarter ─────────────────────────────────────────────

class TestIngestQuarter:
    def test_only_ps_drugs_kept(self, tmp_path):
        demo = _make_demo_df()
        drug = _make_drug_df()
        reac = pd.DataFrame({"primaryid": [100, 101], "caseid": [10, 11],
                              "pt": ["Headache", "Nausea"], "drug_rec_act": [None, None]})
        outc = pd.DataFrame({"primaryid": [100], "caseid": [10], "outc_cod": ["HO"]})

        zip_bytes = _build_fake_zip({"DEMO": demo, "DRUG": drug,
                                     "REAC": reac, "OUTC": outc})
        zip_path = tmp_path / "faers_ascii_2024Q4.zip"
        zip_path.write_bytes(zip_bytes)

        result = ingest_quarter(zip_path)
        assert "DRUG" in result
        drug_out = result["DRUG"]
        # Only ROLE_COD == 'PS' rows should remain
        assert (drug_out["role_cod"] == "PS").all()
        assert len(drug_out) == 2   # rows 0 and 2 from drug df

    def test_drug_names_lowercased(self, tmp_path):
        demo = _make_demo_df()
        drug = _make_drug_df()
        reac = pd.DataFrame({"primaryid": [100], "caseid": [10],
                              "pt": ["Headache"], "drug_rec_act": [None]})
        outc = pd.DataFrame({"primaryid": [100], "caseid": [10], "outc_cod": ["HO"]})

        zip_bytes = _build_fake_zip({"DEMO": demo, "DRUG": drug,
                                     "REAC": reac, "OUTC": outc})
        zip_path = tmp_path / "faers_ascii_2024Q4.zip"
        zip_path.write_bytes(zip_bytes)

        result = ingest_quarter(zip_path)
        drug_out = result["DRUG"]
        assert (drug_out["drugname_lower"] == drug_out["drugname_lower"].str.lower()).all()
