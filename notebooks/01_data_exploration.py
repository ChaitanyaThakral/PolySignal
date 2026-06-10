"""
notebooks/01_data_exploration.py
=================================
Day 2 exploration script — inspect the raw structure of every data source.
Can be run as a plain Python script or opened in JupyterLab as a notebook.

Architecture note:
  We load ONE quarter only (2024Q4) for FAERS inspection — not all 24.
  Loading all 24 quarters before Day 4 ETL (which handles deduplication)
  would give misleading row counts because FAERS has ~15% duplicate cases
  across quarters (same case updated in a later quarter). The ETL on Day 4
  resolves this properly; for today we just need to understand the schema.

Data quality checkpoints are embedded throughout — look for ⚠️ markers.
"""

# %% [markdown]
# # PolySignal — Day 2 Data Exploration
#
# **Goal:** Understand the raw structure of each dataset before writing
# any transformation code. Every column we intend to use gets documented.

# %% Imports
import gzip
import zipfile
from io import StringIO
from pathlib import Path

import pandas as pd

ROOT = Path("..").resolve()          # one level up from notebooks/
DATA_RAW = ROOT / "data" / "raw"

pd.set_option("display.max_columns", 50)
pd.set_option("display.max_colwidth", 40)
pd.set_option("display.float_format", "{:.2f}".format)

print("ROOT:", ROOT)
print("DATA_RAW:", DATA_RAW)

# %% [markdown]
# ---
# ## 1. FAERS — 2024Q4 Quarter Inspection

# %% Load FAERS 2024Q4
FAERS_ZIP = DATA_RAW / "faers" / "faers_ascii_2024Q4.zip"

def load_faers_table(zip_path: Path, table_name: str) -> pd.DataFrame:
    """
    Load a named table (DEMO, DRUG, REAC, OUTC) from a FAERS quarterly zip.
    FAERS ASCII files use $ as delimiter and have inconsistent line endings.
    Some quarters use latin-1 encoding for drug name free text.
    """
    with zipfile.ZipFile(zip_path) as zf:
        # File naming within the zip varies: could be ASCII/DEMO24Q4.txt
        # or just DEMO24Q4.txt depending on the quarter
        candidates = [
            n for n in zf.namelist()
            if table_name.upper() in n.upper() and n.endswith(".txt")
        ]
        if not candidates:
            raise FileNotFoundError(
                f"No {table_name} file found in {zip_path.name}. "
                f"Available: {zf.namelist()}"
            )
        filename = candidates[0]
        print(f"  Loading: {filename}")
        with zf.open(filename) as f:
            return pd.read_csv(
                f,
                sep="$",
                encoding="latin-1",
                low_memory=False,
                on_bad_lines="warn",   # don't crash on malformed rows
            )

if not FAERS_ZIP.exists():
    print(f"⚠️  {FAERS_ZIP} not found — run `python etl/download_raw.py --source faers` first")
else:
    print("Loading FAERS 2024Q4 tables…")
    demo = load_faers_table(FAERS_ZIP, "DEMO")
    drug = load_faers_table(FAERS_ZIP, "DRUG")
    reac = load_faers_table(FAERS_ZIP, "REAC")
    outc = load_faers_table(FAERS_ZIP, "OUTC")
    print(f"  DEMO: {len(demo):,} rows × {len(demo.columns)} cols")
    print(f"  DRUG: {len(drug):,} rows × {len(drug.columns)} cols")
    print(f"  REAC: {len(reac):,} rows × {len(reac.columns)} cols")
    print(f"  OUTC: {len(outc):,} rows × {len(outc.columns)} cols")

# %% FAERS — Column inspection
print("\n── DEMO columns and dtypes ──")
print(demo.dtypes)

print("\n── DRUG columns and dtypes ──")
print(drug.dtypes)

print("\n── REAC columns and dtypes ──")
print(reac.dtypes)

print("\n── OUTC columns and dtypes ──")
print(outc.dtypes)

# %% FAERS — Missing value rates
def missing_rate(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "null_count": df.isnull().sum(),
        "null_pct": (df.isnull().mean() * 100).round(2),
    }).sort_values("null_pct", ascending=False)

print("\n── DEMO missing value rates ──")
print(missing_rate(demo))

print("\n── DRUG missing value rates ──")
print(missing_rate(drug))

# %% FAERS — Drug name representation
print("\n── Drug name: are these free text or coded? ──")
print(drug["drugname"].value_counts().head(20))

print("\n── Drug name sample (first 10 unique) ──")
print(drug["drugname"].dropna().unique()[:10])

print("\n── Drug name casing check (10 examples of the same drug different cases) ──")
# Look for a common drug and see how many spelling variants it has
aspirin_variants = drug[
    drug["drugname"].str.contains("aspirin", case=False, na=False)
]["drugname"].value_counts()
print(aspirin_variants.head(20))

# %% FAERS — Role codes in drug table
print("\n── ROLE_COD distribution (what % are primary suspects?) ──")
print(drug["role_cod"].value_counts())
print("""
PS = Primary Suspect (the drug under investigation — we use this)
SS = Secondary Suspect
C  = Concomitant (also being taken, not suspected)
I  = Interacting
""")

# %% FAERS — Outcome code inspection
print("\n── OUTC_COD distribution ──")
print(outc["outc_cod"].value_counts())
print("""
DE = Death
LT = Life-Threatening
HO = Hospitalization — Initial or Prolonged
DS = Disability
CA = Congenital Anomaly
RI = Required Intervention to Prevent Permanent Impairment
OT = Other Serious (Important Medical Event)
""")

# %% FAERS — Data quality checkpoint: PRIMARYID uniqueness
print("\n── ⚠️  Data quality checkpoint: PRIMARYID in DEMO ──")
n_demo = len(demo)
n_unique_pid = demo["primaryid"].nunique()
print(f"  Total DEMO rows       : {n_demo:,}")
print(f"  Unique PRIMARYID      : {n_unique_pid:,}")
if n_demo != n_unique_pid:
    print(f"  ⚠️  {n_demo - n_unique_pid:,} DUPLICATE primaryids in DEMO!")
    print("  This is expected — FAERS uses 'ISR' or 'PRIMARYID' to track case updates.")
    print("  The ETL on Day 4 will keep only the most recent case version (max PRIMARYID per CASEID).")
else:
    print("  ✓ No duplicates in this quarter's DEMO table.")

# %% FAERS — Join DEMO → DRUG → REAC sanity check
print("\n── ⚠️  Join sanity: DEMO ← DRUG ──")
drug_pids = set(drug["primaryid"].dropna().unique())
demo_pids = set(demo["primaryid"].dropna().unique())
orphan_drug_pids = drug_pids - demo_pids
print(f"  DRUG primaryids with no DEMO match: {len(orphan_drug_pids):,}")
if orphan_drug_pids:
    print("  ⚠️  These will be dropped in the ETL inner join — expected (case deletions).")

# %% [markdown]
# ---
# ## 2. TWOSIDES — Drug-pair to side-effect associations

# %% Load TWOSIDES
TWOSIDES_GZ = DATA_RAW / "twosides" / "TWOSIDES.csv.gz"

if not TWOSIDES_GZ.exists():
    print(f"⚠️  {TWOSIDES_GZ} not found — run `python etl/download_raw.py --source twosides` first")
else:
    print("Loading TWOSIDES sample (first 100k rows)…")
    with gzip.open(TWOSIDES_GZ, "rb") as f:
        twosides = pd.read_csv(f, nrows=100_000)

    print(f"  Shape (sample): {twosides.shape}")
    print("\n── Column names and dtypes ──")
    print(twosides.dtypes)
    print("\n── First 5 rows ──")
    print(twosides.head())

    print("\n── ⚠️  Drug identifier check ──")
    print("  TWOSIDES uses RxNorm CUI (rxcui) for drug identifiers.")
    print("  FAERS uses free-text drug names.")
    print("  These will NOT join directly — RxNorm normalization (Day 4) bridges them.")
    print(f"\n  Sample drug1_rxcui values: {twosides.iloc[:5, 0].tolist()}")
    print(f"  Sample drug1_concept_name: {twosides.iloc[:5, 1].tolist()}")

    print("\n── PRR distribution in TWOSIDES ──")
    if "PRR" in twosides.columns:
        print(twosides["PRR"].describe())

# %% [markdown]
# ---
# ## 3. DrugBank — Known drug interactions (ground truth)

# %% Load DrugBank (XML or CSV)
DRUGBANK_XML = DATA_RAW / "drugbank" / "drugbank_all_drugs.xml"
DRUGBANK_CSV = DATA_RAW / "drugbank" / "drugbank_approved.csv"

if DRUGBANK_XML.exists():
    print("DrugBank XML found — parsing subset to check identifier columns…")
    # Parse just the first 100 drug entries to check structure
    # Full XML parse is done in the Day 4 ETL
    import xml.etree.ElementTree as ET
    tree = ET.parse(DRUGBANK_XML)
    root = tree.getroot()
    ns = {"db": "http://www.drugbank.ca"}

    sample_drugs = []
    for drug_el in list(root.findall("db:drug", ns))[:20]:
        name = drug_el.findtext("db:name", namespaces=ns)
        drugbank_id = drug_el.findtext("db:drugbank-id[@primary='true']", namespaces=ns)
        # RxCUI cross-reference
        rxcui = None
        for ext in drug_el.findall(".//db:external-identifier", ns):
            if ext.findtext("db:resource", namespaces=ns) == "RxCUI":
                rxcui = ext.findtext("db:identifier", namespaces=ns)
                break
        sample_drugs.append({"drugbank_id": drugbank_id, "name": name, "rxcui": rxcui})

    db_df = pd.DataFrame(sample_drugs)
    print(db_df)
    print("\n── ⚠️  Identifier check ──")
    print("  DrugBank uses its own 'DB00001' IDs + cross-references to RxCUI.")
    print("  The RxCUI column bridges to FAERS via RxNorm — same normalization needed.")

elif DRUGBANK_CSV.exists():
    print("DrugBank CSV found…")
    db_csv = pd.read_csv(DRUGBANK_CSV)
    print(f"  Shape: {db_csv.shape}")
    print(db_csv.dtypes)
    print(db_csv.head())

else:
    print("⚠️  DrugBank file not found.")
    print("  Complete the manual steps in etl/download_raw.py (MANUAL_STEPS section).")

# %% [markdown]
# ---
# ## 4. RxNorm — RXNCONSO.RRF inspection

# %% Load RxNorm
RXNORM_ZIP = DATA_RAW / "rxnorm" / "RxNorm_full_prescribe_06012026.zip"

if not RXNORM_ZIP.exists():
    print(f"⚠️  {RXNORM_ZIP} not found — run `python etl/download_raw.py --source rxnorm` first")
else:
    print("Loading RXNCONSO.RRF from RxNorm zip (first 50k rows)…")
    with zipfile.ZipFile(RXNORM_ZIP) as zf:
        rrf_files = [n for n in zf.namelist() if "RXNCONSO" in n.upper()]
        print(f"  RRF files found: {rrf_files}")
        with zf.open(rrf_files[0]) as f:
            rxnorm = pd.read_csv(
                f,
                sep="|",
                header=None,
                nrows=50_000,
                names=[
                    "RXCUI","LAT","TS","LUI","STT","SUI","ISPREF",
                    "RXAUI","SAUI","SCUI","SDUI","SAB","TTY","CODE",
                    "STR","SRL","SUPPRESS","CVF","extra"
                ],
                encoding="utf-8",
                on_bad_lines="warn",
            )

    print(f"  Shape (sample): {rxnorm.shape}")
    print("\n── Key columns ──")
    print(rxnorm[["RXCUI", "SAB", "TTY", "STR"]].head(20))

    print("\n── Source vocabulary (SAB) distribution ──")
    print(rxnorm["SAB"].value_counts().head(10))
    print("""
    Key SABs for PolySignal:
      RXNORM = base RxNorm concepts (what we want)
      SNOMEDCT_US = SNOMED cross-reference
      MMSL = Multum drug names (often match FAERS brand names)
      MSH = MeSH
    """)

    print("\n── Term types (TTY) for RXNORM source ──")
    print(rxnorm[rxnorm["SAB"] == "RXNORM"]["TTY"].value_counts())
    print("""
    Key TTYs:
      IN  = Ingredient (generic name — what we normalize TO)
      BN  = Brand Name
      PIN = Precise Ingredient
      MIN = Multiple Ingredients
    """)

    print("\n── ⚠️  Normalization preview ──")
    print("  FAERS drugname 'ASPIRIN' → RxCUI lookup → RXCUI 1191")
    aspirin_rows = rxnorm[
        rxnorm["STR"].str.upper().str.contains("ASPIRIN", na=False) &
        (rxnorm["SAB"] == "RXNORM") &
        (rxnorm["TTY"] == "IN")
    ]
    print(aspirin_rows[["RXCUI", "STR", "TTY"]].head(5))

# %% [markdown]
# ---
# ## Summary — Cross-dataset identifier problem

print("""
╔══════════════════════════════════════════════════════════════════╗
║            Drug Identifier Mismatch Summary                     ║
╠══════════════════════════════════════════════════════════════════╣
║  Dataset       │ Drug Identifier     │ Example                 ║
║  FAERS         │ Free text (DRUGNAME)│ "ASPIRIN", "aspirin 81" ║
║  TWOSIDES      │ RxNorm RXCUI        │ 1191                    ║
║  DrugBank      │ DrugBank ID + RXCUI │ DB00945, 1191           ║
║  RxNorm        │ RXCUI (canonical)   │ 1191                    ║
╠══════════════════════════════════════════════════════════════════╣
║  Solution (Day 4): normalize FAERS drugnames → RXCUI via       ║
║  RxNorm API fuzzy string matching (RXNCONSO STR → RXCUI)       ║
╚══════════════════════════════════════════════════════════════════╝
""")
