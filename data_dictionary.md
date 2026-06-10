# PolySignal — Data Dictionary

**Last updated:** Day 2 (June 10, 2026)  
**Purpose:** Documents every column from every raw source that PolySignal intends to use downstream. Columns not listed here are explicitly excluded and will be dropped in the Day 4 ETL.

> [!IMPORTANT]
> The core problem this dictionary documents: **all four datasets use different drug identifiers**. FAERS uses free-text names; TWOSIDES and DrugBank use RxNorm RXCUI; DrugBank also has its own proprietary IDs. RxNorm normalization (Day 4) is the bridge that makes these joinable.

---

## Source 1: FAERS Quarterly ASCII Files

**Coverage:** 2020Q1 – 2025Q4 (24 quarters)  
**Format:** ZIP → pipe/`$`-delimited `.txt` files  
**Encoding:** latin-1 (drug names contain non-ASCII brand characters)  
**Primary key across tables:** `PRIMARYID` (a version-aware case ID)  

> [!NOTE]
> `PRIMARYID` is NOT unique across quarters. A case can be updated in a later quarter with a new `PRIMARYID` but the same `CASEID`. The ETL will resolve this by keeping `MAX(PRIMARYID)` per `CASEID`.

### DEMO — Demographics and Case Metadata

| Column | Type | Description | Used? | Notes |
|---|---|---|---|---|
| `PRIMARYID` | INT | Version-aware case identifier | ✅ | Join key to all other tables |
| `CASEID` | INT | Case identifier (stable across quarters) | ✅ | Deduplication key |
| `FDA_DT` | TEXT | Date FDA received report (`YYYYMMDD`) | ✅ | For temporal analysis |
| `REPT_DT` | TEXT | Date report was submitted | ❌ | Redundant with FDA_DT |
| `REPT_COD` | TEXT | Report type: EXP (expedited), DIR (direct) | ✅ | EXP = more serious |
| `MFR_SNDR` | TEXT | Manufacturer name (if industry report) | ❌ | Too noisy |
| `GNDR_COD` | TEXT | Patient sex: M, F, UNK | ✅ | Covariate for causal estimation |
| `AGE` | FLOAT | Patient age (in units given by AGE_COD) | ✅ | Covariate |
| `AGE_COD` | TEXT | Age unit: YR, MON, WK, DY, HR, DEC | ✅ | Must normalize to years |
| `WT` | FLOAT | Patient weight (in WT_COD units) | ✅ | Covariate |
| `WT_COD` | TEXT | Weight unit: KG, LBS | ✅ | Must normalize to kg |
| `OCCP_COD` | TEXT | Reporter occupation: MD, PH, CN, LW, HP, OT | ✅ | MD reports are higher quality |
| `REPORTER_COUNTRY` | TEXT | ISO 2-letter country code | ✅ | Covariate + QC filter |
| `INIT_FAX` | TEXT | Initial fax ID | ❌ | Not used |

**Excluded columns:** `DEATH_DT`, `TO_MFR`, `CONFID`, `AUTH_NUM`, `MFR_CNTRL`, `MFR_DT`, `LIT_REF`

---

### DRUG — Drugs Mentioned in Each Case

| Column | Type | Description | Used? | Notes |
|---|---|---|---|---|
| `PRIMARYID` | INT | Join key to DEMO | ✅ | |
| `CASEID` | INT | Case ID | ✅ | |
| `DRUG_SEQ` | INT | Drug sequence number within a case | ✅ | Needed for dedup |
| `ROLE_COD` | TEXT | Drug role: PS, SS, C, I | ✅ | **PS-only filter** in signal detection |
| `DRUGNAME` | TEXT | Free-text drug name as reported | ✅ | **Primary normalization target** |
| `PROD_AI` | TEXT | Active ingredient(s) as reported | ✅ | Fallback if DRUGNAME is a brand name |
| `VAL_VBM` | INT | Validity code for dose | ❌ | Not used |
| `ROUTE` | TEXT | Administration route (oral, IV, etc.) | ✅ | Covariate |
| `DOSE_VBM` | TEXT | Dose as reported (free text) | ❌ | Too noisy, not normalized |
| `NDA_NUM` | TEXT | NDA/BLA application number | ❌ | Not used |

**Key insight from exploration:** `DRUGNAME` is free text — same drug appears as "ASPIRIN", "aspirin", "Aspirin 81 MG", "ACETYLSALICYLIC ACID", etc. `PROD_AI` sometimes has the generic name when `DRUGNAME` is a brand. The Day 4 ETL will lowercase both and try `PROD_AI` as a fallback.

**Excluded columns:** `CUM_DOSE_CHR`, `CUM_DOSE_UNIT`, `DECHAL`, `RECHAL`, `LOT_NUM`, `EXP_DT`, `NDA_NUM`

---

### REAC — Adverse Reactions/Events

| Column | Type | Description | Used? | Notes |
|---|---|---|---|---|
| `PRIMARYID` | INT | Join key to DEMO | ✅ | |
| `CASEID` | INT | Case ID | ✅ | |
| `PT` | TEXT | MedDRA Preferred Term (the adverse event) | ✅ | **Primary signal target** |
| `DRUG_REC_ACT` | TEXT | Drug reaction action (drug withdrawn, dose reduced, etc.) | ✅ | Evidence quality signal |

**Key insight:** `PT` is already standardized MedDRA terminology — no normalization needed. This is our `event_name` throughout the pipeline.

---

### OUTC — Patient Outcomes

| Column | Type | Description | Used? | Notes |
|---|---|---|---|---|
| `PRIMARYID` | INT | Join key to DEMO | ✅ | |
| `CASEID` | INT | Case ID | ✅ | |
| `OUTC_COD` | TEXT | Outcome severity code | ✅ | See codes below |

**OUTC_COD values:**
| Code | Meaning |
|---|---|
| `DE` | Death |
| `LT` | Life-Threatening |
| `HO` | Hospitalization (initial or prolonged) |
| `DS` | Disability |
| `CA` | Congenital Anomaly |
| `RI` | Required Intervention to Prevent Permanent Impairment |
| `OT` | Other Serious (Important Medical Event) |

---

## Source 2: TWOSIDES (nSIDES Project)

**URL:** Zenodo DOI 10.5281/zenodo.10975016  
**Format:** CSV.gz (~1.9 GB uncompressed)  
**Rows:** ~4.6 million drug-pair to side-effect associations  
**Role in PolySignal:** Ground-truth validation set for GNN link prediction (Day 12–14)

| Column | Type | Description | Used? | Notes |
|---|---|---|---|---|
| `drug1_rxcui` | INT | RxNorm RXCUI of drug 1 | ✅ | Join key to RxNorm |
| `drug1_concept_name` | TEXT | Human-readable name for drug 1 | ✅ | For display/debugging |
| `drug2_rxcui` | INT | RxNorm RXCUI of drug 2 | ✅ | Join key to RxNorm |
| `drug2_concept_name` | TEXT | Human-readable name for drug 2 | ✅ | |
| `condition_meddra_id` | INT | MedDRA concept ID for the side effect | ✅ | Maps to FAERS PT via MedDRA |
| `condition_concept_name` | TEXT | MedDRA name of the side effect | ✅ | Should match FAERS PT |
| `A` | INT | Contingency cell: drug combo + event | ✅ | For PRR recomputation |
| `B` | INT | Drug combo without event | ✅ | |
| `C` | INT | Event without drug combo | ✅ | |
| `D` | INT | Neither | ✅ | |
| `PRR` | FLOAT | Proportional Reporting Ratio | ✅ | Cross-validate against our PRR |
| `PRR_error` | FLOAT | Standard error of PRR | ✅ | |
| `mean_reporting_frequency` | FLOAT | Reporting frequency across quarters | ❌ | Not used directly |

> [!WARNING]
> TWOSIDES drug identifiers are **RxNorm RXCUI**, not free-text names. They will NOT match FAERS `DRUGNAME` directly. The Day 4 ETL RXCUI normalization step is what allows these to be joined.

---

## Source 3: DrugBank (Free Academic Tier)

**URL:** https://go.drugbank.com/releases/latest (requires free academic account)  
**Format:** XML (full) or CSV (approved drugs subset)  
**Role in PolySignal:** Known drug-drug interaction pairs as a second ground-truth layer

### From the XML (preferred) or approved CSV

| Column / XPath | Type | Description | Used? | Notes |
|---|---|---|---|---|
| `drugbank-id[@primary='true']` | TEXT | DrugBank canonical ID (e.g. DB00945) | ✅ | Internal join key |
| `name` | TEXT | Generic/official drug name | ✅ | For display |
| `groups/group` | TEXT | Approved, illicit, investigational, etc. | ✅ | Filter to `approved` only |
| `atc-codes/atc-code[@code]` | TEXT | ATC pharmacological class code | ✅ | Feature for GNN node embeddings |
| `external-identifiers/.../RxCUI` | INT | RxNorm cross-reference | ✅ | **Bridge to FAERS and TWOSIDES** |
| `drug-interactions/drug-interaction` | LIST | Known DDI pairs (drugbank-id + description) | ✅ | Ground-truth negative-edge filter |

**Excluded:** Full drug descriptions, pharmacokinetics, protein binding data, references, mixtures

> [!NOTE]
> DrugBank DDI pairs will be used to **filter negative samples** during GNN link prediction training. We don't want to train the model to predict "no link" for a pair that DrugBank already says interacts (that would be a false negative in training).

---

## Source 4: RxNorm Prescribable Content

**URL:** https://download.nlm.nih.gov/rxnorm/RxNorm_full_prescribe_06012026.zip  
**License:** No license required (Prescribable Content subset)  
**Format:** Pipe-delimited RRF files  
**Role in PolySignal:** Canonical drug name normalization — bridge between FAERS free text and TWOSIDES/DrugBank RXCUI

### RXNCONSO.RRF — Concept Names and Sources

| Column | Position | Description | Used? | Notes |
|---|---|---|---|---|
| `RXCUI` | 0 | RxNorm Concept Unique Identifier | ✅ | The canonical drug ID we normalize TO |
| `LAT` | 1 | Language (always ENG) | ❌ | |
| `SAB` | 11 | Source vocabulary abbreviation | ✅ | Filter: RXNORM, MMSL |
| `TTY` | 12 | Term type | ✅ | Filter: IN (ingredient), BN (brand), PIN |
| `CODE` | 13 | Code in source vocabulary | ✅ | For SAB=MMSL: Multum drug code |
| `STR` | 14 | String (the drug name in this vocabulary) | ✅ | **Lookup target for FAERS normalization** |

**Key TTY values we use:**
| TTY | Meaning | Use |
|---|---|---|
| `IN` | Ingredient | Normalize TO this (generic name) |
| `BN` | Brand Name | Match FAERS brand names FROM this |
| `PIN` | Precise Ingredient | Fallback for salts/esters |
| `MIN` | Multiple Ingredients | Drug combos |

**Normalization strategy (Day 4):**
1. Lowercase FAERS `DRUGNAME`
2. Look up in RXNCONSO `STR` (all TTYs) → get `RXCUI`
3. If found, map to the `IN` (ingredient) concept for that RXCUI
4. If not found, try `PROD_AI` column as fallback
5. If still not found, flag as `UNMAPPED` for manual review

---

## Join Map

```
FAERS.DRUG.DRUGNAME
      │
      │ (Day 4: RxNorm fuzzy lookup)
      ▼
   RXNCONSO.STR → RXCUI
      │                │
      │                └──► TWOSIDES.drug1_rxcui / drug2_rxcui
      │                └──► DrugBank external-identifier/RxCUI
      │
      ▼
FAERS.DRUG.DRUGNAME_RXCUI   (new column added by ETL)
      │
      ├─► signals.disproportionality (drug_name = RXCUI canonical name)
      └─► graph.HeteroData drug nodes
```

---

## Data Quality Rules (enforced in Day 4 ETL)

| Rule | Check | Action on failure |
|---|---|---|
| DEMO uniqueness | `MAX(PRIMARYID) per CASEID` | Keep latest, drop older versions |
| DRUG role filter | `ROLE_COD == 'PS'` | Drop non-primary-suspect drugs |
| Age normalization | Convert MON/WK/DY/HR/DEC → years | Set to NULL if unit unknown |
| Weight normalization | Convert LBS → KG | `weight_kg = weight_lbs / 2.205` |
| Minimum reports | `n_reports >= 3` per drug-event pair | Exclude from signal detection |
| Drug name mapped | RXCUI found in RxNorm | Flag as UNMAPPED, exclude from GNN |

---

## Row Count Checkpoints (to be filled in after download)

| Dataset | Raw rows | After ROLE_COD=PS filter | After RxNorm mapping | Notes |
|---|---|---|---|---|
| FAERS DRUG (2024Q4) | _TBD_ | _TBD_ | _TBD_ | |
| FAERS DEMO (2024Q4) | _TBD_ | — | — | |
| TWOSIDES (sample 100k) | 100,000 | — | — | |
| TWOSIDES (full) | ~4.6M | — | — | |
| RxNorm RXNCONSO (sample) | _TBD_ | — | — | Only IN+BN rows |
