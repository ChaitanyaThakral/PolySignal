-- PolySignal PostgreSQL initialization script
-- Runs once when the container is first created.
--
-- Design decisions:
--   - All tables use BIGSERIAL PKs (not UUID) for join performance on
--     analytical queries. UUIDs are prettier but 4x slower on large indexes.
--   - drug_name / event_name are stored as TEXT with a LOWER() functional
--     index because FAERS has wildly inconsistent casing.
--   - The signals table stores BOTH methods' outputs so cross-validation
--     queries are a single self-join rather than joining two tables.

-- ── Schema ────────────────────────────────────────────────────────────────────

CREATE SCHEMA IF NOT EXISTS faers;
CREATE SCHEMA IF NOT EXISTS signals;

-- ── FAERS core tables ─────────────────────────────────────────────────────────

-- One row per FAERS case report
CREATE TABLE IF NOT EXISTS faers.reports (
    report_id       BIGINT PRIMARY KEY,
    quarter         CHAR(6)   NOT NULL,   -- e.g. '24Q1'
    age_years       SMALLINT,
    sex             CHAR(1),              -- 'M' | 'F' | 'U'
    weight_kg       NUMERIC(6,1),
    reporter_type   TEXT,
    country         CHAR(2),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Drugs mentioned in a report (many-to-many via report_id)
CREATE TABLE IF NOT EXISTS faers.report_drugs (
    id              BIGSERIAL PRIMARY KEY,
    report_id       BIGINT  NOT NULL REFERENCES faers.reports(report_id),
    drug_name       TEXT    NOT NULL,
    role            TEXT,                 -- 'PS' primary suspect | 'SS' | 'C' | 'I'
    route           TEXT,
    dose_unit       TEXT
);

-- Adverse events / outcomes mentioned in a report
CREATE TABLE IF NOT EXISTS faers.report_events (
    id              BIGSERIAL PRIMARY KEY,
    report_id       BIGINT  NOT NULL REFERENCES faers.reports(report_id),
    meddra_pt       TEXT    NOT NULL,     -- MedDRA preferred term
    outcome_code    TEXT                  -- 'DE'=death, 'HO'=hospitalization, etc.
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_report_drugs_name
    ON faers.report_drugs (LOWER(drug_name));

CREATE INDEX IF NOT EXISTS idx_report_events_pt
    ON faers.report_events (LOWER(meddra_pt));

-- ── Signal detection output tables ────────────────────────────────────────────

-- Disproportionality statistics (PRR, ROR, EBGM) — one row per drug-event pair
CREATE TABLE IF NOT EXISTS signals.disproportionality (
    id              BIGSERIAL PRIMARY KEY,
    drug_name       TEXT    NOT NULL,
    event_name      TEXT    NOT NULL,
    n_reports       INT     NOT NULL,     -- raw co-occurrence count
    prr             NUMERIC(10,4),        -- Proportional Reporting Ratio
    prr_lower95     NUMERIC(10,4),
    ror             NUMERIC(10,4),        -- Reporting Odds Ratio
    ror_lower95     NUMERIC(10,4),
    ebgm            NUMERIC(10,4),        -- Empirical Bayes Geometric Mean
    eb05            NUMERIC(10,4),        -- 5th percentile (conservative threshold)
    computed_at     TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (drug_name, event_name)
);

-- GNN link-prediction scores — one row per drug-event pair
CREATE TABLE IF NOT EXISTS signals.gnn_scores (
    id              BIGSERIAL PRIMARY KEY,
    drug_name       TEXT    NOT NULL,
    event_name      TEXT    NOT NULL,
    score           NUMERIC(8,6),         -- sigmoid output [0,1]
    model_version   TEXT    NOT NULL,
    computed_at     TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (drug_name, event_name, model_version)
);

-- Cross-validated signal pairs — only rows where BOTH methods agree
CREATE TABLE IF NOT EXISTS signals.validated_pairs (
    id              BIGSERIAL PRIMARY KEY,
    drug_name       TEXT    NOT NULL,
    event_name      TEXT    NOT NULL,
    disp_id         BIGINT  REFERENCES signals.disproportionality(id),
    gnn_id          BIGINT  REFERENCES signals.gnn_scores(id),
    agreement_score NUMERIC(6,4),         -- composite confidence metric
    status          TEXT    DEFAULT 'pending',  -- 'pending' | 'causal_estimated'
    validated_at    TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (drug_name, event_name)
);
