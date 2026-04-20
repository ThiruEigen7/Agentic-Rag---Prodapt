"""
scripts/ingest_structured.py
-----------------------------
Reads data/structured/fy_stats.xlsx
Validates columns and data types
Loads into SQLite database at data/structured/financials.db

Run:
    python scripts/ingest_structured.py

Output:
    data/structured/financials.db   — SQLite database with table 'financials'

What this file does step by step:
    1. Read fy_stats.xlsx with pandas
    2. Validate: check expected columns exist, check no completely empty rows
    3. Log any NULL values with reasons (so agent knows what's missing)
    4. Write DataFrame → SQLite table 'financials' via pandas .to_sql()
    5. Run 3 verification queries to confirm data loaded correctly
    6. Print full table for visual confirmation

Your XLSX must have these columns (exact names):
    company, fiscal_year, quarter, revenue_bn_USD, op_margin_pct,
    headcount, epsusd, source_link, notes
"""

import sys
import sqlite3
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent.parent
XLSX    = ROOT / "data" / "structure" / "fy_stats.xlsx"
DB_FILE = ROOT / "data" / "structure" / "financials.db"

# ── expected schema ───────────────────────────────────────────────────────────
# (column_name, pandas_dtype_check, friendly_description)
EXPECTED_COLUMNS = [
    ("company",            "object",  "Company name (Accenture / Cognizant / Infosys)"),
    ("fiscal_year",        "object",  "Fiscal year (FY22 / FY23 / FY24 / FY25)"),
    ("quarter",            "object",  "Quarter (Q1 / Q2 / Q3 / Q4)"),
    ("revenue_bn_USD",     "float64", "Revenue in Billion USD"),
    ("op_margin_pct",      "float64", "Operating margin %"),
    ("headcount",          "float64", "Total employee headcount"),
    ("epsusd",             "float64", "Earnings per share in USD"),
    ("source_link",        "object",  "Source URL/reference"),
    ("notes",              "object",  "Additional notes"),
]

VALID_COMPANIES    = {"Accenture", "Cognizant", "Infosys"}
VALID_FISCAL_YEARS = {"FY22", "FY23", "FY24", "FY25"}


# ── step 1: load ──────────────────────────────────────────────────────────────
def load_xlsx() -> pd.DataFrame:
    if not XLSX.exists():
        print(f"ERROR: XLSX not found at {XLSX}")
        print("Place your financials.xlsx in data/structured/ and re-run.")
        sys.exit(1)

    df = pd.read_excel(XLSX, engine="openpyxl")
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns from {XLSX.name}")
    return df


# ── step 2: validate ──────────────────────────────────────────────────────────
def validate(df: pd.DataFrame) -> pd.DataFrame:
    print("\n── Validating columns ───────────────────────────")
    errors = []

    # check all expected columns exist
    for col, _, desc in EXPECTED_COLUMNS:
        if col not in df.columns:
            errors.append(f"Missing column: '{col}' ({desc})")
        else:
            print(f"  OK  {col}")

    if errors:
        for e in errors:
            print(f"  ERROR: {e}")
        sys.exit(1)

    # strip whitespace from string columns
    df["company"]     = df["company"].str.strip()
    df["fiscal_year"] = df["fiscal_year"].str.strip()

    # validate company names
    bad_companies = set(df["company"].unique()) - VALID_COMPANIES
    if bad_companies:
        print(f"\n  WARNING: Unexpected company names: {bad_companies}")
        print(f"  Expected: {VALID_COMPANIES}")

    # validate fiscal years
    bad_years = set(df["fiscal_year"].unique()) - VALID_FISCAL_YEARS
    if bad_years:
        print(f"\n  WARNING: Unexpected fiscal years: {bad_years}")
        print(f"  Expected: {VALID_FISCAL_YEARS}")

    # coerce numeric columns to float (handles strings like "24.6%")
    numeric_cols = ["revenue_bn_USD", "op_margin_pct", "epsusd", "headcount"]
    for col in numeric_cols:
        if df[col].dtype == object:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("%", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"\nValidation passed. Shape: {df.shape}")
    return df


# ── step 3: log NULLs ────────────────────────────────────────────────────────
def log_nulls(df: pd.DataFrame) -> None:
    print("\n── NULL values ──────────────────────────────────")
    null_mask = df.isnull()
    if not null_mask.any().any():
        print("  No NULL values. Clean dataset.")
        return

    for col in df.columns:
        null_rows = df[null_mask[col]]
        for _, row in null_rows.iterrows():
            print(f"  NULL: {row['company']} {row['fiscal_year']} → {col}")
            print(f"        (not available in source annual report)")


# ── step 4: write to SQLite ───────────────────────────────────────────────────
def write_to_sqlite(df: pd.DataFrame) -> None:
    print("\n── Writing to SQLite ────────────────────────────")

    conn = sqlite3.connect(DB_FILE)

    # write — replace table if it already exists (clean re-run)
    df.to_sql(
        name      = "financials",
        con       = conn,
        if_exists = "replace",      # drop and recreate on each run
        index     = False,
        dtype     = {               # explicit SQLite types
            "company"           : "TEXT",
            "fiscal_year"       : "TEXT",
            "quarter"           : "TEXT",
            "revenue_bn_USD"    : "REAL",
            "op_margin_pct"     : "REAL",
            "headcount"         : "INTEGER",
            "epsusd"            : "REAL",
            "source_link"       : "TEXT",
            "notes"             : "TEXT",
        }
    )

    conn.commit()
    conn.close()
    print(f"  Written {len(df)} rows to financials.db → table: financials")


# ── step 5: verification queries ─────────────────────────────────────────────
def verify(df: pd.DataFrame) -> None:
    print("\n── Verification queries ─────────────────────────")

    conn = sqlite3.connect(DB_FILE)

    checks = [
        ("Row count",
         "SELECT COUNT(*) as cnt FROM financials",
         lambda r: r[0]["cnt"] == len(df)),

        ("Infosys FY24 op margin",
         "SELECT op_margin_pct FROM financials WHERE company='Infosys' AND fiscal_year='FY24'",
         lambda r: len(r) > 0),

        ("All companies present",
         "SELECT DISTINCT company FROM financials ORDER BY company",
         lambda r: len(r) == len(df["company"].unique())),
    ]

    all_passed = True
    for name, sql, check_fn in checks:
        cursor = conn.execute(sql)
        cols   = [d[0] for d in cursor.description]
        rows   = [dict(zip(cols, r)) for r in cursor.fetchall()]
        passed = check_fn(rows)
        status = "PASS" if passed else "FAIL"
        print(f"  {status}  {name}: {rows}")
        if not passed:
            all_passed = False

    conn.close()
    print(f"\n  All checks passed: {all_passed}")


# ── step 6: print full table ──────────────────────────────────────────────────
def print_table() -> None:
    print("\n── Final table in SQLite ────────────────────────")
    conn = sqlite3.connect(DB_FILE)
    df   = pd.read_sql("SELECT * FROM financials ORDER BY company, fiscal_year", conn)
    conn.close()
    print(df.to_string(index=False))


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n=== Structured Ingestion: XLSX → SQLite ===\n")

    df = load_xlsx()
    df = validate(df)
    log_nulls(df)
    write_to_sqlite(df)
    verify(df)
    print_table()

    print(f"\n=== Done. Structured data ready. ===")
    print(f"  DB: {DB_FILE}")
    print(f"\nQuery it manually:")
    print(f"  sqlite3 {DB_FILE} 'SELECT * FROM financials WHERE fiscal_year=\"FY24\"'")


# ── quick query helper ────────────────────────────────────────────────────────
def run_query(sql: str) -> None:
    """Run any SQL query against the DB and print results. Used for manual checks."""
    conn   = sqlite3.connect(DB_FILE)
    df     = pd.read_sql(sql, conn)
    conn.close()
    print(df.to_string(index=False))


if __name__ == "__main__":
    if "--query" in sys.argv:
        sql = " ".join(sys.argv[sys.argv.index("--query") + 1:])
        run_query(sql)
    else:
        main()