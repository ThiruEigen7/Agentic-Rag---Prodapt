"""
tools/query_data.py
--------------------
Converts natural language questions → SQL → executes on financials.db → returns result.

Flow:
    User question (natural lang)
        │
        ▼
    LLM (Gemini) — given schema + question → outputs SQL
        │
        ▼
    sqlite3 executes SQL on financials.db
        │
        ▼
    Returns {result, sql, columns, row_count, source}

Standalone test:
    python tools/query_data.py "What was Infosys revenue in FY24?"
    python tools/query_data.py "Compare operating margins of all companies in FY24"
    python tools/query_data.py "Which company had highest headcount in FY23?"
    python tools/query_data.py --sql "SELECT * FROM financials WHERE fiscal_year='FY24'"
"""

import os
import sys
import sqlite3
from pathlib import Path
import warnings
import time

# Suppress FutureWarning about google.generativeai deprecation
warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")

import google.generativeai as genai
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ── Gemini configuration ──────────────────────────────────────────────────────
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")
genai.configure(api_key=API_KEY)
# Using gemini-2.5-flash (latest free model)
MODEL = "gemini-2.5-flash"

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent.parent
DB_FILE = ROOT / "data" / "structure" / "financials.db"

# fallback paths — checks both folder names

    
if not DB_FILE.exists():
    DB_FILE = ROOT / "data" / "structure" / "financials.db"

LLM_MODEL = "gemini-2.5-flash"

# ── exact schema shown to LLM ─────────────────────────────────────────────────
# This is what the LLM reads to generate correct SQL.
# Update this if your schema changes.
DB_SCHEMA = """
SQLite database: financials.db
Table: financials

Columns:
  company            TEXT     -- Company name. Values: 'Infosys', 'TCS', 'Accenture', 'Cognizant', 'Wipro'
  fiscal_year        TEXT     -- Fiscal year. Values: 'FY22', 'FY23', 'FY24', 'FY25'
  quarter            TEXT     -- Quarter. Values: 'Q1', 'Q2', 'Q3', 'Q4'
  revenue_bn_USD     REAL     -- Total revenue in billion USD
  op_margin_pct      REAL     -- Operating margin as percentage. e.g. 24.6 means 24.6%
  headcount          INTEGER  -- Total number of employees
  epsusd             REAL     -- Earnings per share in USD
  source_link        TEXT     -- Source URL
  notes              TEXT     -- Additional notes

Sample rows:
  ('Infosys', 'FY24', 'Q4', 24.1, 20.7, 317240, 63.1, 'https://...', 'Q4 FY24')
  ('TCS', 'FY24', 'Q4', 60.2, 24.6, 601546, 126.7, 'https://...', 'Q4 FY24')
  ('Accenture', 'FY24', 'Q4', 17.1, 15.2, 738000, 8.5, 'https://...', 'Q4 FY24')

All data across 4+ companies x 4 fiscal years (FY22, FY23, FY24, FY25).
"""

# ── SQL generation prompt ─────────────────────────────────────────────────────
SQL_SYSTEM_PROMPT = """You are a SQLite expert. Your only job is to convert a natural language question into a valid SQLite SELECT query.

RULES — follow all of them:
1. Output ONLY the raw SQL query. No explanation, no markdown, no backticks, no preamble.
2. Use exact column names from the schema. IMPORTANT: Column names are CASE-SENSITIVE.
3. Use exact company name strings (case-sensitive): 'Infosys', 'TCS', 'Accenture', 'Cognizant', 'Wipro'
4. Use exact fiscal year format: 'FY22', 'FY23', 'FY24', 'FY25' (strings, not integers)
5. Always include company and fiscal_year in SELECT unless asking for a single aggregate value.
6. For growth calculations, use subqueries or window functions.
7. For "best" or "highest" or "lowest", use ORDER BY + LIMIT 1.
8. For comparisons across companies, return all relevant rows (no LIMIT unless asked).
9. If the question asks for something not in the schema, output exactly: SELECT 'NOT_IN_SCHEMA' AS error
10. Never use DROP, INSERT, UPDATE, DELETE or any non-SELECT statement.

Database schema:
{schema}
"""


# ── step 1: natural language → SQL ───────────────────────────────────────────
def _generate_sql(question: str, max_retries: int = 3) -> str:
    """
    Call Gemini to convert natural language question to a SQL SELECT query.
    Includes retry logic with exponential backoff for rate limits.
    Returns raw SQL string.
    """
    model = genai.GenerativeModel(LLM_MODEL)
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                f"{SQL_SYSTEM_PROMPT.format(schema=DB_SCHEMA)}\n\nQuestion: {question}\nSQL:"
            )
            
            sql = response.text.strip()
            
            # strip markdown code fences if LLM adds them despite instructions
            if sql.startswith("```"):
                lines = sql.split("\n")
                sql   = "\n".join(
                    line for line in lines
                    if not line.startswith("```")
                ).strip()
            
            return sql
            
        except Exception as e:
            error_msg = str(e)
            
            # Check if it's a rate limit error (503, 429, or UNAVAILABLE)
            is_rate_limit = any(code in error_msg for code in ["503", "429", "UNAVAILABLE", "overloaded"])
            
            if is_rate_limit and attempt < max_retries - 1:
                # Calculate exponential backoff: 2^attempt seconds
                wait_time = 2 ** attempt
                print(f"  ⏳ Rate limited. Retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})...", file=sys.stderr)
                time.sleep(wait_time)
                continue
            else:
                # Not a rate limit error, or last attempt failed
                raise e
    
    # Should not reach here
    raise Exception(f"Failed to generate SQL after {max_retries} attempts")


# ── step 2: execute SQL ───────────────────────────────────────────────────────
def _execute_sql(sql: str) -> tuple[list[dict], list[str]]:
    """
    Execute a SQL SELECT query against financials.db.
    Returns (rows_as_list_of_dicts, column_names).
    Raises sqlite3.Error if SQL is invalid.
    """
    if not DB_FILE.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_FILE}\n"
            "Run: python scripts/ingest_structured.py"
        )

    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row   # rows behave like dicts

    try:
        cursor = conn.execute(sql)
        rows   = cursor.fetchall()
        cols   = [d[0] for d in cursor.description] if cursor.description else []
        result = [dict(row) for row in rows]
    except sqlite3.Error as e:
        conn.close()
        raise sqlite3.Error(f"SQL failed: {e} | SQL was: {sql}")
    finally:
        conn.close()

    return result, cols


# ── step 3: format source reference ──────────────────────────────────────────
def _source_ref(rows: list[dict]) -> str:
    """Build a human-readable source reference for the trace log."""
    if not rows:
        return "financials.db (no rows matched)"
    companies = sorted({r.get("company", "?") for r in rows if r.get("company")})
    years     = sorted({r.get("fiscal_year", "?") for r in rows if r.get("fiscal_year")})
    parts     = []
    if companies:
        parts.append(", ".join(companies))
    if years:
        parts.append(", ".join(years))
    return f"financials.db — {' | '.join(parts)} ({len(rows)} row{'s' if len(rows)!=1 else ''})"


# ── main tool function ────────────────────────────────────────────────────────
def query_data(question: str) -> dict:
    """
    Convert a natural language question to SQL, execute it, return result.

    Args:
        question: Natural language question about the financial data.
                  e.g. "What was Infosys operating margin in FY24?"
                  e.g. "Which company had the highest revenue in FY23?"
                  e.g. "Show headcount for all companies across all years"

    Returns:
        dict with keys:
            result      : list of row dicts, or scalar value if single cell
            columns     : list of column names
            row_count   : number of rows
            sql         : the SQL that was executed (for trace logging)
            source      : human-readable source reference
            error       : present only if something went wrong

    Example return for single value:
        {
            "result": 20.7,
            "columns": ["op_margin_pct"],
            "row_count": 1,
            "sql": "SELECT op_margin_pct FROM financials WHERE ...",
            "source": "financials.db — Infosys | FY24 (1 row)"
        }

    Example return for table:
        {
            "result": [
                {"company": "Infosys", "fiscal_year": "FY24", "op_margin_pct": 20.7},
                {"company": "TCS",     "fiscal_year": "FY24", "op_margin_pct": 24.6},
            ],
            "columns": ["company", "fiscal_year", "op_margin_pct"],
            "row_count": 2,
            "sql": "SELECT ...",
            "source": "financials.db — Infosys, TCS | FY24 (2 rows)"
        }
    """
    # ── generate SQL ──────────────────────────────────────────────────────────
    try:
        sql = _generate_sql(question)
    except Exception as e:
        return {
            "result"   : None,
            "columns"  : [],
            "row_count": 0,
            "sql"      : "",
            "source"   : "financials.db",
            "error"    : f"SQL generation failed: {e}",
        }

    # ── check if LLM said question is not in schema ───────────────────────────
    if "NOT_IN_SCHEMA" in sql:
        return {
            "result"   : None,
            "columns"  : [],
            "row_count": 0,
            "sql"      : sql,
            "source"   : "financials.db",
            "error"    : "Question cannot be answered from the structured data. "
                         "Try search_docs or web_search instead.",
        }

    # ── execute SQL ───────────────────────────────────────────────────────────
    try:
        rows, cols = _execute_sql(sql)
    except sqlite3.Error as e:
        return {
            "result"   : None,
            "columns"  : [],
            "row_count": 0,
            "sql"      : sql,
            "source"   : "financials.db",
            "error"    : str(e),
        }
    except FileNotFoundError as e:
        return {
            "result"   : None,
            "columns"  : [],
            "row_count": 0,
            "sql"      : sql,
            "source"   : "financials.db",
            "error"    : str(e),
        }

    # ── no rows returned ──────────────────────────────────────────────────────
    if not rows:
        return {
            "result"   : [],
            "columns"  : cols,
            "row_count": 0,
            "sql"      : sql,
            "source"   : "financials.db (no rows matched — check company name or fiscal year)",
        }

    # ── single cell → return scalar directly ─────────────────────────────────
    if len(rows) == 1 and len(cols) == 1:
        return {
            "result"   : rows[0][cols[0]],
            "columns"  : cols,
            "row_count": 1,
            "sql"      : sql,
            "source"   : _source_ref(rows),
        }

    # ── multiple rows or columns → return table ───────────────────────────────
    return {
        "result"   : rows,
        "columns"  : cols,
        "row_count": len(rows),
        "sql"      : sql,
        "source"   : _source_ref(rows),
    }


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print('  python tools/query_data.py "your question in natural language"')
        print('  python tools/query_data.py --sql "SELECT * FROM financials LIMIT 3"')
        print()
        print("Examples:")
        print('  python tools/query_data.py "What was Infosys revenue in FY24?"')
        print('  python tools/query_data.py "Compare operating margins of all companies in FY24"')
        print('  python tools/query_data.py "Which company had highest headcount in FY23?"')
        print('  python tools/query_data.py "Show revenue growth for TCS across all years"')
        sys.exit(1)

    args = sys.argv[1:]

    # ── direct SQL mode (bypass LLM — for manual debugging) ──────────────────
    if "--sql" in args:
        sql_idx = args.index("--sql")
        raw_sql = " ".join(args[sql_idx + 1:])
        print(f"Running SQL directly: {raw_sql}\n")
        try:
            rows, cols = _execute_sql(raw_sql)
            if rows:
                df = pd.DataFrame(rows)
                print(df.to_string(index=False))
            else:
                print("No rows returned.")
        except Exception as e:
            print(f"Error: {e}")
        sys.exit(0)

    # ── natural language mode ─────────────────────────────────────────────────
    question = " ".join(args)
    print(f"Question : {question}")
    print(f"{'─' * 60}")

    result = query_data(question)

    # show error if present
    if "error" in result:
        print(f"Error    : {result['error']}")
        if result.get("sql"):
            print(f"SQL      : {result['sql']}")
        sys.exit(1)

    print(f"SQL      : {result['sql']}")
    print(f"Source   : {result['source']}")
    print(f"Rows     : {result['row_count']}")
    print()

    # display result
    if isinstance(result["result"], list) and result["result"]:
        df = pd.DataFrame(result["result"])
        print(df.to_string(index=False))
    elif isinstance(result["result"], list) and not result["result"]:
        print("No data matched.")
    else:
        # scalar
        col = result["columns"][0] if result["columns"] else "value"
        print(f"{col} = {result['result']}")