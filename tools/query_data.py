"""
tools/query_data.py
--------------------
Converts natural language questions → SQL → executes on financials.db → returns result.

Flow:
    User question (natural lang)
        │
        ▼
    LLM (OpenRouter — free model) — given schema + question → outputs SQL
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
import time
import json
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from groq import Groq
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ── Groq config ───────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found in .env file.\n"
        "Get a free key (no credit card): https://console.groq.com/"
    )

# llama-3.1-8b-instant: fast, higher rate limits — ideal for SQL generation
# llama-3.3-70b-versatile: higher quality but slower (overkill for SQL)
SQL_MODEL      = os.getenv("GROQ_SQL_MODEL",  "llama-3.1-8b-instant")
SQL_MODEL_SLOW = os.getenv("GROQ_MODEL",      "llama-3.3-70b-versatile")

# ── paths — tries both "structure" and "structured" folder names ──────────────
ROOT = Path(__file__).parent.parent
_db_candidates = [
    ROOT / "data" / "structure"  / "financials.db",
    ROOT / "data" / "structured" / "financials.db",
]
DB_FILE = next((p for p in _db_candidates if p.exists()), _db_candidates[0])

# ── exact schema shown to LLM ─────────────────────────────────────────────────
DB_SCHEMA = """
SQLite database: financials.db
Table: financials

Columns:
  company            TEXT     -- Company name. Values: 'Infosys', 'Accenture', 'Cognizant'
  fiscal_year        TEXT     -- Fiscal year. Values: 'FY22', 'FY23', 'FY24', 'FY25'
  quarter            TEXT     -- Quarter. Values: 'Q1', 'Q2', 'Q3', 'Q4', 'Annual'
  revenue_bn_USD     REAL     -- Total revenue in billion USD
  op_margin_pct      REAL     -- Operating margin as percentage (e.g. 24.6 means 24.6%)
  headcount          INTEGER  -- Total number of employees
  epsusd             REAL     -- Earnings per share in USD
  source_link        TEXT     -- Source URL
  notes              TEXT     -- Additional notes

Sample rows:
  ('Infosys',   'FY24', 'Q4',     24.1, 20.7, 317240, 63.1,  'https://...', 'Q4 FY24')
  ('Accenture', 'FY24', 'Annual', 64.9, 15.2, 738000,  8.5,  'https://...', 'FY24 Annual')
  ('Cognizant', 'FY24', 'Annual', 19.4, 14.8, 347700, 4.56,  'https://...', 'FY24 Annual')

All data: 3 companies x 4 fiscal years (FY22, FY23, FY24, FY25), with quarterly breakdowns.
"""

# ── SQL generation prompt ─────────────────────────────────────────────────────
SQL_SYSTEM_PROMPT = """You are a SQLite expert. Convert the natural language question into a valid SQLite SELECT query.

RULES — follow all of them strictly:
1. Output ONLY the raw SQL query. No explanation, no markdown, no backticks, no preamble.
2. Use exact column names from the schema (case-sensitive).
3. Use exact company strings: 'Infosys', 'Accenture', 'Cognizant'
4. Use exact fiscal year strings: 'FY22', 'FY23', 'FY24', 'FY25'
5. Always include company and fiscal_year in SELECT unless asking for a single scalar.
6. For "best/highest/lowest" → use ORDER BY + LIMIT 1.
7. For comparisons across companies → return all relevant rows (no LIMIT unless asked).
8. For growth calculations → use subqueries or LAG() window function.
9. If the question cannot be answered from this schema → output: SELECT 'NOT_IN_SCHEMA' AS error
10. Never use DROP, INSERT, UPDATE, DELETE — SELECT only.

Database schema:
{schema}
"""


# ── step 1: natural language → SQL ───────────────────────────────────────────
def _generate_sql(question: str, max_retries: int = 3) -> str:
    """
    Call Groq to convert natural language to SQL SELECT query.
    
    Uses llama-3.1-8b-instant (fast, generous limits) for SQL generation.
    Falls back to llama-3.3-70b-versatile on rate limit.
    Returns raw SQL string.
    """
    client = Groq(api_key=GROQ_API_KEY)
    model  = SQL_MODEL   # start with fast model

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model       = model,
                messages    = [
                    {"role": "system", "content": SQL_SYSTEM_PROMPT.format(schema=DB_SCHEMA)},
                    {"role": "user",   "content": f"Question: {question}\nSQL:"},
                ],
                temperature = 0.1,    # low temp = deterministic SQL
                max_tokens  = 250,
            )
            sql = resp.choices[0].message.content.strip()

            # strip markdown fences if model adds them
            if "```" in sql:
                sql = "\n".join(
                    l for l in sql.split("\n")
                    if not l.strip().startswith("```")
                ).strip()

            return sql

        except Exception as e:
            err    = str(e)
            is_rate = "429" in err or "rate_limit" in err.lower()

            if is_rate and model == SQL_MODEL:
                # step up to larger model — different quota bucket
                print(f"  ⚡ Rate limit on {model} → switching to {SQL_MODEL_SLOW}", file=sys.stderr)
                model = SQL_MODEL_SLOW
                continue

            elif is_rate and attempt < max_retries - 1:
                wait = 10 * (attempt + 1)
                print(f"  ⏳ Rate limit. Waiting {wait}s ({attempt+1}/{max_retries})", file=sys.stderr)
                time.sleep(wait)
                continue

            else:
                raise

    raise Exception(f"SQL generation failed after {max_retries} attempts")


# ── step 2: execute SQL ───────────────────────────────────────────────────────
def _execute_sql(sql: str) -> tuple[list[dict], list[str]]:
    """Execute SQL against financials.db. Returns (rows, column_names)."""
    if not DB_FILE.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_FILE}\n"
            "Run: python scripts/ingest_structured.py"
        )

    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(sql)
        rows   = cursor.fetchall()
        cols   = [d[0] for d in cursor.description] if cursor.description else []
        return [dict(r) for r in rows], cols
    except sqlite3.Error as e:
        raise sqlite3.Error(f"SQL failed: {e} | SQL was: {sql}")
    finally:
        conn.close()


# ── step 3: source reference ──────────────────────────────────────────────────
def _source_ref(rows: list[dict]) -> str:
    if not rows:
        return "financials.db (no rows matched)"
    companies = sorted({r.get("company", "?") for r in rows if r.get("company")})
    years     = sorted({r.get("fiscal_year", "?") for r in rows if r.get("fiscal_year")})
    parts     = []
    if companies: parts.append(", ".join(companies))
    if years:     parts.append(", ".join(years))
    n = len(rows)
    return f"financials.db — {' | '.join(parts)} ({n} row{'s' if n!=1 else ''})"


# ── main tool function ────────────────────────────────────────────────────────
def query_data(question: str) -> dict:
    """
    Convert natural language question → SQL → execute → return result.

    Returns dict with keys:
        result      : list of row dicts, or scalar if single cell
        columns     : column name list
        row_count   : number of rows
        sql         : executed SQL (for trace logging)
        source      : human-readable source reference
        error       : only present on failure
    """
    # generate SQL
    try:
        sql = _generate_sql(question)
    except Exception as e:
        return {"result": None, "columns": [], "row_count": 0,
                "sql": "", "source": "financials.db",
                "error": f"SQL generation failed: {e}"}

    # check schema boundary
    if "NOT_IN_SCHEMA" in sql:
        return {"result": None, "columns": [], "row_count": 0,
                "sql": sql, "source": "financials.db",
                "error": "Question cannot be answered from structured data. "
                         "Try search_docs or web_search instead."}

    # execute
    try:
        rows, cols = _execute_sql(sql)
    except (sqlite3.Error, FileNotFoundError) as e:
        return {"result": None, "columns": [], "row_count": 0,
                "sql": sql, "source": "financials.db", "error": str(e)}

    # no rows
    if not rows:
        return {"result": [], "columns": cols, "row_count": 0,
                "sql": sql,
                "source": "financials.db (no rows matched — check company name or fiscal year)"}

    # single cell → scalar
    if len(rows) == 1 and len(cols) == 1:
        return {"result": rows[0][cols[0]], "columns": cols, "row_count": 1,
                "sql": sql, "source": _source_ref(rows)}

    # table
    return {"result": rows, "columns": cols, "row_count": len(rows),
            "sql": sql, "source": _source_ref(rows)}


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print('  python tools/query_data.py "your question"')
        print('  python tools/query_data.py --sql "SELECT * FROM financials LIMIT 3"')
        sys.exit(1)

    args = sys.argv[1:]

    # direct SQL mode — bypasses LLM for debugging
    if "--sql" in args:
        raw_sql = " ".join(args[args.index("--sql") + 1:])
        print(f"SQL: {raw_sql}\n")
        try:
            rows, cols = _execute_sql(raw_sql)
            print(pd.DataFrame(rows).to_string(index=False) if rows else "No rows.")
        except Exception as e:
            print(f"Error: {e}")
        sys.exit(0)

    # natural language mode
    question = " ".join(args)
    print(f"Question : {question}")
    print("─" * 60)

    result = query_data(question)

    if "error" in result:
        print(f"Error : {result['error']}")
        if result.get("sql"): print(f"SQL   : {result['sql']}")
        sys.exit(1)

    print(f"SQL    : {result['sql']}")
    print(f"Source : {result['source']}")
    print(f"Rows   : {result['row_count']}\n")

    res = result["result"]
    if isinstance(res, list) and res:
        print(pd.DataFrame(res).to_string(index=False))
    elif isinstance(res, list):
        print("No data matched.")
    else:
        print(f"{result['columns'][0]} = {res}")