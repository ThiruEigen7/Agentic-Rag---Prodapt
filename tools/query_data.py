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
# IMPORTANT: quarter='FY' is the full-year summary row (named 'FY', NOT 'Annual').
DB_SCHEMA = """
SQLite database: financials.db
Table: financials

Columns:
  company            TEXT     -- Company name. Values: 'Infosys', 'Accenture', 'Cognizant'
  fiscal_year        TEXT     -- Fiscal year. Values: 'FY22', 'FY23', 'FY24', 'FY25'
  quarter            TEXT     -- Quarter. Values: 'Q1', 'Q2', 'Q3', 'Q4', 'FY'
                              --   'FY' = full-year summary row (NOT 'Annual')
  revenue_bn_USD     REAL     -- Total revenue in billion USD
  op_margin_pct      REAL     -- Operating margin as percentage (e.g. 20.1 means 20.1%)
  headcount          INTEGER  -- Total number of employees
  epsusd             REAL     -- Earnings per share in USD
  source_link        TEXT     -- Source URL
  notes              TEXT     -- Additional notes

Sample rows (quarter 'FY' = full-year):
  ('Infosys',   'FY24', 'FY', 18.56, 20.7, 317240, 73.1,  'https://...', 'FY24 full year')
  ('Infosys',   'FY24', 'Q1',  4.62, 20.8, 335186, 18.2,  'https://...', 'Q1 FY24')
  ('Accenture', 'FY24', 'FY', 64.90, 15.5, 733000,  8.5,  'https://...', 'FY24 full year')
  ('Cognizant', 'FY24', 'FY', 19.40, 15.4, 331600,  4.56, 'https://...', 'FY24 full year')

Key rule: to get the full-year figure use WHERE quarter='FY'.
For quarterly breakdown use WHERE quarter IN ('Q1','Q2','Q3','Q4').
To get ALL rows (quarterly + annual) use WHERE quarter IN ('Q1','Q2','Q3','Q4','FY').

All data: 3 companies x 4 fiscal years (FY22, FY23, FY24, FY25) x 5 rows each (Q1-Q4 + FY).
"""

# ── SQL generation prompt ─────────────────────────────────────────────────────
SQL_SYSTEM_PROMPT = """You are a SQLite expert. Convert the natural language question into a valid SQLite SELECT query.

RULES — follow all of them strictly:
1. Output ONLY the raw SQL query. No explanation, no markdown, no backticks, no preamble.
2. Use exact column names from the schema (case-sensitive).
3. Use exact company strings: 'Infosys', 'Accenture', 'Cognizant'
4. Use exact fiscal year strings: 'FY22', 'FY23', 'FY24', 'FY25'
5. CRITICAL: The full-year row uses quarter='FY' (NOT 'Annual'). Never use 'Annual'.
6. ALWAYS include company, fiscal_year, and quarter in SELECT.
7. ALWAYS include revenue_bn_USD and op_margin_pct in SELECT if asking for financials or metrics.
8. For "best/highest/lowest" → use ORDER BY + LIMIT 1 (filter to quarter='FY' first).
9. For comparisons across companies → return all relevant rows (no LIMIT unless asked).
10. For growth calculations → use subqueries or LAG() window function on quarter='FY' rows.
11. If the question cannot be answered from this schema → output: SELECT 'NOT_IN_SCHEMA' AS error
12. Never use DROP, INSERT, UPDATE, DELETE — SELECT only.

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


# ── FY expansion: ensure every FY query returns all quarters + FY row ─────────
import re as _re

def _expand_fy_sql(sql: str) -> str:
    """
    Post-process LLM-generated SQL so that any query that filters to a specific
    fiscal_year (e.g. WHERE fiscal_year='FY24') gets all 5 rows: Q1, Q2, Q3, Q4, FY.

    Also fixes the common LLM mistake of using quarter='Annual' → quarter='FY'.
    """
    # Fix LLM hallucination: 'Annual' is not a valid quarter value
    sql = sql.replace("'Annual'", "'FY'").replace('"Annual"', "'FY'")

    # If query restricts to exactly 'FY', expand it to include quarters as well
    # per user request: "whenever user ask fy data split into 4 q and final fy too"
    sql = _re.sub(
        r"quarter\s*=\s*'FY'",
        "quarter IN ('Q1','Q2','Q3','Q4','FY')",
        sql,
        flags=_re.IGNORECASE,
    )

    # If already using an IN clause, just return
    if _re.search(r"quarter\s+IN", sql, _re.IGNORECASE):
        return sql

    # If filtering on fiscal_year without any quarter filter, inject full breakdown
    if _re.search(r"WHERE\b", sql, _re.IGNORECASE) and not _re.search(r"quarter\s*=", sql, _re.IGNORECASE):
        if _re.search(r"fiscal_year\s*=\s*'FY\d{2}'", sql, _re.IGNORECASE):
            sql = _re.sub(
                r"(fiscal_year\s*=\s*'FY\d{2}')",
                r"\1 AND quarter IN ('Q1','Q2','Q3','Q4','FY')",
                sql,
                count=1,
                flags=_re.IGNORECASE,
            )
    return sql


def _ensure_metrics_selected(sql: str) -> str:
    """
    Heuristically ensure revenue and margin are selected if one is missing
    but the query seems to be about general financials.
    """
    if "SELECT *" in sql.upper():
        return sql
    
    # If it's a SELECT query
    if sql.upper().startswith("SELECT"):
        select_part = sql.split("FROM")[0]
        
        needed = []
        if "revenue_bn_USD" not in select_part and "revenue" in select_part.lower():
             # Already have some revenue mention but maybe not the exact column? 
             # LLM usually uses exact column but just in case.
             pass
        
        if "revenue_bn_USD" not in select_part:
            needed.append("revenue_bn_USD")
        if "op_margin_pct" not in select_part:
            needed.append("op_margin_pct")
            
        if needed:
            last_col_match = _re.search(r"(\w+)\s+FROM", sql, _re.IGNORECASE)
            if last_col_match:
                insertion = ", " + ", ".join(needed)
                sql = sql.replace(" FROM", insertion + " FROM")
    return sql



def _fallback_sql(original_sql: str, question: str) -> str | None:
    """
    When the LLM SQL returns 0 rows, try a broadened fallback query.

    Strategy (in order):
      1. If quarter was filtered to a single value → remove the quarter filter
         to catch all rows for the fiscal year.
      2. If company was filtered → also drop company filter to catch any company.
      3. Otherwise → return None (signal: no fallback available).

    The fallback SQL is returned raw; the caller decides whether to use it.
    """
    sql = original_sql

    # Try: remove a single quarter restriction (e.g. quarter='Annual')
    relaxed = _re.sub(
        r"\s+AND\s+quarter\s*=\s*'[^']+'",
        "",
        sql,
        flags=_re.IGNORECASE,
    )
    if relaxed != sql:
        return relaxed.strip()

    # Try: expand to all quarters if quarter IN clause is too narrow
    relaxed2 = _re.sub(
        r"quarter\s+IN\s+\([^)]+\)",
        "quarter IN ('Q1','Q2','Q3','Q4','FY')",
        sql,
        flags=_re.IGNORECASE,
    )
    if relaxed2 != sql:
        return relaxed2.strip()

    return None  # no simple fallback found


# ── main tool function ────────────────────────────────────────────────────────
def query_data(question: str) -> dict:
    """
    Convert natural language question → SQL → execute → return result.

    Features:
      • Auto-corrects quarter='Annual' → quarter='FY' (LLM hallucination fix)
      • Auto-expands FY queries to include Q1/Q2/Q3/Q4/FY breakdown rows
      • Falls back to a broadened SQL automatically when 0 rows returned

    Returns dict with keys:
        result      : list of row dicts, or scalar if single cell
        columns     : column name list
        row_count   : number of rows
        sql         : executed SQL (for trace logging)
        source      : human-readable source reference
        error       : only present on failure
    """
    # ── Step 1: generate SQL ──────────────────────────────────────────────────
    try:
        sql = _generate_sql(question)
    except Exception as e:
        return {"result": None, "columns": [], "row_count": 0,
                "sql": "", "source": "financials.db",
                "error": f"SQL generation failed: {e}"}

    # ── Step 2: schema boundary check ────────────────────────────────────────
    if "NOT_IN_SCHEMA" in sql:
        return {"result": None, "columns": [], "row_count": 0,
                "sql": sql, "source": "financials.db",
                "error": "Question cannot be answered from structured data. "
                         "Try search_docs or web_search instead."}

    # ── Step 3: FY expansion (inject quarterly breakdown) ────────────────────
    sql = _expand_fy_sql(sql)

    # ── Step 3b: Ensure metrics are selected ─────────────────────────────────
    sql = _ensure_metrics_selected(sql)

    # ── Step 4: execute ───────────────────────────────────────────────────────
    try:
        rows, cols = _execute_sql(sql)
    except (sqlite3.Error, FileNotFoundError) as e:
        return {"result": None, "columns": [], "row_count": 0,
                "sql": sql, "source": "financials.db", "error": str(e)}

    # ── Step 5: fallback on 0 rows ────────────────────────────────────────────
    if not rows:
        fallback = _fallback_sql(sql, question)
        if fallback:
            print(f"  ↩ 0 rows — retrying with broadened SQL", file=sys.stderr)
            try:
                rows, cols = _execute_sql(fallback)
                if rows:
                    sql = fallback + "  -- [fallback: broadened query]"
            except Exception:
                pass  # ignore fallback errors; original empty result returned below

    # ── Step 6: return result ─────────────────────────────────────────────────
    if not rows:
        return {"result": [], "columns": cols if cols else [], "row_count": 0,
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