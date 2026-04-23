# Agentic RAG — Indian IT Company Financials

A fully autonomous LLM reasoning agent that answers complex questions about company financials by integrating three data sources:

- **Unstructured Documents** (markdown annual reports for Infosys, Accenture, Cognizant — FY22–FY25)
- **Structured Financials** (SQLite database with metrics: revenue, operating margin, headcount, EPS)
- **Live Web Search** (Tavily API for real-time stock prices, news, analyst ratings)

The agent implements a complete reasoning loop: **query rewrite** → **gate check** → **planning** → **tool selection** → **execution** → **sufficiency check** → **answer composition** (max 8 steps).

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/ThiruEigen7/Agentic-Rag---Prodapt.git
cd Agentic-Rag---Prodapt
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Fill in your `.env` file:
```ini
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_MODEL_FAST=llama-3.1-8b-instant
TAVILY_API_KEY=your_tavily_api_key_here
```

**Get Free API Keys:**
- **Groq** (LLM): https://console.groq.com/ (free tier: 30 RPM)
- **Tavily** (Web Search): https://tavily.com/ (free tier: 1000 calls/month)

### 3. Build Indexes (First Time Only)

```bash
# Embed unstructured markdown docs and build FAISS index
python scripts/ingest_unstructure.py

# Ingest structured financial data into SQLite (if CSV provided)
python scripts/ingest_structure.py
```

This creates:
- `indexes/faiss.index` — vector index for semantic search
- `indexes/chunks.pkl` — chunk metadata (text, source, company, section)
- `data/structure/financials.db` — SQLite database with company financials

---

## Usage

### Run Individual Tools

#### Search Documents (Semantic Search)
```bash
# Search all companies (default)
python tools/search_docs.py "What is Infosys AI strategy?"

# Search single company
python tools/search_docs.py "revenue trends" --company Infosys

# Balanced search (compare multiple companies)
python tools/search_docs.py "operating margin improvement" --balanced
```

#### Query Structured Data
```bash
# Natural language query (LLM converts to SQL)
python tools/query_data.py "What was Infosys revenue in FY24?"
python tools/query_data.py "Compare operating margins of all companies in FY24"

# Direct SQL query (bypasses LLM, useful for debugging)
python tools/query_data.py --sql "SELECT company, fiscal_year, revenue_bn_USD FROM financials WHERE fiscal_year='FY24'"
```

#### Search Web
```bash
python tools/web_search.py "Infosys stock price today"
python tools/web_search.py "Accenture Q1 FY25 earnings"
```

### Run the Agent

The agent automatically selects tools based on the question:

```bash
# Simple fact queries (1-2 steps)
python agent.py "What was Infosys revenue in FY24?"

# Comparison queries (2-4 steps)
python agent.py "compare operating margin of infosys and cognizant"

# Multi-source queries (3-5 steps)
python agent.py "What is Infosys current stock price and how does it compare to their FY24 revenue?"

# Verbose trace output (shows all steps)
python agent.py "compare revenue trends across all three companies" --trace
```

---

## Architecture

### Data Flow
```
User Question
    ↓
① Query Rewrite (clean & normalize)
    ↓
② Gate Check (trivial / refuse / proceed)
    ↓
③ Planning (decide tools & order)
    ↓
④ Tool Selection & Execution (max 8 iterations)
    ├─ search_docs (unstructured, qualitative)
    ├─ query_data (structured, numeric)
    └─ web_search (live, real-time)
    ↓
⑤ Sufficiency Check (have enough context?)
    ↓
⑥ Answer Composition (with citations)
    ↓
Final Answer + Trace
```

### Available Data

| Source | Tool | Companies | Years | Data Type |
|--------|------|-----------|-------|-----------|
| Markdown Reports | `search_docs` | Infosys, Accenture, Cognizant | FY22–FY25 | MD&A, strategy, text |
| SQLite DB | `query_data` | Infosys, Accenture, Cognizant | FY22–FY25 | Revenue, margin, headcount, EPS |
| Web API | `web_search` | Any company | Real-time | Prices, news, ratings |

**Database Schema** (`financials.db`):
```sql
CREATE TABLE financials (
    company TEXT,              -- 'Infosys', 'Accenture', 'Cognizant'
    fiscal_year TEXT,          -- 'FY22', 'FY23', 'FY24', 'FY25'
    quarter TEXT,              -- 'Q1', 'Q2', 'Q3', 'Q4', 'Annual'
    revenue_bn_USD REAL,       -- Revenue in billion USD
    op_margin_pct REAL,        -- Operating margin as percentage
    headcount INTEGER,         -- Total employees
    epsusd REAL,               -- Earnings per share
    source_link TEXT,          -- Source URL
    notes TEXT                 -- Additional notes
);
```

---

## Routing Rules (Agent Logic)

The agent uses these rules to decide which tool to use:

| Question Type | Tool | Reason |
|---------------|------|--------|
| "What was Infosys FY24 revenue?" | `query_data` | Numeric historical data |
| "Why did margins improve?" | `search_docs` | Qualitative explanation from reports |
| "Compare all three companies" | `query_data` + `search_docs` | Multiple companies, may need both |
| "Current stock price" | `web_search` | Real-time data, not in historical DB |
| "What is EPS mean?" | LLM (direct) | General knowledge, no tools needed |
| "Should I buy Infosys?" | Refuse | Investment advice not supported |
| "Will revenue grow in FY27?" | Refuse | Prediction beyond available data |

---

## Project Structure

```
Agentic-Rag---Prodapt/
├── agent.py                    # Main agent loop (752 lines, 8-step max)
├── tools/
│   ├── search_docs.py          # Semantic search over FAISS index
│   ├── query_data.py           # Natural language → SQL on SQLite
│   ├── web_search.py           # Tavily API wrapper
│   └── __init__.py
├── scripts/
│   ├── ingest_unstructure.py   # Embed markdown docs → FAISS index
│   ├── ingest_structure.py     # Excel/CSV → SQLite
│   └── patch_chunks.py         # Utility: repair chunk metadata
├── data/
│   ├── unstructure/            # Input: markdown annual reports
│   │   ├── infosys-fy25.md
│   │   ├── accenture-fy25.md
│   │   └── cts-fy25.md (Cognizant)
│   └── structure/              # Output: SQLite database
│       └── financials.db
├── indexes/
│   ├── faiss.index             # FAISS vector index
│   └── chunks.pkl              # Chunk metadata + embeddings
├── traces/
│   └── YYYYMMDD_HHMMSS.json    # Execution traces (auto-generated)
├── requirements.txt            # Python dependencies
├── .env                        # API keys (git-ignored)
├── .env.example                # Template for .env
└── README.md                   # This file
```

---

## Dependencies & LLM Stack

### Python Packages (11 total)

| Package | Purpose | Used By |
|---------|---------|---------|
| `groq` | LLM API client (Groq) | agent.py, query_data.py |
| `sentence-transformers` | Embedding model (768-dim) | search_docs.py |
| `faiss-cpu` | Vector similarity search | search_docs.py, scripts |
| `tavily-python` | Web search API | web_search.py |
| `numpy` | Numerical computing | search_docs.py, scripts |
| `pandas` | Data manipulation | query_data.py, scripts |
| `python-dotenv` | Environment variables | agent.py, tools |
| `tqdm` | Progress bars | scripts/ingest_unstructure.py |
| `openpyxl` | Excel support | scripts/ingest_structure.py |
| `sqlite3` | SQLite driver | query_data.py (built-in) |
| `requests` | HTTP support | agent.py |

### Embedding Model
- **Model**: `nomic-ai/nomic-embed-text-v1.5`
- **Dimensions**: 768
- **Type**: Sentence-level embeddings
- **Source**: HuggingFace (auto-downloaded on first use)

### LLM Configuration

**Primary Model** (best quality):
- `llama-3.3-70b-versatile` (Groq: 30 RPM free)
- Used for: gate checks, planning, sufficiency checks, answer composition

**Fallback Model** (rate-limited):
- `llama-3.1-8b-instant` (Groq: higher limits)
- Automatically used when primary model rate-limits
- Used for: SQL generation in query_data.py

**Retry Logic**:
- Exponential backoff: 10s, 20s, 30s between retries
- Max 3 attempts per request
- Automatic model fallback on 429 (rate limit)
