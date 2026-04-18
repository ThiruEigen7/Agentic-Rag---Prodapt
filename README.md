# Agentic RAG — Public Company Financials

An LLM reasoning agent that answers questions over three data sources:
- **Annual report PDFs** (Infosys, TCS, Wipro — FY21–FY24)
- **Structured financials table** (CSV → SQLite, 12 rows × 7 columns)
- **Live web** (Tavily API for current prices, news, analyst ratings)

The agent loop is written from scratch in ~70 lines of Python. No `initialize_agent()` wrapper is used.

---

## Setup

```bash
git clone https://github.com/ThiruEigen7/Agentic-Rag---Prodapt
cd Agentic-Rag---Prodapt
pip install -r requirements.txt
cp .env.example .env          # fill in ANTHROPIC_API_KEY, OPENAI_API_KEY, TAVILY_API_KEY
```

Add annual report PDFs to `data/pdfs/`:
```
data/pdfs/infosys_ar_fy24.pdf
data/pdfs/tcs_ar_fy24.pdf
data/pdfs/wipro_ar_fy24.pdf
```

Build all indexes in one command:
```bash
python scripts/setup.py
```

---

## Running each tool individually

```bash
# search_docs — semantic search over PDFs
python tools/search_docs.py "what reason did Infosys give for margin improvement"

# query_data — structured financials
python tools/query_data.py "What was TCS operating margin in FY24?"
python tools/query_data.py "Compare headcount across all companies FY24"

# web_search — live web
python tools/web_search.py "Infosys stock price today"
python tools/web_search.py "TCS Q1 FY25 results"
```

---

## Running the agent

```bash
python agent/loop.py --question "What was Infosys operating margin in FY24?"
python agent/loop.py --question "How did TCS explain its margin improvement?"
python agent/loop.py --question "Compare margins of Infosys and TCS in FY24 and what drove each?"
```

## Running the evaluation set

```bash
python eval/run_eval.py
```

---

## Known failure modes

1. **Temporal ambiguity on "latest."** Questions with "latest" or "current" sometimes route to `query_data` (returning FY24 CSV data) when the user intended live web data. Workaround: rephrase as "current stock price" vs "FY24 revenue."

2. **Low-score retrieval on sparse topics.** If search_docs returns chunks with score < 0.55, the LLM may infer rather than cite. The trace logs scores so you can spot this.

3. **FY21 Wipro headcount is NULL** in the CSV (not reported in that year's AR). The agent correctly says "not in data" but the message is terse.

---
