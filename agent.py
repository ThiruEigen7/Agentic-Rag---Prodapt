"""
agent.py
--------
Agentic RAG — main entry point.

Implements the loop from the architecture diagram:
    ① Query rewrite         — clean the user question for better retrieval
    ② LLM gate              — does this need tools, or answer directly, or refuse?
    ③ Planning              — which tool(s) and in what order?
    ④ Tool execution        — call the chosen tool
    ⑤ LLM relevance check  — is the retrieved content sufficient?
    ⑥ Answer / loop back   — compose final answer OR retry (max 8 steps)

Hard cap: 8 tool calls max. Enforced as a raised exception — not a soft check.
Every run produces a structured trace in traces/YYYYMMDD_HHMMSS.json

Run:
    python agent.py "What was Infosys revenue in FY24?"
    python agent.py "Compare operating margins of all three companies in FY24"
    python agent.py "What is Accenture current stock price?"
    python agent.py --trace "Which company had highest headcount?"
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict

from groq import Groq
from dotenv import load_dotenv

from tools.search_docs import search_docs, detect_company_in_query
from tools.query_data import query_data
from tools.web_search import web_search

load_dotenv()

# ── config ────────────────────────────────────────────────────────────────────
MAX_STEPS      = 8        # hard cap — never change this
TRACES_DIR     = Path("traces")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
# llama-3.3-70b-versatile : best quality, ~30 RPM free
# llama-3.1-8b-instant    : faster, higher rate limits — use as fallback
GROQ_MODEL     = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_MODEL_FAST= os.getenv("GROQ_MODEL_FAST", "llama-3.1-8b-instant")

COMPANY_KEYWORDS = {
    "infosys"    : "Infosys",
    "accenture"  : "Accenture",
    "cognizant"  : "Cognizant",
    "cts"        : "Cognizant",
}


# ── single source of truth about what data exists ─────────────────────────────
# Update this block whenever your corpus or DB changes.
# Every prompt references DATA_CONTEXT so the LLM always knows what's available.
DATA_CONTEXT = """
AVAILABLE TOOLS AND SCHEMAS:

1. search_docs: Semantic search over unstructured documents.
   Description: Use this tool to perform semantic search over annual reports and textual documents. It is ideal for answering "why", "how", or strategy-related questions by retrieving qualitative context from MD&A, CEO letters, and risk factors. Use this when you need insights that are not captured in structured numeric tables.
   Inputs: Natural language query string.
   Outputs: Top-3 relevant text chunks with source filename and section heading (as page number).

2. query_data: Query the structured financial / stats table.
   Description: Use this tool to retrieve precise numeric data, metrics, and financial statistics from a structured SQLite database. It can handle SQL queries or natural language questions about revenue, margins, headcount, and EPS. Always use this for factual, quantitative lookups for known fiscal years (FY22-FY25).
   Inputs: A pandas or SQL query, or a natural-language question about the data.
   Outputs: A table or scalar value with column names and row count.

3. web_search: Search the live web for recent information.
   Description: Use this tool to search the live internet for real-time information, current stock prices, analyst ratings, or news events occurring after FY25. Keep the search query extremely concise. Do NOT use this for historical financial data that is already available in search_docs or query_data.
   Inputs: A short search query string (under 10 words).
   Outputs: Top-3 result snippets with URL and publication date.

ROUTING RULES:
- FY22, FY23, FY24, FY25 EXISTS in search_docs and query_data — always use those first.
- Only use web_search for live prices, news, or data beyond FY25.
- "Latest" or "current" financials = FY25 (in database).
"""

# ── exceptions ────────────────────────────────────────────────────────────────
class HardCapExceeded(Exception):
    """Raised when agent hits the 8-step limit. Never caught silently."""
    pass

class AgentRefusal(Exception):
    """Raised when question should be declined without tool use."""
    pass


# ── data classes ──────────────────────────────────────────────────────────────
@dataclass
class ToolCall:
    step       : int
    tool       : str        # "search_docs" | "query_data" | "web_search"
    input      : str        # what was sent to the tool
    output     : dict       # raw tool return dict
    latency_ms : int        # milliseconds
    sufficient : bool       # did LLM say this is enough to answer?

@dataclass
class AgentResult:
    question       : str
    rewritten_q    : str
    plan           : str
    tool_calls     : list[ToolCall] = field(default_factory=list)
    final_answer   : str  = ""
    citations      : list[str] = field(default_factory=list)
    steps_used     : int  = 0
    total_time_ms  : int  = 0
    status         : str  = "ok"   # "ok" | "refused" | "cap_exceeded" | "error"
    refusal_reason : str  = ""


# ── Groq client ──────────────────────────────────────────────────────────────
def _get_groq() -> Groq:
    if not GROQ_API_KEY:
        raise EnvironmentError(
            "GROQ_API_KEY not set. Add it to .env file.\n"
            "Get free key (no credit card): https://console.groq.com/"
        )
    return Groq(api_key=GROQ_API_KEY)


def _llm(prompt: str, max_retries: int = 3, fast: bool = False) -> str:
    """
    Groq API call with automatic fallback to faster model on rate limit.
    
    fast=False : uses GROQ_MODEL (llama-3.3-70b-versatile) — best quality
    fast=True  : uses GROQ_MODEL_FAST (llama-3.1-8b-instant) — higher limits
    
    On 429 rate limit: automatically retries with fast model before giving up.
    """
    client = _get_groq()
    model  = GROQ_MODEL_FAST if fast else GROQ_MODEL

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model       = model,
                messages    = [{"role": "user", "content": prompt}],
                temperature = 0.3,
                max_tokens  = 500,
            )
            return resp.choices[0].message.content.strip()

        except Exception as e:
            err = str(e)
            is_rate = "429" in err or "rate_limit" in err.lower()

            if is_rate and not fast:
                # step down to fast model — no wait needed
                print(f"  ⚡ Rate limit on {model} → switching to {GROQ_MODEL_FAST}", file=sys.stderr)
                model = GROQ_MODEL_FAST
                continue

            elif is_rate and attempt < max_retries - 1:
                wait = 10 * (attempt + 1)   # groq resets per minute: wait 10-20s
                print(f"  ⏳ Rate limit. Waiting {wait}s ({attempt+1}/{max_retries})", file=sys.stderr)
                time.sleep(wait)
                continue

            else:
                raise

    raise Exception(f"Groq failed after {max_retries} attempts on {model}")


def _llm_json(prompt: str, max_retries: int = 3) -> dict:
    """
    Groq call expecting JSON output.
    Strips markdown fences. Falls back to empty dict on parse failure.
    """
    raw = _llm(prompt, max_retries=max_retries)

    # strip ```json ... ``` fences
    if "```" in raw:
        lines = [l for l in raw.split("\n") if not l.strip().startswith("```")]
        raw   = "\n".join(lines).strip()

    # extract JSON object if model added surrounding text
    if "{" in raw:
        try:
            start = raw.index("{")
            end   = raw.rindex("}") + 1
            raw   = raw[start:end]
        except ValueError:
            pass

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


# ── pre-loop step ①: query rewrite ───────────────────────────────────────────
def rewrite_query(question: str) -> str:
    """
    Rewrite vague/conversational question into a clean retrieval query.
    "how did they do on margins last year" → "Infosys TCS Cognizant operating margin FY24"
    Does NOT count toward the 8-step cap.
    """
    prompt = f"""Rewrite the user's question into a clear, specific retrieval query.
Fix vague pronouns, relative time references, and ambiguous terms.
Keep company names and fiscal years explicit.
Output ONLY the rewritten query — no explanation.

Companies in scope: Infosys, Accenture, Cognizant (also called CTS)
Fiscal years in scope: FY21, FY22, FY23, FY24, FY25

User question: {question}
Rewritten query:"""
    try:
        rewritten = _llm(prompt)
        # Normalize common company aliases (deterministic post-processing)
        return _normalize_company_aliases(rewritten)
    except Exception:
        return _normalize_company_aliases(question)   # fallback: use original if rewrite fails


def _normalize_company_aliases(text: str) -> str:
    """Normalize common company aliases to canonical names.

    Maps:
      - 'TCS' or 'CTS' -> 'Cognizant'
      - preserves case for other words
    This is a simple, deterministic replacement to avoid LLM rewrite inconsistencies.
    """
    import re
    if not text:
        return text
    # replace standalone TCS or CTS (case-insensitive) with Cognizant
    text = re.sub(r"\bTCS\b", "Cognizant", text, flags=re.IGNORECASE)
    text = re.sub(r"\bCTS\b", "Cognizant", text, flags=re.IGNORECASE)
    return text


# ── pre-loop step ②: gate — trivial / refuse / proceed ───────────────────────
GATE_PROMPT = """You are a classifier for a financial data agent.
Today's date is {current_date}.

Classify this question into exactly one of three categories:

TRIVIAL  — answerable directly with no data lookup (math, general knowledge, greetings, current date/time)
           e.g. "what is 2+2", "what does EPS mean", "hello", "what is today's date"

REFUSE   — must be declined: investment advice ("should I buy X"), predictions beyond
           available data (FY26+), or harmful/irrelevant requests
           e.g. "should I invest in Accenture", "will revenue grow in FY27"

PROCEED  — needs tool use to answer from company data (everything else)
           e.g. ANY question about Infosys / Accenture / Cognizant facts, numbers,
           strategy, comparisons, trends for FY22-FY25

{DATA_CONTEXT}

CRITICAL: The following ALWAYS map to PROCEED:
  - Any FY22 / FY23 / FY24 / FY25 question → data EXISTS, use tools
  - "Compare X and Y" questions → PROCEED
  - "What is the operating margin / revenue / headcount" → PROCEED
  - "Latest" or "current" financials → PROCEED (FY25 is in the database)
  - "Current stock price / analyst rating" → PROCEED (use web_search)

Respond with ONLY valid JSON — no markdown, no explanation:
{{"decision": "TRIVIAL"|"REFUSE"|"PROCEED", "reason": "one sentence"}}

Question: {question}"""


def gate_check(question: str) -> tuple[str, str]:
    """
    Returns (decision, reason).
    decision: "TRIVIAL" | "REFUSE" | "PROCEED"
    """
    current_date = datetime.now().strftime("%A, %B %d, %Y")
    result = _llm_json(GATE_PROMPT.format(
        DATA_CONTEXT=DATA_CONTEXT, 
        question=question, 
        current_date=current_date
    ))
    decision = result.get("decision", "PROCEED").upper()
    reason   = result.get("reason", "")
    if decision not in ("TRIVIAL", "REFUSE", "PROCEED"):
        decision = "PROCEED"
    return decision, reason



# ── pre-loop step ③: planning ─────────────────────────────────────────────────
PLAN_PROMPT = """You are planning tool calls for a financial data agent.

{DATA_CONTEXT}

TOOL SELECTION GUIDE:
  search_docs  → use for WHY / HOW / WHAT STRATEGY questions
                 e.g. "why did margin improve", "what did CEO say", "AI strategy"
                 Searches: Infosys FY22-FY25, Accenture FY22-FY25, Cognizant FY22-FY25 annual reports

  query_data   → use for NUMBERS / METRICS / COMPARISONS questions
                 e.g. "revenue in FY25", "compare margins", "headcount trend"
                 Database has: revenue_bn_USD, op_margin_pct, headcount, epsusd for FY22-FY25

  web_search   → use ONLY for live/current data not in database
                 e.g. "current stock price", "today's analyst rating", "news this week"
                 Do NOT use for any FY22-FY25 historical data — that's in the database

MULTI-TOOL STRATEGY:
  "Compare margins AND explain why" → query_data first (numbers), then search_docs (explanation)
  "Revenue trend AND current price"  → query_data first (history), then web_search (live price)
  "What did Infosys say about AI"    → search_docs only (qualitative)
  "Operating margin FY24 vs FY25"    → query_data only (both years in database)

Write a 1-2 sentence plan, then output ONLY valid JSON:
{{"plan": "...", "first_tool": "search_docs"|"query_data"|"web_search", "first_input": "short specific query string"}}

Question: {question}"""

def make_plan(question: str) -> tuple[str, str, str]:
    """
    Returns (plan_text, first_tool, first_input).

    Priority overrides (deterministic, before LLM):
      1. Live/current data keywords   → web_search first
      2. Multi-company numeric comparison for known FY years → query_data with all-company input
      3. Everything else              → LLM decides
    """
    import re
    q_lower = question.lower()

    # ── Override 1: live/current data ────────────────────────────────────────
    needs_live_data = any(w in q_lower for w in [
        "current", "today", "live", "stock price", "analyst rating",
        "recent", "news", "latest", "right now"
    ])

    if needs_live_data:
        company_match = None
        for company in ["Infosys", "Accenture", "Cognizant", "CTS", "TCS"]:
            if company.lower() in q_lower:
                company_match = company
                break

        if "stock price" in q_lower or "price" in q_lower:
            web_input = f"{company_match or 'IT company'} stock price"
        elif "analyst" in q_lower or "rating" in q_lower:
            web_input = f"{company_match or 'IT company'} analyst ratings"
        else:
            web_input = f"{company_match or 'IT company'} news"

        if any(w in q_lower for w in ["compare", "versus", "vs", "revenue", "margin", "fy24", "fy25"]):
            plan = "First get current data via web_search, then retrieve historical metrics"
        else:
            plan = f"Get current/live data about {company_match or 'the company'}"

        return plan, "web_search", web_input

    # ── Override 2: multi-company numeric comparison with explicit FY year ───
    # e.g. "Compare margins of all three in FY24" → query_data with all-company input
    is_multi  = _is_multi_company_query(question)
    is_numeric = any(w in q_lower for w in ["revenue", "margin", "headcount", "eps"])
    fy_matches = re.findall(r'fy\d{1,2}', q_lower)

    if is_multi and is_numeric and fy_matches:
        fy_years = ", ".join(m.upper() for m in fy_matches)

        # Which metric?
        metric_map = {
            "revenue": "revenue", "margin": "operating margin",
            "headcount": "headcount", "eps": "EPS",
        }
        metric = "financials"
        for kw, m in metric_map.items():
            if kw in q_lower:
                metric = m
                break

        # Which companies explicitly named? Fall back to all three.
        company_names: list[str] = []
        for c in ["Infosys", "Accenture", "Cognizant", "CTS", "TCS"]:
            if c.lower() in q_lower:
                canonical = "Cognizant" if c.lower() in ("tcs", "cts") else c
                if canonical not in company_names:
                    company_names.append(canonical)
        if not company_names:
            company_names = ["Infosys", "Accenture", "Cognizant"]

        companies_str = ", ".join(company_names)
        first_input   = f"{companies_str} {metric} for {fy_years}"
        plan = (f"Fetch {metric} for {companies_str} ({fy_years}) from query_data, "
                f"then enrich with search_docs for qualitative context.")
        return plan, "query_data", first_input

    # ── Override 3: LLM decides ───────────────────────────────────────────────
    result     = _llm_json(PLAN_PROMPT.format(DATA_CONTEXT=DATA_CONTEXT, question=question))
    plan       = result.get("plan", "")
    first_tool = result.get("first_tool", "query_data")
    first_input = result.get("first_input", question)
    if first_tool not in ("search_docs", "query_data", "web_search"):
        first_tool = "query_data"
    return plan, first_tool, first_input


# ── loop step ④: tool selection ───────────────────────────────────────────────
TOOL_SELECT_PROMPT = """You are deciding the next tool call for a financial data agent.

{DATA_CONTEXT}

DECISION RULES — follow strictly:
  query_data   → numbers, metrics, comparisons, trends (revenue, margin, headcount, EPS)
                 ALWAYS try this first for any FY22-FY25 numeric question
  search_docs  → qualitative text: strategy, explanations, CEO commentary, MD&A
                 Use after query_data when answer also needs "why" or "how"
  web_search   → ONLY for: current stock price, live analyst rating, news beyond FY25
                 Do NOT use web_search for FY22-FY25 data — it is in the database
  DONE         → context already contains a complete answer — stop calling tools

QUESTION: {question}

CONTEXT COLLECTED SO FAR:
{context}

What is the NEXT best tool to call?
- If context has the answer already → {{"tool": "DONE", "input": ""}}
- If context has numbers but no explanation → {{"tool": "search_docs", "input": "..."}}
- If context is empty and question needs numbers → {{"tool": "query_data", "input": "..."}}
- If question needs live/current data → {{"tool": "web_search", "input": "short query"}}

Output ONLY valid JSON — no markdown, no explanation:
{{"tool": "search_docs"|"query_data"|"web_search"|"DONE", "input": "exact query string"}}"""

def select_tool(question: str, context: str) -> tuple[str, str]:
    """
    Returns (tool_name, tool_input).
    tool_name can be "DONE" meaning agent has enough to answer.
    
    KEY FEATURES:
    1. If question requires live/current data → force web_search
    2. When query_data returned error or no results for requested company → fallback to search_docs
    3. When user asks for FY data → split into Q1,Q2,Q3,Q4,Annual for full picture
    """
    import re
    q_lower = question.lower()
    
    # ── Rule 1: Force web_search for live/current data ───────────────────────
    needs_live_data = any(w in q_lower for w in [
        "current", "today", "live", "stock price", "analyst rating",
        "recent", "news", "latest", "right now"
    ])
    has_web_results = "[Step" in context and "web_search" in context and "Result:" in context
    
    if needs_live_data and not has_web_results:
        company_match = None
        for company in ["Infosys", "Accenture", "Cognizant", "CTS"]:
            if company.lower() in q_lower:
                company_match = company
                break
        
        if company_match:
            if "stock price" in q_lower or "price" in q_lower:
                web_input = f"{company_match} stock price"
            elif "analyst" in q_lower or "rating" in q_lower:
                web_input = f"{company_match} analyst rating"
            elif "news" in q_lower:
                web_input = f"{company_match} news"
            else:
                web_input = f"{company_match} latest"
        else:
            web_input = "IT company latest news"
        
        return "web_search", web_input
    
    # ── Rule 2: query_data error → fallback to search_docs ──────────────────
    if ("NOT_IN_SCHEMA" in context or "ERROR" in context) and "query_data" in context:
        return "search_docs", question

    # ── Rule 3: multi-company comparison → ensure BOTH tools are called ──────
    # "SQL:" tag is always emitted by _format_context for query_data results.
    # This is more reliable than checking for "Row" (capital R never appears).
    has_query_data  = "SQL:" in context and "query_data" in context
    has_search_docs = "search_docs" in context
    is_comparison   = any(w in q_lower for w in [
        "compare", "versus", "vs", "differ", "all", "between", "both", "companies"
    ])
    also_asks_why = any(w in q_lower for w in ["why", "reason", "explain", "how"])

    if has_query_data and not has_search_docs and (is_comparison or also_asks_why):
        return "search_docs", question

    # ── Rule 4: de-duplication guard ─────────────────────────────────────────
    # Parse the last tool+input from context to avoid spinning on the same call.
    last_tool_in_ctx  = None
    last_input_in_ctx = None
    for line in reversed(context.splitlines()):
        stripped = line.strip()
        if stripped.startswith("[Step") and "—" in stripped:
            last_tool_in_ctx = stripped.split("—")[-1].strip().rstrip("]")
        elif stripped.startswith("Input:"):
            last_input_in_ctx = stripped[len("Input:"):].strip()
            break

    # ── Rule 5: LLM decides, then enhance with FY + company context ──────────
    result     = _llm_json(TOOL_SELECT_PROMPT.format(
        DATA_CONTEXT=DATA_CONTEXT, question=question, context=context
    ))
    tool       = result.get("tool", "query_data")
    tool_input = result.get("input", question)

    # If LLM is about to repeat the exact same call as last time → advance
    if tool == last_tool_in_ctx and tool_input == last_input_in_ctx:
        if tool == "query_data" and not has_search_docs:
            tool, tool_input = "search_docs", question
        elif tool == "search_docs" and has_query_data:
            tool, tool_input = "DONE", ""
        else:
            tool, tool_input = "DONE", ""

    # Build a better query_data input covering all companies + FY years
    if tool == "query_data":
        fy_matches = re.findall(r'fy\d{1,2}', q_lower)
        if fy_matches:
            fy_years = [m.upper() for m in fy_matches]
            company_names: list[str] = []
            for company in ["Infosys", "Accenture", "Cognizant", "CTS", "TCS"]:
                if company.lower() in q_lower:
                    canonical = "Cognizant" if company.lower() in ("tcs", "cts") else company
                    if canonical not in company_names:
                        company_names.append(canonical)
            if not company_names and _is_multi_company_query(question):
                company_names = ["Infosys", "Accenture", "Cognizant"]
            metric_map = {
                "revenue": "revenue", "margin": "operating margin",
                "headcount": "headcount", "eps": "EPS", "employee": "headcount",
            }
            metric_type = "financials"
            for kw, metric in metric_map.items():
                if kw in q_lower:
                    metric_type = metric
                    break
            if company_names:
                tool_input = f"{', '.join(company_names)} {metric_type} for {', '.join(fy_years)}"

    if tool not in ("search_docs", "query_data", "web_search", "DONE"):
        tool = "query_data"
    return tool, tool_input


# ── helpers for multi-company detection ──────────────────────────────────────
_MULTI_COMPANY_KEYWORDS = [
    "compare", "all", "three", "each", "versus", "vs", "both",
    "companies", "all three", "every company", "which company",
    "infosys and accenture", "infosys and cognizant", "accenture and cognizant",
]

def _is_multi_company_query(text: str) -> bool:
    """Return True if the text appears to span multiple companies."""
    t = text.lower()
    # Check for multi-company keywords
    if any(kw in t for kw in _MULTI_COMPANY_KEYWORDS):
        return True
    # Check for 2+ company names explicitly mentioned
    mentioned = sum(
        1 for name in ["infosys", "accenture", "cognizant", "cts", "tcs"]
        if name in t
    )
    return mentioned >= 2


# ── loop step ④: execute tool ─────────────────────────────────────────────────
def execute_tool(tool: str, tool_input: str, question: str) -> dict:
    """
    Call the selected tool and return its raw result dict.
    Errors are caught and returned as structured error dicts — never crash.

    For search_docs, routing logic:
      - multi-company query (question OR tool_input) → balanced=True (top_k per company)
      - single company mentioned in tool_input       → company= filter
      - otherwise                                    → standard full-index search
    """
    try:
        if tool == "search_docs":
            # Multi-company check: look at BOTH the question and the tool_input
            is_multi = _is_multi_company_query(question) or _is_multi_company_query(tool_input)

            if is_multi:
                # Balanced retrieval: top_k per company so no company is drowned out
                return search_docs(tool_input, top_k=2, balanced=True)

            # Single-company: detect from tool_input first (more specific), then question
            company = detect_company_in_query(tool_input) or detect_company_in_query(question)
            if company:
                return search_docs(tool_input, top_k=4, company=company)

            # Fallback: generic full-index search
            return search_docs(tool_input, top_k=4)

        elif tool == "query_data":
            return query_data(tool_input)

        elif tool == "web_search":
            return web_search(tool_input)

        else:
            return {"error": f"Unknown tool: {tool}"}

    except Exception as e:
        return {"error": f"Tool '{tool}' raised an exception: {e}"}


# ── loop step ⑤: relevance / sufficiency check ─────────────────────────────
SUFFICIENCY_PROMPT = """You are checking whether enough information has been collected to answer a question.

QUESTION: {question}

CONTEXT COLLECTED:
{context}

Is the context sufficient to write a complete, accurate, cited answer?

Sufficiency rules:
  SUFFICIENT (true) when:
    - All numbers/metrics asked for are present in context
    - Explanation text is present if the question asks "why" or "how"
    - At least one result was returned (not all errors/empty)

  NOT SUFFICIENT (false) when:
    - Context is empty or all tool calls returned errors
    - Question asks for multiple things and only some are answered
    - Numbers are present but question also needs qualitative explanation (or vice versa)
    - A comparison question is only half answered (one company missing)

  IMPORTANT: Do NOT say insufficient just because you want more context.
  If the core question is answered with cited data → sufficient = true.

Respond with ONLY valid JSON:
{{"sufficient": true|false, "reason": "one sentence explaining your decision"}}"""

def is_sufficient(question: str, context: str) -> tuple[bool, str]:
    """
    Returns (sufficient, reason).
    
    Heuristic: For queries that need both live data and historical data:
    - If context has web_search results AND query_data results → ALWAYS sufficient
    - For pure historical queries with comparison → sufficient if company data found
    - If query_data failed/errored and search_docs was called → sufficient for qualitative questions
    - Special: partial company data (e.g., Infosys but missing TCS) + search_docs → sufficient for comparison
    """
    q_lower = question.lower()
    
    # Check if question requires live/current data
    needs_live_data = any(w in q_lower for w in [
        "current", "today", "live", "stock price", "analyst rating",
        "recent", "news", "latest", "right now"
    ])
    
    # Check if question also asks for historical/comparison data
    also_asks_historical = any(w in q_lower for w in [
        "compare", "versus", "vs", "revenue", "margin", "fy", "fiscal year",
        "trend", "differ", "between", "how does", "against", "reason", "why"
    ])

    # Detect how many companies the question is asking about
    requested_companies = sum(
        1 for name in ["infosys", "accenture", "cognizant", "tcs", "cts"]
        if name in q_lower
    )
    is_multi_company_q = _is_multi_company_query(question)
    num_companies_requested = max(requested_companies, 3 if is_multi_company_q and requested_companies == 0 else requested_companies)

    # Check what we have in context
    # Note: "SQL:" tag is reliably emitted for every query_data call in _format_context.
    # "Row" (capital R) never appears in context rows (they are plain dict reprs).
    has_web_results  = "[Step" in context and "web_search"  in context and "Result:" in context
    has_query_data   = "[Step" in context and "query_data"  in context and "SQL:"    in context
    has_search_docs  = "[Step" in context and "search_docs" in context

    # Check if query_data failed with error (missing company, schema issue, etc.)
    query_data_failed = (
        "[Step" in context and "query_data" in context
        and ("ERROR" in context or "NOT_IN_SCHEMA" in context)
    )

    # Count how many distinct companies are present in retrieved context
    companies_in_context = sum(
        1 for name in ["Infosys", "Accenture", "Cognizant"]
        if name in context
    )

    # ── Rule: live data + historical comparison → need BOTH sources ───────────
    if needs_live_data and also_asks_historical:
        if has_web_results and (has_query_data or has_search_docs):
            return True, "Collected both live data (web_search) and historical data"
        elif has_web_results:
            return False, "Need historical/comparison data in addition to current stock price"
        else:
            return False, "Need current stock price data from web_search"

    # ── Rule: live data only ──────────────────────────────────────────────────
    if needs_live_data and not also_asks_historical:
        if has_web_results:
            return True, "Current/live data obtained via web_search"
        else:
            return False, "Question requires current/live data — web_search results needed"

    # ── Rule: query_data failed → qualitative fallback via search_docs ────────
    if query_data_failed and has_search_docs:
        if any(w in q_lower for w in ["reason", "why", "explain", "margin", "guidance"]):
            return True, "Search_docs retrieved qualitative explanation (MD&A, rationale)"

    # ── Rule: multi-company comparison questions ──────────────────────────────
    is_comparison = any(w in q_lower for w in [
        "compare", "versus", "vs", "differ", "all", "between", "both", "companies"
    ])
    is_numeric    = any(w in q_lower for w in ["revenue", "margin", "headcount", "eps", "number"])

    if is_comparison and not needs_live_data:
        if is_multi_company_q:
            # Need all 3 companies' data for a full comparison
            if companies_in_context >= 3 and (has_query_data or has_search_docs):
                return True, f"All 3 companies present in context — comparison is complete"
            elif companies_in_context >= 2 and has_query_data and has_search_docs:
                # Have 2 companies' structured + qualitative context — good enough
                return True, f"{companies_in_context} companies with both numeric + qualitative data"
            elif companies_in_context >= 1 and has_query_data and has_search_docs:
                # Have some structured + qualitative — let LLM decide
                pass  # fall through to LLM check
            else:
                # Missing either structured or qualitative data
                if has_query_data and not has_search_docs:
                    return False, "Have numbers but still need qualitative context from search_docs"
                elif has_search_docs and not has_query_data:
                    return False, "Have qualitative context but still need numbers from query_data"
                else:
                    return False, "Need data from at least one source"
        elif is_numeric:
            # Single metric comparison — structured data alone is sufficient
            infosys_count   = context.count("'Infosys'")   + context.count('"Infosys"')
            accenture_count = context.count("'Accenture'") + context.count('"Accenture"')
            cognizant_count = context.count("'Cognizant'") + context.count('"Cognizant"')
            total_entries   = infosys_count + accenture_count + cognizant_count
            if total_entries >= 4 or (has_search_docs and companies_in_context >= 1):
                return True, f"Sufficient numeric comparison data ({total_entries} entries)"

    # ── Rule: enough structured data for non-comparison numeric queries ────────
    # "Source:" is emitted once per query_data call in _format_context.
    # "Row" (capital R) never appears in context — rows are plain Python dict reprs.
    if has_query_data and context.count("Source:") >= 1 and companies_in_context >= 1:
        return True, "Sufficient historical data collected"

    # ── Fallback: let LLM decide (guarded — network errors must not crash agent) ──
    try:
        result    = _llm_json(SUFFICIENCY_PROMPT.format(question=question, context=context))
        sufficient = result.get("sufficient", False)
        reason     = result.get("reason", "LLM fallback")
        return bool(sufficient), reason
    except Exception as e:
        # If the LLM call itself fails (rate-limit, connection error, etc.) we
        # conservatively say "not sufficient" so the agent keeps trying with the
        # next tool — unless we already have data from both major sources, in which
        # case we declare sufficient to avoid burning all 8 steps.
        if has_query_data and has_search_docs:
            return True, f"LLM sufficiency check failed ({e}); declaring sufficient (both sources present)"
        if has_query_data and companies_in_context >= 3:
            return True, f"LLM sufficiency check failed ({e}); declaring sufficient (all 3 companies in query_data)"
        return False, f"LLM sufficiency check failed: {e}"


# ── loop step ⑥: answer composition ──────────────────────────────────────────
COMPOSE_PROMPT = """You are composing a final answer from retrieved context.

QUESTION: {question}

CONTEXT (retrieved from tools):
{context}

Write a clear, accurate answer using ONLY information present in the context above.
- Answer the question directly in the first sentence
- Cite every factual claim: mention the source (tool name + file/URL/table)
- Do not introduce any information not in the context
- If numbers, include units (crore INR, %, etc.)
- Keep it concise — 3 to 6 sentences max

Then on a new line write:
CITATIONS: <comma-separated list of sources used>"""

def compose_answer(question: str, context: str) -> tuple[str, list[str]]:
    """
    Returns (answer_text, citations_list).
    Uses increased max_tokens so multi-source / multi-company answers are not truncated.
    """
    client = _get_groq()
    model  = GROQ_MODEL
    prompt = COMPOSE_PROMPT.format(question=question, context=context)
    try:
        resp = client.chat.completions.create(
            model       = model,
            messages    = [{"role": "user", "content": prompt}],
            temperature = 0.3,
            max_tokens  = 1024,   # larger budget for multi-source answers
        )
        raw = resp.choices[0].message.content.strip()
    except Exception:
        # fallback to shared helper (500 tokens)
        raw = _llm(prompt)

    # split answer from citations
    if "CITATIONS:" in raw:
        parts     = raw.split("CITATIONS:", 1)
        answer    = parts[0].strip()
        citations = [c.strip() for c in parts[1].split(",") if c.strip()]
    else:
        answer    = raw.strip()
        citations = []

    return answer, citations


# ── context builder ───────────────────────────────────────────────────────────
def _format_context(tool_calls: list[ToolCall]) -> str:
    """Convert list of tool calls into a readable context string for LLM prompts."""
    if not tool_calls:
        return "(no context collected yet)"

    parts = []
    for tc in tool_calls:
        parts.append(f"[Step {tc.step} — {tc.tool}]")
        parts.append(f"Input: {tc.input}")

        output = tc.output
        if "error" in output:
            parts.append(f"Result: ERROR — {output['error']}")

        elif tc.tool == "search_docs" and "chunks" in output:
            for chunk in output["chunks"]:
                parts.append(
                    f"Result: [{chunk.get('company','')}] "
                    f"{chunk.get('source','')} § {chunk.get('section','')}"
                )
                parts.append(f"  {chunk.get('text','')[:300]}")

        elif tc.tool == "query_data":
            parts.append(f"SQL: {output.get('sql','')}")
            parts.append(f"Source: {output.get('source','')}")
            result = output.get("result")
            if isinstance(result, list):
                # Sort so FY (full-year) rows appear first — most useful for multi-company comparisons.
                # Within each company: FY > Q4 > Q3 > Q2 > Q1.
                _quarter_order = {"FY": 0, "Q4": 1, "Q3": 2, "Q2": 3, "Q1": 4}
                try:
                    sorted_rows = sorted(
                        result,
                        key=lambda r: (
                            r.get("company", ""),
                            _quarter_order.get(r.get("quarter", ""), 9),
                        ),
                    )
                except Exception:
                    sorted_rows = result
                for row in sorted_rows[:20]:   # cap raised to 20 for multi-company results
                    parts.append(f"  Row: {row}")
            else:
                col = output.get("columns", ["value"])[0]
                parts.append(f"  {col} = {result}")

        elif tc.tool == "web_search" and "results" in output:
            for r in output["results"]:
                parts.append(
                    f"Result: {r.get('title','')} "
                    f"({r.get('published_date','unknown date')})"
                )
                parts.append(f"  {r.get('snippet','')[:300]}")
                parts.append(f"  URL: {r.get('url','')}")

        parts.append("")   # blank line between steps

    return "\n".join(parts)


# ── trace writer ──────────────────────────────────────────────────────────────
def _save_trace(result: AgentResult) -> Path:
    """Write structured trace JSON to traces/ directory."""
    TRACES_DIR.mkdir(exist_ok=True)
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_file = TRACES_DIR / f"{timestamp}.json"

    # convert dataclasses to plain dicts for JSON serialisation
    trace = asdict(result)
    with open(trace_file, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2, ensure_ascii=False)

    return trace_file


def _print_trace(result: AgentResult) -> None:
    """Pretty-print the trace to stdout."""
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"Question  : {result.question}")
    print(f"Rewritten : {result.rewritten_q}")
    print(f"Plan      : {result.plan}")
    print(f"Status    : {result.status}")
    print(sep)

    for tc in result.tool_calls:
        print(f"\nStep {tc.step}/{MAX_STEPS} — {tc.tool}")
        print(f"  Input      : {tc.input}")
        print(f"  Latency    : {tc.latency_ms}ms")
        print(f"  Sufficient : {tc.sufficient}")

        output = tc.output
        if "error" in output:
            print(f"  Result     : ERROR — {output['error']}")
        elif tc.tool == "search_docs" and "chunks" in output:
            for c in output["chunks"]:
                print(
                    f"  Result     : [{c.get('company','')}] "
                    f"{c.get('source','')} p.{c.get('page','?')} "
                    f"score={c.get('score',0):.3f}"
                )
        elif tc.tool == "query_data":
            print(f"  SQL        : {output.get('sql','')}")
            print(f"  Source     : {output.get('source','')}")
            res = output.get("result")
            if isinstance(res, list):
                for row in res[:5]:
                    print(f"  Row        : {row}")
            else:
                print(f"  Value      : {res}")
        elif tc.tool == "web_search" and "results" in output:
            for r in output["results"]:
                print(f"  Result     : {r.get('title','')} — {r.get('url','')}")

    print(f"\n{sep}")
    print(f"Final Answer:\n{result.final_answer}")
    print(f"\nCitations: {', '.join(result.citations) if result.citations else 'none'}")
    print(f"\nSteps used : {result.steps_used} / {MAX_STEPS} max")
    print(f"Total time : {result.total_time_ms}ms")
    print(sep)


# ── MAIN AGENT LOOP ───────────────────────────────────────────────────────────
def run_agent(question: str, verbose: bool = False) -> AgentResult:
    """
    Main agent entry point. Implements the full architecture diagram:

    ① Rewrite query
    ② Gate check (trivial / refuse / proceed)
    ③ Make plan
    ④ Tool selection + execution   ┐
    ⑤ Sufficiency check            ├─ loop, max 8 iterations
    ⑥ Answer OR loop back          ┘

    Args:
        question : user's natural language question
        verbose  : if True, print trace to stdout

    Returns:
        AgentResult dataclass with answer, citations, trace, status
    """
    start_time = time.time()
    result     = AgentResult(question=question, rewritten_q="", plan="")

    try:
        # ── ① gate check (on original question) ───────────────────────────────
        # We check the original question before rewrite so trivial questions 
        # (e.g. "what is today's date") aren't accidentally "financialized" by the rewriter.
        decision, reason = gate_check(question)

        if decision == "TRIVIAL":
            # answer directly, no tools
            current_date = datetime.now().strftime("%A, %B %d, %Y")
            result.final_answer = _llm(
                f"Today's date is {current_date}.\n"
                f"Answer this question directly and concisely:\n{question}"
            )
            result.status    = "ok"
            result.plan      = "Answered directly — no tools needed."
            result.steps_used = 0

        elif decision == "REFUSE":
            raise AgentRefusal(reason)

        else:
            # ── ② query rewrite ───────────────────────────────────────────────────
            result.rewritten_q = rewrite_query(question)
            working_q          = result.rewritten_q

            # ── ③ planning ────────────────────────────────────────────────────
            plan, next_tool, next_input = make_plan(working_q)
            result.plan = plan


            context    = ""
            tool_calls : list[ToolCall] = []
            step       = 0

            # ── ④⑤⑥ loop ──────────────────────────────────────────────────────
            while True:

                # hard cap enforcement — never bypass this
                if step >= MAX_STEPS:
                    raise HardCapExceeded(
                        f"Agent reached the {MAX_STEPS}-step hard cap "
                        f"without a sufficient answer. "
                        f"The question may be unanswerable from available sources."
                    )

                step += 1

                # ── ④ execute tool ────────────────────────────────────────────
                t0     = time.time()
                output = execute_tool(next_tool, next_input, working_q)
                ms     = int((time.time() - t0) * 1000)

                # update context
                context = _format_context(
                    tool_calls + [ToolCall(
                        step=step, tool=next_tool,
                        input=next_input, output=output,
                        latency_ms=ms, sufficient=False
                    )]
                )

                # ── ⑤ sufficiency check ───────────────────────────────────────
                sufficient, suf_reason = is_sufficient(working_q, context)

                tc = ToolCall(
                    step       = step,
                    tool       = next_tool,
                    input      = next_input,
                    output     = output,
                    latency_ms = ms,
                    sufficient = sufficient,
                )
                tool_calls.append(tc)
                result.tool_calls = tool_calls

                # ── ⑥ answer or loop back ─────────────────────────────────────
                if sufficient:
                    # compose final answer
                    answer, citations      = compose_answer(working_q, context)
                    result.final_answer   = answer
                    result.citations      = citations
                    result.status         = "ok"
                    result.steps_used     = step
                    break

                else:
                    # not sufficient — select next tool
                    next_tool, next_input = select_tool(working_q, context)

                    if next_tool == "DONE":
                        # LLM says DONE even though sufficiency said no
                        # trust DONE — compose with what we have
                        answer, citations    = compose_answer(working_q, context)
                        result.final_answer  = answer
                        result.citations     = citations
                        result.status        = "ok"
                        result.steps_used    = step
                        break

    except HardCapExceeded as e:
        result.final_answer = (
            f"I was unable to answer this question within the {MAX_STEPS} "
            f"allowed tool calls. The question may require information not "
            f"available in the current sources, or may be too broad. "
            f"Please try a more specific question.\n\nDetails: {e}"
        )
        result.status    = "cap_exceeded"
        result.steps_used = MAX_STEPS

    except AgentRefusal as e:
        result.final_answer = (
            f"I'm not able to help with that. {e}\n\n"
            f"I can answer factual questions about Infosys, Accenture, and "
            f"Cognizant financials from their annual reports (FY21–FY24), "
            f"or search for recent news about these companies."
        )
        result.status    = "refused"
        result.steps_used = 0

    except Exception as e:
        result.final_answer = f"An unexpected error occurred: {e}"
        result.status       = "error"

    finally:
        result.total_time_ms = int((time.time() - start_time) * 1000)
        trace_file           = _save_trace(result)
        if verbose:
            _print_trace(result)
            print(f"\nTrace saved → {trace_file}")

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Agentic RAG — ask questions about company financials"
    )
    parser.add_argument(
        "question",
        nargs  = "+",
        help   = "Your question in natural language"
    )
    parser.add_argument(
        "--trace", "-t",
        action  = "store_true",
        help    = "Print full trace to stdout"
    )
    args = parser.parse_args()

    question = " ".join(args.question)
    result   = run_agent(question, verbose=args.trace)

    if not args.trace:
        # minimal output when not in trace mode
        print(f"\nAnswer: {result.final_answer}")
        if result.citations:
            print(f"\nCitations: {', '.join(result.citations)}")
        print(f"\n[{result.steps_used} tool calls | {result.total_time_ms}ms | status: {result.status}]")