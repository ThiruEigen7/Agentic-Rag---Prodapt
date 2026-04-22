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

from google import genai
from google.genai import types
from dotenv import load_dotenv

from tools import search_docs, query_data, web_search
from tools.search_docs import detect_company_in_query

load_dotenv()

# ── config ────────────────────────────────────────────────────────────────────
MAX_STEPS     = 8           # hard cap — never change this
TRACES_DIR    = Path("traces")
GEMINI_MODEL  = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")   # free tier

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
AVAILABLE DATA SOURCES:

1. search_docs — Annual Report PDFs (qualitative text):
   Companies : Infosys, Accenture, Cognizant
   Years     : FY22, FY23, FY24, FY25
   Contains  : MD&A commentary, CEO letters, strategic priorities, business segment
               descriptions, risk factors, operational highlights, headcount narrative,
               deal wins, AI/cloud strategy, margin explanation text

2. query_data — Structured Financial Database (numbers):
   Companies : Infosys, Accenture, Cognizant
   Years     : FY22, FY23, FY24, FY25
   Columns   : company, fiscal_year, quarter, revenue_bn_USD, op_margin_pct,
               headcount, epsusd, source_link, notes
   Contains  : revenue (USD billions), operating margin %, headcount,
               EPS (USD), quarterly and annual breakdowns

3. web_search — Live Web (real-time):
   Contains  : current stock prices, analyst ratings, news from last 90 days,
               earnings releases beyond FY25, leadership changes, deal announcements
   Use when  : question asks for "current", "today", "latest", "recent",
               or any period BEYOND FY25

ROUTING RULES:
- FY22, FY23, FY24, FY25 EXISTS in search_docs and query_data — always use those first
- "Latest" or "current" financials = FY25 → query_data or search_docs
- Only use web_search for live prices, news, or data beyond FY25
- NEVER say data is unavailable for FY22-FY25 — it is in the database
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


# ── Gemini client ─────────────────────────────────────────────────────────────
def _get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not set. Add it to .env file.\n"
            "Get free key: https://aistudio.google.com/app/apikey"
        )
    return genai.Client(api_key=api_key)


def _llm(prompt: str) -> str:
    """Single Gemini call. Returns response text stripped of whitespace."""
    client   = _get_client()
    response = client.models.generate_content(
        model    = GEMINI_MODEL,
        contents = prompt,
    )
    return response.text.strip()


def _llm_json(prompt: str) -> dict:
    """
    Gemini call expecting JSON output.
    Strips markdown fences if model adds them despite instructions.
    Falls back to empty dict on parse failure.
    """
    raw = _llm(prompt)
    # strip ```json ... ``` fences
    if raw.startswith("```"):
        lines = [l for l in raw.split("\n") if not l.strip().startswith("```")]
        raw   = "\n".join(lines).strip()
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
        return _llm(prompt)
    except Exception:
        return question   # fallback: use original if rewrite fails


# ── pre-loop step ②: gate — trivial / refuse / proceed ───────────────────────
GATE_PROMPT = """You are a classifier for a financial data agent.

Classify this question into exactly one of three categories:

TRIVIAL  — answerable directly with no data lookup (math, general knowledge, greetings)
           e.g. "what is 2+2", "what does EPS mean", "hello"

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

Question: {{question}}"""

def gate_check(question: str) -> tuple[str, str]:
    """
    Returns (decision, reason).
    decision: "TRIVIAL" | "REFUSE" | "PROCEED"
    """
    result = _llm_json(GATE_PROMPT.format(DATA_CONTEXT=DATA_CONTEXT, question=question))
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

Question: {{question}}"""

def make_plan(question: str) -> tuple[str, str, str]:
    """
    Returns (plan_text, first_tool, first_input).
    """
    result     = _llm_json(PLAN_PROMPT.format(DATA_CONTEXT=DATA_CONTEXT, question=question))
    plan       = result.get("plan", "")
    first_tool = result.get("first_tool", "query_data")
    first_input= result.get("first_input", question)
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

QUESTION: {{question}}

CONTEXT COLLECTED SO FAR:
{{context}}

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
    """
    result     = _llm_json(TOOL_SELECT_PROMPT.format(
        DATA_CONTEXT=DATA_CONTEXT, question=question, context=context
    ))
    tool       = result.get("tool", "query_data")
    tool_input = result.get("input", question)
    if tool not in ("search_docs", "query_data", "web_search", "DONE"):
        tool = "query_data"
    return tool, tool_input


# ── loop step ④: execute tool ─────────────────────────────────────────────────
def execute_tool(tool: str, tool_input: str, question: str) -> dict:
    """
    Call the selected tool and return its raw result dict.
    Errors are caught and returned as structured error dicts — never crash.
    """
    try:
        if tool == "search_docs":
            # detect company for targeted search
            company  = detect_company_in_query(question)
            balanced = any(
                w in question.lower()
                for w in ["compare", "all", "three", "each", "versus", "vs"]
            )
            if company:
                return search_docs(tool_input, top_k=3, company=company)
            elif balanced:
                return search_docs(tool_input, top_k=1, balanced=True)
            else:
                return search_docs(tool_input, top_k=3)

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

QUESTION: {{question}}

CONTEXT COLLECTED:
{{context}}

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
    """Returns (sufficient, reason)."""
    result = _llm_json(SUFFICIENCY_PROMPT.format(
        question=question, context=context
    ))
    sufficient = result.get("sufficient", False)
    reason     = result.get("reason", "")
    return bool(sufficient), reason


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
    """
    raw = _llm(COMPOSE_PROMPT.format(question=question, context=context))

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
                for row in result[:10]:   # cap at 10 rows in context
                    parts.append(f"  {row}")
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
        # ── ① query rewrite ───────────────────────────────────────────────────
        result.rewritten_q = rewrite_query(question)
        working_q          = result.rewritten_q

        # ── ② gate check ──────────────────────────────────────────────────────
        decision, reason = gate_check(working_q)

        if decision == "TRIVIAL":
            # answer directly, no tools
            result.final_answer = _llm(
                f"Answer this question directly and concisely:\n{question}"
            )
            result.status    = "ok"
            result.plan      = "Answered directly — no tools needed."
            result.steps_used = 0

        elif decision == "REFUSE":
            raise AgentRefusal(reason)

        else:
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