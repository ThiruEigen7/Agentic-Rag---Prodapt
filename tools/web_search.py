"""
tools/web_search.py
--------------------
Live web search using Tavily API (free tier).
Used for anything NOT in the annual report PDFs or the structured CSV:
    - current stock prices
    - recent earnings announcements
    - analyst ratings and target prices
    - leadership changes / executive news
    - recent deal wins, client announcements
    - IT sector news from the last few months

Tavily free tier limits:
    - 1000 API calls / month
    - Use search_depth="basic" to stay within limits
    - Each call here = 1 API credit

Standalone test:
    python tools/web_search.py "Infosys stock price today"
    python tools/web_search.py "Accenture Q1 FY25 earnings results"
    python tools/web_search.py "Cognizant CEO latest news 2025"
    python tools/web_search.py "IT sector layoffs 2025"
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass

from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

# ── config ────────────────────────────────────────────────────────────────────
MAX_RESULTS    = 3       # max results per search (3 = good balance for free tier)
SEARCH_DEPTH   = "basic" # "basic" uses 1 credit | "advanced" uses 2 credits
MAX_QUERY_LEN  = 100     # Tavily works best with short, focused queries


# ── result data class ─────────────────────────────────────────────────────────
@dataclass
class WebResult:
    title          : str
    snippet        : str   # 2-3 sentence excerpt
    url            : str
    published_date : str   # ISO date string or "" if unknown
    score          : float # Tavily relevance score (0-1)


# ── main tool function ────────────────────────────────────────────────────────
def web_search(query: str) -> dict:
    """
    Search the live web for recent information about IT companies.

    Use this tool for:
        - Current stock prices, market cap, P/E ratios
        - Recent quarterly earnings (beyond FY24 annual reports)
        - Analyst ratings, buy/sell/hold recommendations
        - Leadership changes, executive appointments
        - Recent deal wins, partnership announcements
        - IT sector news, macro trends

    Do NOT use for:
        - Historical financials FY21-FY24 → use query_data instead
        - Annual report content → use search_docs instead

    Args:
        query: Short, focused search query. Under 10 words works best.
               Good: "Infosys stock price NSE today"
               Good: "Accenture Q2 FY25 quarterly results"
               Bad:  "tell me everything about Infosys financials and future"

    Returns:
        dict with key 'results': list of up to 3 dicts, each containing:
            title          : page title
            snippet        : 2-3 sentence excerpt
            url            : full source URL
            published_date : publication date (ISO) or "" if unknown
            score          : Tavily relevance score (0.0 to 1.0)

        On error, returns:
            {'results': [], 'error': 'error message'}
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {
            "results" : [],
            "error"   : "TAVILY_API_KEY not set. Add it to your .env file.",
        }

    # trim very long queries
    if len(query) > MAX_QUERY_LEN:
        query = query[:MAX_QUERY_LEN].rsplit(" ", 1)[0]  # trim at word boundary

    try:
        client   = TavilyClient(api_key=api_key)
        response = client.search(
            query        = query,
            search_depth = SEARCH_DEPTH,
            max_results  = MAX_RESULTS,
        )

        results = []
        for item in response.get("results", [])[:MAX_RESULTS]:
            results.append(WebResult(
                title          = item.get("title", ""),
                snippet        = item.get("content", ""),
                url            = item.get("url", ""),
                published_date = item.get("published_date", ""),
                score          = round(float(item.get("score", 0.0)), 4),
            ))

        return {
            "results" : [
                {
                    "title"         : r.title,
                    "snippet"       : r.snippet,
                    "url"           : r.url,
                    "published_date": r.published_date,
                    "score"         : r.score,
                }
                for r in results
            ]
        }

    except Exception as e:
        return {
            "results" : [],
            "error"   : f"Tavily search failed: {e}",
        }


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print('  python tools/web_search.py "your search query"')
        print()
        print("Examples:")
        print('  python tools/web_search.py "Infosys stock price today"')
        print('  python tools/web_search.py "Accenture Q1 FY25 earnings"')
        print('  python tools/web_search.py "Cognizant CEO 2025"')
        print('  python tools/web_search.py "IT sector layoffs 2025"')
        sys.exit(1)

    query  = " ".join(sys.argv[1:])
    print(f"Searching: '{query}'\n")

    result = web_search(query)

    if "error" in result:
        print(f"Error: {result['error']}")
        sys.exit(1)

    if not result["results"]:
        print("No results returned.")
        sys.exit(0)

    for i, r in enumerate(result["results"], 1):
        print(f"── Result {i} {'─'*45}")
        print(f"Title   : {r['title']}")
        print(f"Date    : {r['published_date'] or 'unknown'}")
        print(f"Score   : {r['score']}")
        print(f"URL     : {r['url']}")
        print(f"Snippet : {r['snippet'][:300]}...")
        print()