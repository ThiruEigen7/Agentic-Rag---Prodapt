"""
Tool 3: web_search
------------------
Live web search using Tavily API. Returns top-3 snippets with URLs and dates.
Designed for recent news, current stock prices, analyst ratings, and events
that post-date the annual reports in the corpus.

Standalone usage:
    python tools/web_search.py "Infosys stock price today"
    python tools/web_search.py "TCS Q1 FY25 earnings results"
    python tools/web_search.py "Wipro CEO latest news 2024"
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass

try:
    from tavily import TavilyClient
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

load_dotenv()

# ── config ────────────────────────────────────────────────────────────────────
MAX_RESULTS   = 3
MAX_QUERY_LEN = 100     # Tavily works best with short, focused queries


# ── data types ────────────────────────────────────────────────────────────────
@dataclass
class WebResult:
    title: str
    snippet: str
    url: str
    published_date: str     # ISO date string or empty string if unavailable


# ── main tool function ────────────────────────────────────────────────────────
def web_search(query: str) -> dict:
    """
    Search the live web for recent information about IT companies,
    stock prices, earnings, analyst ratings, and current events.

    Args:
        query: Short, focused search query (ideally under 10 words).
               e.g. "Infosys current stock price NSE"
               e.g. "TCS Q4 FY25 results analyst reaction"

    Returns:
        dict with key "results" — list of up to 3 results, each containing:
            title          : page title
            snippet        : 2–3 sentence excerpt from the page
            url            : full URL of the source
            published_date : publication date (ISO format) or "" if unknown

    Raises:
        EnvironmentError  : if TAVILY_API_KEY is not set
        Exception         : if the Tavily API call fails
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "TAVILY_API_KEY not set. Add it to your .env file."
        )

    # truncate very long queries — Tavily works best short
    if len(query) > MAX_QUERY_LEN:
        query = query[:MAX_QUERY_LEN]

    client = TavilyClient(api_key=api_key)

    response = client.search(
        query=query,
        search_depth="basic",       # "basic" is sufficient and cheaper
        max_results=MAX_RESULTS,
        include_answer=False,       # we want raw results, not a summary
    )

    results = []
    for item in response.get("results", [])[:MAX_RESULTS]:
        results.append(
            WebResult(
                title=item.get("title", ""),
                snippet=item.get("content", ""),
                url=item.get("url", ""),
                published_date=item.get("published_date", ""),
            )
        )

    return {
        "results": [
            {
                "title": r.title,
                "snippet": r.snippet,
                "url": r.url,
                "published_date": r.published_date,
            }
            for r in results
        ]
    }


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print('  python tools/web_search.py "your query here"')
        print()
        print("Example queries:")
        print('  python tools/web_search.py "Infosys stock price today"')
        print('  python tools/web_search.py "TCS Q4 FY25 earnings results"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    print(f"\nSearching web for: '{query}'\n")

    try:
        result = web_search(query)

        if not result["results"]:
            print("No results returned.")
        else:
            for i, r in enumerate(result["results"], 1):
                print(f"── Result {i} ──────────────────────────────────")
                print(f"Title   : {r['title']}")
                print(f"Date    : {r['published_date'] or 'unknown'}")
                print(f"URL     : {r['url']}")
                print(f"Snippet : {r['snippet'][:300]}...")
                print()

    except EnvironmentError as e:
        print(f"Config error: {e}")
    except Exception as e:
        print(f"Search failed: {e}")