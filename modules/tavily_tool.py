# modules/tavily_tool.py

"""
Tavily search wrapper.


This module exposes a LangChain-compatible tool named `tavily_search` (and a lightweight
fallback) that performs evidence searches via the Tavily API and returns summarized
results with titles, snippets and URLs.


IMPORTANT (agent guidance):
When the user explicitly asks you to "search the web", "give sources", "cite", or
"find studies/guidelines", CALL the tool named **"tavily_search"** with the user's
search query. The tool returns summarized results and URLs. Always include an
EVIDENCE section listing the Tavily results (if the tool was used).


This file intentionally avoids importing langchain at module-import time. Call
`get_tavily_tool()` to obtain a tool object that is compatible with LangChain when
langchain is installed, or a small fallback object otherwise.
"""


import os
import logging
import requests
from typing import Dict, Any, List


# Configure module logger

logger = logging.getLogger(__name__)
if not logger.handlers:
    # Basic configuration – Streamlit or your app runner may override this
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# Config (read ENV on use)

def _get_tavily_config():
    return {
        "url": os.getenv("TAVILY_API_URL", "https://api.tavily.ai/v1/search"),
        "key": os.getenv("TAVILY_API_KEY", ""),
    }




def _format_tavily_items(items: List[Dict[str, Any]], max_results: int = 5) -> str:
    """Format Tavily result items into a plain-text summary the agent can show."""
    text_out = []
    for i, it in enumerate(items[:max_results]):
        title = it.get("title") or it.get("headline") or it.get("name") or "(no title)"
        snippet = it.get("snippet") or it.get("excerpt") or it.get("summary") or it.get("text") or ""
        url = it.get("url") or it.get("link") or ""
        entry = f"{i+1}. {title}\n{snippet}\n{url}"
        text_out.append(entry)
    return "\n\n".join(text_out) if text_out else "No results from Tavily."




def tavily_search(query: str, max_results: int = 5) -> str:
    """
    Query the Tavily API if a key is available; otherwise return a fallback message.
    Returns a plain text string summarizing top results (titles, snippets, urls).


    The function logs execution for debugging so you can verify the tool ran.
    Args:
        query: The search query string.
        max_results: Maximum number of results to return."""
    cfg = _get_tavily_config()
    api_key = cfg["key"]
    api_url = cfg["url"]


    logger.info("tavily_search called with query=%s max_results=%d", query, max_results)


    if not api_key:
        msg = "Tavily API key not configured. Please provide TAVILY_API_KEY to enable Tavily Search."
        logger.info("%s", msg)
        return msg


    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"q": query, "k": max_results}
    try:
        resp = requests.post(api_url, json=payload, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("results") or data.get("data") or []
        formatted = _format_tavily_items(items, max_results=max_results)
        logger.info("tavily_search returned %d items", len(items))
        return formatted
    except Exception as e:
        logger.exception("Tavily search failed")
        return f"Tavily search failed: {e}"




# Fallback simple tool object usable by agent_engine.create_health_agent()
class _SimpleTool:
    def __init__(self, name: str, func, description: str = ""):
        self.name = name
        self.func = func
        self.description = description


    def __call__(self, x):
        # LangChain Tool.func usually expects a single str input; keep behavior compatible
        # Accept either (str) or (str, int) tuples if needed by converting accordingly.
        try:
            if isinstance(x, (list, tuple)) and len(x) == 2 and isinstance(x[0], str):
                return self.func(x[0], max_results=int(x[1]))
            return self.func(x)
        except Exception as e:
            logger.exception("Error in fallback tavily tool call")
            return f"Tavily fallback tool error: {e}"




def get_tavily_tool():
    """
    Return a tool object. If langchain is installed, return a langchain.tools.Tool instance
    named `tavily_search`. Otherwise, return a lightweight fallback with .name and .func
    attributes. The agent should call the tool with a single string query.
    """
    try:
        # Import Tool lazily so top-level import won't pull langchain unless this function is used.
        from langchain.tools import Tool as LC_Tool # type: ignore


        return LC_Tool(
            name="tavily_search",
            func=tavily_search,
            description=(
                "Search the web for up-to-date medical guidance, clinical guidelines, "
                "drug labels, and peer-reviewed studies. Input: a short search query string. "
                "Return: summarized results with titles, short snippets, and URLs."
            ),
        )
    except Exception:
        # Return fallback tool compatible with agent_engine's conversion logic
        return _SimpleTool(
            name="tavily_search",
            func=tavily_search,
            description="(Fallback) Tavily search tool. LangChain not installed; returns text results.",
        )