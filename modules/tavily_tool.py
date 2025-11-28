# modules/tavily_tool.py
"""
Tavily search wrapper.

This file avoids importing langchain at module import time to prevent hitting heavy
dependencies when the optional Tavily tool is not used. get_tavily_tool() will try
to return a LangChain Tool if langchain is available; otherwise it returns a small
fallback object with `.name`, `.func`, and `.description` attributes that is
compatible with the agent conversion logic in agent_engine.py.
"""

import os
import requests
from typing import Dict, Any

# Config (read ENV on use)
def _get_tavily_config():
    return {
        "url": os.getenv("TAVILY_API_URL", "https://api.tavily.ai/v1/search"),
        "key": os.getenv("TAVILY_API_KEY", ""),
    }

def tavily_search(query: str, max_results: int = 5) -> str:
    """
    Query Tavily API if key available; otherwise return a short fallback message.
    Returns a plain text string summarizing top results.
    """
    cfg = _get_tavily_config()
    api_key = cfg["key"]
    api_url = cfg["url"]

    if not api_key:
        return "Tavily API key not configured. Please provide TAVILY_API_KEY to enable Tavily Search."

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"q": query, "k": max_results}
    try:
        resp = requests.post(api_url, json=payload, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("results") or data.get("data") or []
        text_out = []
        for i, it in enumerate(items[:max_results]):
            title = it.get("title") or it.get("headline") or it.get("name") or ""
            snippet = it.get("snippet") or it.get("excerpt") or it.get("summary") or it.get("text") or ""
            url = it.get("url") or it.get("link") or ""
            text_out.append(f"{i+1}. {title}\n{snippet}\n{url}")
        return "\n\n".join(text_out) if text_out else "No results from Tavily."
    except Exception as e:
        return f"Tavily search failed: {e}"

# Fallback simple tool object usable by agent_engine.create_health_agent()
class _SimpleTool:
    def __init__(self, name: str, func, description: str = ""):
        self.name = name
        self.func = func
        self.description = description

    def __call__(self, x):
        # langchain Tool.func usually expects a single str input; keep behavior compatible
        return self.func(x)

def get_tavily_tool():
    """
    Return a tool object. If langchain is installed, return a langchain.tools.Tool instance.
    Otherwise, return a lightweight fallback with .name and .func attributes.
    """
    try:
        # Import Tool lazily so top-level import won't pull langchain unless this function is used.
        from langchain.tools import Tool as LC_Tool  # type: ignore
        return LC_Tool(
            name="TavilySearch",
            func=tavily_search,
            description="Search the web for the latest health information and guidelines. Input is a search query; returns summarized results."
        )
    except Exception:
        # Return fallback tool compatible with agent_engine's conversion logic
        return _SimpleTool(
            name="TavilySearch",
            func=tavily_search,
            description="(Fallback) Tavily search tool. LangChain not installed; this will return plain text results."
        )