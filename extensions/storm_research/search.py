"""Pluggable web search backends for the STORM pipeline.

Provides a simple protocol and two implementations:
- DuckDuckGoSearch (free, no API key)
- TavilySearch (requires TAVILY_API_KEY)

The backends are auto-detected based on available packages / env vars.
"""
from __future__ import annotations

import logging
import os
from typing import Protocol

from .data import SearchResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class SearchBackend(Protocol):
    """Interface for web search backends."""

    name: str

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Run a web search and return results."""
        ...


# ---------------------------------------------------------------------------
# DuckDuckGo (free, no API key)
# ---------------------------------------------------------------------------

class DuckDuckGoSearch:
    """Free web search via the ``duckduckgo-search`` package."""

    name = "duckduckgo"

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            logger.warning(
                "duckduckgo-search not installed. "
                "Install with: pip install duckduckgo-search"
            )
            return []

        results: list[SearchResult] = []
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=top_k):
                    results.append(
                        SearchResult(
                            url=r.get("href", r.get("link", "")),
                            title=r.get("title", ""),
                            description=r.get("body", r.get("snippet", "")),
                            snippets=[r.get("body", r.get("snippet", ""))],
                        )
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning("DuckDuckGo search failed for %r: %s", query, exc)
        return results


# ---------------------------------------------------------------------------
# Tavily (requires API key — higher quality)
# ---------------------------------------------------------------------------

class TavilySearch:
    """Web search via Tavily API (requires TAVILY_API_KEY env var)."""

    name = "tavily"

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("TAVILY_API_KEY", "")

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if not self._api_key:
            logger.warning("TAVILY_API_KEY not set — cannot use Tavily search.")
            return []

        try:
            import httpx
        except ImportError:
            try:
                import requests as httpx  # type: ignore[no-redef]
            except ImportError:
                logger.warning("Neither httpx nor requests is installed for Tavily.")
                return []

        results: list[SearchResult] = []
        try:
            resp = httpx.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self._api_key,
                    "query": query,
                    "max_results": top_k,
                    "include_raw_content": False,
                },
                timeout=15,
            )
            data = resp.json()
            for r in data.get("results", []):
                results.append(
                    SearchResult(
                        url=r.get("url", ""),
                        title=r.get("title", ""),
                        description=r.get("content", ""),
                        snippets=[r.get("content", "")],
                    )
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tavily search failed for %r: %s", query, exc)
        return results


# ---------------------------------------------------------------------------
# Auto-detect best available backend
# ---------------------------------------------------------------------------

def get_search_backend() -> SearchBackend:
    """Return the best available search backend.

    Priority:
    1. Tavily (if TAVILY_API_KEY is set)
    2. DuckDuckGo (if duckduckgo-search is installed)
    3. Fallback (returns empty results with a warning)
    """
    # Check Tavily
    tavily_key = os.environ.get("TAVILY_API_KEY", "")
    if tavily_key:
        logger.debug("Using Tavily search backend.")
        return TavilySearch(api_key=tavily_key)

    # Check DuckDuckGo
    try:
        import duckduckgo_search  # noqa: F401
        logger.debug("Using DuckDuckGo search backend.")
        return DuckDuckGoSearch()
    except ImportError:
        pass

    logger.warning(
        "No search backend available. Install duckduckgo-search "
        "or set TAVILY_API_KEY. Research quality will be limited."
    )
    return _FallbackSearch()


class _FallbackSearch:
    """Fallback when no search backend is available."""

    name = "none"

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        logger.warning("No search backend — returning empty results for %r", query)
        return []
