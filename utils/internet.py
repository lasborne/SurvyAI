"""
Internet sourcing utilities (no API key required).

SurvyAI uses this module for optional, user-permissioned internet lookups.
All returned results must be clearly marked as internet-sourced by the caller.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

def wikipedia_search(query: str, timeout_seconds: int = 15, limit: int = 8) -> Dict[str, Any]:
    """
    Search Wikipedia (no key) via MediaWiki API and return result snippets + URLs.
    """
    if not query or not query.strip():
        return {"success": False, "error": "Empty query"}

    q = query.strip()
    limit = max(1, min(10, int(limit or 8)))

    params = {
        "action": "query",
        "list": "search",
        "srsearch": q,
        "srlimit": str(limit),
        "format": "json",
        "utf8": "1",
    }
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "SurvyAI/1.0 (internet_search; Wikipedia)",
                "Accept": "application/json",
            },
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        data = json.loads(raw)
    except Exception as e:
        logger.warning(f"Wikipedia search failed: {e}")
        return {"success": False, "error": str(e), "provider": "wikipedia"}

    results: List[Dict[str, str]] = []
    for item in (data.get("query", {}).get("search", []) or []):
        title = (item.get("title") or "").strip()
        snippet = (item.get("snippet") or "").strip()
        # snippet may contain HTML tags; keep minimal cleanup
        snippet = snippet.replace("<span class=\"searchmatch\">", "").replace("</span>", "")
        if not title:
            continue
        page_url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
        results.append({"title": title, "url": page_url, "snippet": snippet})

    return {
        "success": True,
        "provider": "wikipedia",
        "query": q,
        "results": results,
        "note": "INTERNET_SOURCED",
    }


def duckduckgo_instant_answer_search(query: str, timeout_seconds: int = 15) -> Dict[str, Any]:
    """
    Search using DuckDuckGo's Instant Answer API (no key).

    Note: This is *not* a full web crawler; it returns limited structured results.
    """
    if not query or not query.strip():
        return {"success": False, "error": "Empty query"}

    q = query.strip()
    params = {
        "q": q,
        "format": "json",
        "no_redirect": "1",
        "no_html": "1",
        "skip_disambig": "1",
    }
    url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode(params)

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "SurvyAI/1.0 (internet_search; +https://example.invalid)",
                "Accept": "application/json",
            },
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        data = json.loads(raw)
    except Exception as e:
        logger.warning(f"DuckDuckGo search failed: {e}")
        return {"success": False, "error": str(e), "provider": "duckduckgo_instant_answer"}

    results: List[Dict[str, str]] = []

    # Primary abstract / answer
    abstract = (data.get("AbstractText") or "").strip()
    abstract_url = (data.get("AbstractURL") or "").strip()
    heading = (data.get("Heading") or "").strip()
    if abstract:
        results.append(
            {
                "title": heading or "DuckDuckGo Abstract",
                "url": abstract_url or "",
                "snippet": abstract,
            }
        )

    # Related topics (may be nested)
    def _walk_related(items: list) -> None:
        for it in items or []:
            if isinstance(it, dict) and "Topics" in it:
                _walk_related(it.get("Topics") or [])
                continue
            if not isinstance(it, dict):
                continue
            txt = (it.get("Text") or "").strip()
            first_url = (it.get("FirstURL") or "").strip()
            if txt:
                results.append(
                    {
                        "title": txt.split(" - ", 1)[0][:120] if " - " in txt else txt[:120],
                        "url": first_url,
                        "snippet": txt,
                    }
                )

    _walk_related(data.get("RelatedTopics") or [])

    # De-dup by URL+snippet
    seen = set()
    deduped: List[Dict[str, str]] = []
    for r in results:
        key = (r.get("url", "") + "|" + r.get("snippet", "")).lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    return {
        "success": True,
        "provider": "duckduckgo_instant_answer",
        "query": q,
        "results": deduped[:10],
        "note": "INTERNET_SOURCED",
    }


def internet_search(query: str, timeout_seconds: int = 15) -> Dict[str, Any]:
    """
    Best-effort internet search without an API key.

    Strategy:
    - DuckDuckGo Instant Answer (fast, sometimes empty)
    - Wikipedia search as a reliable fallback for many technical terms
    """
    ddg = duckduckgo_instant_answer_search(query, timeout_seconds=timeout_seconds)
    results: List[Dict[str, str]] = []
    providers: List[str] = []

    if ddg.get("success"):
        providers.append(ddg.get("provider", "duckduckgo_instant_answer"))
        for r in ddg.get("results", []) or []:
            rr = dict(r)
            rr["provider"] = ddg.get("provider", "duckduckgo_instant_answer")
            results.append(rr)

    if not results:
        wiki = wikipedia_search(query, timeout_seconds=timeout_seconds, limit=8)
        if wiki.get("success"):
            providers.append(wiki.get("provider", "wikipedia"))
            for r in wiki.get("results", []) or []:
                rr = dict(r)
                rr["provider"] = wiki.get("provider", "wikipedia")
                results.append(rr)
        else:
            return {
                "success": False,
                "error": wiki.get("error", "No results and fallback failed"),
                "providers_attempted": providers or ["duckduckgo_instant_answer", "wikipedia"],
                "query": query,
                "note": "INTERNET_SOURCED",
            }

    return {
        "success": True,
        "providers": providers or ["duckduckgo_instant_answer", "wikipedia"],
        "query": query.strip(),
        "results": results[:10],
        "note": "INTERNET_SOURCED",
    }


