"""
Internet sourcing utilities for SurvyAI.

This module implements a production-grade, **domain-agnostic** multi-stage web
retrieval pipeline modelled on how enterprise web-connected assistants
(ChatGPT browsing, Claude, Gemini, Perplexity) actually retrieve real-time
information. The guiding principle:

    Search retrieves *evidence*, not *answers*. The LLM synthesises and reasons
    over trustworthy evidence; the retrieval system's job is to find that
    evidence efficiently.

Pipeline (see `research()`):

    Query variants  →  Multi-source retrieval (over-fetch)  →  De-dup
        →  Source trust scoring  →  Relevance re-ranking
        →  Page content extraction (top sources, concurrent)
        →  Evidence pack + cross-source confidence

NOTHING in here is hard-coded to a specific question or domain. Trust scoring is
based on general signals (TLD class, official/site structure, content-farm
patterns) that apply to any topic — surveying, medicine, companies, sports, etc.

Backends:
  * If a search API key is configured (Tavily / Brave / SerpAPI), it is used for
    high-quality ranked web results (enterprise path).
  * Otherwise the pipeline falls back to key-free providers (DuckDuckGo HTML +
    Wikipedia), so the feature works out-of-the-box at zero cost.

All returned results must be clearly marked as internet-sourced by the caller.
"""

from __future__ import annotations

import concurrent.futures
import html
import json
import os
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 SurvyAI/1.0"
)

# Lightweight English stopword set for relevance scoring (kept small on purpose).
_STOPWORDS = frozenset(
    """
    a an the of to in on at for and or but is are was were be been being
    this that these those it its as by with from into about over under
    do does did done has have had what which who whom whose when where why how
    i you he she they we me him her them my your his their our
    """.split()
)


# ==========================================================================
# 0. Tokenisation & generic helpers
# ==========================================================================

def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _content_tokens(text: str) -> List[str]:
    return [t for t in _tokens(text) if t not in _STOPWORDS and len(t) > 1]


def _domain_of(url: str) -> str:
    try:
        netloc = urllib.parse.urlparse(url).netloc.lower()
        return netloc[4:] if netloc.startswith("www.") else netloc
    except Exception:
        return ""


def _http_get(url: str, timeout_seconds: int = 12, accept: str = "*/*") -> Optional[str]:
    """Plain GET that returns decoded text, or None on any failure."""
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": _DEFAULT_UA, "Accept": accept,
                     "Accept-Language": "en-US,en;q=0.9"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            raw = resp.read(_MAX_DOWNLOAD_BYTES)
        return raw.decode(charset, errors="replace")
    except Exception as e:  # noqa: BLE001 - network failures are expected/handled
        logger.debug("HTTP GET failed for %s: %s", url, e)
        return None


_MAX_DOWNLOAD_BYTES = 2_500_000  # cap page downloads (~2.5MB) to bound cost/latency


# ==========================================================================
# 1. Source trust scoring  (GENERAL, domain-class based — never query-specific)
# ==========================================================================

# Generic, widely-recognised high-trust web properties. This is NOT a per-query
# allow-list; it is a small set of cross-domain reference/authority sites that
# are reliable for almost any topic. Topic-specific authority is handled by the
# TLD-class rules below (e.g. *.gov / *.edu for official/academic matters).
_KNOWN_REFERENCE_DOMAINS = {
    "wikipedia.org": 0.78,
    "britannica.com": 0.80,
    "reuters.com": 0.90,
    "apnews.com": 0.90,
    "bbc.com": 0.86,
    "bbc.co.uk": 0.86,
    "nature.com": 0.92,
    "science.org": 0.92,
    "who.int": 0.93,
    "nih.gov": 0.95,
    "ncbi.nlm.nih.gov": 0.95,
    "github.com": 0.82,
    "stackoverflow.com": 0.74,
    "arxiv.org": 0.84,
}

# Penalised patterns: aggregators / low-signal / SEO content farms.
_LOW_TRUST_HINTS = (
    "pinterest.", "quora.com", "answers.com", "ehow.", "wikihow.",
    "blogspot.", "wordpress.com", "medium.com", "tumblr.",
    "facebook.com", "twitter.com", "x.com", "tiktok.com", "instagram.com",
)


def domain_trust(url: str) -> float:
    """Return a 0..1 trust prior for a URL based on **general** signals only.

    Signals (all topic-independent):
      * Government / military / international-org TLDs → very high.
      * Academic (.edu / .ac.*) → high.
      * Known cross-domain reference/news/science authorities → high.
      * Aggregators / social / content-farms → low.
      * Everything else → neutral baseline.
    """
    d = _domain_of(url)
    if not d:
        return 0.30

    # Government & official institutional TLDs (any country).
    if d.endswith(".gov") or ".gov." in d or d.endswith(".mil") or d.endswith(".int"):
        return 0.97
    if d.endswith(".edu") or ".edu." in d or ".ac." in d:
        return 0.90

    for known, score in _KNOWN_REFERENCE_DOMAINS.items():
        if d == known or d.endswith("." + known):
            return score

    if any(h in d for h in _LOW_TRUST_HINTS):
        return 0.28

    # Official-looking org/primary-source TLDs get a mild boost.
    if d.endswith(".org") or ".org." in d:
        return 0.62
    if d.endswith(".gov.ng") or d.endswith(".go.ke"):  # explicit common gov forms
        return 0.97

    return 0.50


def _freshness_boost(published_iso: Optional[str]) -> float:
    """Small recency boost (0..0.15) when a result exposes a recent date."""
    if not published_iso:
        return 0.0
    try:
        dt = datetime.fromisoformat(published_iso.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - dt).days
        if age_days <= 1:
            return 0.15
        if age_days <= 7:
            return 0.11
        if age_days <= 31:
            return 0.07
        if age_days <= 365:
            return 0.03
    except Exception:
        return 0.0
    return 0.0


# ==========================================================================
# 2. Query understanding / rewriting (rule-based, domain-agnostic)
# ==========================================================================

def rule_based_query_variants(question: str, max_variants: int = 5) -> List[str]:
    """Generate diverse search strings from a question WITHOUT any LLM call.

    Domain-agnostic transformations only:
      * the cleaned question,
      * the question stripped of leading interrogatives ("who is", "what is", …),
      * a keyword-only version (content words),
      * a quoted key-phrase version for exact matching.
    """
    q = " ".join((question or "").split()).strip().rstrip("?").strip()
    if not q:
        return []
    out: List[str] = []
    seen: set[str] = set()

    def _add(s: str) -> None:
        s = " ".join((s or "").split()).strip()
        if s and s.lower() not in seen and len(s) > 1:
            seen.add(s.lower())
            out.append(s)

    _add(q)

    # Strip a leading interrogative phrase to expose the core entity/topic.
    lead = re.match(
        r"^(who(?:'s| is| are| was| were)?|what(?:'s| is| are| was| were)?|"
        r"when(?:'s| is| was| did)?|where(?:'s| is| was)?|which|how(?: many| much| do| does| to)?|"
        r"why(?:'s| is| does| do)?|list(?: of)?|name(?: the)?|tell me(?: about)?)\s+(the\s+)?",
        q.lower(),
    )
    if lead:
        core = q[lead.end():].strip()
        _add(core)

    # Keyword-only (content tokens, original casing preserved where possible).
    content = _content_tokens(q)
    if content and len(content) >= 2:
        _add(" ".join(content))

    # Exact-phrase variant for the core (helps for proper nouns / titles).
    if lead:
        core = q[lead.end():].strip()
        if 1 < len(core.split()) <= 8:
            _add(f'"{core}"')

    return out[:max_variants]


# ==========================================================================
# 3. Search providers
# ==========================================================================

def _provider_tavily(query: str, api_key: str, max_results: int, timeout_seconds: int) -> List[Dict[str, Any]]:
    """Tavily Search API (optional, high quality). Returns [] on any failure."""
    try:
        payload = json.dumps({
            "api_key": api_key,
            "query": query,
            "max_results": max(5, min(20, max_results)),
            "search_depth": "advanced",
            "include_answer": False,
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://api.tavily.com/search",
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": _DEFAULT_UA},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        out = []
        for r in data.get("results", []) or []:
            out.append({
                "title": (r.get("title") or "").strip(),
                "url": (r.get("url") or "").strip(),
                "snippet": (r.get("content") or "").strip(),
                "published": r.get("published_date"),
                "provider": "tavily",
                "provider_score": float(r.get("score") or 0.0),
            })
        return out
    except Exception as e:
        logger.debug("Tavily search failed: %s", e)
        return []


def _provider_brave(query: str, api_key: str, max_results: int, timeout_seconds: int) -> List[Dict[str, Any]]:
    """Brave Search API (optional). Returns [] on any failure."""
    try:
        url = "https://api.search.brave.com/res/v1/web/search?" + urllib.parse.urlencode(
            {"q": query, "count": max(5, min(20, max_results))}
        )
        req = urllib.request.Request(
            url,
            headers={"X-Subscription-Token": api_key, "Accept": "application/json",
                     "User-Agent": _DEFAULT_UA},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        out = []
        for r in (data.get("web", {}) or {}).get("results", []) or []:
            out.append({
                "title": (r.get("title") or "").strip(),
                "url": (r.get("url") or "").strip(),
                "snippet": (r.get("description") or "").strip(),
                "published": (r.get("age") or None),
                "provider": "brave",
            })
        return out
    except Exception as e:
        logger.debug("Brave search failed: %s", e)
        return []


def _provider_serpapi(query: str, api_key: str, max_results: int, timeout_seconds: int) -> List[Dict[str, Any]]:
    """SerpAPI (Google results, optional). Returns [] on any failure."""
    try:
        url = "https://serpapi.com/search.json?" + urllib.parse.urlencode(
            {"q": query, "engine": "google", "num": max(5, min(20, max_results)), "api_key": api_key}
        )
        raw = _http_get(url, timeout_seconds=timeout_seconds, accept="application/json")
        if not raw:
            return []
        data = json.loads(raw)
        out = []
        for r in data.get("organic_results", []) or []:
            out.append({
                "title": (r.get("title") or "").strip(),
                "url": (r.get("link") or "").strip(),
                "snippet": (r.get("snippet") or "").strip(),
                "published": r.get("date"),
                "provider": "serpapi",
            })
        return out
    except Exception as e:
        logger.debug("SerpAPI search failed: %s", e)
        return []


def _provider_duckduckgo_html(query: str, max_results: int, timeout_seconds: int) -> List[Dict[str, Any]]:
    """Key-free real web results by parsing DuckDuckGo's HTML endpoint.

    The keyless endpoint occasionally rate-limits; we try both the html and lite
    endpoints and retry once with a short backoff before giving up (the caller
    still has Wikipedia as a final fallback)."""
    import time as _time

    out: List[Dict[str, Any]] = []
    raw = None
    attempts = [
        ("https://html.duckduckgo.com/html/", "POST"),
        ("https://lite.duckduckgo.com/lite/", "POST"),
        ("https://html.duckduckgo.com/html/", "POST"),  # one retry after backoff
    ]
    for idx, (endpoint, method) in enumerate(attempts):
        if idx == len(attempts) - 1:
            _time.sleep(0.6)  # brief backoff before the final retry
        try:
            data = urllib.parse.urlencode({"q": query}).encode("utf-8")
            req = urllib.request.Request(
                endpoint, data=data,
                headers={"User-Agent": _DEFAULT_UA, "Accept": "text/html",
                         "Accept-Language": "en-US,en;q=0.9",
                         "Content-Type": "application/x-www-form-urlencoded"},
                method=method,
            )
            with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            if raw and ("result__a" in raw or "uddg=" in raw):
                break
            raw = None
        except Exception as e:
            logger.debug("DuckDuckGo endpoint %s failed: %s", endpoint, e)
            raw = None
    if not raw:
        return out

    # Result anchors: class="result__a" href="...". Fall back to generic uddg links.
    anchors = re.findall(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', raw, re.S)
    if not anchors:
        anchors = re.findall(r'<a[^>]+href="(https?://[^"]*uddg=[^"]+)"[^>]*>(.*?)</a>', raw, re.S)

    # Snippets (best-effort, parallel order).
    snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', raw, re.S)

    for i, (href, title_html) in enumerate(anchors):
        url = _ddg_unwrap(href)
        title = _strip_tags(title_html)
        snippet = _strip_tags(snippets[i]) if i < len(snippets) else ""
        if url and title:
            out.append({"title": title, "url": url, "snippet": snippet, "provider": "duckduckgo"})
        if len(out) >= max_results:
            break
    return out


def _ddg_unwrap(href: str) -> str:
    """DuckDuckGo wraps target URLs in a redirect (uddg=...). Unwrap it."""
    try:
        if "uddg=" in href:
            qs = urllib.parse.urlparse(href).query
            params = urllib.parse.parse_qs(qs)
            if "uddg" in params:
                return urllib.parse.unquote(params["uddg"][0])
        if href.startswith("//"):
            return "https:" + href
        return href
    except Exception:
        return href


def _provider_wikipedia(query: str, max_results: int, timeout_seconds: int) -> List[Dict[str, Any]]:
    res = wikipedia_search(query, timeout_seconds=timeout_seconds, limit=max_results)
    if not res.get("success"):
        return []
    out = []
    for r in res.get("results", []) or []:
        rr = dict(r)
        rr["provider"] = "wikipedia"
        out.append(rr)
    return out


def _configured_search_api() -> Tuple[Optional[str], Optional[str]]:
    """Return (provider_name, api_key) for the first configured search API, else (None, None)."""
    for env, name in (
        ("TAVILY_API_KEY", "tavily"),
        ("BRAVE_SEARCH_API_KEY", "brave"),
        ("SERPAPI_API_KEY", "serpapi"),
    ):
        key = (os.environ.get(env) or "").strip()
        if key:
            return name, key
    return None, None


# ==========================================================================
# 4. HTML → text extraction & chunking
# ==========================================================================

def _strip_tags(fragment: str) -> str:
    txt = re.sub(r"<[^>]+>", " ", fragment or "")
    txt = html.unescape(txt)
    return " ".join(txt.split()).strip()


def html_to_text(page_html: str) -> str:
    """Extract readable body text from an HTML page (no external deps)."""
    if not page_html:
        return ""
    # Drop non-content regions entirely.
    cleaned = re.sub(r"(?is)<(script|style|noscript|template|svg|head)[^>]*>.*?</\1>", " ", page_html)
    cleaned = re.sub(r"(?is)<!--.*?-->", " ", cleaned)
    # Turn block boundaries into newlines so paragraphs survive.
    cleaned = re.sub(r"(?i)</(p|div|section|article|li|h[1-6]|br|tr)\s*>", "\n", cleaned)
    text = _strip_tags(cleaned)
    # Re-introduce sentence spacing collapsed by the strip.
    return text


def chunk_text(text: str, *, chunk_chars: int = 700, max_chunks: int = 8) -> List[str]:
    """Split text into overlapping-ish chunks bounded by sentence/paragraph breaks."""
    if not text:
        return []
    pieces = re.split(r"(?<=[\.\!\?])\s+|\n+", text)
    chunks: List[str] = []
    buf = ""
    for p in pieces:
        p = p.strip()
        if not p:
            continue
        if len(buf) + len(p) + 1 <= chunk_chars:
            buf = (buf + " " + p).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = p[:chunk_chars]
        if len(chunks) >= max_chunks:
            break
    if buf and len(chunks) < max_chunks:
        chunks.append(buf)
    return chunks


# ==========================================================================
# 5. Relevance scoring / re-ranking
# ==========================================================================

def _lexical_overlap(query_tokens: List[str], text: str) -> float:
    """Fraction of distinct query content-tokens present in text (0..1), with a
    small bonus for multi-occurrence (term frequency)."""
    if not query_tokens:
        return 0.0
    qset = set(query_tokens)
    ttoks = _content_tokens(text)
    if not ttoks:
        return 0.0
    tset = set(ttoks)
    present = qset & tset
    coverage = len(present) / len(qset)
    # Term-frequency bonus (capped) rewards focused, on-topic pages.
    tf = sum(min(ttoks.count(w), 3) for w in present)
    tf_bonus = min(0.25, tf / (len(ttoks) + 1) * 2.0)
    return min(1.0, coverage + tf_bonus)


def rerank_results(
    question: str,
    results: List[Dict[str, Any]],
    *,
    top_k: int = 8,
) -> List[Dict[str, Any]]:
    """Re-rank for *relevance* (not popularity) by combining:
        relevance(title+snippet) · trust · freshness · provider prior.
    Domain-agnostic; works for any topic.
    """
    qtok = _content_tokens(question)
    scored: List[Dict[str, Any]] = []
    for r in results:
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        url = r.get("url", "")
        rel = 0.65 * _lexical_overlap(qtok, title) + 0.35 * _lexical_overlap(qtok, snippet)
        trust = domain_trust(url)
        fresh = _freshness_boost(r.get("published"))
        provider_prior = 0.05 if r.get("provider") in ("tavily", "brave", "serpapi") else 0.0
        # Weighted blend: relevance dominates, trust strongly modulates.
        score = (0.58 * rel) + (0.32 * trust) + fresh + provider_prior
        rr = dict(r)
        rr["relevance"] = round(rel, 4)
        rr["trust"] = round(trust, 4)
        rr["rank_score"] = round(score, 4)
        rr["domain"] = _domain_of(url)
        scored.append(rr)
    scored.sort(key=lambda x: x.get("rank_score", 0.0), reverse=True)
    return scored[:top_k]


# ==========================================================================
# 6. Public: legacy-compatible simple search (kept for existing call-sites)
# ==========================================================================

def wikipedia_search(query: str, timeout_seconds: int = 15, limit: int = 8) -> Dict[str, Any]:
    """Search Wikipedia (no key) via the MediaWiki API."""
    if not query or not query.strip():
        return {"success": False, "error": "Empty query"}
    q = query.strip()
    limit = max(1, min(10, int(limit or 8)))
    params = {
        "action": "query", "list": "search", "srsearch": q,
        "srlimit": str(limit), "format": "json", "utf8": "1",
    }
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    raw = _http_get(url, timeout_seconds=timeout_seconds, accept="application/json")
    if raw is None:
        return {"success": False, "error": "request_failed", "provider": "wikipedia"}
    try:
        data = json.loads(raw)
    except Exception as e:
        return {"success": False, "error": str(e), "provider": "wikipedia"}

    results: List[Dict[str, str]] = []
    for item in (data.get("query", {}).get("search", []) or []):
        title = (item.get("title") or "").strip()
        snippet = _strip_tags(item.get("snippet") or "")
        if not title:
            continue
        page_url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
        results.append({"title": title, "url": page_url, "snippet": snippet})
    return {"success": True, "provider": "wikipedia", "query": q,
            "results": results, "note": "INTERNET_SOURCED"}


def duckduckgo_instant_answer_search(query: str, timeout_seconds: int = 15) -> Dict[str, Any]:
    """DuckDuckGo Instant Answer API (no key). Limited structured results."""
    if not query or not query.strip():
        return {"success": False, "error": "Empty query"}
    q = query.strip()
    url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode(
        {"q": q, "format": "json", "no_redirect": "1", "no_html": "1", "skip_disambig": "1"}
    )
    raw = _http_get(url, timeout_seconds=timeout_seconds, accept="application/json")
    if raw is None:
        return {"success": False, "error": "request_failed", "provider": "duckduckgo_instant_answer"}
    try:
        data = json.loads(raw)
    except Exception as e:
        return {"success": False, "error": str(e), "provider": "duckduckgo_instant_answer"}

    results: List[Dict[str, str]] = []
    abstract = (data.get("AbstractText") or "").strip()
    if abstract:
        results.append({"title": (data.get("Heading") or "DuckDuckGo Abstract").strip(),
                        "url": (data.get("AbstractURL") or "").strip(), "snippet": abstract})

    def _walk(items: list) -> None:
        for it in items or []:
            if isinstance(it, dict) and "Topics" in it:
                _walk(it.get("Topics") or [])
                continue
            if not isinstance(it, dict):
                continue
            txt = (it.get("Text") or "").strip()
            if txt:
                results.append({"title": txt[:120], "url": (it.get("FirstURL") or "").strip(),
                                "snippet": txt})

    _walk(data.get("RelatedTopics") or [])
    seen, deduped = set(), []
    for r in results:
        key = (r.get("url", "") + "|" + r.get("snippet", "")).lower()
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    return {"success": True, "provider": "duckduckgo_instant_answer", "query": q,
            "results": deduped[:10], "note": "INTERNET_SOURCED"}


def internet_search(query: str, timeout_seconds: int = 15) -> Dict[str, Any]:
    """Best-effort key-free search (real web links via DuckDuckGo HTML, then
    Wikipedia + instant-answer as fallbacks). Kept for backward compatibility."""
    q = (query or "").strip()
    if not q:
        return {"success": False, "error": "Empty query"}

    results: List[Dict[str, Any]] = []
    providers: List[str] = []

    ddg_html = _provider_duckduckgo_html(q, max_results=10, timeout_seconds=timeout_seconds)
    if ddg_html:
        providers.append("duckduckgo")
        results.extend(ddg_html)

    if len(results) < 3:
        ia = duckduckgo_instant_answer_search(q, timeout_seconds=timeout_seconds)
        if ia.get("success") and ia.get("results"):
            providers.append("duckduckgo_instant_answer")
            results.extend(ia["results"])

    if len(results) < 3:
        wiki = _provider_wikipedia(q, max_results=8, timeout_seconds=timeout_seconds)
        if wiki:
            providers.append("wikipedia")
            results.extend(wiki)

    if not results:
        return {"success": False, "error": "No results", "query": q,
                "providers_attempted": providers or ["duckduckgo", "wikipedia"],
                "note": "INTERNET_SOURCED"}

    # De-dup by URL.
    seen, deduped = set(), []
    for r in results:
        u = (r.get("url") or "").strip().lower()
        k = u or (r.get("snippet", "")[:80]).lower()
        if k and k not in seen:
            seen.add(k)
            deduped.append(r)
    return {"success": True, "providers": providers, "query": q,
            "results": deduped[:10], "note": "INTERNET_SOURCED"}


def internet_search_variants(queries: List[str], timeout_seconds: int = 15) -> Dict[str, Any]:
    """Try several query phrasings until one returns usable results."""
    tried: List[str] = []
    for q in queries:
        q = (q or "").strip()
        if not q or q.lower() in {t.lower() for t in tried}:
            continue
        tried.append(q)
        res = internet_search(q, timeout_seconds=timeout_seconds)
        if res.get("success") and (res.get("results") or []):
            res["variants_tried"] = tried
            return res
    return {"success": False, "error": "No results from any query variant",
            "query": tried[0] if tried else "", "variants_tried": tried,
            "note": "INTERNET_SOURCED"}


# ==========================================================================
# 7. Public: full multi-stage research pipeline
# ==========================================================================

def multi_source_search(
    queries: List[str],
    *,
    max_results_per_query: int = 12,
    timeout_seconds: int = 12,
) -> List[Dict[str, Any]]:
    """Run every query against the best available provider and aggregate +
    de-duplicate the raw results (over-fetch stage)."""
    provider_name, api_key = _configured_search_api()
    aggregated: List[Dict[str, Any]] = []

    for q in queries:
        q = (q or "").strip()
        if not q:
            continue
        batch: List[Dict[str, Any]] = []
        if provider_name == "tavily":
            batch = _provider_tavily(q, api_key, max_results_per_query, timeout_seconds)
        elif provider_name == "brave":
            batch = _provider_brave(q, api_key, max_results_per_query, timeout_seconds)
        elif provider_name == "serpapi":
            batch = _provider_serpapi(q, api_key, max_results_per_query, timeout_seconds)

        # Always supplement with key-free web results (and as full fallback).
        if not batch:
            batch = _provider_duckduckgo_html(q, max_results_per_query, timeout_seconds)
        # Wikipedia as an authoritative supplement for entity/topic questions.
        batch.extend(_provider_wikipedia(q, max_results=4, timeout_seconds=timeout_seconds))

        for r in batch:
            r.setdefault("source_query", q)
            aggregated.append(r)

    # De-dup by normalised URL, keeping the richest snippet.
    by_url: Dict[str, Dict[str, Any]] = {}
    for r in aggregated:
        u = (r.get("url") or "").strip()
        if not u:
            continue
        key = u.lower().rstrip("/")
        if key not in by_url:
            by_url[key] = r
        else:
            if len(r.get("snippet", "")) > len(by_url[key].get("snippet", "")):
                by_url[key]["snippet"] = r.get("snippet", "")
    return list(by_url.values())


def _fetch_and_extract(result: Dict[str, Any], question_tokens: List[str], timeout_seconds: int) -> Dict[str, Any]:
    """Fetch a page and attach the most relevant extracted chunk as evidence."""
    url = result.get("url", "")
    page = _http_get(url, timeout_seconds=timeout_seconds, accept="text/html")
    if not page:
        return result
    text = html_to_text(page)
    if not text:
        return result
    chunks = chunk_text(text, chunk_chars=700, max_chunks=10)
    if not chunks:
        return result
    best, best_score = "", -1.0
    for c in chunks:
        s = _lexical_overlap(question_tokens, c)
        if s > best_score:
            best, best_score = c, s
    enriched = dict(result)
    enriched["extracted"] = best[:1200]
    enriched["extract_score"] = round(best_score, 4)
    # Strengthen the snippet with the best on-page evidence when more relevant.
    if best_score > _lexical_overlap(question_tokens, result.get("snippet", "")):
        enriched["snippet"] = (best[:500] or result.get("snippet", "")).strip()
    return enriched


def research(
    question: str,
    *,
    query_variants: Optional[List[str]] = None,
    max_sources: int = 8,
    fetch_pages: int = 4,
    timeout_seconds: int = 12,
    read_pages: bool = True,
) -> Dict[str, Any]:
    """Domain-agnostic multi-stage retrieval producing an *evidence pack*.

    Stages: variants → multi-source retrieval (over-fetch) → trust+relevance
    re-rank → page reading (top sources) → evidence + cross-source confidence.

    Returns a dict:
        {
          success, question, queries, evidence: [ {title,url,domain,trust,
              relevance,rank_score,snippet,extracted} ],
          sources: [urls], confidence: 0..1, providers, note
        }
    """
    base_q = " ".join((question or "").split()).strip()
    if not base_q:
        return {"success": False, "error": "empty_question", "question": question}

    variants = [v for v in (query_variants or []) if v and v.strip()]
    for v in rule_based_query_variants(base_q):
        if v.lower() not in {x.lower() for x in variants}:
            variants.append(v)
    variants = variants[:6] or [base_q]

    raw = multi_source_search(variants, max_results_per_query=12, timeout_seconds=timeout_seconds)
    if not raw:
        return {"success": False, "error": "no_web_results", "question": base_q,
                "queries": variants, "note": "INTERNET_SOURCED"}

    ranked = rerank_results(base_q, raw, top_k=max(max_sources, fetch_pages))
    qtok = _content_tokens(base_q)

    # Page reading (concurrent) for the top sources — most value vs snippet-only.
    if read_pages and fetch_pages > 0:
        to_fetch = ranked[:fetch_pages]
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(5, len(to_fetch) or 1)) as ex:
                futures = {ex.submit(_fetch_and_extract, r, qtok, min(8, timeout_seconds)): i
                           for i, r in enumerate(to_fetch)}
                for fut in concurrent.futures.as_completed(futures, timeout=timeout_seconds + 6):
                    i = futures[fut]
                    try:
                        ranked[i] = fut.result()
                    except Exception:
                        pass
        except Exception as e:
            logger.debug("Page-reading stage failed/partial: %s", e)
        # Re-rank again now that on-page evidence has enriched snippets.
        ranked = rerank_results(base_q, ranked, top_k=max_sources)
    else:
        ranked = ranked[:max_sources]

    evidence = [r for r in ranked if (r.get("snippet") or r.get("extracted"))]
    # Cross-source agreement → confidence: distinct trustworthy domains that are
    # actually on-topic. (General signal; no per-topic logic.)
    on_topic = [r for r in evidence if r.get("relevance", 0) >= 0.34]
    distinct_domains = {r.get("domain") for r in on_topic if r.get("domain")}
    avg_trust = (sum(r.get("trust", 0.0) for r in on_topic) / len(on_topic)) if on_topic else 0.0
    confidence = min(1.0, 0.18 * len(distinct_domains) + 0.5 * avg_trust)

    providers = sorted({r.get("provider", "") for r in raw if r.get("provider")})
    return {
        "success": True,
        "question": base_q,
        "queries": variants,
        "evidence": evidence,
        "sources": [r.get("url") for r in evidence if r.get("url")],
        "confidence": round(confidence, 3),
        "distinct_domains": len(distinct_domains),
        "providers": providers,
        "note": "INTERNET_SOURCED",
    }
