"""
Coordinate parsing utilities (survey-aware).

Goals:
- Extract coordinates from free-form text (including DMS/DM formats and hemisphere letters)
- Normalize geodetic coordinates to decimal degrees
- Provide lightweight CRS inference helpers (EPSG/UTM/WGS84 hints)

This is intentionally heuristic and conservative: when ambiguous, we avoid guessing
silently and instead return fewer results.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ParsedPoint:
    """A parsed coordinate pair normalized to the x/y expected by pyproj(always_xy=True)."""

    # Always XY order as expected by pyproj(always_xy=True):
    # - for geodetic: x=lon, y=lat
    # - for projected: x=easting, y=northing
    x: float
    y: float
    kind: str  # "geodetic" | "projected"
    source_text: str
    notes: str = ""


_NUM = r"[-+]?\d+(?:\.\d+)?"


def _to_float(num_text: str) -> float:
    # Accept "123,456.78"
    return float(num_text.replace(",", "").strip())


def dms_to_decimal(deg: float, minutes: float = 0.0, seconds: float = 0.0, hemisphere: Optional[str] = None) -> float:
    """Convert degrees/minutes/seconds to signed decimal degrees."""
    hemi = (hemisphere or "").strip().upper() or None
    sign = -1.0 if deg < 0 else 1.0
    if hemi in ("S", "W"):
        sign = -1.0
    if hemi in ("N", "E"):
        sign = 1.0

    dec = abs(deg) + (abs(minutes) / 60.0) + (abs(seconds) / 3600.0)
    return sign * dec


def parse_angle(text: str) -> Optional[float]:
    """
    Parse a single angular coordinate in common survey formats:
    - Decimal degrees: 6.1234, -1.2345
    - Hemisphere suffix/prefix: 6.1234N, W 3.4567
    - DMS/DM: 6°12'30.5\"N, 6 12 30.5 N, 6°12.5'N
    """
    if not text or not str(text).strip():
        return None

    raw = str(text).strip()
    u = raw.upper()

    hemi = None
    m_hemi = re.search(r"\b([NSEW])\b", u)
    if m_hemi:
        hemi = m_hemi.group(1)

    # Remove hemisphere letters and degree/min/sec symbols to extract numbers
    cleaned = re.sub(r"[NSEW]", " ", u)
    cleaned = cleaned.replace("°", " ").replace("º", " ").replace("'", " ").replace('"', " ")
    cleaned = cleaned.replace("D", " ").replace("M", " ").replace("S", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    nums = re.findall(_NUM, cleaned)
    if not nums:
        return None

    try:
        values = [_to_float(n) for n in nums[:3]]
    except Exception:
        return None

    if len(values) == 1:
        deg = values[0]
        return dms_to_decimal(deg, 0.0, 0.0, hemisphere=hemi)
    if len(values) == 2:
        deg, minutes = values
        return dms_to_decimal(deg, minutes, 0.0, hemisphere=hemi)
    deg, minutes, seconds = values[0], values[1], values[2]
    return dms_to_decimal(deg, minutes, seconds, hemisphere=hemi)


def _looks_like_lat(v: float) -> bool:
    return abs(v) <= 90.0


def _looks_like_lon(v: float) -> bool:
    return abs(v) <= 180.0


def _parse_geodetic_pair_from_tokens(a: str, b: str) -> Optional[ParsedPoint]:
    a_u, b_u = a.upper(), b.upper()
    a_has_ns = bool(re.search(r"\b[NS]\b", a_u) or re.search(r"[NS]$", a_u))
    a_has_ew = bool(re.search(r"\b[EW]\b", a_u) or re.search(r"[EW]$", a_u))
    b_has_ns = bool(re.search(r"\b[NS]\b", b_u) or re.search(r"[NS]$", b_u))
    b_has_ew = bool(re.search(r"\b[EW]\b", b_u) or re.search(r"[EW]$", b_u))

    a_val = parse_angle(a)
    b_val = parse_angle(b)
    if a_val is None or b_val is None:
        return None

    # Hemisphere-driven ordering (most reliable)
    if a_has_ns and b_has_ew:
        lat, lon = a_val, b_val
        return ParsedPoint(x=lon, y=lat, kind="geodetic", source_text=f"{a} {b}", notes="lat(N/S) then lon(E/W)")
    if a_has_ew and b_has_ns:
        lon, lat = a_val, b_val
        return ParsedPoint(x=lon, y=lat, kind="geodetic", source_text=f"{a} {b}", notes="lon(E/W) then lat(N/S)")
    if b_has_ns and a_has_ew:
        lon, lat = a_val, b_val
        return ParsedPoint(x=lon, y=lat, kind="geodetic", source_text=f"{a} {b}", notes="lon(E/W) then lat(N/S)")
    if b_has_ew and a_has_ns:
        lat, lon = a_val, b_val
        return ParsedPoint(x=lon, y=lat, kind="geodetic", source_text=f"{a} {b}", notes="lat(N/S) then lon(E/W)")

    # Value-driven ordering (fallback)
    if _looks_like_lat(a_val) and _looks_like_lon(b_val) and not (_looks_like_lon(a_val) and _looks_like_lat(b_val)):
        lat, lon = a_val, b_val
        return ParsedPoint(x=lon, y=lat, kind="geodetic", source_text=f"{a} {b}", notes="value-heuristic lat,lon")
    if _looks_like_lon(a_val) and _looks_like_lat(b_val):
        lon, lat = a_val, b_val
        return ParsedPoint(x=lon, y=lat, kind="geodetic", source_text=f"{a} {b}", notes="value-heuristic lon,lat")

    return None


def extract_points(text: str, max_points: int = 20) -> List[ParsedPoint]:
    """
    Extract coordinate pairs from free-form text.

    Supports:
    - Projected: "E 123456.78 N 7654321.00", "X=... Y=...", "Easting: ... Northing: ..."
    - Geodetic: "6°12'30.5\"N 3°21'10\"E", "6.1234N, 3.4567E", "lat 6 12 30 N lon 3 21 10 E"
    - Geodetic numeric pairs: "6.1234, 3.4567" (heuristic)
    """
    if not text:
        return []

    t = str(text)
    out: List[ParsedPoint] = []

    # --- Projected patterns (E/N, X/Y) ---
    projected_patterns = [
        # Easting/Northing with labels
        re.compile(
            rf"\bE(?:ASTING)?\s*[:=]?\s*(?P<x>\d[\d,]*\.?\d*)\s*[,;\s]+N(?:ORTHING)?\s*[:=]?\s*(?P<y>\d[\d,]*\.?\d*)\b",
            flags=re.IGNORECASE,
        ),
        re.compile(
            rf"\bX\s*[:=]?\s*(?P<x>\d[\d,]*\.?\d*)\s*[,;\s]+Y\s*[:=]?\s*(?P<y>\d[\d,]*\.?\d*)\b",
            flags=re.IGNORECASE,
        ),
    ]

    for pat in projected_patterns:
        for m in pat.finditer(t):
            if len(out) >= max_points:
                return out
            try:
                x = _to_float(m.group("x"))
                y = _to_float(m.group("y"))
            except Exception:
                continue
            # Conservative: UTM-like magnitudes (avoid matching years, etc.)
            if abs(x) < 1000 or abs(y) < 1000:
                continue
            out.append(ParsedPoint(x=x, y=y, kind="projected", source_text=m.group(0), notes="E/N or X/Y"))

    # --- Geodetic tokens with hemisphere or degree sign ---
    # Capture "token token" where each token contains either ° or hemisphere marker.
    geo_token = r"(?:[NSEW]\s*)?(?:\d[\d,]*\.?\d*(?:\s*[°º]\s*\d[\d,]*\.?\d*)?(?:\s*'\s*\d[\d,]*\.?\d*)?(?:\s*\"\s*\d[\d,]*\.?\d*)?\s*[NSEW]?)"
    geo_pair_pat = re.compile(rf"(?P<a>{geo_token})\s*[,;\s]\s*(?P<b>{geo_token})", flags=re.IGNORECASE)

    for m in geo_pair_pat.finditer(t):
        if len(out) >= max_points:
            return out
        a = m.group("a").strip()
        b = m.group("b").strip()
        # Only consider pairs likely to be geodetic (° or hemisphere present in either)
        if not (re.search(r"[°º]", a) or re.search(r"[°º]", b) or re.search(r"[NSEW]", a, re.I) or re.search(r"[NSEW]", b, re.I)):
            continue
        p = _parse_geodetic_pair_from_tokens(a, b)
        if p:
            # Guard rails
            if _looks_like_lon(p.x) and _looks_like_lat(p.y):
                out.append(p)

    # --- Plain decimal pair fallback (heuristic) ---
    # Example: "6.12345, 3.45678"
    dec_pair = re.compile(rf"\b(?P<a>{_NUM})\s*[,/]\s*(?P<b>{_NUM})\b")
    for m in dec_pair.finditer(t):
        if len(out) >= max_points:
            return out
        try:
            a = _to_float(m.group("a"))
            b = _to_float(m.group("b"))
        except Exception:
            continue
        # Only accept if it looks like geodetic-ish
        if _looks_like_lat(a) and _looks_like_lon(b):
            out.append(ParsedPoint(x=b, y=a, kind="geodetic", source_text=m.group(0), notes="decimal lat,lon"))
        elif _looks_like_lon(a) and _looks_like_lat(b):
            out.append(ParsedPoint(x=a, y=b, kind="geodetic", source_text=m.group(0), notes="decimal lon,lat"))

    return out


def infer_crs_from_text(text: str) -> Dict[str, Any]:
    """
    Heuristic CRS inference from a free-form query.

    Returns keys:
    - source_crs, target_crs: Optional[str]
    - source_zone, target_zone: Optional[int] (rarely needed; often encoded in UTM string)
    """
    q = (text or "").strip()
    q_low = q.lower()

    def _clean_crs(s: str) -> str:
        s2 = re.sub(r"\s+", " ", (s or "").strip())
        # avoid trailing punctuation
        s2 = s2.strip(" ,;:.")
        return s2

    # EPSG explicit
    epsg_codes = re.findall(r"(?i)\bEPSG\s*[: ]\s*(\d{3,6})\b", q)
    wkid_codes = re.findall(r"(?i)\bWKID\s*[: ]\s*(\d{3,6})\b", q)
    codes = [*epsg_codes, *wkid_codes]

    # "from ... to ..." extraction
    m = re.search(r"(?i)\bfrom\s+([^,\n;:]+?)\s+\bto\s+([^,\n;:]+)", q)
    src = _clean_crs(m.group(1)) if m else None
    dst = _clean_crs(m.group(2)) if m else None

    # If EPSG codes appear and from/to wasn't clean, map in order
    if codes and (not src or not dst):
        if len(codes) >= 2:
            src = src or f"EPSG:{codes[0]}"
            dst = dst or f"EPSG:{codes[1]}"
        elif len(codes) == 1:
            # If query says "to EPSG:xxxx" treat as target; else source
            if " to " in q_low:
                dst = dst or f"EPSG:{codes[0]}"
            else:
                src = src or f"EPSG:{codes[0]}"

    # UTM hint (keep as a user-friendly string; converter can resolve)
    utm = re.search(r"(?i)\bUTM\b.*?\bZONE\b\s*(\d{1,2})\s*([NS])\b", q)
    if utm:
        zone = int(utm.group(1))
        hemi = utm.group(2).upper()
        utm_str = f"UTM Zone {zone}{hemi}"
        if src and "utm" in src.lower():
            src = utm_str
        elif dst and "utm" in dst.lower():
            dst = utm_str
        elif not src:
            src = utm_str

    # Common named systems
    if not src:
        if "wgs84" in q_low or "wgs 84" in q_low:
            src = "WGS84"
    if not dst:
        if "wgs84" in q_low or "wgs 84" in q_low:
            dst = "WGS84"

    return {
        "source_crs": src,
        "target_crs": dst,
        "source_zone": None,
        "target_zone": None,
    }


