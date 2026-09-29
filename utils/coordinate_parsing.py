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


# Official EPSG-style datum labels, longest/most specific first.
# Used to keep NAD83/ETRS89/SIRGAS/etc. on UTM instead of silently substituting WGS 84.
_NAMED_DATUM_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (r"international terrestrial reference frame\s*2014", "ITRF2014"),
    (r"international terrestrial reference frame\s*2008", "ITRF2008"),
    (r"international terrestrial reference frame\s*2000", "ITRF2000"),
    (r"nad\s*83\s*\(\s*2011\s*\)", "NAD83(2011)"),
    (r"nad\s*83\s*\(\s*csrs\s*\)", "NAD83(CSRS)"),
    (r"nad\s*83\s*\(\s*harn\s*\)", "NAD83(HARN)"),
    (r"nad\s*83\s*\(\s*nsrs2007\s*\)", "NAD83(NSRS2007)"),
    (r"sirgas\s*2000", "SIRGAS 2000"),
    (r"hartebeesthoek\s*94|hartebeesthoek94", "Hartebeesthoek94"),
    (r"timbalai\s*1948", "Timbalai 1948"),
    (r"pulkovo\s*1942", "Pulkovo 1942"),
    (r"pulkovo\s*1995", "Pulkovo 1995"),
    (r"arc\s*1960", "Arc 1960"),
    (r"arc\s*1950", "Arc 1950"),
    (r"gda\s*2020|gda2020", "GDA2020"),
    (r"gda\s*94|gda94", "GDA94"),
    (r"etrs[\s\-]?89", "ETRS89"),
    (r"itrf[\s\-]?2014", "ITRF2014"),
    (r"itrf[\s\-]?2008", "ITRF2008"),
    (r"itrf[\s\-]?2000", "ITRF2000"),
    (r"nad\s*83", "NAD83"),
    (r"nad\s*27", "NAD27"),
    (r"sad[\s\-]?69", "SAD69"),
    (r"psad[\s\-]?56", "PSAD56"),
    (r"ed[\s\-]?50", "ED50"),
    (r"wgs[\s\-]?84", "WGS 84"),
    (r"osgb[\s\-]?36|british national grid", "OSGB36"),
    (r"nzgd[\s\-]?2000", "NZGD2000"),
    (r"nzgd[\s\-]?49", "NZGD49"),
    (r"agd\s*66|agd66", "AGD66"),
    (r"agd\s*84|agd84", "AGD84"),
    (r"kertau(?:\s*1968)?", "Kertau 1968"),
    (r"tokyo", "Tokyo"),
    (r"minna", "Minna"),
)

_ELLIPSOID_ONLY_RE = re.compile(
    r"^(?:grs[\s\-]?80|grs80|clarke[\s\-]?1880|clarke[\s\-]?1866|"
    r"airy(?:\s*1830)?|bessel(?:\s*1841)?|krass?ovsky(?:\s*1940)?|"
    r"everest(?:\s*1830)?|wgs[\s\-]?72)$",
    flags=re.IGNORECASE,
)

# Tokens that mean the user named a real CRS/datum (not "convert this Excel to CAD").
CRS_FAMILY_TOKEN = (
    r"utm|wgs|epsg|wkid|minna|nad|osgb|gda|mga|etrs|itrf|sirgas|sad|psad|"
    r"ed\s*50|ed50|arc\s*19|hartebeest|pulkovo|nzgd|agd|ntm|bng|lambert|"
    r"mercator|stereographic|tokyo|kertau|timbalai|nsidc|clarke|datum"
)


def extract_named_datum(text: str) -> Optional[str]:
    """Return a canonical datum label if the text names one (else None)."""
    raw = (text or "").strip()
    if not raw:
        return None
    low = re.sub(r"[\s_\-]+", " ", raw.lower())
    for pat, label in _NAMED_DATUM_PATTERNS:
        if re.search(pat, low, flags=re.IGNORECASE):
            return label
    return None


def is_ellipsoid_only_crs_label(text: str) -> bool:
    """True when the string is an ellipsoid name, not a coordinate reference system."""
    t = re.sub(r"[\s_\-]+", " ", (text or "").strip().lower())
    return bool(t and _ELLIPSOID_ONLY_RE.match(t))


def _utm_zone_hemi(text: str) -> Optional[Tuple[str, str]]:
    low = (text or "").lower()
    m = re.search(r"utm\s*(?:zone\s*)?(\d{1,2})\s*([ns])\b", low)
    if not m:
        m = re.search(r"\butm\b\s*(?:zone\s*)?(\d{1,2})\b", low)
        if not m:
            return None
        return m.group(1), ""
    return m.group(1), m.group(2).upper()


def canonical_projected_crs_label(text: str) -> Optional[str]:
    """
    Keep datum + projection together.

    'ETRS89 / UTM zone 32N' stays that name. Bare 'UTM Zone 32N' stays WGS 84 UTM wording.
    """
    label = (text or "").strip()
    if not label:
        return None
    low = label.lower()
    datum = extract_named_datum(label)

    mga = re.search(r"\bmga\s*(?:zone\s*)?(\d{1,2})\b", low)
    if mga and datum in ("GDA94", "GDA2020"):
        return f"{datum} / MGA zone {mga.group(1)}"

    um = _utm_zone_hemi(label)
    if um:
        zone, hemi = um
        if datum and datum != "WGS 84":
            if hemi:
                return f"{datum} / UTM zone {zone}{hemi}"
            return f"{datum} / UTM zone {zone}"
        if datum == "WGS 84":
            if hemi:
                return f"WGS 84 / UTM zone {zone}{hemi}"
            return f"WGS 84 / UTM zone {zone}"
        if hemi:
            return f"UTM Zone {zone}{hemi}"
        return f"UTM Zone {zone}"
    return None


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
        # Trim trailing clause words that are not part of a CRS name.
        s2 = re.split(
            r"\b(?:and|then|using|with|save|plot|generate|create|before|after)\b",
            s2,
            maxsplit=1,
            flags=re.I,
        )[0]
        s2 = s2.strip(" ,;:.\"'()[]")
        return s2

    # EPSG explicit
    epsg_codes = re.findall(r"(?i)\bEPSG\s*[: ]\s*(\d{3,6})\b", q)
    wkid_codes = re.findall(r"(?i)\bWKID\s*[: ]\s*(\d{3,6})\b", q)
    codes = [*epsg_codes, *wkid_codes]

    # "from … to/into …" — stop before the next clause (save / plot / comma).
    m = re.search(
        r"(?i)\bfrom\s+(.+?)\s+\b(?:to|into)\s+(.+?)(?:"
        r",|\band\s+save\b|\bthen\b|\busing\b|\bwith\b|\band\s+use\b|"
        r"\bplot\b|\bgenerate\b|\n|$)",
        q,
    )
    src = _clean_crs(m.group(1)) if m else None
    dst = _clean_crs(m.group(2)) if m else None
    if not src or not dst:
        m2 = re.search(
            r"(?i)\b(?:convert(?:ed|ing)?|reproject(?:ed|ing)?|transform(?:ed|ing)?)\s+"
            r"(?:(?:it|them|coordinates?|these|the)\s+)?"
            r"(?:from\s+)?"
            rf"((?:(?!\b(?:to|into)\b).)*?(?:{CRS_FAMILY_TOKEN})[^,\n;]*?)"
            r"\s+\b(?:to|into)\s+"
            r"(.+?)(?:"
            r",|\band\s+save\b|\bthen\b|\busing\b|\bwith\b|\band\s+use\b|"
            r"\bplot\b|\bgenerate\b|\n|$)",
            q,
        )
        if m2:
            src = src or _clean_crs(m2.group(1))
            dst = dst or _clean_crs(m2.group(2))

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

    def _canon_named(label: Optional[str]) -> Optional[str]:
        if not label:
            return label
        low = label.lower()
        if "minna" in low and re.search(r"mid[\s\-]?belt", low):
            return "Minna / Nigeria Mid Belt"
        if "minna" in low and re.search(r"west[\s\-]?belt", low):
            return "Minna / Nigeria West Belt"
        if "minna" in low and re.search(r"east[\s\-]?belt", low):
            return "Minna / Nigeria East Belt"
        projected = canonical_projected_crs_label(label)
        if projected:
            return projected
        if re.search(r"\bwgs\s*84\b", low) and "utm" not in low and "ups" not in low:
            return "WGS84"
        return label

    # Bare UTM in the query fills a missing source only when that side has no datum.
    if (not src) or (src and "utm" in src.lower() and not extract_named_datum(src)):
        projected = canonical_projected_crs_label(src or q)
        if projected and "UTM" in projected.upper():
            if not src:
                src = projected
            elif src and extract_named_datum(src) is None:
                src = projected

    src = _canon_named(src)
    dst = _canon_named(dst)

    if not dst and re.search(r"minna", q_low) and re.search(r"mid[\s\-]?belt", q_low):
        if not (src and "minna" in src.lower()):
            dst = "Minna / Nigeria Mid Belt"

    # Common named systems — fill only a missing *source*, never invent a target
    # just because WGS84 appeared as the from-system.
    if not src:
        if "wgs84" in q_low or "wgs 84" in q_low:
            src = "WGS84"

    return {
        "source_crs": src,
        "target_crs": dst,
        "source_zone": None,
        "target_zone": None,
    }


