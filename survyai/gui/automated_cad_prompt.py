"""Compose and parse Automated CAD section values as conventional cadastral prompts.

The Console still accepts free-form CAD prompts. This module only translates the
structured form (and the default CAD prompt template) into the same field language
the cadastral fastpath already understands.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

MODE_COORDINATES = "coordinates"
MODE_BEARINGS = "bearings"

FENCE_NONE = "None"
FENCE_DWARF = "Dwarf Wall Fence"
FENCE_CONCRETE = "Concrete Wall Fence"

ADJUST_BEARING = "bearing"
ADJUST_BOWDITCH = "bowditch"

# Canonical sentences the cadastral engine already honours.
BEARING_ADJUST_INSTRUCTION = (
    "Use bearing adjustment (hold distances constant) to close the traverse."
)
BOWDITCH_ADJUST_INSTRUCTION = (
    "Use Bowditch adjustment method to close the traverse."
)
CAD_SESSION_KEEP_LINE = (
    "Keep this generated parcel (pillars, coordinates or bearings and distances, "
    "area, and the output DWG) available for follow-up site analysis in this conversation."
)

# Scales the plot engine will honour from an explicit prompt (see agent/prompts.py).
ALLOWED_SCALE_DENOMS = frozenset(
    {250, 500, 1000, 2000, 2500, 5000, 10000, 20000, 25000}
)

_NUM = r"[-+]?\d+(?:\.\d+)?"
_PILLAR_TOKEN_RE = re.compile(
    r"\b[A-Z]{1,8}\s*/\s*[A-Z]{0,10}\s*\d+[A-Z]{0,4}"
    r"|\b[A-Z]{2,10}\s+\d+[A-Z]{0,4}\b",
    re.IGNORECASE,
)
_COORD_PAIR_RE = re.compile(
    rf"\(\s*({_NUM})\s*(?:m\s*)?E\s*[,;]\s*({_NUM})\s*(?:m\s*)?N\s*\)"
    rf"|\(\s*({_NUM})\s*[,;]\s*({_NUM})\s*\)",
    re.IGNORECASE,
)
_BEARING_LEG_RE = re.compile(
    r"bearing\s*[:=]?\s*"
    rf"({_NUM})\s*(?:d|deg|degree|degrees)?\s*"
    rf"(?:({_NUM})\s*(?:m|min|minute|minutes|'))?\s*"
    rf"(?:({_NUM})\s*(?:s|sec|second|seconds|\"|″)?)?\s*"
    r"[,;]?\s*"
    rf"(?:measured\s+)?(?:distance|dist\.?)\s*[:=]?\s*"
    rf"({_NUM})\s*(?:m\b)?",
    re.IGNORECASE,
)
_METADATA_LABELS: Tuple[Tuple[str, str], ...] = (
    ("buyer_name", r"buyer\s*names?"),
    ("location", r"location"),
    ("lga", r"local\s+gov(?:ernment|t\.?)\s*area"),
    ("state", r"\bstate\b"),
    ("origin", r"origin_crs|crs_origin|\borigin\b"),
    ("plan_number", r"plan\s*(?:number|no\.?)"),
    ("cert_date", r"date\s+on\s+the\s+certification|certification\s+date"),
    ("surveyor_name", r"surveyor\s+name"),
    ("surveyor_company_address", r"surveyor\s+company\s+and\s+address"),
    ("surveyor_company", r"surveyor\s+company"),
    ("surveyor_address", r"surveyor\s+address"),
    ("pillars", r"pillar\s+numbers?"),
    ("scale", r"plot\s+using\s+scale|\bscale\b"),
)


@dataclass
class TraverseLeg:
    degrees: str = ""
    minutes: str = ""
    seconds: str = ""
    distance: str = ""


@dataclass
class AccessRoad:
    width: str = ""
    start_pillar: str = ""
    end_pillar: str = ""


@dataclass
class WallFence:
    fence_type: str = FENCE_NONE
    start_pillar: str = ""
    end_pillar: str = ""


@dataclass
class CadFormState:
    mode: str = MODE_COORDINATES
    save_as: str = ""
    owners: List[str] = field(default_factory=lambda: [""])
    location: str = ""
    lga: str = ""
    state: str = ""
    auto_scale: bool = True
    scale: str = ""
    origin: str = ""
    pillars: List[str] = field(default_factory=lambda: ["", "", ""])
    coordinates: List[str] = field(default_factory=lambda: ["", "", ""])
    start_coordinate: str = ""
    legs: List[TraverseLeg] = field(
        default_factory=lambda: [TraverseLeg(), TraverseLeg(), TraverseLeg()]
    )
    roads: List[AccessRoad] = field(default_factory=lambda: [AccessRoad()])
    fences: List[WallFence] = field(default_factory=lambda: [WallFence()])
    plan_number: str = ""
    surveyor_name: str = ""
    surveyor_company: str = ""
    surveyor_address: str = ""
    traverse_adjustment: str = ADJUST_BEARING


def ordinal_label(n: int) -> str:
    """1 → 1st, 2 → 2nd, 3 → 3rd, 11 → 11th, 21 → 21st."""
    if n <= 0:
        n = 1
    if 10 <= (n % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def parse_scale_denom(text: str) -> Optional[int]:
    """Return an allowed scale denominator, or None for blank/invalid (auto-scale)."""
    raw = (text or "").strip().replace(" ", "")
    if not raw:
        return None
    m = re.fullmatch(r"(?:1[:/])?(\d+)", raw, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"1\s*[:/]\s*(\d+)", text or "", flags=re.IGNORECASE)
        if not m:
            m = re.search(r"(\d+)", text or "")
            if not m:
                return None
    try:
        denom = int(m.group(1))
    except (TypeError, ValueError):
        return None
    if denom in ALLOWED_SCALE_DENOMS:
        return denom
    return None


def parse_en_pair(text: str) -> Optional[Tuple[float, float]]:
    """Parse an Easting, Northing pair from a value box or prompt fragment."""
    raw = (text or "").strip()
    if not raw:
        return None
    m = _COORD_PAIR_RE.search(raw)
    if m:
        if m.group(1) is not None:
            return float(m.group(1)), float(m.group(2))
        return float(m.group(3)), float(m.group(4))
    cleaned = re.sub(r"(?:m\s*)?[EN]\b", " ", raw, flags=re.IGNORECASE)
    cleaned = cleaned.replace("(", " ").replace(")", " ")
    parts = [p.strip() for p in re.split(r"[,;]\s*", cleaned) if p.strip()]
    nums: List[float] = []
    for part in parts:
        token = part.strip()
        if re.fullmatch(_NUM, token):
            nums.append(float(token))
    if len(nums) >= 2:
        return nums[0], nums[1]
    tokens = [t for t in re.findall(_NUM, cleaned)]
    if len(tokens) >= 2:
        return float(tokens[0]), float(tokens[1])
    return None


def format_en_pair(easting: float, northing: float) -> str:
    return f"({_fmt_num(easting)}mE, {_fmt_num(northing)}mN)"


def format_en_display(easting: float, northing: float) -> str:
    return f"{_fmt_num(easting)}, {_fmt_num(northing)}"


def _fmt_num(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _sanitize_dwg_stem(name: str) -> str:
    cleaned = (name or "").strip()
    if cleaned.lower().endswith(".dwg"):
        cleaned = cleaned[:-4].strip()
    cleaned = re.sub(r'[<>:"/\\|?*]', "", cleaned)
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = cleaned.strip("._") or "survey_plan"
    return cleaned[:80]


def _nonempty(values: Iterable[str]) -> List[str]:
    return [v.strip() for v in values if str(v or "").strip()]


def _join_owners(owners: Sequence[str]) -> str:
    """Join owners the way the title-block formatter expects (AND before the last)."""
    names = _nonempty(owners)
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} AND {names[1]}"
    return ", ".join(names[:-1]) + f" AND {names[-1]}"


def compose_cad_prompt(state: CadFormState) -> Tuple[str, str]:
    """Build a conventional cadastral prompt. Returns (prompt, error)."""
    mode = (state.mode or MODE_COORDINATES).strip().lower()
    if mode not in {MODE_COORDINATES, MODE_BEARINGS}:
        mode = MODE_COORDINATES

    owners = _nonempty(state.owners)
    pillars = _nonempty(state.pillars)
    lines: List[str] = []

    save_as = (state.save_as or "").strip()
    stem_source = save_as or (owners[0] if owners else "survey_plan")
    stem = _sanitize_dwg_stem(stem_source)
    lines.append(f"Generate '{stem}.dwg' in the same folder as this project")
    if owners:
        lines.append(f"buyer name: {_join_owners(owners)}")
    if state.location.strip():
        lines.append(f"location: {state.location.strip()}")
    if state.lga.strip():
        lines.append(f"local government area: {state.lga.strip()}")
    if state.state.strip():
        lines.append(f"state: {state.state.strip()}")
    if state.origin.strip():
        lines.append(f"origin_crs: {state.origin.strip()}")

    use_manual_scale = not bool(state.auto_scale)
    denom = parse_scale_denom(state.scale) if use_manual_scale else None
    if denom:
        lines.append(f"Plot using scale 1:{denom}")

    if state.plan_number.strip():
        lines.append(f"plan number: {state.plan_number.strip()}")
    if state.surveyor_name.strip():
        lines.append(f"Surveyor name: {state.surveyor_name.strip()}")
    company = state.surveyor_company.strip()
    address = state.surveyor_address.strip()
    if company or address:
        first = company or address
        lines.append(f"Surveyor company and address: {first}")
        if company and address:
            lines.append(address)

    if pillars:
        lines.append("pillar numbers: " + ", ".join(pillars))

    if mode == MODE_COORDINATES:
        pairs: List[str] = []
        for raw in state.coordinates:
            parsed = parse_en_pair(raw)
            if parsed is None:
                continue
            pairs.append(format_en_pair(*parsed))
        if len(pairs) < 3:
            return "", (
                "Enter at least 3 coordinates (Easting, Northing) to form a traverse."
            )
        lines.append("coordinates for the points: " + "; ".join(pairs))
    else:
        start = parse_en_pair(state.start_coordinate)
        if start is None:
            return "", (
                "Enter the starting coordinate in Easting, Northing (E, N) format."
            )
        usable_legs: List[TraverseLeg] = []
        for leg in state.legs or []:
            if not str(leg.degrees or "").strip():
                continue
            if not str(leg.distance or "").strip():
                continue
            usable_legs.append(leg)
        if len(usable_legs) < 3:
            return "", (
                "Enter at least 3 traverse legs with bearing degrees and distances. "
                "Minutes and seconds may be left blank (treated as 0)."
            )
        parts = [f"coordinates for the point: {format_en_pair(*start)}, with"]
        for i, leg in enumerate(usable_legs):
            bearing = _format_bearing_phrase(leg)
            dist = _format_distance_m(leg.distance)
            if i == 0:
                label = "first traverse leg"
            elif i == len(usable_legs) - 1:
                label = "the final traverse leg"
            else:
                label = f"{ordinal_label(i + 1)} traverse leg"
            prefix = "bearing" if i == 0 else "bearing"
            parts.append(f"{prefix}: {bearing}, distance = {dist} ({label})")
        lines.append(" ".join(parts[0:1]) + " " + "; ".join(parts[1:]) + ".")
        adj = (state.traverse_adjustment or ADJUST_BEARING).strip().lower()
        if adj == ADJUST_BOWDITCH:
            lines.append(BOWDITCH_ADJUST_INSTRUCTION)
        else:
            lines.append(BEARING_ADJUST_INSTRUCTION)

    for road in state.roads or []:
        width = (road.width or "").strip()
        start_p = (road.start_pillar or "").strip()
        end_p = (road.end_pillar or "").strip()
        if not width or not start_p or not end_p:
            continue
        width_num = re.sub(r"[^\d.]+", "", width) or width
        lines.append(
            f"Add an access of width {width_num}m on the side of {start_p} and {end_p}."
        )

    for fence in state.fences or []:
        kind = (fence.fence_type or FENCE_NONE).strip()
        if kind == FENCE_NONE or not kind:
            continue
        start_p = (fence.start_pillar or "").strip()
        end_p = (fence.end_pillar or "").strip()
        if not start_p or not end_p:
            continue
        title = (
            "Dwarf Concrete Wall Fence"
            if kind == FENCE_DWARF
            else "Concrete wall fence"
        )
        lines.append(f"Add {title} on the sides joining {start_p} and {end_p}.")

    lines.append(CAD_SESSION_KEEP_LINE)
    return "\n".join(lines), ""


def _format_bearing_phrase(leg: TraverseLeg) -> str:
    deg = (leg.degrees or "").strip()
    minutes = (leg.minutes or "").strip() or "0"
    seconds = (leg.seconds or "").strip() or "0"
    parts = [f"{deg}degrees"]
    if minutes not in {"", "0"}:
        parts.append(f"{minutes} min")
    elif seconds not in {"", "0"}:
        parts.append("0 min")
    if seconds not in {"", "0"}:
        parts.append(f"{seconds} sec")
    return " ".join(parts)


def _format_distance_m(text: str) -> str:
    raw = (text or "").strip()
    m = re.search(_NUM, raw)
    if not m:
        return raw
    value = m.group(0)
    return f"{value}m"


def parse_cad_prompt(text: str) -> CadFormState:
    """Best-effort fill of the Automated CAD form from a conventional CAD prompt."""
    state = CadFormState()
    raw = (text or "").strip()
    if not raw:
        return state

    fields = _extract_labeled_fields(raw)
    save_as = _extract_generate_stem(raw)
    if save_as:
        state.save_as = save_as
    owners = _split_owner_names(fields.get("buyer_name", ""))
    if owners:
        state.owners = owners
    if fields.get("location"):
        state.location = fields["location"]
    if fields.get("lga"):
        state.lga = fields["lga"]
    if fields.get("state"):
        state.state = _trim_trailing_state_word(fields["state"])
    origin = fields.get("origin") or ""
    if origin:
        state.origin = origin
    if fields.get("plan_number"):
        state.plan_number = fields["plan_number"]
    if fields.get("surveyor_name"):
        state.surveyor_name = fields["surveyor_name"]

    company, address = _split_company_address(
        fields.get("surveyor_company_address")
        or fields.get("surveyor_company")
        or "",
        fields.get("surveyor_address") or "",
    )
    state.surveyor_company = company
    state.surveyor_address = address

    pillars = _split_pillars(fields.get("pillars", ""))
    if not pillars:
        pillars = _split_pillars(_slice_after_label(raw, r"pillar\s+numbers?"))
    if pillars:
        state.pillars = pillars

    denom = parse_scale_denom(fields.get("scale", "")) or parse_scale_denom(raw)
    # Only treat as manual scale when the prompt actually asked for one.
    if denom and re.search(
        r"plot\s+using\s+scale|\bscale\s*[:=]\s*1\s*[:/]\s*\d+|\bscale\s+1\s*[:/]\s*\d+",
        raw,
        flags=re.IGNORECASE,
    ):
        state.auto_scale = False
        state.scale = f"1:{denom}"
    else:
        state.auto_scale = True
        state.scale = ""

    coord_pairs = _extract_coordinate_pairs(raw)
    legs = _extract_bearing_legs(raw)
    if len(legs) >= 3:
        state.mode = MODE_BEARINGS
        if coord_pairs:
            state.start_coordinate = format_en_display(*coord_pairs[0])
        state.legs = legs
        state.coordinates = [format_en_display(*p) for p in coord_pairs] or ["", "", ""]
        while len(state.coordinates) < 3:
            state.coordinates.append("")
    elif len(coord_pairs) >= 3:
        state.mode = MODE_COORDINATES
        state.coordinates = [format_en_display(*p) for p in coord_pairs]
        if coord_pairs:
            state.start_coordinate = format_en_display(*coord_pairs[0])
    elif coord_pairs:
        state.mode = MODE_BEARINGS if legs else MODE_COORDINATES
        state.start_coordinate = format_en_display(*coord_pairs[0])
        state.coordinates = [format_en_display(*p) for p in coord_pairs]
        while len(state.coordinates) < 3:
            state.coordinates.append("")
        if legs:
            state.legs = legs
            state.mode = MODE_BEARINGS

    while len(state.pillars) < 3:
        state.pillars.append("")
    while len(state.legs) < 3:
        state.legs.append(TraverseLeg())
    while len(state.coordinates) < 3:
        state.coordinates.append("")

    roads = _extract_access_roads(raw)
    state.roads = roads or [AccessRoad()]
    fences = _extract_wall_fences(raw)
    state.fences = fences or [WallFence()]
    if re.search(
        r"\b(?:bowditch|bowdich|bodwitch|compass\s+(?:rule|method|adjustment))\b",
        raw,
        flags=re.IGNORECASE,
    ) and not re.search(
        r"\bbearing\s+adjustment\b|\bhold\s+distances?\s+constant\b",
        raw,
        flags=re.IGNORECASE,
    ):
        state.traverse_adjustment = ADJUST_BOWDITCH
    else:
        state.traverse_adjustment = ADJUST_BEARING
    return state


def _extract_labeled_fields(text: str) -> dict[str, str]:
    matches: List[Tuple[int, int, str]] = []
    for key, pattern in _METADATA_LABELS:
        for m in re.finditer(
            rf"(?<![A-Za-z0-9_])({pattern})\s*[:=]\s*",
            text,
            flags=re.IGNORECASE,
        ):
            start = m.end()
            matches.append((m.start(1), start, key))
    matches.sort(key=lambda item: item[0])
    # Prefer the more specific company+address label over company/address alone.
    filtered: List[Tuple[int, int, str]] = []
    occupied: List[Tuple[int, int]] = []
    for start, value_at, key in matches:
        if any(start >= a and start < b for a, b in occupied):
            continue
        filtered.append((start, value_at, key))
        occupied.append((start, value_at))
    fields: dict[str, str] = {}
    for i, (_label_at, value_at, key) in enumerate(filtered):
        end = filtered[i + 1][0] if i + 1 < len(filtered) else len(text)
        value = text[value_at:end]
        value = re.split(
            r"\b(?:Add\s+an?\s+access|Add\s+\d*\s*(?:Concrete|Dwarf)|"
            r"with\s+bearing|coordinates\s+for\s+the\s+points?)\b",
            value,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        keep_nl = key in {"surveyor_company_address", "surveyor_company", "surveyor_address"}
        fields[key] = _clean_field_value(value, keep_newlines=keep_nl)
    return fields


def _clean_field_value(value: str, *, keep_newlines: bool = False) -> str:
    text = (value or "").strip().strip(" ,;:")
    text = text.strip("'\"")
    if keep_newlines:
        text = re.sub(r"[^\S\n]+", " ", text)
        text = re.sub(r"\n{2,}", "\n", text)
        return text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _extract_generate_stem(text: str) -> str:
    m = re.search(
        r"(?:generate|create|produce)\s*[-]?\s+"
        r"(?:cad\s+drawing\s+|cad\s+|drawing\s+|file\s+)?"
        r"['\"]([^'\"]+?)\.dwg['\"]",
        text or "",
        flags=re.IGNORECASE,
    )
    if not m:
        m = re.search(
            r"(?:generate|create|produce)\s*[-]?\s+"
            r"(?:cad\s+drawing\s+|cad\s+|drawing\s+|file\s+)?"
            r"([^\s'\",]+?)\.dwg\b",
            text or "",
            flags=re.IGNORECASE,
        )
    if not m:
        return ""
    return _sanitize_dwg_stem(m.group(1) or "")


def _trim_trailing_state_word(value: str) -> str:
    return re.sub(r"\s+state\s*$", "", value or "", flags=re.IGNORECASE).strip()


def _split_owner_names(value: str) -> List[str]:
    raw = (value or "").strip()
    if not raw:
        return []
    chunks = [p.strip() for p in re.split(r"\s*;\s+|\n+", raw) if p.strip()]
    names: List[str] = []
    for chunk in chunks:
        comma_parts = [p.strip() for p in re.split(r",\s*", chunk) if p.strip()]
        for part in comma_parts or [chunk]:
            and_parts = [
                p.strip()
                for p in re.split(r"\s+(?:and|&)\s+", part, flags=re.IGNORECASE)
                if p.strip()
            ]
            if len(and_parts) <= 1:
                names.append(and_parts[0] if and_parts else part)
            else:
                names.extend(and_parts)
    return names or [raw]


def _split_company_address(company_blob: str, address_only: str) -> Tuple[str, str]:
    blob = (company_blob or "").strip()
    extra = (address_only or "").strip()
    if extra and not blob:
        return "", extra
    if not blob:
        return "", extra
    lines = [ln.strip() for ln in re.split(r"[\n;]+", blob) if ln.strip()]
    if len(lines) >= 2:
        return lines[0], " ".join(lines[1:] + ([extra] if extra else []))
    m = re.search(
        r"(.+?\b(?:LTD\.|LIMITED|LLC|INC\.|NIG\.))\s*(.*)$",
        blob,
        flags=re.IGNORECASE,
    )
    if not m:
        m = re.search(r"(.+?\bLTD)\s*(.*)$", blob, flags=re.IGNORECASE)
    if m:
        company = m.group(1).strip(" ,;")
        rest = (m.group(2) or "").strip(" ,;.")
        address = " ".join(p for p in (rest, extra) if p).strip(" ,;.")
        return company, address
    return blob, extra


def _split_pillars(value: str) -> List[str]:
    raw = (value or "").strip().strip(" ,;")
    if not raw:
        return []
    found = [m.group(0).strip() for m in _PILLAR_TOKEN_RE.finditer(raw)]
    if found:
        return found
    parts = [p.strip().strip("'\"") for p in re.split(r"[,;\n]+", raw) if p.strip()]
    return parts


def _slice_after_label(text: str, label: str) -> str:
    m = re.search(rf"{label}\s*[:=]\s*", text, flags=re.IGNORECASE)
    if not m:
        return ""
    return text[m.end() :]


def _extract_coordinate_pairs(text: str) -> List[Tuple[float, float]]:
    pairs: List[Tuple[float, float]] = []
    seen: set[Tuple[float, float]] = set()
    for m in _COORD_PAIR_RE.finditer(text or ""):
        if m.group(1) is not None:
            pair = (float(m.group(1)), float(m.group(2)))
        else:
            pair = (float(m.group(3)), float(m.group(4)))
        key = (round(pair[0], 6), round(pair[1], 6))
        if key in seen:
            continue
        seen.add(key)
        pairs.append(pair)
    return pairs


def _extract_bearing_legs(text: str) -> List[TraverseLeg]:
    legs: List[TraverseLeg] = []
    for m in _BEARING_LEG_RE.finditer(text or ""):
        legs.append(
            TraverseLeg(
                degrees=m.group(1) or "",
                minutes=(m.group(2) or "").strip(),
                seconds=(m.group(3) or "").strip(),
                distance=m.group(4) or "",
            )
        )
    return legs


def _extract_access_roads(text: str) -> List[AccessRoad]:
    roads: List[AccessRoad] = []
    patterns = (
        rf"(?:an?\s+)?access(?:\s+of|\s+road)?\s+(?:width\s+)?({_NUM})\s*m\s+"
        r".*?(?:side\s+of|on\s+side)\s+(.+?)(?=;|\.\s|and\s+another|\nAdd|\n[A-Z]|$)",
        rf"(?:another\s+)?road\s+({_NUM})\s*m\s+wide\s+"
        r".*?(?:side\s+of|on\s+side)\s+(.+?)(?=;|\.\s|and\s+another|\nAdd|$)",
        rf"(?:another\s+)?road\s+(?:of\s+)?(?:width\s+)?({_NUM})\s*m\s+"
        r".*?(?:side\s+of|on\s+side)\s+(.+?)(?=;|\.\s|\nAdd|$)",
    )
    seen: set[Tuple[str, str, str]] = set()
    for pat in patterns:
        for m in re.finditer(pat, text or "", flags=re.IGNORECASE | re.DOTALL):
            width = (m.group(1) or "").strip()
            pillar_blob = m.group(2) or ""
            tokens = [t.strip() for t in _PILLAR_TOKEN_RE.findall(pillar_blob)]
            if len(tokens) < 2:
                bits = re.split(r"\s+(?:and|to)\s+", pillar_blob, flags=re.IGNORECASE)
                tokens = [b.strip(" .,;") for b in bits if b.strip()]
            if len(tokens) < 2:
                continue
            key = (width, tokens[0], tokens[1])
            if key in seen:
                continue
            seen.add(key)
            roads.append(
                AccessRoad(width=width, start_pillar=tokens[0], end_pillar=tokens[1])
            )
    return roads


def _extract_wall_fences(text: str) -> List[WallFence]:
    fences: List[WallFence] = []
    seen: set[Tuple[str, str, str]] = set()
    for m in re.finditer(
        r"(Dwarf(?:\s+Concrete)?\s+Wall\s+Fence|Concrete\s+wall\s+fence|"
        r"D\.C\.W\.F\.?|C\.W\.F\.?|DCWF|CWF).{0,80}?"
        r"(?:joining|connecting|sides?|along)\s+(.+?)(?=\.\s|;\s|\nAdd|$)",
        text or "",
        flags=re.IGNORECASE | re.DOTALL,
    ):
        kind_raw = m.group(1) or ""
        fence_type = (
            FENCE_DWARF
            if re.search(r"dwarf|d\.c\.w\.f|dcwf", kind_raw, flags=re.IGNORECASE)
            else FENCE_CONCRETE
        )
        tokens = [t.strip() for t in _PILLAR_TOKEN_RE.findall(m.group(2) or "")]
        if len(tokens) < 2:
            continue
        for a, b in zip(tokens, tokens[1:]):
            key = (fence_type, a, b)
            if key in seen:
                continue
            seen.add(key)
            fences.append(
                WallFence(fence_type=fence_type, start_pillar=a, end_pillar=b)
            )
    return fences


def format_cad_prompt_for_display(text: str) -> str:
    """Pretty transcript for a conventional cadastral Generate prompt. Raw text if not CAD."""
    raw = (text or "").strip()
    if not raw:
        return text or ""
    low = raw.lower()
    if "generate" not in low or ".dwg" not in low:
        return raw
    if "buyer name" not in low and "pillar" not in low:
        return raw
    if "coordinates" not in low and "bearing" not in low:
        return raw
    try:
        state = parse_cad_prompt(raw)
    except Exception:
        return raw
    owners = [o for o in (state.owners or []) if str(o or "").strip()]
    pillars = [p for p in (state.pillars or []) if str(p or "").strip()]
    usable_legs = [
        lg
        for lg in (state.legs or [])
        if str(lg.degrees or "").strip() and str(lg.distance or "").strip()
    ]
    usable_coords = [c for c in (state.coordinates or []) if str(c or "").strip()]
    if (not owners and not pillars) or (len(usable_legs) < 3 and len(usable_coords) < 3):
        return raw

    lines: List[str] = ["Plot request"]
    if state.save_as:
        lines.append(f"Save as: {state.save_as}.dwg")
    if owners:
        if len(owners) == 1:
            lines.append(f"Owner: {owners[0]}")
        else:
            lines.append("Owners:")
            for i, name in enumerate(owners, start=1):
                lines.append(f"  {i}. {name}")
    if state.location.strip():
        lines.append(f"Location: {state.location.strip()}")
    if state.lga.strip():
        lines.append(f"LGA: {state.lga.strip()}")
    if state.state.strip():
        lines.append(f"State: {state.state.strip()}")
    if state.origin.strip():
        lines.append(f"Origin: {state.origin.strip()}")
    if state.plan_number.strip():
        lines.append(f"Plan number: {state.plan_number.strip()}")
    if state.surveyor_name.strip():
        lines.append(f"Surveyor: {state.surveyor_name.strip()}")
    if state.surveyor_company.strip():
        lines.append(f"Company: {state.surveyor_company.strip()}")
    if state.surveyor_address.strip():
        lines.append(f"Address: {state.surveyor_address.strip()}")
    if pillars:
        lines.append("Pillars: " + ", ".join(pillars))

    if state.mode == MODE_BEARINGS:
        if state.start_coordinate.strip():
            sc = state.start_coordinate.strip()
            parts = [p.strip() for p in sc.split(",")]
            if len(parts) == 2 and "E" not in sc.upper():
                sc = f"{parts[0]} E, {parts[1]} N"
            lines.append(f"Start coordinate: {sc}")
        usable = [
            lg
            for lg in (state.legs or [])
            if str(lg.degrees or "").strip() and str(lg.distance or "").strip()
        ]
        if usable:
            lines.append("Traverse legs:")
            for i, lg in enumerate(usable, start=1):
                mins = (lg.minutes or "0").strip() or "0"
                secs = (lg.seconds or "0").strip() or "0"
                bearing = f"{(lg.degrees or '').strip()}° {mins}′"
                if secs not in {"", "0"}:
                    bearing += f" {secs}″"
                lines.append(f"  {i}. {bearing}  ·  {(lg.distance or '').strip()} m")
        adj = (state.traverse_adjustment or ADJUST_BEARING).strip().lower()
        if adj == ADJUST_BOWDITCH:
            lines.append("Adjustment: Bowditch (compass rule)")
        else:
            lines.append("Adjustment: Bearing (distances held constant)")
    else:
        pairs = [c for c in (state.coordinates or []) if str(c or "").strip()]
        if pairs:
            lines.append("Coordinates:")
            for i, c in enumerate(pairs, start=1):
                lines.append(f"  {i}. {c}")

    for i, road in enumerate(state.roads or [], start=1):
        if not (road.width and road.start_pillar and road.end_pillar):
            continue
        lines.append(
            f"Access {i}: {road.width} m  ·  {road.start_pillar} to {road.end_pillar}"
        )
    for i, fence in enumerate(state.fences or [], start=1):
        kind = (fence.fence_type or FENCE_NONE).strip()
        if kind == FENCE_NONE or not fence.start_pillar or not fence.end_pillar:
            continue
        lines.append(
            f"Fence {i}: {kind}  ·  {fence.start_pillar} to {fence.end_pillar}"
        )
    return "\n".join(lines)
