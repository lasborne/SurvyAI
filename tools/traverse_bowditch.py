"""
Bowditch (compass-rule) traverse adjustment.

Distributes linear misclosure in proportion to each observed leg length.
Both bearings and distances of the closed ring are then recomputed from the
adjusted coordinates. This is the classical compass rule used in cadastral
traverse reduction — not a bearing-only least-squares hold of distances.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple


def _azimuth_deg(de: float, dn: float) -> float:
    """Whole-circle bearing from north, clockwise (surveyor convention)."""
    return (math.degrees(math.atan2(de, dn)) + 360.0) % 360.0


def _legs_from_closed_ring(pts: Sequence[Dict[str, float]]) -> List[Dict[str, float]]:
    n = len(pts)
    out: List[Dict[str, float]] = []
    for i in range(n):
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        de = float(p2["e"]) - float(p1["e"])
        dn = float(p2["n"]) - float(p1["n"])
        out.append(
            {
                "bearing_deg": _azimuth_deg(de, dn),
                "distance": float(math.hypot(de, dn)),
            }
        )
    return out


def apply_bowditch_rule(
    start_e: float,
    start_n: float,
    bearings_deg: Sequence[float],
    distances: Sequence[float],
) -> Dict[str, Any]:
    """
    Apply the Bowditch / compass rule to a start coordinate plus observed legs.

    The start station is held. Each observed departure/latitude is corrected by
        -ΔE * (Li / P)  and  -ΔN * (Li / P)
    where P is the perimeter. The implied closing vertex is forced onto the
    start; unique parcel vertices omit that duplicate.

    Returns a dict with unadjusted/adjusted points, recomputed legs, and
    misclosure metrics. Does not decide whether the correction should be
    applied — the caller compares ``misclosure_m`` to its own threshold.
    """
    if len(bearings_deg) != len(distances):
        raise ValueError("bearings_deg and distances must be the same length")
    if len(distances) < 1:
        raise ValueError("at least one traverse leg is required")

    e0 = float(start_e)
    n0 = float(start_n)
    b_list = [float(b) for b in bearings_deg]
    d_list = [float(d) for d in distances]

    deltas: List[Tuple[float, float, float]] = []
    unadj: List[Dict[str, float]] = [{"e": e0, "n": n0}]
    ce, cn = e0, n0
    total_len = 0.0
    for bdeg, dist in zip(b_list, d_list):
        br = math.radians(bdeg)
        de = dist * math.sin(br)
        dn = dist * math.cos(br)
        deltas.append((de, dn, dist))
        total_len += dist
        ce += de
        cn += dn
        unadj.append({"e": float(ce), "n": float(cn)})

    mis_e = float(unadj[-1]["e"] - unadj[0]["e"])
    mis_n = float(unadj[-1]["n"] - unadj[0]["n"])
    mis = float(math.hypot(mis_e, mis_n))

    adj: List[Dict[str, float]] = [{"e": e0, "n": n0}]
    ce2, cn2 = e0, n0
    if total_len > 1e-12:
        for de, dn, li in deltas:
            cde = (-mis_e) * (li / total_len)
            cdn = (-mis_n) * (li / total_len)
            ce2 += de + cde
            cn2 += dn + cdn
            adj.append({"e": float(ce2), "n": float(cn2)})
    else:
        adj = [dict(p) for p in unadj]

    max_shift = 0.0
    for k in range(1, min(len(unadj), len(adj))):
        sh = math.hypot(adj[k]["e"] - unadj[k]["e"], adj[k]["n"] - unadj[k]["n"])
        if sh > max_shift:
            max_shift = float(sh)

    vertices = adj[:-1] if len(adj) >= 2 else list(adj)
    adjusted_legs = _legs_from_closed_ring(vertices) if len(vertices) >= 2 else []
    observed_legs = [
        {"bearing_deg": float(b), "distance": float(d)} for b, d in zip(b_list, d_list)
    ]

    delta_bearing: List[float] = []
    delta_distance: List[float] = []
    n_cmp = min(len(observed_legs), len(adjusted_legs))
    for i in range(n_cmp):
        db = (adjusted_legs[i]["bearing_deg"] - observed_legs[i]["bearing_deg"] + 180.0) % 360.0 - 180.0
        dd = adjusted_legs[i]["distance"] - observed_legs[i]["distance"]
        delta_bearing.append(float(db))
        delta_distance.append(float(dd))

    return {
        "unadjusted_points": unadj,
        "adjusted_points_with_closure": adj,
        "adjusted_vertices": vertices,
        "observed_legs": observed_legs,
        "adjusted_legs": adjusted_legs,
        "misclosure_e_m": mis_e,
        "misclosure_n_m": mis_n,
        "misclosure_m": mis,
        "perimeter_m": float(total_len),
        "max_point_shift_m": float(max_shift),
        "delta_bearing_deg": delta_bearing,
        "delta_distance_m": delta_distance,
    }
