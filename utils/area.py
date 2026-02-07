"""
Survey area utilities.

- If coordinates are geodetic (lon/lat in degrees), compute geodesic area on WGS84 ellipsoid.
- If coordinates are projected (x/y in meters/feet), compute planar polygon area (shoelace).

The caller should pass points in the correct order around the boundary. We will
close the polygon if needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class AreaResult:
    method: str  # "geodesic_wgs84" | "planar"
    area_m2: float
    perimeter_m: Optional[float] = None

    @property
    def hectares(self) -> float:
        return self.area_m2 / 10_000.0

    @property
    def acres(self) -> float:
        return self.area_m2 * 0.0002471053814671653

    @property
    def ft2(self) -> float:
        return self.area_m2 * 10.763910416709722


def _close(points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not points:
        return []
    if points[0] == points[-1]:
        return list(points)
    return list(points) + [points[0]]


def planar_polygon_area(points_xy: Sequence[Tuple[float, float]]) -> float:
    """Shoelace area for planar x/y coordinates. Units are squared input units."""
    pts = _close(points_xy)
    if len(pts) < 4:
        return 0.0
    s = 0.0
    for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
        s += (x1 * y2) - (x2 * y1)
    return abs(s) * 0.5


def geodesic_area_wgs84(points_lonlat: Sequence[Tuple[float, float]]) -> AreaResult:
    """
    Geodesic polygon area on WGS84.
    points_lonlat: (lon, lat) in degrees.
    """
    try:
        from pyproj import Geod
    except Exception as e:
        raise ImportError("pyproj is required for geodesic area calculation") from e

    pts = _close(points_lonlat)
    if len(pts) < 4:
        return AreaResult(method="geodesic_wgs84", area_m2=0.0, perimeter_m=0.0)

    geod = Geod(ellps="WGS84")
    lons = [p[0] for p in pts]
    lats = [p[1] for p in pts]
    area, perim = geod.polygon_area_perimeter(lons, lats)
    return AreaResult(method="geodesic_wgs84", area_m2=abs(area), perimeter_m=float(perim))


def best_area(
    points: Sequence[Tuple[float, float]],
    *,
    crs_hint: str = "",
    assume_lonlat: Optional[bool] = None,
) -> AreaResult:
    """
    Pick a sane best-effort area method.

    - If CRS hint looks geodetic/WGS84/EPSG:4326 -> geodesic
    - If assume_lonlat explicitly set -> respect it
    - Else: infer by value ranges (lon within 180, lat within 90)
    """
    hint = (crs_hint or "").lower()
    if assume_lonlat is True:
        return geodesic_area_wgs84(points)
    if assume_lonlat is False:
        # Caller asserts planar meters; return in m² (caller should ensure units are meters)
        return AreaResult(method="planar", area_m2=float(planar_polygon_area(points)), perimeter_m=None)

    if any(k in hint for k in ("wgs84", "wgs 84", "epsg:4326", "wkid 4326", "geographic")):
        return geodesic_area_wgs84(points)

    # Range inference
    if points:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        if all(abs(x) <= 180.0 for x in xs) and all(abs(y) <= 90.0 for y in ys):
            return geodesic_area_wgs84(points)

    return AreaResult(method="planar", area_m2=float(planar_polygon_area(points)), perimeter_m=None)


