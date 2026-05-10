"""
================================================================================
SurvyAI — GeoPandas / Shapely Execution Environment
================================================================================

Provides a safe, subprocess-isolated execution layer for arbitrary
GeoPandas / Shapely / pyproj / ezdxf Python code generated on-the-fly by the
LLM agent.

This fills the critical gap left by the ArcGIS-only execution path:
  • No ArcGIS Pro licence required
  • Faster (no Pro startup overhead)
  • Deterministic — the LLM writes standard Python, not ArcPy API calls
  • Handles: CSV/Excel points, DWG/DXF polygons, shapefiles, GeoJSON, GDB
  • Produces Excel/CSV/shapefile outputs with RESULT_ structured logging

Architecture note
-----------------
The executor injects a preamble with production-ready helper functions
(read_dwg_polygons, read_csv_points, points_within_polygon, export_to_excel,
result_log) into every script so that the LLM can call them without
re-implementing them, reducing hallucination surface area.
"""

from __future__ import annotations

import datetime
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Preamble injected at the top of every LLM-generated script
# ---------------------------------------------------------------------------

GEOPANDAS_SCRIPT_PREAMBLE = textwrap.dedent("""
import sys, os, json, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import (
    Point, Polygon, MultiPolygon, LineString, MultiLineString, GeometryCollection
)
from shapely.ops import unary_union
from pathlib import Path

# ── Structured output helpers ────────────────────────────────────────────────
def result_log(key: str, value) -> None:
    \"\"\"Print a RESULT_ key-value pair so the agent can parse structured output.\"\"\"
    print(f"RESULT_{key.upper()}: {value}", flush=True)


# ── DWG / DXF polygon reader ─────────────────────────────────────────────────
def read_dwg_polygons(dwg_path, layer_filter=None, crs=None):
    \"\"\"
    Read closed polygon / polyline features from a DWG or DXF file using ezdxf.

    Parameters
    ----------
    dwg_path : str or Path
        Path to a .dwg or .dxf file.
    layer_filter : list[str] or None
        If given, only entities on these layer names are included.
    crs : str or int or None
        CRS to assign (e.g. 'EPSG:26392'). No re-projection is performed;
        only the CRS attribute is set.

    Returns
    -------
    GeoDataFrame with columns: geometry, layer, entity_type, source.
    \"\"\"
    import ezdxf

    doc = ezdxf.readfile(str(dwg_path))
    msp = doc.modelspace()
    records = []

    for entity in msp:
        lyr = getattr(entity.dxf, "layer", "") or ""
        if layer_filter and lyr not in layer_filter:
            continue

        pts = None
        etype = entity.dxftype()

        if etype == "LWPOLYLINE":
            try:
                pts = [(p[0], p[1]) for p in entity.get_points()]
            except Exception:
                pass
        elif etype in ("POLYLINE", "MESH"):
            try:
                pts = [(v.dxf.location.x, v.dxf.location.y) for v in entity.vertices]
            except Exception:
                pass
        elif etype == "HATCH":
            for boundary_path in entity.paths:
                seg_pts = []
                if hasattr(boundary_path, "vertices"):
                    seg_pts = [(v[0], v[1]) for v in boundary_path.vertices]
                elif hasattr(boundary_path, "edges"):
                    for edge in boundary_path.edges:
                        if hasattr(edge, "start"):
                            seg_pts.append((edge.start.x, edge.start.y))
                if len(seg_pts) >= 3:
                    pts = seg_pts
                    break  # use first boundary path
        elif etype == "INSERT":
            # Block reference — skip (requires block expansion)
            continue

        if pts and len(pts) >= 3:
            # Ensure closure
            if pts[0] != pts[-1]:
                pts.append(pts[0])
            try:
                poly = Polygon(pts)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if poly.is_valid and not poly.is_empty:
                    records.append({
                        "geometry": poly,
                        "layer": lyr,
                        "entity_type": etype,
                        "source": str(dwg_path),
                    })
            except Exception:
                pass

    gdf = gpd.GeoDataFrame(records, geometry="geometry")
    if crs:
        gdf = gdf.set_crs(crs)
    result_log("DWG_POLYGONS_LOADED", len(gdf))
    return gdf


# ── CSV / Excel point reader ─────────────────────────────────────────────────
def read_csv_points(path, e_col=None, n_col=None, crs=None):
    \"\"\"
    Read a CSV or Excel file into a GeoDataFrame of Point geometries.

    Auto-detects Easting/Northing (or X/Y, Longitude/Latitude) columns when
    e_col / n_col are not specified.

    Parameters
    ----------
    path : str or Path
    e_col : str or None    (column name for Easting / X)
    n_col : str or None    (column name for Northing / Y)
    crs   : str or int or None

    Returns
    -------
    GeoDataFrame
    \"\"\"
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    cols_lower = {c.strip().lower(): c for c in df.columns}

    if e_col is None:
        for candidate in ["e", "easting", "east", "x", "longitude", "lon", "long"]:
            if candidate in cols_lower:
                e_col = cols_lower[candidate]
                break
    if n_col is None:
        for candidate in ["n", "northing", "north", "y", "latitude", "lat"]:
            if candidate in cols_lower:
                n_col = cols_lower[candidate]
                break

    if not e_col or not n_col:
        raise ValueError(
            f"Cannot auto-detect E/N columns. "
            f"Available columns: {list(df.columns)}. "
            f"Specify e_col and n_col explicitly."
        )

    df[e_col] = pd.to_numeric(df[e_col], errors="coerce")
    df[n_col] = pd.to_numeric(df[n_col], errors="coerce")
    df = df.dropna(subset=[e_col, n_col]).reset_index(drop=True)

    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(float(e), float(n)) for e, n in zip(df[e_col], df[n_col])],
    )
    if crs:
        gdf = gdf.set_crs(crs)

    result_log("CSV_POINTS_LOADED", len(gdf))
    result_log("E_COL_USED", e_col)
    result_log("N_COL_USED", n_col)
    return gdf


# ── Spatial operations ───────────────────────────────────────────────────────
def points_within_polygon(points_gdf, polygon_gdf, predicate="within"):
    \"\"\"
    Return only those points that satisfy the spatial predicate against polygons.

    Handles CRS mismatch by reprojecting points to the polygon CRS.

    Parameters
    ----------
    points_gdf   : GeoDataFrame of points
    polygon_gdf  : GeoDataFrame of polygons
    predicate    : 'within' | 'intersects' | 'contains' (default 'within')

    Returns
    -------
    GeoDataFrame — subset of points_gdf that pass the predicate.
    \"\"\"
    if (
        points_gdf.crs is not None
        and polygon_gdf.crs is not None
        and points_gdf.crs != polygon_gdf.crs
    ):
        points_gdf = points_gdf.to_crs(polygon_gdf.crs)

    joined = gpd.sjoin(
        points_gdf,
        polygon_gdf[["geometry"]],
        how="inner",
        predicate=predicate,
    )
    joined = joined.drop(columns=["index_right"], errors="ignore")
    result_log("POINTS_WITHIN_COUNT", len(joined))
    result_log("POINTS_OUTSIDE_COUNT", len(points_gdf) - len(joined))
    return joined


def merge_point_attributes(points_gdf, polygon_gdf, how="inner", predicate="within"):
    \"\"\"
    Spatial join that brings polygon attributes onto each matching point row.

    Parameters
    ----------
    points_gdf  : GeoDataFrame of points
    polygon_gdf : GeoDataFrame of polygons (may have extra attribute columns)
    how         : 'inner' (only matched) | 'left' (all points, NaN if no match)
    predicate   : 'within' | 'intersects'

    Returns
    -------
    GeoDataFrame with original point columns + polygon attribute columns.
    \"\"\"
    if (
        points_gdf.crs is not None
        and polygon_gdf.crs is not None
        and points_gdf.crs != polygon_gdf.crs
    ):
        points_gdf = points_gdf.to_crs(polygon_gdf.crs)

    joined = gpd.sjoin(points_gdf, polygon_gdf, how=how, predicate=predicate)
    joined = joined.drop(columns=["index_right"], errors="ignore")
    result_log("MERGED_ROW_COUNT", len(joined))
    return joined


# ── Export helpers ───────────────────────────────────────────────────────────
def export_to_excel(gdf, output_path, sheet_name="Results", drop_geometry=True):
    \"\"\"
    Export a GeoDataFrame to Excel (.xlsx).

    Geometry is dropped by default (professional attribute table).
    \"\"\"
    df = gdf.copy()
    if drop_geometry:
        df = df.drop(columns=["geometry"], errors="ignore")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(str(output_path), index=False, sheet_name=sheet_name)
    result_log("OUTPUT_FILE", str(output_path))
    result_log("ROW_COUNT", len(df))
    result_log("COLUMN_COUNT", len(df.columns))
    result_log("COLUMNS", ", ".join(df.columns.tolist()))
    return str(output_path)


def export_to_csv(gdf, output_path, drop_geometry=True):
    \"\"\"Export a GeoDataFrame to CSV.\"\"\"
    df = gdf.copy()
    if drop_geometry:
        df = df.drop(columns=["geometry"], errors="ignore")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(output_path), index=False)
    result_log("OUTPUT_FILE", str(output_path))
    result_log("ROW_COUNT", len(df))
    return str(output_path)


def export_to_shapefile(gdf, output_path):
    \"\"\"Export a GeoDataFrame to shapefile.\"\"\"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(str(output_path))
    result_log("OUTPUT_FILE", str(output_path))
    result_log("ROW_COUNT", len(gdf))
    return str(output_path)


def read_shapefile_or_geojson(path, crs=None):
    \"\"\"
    Read any vector format supported by GeoPandas / Fiona:
    shapefile (.shp), GeoJSON (.json/.geojson), GPKG, KML, GDB layer, etc.

    For GDB pass: read_shapefile_or_geojson('path/to/file.gdb/LayerName')
    \"\"\"
    gdf = gpd.read_file(str(path))
    if crs and gdf.crs != crs:
        gdf = gdf.to_crs(crs)
    result_log("VECTOR_FEATURES_LOADED", len(gdf))
    return gdf
""")


# ---------------------------------------------------------------------------
# GeoPandasExecutor
# ---------------------------------------------------------------------------

class GeoPandasExecutor:
    """
    Executes LLM-generated GeoPandas / Shapely Python code in an isolated
    subprocess, injects production-ready helper functions via a preamble, and
    parses structured ``RESULT_`` lines from stdout.

    This is the dynamic GIS execution engine for tasks that:
      • Do not need ArcGIS Pro visualisation
      • Require no ArcGIS licence
      • Involve spatial joins, point-in-polygon, attribute export, vector ops
    """

    def __init__(self, timeout_seconds: int = 300):
        self.timeout_seconds = timeout_seconds

    # ------------------------------------------------------------------
    def execute_script(
        self,
        code: str,
        script_name: Optional[str] = None,
        working_dir: Optional[str] = None,
        expected_output_files: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute GeoPandas Python code in a subprocess with the shared preamble.

        Parameters
        ----------
        code                  : Python source (LLM-generated). Must use helpers
                                from the preamble (read_dwg_polygons etc.) and
                                emit ``RESULT_KEY: value`` lines for metrics.
        script_name           : Human-readable name for the saved script file.
        working_dir           : Directory to save the script and use as cwd.
        expected_output_files : Paths the script is expected to create; missing
                                ones are reported as failures.

        Returns
        -------
        Dict with keys: success, script_path, stdout, stderr, parsed_results,
        output_files, missing_outputs, return_code.
        """
        full_code = GEOPANDAS_SCRIPT_PREAMBLE + "\n\n# ── LLM-GENERATED SCRIPT ────\n" + code

        # Resolve working dir
        wdir = Path(working_dir).resolve() if working_dir else Path.cwd().resolve()
        try:
            wdir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            return self._error(f"Cannot create working directory {wdir}: {exc}")

        # Save script for audit trail
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = (
            (script_name or "geopandas_script")
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )[:80]
        script_path = wdir / f"{safe_name}_{stamp}.py"
        try:
            script_path.write_text(full_code, encoding="utf-8")
        except Exception as exc:
            return self._error(f"Cannot write script to {script_path}: {exc}")

        # Execute in subprocess
        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                cwd=str(wdir),
            )
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            success = proc.returncode == 0
            return_code = proc.returncode
        except subprocess.TimeoutExpired:
            return self._error(
                f"Script timed out after {self.timeout_seconds}s.",
                script_path=str(script_path),
            )
        except Exception as exc:
            return self._error(str(exc), script_path=str(script_path))

        # Parse RESULT_ lines
        parsed = {}
        for line in stdout.splitlines():
            line = line.strip()
            if line.startswith("RESULT_"):
                try:
                    raw_key, _, raw_val = line.partition(": ")
                    parsed[raw_key[7:]] = raw_val.strip()
                except Exception:
                    pass

        # Verify expected output files
        found: List[str] = []
        missing: List[str] = []
        for p in (expected_output_files or []):
            (found if Path(p).exists() else missing).append(p)

        # Also pick up OUTPUT_FILE entries from RESULT_ lines
        for val in parsed.values():
            if isinstance(val, str) and Path(val).exists() and val not in found:
                found.append(val)

        return {
            "success": success and not missing,
            "script_path": str(script_path),
            "stdout": stdout,
            "stderr": stderr if not success else "",
            "parsed_results": parsed,
            "output_files": found,
            "missing_outputs": missing,
            "return_code": return_code,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _error(msg: str, script_path: str = "") -> Dict[str, Any]:
        return {
            "success": False,
            "error": msg,
            "script_path": script_path,
            "stdout": "",
            "stderr": "",
            "parsed_results": {},
            "output_files": [],
            "missing_outputs": [],
            "return_code": -1,
        }

    # ------------------------------------------------------------------
    def format_result(self, result: Dict[str, Any]) -> str:
        """Format the execution result as a human-readable string for the LLM."""
        lines = []
        if result.get("success"):
            lines.append("✅ GeoPandas script executed successfully.")
        else:
            err = result.get("error", "")
            lines.append(f"❌ GeoPandas script failed. {err}".strip())

        if result.get("parsed_results"):
            lines.append("\nKey results:")
            for k, v in result["parsed_results"].items():
                lines.append(f"  {k}: {v}")

        if result.get("output_files"):
            lines.append("\nOutput files verified:")
            for f in result["output_files"]:
                lines.append(f"  ✓ {f}")

        if result.get("missing_outputs"):
            lines.append("\nExpected output files NOT found:")
            for f in result["missing_outputs"]:
                lines.append(f"  ✗ {f}")

        stdout = (result.get("stdout") or "").strip()
        if stdout:
            preview = stdout[-2000:] if len(stdout) > 2000 else stdout
            lines.append(f"\nScript output (last 2000 chars):\n{preview}")

        stderr = (result.get("stderr") or "").strip()
        if stderr:
            preview = stderr[-1000:] if len(stderr) > 1000 else stderr
            lines.append(f"\nErrors / warnings:\n{preview}")

        if result.get("script_path"):
            lines.append(f"\nScript saved: {result['script_path']}")

        return "\n".join(lines)
