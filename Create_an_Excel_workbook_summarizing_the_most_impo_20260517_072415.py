
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
    """Print a RESULT_ key-value pair so the agent can parse structured output."""
    print(f"RESULT_{key.upper()}: {value}", flush=True)


# ── DWG / DXF polygon reader ─────────────────────────────────────────────────
def read_dwg_polygons(dwg_path, layer_filter=None, crs=None):
    """
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
    """
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
    """
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
    """
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
    """
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
    """
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
    """
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
    """
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
    """
    Export a GeoDataFrame to Excel (.xlsx).

    Geometry is dropped by default (professional attribute table).
    """
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
    """Export a GeoDataFrame to CSV."""
    df = gdf.copy()
    if drop_geometry:
        df = df.drop(columns=["geometry"], errors="ignore")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(output_path), index=False)
    result_log("OUTPUT_FILE", str(output_path))
    result_log("ROW_COUNT", len(df))
    return str(output_path)


def export_to_shapefile(gdf, output_path):
    """Export a GeoDataFrame to shapefile."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(str(output_path))
    result_log("OUTPUT_FILE", str(output_path))
    result_log("ROW_COUNT", len(gdf))
    return str(output_path)


def read_shapefile_or_geojson(path, crs=None):
    """
    Read any vector format supported by GeoPandas / Fiona:
    shapefile (.shp), GeoJSON (.json/.geojson), GPKG, KML, GDB layer, etc.

    For GDB pass: read_shapefile_or_geojson('path/to/file.gdb/LayerName')
    """
    gdf = gpd.read_file(str(path))
    if crs and gdf.crs != crs:
        gdf = gdf.to_crs(crs)
    result_log("VECTOR_FEATURES_LOADED", len(gdf))
    return gdf


# ── LLM-GENERATED SCRIPT ────
import pandas as pd
from pathlib import Path

out1 = Path(r"C:\Users\UZOR\PycharmProjects\untitled\venv\SurvyAI\Plan_details_Extract.xlsx")
out2 = Path(r"C:\Users\UZOR\PycharmProjects\untitled\venv\SurvyAI\Plan_details_Extract2.xlsx")

# Summary for Plan_details_Extract.docx
rows1 = [
    {
        "Drawing File": "NWUNNE_FORTUNE.dwg",
        "Owner/Primary Name": "MR. NWUNE FORTUNE OKECHUKWU",
        "Plan Type": "PLAN SHEWING LANDED PROPERTY",
        "Location": "ELDER OBINALI CLOSE, BESIDE DEESALEM INT'L COLLEGE; OFF STANDARD ROAD, RUMUODUNWERE; ELELENWO; OBIO/AKPOR LOCAL GOVERNMENT AREA; RIVERS STATE, NIGERIA",
        "Plan Number": "RV/2023/016SP",
        "Scale": "1:500",
        "CRS/Origin": "UTM ZONE 32N",
        "Certified Date": "02-03-2023",
        "Stated Area": "815.88 SQ. MTRS",
        "Reference Northing": "534 894.00 m.N",
        "Reference Easting": "285 813.00 m.E",
        "Pillar Labels": "SC/BP P2306; SC/BP P2307; SC/BP P2308; SC/BP P2309; SC/AB P6179; SC/AB P6180; SC/AB P6181; SC/AB P6182",
        "Bearings/Distances": "019°00' / 43.37m; 108°21' / 15.48m; 186°35' / 15.35m; 191°12' / 6.19m; 198°07' / 13.05m; 204°40' / 11.10m; 297°14' / 17.80m; 251°19' / 1.40m",
        "Other Annotations": "ACCESS CLOSE; u/c BUILDING; C.W.F",
        "Surveyor/Company": "MICHAEL C. OKERE; SUNRISE GLOBAL MAPPING LIMITED; NO. 16 IGWURUTA ROAD, RUMUOKWURUSI; sunriseglobalmapping@gmail.com"
    },
    {
        "Drawing File": "JONATHAN_ODIGIE.dwg",
        "Owner/Primary Name": "JONATHAN ODIGIE",
        "Plan Type": "PLAN SHEWING LANDED PROPERTY",
        "Location": "OHIA IZOR MINI NKPUKPA; MGBUCHI COMMUNITY, RUKPOKWU; OBIO/AKPOR LOCAL GOVERNMENT AREA; RIVERS STATE, NIGERIA",
        "Plan Number": "RV/4153/2021/025SP",
        "Scale": "1:500",
        "CRS/Origin": "UTM ZONE 32N",
        "Certified Date": "05-05-2021",
        "Stated Area": "472.82 SQ. MTRS",
        "Reference Northing": "545 108.070m.N",
        "Reference Easting": "275 812.150m.E",
        "Pillar Labels": "SC/RV P7241; SC/RV P7242; SC/RV P7243; SC/RV P7244",
        "Bearings/Distances": "051°50' / 17.70m; 153°02' / 30.00m; 226°55' / 14.00m; 325°43' / 30.70m",
        "Other Annotations": "ACCESS ROAD",
        "Surveyor/Company": "UMEH CASMIR N., B.SC, MNIS; SURVEYOR; 37 EZE BARABARA STREET, OGOLO; RUMUIGBO, PORT HARCOURT; RIVERS STATE"
    },
    {
        "Drawing File": "MR.IKECHUKWU_OLEKA.dwg",
        "Owner/Primary Name": "MR. IKECHUKWU OLEKA",
        "Plan Type": "PLAN SHEWING LANDED PROPERTY",
        "Location": "UMUNWAELILE FARMLAND, UMUOMETA COMMUNITY; EDEGELEM, IGBO-ETCHE; ETCHE LOCAL GOVERNMENT AREA; RIVERS STATE, NIGERIA",
        "Plan Number": "RV/4153/2022/...",
        "Scale": "1:500",
        "CRS/Origin": "UTM ZONE 32N",
        "Certified Date": "12-09-2022",
        "Stated Area": "468.15 SQ. MTRS",
        "Reference Northing": "547 138.151m.N",
        "Reference Easting": "284 619.410m.E",
        "Pillar Labels": "SC/BN P6205; SC/BN P6204; SC/BC P1107; SP P4047",
        "Bearings/Distances": "078°20' / 30.73m; 153°11' / 15.30m; 257°53' / 32.08m; 338°20' / 15.25m",
        "Other Annotations": "ACCESS ROAD; NOTE: ALL PILLAR NUMBERS HERE MIGHT BE UNREGISTERED. PLEASE CHECK!",
        "Surveyor/Company": "UMEH CASMIR N., B.SC, MNIS; SURVEYOR; 37 EZE BARABARA STREET, OGOLO; RUMUIGBO, PORT HARCOURT; RIVERS STATE"
    },
    {
        "Drawing File": "KELECHI_SUSAN_AKAEZE.dwg",
        "Owner/Primary Name": "KELECHI SUSAN AKAEZE",
        "Plan Type": "PLAN SHEWING LANDED PROPERTY",
        "Location": "OHIA OKARAMINI, OMUOGWA; OMUIKE, ALUU; IKWERRE LOCAL GOVERNMENT AREA; RIVERS STATE, NIGERIA",
        "Plan Number": "RV/1124/2023/019",
        "Scale": "1:500",
        "CRS/Origin": "UTM ZONE 32N",
        "Certified Date": "24-02-2023",
        "Stated Area": "410.79 SQ. MTRS",
        "Reference Northing": "548 322.610m.N",
        "Reference Easting": "270 167.220m.E",
        "Pillar Labels": "SC/BY P0752; SC/BY P0753; SC/BY P0754; SC/BY P0755",
        "Bearings/Distances": "073°40' / 19.66m; 165°37' / 20.18m; 264°52' / 26.40m; 008°23' / 16.56m",
        "Other Annotations": "ACCESS ROAD",
        "Surveyor/Company": "SURV. O.R. EDE, B.Tech (ANIS); (REGISTERED SURVEYOR); NO. 7B WOJI ESTATE ROAD, WOJI; PORT HARCOURT, RIVERS STATE"
    }
]

df1 = pd.DataFrame(rows1)
with pd.ExcelWriter(out1, engine='openpyxl') as writer:
    df1.to_excel(writer, index=False, sheet_name='Summary')

# Summary for Plan_details_Extract2.docx
rows2 = [
    {"Contractor/Label": "VIC-OLLY GLOBAL MERCHANDISE NIG", "Work Type": "GRASS CUTTING", "Description": "", "Stated Area (SQM)": "1332.628", "Drawing Name": "EGBEMA MAIN GRASS CUTTING AND SURVEILLANCE.dwg"},
    {"Contractor/Label": "BLUE SPORT ENTERPRISE", "Work Type": "GRASS CUTTING", "Description": "", "Stated Area (SQM)": "8255.388", "Drawing Name": "EGBEMA MAIN GRASS CUTTING AND SURVEILLANCE.dwg"},
    {"Contractor/Label": "LN CHUCKS & CO", "Work Type": "GRASS CUTTING & SURVEILLANCE", "Description": "FOR WELL 1 & FLOWLINE", "Stated Area (SQM)": "16607.994", "Drawing Name": "EGBEMA MAIN GRASS CUTTING AND SURVEILLANCE.dwg"},
    {"Contractor/Label": "CHIBUIKE UGWUEZU NIGERIA ENTERPRISES", "Work Type": "GRASS CUTTING & SURVEILLANCE", "Description": "FOR WELL 2 & ACCESS ROAD", "Stated Area (SQM)": "19172.754", "Drawing Name": "EGBEMA MAIN GRASS CUTTING AND SURVEILLANCE.dwg"},
    {"Contractor/Label": "NWACHOGUFEB NIGERIA ENTERPRISE", "Work Type": "GRASS CUTTING & SURVEILLANCE", "Description": "FOR WELL 3 & FLOWLINE", "Stated Area (SQM)": "21347.031", "Drawing Name": "EGBEMA MAIN GRASS CUTTING AND SURVEILLANCE.dwg"},
    {"Contractor/Label": "EGBEMA LANDOWNERS", "Work Type": "GRASS CUTTING & SURVEILLANCE", "Description": "FOR WELL 7 & FLOWLINE", "Stated Area (SQM)": "25063.382", "Drawing Name": "EGBEMA MAIN GRASS CUTTING AND SURVEILLANCE.dwg"},
    {"Contractor/Label": "FABUCO ENTERPRISES NIGERIA LIMITED", "Work Type": "GRASS CUTTING", "Description": "FOR ACCESS ROAD FROM POLICE STATION TO ABAEZI T.T.C JUNCTION", "Stated Area (SQM)": "64065.590", "Drawing Name": "EGBEMA MAIN GRASS CUTTING AND SURVEILLANCE.dwg"},
    {"Contractor/Label": "E.O. NWOKECHA AND SONS", "Work Type": "GRASS CUTTING & SURVEILLANCE", "Description": "", "Stated Area (SQM)": "96179.289", "Drawing Name": "EGBEMA MAIN GRASS CUTTING AND SURVEILLANCE.dwg"},
    {"Contractor/Label": "ISYBLESS NIG ENTERPRISE", "Work Type": "GRASS CUTTING & SURVEILLANCE", "Description": "FOR WELL 2 FLOWLINE & ACCESS ROAD", "Stated Area (SQM)": "15120.487", "Drawing Name": "EGBEMA MAIN GRASS CUTTING AND SURVEILLANCE.dwg"},
    {"Contractor/Label": "HASSAN GLOBAL RESOURCES", "Work Type": "GRASS CUTTING & SURVEILLANCE", "Description": "", "Stated Area (SQM)": "134281.713", "Drawing Name": "EGBEMA MAIN GRASS CUTTING AND SURVEILLANCE.dwg"}
]
coords2 = [
    {"Point": "3", "Easting (m)": "478344.168", "Northing (m)": "171776.533", "Height (m)": "16.97"},
    {"Point": "4", "Easting (m)": "478429.866", "Northing (m)": "171792.574", "Height (m)": "18.73"}
]
meta2 = [
    {"Field": "Drawing Name", "Value": "EGBEMA MAIN GRASS CUTTING AND SURVEILLANCE.dwg"},
    {"Field": "Source File", "Value": r"C:\Users\USER\Documents\SPDC\GRASSCUTTING_SURVEILLANCE_CONTRACTS\EGBEMA MAIN GRASS CUTTING AND SURVEILLANCE.dwg"},
    {"Field": "Units", "Value": "Unitless"},
    {"Field": "Layers", "Value": "0, POINTS, BORDER"},
    {"Field": "Entity Count", "Value": "389"},
    {"Field": "Verified Closed Entity Handle", "Value": "1109"},
    {"Field": "Computed Entity Area", "Value": "56011.368045177675 square units"}
]

df2 = pd.DataFrame(rows2)
df2_coords = pd.DataFrame(coords2)
df2_meta = pd.DataFrame(meta2)
with pd.ExcelWriter(out2, engine='openpyxl') as writer:
    df2.to_excel(writer, index=False, sheet_name='Contracts_Summary')
    df2_coords.to_excel(writer, index=False, sheet_name='Visible_Coordinates')
    df2_meta.to_excel(writer, index=False, sheet_name='Drawing_Info')

result_log('OUTPUT_FILE_1', str(out1))
result_log('OUTPUT_FILE_2', str(out2))
result_log('ROWS_PLAN_DETAILS_EXTRACT', len(df1))
result_log('ROWS_PLAN_DETAILS_EXTRACT2_CONTRACTS', len(df2))
result_log('ROWS_PLAN_DETAILS_EXTRACT2_COORDS', len(df2_coords))