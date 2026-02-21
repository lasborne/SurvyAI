
import arcpy, os
arcpy.env.overwriteOutput = True
workspace = r"C:\Users\UZOR\PycharmProjects\untitled\venv\SurvyAI"
gdb_name = "FillVolume_analysis"
gdb_path = os.path.join(workspace, gdb_name + ".gdb")
if not arcpy.Exists(gdb_path):
    arcpy.management.CreateFileGDB(workspace, gdb_name + ".gdb")
arcpy.env.workspace = gdb_path

excel_table = os.path.join(gdb_path, "data_table")
if arcpy.Exists(excel_table):
    arcpy.management.Delete(excel_table)
arcpy.conversion.ExcelToTable(r"C:\Users\UZOR\PycharmProjects\untitled\venv\SurvyAI\WELL 8 POST DATA.xlsx", excel_table, "WELL 8 POST FILL DATA FINAL")

# Resolve actual field names (ArcGIS may alter names: spaces->underscores, case, etc.)
fields = [f.name for f in arcpy.ListFields(excel_table) if f.type not in ("OID", "Geometry")]
def _norm(s): return (s or "").lower().replace(" ", "_").replace("-", "_").strip("_")
def _pick(hints, exclude):
    for h in hints:
        hn = _norm(h)
        for f in fields:
            if f in exclude: continue
            if _norm(f) == hn or hn in _norm(f) or _norm(f) in hn:
                return f
    return None
x_f = _pick(["Eastings", "easting", "east", "x", "longitude"], [])
y_f = _pick(["Northings", "northing", "north", "y", "latitude"], [])
post_f = _pick(["post fill", "post_fill", "postfill", "post fill", "post"], [x_f, y_f])
pre_f = _pick(["pre fill", "pre_fill", "prefill", "pre fill", "pre"], [x_f, y_f, post_f] if post_f else [x_f, y_f])
if not all([x_f, y_f, post_f, pre_f]):
    raise RuntimeError("Resolved fields: " + str({"x": x_f, "y": y_f, "post_z": post_f, "pre_z": pre_f}) + "; available: " + ",".join(fields))

sr = arcpy.SpatialReference(26392)
points_fc = os.path.join(gdb_path, "points")
if arcpy.Exists(points_fc):
    arcpy.management.Delete(points_fc)
arcpy.management.CreateFeatureclass(gdb_path, "points", "POINT", spatial_reference=sr)
arcpy.management.AddField(points_fc, "Post", "DOUBLE")
arcpy.management.AddField(points_fc, "Pre", "DOUBLE")
with arcpy.da.SearchCursor(excel_table, [x_f, y_f, post_f, pre_f]) as sc, \
     arcpy.da.InsertCursor(points_fc, ["SHAPE@XY", "Post", "Pre"]) as ic:
    for x_raw, y_raw, pz, prz in sc:
        try:
            x = float(str(x_raw).replace(",", ""))
            y = float(str(y_raw).replace(",", ""))
            post_z = float(str(pz).replace(",", "")) if pz not in (None, "") else None
            pre_z = float(str(prz).replace(",", "")) if prz not in (None, "") else None
        except Exception:
            continue
        ic.insertRow(((x, y), post_z, pre_z))

post_pts = os.path.join(gdb_path, "post_points")
if arcpy.Exists(post_pts):
    arcpy.management.Delete(post_pts)
arcpy.analysis.Select(points_fc, post_pts, "Post IS NOT NULL")
hull_fc = os.path.join(gdb_path, "post_hull")
if arcpy.Exists(hull_fc):
    arcpy.management.Delete(hull_fc)
arcpy.management.MinimumBoundingGeometry(post_pts, hull_fc, "CONVEX_HULL", group_option="ALL")

cell_size = 1.0
arcpy.env.mask = hull_fc
extent_hull = arcpy.Describe(hull_fc).extent
arcpy.env.extent = extent_hull
post_raster = os.path.join(gdb_path, "post_idw")
pre_raster = os.path.join(gdb_path, "pre_idw")
for r in [post_raster, pre_raster]:
    if arcpy.Exists(r):
        arcpy.management.Delete(r)
arcpy.sa.Idw(post_pts, "Post", cell_size=cell_size).save(post_raster)
arcpy.env.extent = extent_hull
arcpy.env.snapRaster = post_raster
pre_pts = os.path.join(gdb_path, "pre_points")
if arcpy.Exists(pre_pts):
    arcpy.management.Delete(pre_pts)
arcpy.analysis.Select(points_fc, pre_pts, "Pre IS NOT NULL")
arcpy.sa.Idw(pre_pts, "Pre", cell_size=cell_size).save(pre_raster)

cutfill_raster = os.path.join(gdb_path, "cutfill")
if arcpy.Exists(cutfill_raster):
    arcpy.management.Delete(cutfill_raster)
# CutFill(before, after): before=pre-fill, after=post-fill. Negative Volume = fill (material added).
arcpy.sa.CutFill(pre_raster, post_raster).save(cutfill_raster)

# IMPORTANT:
# - The CutFill output raster itself may store REGION IDs in its VALUE field (not per-cell height deltas),
#   so attempting to compute volume from its VALUE/COUNT alone can be wrong (it can collapse to a single zone).
# - To match manual ArcGIS Pro behavior, compute dz = (post_idw - pre_idw) within the post boundary mask,
#   and then create a "CutFill table" with a true Volume column derived from dz and cell area.
fill_volume = 0.0
cut_volume = 0.0

# Force CutFill to be clipped to the actual area of work (post boundary).
cutfill_masked = os.path.join(gdb_path, "cutfill_masked")
if arcpy.Exists(cutfill_masked):
    arcpy.management.Delete(cutfill_masked)
arcpy.sa.ExtractByMask(cutfill_raster, hull_fc).save(cutfill_masked)

# Difference raster (meters): positive=fill (post>pre), negative=cut (post<pre)
dz_raster = os.path.join(gdb_path, "dz_post_minus_pre")
if arcpy.Exists(dz_raster):
    arcpy.management.Delete(dz_raster)
dz = arcpy.sa.Minus(arcpy.sa.Raster(post_raster), arcpy.sa.Raster(pre_raster))
arcpy.sa.ExtractByMask(dz, hull_fc).save(dz_raster)
del dz

# Cell area from output raster (m²)
dz_desc = arcpy.Describe(dz_raster)
try:
    cell_area = float(dz_desc.meanCellWidth) * float(dz_desc.meanCellHeight)
except Exception:
    cell_area = float(cell_size) * float(cell_size)

# Compute TOTAL fill/cut volumes from dz using zonal SUM (robust in headless ProPy).
# dz = (post - pre) in meters; Volume = SUM(dz_pos) * cell_area in m³, and SUM(-dz_neg) * cell_area in m³.
fill_volume = 0.0
cut_volume = 0.0
oid_field = arcpy.Describe(hull_fc).OIDFieldName
fill_depth = arcpy.sa.Con(arcpy.sa.Raster(dz_raster) > 0, arcpy.sa.Raster(dz_raster), 0)
cut_depth = arcpy.sa.Con(arcpy.sa.Raster(dz_raster) < 0, -arcpy.sa.Raster(dz_raster), 0)
fill_tbl = os.path.join(gdb_path, "fill_depth_sum")
cut_tbl = os.path.join(gdb_path, "cut_depth_sum")
for t in [fill_tbl, cut_tbl]:
    if arcpy.Exists(t):
        arcpy.management.Delete(t)
arcpy.sa.ZonalStatisticsAsTable(hull_fc, oid_field, fill_depth, fill_tbl, "DATA", "SUM")
arcpy.sa.ZonalStatisticsAsTable(hull_fc, oid_field, cut_depth, cut_tbl, "DATA", "SUM")
del fill_depth
del cut_depth

def _sum_from(tbl):
    flds = [f.name for f in arcpy.ListFields(tbl)]
    sf = next((f for f in flds if (f or "").strip().lower() == "sum"), None) or next((f for f in flds if "sum" in (f or "").strip().lower()), None)
    if not sf:
        return 0.0
    with arcpy.da.SearchCursor(tbl, [sf]) as c:
        for (v,) in c:
            return float(v) if v is not None else 0.0
    return 0.0

fill_volume = _sum_from(fill_tbl) * cell_area
cut_volume = _sum_from(cut_tbl) * cell_area

# Create CutFill table with Volume column from dz sums per CutFill zone.
# (This mimics the Volume column ArcGIS often provides automatically.)
cutfill_table = os.path.join(gdb_path, "cutfill_table")
if arcpy.Exists(cutfill_table):
    arcpy.management.Delete(cutfill_table)

# Ensure the zone raster has an attribute table (for the Value zone field)
try:
    arcpy.management.BuildRasterAttributeTable(cutfill_masked, "Overwrite")
except Exception:
    pass

zone_field = "Value"
# Zonal statistics: sum of dz values per zone (units: meters). Multiply by cell_area => m³.
arcpy.sa.ZonalStatisticsAsTable(cutfill_masked, zone_field, dz_raster, cutfill_table, "DATA", "SUM")

# Add Volume (m³) and compute it from SUM(dz) * cell_area
tbl_fields = [f.name for f in arcpy.ListFields(cutfill_table)]
sum_f = next((f for f in tbl_fields if (f or "").strip().lower() == "sum"), None)
if not sum_f:
    # Sometimes SUM is named SUM_ or similar depending on store; pick first field containing 'sum'
    sum_f = next((f for f in tbl_fields if "sum" in (f or "").strip().lower()), None)
if not sum_f:
    raise RuntimeError("Zonal stats table missing SUM field. Fields: " + ", ".join(tbl_fields))

if "Volume" not in tbl_fields and "VOLUME" not in [f.upper() for f in tbl_fields]:
    arcpy.management.AddField(cutfill_table, "Volume", "DOUBLE")
# Populate Volume using an UpdateCursor (avoids CalculateField expression quirks in some Pro headless runs)
tbl_fields2 = [f.name for f in arcpy.ListFields(cutfill_table)]
vol_f = next((f for f in tbl_fields2 if (f or "").strip().lower() == "volume"), "Volume")
with arcpy.da.UpdateCursor(cutfill_table, [sum_f, vol_f]) as ucur:
    for s, v in ucur:
        try:
            sv = float(s) if s is not None else 0.0
        except Exception:
            sv = 0.0
        ucur.updateRow((s, sv * cell_area))

# Note: `cutfill_table.Volume` is the NET dz volume per CutFill zone.
# We keep the per-cell `fill_volume` / `cut_volume` computed above as the authoritative totals.

# Area of post_hull polygon (sq m)
hull_area = 0.0
with arcpy.da.SearchCursor(hull_fc, ["SHAPE@AREA"]) as cur:
    for row in cur:
        if row[0] is not None:
            hull_area += float(row[0])
            break

# Create summary table with Area and Volume (not zonal stats, which can confuse area vs volume)
out_excel = r"C:\Users\UZOR\PycharmProjects\untitled\venv\SurvyAI\results_fill.xlsx"
if arcpy.Exists(out_excel):
    arcpy.management.Delete(out_excel)
summary_tbl = os.path.join(gdb_path, "fill_volume_summary")
if arcpy.Exists(summary_tbl):
    arcpy.management.Delete(summary_tbl)
arcpy.management.CreateTable(gdb_path, "fill_volume_summary")
arcpy.management.AddField(summary_tbl, "Metric", "TEXT", field_length=64)
arcpy.management.AddField(summary_tbl, "Value", "DOUBLE")
with arcpy.da.InsertCursor(summary_tbl, ["Metric", "Value"]) as ic:
    ic.insertRow(["Area_sq_m", hull_area])
    ic.insertRow(["Fill_Volume_m3", fill_volume])
    ic.insertRow(["Cut_Volume_m3", cut_volume])
arcpy.conversion.TableToExcel(summary_tbl, out_excel)
print("RESULT_AREA_SQ_M:", hull_area)
print("RESULT_FILL_VOLUME_CUBIC_METERS:", fill_volume)
print("RESULT_CUT_VOLUME_CUBIC_METERS:", cut_volume)
print("RESULT_EXCEL_PATH:", out_excel)
