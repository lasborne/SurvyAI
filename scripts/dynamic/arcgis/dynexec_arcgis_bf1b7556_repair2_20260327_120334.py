# SurvyAI Dynamic Execution
# Domain: arcgis  Task: bf1b7556  Attempt: 3
# Generated: 2026-03-27T12:03:34.804493

import arcpy, os
from arcpy import env

# Paths
workspace = r"C:\Users\USER\Documents\SPDC\ADIBAWA WELL 13\Adibawa Well 13A BORROWPIT-Contractor\ADIBAWA WELL 13 BORROW PIT VOLUME COMPUTATION_074938"
pre_excel = os.path.join(workspace, "csv_adibawa__020416_0411510.xlsx")
post_excel = os.path.join(workspace, "csv_Adi post_031618_062229.xlsx")
dwg_path = r"C:\Users\USER\Documents\SPDC\ADIBAWA WELL 13\Adibawa Well 13A BORROWPIT-Contractor\REVIEW\POST SURVEY FOR ADIBAWA WELL 13 BORROW PIT 13A RESTORATION.dwg"
output_excel = os.path.join(workspace, "Adibawa_VolumeResult.xlsx")

# Set environment
env.workspace = workspace
arcpy.CheckOutExtension("Spatial")

# Create file geodatabase
gdb = os.path.join(workspace, "adibawa_volume.gdb")
if not arcpy.Exists(gdb):
    arcpy.management.CreateFileGDB(workspace, "adibawa_volume.gdb")

env.workspace = gdb

# Import DWG polygon to geodatabase
cad_dataset = os.path.join(gdb, "adibawa_cad")
if not arcpy.Exists(cad_dataset):
    arcpy.conversion.CADToGeodatabase(dwg_path, gdb, "adibawa_cad", 1000)
polygon_fc = os.path.join(gdb, "adibawa_cad", "Polygon")

# Get extent from polygon and set as environment extent and mask
if not arcpy.Exists(polygon_fc):
    raise Exception("Polygon feature class from DWG not found")

desc = arcpy.Describe(polygon_fc)
env.extent = desc.extent
env.mask = polygon_fc

# Define spatial reference (Minna / Nigeria Mid Belt)
spatial_ref = arcpy.SpatialReference(26392)

# Helper to create points from Excel
def create_points_from_excel(excel_path, sheet_name, out_name):
    temp_table = os.path.join(gdb, out_name + "_tbl")
    if arcpy.Exists(temp_table):
        arcpy.management.Delete(temp_table)
    arcpy.conversion.ExcelToTable(excel_path, temp_table, sheet_name)
    # Create point feature class
    pts_fc = os.path.join(gdb, out_name)
    if arcpy.Exists(pts_fc):
        arcpy.management.Delete(pts_fc)
    arcpy.management.CreateFeatureclass(gdb, out_name, "POINT", spatial_reference=spatial_ref)
    arcpy.management.AddField(pts_fc, "Z", "DOUBLE")
    # Find fields
    fields = [f.name for f in arcpy.ListFields(temp_table)]
    x_field = "E" if "E" in fields else None
    y_field = "N" if "N" in fields else None
    z_field = "Z" if "Z" in fields else None
    if not (x_field and y_field and z_field):
        raise Exception("Required fields E, N, Z not found in Excel table")
    with arcpy.da.SearchCursor(temp_table, [x_field, y_field, z_field]) as sc, \
         arcpy.da.InsertCursor(pts_fc, ["SHAPE@XY", "Z"]) as ic:
        for x, y, z in sc:
            try:
                x_f = float(str(x).replace(",", ""))
                y_f = float(str(y).replace(",", ""))
                z_f = float(str(z).replace(",", ""))
                ic.insertRow(((x_f, y_f), z_f))
            except Exception:
                continue
    return pts_fc

# Create PRE and POST points
pre_pts = create_points_from_excel(pre_excel, "Sheet1", "pre_points")
post_pts = create_points_from_excel(post_excel, "Sheet1", "post_points")

# Set output coordinate system
env.outputCoordinateSystem = spatial_ref

# Determine cell size based on polygon extent
ext = env.extent
cell_size = min((ext.XMax - ext.XMin), (ext.YMax - ext.YMin)) / 200.0

# Create IDW rasters
pre_raster = os.path.join(gdb, "pre_idw")
post_raster = os.path.join(gdb, "post_idw")

if arcpy.Exists(pre_raster):
    arcpy.management.Delete(pre_raster)
if arcpy.Exists(post_raster):
    arcpy.management.Delete(post_raster)

arcpy.env.snapRaster = None

pre_idw = arcpy.sa.Idw(pre_pts, "Z", cell_size=cell_size)
pre_idw.save(pre_raster)

post_idw = arcpy.sa.Idw(post_pts, "Z", cell_size=cell_size)
post_idw.save(post_raster)

# CutFill
cutfill_raster = os.path.join(gdb, "cutfill")
if arcpy.Exists(cutfill_raster):
    arcpy.management.Delete(cutfill_raster)

cf = arcpy.sa.CutFill(pre_raster, post_raster)
cf.save(cutfill_raster)

# Zonal statistics over polygon to get total volume
# Positive = fill, Negative = cut. We'll compute separate sums.

# Convert raster to table with histogram-like stats
cut_table = os.path.join(gdb, "cutfill_stats")
if arcpy.Exists(cut_table):
    arcpy.management.Delete(cut_table)

arcpy.sa.ZonalStatisticsAsTable(polygon_fc, "FID", cutfill_raster, cut_table, "DATA", "SUM")

# But ZonalStatistics SUM will give net volume; instead, compute cut and fill separately from raster

fill_volume = 0.0
cut_volume = 0.0

# Iterate raster cells within mask polygon extent
with arcpy.da.SearchCursor(cutfill_raster, ["VALUE", "COUNT"]) as rc:
    for val, cnt in rc:
        if val is None:
            continue
        vol = float(val) * float(cnt) * (cell_size ** 2)
        if val > 0:
            fill_volume += vol
        elif val < 0:
            cut_volume += vol

net_volume = fill_volume + cut_volume

# Export results to Excel
import xlsxwriter

if os.path.exists(output_excel):
    os.remove(output_excel)

workbook = xlsxwriter.Workbook(output_excel)
worksheet = workbook.add_worksheet("Volume")
worksheet.write(0, 0, "Metric")
worksheet.write(0, 1, "Value (m^3)")
worksheet.write(1, 0, "Fill Volume (positive)")
worksheet.write(1, 1, fill_volume)
worksheet.write(2, 0, "Cut Volume (negative)")
worksheet.write(2, 1, cut_volume)
worksheet.write(3, 0, "Net Volume")
worksheet.write(3, 1, net_volume)
workbook.close()

print("RESULT_FILE:", output_excel)
print("RESULT_FILL_VOLUME:", fill_volume)
print("RESULT_CUT_VOLUME:", cut_volume)
print("RESULT_NET_VOLUME:", net_volume)