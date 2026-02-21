
import arcpy, os
project_path = r"C:\\Users\\UZOR\\PycharmProjects\\untitled\\venv\\SurvyAI\\FillVolume_Results\\FillVolume_Results.aprx"
gdb_path = r"C:\\Users\\UZOR\\PycharmProjects\\untitled\\venv\\SurvyAI\\FillVolume_analysis.gdb"
workspace_dir = r"C:\\Users\\UZOR\\PycharmProjects\\untitled\\venv\\SurvyAI"
aprx = arcpy.mp.ArcGISProject(project_path)
aprx.defaultGeodatabase = gdb_path
try:
    aprx.addFolderConnection(workspace_dir)
except Exception:
    pass
maps = aprx.listMaps()
if maps:
    m = aprx.activeMap if aprx.activeMap else maps[0]
    try:
        if m.spatialReference is None:
            m.spatialReference = arcpy.SpatialReference(26392)
    except Exception:
        pass
    layer_order = ["post_hull", "points", "post_points", "pre_idw", "post_idw", "dz_post_minus_pre", "cutfill_masked"]
    for name in layer_order:
        path = os.path.join(gdb_path, name)
        if arcpy.Exists(path):
            try:
                lyr = m.addDataFromPath(path)
                try:
                    # Ensure visibility matches "manual add" behavior.
                    lyr.visible = True
                except Exception:
                    pass
            except Exception as e:
                print("RESULT_LAYER_WARN:", name, str(e))
    try:
        m.addBasemap("Imagery Hybrid")
    except Exception:
        pass
    zoom_ext = None
    hull_path = os.path.join(gdb_path, "post_hull")
    if arcpy.Exists(hull_path):
        zoom_ext = arcpy.Describe(hull_path).extent
    if zoom_ext is None and arcpy.Exists(os.path.join(gdb_path, "cutfill")):
        zoom_ext = arcpy.Describe(os.path.join(gdb_path, "cutfill")).extent
    if zoom_ext:
        try:
            m.defaultCamera.setExtent(zoom_ext)
        except Exception:
            pass
    aprx.save()
    print("RESULT_PROJECT_READY:", project_path)
aprx = None
