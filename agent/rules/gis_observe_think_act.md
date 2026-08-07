# SurvyAI GIS Observe → Think → Act

This rule is the durable semantic protocol for multi-step GIS work
(CRS conversion, polygons, buffers, overlaps, Pro symbology).
Runtime behaviour is driven by `agent/prompts.py` (`SYSTEM_PROMPT`); keep this file
aligned when editing that prompt.

SurvyAI is the **context engineer**: recover practical GIS intent from messy or
differently styled prompts. Do not require the user to be a prompt engineer.
Do **not** hard-code a single training example’s filenames, owners, buffer
distances, or colours into agent logic.

## Observe

From the **current** user message only, list:

- Source files (Excel / CSV / DWG / shapefile / GDB)
- Spreadsheet layout: headed XY table vs family/owner-block (owner title rows + E/N)
- Source CRS and target CRS (informal names OK; resolve to EPSG when converting)
- Whether they asked to **save a converted workbook** (any basename)
- Whether they want **ArcGIS Pro visualization / symbology** (fill colours, layers)
- Whether they want **buffers**, **overlaps**, **areas**, or only conversion
- Output names / folders

Critical observation: `Unnamed:` columns or an owner/family name as the first
“header” means the sheet is **not** a headed XY table — SurvyAI already has a
family-block parser; do not invent a new ArcPy parser.

## Think

Map each observed need to existing SurvyAI tools first:

1. `excel_inspect_workbook` — discover sheets/columns + `ownership.hint`
2. `excel_normalize_ownership_workbook` when layout is family/owner blocks
3. `excel_coordinate_converter` (pyproj) for CRS conversion into the user-named file
4. Visualization / Pro symbology → `arcgis_execute_python_code` on the **converted headed** table
5. Vector metrics without Pro colours → `geopandas_execute` is fine

Never route polygon/buffer/overlap requests to fill-volume / IDW / CutFill tools
unless the user also asked for PRE/POST elevation volumes.

## Act

1. Normalize ownership Excel when needed (Easting / Northing / Pillar / Owner)
2. Convert with SurvyAI (`excel_coordinate_converter`); keep Owner through conversion
3. Build tight-fitting polygons **in the target projected CRS** (group by Owner)
4. Apply symbology the user asked for (e.g. blue fills) via ArcPy when Pro viz is required
5. Buffers / overlaps / areas as requested; colour overlap zones when asked
6. Print `RESULT_*` lines and verify files exist before claiming success
7. On failure: read the tool error, normalize or fix columns, retry — do not invent areas

## Session follow-ups (retain context)

After verified `.aprx` / `.gdb` / owner polygons exist in **this** chat:

- Later asks about "the open ArcGIS result", "these parcels", "each of them" reuse those paths
- Never ask the user to re-send parcel layers / `.aprx` paths you already created
- "Would landmark/structure X fit?" → ask once if they have a footprint, else request
  internet permission for published base dimensions, then geometric fit vs each parcel
- "Deep learning analysis" without a supplied model **and without** Living Atlas /
  pretrained land-cover intent → geometric fit / containment analysis
  (do **not** refuse for lack of a DL model)
- Living Atlas / already-trained land-cover / object-detection on session parcels:
  OBSERVE session `.aprx`/`.gdb`/parcels → THINK catalog + classify + clip (not
  SHAPE@AREA proxy) → ACT with `internet_search` (after one grant) +
  `arcgis_execute_python_code`. On "yes" / "yes go ahead", never re-ask permission.
  If Living Atlas/imagery/model is unavailable after a real attempt, honest fallback
  only — never invent class areas, never permission-loop
- Prefer `geopandas_execute` on converted Excel Owner+X/Y or parcel `.shp`; use
  `arcgis_execute_python_code` for Pro work. SurvyAI's own Python often lacks arcpy —
  that is not a blocker. Do not gate on `arcgis_get_project_info`.
- When ArcGIS Pro is already open or the user pastes `.aprx` / `.shp` / `.gdb` paths,
  analyze those files immediately for fit/containment — never re-ask for paths already
  listed, and never re-run the full convert/plot/buffer job unless they asked to.
- "Option N" + internet "yes" → search the reference object, then continue the same fit
  workflow — do not restart as a blank "send two layers" comparison
- Affirming "practical GIS-based fit comparison" means run the parcel/landmark geometric
  fit — NEVER answer with a GeoPandas vs ArcGIS vs AutoCAD tools essay

## Task repeats (re-execute)

- If the user pastes the **same full** convert → plot → buffer → overlap (or similar)
  workflow again, **re-run the tools**. Do not answer "Already completed".
- Users may have deleted outputs, moved folders, or want a fresh run.
- Only summarize prior outputs when they explicitly ask to recall/summarize without
  restating the operational workflow.
- Never divert Excel→CRS→ArcGIS polygon/buffer/overlap jobs into `Report.docx`
  essay fast-paths ("save as … excel" ≠ Word report)

## Non-goals

- Do not invent coordinates, areas, or overlap results
- Do not re-parse family-block Excel inside ad-hoc ArcPy when normalize exists
- Do not treat “blue polygons / red overlaps” as a volume/CutFill task
- Do not overfit one job’s owner names, buffer distance, or output basename into code
- Do not drop session GIS artifacts when the user affirms an option or grants internet access
- Do not refuse landmark-fit follow-ups because local `import arcpy` failed
