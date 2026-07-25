# Getting started with SurvyAI

A short playbook for your first time. Open **Help → Getting started guide** anytime.

---

## 1. Five-minute start

1. **Workspace** — SurvyAI stores output in this folder by default and reads input (except, when you explicitly define a full path where an input or output should be stored). Point the Workspace bar at the folder that holds your drawings, PDFs, Excel/CSV files, and templates. SurvyAI reads inputs from here and writes outputs here by default ().
2. **Choose an AI path**
   - **Local (Ollama)** — free, private, works offline after models are installed (Not good for complex tasks).
   - **Hosted (cloud sign-in)** — faster / stronger models; uses your credit balance (Good for complex tasks).
3. **Ask** — Type in the console and press **Send** (or `Enter`). Use **`Shift+Enter`** for a new line.
4. Watch **Activity** / the conversation for progress. Long CAD or GIS jobs can take a few minutes.

Tip: Keep related files in one workspace so you can mention short file names in prompts.

---

## 2. Automatically plot a CAD cadastral plan

**Need:** Atleast, AutoCAD version 2007 or later installed (and preferably running); works with any model of AutoCAD from 2007 and later. A survey plan template `.dwg` helps (SurvyAI ships a default template).

**Do this:**

1. Put your template (or use the bundled default) in the workspace (**This is Optional: SurvyAI comes with a CAD template embedded**).
2. Gather buyer name, location, LGA/state, CRS, plan number, surveyor details, pillar numbers, and bearings/distances (or coordinates).
3. You call also include instructions to plot other features like roads, concrete wall fences (**Note that SurvyAI only allows plotting of Roads and Concrete wall fences/Dwarf Concrete Wall fences (c.w.f or d.c.w.f) for this version**). Roads are plotted only in dashed lines and with the user-specified width.
4. Send a clear request. Example:

```text
Generate Buyer_Name.dwg in this workspace using the survey plan template.
Buyer: Mr. Richyblue James Doe
Location: Livingstone Chokogba Farmland, Chokota Etche
LGA: Etche LGA, Rivers State
CRS: UTM Zone 32N
Plan number: RV/0000/2026/001
Surveyor: Surv. Robotics John Doe (mnis), SURVYAI GEO-NET SERVICES LTD
Pillars: SP/RV 1000 … SP/RV 1003 with bearings and distances:
  59°58' / 30.50m; 154°34' / 15.25m; 239°50' / 30.50m; 334°39' / 15.25m
Add a 6 m access road on SP/RV 1000–1001 and a 10 m road on SP/RV 1002–1003.
```

**Follow-ups** (same conversation, after the plan exists):

- `Add a road on the eastern boundary of the plan we just made.`
- `Change the title block buyer name to ABC Limited.`
- `Move the access road to the opposite side.`

Settings also has a **default CAD prompt** you can customize for repeat jobs.
**To customize your default CAD prompt, click on 'Account' --> 'Edit Default CAD Prompt...', edit the Default CAD prompt to suit you, such as the Surveyor's name, company and address, Plan number, etc., and click on 'Apply Change'.**

---

## 3. Scan a PDF and replot a CAD plan

**Typical flow:** PDF/report → extract facts/coords → plot DWG.

**Examples:**

```text
Open survey_deed.pdf in this workspace. Extract owner name, location, pillar numbers,
and all bearings/distances or coordinates. List them clearly in a table.
```

```text
Using the coordinates you extracted from survey_deed.pdf, plot a cadastral plan
to New_Plan.dwg with our survey plan template. CRS: UTM Zone 32N.
```

```text
Compare the bearings in field_notes.pdf with the plan we just plotted and flag mismatches.
```

Tips:

- Prefer text PDFs; scanned-only PDFs may need clearer pages or OCR-friendly exports.
- If the PDF is huge, ask for one section first (schedule of bearings, then plot).

---

## 4. Automate geospatial work in ArcGIS Pro

**Need:** ArcGIS Pro installed. SurvyAI can launch projects and drive common analysis when available.

**Examples:**

```text
Create an ArcGIS Pro project in this workspace named SiteA_Fill, set CRS to
Minna / Nigeria Mid Belt, and open it.
```

```text
Using the elevation points in levels.xlsx, build an IDW surface and compute
cut/fill volume against the design surface in design.tif. Save results in this workspace.
```

```text
Import boundary.shp into the current ArcGIS project and summarize area in hectares.
```

If ArcGIS is not detected, SurvyAI will say so — CAD and document tools still work.

---

## 5. Other key abilities

| Task | Example ask |
|------|-------------|
| Excel / CSV coordinates | `Read points.xlsx, convert from WGS84 to UTM 32N, save as converted.xlsx.` |
| Area / traverse | `Compute closed traverse area and Bowditch-adjust these bearings…` |
| Documents / reports | `Summarize report.docx and save summary.docx in this folder.` |
| Coordinate tools | Blue Marble if installed; otherwise pyproj fallback. |

Optional integrations (not bundled): **AutoCAD**, **ArcGIS Pro**, **Ollama**.

---

## 6. Conversations and good habits

- **New** starts a fresh conversation; **Delete** removes one.
- Prefer **one clear job per message**. Unrelated topics do not continue the previous CAD/GIS job.
- Use **Safe Mode** in Settings if you need to limit advanced integrations while troubleshooting.
- Export diagnostics from Settings/Help paths when contacting support (sensitive values are redacted) and send to **support@survyai.com** (stating the exact issues you encountered in the body of the mail).

---

## 7. Where to go next

- **Help → Documentation (README)** — product overview, billing, privacy.
- **Help → First-run tutorial** — account / data folder / capability wizard again.
- **Help → Getting started guide** — this playbook.

You’re ready: set a workspace, and send your first prompt.
