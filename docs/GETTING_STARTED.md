# Getting started with SurvyAI

A short playbook for your first session. Open **Help → Getting started guide**
anytime. Hover a control in the app for a one-line tip.

---

## 1. Five-minute start

1. **Workspace** — Point the Workspace bar at the folder that holds your
   drawings, PDFs, Excel/CSV files, and templates. SurvyAI reads inputs from
   here and writes outputs here unless you give a full path.
2. **Appearance** — Use the sun/moon control (top right) for light or dark
   mode. The choice is saved on this PC.
3. **Choose an AI path**
   - **Local (Ollama)** — free, private, works offline after models are
     installed. Prefer this for simple questions, not complex CAD or GIS.
   - **Hosted (cloud sign-in)** — stronger models for CAD, documents, and GIS.
     Uses your credit balance.
4. **Pick a tab**
   - **Automated CAD section** — fill the form, press **Send**. No prompt
     writing.
   - **Console** — type a request, attach files if needed, press **Send**
     (**Enter**). **Shift+Enter** starts a new line in the prompt box
     (same as most chat apps). Hover or click the box to see this shortcut.
   - **Output History** — review past runs.
5. Watch **Live activity** and the conversation for progress. Long CAD or GIS
   jobs can take a few minutes.

Tip: Keep related files in one workspace so you can mention short file names.

**Fast mode** and **Use fallback LLM** start unchecked. Leave them off unless
you want quicker general answers (Fast mode) or to force the fallback provider.

---

## 2. Plot a cadastral plan (Automated CAD)

**Need:** AutoCAD 2007 or later (preferably running). A survey-plan template
helps; SurvyAI also ships a default template.

**Do this:**

1. Open the **Automated CAD section** tab.
2. Choose **Coordinates** or **Bearings and distances**.
3. Fill **Save File As** only if you want a specific `.dwg` name. Leave it
   blank to use the first owner name.
4. Enter every real **Owner / buyer**. Use **+** on the last row to add
   another. Names appear together on the title block.
5. Fill site, origin (CRS), and at least three pillar numbers.
6. Enter coordinates, or one start coordinate plus traverse legs.
7. Optional: access roads (width + start/end pillars) and wall fences
   (concrete or dwarf concrete only).
8. Choose one traverse adjustment: **Bearing adjustment** (default) or
   **Bowditch**.
9. Fill certification (plan number, surveyor, company, address) as needed.
10. Press **Send**. If the file already exists, confirm overwrite or pick
    another name.

**Input CAD plan prompt** fills the form from
**Account → Edit Default CAD Prompt** (surveyor defaults, plan number, and
similar). Console still inserts the full prompt text.

---

## 3. Follow-ups in the same conversation

Stay in the conversation that created the plan. Then in **Console** (or a
follow-up after a form plot). In the Console box, **Enter** sends;
**Shift+Enter** starts a new line:

```text
Add a 6 m access road on SP/RV 1000–SP/RV 1001.
```

```text
Change the title-block buyer name to ABC Limited.
```

```text
Subdivide the parcel for the three owners. The second owner has half;
the rest share equally.
```

```text
Save as buyer_name1.dwg
```

**Subdivision notes**

- SurvyAI splits the existing parcel. It does not sketch extra crossing
  internals by hand.
- Equal shares are the default when you do not specify otherwise.
- You can name shares (“first has two-fifths”, “30 square metres more than
  the third”).
- New cut corners get pegs. New pillar numbers appear only if you supply them.

**Save as** copies from the parent drawing into the name you give, then edits
that file. Relative names land next to the parent drawing. If you omit a
filename, SurvyAI edits the current plan after you confirm.

---

## 4. Scan a PDF and replot a CAD plan

**Typical flow:** PDF or report → extract facts and coordinates → plot DWG.

```text
Open survey_deed.pdf in this workspace. Extract owner name, location, pillar
numbers, and all bearings/distances or coordinates. List them in a table.
```

```text
Using the coordinates you extracted from survey_deed.pdf, plot a cadastral
plan to New_Plan.dwg with our survey plan template. CRS: UTM Zone 32N.
```

```text
Compare the bearings in field_notes.pdf with the plan we just plotted and
flag mismatches.
```

Tips:

- Prefer text PDFs; scanned pages work better when they are sharp or
  OCR-friendly.
- For a huge PDF, ask for one section first (schedule of bearings, then plot).
- You can attach the PDF with **+** instead of typing the path.

---

## 5. Automate geospatial work in ArcGIS Pro

**Need:** ArcGIS Pro installed. SurvyAI can launch projects and drive common
analysis when it is detected.

```text
Create an ArcGIS Pro project in this workspace named SiteA_Fill, set CRS to
Minna / Nigeria Mid Belt, and open it.
```

```text
Using the elevation points in levels.xlsx, build an IDW surface and compute
cut/fill volume against the design surface in design.tif. Save results in
this workspace.
```

```text
Import boundary.shp into the current ArcGIS project and summarize area in
hectares.
```

If ArcGIS is not detected, SurvyAI will say so. CAD and document tools still
work.

---

## 6. Other key abilities

| Task | Example |
|------|---------|
| Excel / CSV coordinates | `Read points.xlsx, convert from WGS84 to UTM 32N, save as converted.xlsx.` |
| Area / traverse | `Compute closed traverse area and Bowditch-adjust these bearings…` |
| Documents / reports | `Summarize report.docx and save summary.docx in this folder.` |
| Coordinate tools | Blue Marble if installed; otherwise a local projection fallback. |

Optional integrations (not bundled): **AutoCAD**, **ArcGIS Pro**,
**Geographic Calculator**, **Ollama**.

---

## 7. Credits, Fast Mode, and good habits

- **New** starts a fresh conversation; **Delete** removes the selected one.
- Prefer **one clear job per message**. Unrelated topics do not continue the
  previous CAD or GIS job.
- **Account → Credits & Usage** shows pool, used, remaining, and recent billed
  runs. Local Ollama usage is free.
- Console reminders appear near 50%, 80%, and 95% of the period pool.
- **Safe Mode** in Settings limits advanced integrations while you
  troubleshoot.
- Export a diagnostics bundle from **File** when contacting
  **support@survyai.com**. Sensitive values are redacted.

---

## 8. Where to go next

- **Help → Documentation (README)** — product overview, billing, privacy.
- **Help → First-run tutorial** — account, data folder, and capability wizard.
- **Help → Getting started guide** — this playbook.

Set a workspace, then either fill Automated CAD or send your first Console
prompt.
