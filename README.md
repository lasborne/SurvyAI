# SurvyAI

SurvyAI is an AI assistant for land surveyors and geospatial professionals. It
understands natural-language requests and runs real survey workflows on your
machine — reading and analyzing documents, generating and editing AutoCAD
cadastral plans, running coordinate and area computations, performing GIS
analysis, and producing reports.

---

## Getting started

1. **Choose a workspace.** Use the **Workspace** bar at the top to point SurvyAI
   at the folder that contains your project files (drawings, spreadsheets,
   documents, CSVs). SurvyAI reads inputs from and writes outputs to this folder.
2. **Pick how you run the AI.**
   - **Local models (Ollama)** — free, runs entirely on your PC, no account
     required. Best for privacy and offline use.
   - **Hosted models** — faster, higher-quality responses. Requires signing in
     and a credit balance (see **Billing & credits** below).
3. **Type a request** in the console and press **Send** (or `Enter`).
   Use `Shift+Enter` for a new line.

---

## What you can ask it to do

- **CAD / cadastral plans** — "Plot a cadastral plan from `template.dwg` using
  these coordinates…", then follow up with "add a road on the eastern boundary"
  or "change the title to ABC Limited".
- **Document analysis** — "Summarize `report.docx` and save the summary as a Word
  document", or extract specific sections from large reports.
- **Coordinate & area work** — parse coordinates, transform between systems,
  compute areas and bearings/distances, run Bowditch adjustment.
- **GIS analysis** — IDW surfaces, cut/fill volumes, and other ArcGIS Pro
  workflows when ArcGIS is installed.
- **General questions** — surveying knowledge, standards, and explanations.

SurvyAI answers each request on its own. If you switch to an unrelated topic, it
will not carry over or resume the previous task.

---

## Conversations

- Each conversation tab keeps its own history and session.
- Use **New** to start a fresh conversation and **Delete** to remove one.
- A task started in one conversation always returns its result to that same
  conversation, even if you switch tabs while it runs.

---

## Optional integrations

SurvyAI detects and uses these when they are installed; none are bundled:

| Integration | Enables |
|---|---|
| **AutoCAD** | CAD plan generation and table editing |
| **ArcGIS Pro** | GIS analysis and generated ArcPy execution |
| **Blue Marble Geographic Calculator** | advanced coordinate transformations |
| **Ollama** | free local LLM models (installable from inside the app) |

---

## Billing & credits

- The **free plan** uses local models (Ollama) only — no charges.
- **Hosted models** draw from your purchased credit balance. Usage is metered on
  the server per request.
- You'll see reminders as you approach your limit (at roughly 50%, 80%, and 95%).
- When credits run out, SurvyAI automatically switches to a free local model and
  notifies you so work can continue. Top up to resume hosted models.

---

## Updates

SurvyAI checks for updates and verifies the integrity and digital signature of
any downloaded installer before applying it. You stay in control of when an
update is installed.

---

## Privacy & local data

- Your files stay on your machine. Local tool operations (CAD, parsing,
  computation) run on your PC.
- Account tokens and other secrets are stored encrypted in your Windows user
  profile (DPAPI), not in plaintext.
- Diagnostics exports are redacted of sensitive values before they are saved.

---

## Support

Use **Help → Documentation** in the app for this guide, **Help → Tutorial** for a
guided walkthrough, and **Help → About** for version information.
