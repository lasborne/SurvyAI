# SurvyAI

SurvyAI is an AI assistant for land surveyors and geospatial professionals. It
understands natural-language requests and runs real survey workflows on your
Windows PC — reading documents, generating and editing AutoCAD cadastral plans,
running coordinate and area computations, performing GIS analysis, and producing
reports.

---

## Getting started

1. **Choose a workspace.** Use the **Workspace** bar at the top to point SurvyAI
   at the folder that holds your project files (drawings, spreadsheets,
   documents, CSVs). SurvyAI reads inputs from and writes outputs to this folder
   unless you give a full path.
2. **Pick how you run the AI.**
   - **Local models (Ollama)** — free, runs entirely on your PC. Best for
     privacy and offline use. Not the best choice for complex CAD or GIS jobs.
   - **Hosted models** — faster, higher-quality responses. Requires signing in
     and a credit balance (see **Billing & credits**).
3. **Choose a workspace tab.**
   - **Console** — type a free-form request and press **Send** (or **Enter**).
     **Shift+Enter** starts a new line in the prompt box (same as most chat
     apps). Attach files with **+** or drag-and-drop.
   - **Automated CAD section** — fill the survey fields and press **Send**. You
     do not write a prompt. Best for a first cadastral plot.
   - **Output History** — inspect past runs and reuse a query.
4. **Appearance.** Use the sun/moon control in the title bar to switch light and
   dark mode. The choice is saved on this PC.

Use **Help → Getting started guide** in the app for a short playbook with
examples (also shown once after first install).

---

## Workspace tabs

| Tab | Use it for |
|---|---|
| **Console** | Free-form prompts, PDF-to-CAD, follow-up edits, GIS, documents, and questions. **Enter** sends; **Shift+Enter** starts a new line. |
| **Automated CAD section** | Structured cadastral plotting from coordinates or bearings and distances |
| **Output History** | Past agent runs, costs, and “use this query again” |

**Conversations** (left) and **Live activity** (right) stay visible on Console and
Automated CAD so you can switch tabs without losing the thread.

---

## Automated CAD section

Fill the form, then press **Send**. SurvyAI plots from those values.

- **Coordinates** or **Bearings and distances** — pick one mode. Bearings need
  one start coordinate (E, N) plus traverse legs. Blank minutes or seconds are
  treated as 0.
- **Save File As** — optional `.dwg` name. Leave blank to use the first owner
  name. If a file already exists, SurvyAI asks before overwriting.
- **Owners** — add every real owner or buyer. Several names appear together on
  the title block. Do not invent names.
- **Scale** — Auto-scale is on by default. Turn it off only to type a standard
  survey scale (for example `1:500`).
- **Roads** — dashed access roads between named pillars, at the width you give.
- **Wall fences** — concrete or dwarf concrete wall only, between named pillars.
- **Traverse adjustment** — choose **Bearing adjustment** (default; distances
  held) or **Bowditch / compass rule**. Only one method is used.

After the plan exists, stay in the **same conversation** and use Console (or a
follow-up prompt) for edits: move a road, change the title, subdivide, or
**save as** a new drawing.

---

## What you can ask in Console

- **CAD / cadastral plans** — plot from a template, then follow up with
  “add a 6 m road on SP/RV 1000–1001”, “change the title to ABC Limited”,
  “subdivide for the three owners (second owner has half)”, or
  “save as `buyer_name1.dwg`”.
- **Subdivisions** — SurvyAI splits the existing parcel. It does not freehand
  extra crossing lines. Owner shares follow what you write (equal by default).
- **Document analysis** — summarize or extract from PDF, Word, or Excel.
- **Coordinate & area work** — transforms, areas, bearings/distances, traverse
  adjustment.
- **GIS analysis** — IDW, cut/fill, and other ArcGIS Pro workflows when ArcGIS
  is installed.
- **General questions** — surveying knowledge and explanations.

Each conversation keeps its own history. A run always returns to the
conversation that started it, even if you switch tabs while it is working.

---

## Fast Mode and Fallback LLM

Both start **off** on a new install. Turn them on only when you need them.

| Control | When to use it |
|---|---|
| **Fast mode (non-file prompts)** | Quicker answers to general questions by skipping tool planning. CAD, documents, ArcGIS, and other file jobs are unchanged. |
| **Use fallback LLM** | Force the fallback provider even when the primary is healthy. Leave off unless you are testing or the primary is failing. Disabled while Primary is Ollama. |

Primary LLM defaults to **Auto** on each launch (best paid hosted model for the
task). You can lock a provider in Settings for the session.

---

## Optional integrations

SurvyAI detects and uses these when they are installed; none are bundled:

| Integration | Enables |
|---|---|
| **AutoCAD** (2007 or later) | Cadastral plan generation and drawing edits |
| **ArcGIS Pro** | GIS analysis and generated ArcPy execution |
| **Blue Marble Geographic Calculator** | Advanced coordinate transformations |
| **Ollama** | Free local LLM models (installable from inside the app) |

---

## Billing & credits

- The **free plan** uses local models (Ollama) only — no hosted charges.
- **Hosted models** draw from your purchased credit balance. Usage is metered
  on the server per request.
- Open **Account → Credits & Usage** to see the pool, used amount, remaining
  balance, and a short activity log. The page opens immediately; cloud numbers
  refresh in the background.
- Reminders appear under the console prompt at about 50%, 80%, and 95%.
- When credits run out, SurvyAI can continue on a free local model. Top up to
  resume hosted models.

### Account password

- Cloud passwords must be at least **10 characters**, with upper and lower case,
  a digit, and a special character.
- Use **Forgot password…** on the sign-in screen for a one-time email code.
- When signed in, open **Settings → Change password…** (other devices are
  signed out).

---

## Updates

SurvyAI can check for updates and verifies the integrity and digital signature
of any downloaded installer before applying it. You stay in control of when an
update is installed. Enable automatic checks in Settings, or use
**Help → Check for updates…**.

---

## Privacy & local data

- Your project files stay on your machine. CAD, parsing, and computation run
  locally.
- Account tokens and secrets are stored encrypted in your Windows user profile
  (DPAPI), not in plaintext.
- If a destination file already exists, SurvyAI asks before overwriting.
- Diagnostics exports are redacted of sensitive values before they are saved.

---

## Support

| In the app | What you get |
|---|---|
| **Help → Getting started guide** | Short playbook: Automated CAD, Console follow-ups, PDF-to-CAD, ArcGIS |
| **Help → Documentation** | This overview (billing, privacy, controls) |
| **Help → First-run tutorial** | Setup wizard again |
| **Help → About** | Version and notices |
| **File → Export diagnostics bundle…** | Redacted ZIP for **support@survyai.com** |

Hover any toolbar control, tab, or menu item for a one-line explanation.
