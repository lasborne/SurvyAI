# SurvyAI productization — Phase 1 and Phase 2

## Phase 1 — Essence (what it is)

Phase 1 **does not change how tools work internally**. It adds a **thin product layer** so a future Windows GUI and installer do not talk to `SurvyAIAgent` raw dicts and global config only.

| Idea | What we added |
|------|----------------|
| **Stable API** | `SurvyAIAgentService` with `run_task()` → typed `AgentRunResult` |
| **Injected config** | `SurvyAIAgent(settings=...)` and `merge_settings()` for cloud tokens later |
| **Machine probe** | `scan_machine_capabilities()` without starting AutoCAD |
| **Version** | `survyai.version.__version__` |
| **CLI** | `survyai/cli.py` + root `cli.py` entry |

Phase 1 = **structure and boundaries**. The agent graph and `tools/` are unchanged.

---

## Phase 2 — Essence (what it adds)

Phase 2 **enforces license / feature flags on the LangGraph tool list**.

- **Who is allowed which tools** is decided at **agent startup** by removing disallowed tools from the list bound to the LLM.
- **No payment tiers** in this repo: only two **license modes**:
  - **`builder`** (default for development) — you keep **full** integration tools for free while building and testing. Per-feature `SURVYAI_FEATURE_*=0` is **ignored** so you never accidentally disable CAD/GIS on your dev machine.
  - **`pro`** — what you ship to **paying customers** (single product: “Pro”). Optional `SURVYAI_FEATURE_*=0` can remove tool families (support, abuse, or future policy).

**Tool mapping** (see `categorize_tool_for_license()` in `survyai/feature_flags.py`):

| Category | Tool name pattern / names |
|----------|---------------------------|
| autocad | `autocad_*` |
| internet | `internet_search` |
| arcgis | `arcgis_*` |
| blue_marble | `geographic_calculator_*` |
| vector_store | `semantic_search`, `store_document`, `vector_store_stats` |
| *(core, always on)* | Excel, documents, coordinates, `filesystem_stat`, etc. |

**Not** changed in Phase 2: internal RAG / vector-store retrieval inside `process_query` when the vector DB is enabled in settings — only the **tool** surface is gated. Tightening RAG by license can be a later step.

---

## Environment variables

| Variable | Meaning |
|----------|---------|
| `SURVYAI_LICENSE_MODE` | `builder` (default) or `pro` |
| `SURVYAI_FEATURE_AUTOCAD` | `1`/`0` — only when `pro` |
| `SURVYAI_FEATURE_ARCGIS` | same |
| `SURVYAI_FEATURE_BLUE_MARBLE` | same |
| `SURVYAI_FEATURE_INTERNET` | same |
| `SURVYAI_FEATURE_VECTOR_STORE` | same |

## Modules (reference)

| Module | Role |
|--------|------|
| `survyai/agent_service.py` | Session + `run_task()`; passes `FeatureFlags` into `SurvyAIAgent` |
| `survyai/types.py` | `AgentRunResult` |
| `survyai/app_config.py` | `merge_settings()` |
| `survyai/capabilities.py` | Machine scan |
| `survyai/feature_flags.py` | License + `is_tool_allowed()`, `categorize_tool_for_license()` |
| `agent/agent.py` | `_filter_tools_by_feature_flags()` after `_create_tools()` |
