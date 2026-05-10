# Phase 2 — Windows desktop GUI (PySide6)

## What was built

Phase 2 is now a **real desktop product shell**, not only a chat window.

| Piece | Purpose |
|-------|---------|
| [`survyai/gui/main.py`](../survyai/gui/main.py) | `QApplication` bootstrap, Windows app identity, argv prefill/`--run` handling |
| [`survyai/gui/main_window.py`](../survyai/gui/main_window.py) | Full desktop shell: console, history, settings, diagnostics, account/license cards |
| [`survyai/gui/worker.py`](../survyai/gui/worker.py) | Background execution, workspace-aware runs, soft cancel, progress messages |
| [`survyai/gui/state.py`](../survyai/gui/state.py) | Persistent desktop state: onboarding, profile, workspace, safe mode, output history |
| [`survyai/gui/onboarding.py`](../survyai/gui/onboarding.py) | Onboarding wizard, account dialog, environment validation summary |
| [`survyai/gui/launch_args.py`](../survyai/gui/launch_args.py) | CLI-to-GUI prompt prefill and optional auto-run |
| [`survyai/gui/styles.py`](../survyai/gui/styles.py) | Shared QSS styling for a professional Windows feel |

## Delivered product features

- Login/profile section
- License / subscription status card
- Project/workspace selector
- Prompt/task console
- Tool permission dialog for internet usage
- Persistent output history
- Settings page
- Diagnostics/log export
- First-run onboarding
- Environment validation
- AutoCAD detection
- Data-folder selection
- First-run tutorial content
- Long-running job progress activity
- Soft cancellation
- Retry last / retry selected history item
- Safe mode that disables external integrations for troubleshooting

## How to run

From the project root (with venv activated and dependencies installed):

```bash
pip install -r requirements.txt
python -m survyai.gui
```

Optional command-line prompt prefill:

```bash
python -m survyai.gui "Generate a cadastral plan for ..."
python -m survyai.gui --run "Generate a cadastral plan for ..."
```

CLI entry points remain available:

```bash
python -m survyai.cli gui
python -m cli gui
```

## Desktop architecture decisions

1. **PySide6 (Qt 6)** — Stable Windows desktop foundation with strong dialog, threading, and packaging support.
2. **Persistent desktop state** — GUI/product state is kept separate from `.env` runtime settings so onboarding/history/workspace survive restarts cleanly.
3. **Workspace-aware runs** — each task executes with the selected workspace as process working directory, which is important for relative file outputs.
4. **Soft cancel** — the UI allows cancellation requests, but because the current agent/AutoCAD/LLM stack is blocking, cancellation takes effect after the current step completes.
5. **Safe mode** — implemented as a restrictive runtime feature-flag layer that disables integration tool families even on builder/dev machines.
6. **Backend-ready account/license UI** — the GUI already exposes account and license surfaces, while the current source of truth remains local feature flags / local profile until Phase 3 backend work lands.

## Current limitations

- Login is currently **local desktop profile scaffolding**, not real cloud authentication yet.
- License/subscription status is currently derived from **local feature flags / token presence**, not live billing.
- Progress is **activity streaming**, not true token-by-token model streaming.
- Cancel is **soft cancel**, not hard interruption of AutoCAD/LLM execution.

## Next steps after Phase 2

- Real backend auth and Stripe-backed entitlements
- Streaming agent callbacks for richer live progress
- Signed installer, auto-update, and production packaging
