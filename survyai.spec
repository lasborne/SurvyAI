# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for the SurvyAI Windows desktop application.

Build:
    pip install -r requirements.txt -r requirements-build.txt
    pyinstaller --noconfirm --clean survyai.spec

Output:
    dist/SurvyAI/SurvyAI.exe   (one-dir build; recommended for installers)

Notes
-----
- One-DIR build (not one-file): faster startup and far simpler signing/installer
  packaging. The Inno Setup script (installer/survyai.iss) wraps dist/SurvyAI.
- Local embedding models (torch / sentence-transformers / transformers) are NOT
  bundled by default to keep the installer lean. Set the env var
  SURVYAI_BUNDLE_LOCAL_EMBEDDINGS=1 before building to include them.
- The cloud backend (survyai_cloud) is intentionally excluded from the desktop
  bundle.
"""

import os
from PyInstaller.utils.hooks import collect_all, collect_submodules

block_cipher = None

# --- Application data files (relative to project root) ----------------------
datas = [
    ("README.md", "."),
    ("docs/GETTING_STARTED.md", "docs"),
    ("agent/agent_config.json", "agent"),
    ("agent/system_prompt.txt", "agent"),
    ("survyai/packaging_manifest.json", "survyai"),
    # Default cadastral DWG + seed profile so CAD plotting works on first install.
    ("bundled_templates/survey_plan_template3.dwg", "bundled_templates"),
    ("bundled_templates/survey_plan_template3.json", "bundled_templates"),
]

binaries = []
hiddenimports = []

# --- Collect data/binaries/submodules for packages that need it ------------
# langchain ecosystem + langgraph rely heavily on dynamic imports.
_collect_all_pkgs = [
    "langchain",
    "langchain_core",
    "langchain_community",
    "langchain_openai",
    "langchain_anthropic",
    "langchain_google_genai",
    "langgraph",
    "tiktoken",
    "tiktoken_ext",
    "pydantic",
    "pydantic_settings",
    # Geospatial stacks ship CRS/proj data and compiled deps:
    "pyproj",
    "fiona",
    "shapely",
    "geopandas",
    "ezdxf",
]

for _pkg in _collect_all_pkgs:
    try:
        d, b, h = collect_all(_pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as exc:  # package optional / not installed in this env
        print(f"[survyai.spec] collect_all skipped for {_pkg}: {exc}")

# Submodules that are imported dynamically:
for _pkg in ("langchain", "langchain_community", "langgraph"):
    try:
        hiddenimports += collect_submodules(_pkg)
    except Exception as exc:
        print(f"[survyai.spec] collect_submodules skipped for {_pkg}: {exc}")

# Windows COM for AutoCAD automation:
hiddenimports += [
    "win32com",
    "win32com.client",
    "pythoncom",
    "pywintypes",
    "psycopg",
    "psycopg_pool",
    "pgvector",
    # Package `tools.geographic_calculator/` shadows a plain module name; load core explicitly.
    "tools.geographic_calculator_core",
]

# --- Optional heavy local-embedding stack ----------------------------------
_bundle_local_embeddings = os.environ.get("SURVYAI_BUNDLE_LOCAL_EMBEDDINGS", "0") == "1"
if _bundle_local_embeddings:
    for _pkg in ("torch", "sentence_transformers", "transformers", "safetensors"):
        try:
            d, b, h = collect_all(_pkg)
            datas += d
            binaries += b
            hiddenimports += h
        except Exception as exc:
            print(f"[survyai.spec] collect_all skipped for {_pkg}: {exc}")

# --- Excludes (keep the bundle lean + avoid server code on the desktop) -----
excludes = [
    "survyai_cloud",
    "alembic",
    "uvicorn",
    "fastapi",
    "tkinter",
    "matplotlib",
    "pytest",
    "IPython",
    "notebook",
]
if not _bundle_local_embeddings:
    excludes += ["torch", "sentence_transformers", "transformers", "safetensors"]


a = Analysis(
    ["run_survyai_desktop.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SurvyAI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # GUI app: no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon="installer/survyai.ico",  # add a real .ico to enable a custom icon
    version=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="SurvyAI",
)
