"""
================================================================================
AutoCAD COM API Processor
================================================================================

This module provides the interface between SurvyAI and AutoCAD, allowing the
AI agent to control AutoCAD through its COM (Component Object Model) API.

WHAT IS COM?
------------
COM (Component Object Model) is a Microsoft technology that allows different
software applications to communicate with each other. AutoCAD exposes a COM
interface that lets Python (via pywin32) control it programmatically.

HOW IT WORKS:
-------------
1. Python connects to AutoCAD using win32com.client
2. We get a reference to the AutoCAD.Application object
3. Through this object, we can:
   - Open and save drawings
   - Read entities (lines, polylines, text, etc.)
   - Execute AutoCAD commands
   - Calculate areas, distances, etc.

REQUIREMENTS:
-------------
- Windows operating system (COM is Windows-only)
- AutoCAD installed (any recent version)
- pywin32 package: pip install pywin32

ENTITY TYPES:
-------------
AutoCAD has many entity types. The most common for surveying are:
- LWPOLYLINE: Lightweight polyline (boundary lines)
- LINE: Simple line segment
- CIRCLE: Circle (control points, etc.)
- TEXT/MTEXT: Text annotations
- HATCH: Filled areas
- POINT: Survey points

COLOR CODES:
------------
AutoCAD uses ACI (AutoCAD Color Index) for colors:
- 1 = Red (often used for property boundaries)
- 2 = Yellow
- 3 = Green
- 4 = Cyan
- 5 = Blue
- 6 = Magenta
- 7 = White/Black (depends on background)

Author: SurvyAI Team
License: MIT
================================================================================
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.logger import get_logger

# ==============================================================================
# LOGGING
# ==============================================================================

# Get logger for this module
# All AutoCAD-related logs will be tagged appropriately
logger = get_logger(__name__)


# ==============================================================================
# CONSTANTS
# ==============================================================================

# AutoCAD space constants
# Model Space is where the actual drawing is created
# Paper Space is for layout/printing
AC_MODEL_SPACE = 1
AC_PAPER_SPACE = 2

# Mapping from AutoCAD's internal entity names to common names
# AutoCAD uses "AcDb" prefix internally (e.g., "AcDbLine")
# We convert these to user-friendly names (e.g., "LINE")
ENTITY_TYPES = {
    "AcDbLine": "LINE",              # Simple line segment
    "AcDbPolyline": "LWPOLYLINE",    # Lightweight polyline (most common)
    "AcDb2dPolyline": "POLYLINE",    # 2D polyline (older format)
    "AcDb3dPolyline": "3DPOLYLINE",  # 3D polyline
    "AcDbCircle": "CIRCLE",          # Circle
    "AcDbArc": "ARC",                # Arc (portion of circle)
    "AcDbText": "TEXT",              # Single-line text
    "AcDbMText": "MTEXT",            # Multi-line text
    "AcDbHatch": "HATCH",            # Hatched/filled area
    "AcDbPoint": "POINT",            # Point entity
    "AcDbSpline": "SPLINE",          # Spline curve
    "AcDbBlockReference": "INSERT",  # Block reference
    "AcDbTable": "TABLE",            # Table object (title blocks, schedules, etc.)
}

# AutoCAD Color Index (ACI) to color name mapping
# These are the standard AutoCAD colors (index 1-9)
# Index 256 = "ByLayer" (inherits layer color)
# Index 0 = "ByBlock" (inherits block color)
ACI_COLORS = {
    1: "red",        # Often used for property boundaries
    2: "yellow",     # 
    3: "green",      # 
    4: "cyan",       # 
    5: "blue",       # 
    6: "magenta",    # 
    7: "white",      # (appears black on white background)
    8: "dark_gray",  # 
    9: "light_gray", # 
}


# ==============================================================================
# MAIN CLASS
# ==============================================================================

class AutoCADProcessor:
    """
    Interface with AutoCAD application via COM automation.
    
    This class provides the bridge between Python and AutoCAD, allowing
    the AI agent to perform CAD operations using AutoCAD's native engine.
    
    Key Capabilities:
    -----------------
    1. Drawing Management:
       - Open DWG and DXF files
       - Get drawing information (units, layers, etc.)
       
    2. Entity Reading:
       - Extract all entities or filter by type/layer/color
       - Get entity properties (coordinates, area, length)
       
    3. Text Extraction:
       - Get all text content from drawings
       - Search for text matching patterns (regex supported)
       
    4. Geometric Calculations:
       - Calculate areas of closed shapes
       - Uses AutoCAD's native precision
       
    5. Command Execution:
       - Execute any AutoCAD command
       
    Usage Example:
    --------------
    ```python
    # Create processor (doesn't connect yet)
    acad = AutoCADProcessor(auto_connect=False)
    
    # Connect to AutoCAD (must be running)
    if acad.connect():
        # Open a drawing
        result = acad.open_drawing("survey.dwg")
        
        # Calculate area of red boundaries
        areas = acad.calculate_area(color="red")
        print(f"Total area: {areas['total_area_sq_units']} sq units")
        
        # Find owner name in text
        texts = acad.search_text("property of")
        for match in texts['matches']:
            print(f"Found: {match['content']}")
    ```
    
    Thread Safety:
    --------------
    COM objects are apartment-threaded. Each thread that uses this class
    must call pythoncom.CoInitialize() first. This is handled automatically
    in the connect() method.
    """

    # How long a successful command-cancel stays trusted before ESC is re-sent.
    _QUIESCE_REARM_S: float = 3.0

    def __init__(self, auto_connect: bool = True):
        """
        Initialize the AutoCAD processor.
        
        Args:
            auto_connect: If True, attempt to connect to AutoCAD immediately.
                         If False, connection is deferred until connect() is called.
                         Set to False if AutoCAD might not be running at startup.
        
        Attributes:
            acad: Reference to AutoCAD.Application COM object
            doc: Reference to the active AutoCAD document
            _connected: Boolean tracking connection status
        """
        # COM object references (None until connected)
        self.acad = None  # AutoCAD.Application
        self.doc = None   # ActiveDocument
        
        # Connection state tracking
        self._connected = False
        # When set, COM operations prefer re-activating this drawing (cadastral pipeline output).
        self._workflow_doc_path: Optional[str] = None
        # COM recovery / warning throttle (separate-owner batches can thrash Documents).
        self._last_com_recover_ts: float = 0.0
        self._workflow_open_fail_until: float = 0.0
        self._last_doc_warn_ts: float = 0.0
        self._last_doc_warn_msg: str = ""
        # Document-scoped entity/table handle cache (invalidated on open/close/recover).
        self._entity_handle_cache: Dict[str, Any] = {}
        self._handle_cache_doc_key: Optional[str] = None
        self._documents_open_count: int = 0
        self._modelspace_scan_count: int = 0
        # Document-scoped ModelSpace classification snapshot (see _modelspace_records).
        self._ms_snapshot: Optional[List[Dict[str, Any]]] = None
        self._ms_snapshot_key: Optional[str] = None
        self._ms_snapshot_count: int = -1
        # Command-cancel (ESC) throttle state (see quiesce_autocad).
        self._last_quiesce_ts: float = 0.0
        self._quiesce_needed: bool = True
        
        # Optionally connect immediately
        if auto_connect:
            self.connect()

    def _invalidate_handle_cache(self) -> None:
        self._entity_handle_cache.clear()
        self._handle_cache_doc_key = None
        self._ms_snapshot = None
        self._ms_snapshot_key = None
        self._ms_snapshot_count = -1
        # Document identity or contents changed; re-arm the command cancel.
        self._quiesce_needed = True

    def _modelspace_records(self, *, force: bool = False) -> List[Dict[str, Any]]:
        """One classified pass over ModelSpace, reused across queries.

        Every ``ms.Item(i)`` and each ``Layer`` / ``ObjectName`` read is a separate COM
        round-trip, so on a survey template with thousands of entities a single scan
        cost seconds — and the cadastral plot scans repeatedly (bbox fitting, table
        discovery, text-height sampling, handle lookups).

        Only *classification* is cached (handle, layer, object name); those change
        solely when entities are created or removed, which always changes
        ``ModelSpace.Count``. The count is therefore re-read (one COM call) as a cheap
        validity check. Geometry is never cached — callers read position/bounding boxes
        from the live proxy so moves and scales stay exact.
        """
        if self.doc is None:
            return []
        try:
            ms = self.doc.ModelSpace
            count = int(getattr(ms, "Count", 0) or 0)
        except Exception:
            self._ms_snapshot = None
            raise
        doc_key = self._handle_cache_key_for_doc()
        if (
            not force
            and self._ms_snapshot is not None
            and self._ms_snapshot_key == doc_key
            and self._ms_snapshot_count == count
        ):
            return self._ms_snapshot

        records: List[Dict[str, Any]] = []
        self._modelspace_scan_count = int(getattr(self, "_modelspace_scan_count", 0) or 0) + 1
        try:
            from survyai.perf import incr

            incr("autocad_modelspace_scans")
        except Exception:
            pass
        for i in range(count):
            try:
                e = ms.Item(i)
                records.append(
                    {
                        "obj": e,
                        "handle": str(getattr(e, "Handle", "") or ""),
                        "layer": str(getattr(e, "Layer", "") or "").upper(),
                        "name": str(getattr(e, "ObjectName", "") or ""),
                    }
                )
            except Exception:
                continue
        self._ms_snapshot = records
        self._ms_snapshot_key = doc_key
        self._ms_snapshot_count = count
        return records

    def _handle_cache_key_for_doc(self) -> Optional[str]:
        try:
            if self.doc is None:
                return None
            full = str(getattr(self.doc, "FullName", "") or "").strip()
            if full:
                return str(Path(full).resolve()).lower()
            name = str(getattr(self.doc, "Name", "") or "").strip().lower()
            return name or None
        except Exception:
            return None

    def _get_entity_by_handle(
        self,
        handle: str,
        *,
        object_name: Optional[str] = None,
        scan_modelspace: bool = True,
    ) -> Any:
        """Resolve an entity by handle using a document-scoped cache; fall back to ModelSpace scan."""
        h = str(handle or "")
        if not h or self.doc is None:
            return None
        doc_key = self._handle_cache_key_for_doc()
        cache_key = f"{h}|{object_name or '*'}"
        force_rescan = False
        if doc_key and self._handle_cache_doc_key == doc_key:
            cached = self._entity_handle_cache.get(cache_key)
            if cached is not None:
                try:
                    _ = getattr(cached, "Handle", None)
                    if object_name and str(getattr(cached, "ObjectName", "")) != object_name:
                        raise RuntimeError("stale object name")
                    try:
                        from survyai.perf import incr

                        incr("autocad_handle_cache_hit")
                    except Exception:
                        pass
                    return cached
                except Exception:
                    self._entity_handle_cache.pop(cache_key, None)
                    # A dead proxy proves AutoCAD replaced this document's objects, so
                    # every cached proxy for it is suspect — not just this one.
                    force_rescan = True
                    try:
                        from survyai.perf import incr

                        incr("autocad_handle_cache_stale")
                    except Exception:
                        pass
        if not scan_modelspace:
            return None

        def _find(records: List[Dict[str, Any]]) -> Any:
            for rec in records:
                try:
                    if rec["handle"] != h:
                        continue
                    if object_name and rec["name"] != object_name:
                        continue
                    e = rec["obj"]
                    if doc_key:
                        self._handle_cache_doc_key = doc_key
                        self._entity_handle_cache[cache_key] = e
                        on = rec["name"]
                        if object_name is None and on:
                            self._entity_handle_cache[f"{h}|{on}"] = e
                    return e
                except Exception:
                    continue
            return None

        try:
            reused = self._ms_snapshot is not None and not force_rescan
            found = _find(self._modelspace_records(force=force_rescan))
            if found is None and reused:
                # Entity churn can leave ModelSpace.Count unchanged; confirm a miss
                # against a fresh pass before reporting "not found".
                found = _find(self._modelspace_records(force=True))
            return found
        except Exception:
            return None

    def _autocad_version_catalog(self) -> List[Tuple[int, List[str]]]:
        """
        Return supported AutoCAD COM ProgID aliases by release year.

        Notes:
        - AutoCAD COM ProgIDs vary by release and can appear with or without a
          trailing ".0" style minor version.
        - We support versions down to AutoCAD 2007 as requested.
        """
        catalog: List[Tuple[int, List[str]]] = [
            (2025, ["AutoCAD.Application.24.4", "AutoCAD.Application.25"]),
            (2024, ["AutoCAD.Application.24.3", "AutoCAD.Application.24"]),
            (2023, ["AutoCAD.Application.24.2"]),
            (2022, ["AutoCAD.Application.24.1", "AutoCAD.Application.23"]),
            (2021, ["AutoCAD.Application.24.0", "AutoCAD.Application.22"]),
            (2020, ["AutoCAD.Application.23.1", "AutoCAD.Application.21"]),
            (2019, ["AutoCAD.Application.23.0", "AutoCAD.Application.20"]),
            (2018, ["AutoCAD.Application.22.0", "AutoCAD.Application.19"]),
            (2017, ["AutoCAD.Application.21.0", "AutoCAD.Application.18"]),
            (2016, ["AutoCAD.Application.20.1"]),
            (2015, ["AutoCAD.Application.20.0"]),
            (2014, ["AutoCAD.Application.19.1"]),
            (2013, ["AutoCAD.Application.19.0"]),
            (2012, ["AutoCAD.Application.18.2"]),
            (2011, ["AutoCAD.Application.18.1"]),
            (2010, ["AutoCAD.Application.18.0"]),
            (2009, ["AutoCAD.Application.17.2"]),
            (2008, ["AutoCAD.Application.17.1"]),
            (2007, ["AutoCAD.Application.17.0"]),
        ]
        return catalog

    def _unique_progids(self, progids: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for progid in progids:
            key = str(progid).strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(str(progid))
        return out

    def _versioned_autocad_progids_newest_first(self) -> List[str]:
        """
        Version-specific ProgIDs only, newest → oldest (2007…recent).
        Used after ``AutoCAD.Application`` when attaching to a running session.
        """
        progids: List[str] = []
        for _year, aliases in sorted(self._autocad_version_catalog(), key=lambda x: x[0], reverse=True):
            progids.extend(aliases)
        return self._unique_progids(progids)

    def _attach_running_autocad(self):
        """
        Connect to an **already running** AutoCAD.

        Order:
        1) ``GetActiveObject("AutoCAD.Application")`` — usually the automation
           server for whichever session the user actually opened (avoids picking
           2021-specific ProgID when e.g. 2019 is the running window).
        2) Version-specific ProgIDs, newest → … → 2007 — for installs that do
           not register the generic ProgID.
        """
        import win32com.client

        for progid in ("AutoCAD.Application",):
            try:
                acad = win32com.client.GetActiveObject(progid)
                return acad, progid
            except Exception:
                continue
        for progid in self._versioned_autocad_progids_newest_first():
            try:
                acad = win32com.client.GetActiveObject(progid)
                return acad, progid
            except Exception:
                continue
        return None, None

    def _startup_autocad_progids(self) -> List[str]:
        """
        Order for starting AutoCAD when none is currently open.
        Preference:
        1. AutoCAD 2021 if installed
        2. Latest installed version available
        3. Generic AutoCAD ProgID as final fallback
        """
        catalog = {year: aliases for year, aliases in self._autocad_version_catalog()}
        progids: List[str] = []

        # Default preferred release
        progids.extend(catalog.get(2021, []))

        # Then newest -> oldest, excluding the already-preferred 2021
        for year, aliases in sorted(self._autocad_version_catalog(), key=lambda x: x[0], reverse=True):
            if year == 2021:
                continue
            progids.extend(aliases)

        progids.append("AutoCAD.Application")
        return self._unique_progids(progids)
    
    # ==========================================================================
    # CONNECTION MANAGEMENT
    # ==========================================================================
    
    def connect(self) -> bool:
        """
        Establish connection to AutoCAD via COM.
        
        This method:
        1. Initializes COM for the current thread
        2. Attempts to connect to a running AutoCAD instance
        3. If none is running, optionally starts a new instance
        
        Returns:
            bool: True if connection successful, False otherwise
            
        Notes:
            - AutoCAD must be installed on the system
            - On first connection to a new instance, AutoCAD may take
              several seconds to fully initialize
        """
        # ------------------------------------------------------------------
        # Step 1: Initialize COM for this thread
        # ------------------------------------------------------------------
        try:
            import pythoncom
            # CoInitialize sets up COM for the calling thread
            # This is required before any COM operations
            pythoncom.CoInitialize()
        except Exception:
            # May already be initialized, which is fine
            pass
        
        # ------------------------------------------------------------------
        # Step 2: Try to connect to AutoCAD
        # ------------------------------------------------------------------
        try:
            import win32com.client

            startup_progids = self._startup_autocad_progids()

            # First, attach to any already running AutoCAD (prefer user's open session).
            connected = False
            self.acad, progid_used = self._attach_running_autocad()
            if self.acad is not None:
                ver = None
                try:
                    ver = str(getattr(self.acad, "Version", "") or "")
                except Exception:
                    ver = None
                logger.info(
                    f"Connected to running AutoCAD via {progid_used}"
                    + (f" (Version: {ver})" if ver else "")
                )
                connected = True

            if not connected:
                # No running instance - start one using the preferred order:
                # 2021 first, then the newest installed version that responds.
                logger.info("No running AutoCAD instance found. Attempting to start preferred AutoCAD version...")

                for progid in startup_progids:
                    try:
                        try:
                            self.acad = win32com.client.DispatchEx(progid)
                            logger.info(f"Started new AutoCAD instance via DispatchEx ({progid})")
                        except Exception:
                            self.acad = win32com.client.Dispatch(progid)
                            logger.info(f"Started new AutoCAD instance via Dispatch ({progid})")

                        self.acad.Visible = True

                        logger.info("Waiting for AutoCAD to initialize...")
                        time.sleep(3)

                        connected = True
                        break

                    except Exception as e:
                        logger.debug(f"Could not connect via {progid}: {e}")
                        continue

                if not connected:
                    logger.error("=" * 60)
                    logger.error("AUTOCAD CONNECTION FAILED")
                    logger.error("=" * 60)
                    logger.error("Could not connect to or start AutoCAD.")
                    logger.error("")
                    logger.error("TROUBLESHOOTING STEPS:")
                    logger.error("1. Open AutoCAD manually first, then run this command again")
                    logger.error("2. Ensure AutoCAD (not AutoCAD LT) is installed")
                    logger.error("3. Check if Python and AutoCAD are both 64-bit or both 32-bit")
                    logger.error("4. Run your terminal/IDE as Administrator")
                    logger.error("5. Check if antivirus is blocking COM automation")
                    logger.error("")
                    logger.error("To check installed AutoCAD versions, run:")
                    logger.error('  reg query "HKEY_LOCAL_MACHINE\\SOFTWARE\\Autodesk\\AutoCAD" /s')
                    logger.error("=" * 60)
                    return False
            
            # Mark as connected
            self._connected = True
            
            # ------------------------------------------------------------------
            # Step 3: Get the active document if one is open
            # ------------------------------------------------------------------
            try:
                self.doc = self.acad.ActiveDocument
                logger.info(f"Active document: {self.doc.Name}")
            except Exception:
                # No document open, which is okay
                self.doc = None
                logger.info("No active document")
            
            return True
            
        except ImportError:
            logger.error("pywin32 not available. Install with: pip install pywin32")
            return False
            
        except Exception as e:
            logger.error(f"Failed to connect to AutoCAD: {e}")
            return False
    
    @property
    def is_connected(self) -> bool:
        """
        Check if currently connected to AutoCAD.
        
        This property verifies the connection is still valid by attempting
        to access a property of the AutoCAD application object.
        
        Returns:
            bool: True if connected and responsive, False otherwise
        """
        if not self._connected or not self.acad:
            return False
            
        try:
            # Try to access a property to verify connection
            # If AutoCAD was closed, this will raise an exception
            _ = self.acad.Name
            return True
        except Exception:
            # Connection lost
            self._connected = False
            return False
    
    # ==========================================================================
    # DRAWING MANAGEMENT
    # ==========================================================================
    
    def close_drawing_if_open(self, file_path: str, *, save_changes: bool = False) -> Dict[str, Any]:
        """
        Close a drawing if it is currently open in AutoCAD.

        This is used to avoid creating unintended extra files when regenerating an
        output DWG that is already open in the UI.
        """
        # Ensure we're connected
        if not self.is_connected:
            if not self.connect():
                return {"success": False, "error": "Not connected to AutoCAD"}

        from pathlib import Path
        import time

        target = Path(file_path).resolve()
        target_name = target.name.lower()

        # Cancel any interactive AutoCAD command (Zoom/Window/etc.) that blocks Close/Open.
        try:
            self.quiesce_autocad(esc_count=3)
        except Exception:
            pass

        try:
            docs = self.acad.Documents
        except Exception as e:
            return {"success": False, "error": f"AutoCAD Documents unavailable: {e}"}

        closed_any = False
        last_err = None
        try:
            n = int(getattr(docs, "Count", 0))
        except Exception:
            n = 0

        # Iterate backwards — Count shrinks when documents are closed.
        for i in range(n - 1, -1, -1):
            try:
                d = getattr(docs, "Item")(i)
            except Exception as e:
                last_err = e
                continue
            try:
                full = str(getattr(d, "FullName", "") or "")
                nm = str(getattr(d, "Name", "") or "").lower()
                match = False
                try:
                    if full and Path(full).resolve() == target:
                        match = True
                except Exception:
                    pass
                if not match and nm == target_name:
                    match = True
                if not match:
                    continue
                try:
                    d.Close(bool(save_changes))
                    closed_any = True
                    if self.doc is d:
                        self.doc = None
                    time.sleep(0.2)
                except Exception as e:
                    last_err = e
                    continue
            except Exception as e:
                last_err = e
                continue

        return {"success": True, "closed": closed_any, "error": str(last_err) if last_err else None}

    def save_and_close_other_drawings(
        self,
        *,
        exclude_paths: Optional[List[str]] = None,
        close_after_save: bool = True,
    ) -> Dict[str, Any]:
        """
        Save unsaved open drawings and optionally close them so a new CAD run is not blocked.

        Excluded paths (typically protected survey-plan templates) are left untouched and
        are never written. Never-saved drawings (empty FullName) are saved under the user's
        Documents folder before close.
        """
        if not self.is_connected:
            if not self.connect():
                return {"success": False, "error": "Not connected to AutoCAD"}

        from pathlib import Path
        import time

        # Cancel Zoom/pan/in-progress commands so Save/Close are not blocked.
        try:
            self.quiesce_autocad(esc_count=3)
        except Exception:
            pass

        exclude: set[str] = set()
        for raw in exclude_paths or []:
            try:
                if raw:
                    exclude.add(str(Path(str(raw)).resolve()).lower())
            except Exception:
                continue

        try:
            docs = self.acad.Documents
            n = int(getattr(docs, "Count", 0))
        except Exception as e:
            return {"success": False, "error": f"AutoCAD Documents unavailable: {e}"}

        saved_paths: List[str] = []
        closed_paths: List[str] = []
        errors: List[str] = []

        # Iterate backwards — Count shrinks when documents are closed.
        for i in range(n - 1, -1, -1):
            try:
                d = getattr(docs, "Item")(i)
            except Exception as e:
                errors.append(str(e))
                continue

            read_only = False
            try:
                read_only = bool(getattr(d, "ReadOnly", False))
            except Exception:
                pass

            full = ""
            name = ""
            try:
                full = str(getattr(d, "FullName", "") or "").strip()
                name = str(getattr(d, "Name", "") or "").strip() or "Drawing1.dwg"
            except Exception as e:
                errors.append(str(e))
                continue

            resolved_key = ""
            if full:
                try:
                    resolved_key = str(Path(full).resolve()).lower()
                except Exception:
                    resolved_key = full.lower()
                if resolved_key in exclude:
                    continue

            try:
                is_saved = bool(getattr(d, "Saved", True))
            except Exception:
                is_saved = True

            if not is_saved and not read_only:
                try:
                    if full:
                        d.Save()
                        saved_paths.append(full)
                    else:
                        dest_dir = Path.home() / "Documents"
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        dest_name = name if name.lower().endswith(".dwg") else f"{name}.dwg"
                        dest = dest_dir / dest_name
                        # Avoid clobbering an unrelated file with the same default name.
                        if dest.exists():
                            stem = dest.stem
                            suffix = dest.suffix
                            k = 1
                            while True:
                                candidate = dest_dir / f"{stem}_{k}{suffix}"
                                if not candidate.exists():
                                    dest = candidate
                                    break
                                k += 1
                        d.SaveAs(str(dest))
                        full = str(dest)
                        resolved_key = str(Path(full).resolve()).lower()
                        saved_paths.append(full)
                except Exception as e:
                    errors.append(f"Save failed for {name or full or '?'}: {e}")
                    continue

            if not close_after_save:
                continue
            if resolved_key and resolved_key in exclude:
                continue
            try:
                d.Close(False)
                closed_paths.append(full or name)
                time.sleep(0.15)
            except Exception as e:
                errors.append(f"Close failed for {name or full or '?'}: {e}")

        return {
            "success": True,
            "saved": saved_paths,
            "closed": closed_paths,
            "errors": errors,
        }

    def is_drawing_open(self, file_path: str) -> bool:
        """
        True if AutoCAD already has this drawing open as a saved document (FullName matches resolved path).

        Used to avoid closing the user's tab or failing to overwrite the DWG on disk: automation can
        activate the open document and edit in place instead.
        """
        if not self.is_connected:
            if not self.connect():
                return False
        from pathlib import Path

        try:
            target = Path(file_path).resolve()
        except Exception:
            return False
        try:
            docs = self.acad.Documents
            n = int(getattr(docs, "Count", 0))
        except Exception:
            return False
        for i in range(n):
            try:
                d = getattr(docs, "Item")(i)
                full = str(getattr(d, "FullName", "") or "").strip()
                if not full:
                    continue
                try:
                    if Path(full).resolve() == target:
                        return True
                except Exception:
                    continue
            except Exception:
                continue
        return False
    
    def open_drawing(self, file_path: str, read_only: bool = False) -> Dict[str, Any]:
        """
        Open a drawing file in AutoCAD.
        
        Supports both DWG (native AutoCAD) and DXF (exchange format) files.
        If the file is already open, it activates that document instead of
        opening a read-only copy.
        
        Args:
            file_path: Path to the .dwg or .dxf file
            read_only: Open the file in read-only mode (default: False)
            
        Returns:
            Dict containing:
            - success: Boolean indicating if operation succeeded
            - file_path: Absolute path to the opened file
            - drawing_name: Name of the drawing
            - units: Drawing units (Meters, Feet, etc.)
            - layers: List of layer names in the drawing
            - entity_count: Total number of entities
            - error: Error message if success is False
            
        Example:
            >>> result = acad.open_drawing("survey.dwg")
            >>> if result["success"]:
            ...     print(f"Opened: {result['drawing_name']}")
            ...     print(f"Units: {result['units']}")
        """
        # Ensure we're connected
        if not self.is_connected:
            if not self.connect():
                return {"success": False, "error": "Not connected to AutoCAD"}

        # Validate file path
        file_path = Path(file_path).resolve()
        if not file_path.exists():
            return {"success": False, "error": f"File not found: {file_path}"}

        file_name = file_path.name.lower()
        full_path_str = str(file_path)

        # Leave any stuck interactive command before Documents.Open (common hang).
        try:
            self.quiesce_autocad(esc_count=3)
        except Exception:
            pass

        try:
            # COM calls can be intermittently rejected when AutoCAD is busy/modal.
            # Bound total open/recovery time so busy AutoCAD cannot consume minutes.
            open_deadline = time.time() + 45.0

            def _com_retry(fn, attempts: int = 10, base_sleep: float = 0.2):
                return self._com_retry(
                    fn, attempts=attempts, base_sleep=base_sleep, deadline_ts=open_deadline
                )

            # AutoCAD COM collections are not always directly iterable.
            # Use index-based access via .Count / .Item() for maximum compatibility.
            def _iter_docs():
                try:
                    docs = _com_retry(lambda: self.acad.Documents, attempts=8)
                    n = int(_com_retry(lambda: getattr(docs, "Count"), attempts=8))
                except Exception:
                    n = 0
                for i in range(n):
                    try:
                        docs = _com_retry(lambda: self.acad.Documents, attempts=8)
                        yield _com_retry(lambda: getattr(docs, "Item")(i), attempts=8)
                    except Exception:
                        continue

            # ------------------------------------------------------------------
            # Step 1: Check if file is already open
            # ------------------------------------------------------------------
            existing_doc = None
            try:
                for doc in _iter_docs():
                    try:
                        doc_path = Path(
                            str(
                                _com_retry(
                                    lambda d=doc: getattr(d, "FullName", "") or "",
                                    attempts=4,
                                )
                            )
                        ).resolve()
                        doc_name = self._safe_doc_name(doc, deadline_ts=open_deadline).lower()
                        if doc_path == file_path or doc_name == file_name:
                            existing_doc = doc
                            logger.info("Drawing already open: %s", doc_name)
                            break
                    except Exception:
                        continue
            except Exception as e:
                logger.debug(f"Could not enumerate documents: {e}")
            
            # ------------------------------------------------------------------
            # Step 2: Activate existing or open new
            # ------------------------------------------------------------------
            if existing_doc:
                # Activate the existing document — already loaded; skip long readiness wait.
                _com_retry(lambda: existing_doc.Activate(), attempts=8)
                self.doc = existing_doc
                self._invalidate_handle_cache()
                logger.info(
                    "Activated existing document: %s",
                    self._safe_doc_name(self.doc, deadline_ts=open_deadline),
                )
                activated_existing = True
            else:
                activated_existing = False
                # If Documents is already wedged, recover before Open.
                try:
                    _ = _com_retry(lambda: int(self.acad.Documents.Count), attempts=3)
                except Exception as probe_exc:
                    if not self._settle_busy_com(probe_exc) and self._com_error_is_broken_proxy(
                        probe_exc
                    ):
                        if not self.recover_com_session(force=True, reason=str(probe_exc)):
                            return {
                                "success": False,
                                "stage": "autocad_open",
                                "error": f"AutoCAD COM proxy stale before open: {probe_exc}",
                            }
                        time.sleep(0.4)
                # Close any existing read-only copies of the same file first
                try:
                    for doc in _iter_docs():
                        try:
                            name = self._safe_doc_name(doc, deadline_ts=open_deadline).lower()
                            if name == file_name and getattr(doc, "ReadOnly", False):
                                logger.info(f"Closing read-only copy: {name}")
                                _com_retry(lambda: doc.Close(False), attempts=6)  # False = don't save
                        except Exception:
                            continue
                except Exception:
                    pass
                
                # Open the drawing fresh.
                # Cap retries: unbounded COM loops + "<unknown>.Open" previously hung the GUI run.
                # RPC_E_CALL_REJECTED (-2147418111) is common while AutoCAD is modal/busy —
                # use longer backoff than generic COM probes.
                try:
                    docs = _com_retry(lambda: self.acad.Documents, attempts=8, base_sleep=0.25)
                    self.doc = _com_retry(
                        lambda: getattr(docs, "Open")(full_path_str, read_only),
                        attempts=12,
                        base_sleep=0.35,
                    )
                    self._documents_open_count = int(getattr(self, "_documents_open_count", 0) or 0) + 1
                    try:
                        from survyai.perf import incr

                        incr("autocad_documents_open")
                    except Exception:
                        pass
                    self._invalidate_handle_cache()
                except Exception as open_exc:
                    # Broken Documents proxy / busy reject — reconnect, then retry patiently once.
                    logger.warning(
                        "Documents.Open failed (%s); reconnecting AutoCAD COM…",
                        open_exc,
                    )
                    try:
                        if not self.recover_com_session(force=True, reason=str(open_exc)):
                            raise open_exc
                        self.quiesce_autocad(esc_count=4)
                        time.sleep(1.0)
                        docs = _com_retry(lambda: self.acad.Documents, attempts=6, base_sleep=0.3)
                        try:
                            self.doc = _com_retry(
                                lambda: getattr(docs, "Open")(full_path_str, read_only),
                                attempts=8,
                                base_sleep=0.4,
                            )
                        except Exception:
                            # Some builds reject read_only Open while busy; try writable open.
                            self.doc = _com_retry(
                                lambda: getattr(docs, "Open")(full_path_str),
                                attempts=8,
                                base_sleep=0.4,
                            )
                    except Exception:
                        raise open_exc

                try:
                    opened_name = self._safe_doc_name(self.doc, deadline_ts=open_deadline)
                except Exception as name_exc:
                    if not self._settle_busy_com(name_exc) and self._com_error_is_broken_proxy(
                        name_exc
                    ):
                        logger.warning(
                            "Post-open .Name failed (%s); recovering COM once…", name_exc
                        )
                        if self.recover_com_session(force=True, reason=f"Open.Name: {name_exc}"):
                            if not self._activate_document_by_path(file_path, deadline_ts=open_deadline):
                                return {
                                    "success": False,
                                    "stage": "autocad_open",
                                    "error": (
                                        f"AutoCAD opened the drawing but the COM document proxy is stale "
                                        f"(Open.Name). Recovered once; please retry. Detail: {name_exc}"
                                    ),
                                }
                            opened_name = self._safe_doc_name(self.doc, deadline_ts=open_deadline)
                        else:
                            return {
                                "success": False,
                                "stage": "autocad_open",
                                "error": f"Open.Name: {name_exc}",
                            }
                    else:
                        raise
                logger.info(f"Opened new document: {opened_name}")
            
            # ------------------------------------------------------------------
            # Step 3: Ensure document is activated (never re-Open after success)
            # ------------------------------------------------------------------
            try:
                cur_name = self._safe_doc_name(self.doc, deadline_ts=open_deadline).lower()
                if cur_name != file_name:
                    if not self._activate_document_by_path(file_path, deadline_ts=open_deadline):
                        logger.warning("Could not activate document by path: %s", file_path)
                else:
                    _com_retry(lambda: self.doc.Activate(), attempts=8)
                    time.sleep(0.05 if activated_existing else 0.2)
            except Exception as e:
                logger.warning(f"Could not activate document: {e}")
                if not self._settle_busy_com(e) and self._com_error_is_broken_proxy(e):
                    self.recover_com_session(force=True, reason=f"activate: {e}")
                    self._activate_document_by_path(file_path, deadline_ts=open_deadline)
            
            # ------------------------------------------------------------------
            # Step 4: Wait for document to fully load and verify it's accessible
            # ------------------------------------------------------------------
            # Already-open docs need only a short ModelSpace probe; new opens may
            # need longer (capped) readiness for complex templates.
            max_wait = 2.0 if activated_existing else min(25.0, max(1.0, open_deadline - time.time()))
            # Poll readiness on a rising interval: a template that settles in ~0.1s
            # should not be billed a full coarse tick, while a slow load still backs
            # off instead of spinning on COM.
            wait_interval = 0.05 if activated_existing else 0.08
            max_interval = 0.15 if activated_existing else 0.5
            waited = 0.0
            attempts_made = 0
            doc_ready = False
            
            while waited < max_wait and time.time() < open_deadline:
                try:
                    # Prefer pinned path activation over ActiveDocument guessing.
                    if not self._activate_document_by_path(file_path, deadline_ts=open_deadline):
                        self.doc = _com_retry(lambda: self.acad.ActiveDocument, attempts=8)

                    # Verify it's the correct document
                    cur_name = self._safe_doc_name(self.doc, deadline_ts=open_deadline).lower()
                    if cur_name != file_name:
                        self._activate_document_by_path(file_path, deadline_ts=open_deadline)
                    
                    # Try to access modelspace - this will fail if doc isn't ready
                    _ = _com_retry(lambda: self.doc.ModelSpace.Count, attempts=8)
                    
                    # Try to access document name to ensure it's fully loaded
                    _ = self._safe_doc_name(self.doc, deadline_ts=open_deadline)
                    
                    # If we get here, document is ready
                    doc_ready = True
                    break
                except Exception as e:
                    attempts_made += 1
                    logger.debug(f"Waiting for document to load... ({waited:.1f}s) - {e}")
                    # Never recover on the first miss: a loading document reports the
                    # same errors as a dead proxy.
                    if self._com_error_is_broken_proxy(e) and attempts_made > 1:
                        self.recover_com_session(force=True, reason=str(e))
                        self._activate_document_by_path(file_path, deadline_ts=open_deadline)
                    time.sleep(wait_interval)
                    waited += wait_interval
                    wait_interval = min(max_interval, wait_interval * 1.6)
            
            if not doc_ready:
                logger.error(f"Document did not become ready after {max_wait} seconds")
                return {
                    "success": False,
                    "stage": "autocad_open",
                    "error": f"Document opened but did not become ready after {max_wait} seconds. The file may be corrupted or AutoCAD may need more time."
                }
            
            # Give a bit more time for the UI to settle (skip heavy settle for re-activate)
            time.sleep(0.05 if activated_existing else 0.5)
            
            # ------------------------------------------------------------------
            # Step 5: Final verification that document is accessible
            # ------------------------------------------------------------------
            try:
                # Activate by path — do NOT call Documents.Open again.
                self._activate_document_by_path(file_path, deadline_ts=open_deadline)
                
                # Verify it's still the correct document
                doc_name = self._safe_doc_name(self.doc, deadline_ts=open_deadline)
                if doc_name.lower() != file_name:
                    logger.warning(f"Active document mismatch: expected {file_name}, got {doc_name}")
                    if not self._activate_document_by_path(file_path, deadline_ts=open_deadline):
                        return {
                            "success": False,
                            "stage": "autocad_open",
                            "error": f"Could not activate drawing {file_name} (active={doc_name})",
                        }
                    doc_name = self._safe_doc_name(self.doc, deadline_ts=open_deadline)
                
                # Final verification - try to access document properties
                is_readonly = getattr(self.doc, 'ReadOnly', False)
                if activated_existing:
                    # Fast path: already-open doc — avoid full ModelSpace / layer scans.
                    entity_count = -1
                    layers: List[str] = []
                else:
                    entity_count = self._count_entities()
                    layers = self._get_layers()
                
                if is_readonly:
                    logger.warning("Document opened in READ-ONLY mode. Some operations may be limited.")
                
                result = {
                    "success": True,
                    "file_path": str(file_path),
                    "drawing_name": doc_name,
                    "units": self._get_units(),
                    "layers": layers,
                    "entity_count": entity_count,
                    "read_only": is_readonly,
                    "activated_existing": bool(activated_existing),
                }
                
                logger.info(
                    "Document ready: %s (%s)",
                    doc_name,
                    "reactivated" if activated_existing else f"{entity_count} entities",
                )
                return result
                
            except Exception as e:
                logger.error(f"Document opened but not accessible: {e}")
                if self._com_error_is_busy(e):
                    # Alive but busy: cancel any pending command, let it settle, and
                    # re-probe once. No teardown — the session is fine.
                    try:
                        self.quiesce_autocad(esc_count=2, force=True)
                    except Exception:
                        pass
                    time.sleep(0.3)
                    try:
                        doc_name = self._safe_doc_name(self.doc, deadline_ts=open_deadline)
                        return {
                            "success": True,
                            "file_path": str(file_path),
                            "drawing_name": doc_name,
                            "units": self._get_units(),
                            "layers": [],
                            "entity_count": -1,
                            "read_only": bool(getattr(self.doc, "ReadOnly", read_only)),
                            "activated_existing": bool(activated_existing),
                        }
                    except Exception:
                        return {
                            "success": False,
                            "stage": "autocad_open",
                            "error": f"AutoCAD busy while opening drawing: {e}",
                        }
                if self._com_error_is_broken_proxy(e):
                    if self.recover_com_session(force=True, reason=str(e)):
                        if self._activate_document_by_path(file_path, deadline_ts=open_deadline):
                            try:
                                doc_name = self._safe_doc_name(self.doc, deadline_ts=open_deadline)
                                return {
                                    "success": True,
                                    "file_path": str(file_path),
                                    "drawing_name": doc_name,
                                    "units": self._get_units(),
                                    "layers": [],
                                    "entity_count": -1,
                                    "read_only": bool(getattr(self.doc, "ReadOnly", read_only)),
                                    "activated_existing": bool(activated_existing),
                                    "recovered_open": True,
                                }
                            except Exception as e3:
                                return {
                                    "success": False,
                                    "stage": "autocad_open",
                                    "error": f"Open.Name recovery failed: {e3}",
                                }
                return {
                    "success": False,
                    "stage": "autocad_open",
                    "error": f"Document opened but not accessible: {e}",
                }
            
        except Exception as e:
            logger.error(f"Failed to open drawing: {e}")
            if self._com_error_is_busy(e):
                # Busy, not broken: wait it out and try to activate the drawing that
                # may already be open, without reconnecting.
                try:
                    self.quiesce_autocad(esc_count=2, force=True)
                except Exception:
                    pass
                time.sleep(0.4)
                try:
                    if self._activate_document_by_path(file_path):
                        return {
                            "success": True,
                            "file_path": str(file_path),
                            "drawing_name": self._safe_doc_name(self.doc),
                            "units": self._get_units(),
                            "layers": [],
                            "entity_count": -1,
                            "read_only": bool(getattr(self.doc, "ReadOnly", read_only)),
                            "activated_existing": True,
                        }
                except Exception:
                    pass
                return {
                    "success": False,
                    "stage": "autocad_open",
                    "error": f"AutoCAD busy while opening drawing: {e}",
                }
            # Last-chance recovery for a dead COM proxy (reference DWG metadata opens).
            # Activate-by-path only — do not issue a second Documents.Open after a prior success path.
            if self._com_error_is_broken_proxy(e):
                try:
                    if self.recover_com_session(force=True, reason=f"open_drawing final: {e}"):
                        self.quiesce_autocad(esc_count=4)
                        time.sleep(0.8)
                        # Prefer activate if already open; only Open when absent.
                        if self._activate_document_by_path(file_path):
                            return {
                                "success": True,
                                "file_path": str(file_path),
                                "drawing_name": self._safe_doc_name(self.doc),
                                "units": self._get_units(),
                                "layers": [],
                                "entity_count": -1,
                                "read_only": bool(getattr(self.doc, "ReadOnly", read_only)),
                                "activated_existing": True,
                                "recovered_open": True,
                            }
                        docs = self._com_retry(
                            lambda: self.acad.Documents, attempts=6, base_sleep=0.3
                        )
                        try:
                            self.doc = self._com_retry(
                                lambda: getattr(docs, "Open")(full_path_str, read_only),
                                attempts=6,
                                base_sleep=0.35,
                            )
                        except Exception:
                            self.doc = self._com_retry(
                                lambda: getattr(docs, "Open")(full_path_str),
                                attempts=6,
                                base_sleep=0.35,
                            )
                        if self.doc is not None:
                            try:
                                self._com_retry(lambda: self.doc.Activate(), attempts=6)
                            except Exception:
                                pass
                            logger.info(
                                "Opened drawing after final COM recovery: %s",
                                self._safe_doc_name(self.doc),
                            )
                            return {
                                "success": True,
                                "file_path": str(file_path),
                                "drawing_name": self._safe_doc_name(self.doc),
                                "units": self._get_units(),
                                "layers": [],
                                "entity_count": -1,
                                "read_only": bool(getattr(self.doc, "ReadOnly", read_only)),
                                "activated_existing": False,
                                "recovered_open": True,
                            }
                except Exception as e2:
                    logger.error("Final open_drawing recovery failed: %s", e2)
                    return {
                        "success": False,
                        "stage": "autocad_open",
                        "error": f"{e}; recovery failed: {e2}",
                    }
            return {"success": False, "stage": "autocad_open", "error": str(e)}

    def open_drawing_resilient(
        self,
        file_path: str,
        *,
        read_only: bool = True,
        attempts: int = 2,
        deadline_s: float = 40.0,
    ) -> Dict[str, Any]:
        """
        Open a drawing with patient COM recovery for reference/input DWG reads.

        Used when metadata must be taken from an existing plan before a plot batch.
        Does not change the successful open_drawing contract — only adds outer retries.
        """
        last: Dict[str, Any] = {"success": False, "error": "open_drawing_resilient: no attempts"}
        path = str(Path(file_path).resolve()) if file_path else ""
        deadline = time.time() + max(5.0, float(deadline_s))
        for i in range(max(1, int(attempts))):
            if time.time() >= deadline:
                last = {
                    "success": False,
                    "stage": "autocad_open",
                    "error": f"Timed out opening drawing after {deadline_s:.0f}s",
                }
                break
            try:
                if not self.is_connected:
                    if not self.connect():
                        last = {"success": False, "stage": "autocad_open", "error": "Not connected to AutoCAD"}
                        time.sleep(0.4 * (i + 1))
                        continue
                if i > 0:
                    # Only reconnect when the previous attempt looked like a dead
                    # session. A busy AutoCAD stays busy across a reconnect, so
                    # recovering there just pays the cost twice.
                    prev_err = Exception(str(last.get("error") or ""))
                    if self._com_error_is_broken_proxy(prev_err) and not self._com_error_is_busy(
                        prev_err
                    ):
                        self.recover_com_session(
                            force=True, reason=f"open_drawing_resilient attempt {i + 1}"
                        )
                    time.sleep(0.5 * i)
                try:
                    self.quiesce_autocad(esc_count=3 + i, force=True)
                except Exception:
                    pass
                # Alternate read_only on later attempts — some AutoCAD builds reject RO Open.
                use_ro = bool(read_only) if i == 0 else False
                result = self.open_drawing(path, read_only=use_ro)
                if result.get("success"):
                    return result
                last = result if isinstance(result, dict) else {"success": False, "error": str(result)}
                last_err = Exception(str(last.get("error") or ""))
                if self._com_error_is_busy(last_err) or self._com_error_is_broken_proxy(last_err):
                    # open_drawing already waited or recovered; avoid long thrash.
                    continue
            except Exception as exc:
                last = {"success": False, "stage": "autocad_open", "error": str(exc)}
            time.sleep(0.4 * (i + 1))
        return last
    
    def execute_command(self, command: str) -> Dict[str, Any]:
        """
        Execute an AutoCAD command.
        
        This sends a command directly to AutoCAD's command line, just as if
        a user typed it. Useful for operations not covered by other methods.
        
        Args:
            command: AutoCAD command string (e.g., "ZOOM E", "REGEN", "AREA")
            
        Returns:
            Dict with success status and message
            
        Common Commands:
            - "ZOOM E" - Zoom to extents (show entire drawing)
            - "REGEN" - Regenerate the drawing display
            - "QSAVE" - Quick save
            - "PURGE ALL" - Remove unused elements
            
        Note:
            Some commands require additional input (like picking points).
            These interactive commands may not work well through this method.
        """
        if not self.ensure_workflow_document():
            return {"success": False, "error": "No active document. Please open a drawing first using autocad_open_drawing."}
        
        try:
            def _send() -> None:
                self.doc.SendCommand(command + "\n")

            self._com_retry(_send, attempts=8, base_sleep=0.25)
            # A dispatched command can leave AutoCAD prompting for input.
            self._quiesce_needed = True

            # Wait for command to complete
            time.sleep(0.3)
            
            return {
                "success": True,
                "command": command,
                "message": f"Command '{command}' executed"
            }
            
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    def set_workflow_document(self, file_path: Optional[str]) -> None:
        """Pin the drawing SurvyAI is automating so COM calls stay on the correct tab."""
        if not file_path:
            self._workflow_doc_path = None
            self._invalidate_handle_cache()
            return
        try:
            new_path = str(Path(file_path).resolve())
        except Exception:
            new_path = str(file_path)
        if self._workflow_doc_path and self._workflow_doc_path.lower() != new_path.lower():
            self._invalidate_handle_cache()
        self._workflow_doc_path = new_path

    @staticmethod
    def _com_error_is_busy(exc: BaseException) -> bool:
        """True when AutoCAD is alive but refusing calls (modal dialog, mid-command).

        ``RPC_E_CALL_REJECTED`` says the callee is busy — the session is perfectly
        healthy. Tearing it down and reconnecting cannot make AutoCAD less busy, and
        the reconnect itself is expensive, so these must be waited out and retried
        rather than recovered.
        """
        low = str(exc or "").lower()
        return (
            "rejected by callee" in low
            or "-2147418111" in low
            or "call was rejected" in low
            or "server is busy" in low
            or "application is busy" in low
        )

    @staticmethod
    def _com_error_is_broken_proxy(exc: BaseException) -> bool:
        """True for genuinely dead COM proxies/servers that only a reconnect can fix."""
        msg = str(exc or "")
        low = msg.lower()
        return (
            "<unknown>." in low
            or "open.name" in low
            or ".name" in low and ("open" in low or "unknown" in low)
            or "rpc server is unavailable" in low
            or "server threw an exception" in low
            or "invalid class string" in low
            or "catastrophic failure" in low
        )

    def _settle_busy_com(self, exc: BaseException, *, wait_s: float = 0.35) -> bool:
        """Let a busy — but healthy — AutoCAD finish what it is doing.

        Returns True when ``exc`` was a busy signal, meaning the caller should retry
        rather than recover the session. Callers use this to keep the settling pause
        that a reconnect used to provide incidentally, without its cost.
        """
        if not self._com_error_is_busy(exc):
            return False
        try:
            self.quiesce_autocad(esc_count=2, force=True)
        except Exception:
            pass
        time.sleep(max(0.0, float(wait_s)))
        try:
            from survyai.perf import incr

            incr("autocad_busy_settled")
        except Exception:
            pass
        return True

    def _log_doc_warn_throttled(self, message: str) -> None:
        """Avoid flooding the CLI when Documents is wedged during a batch."""
        now = time.time()
        if (
            message == getattr(self, "_last_doc_warn_msg", "")
            and (now - float(getattr(self, "_last_doc_warn_ts", 0.0) or 0.0)) < 2.5
        ):
            return
        self._last_doc_warn_msg = message
        self._last_doc_warn_ts = now
        logger.warning(message)

    def recover_com_session(self, *, force: bool = False, reason: str = "") -> bool:
        """
        Drop stale COM references and re-attach to AutoCAD.

        Used when Documents/Open start returning ``<unknown>.Count`` / ``<unknown>.Open``.
        Rate-limited so a wedged session does not reconnect in a tight loop.
        """
        now = time.time()
        last = float(getattr(self, "_last_com_recover_ts", 0.0) or 0.0)
        if not force and (now - last) < 2.0 and self.is_connected:
            return True
        self._last_com_recover_ts = now
        if reason:
            logger.warning("Recovering AutoCAD COM session (%s)…", reason)
        else:
            logger.warning("Recovering AutoCAD COM session…")
        self.doc = None
        self._connected = False
        self.acad = None
        self._invalidate_handle_cache()
        self._quiesce_needed = True
        ok = self.connect()
        if ok:
            time.sleep(0.15)
            try:
                self.quiesce_autocad(esc_count=2, force=True)
            except Exception:
                pass
        return bool(ok)

    def _documents_count_safe(self) -> Optional[int]:
        """Return Documents.Count with COM retry; None if Documents is unusable."""
        if not self._connected or not self.acad:
            if not self.connect():
                return None
        try:
            return int(self._com_retry(lambda: self.acad.Documents.Count, attempts=4, base_sleep=0.15))
        except Exception as e:
            if not self._settle_busy_com(e) and self._com_error_is_broken_proxy(e):
                if self.recover_com_session(reason=f"Documents.Count: {e}"):
                    try:
                        return int(
                            self._com_retry(lambda: self.acad.Documents.Count, attempts=3, base_sleep=0.2)
                        )
                    except Exception:
                        return None
            return None

    def save_and_close_drawing(self, file_path: str, *, save: bool = True) -> Dict[str, Any]:
        """
        Save (optional) and close a specific drawing so batch CAD runs do not pile up tabs.

        Clears the workflow pin when it matches ``file_path``.
        """
        from pathlib import Path

        try:
            target = str(Path(file_path).resolve())
        except Exception:
            target = str(file_path)
        errors: List[str] = []
        if save:
            try:
                opened = self.open_drawing(target, read_only=False)
                if opened.get("success"):
                    try:
                        if bool(getattr(self.doc, "ReadOnly", False)):
                            errors.append("drawing is read-only; close without save")
                        else:
                            self.save_active_drawing()
                    except Exception as e:
                        errors.append(f"save failed: {e}")
                else:
                    errors.append(str(opened.get("error") or "could not activate for save"))
            except Exception as e:
                errors.append(f"activate/save failed: {e}")
        try:
            closed = self.close_drawing_if_open(target, save_changes=False)
            if not closed.get("success") and closed.get("error"):
                errors.append(str(closed.get("error")))
        except Exception as e:
            errors.append(f"close failed: {e}")
        try:
            if self._workflow_doc_path:
                try:
                    if str(Path(self._workflow_doc_path).resolve()) == str(Path(target).resolve()):
                        self.set_workflow_document(None)
                except Exception:
                    self.set_workflow_document(None)
        except Exception:
            pass
        self.doc = None
        time.sleep(0.35)
        # Probe Documents; recover once if the close left COM wedged.
        if self._documents_count_safe() is None:
            self.recover_com_session(reason="after save_and_close_drawing")
        return {"success": len(errors) == 0, "path": target, "errors": errors}

    def save_close_to_release_lock(self, file_path: str) -> Dict[str, Any]:
        """
        Save unsaved view/edits, close the drawing, and drop AutoCAD's file lock.

        Typical case: the user panned, zoomed, or edited the previous output and
        left it open. Closing without save often fails or leaves the DWG locked.
        """
        from pathlib import Path

        try:
            target = str(Path(file_path).resolve())
        except Exception:
            target = str(file_path)
        errors: List[str] = []
        try:
            self.quiesce_autocad(esc_count=4)
        except Exception as e:
            errors.append(f"quiesce: {e}")
        saved = False
        try:
            result = self.save_and_close_drawing(target, save=True)
            saved = bool(result.get("success"))
            for err in result.get("errors") or []:
                if err:
                    errors.append(str(err))
        except Exception as e:
            errors.append(f"save_and_close: {e}")
        closed = False
        try:
            still_open = False
            try:
                still_open = bool(self.is_drawing_open(target))
            except Exception:
                still_open = True
            if still_open:
                forced = self.close_drawing_if_open(target, save_changes=True)
                closed = bool(forced.get("closed"))
                if forced.get("error"):
                    errors.append(str(forced.get("error")))
            else:
                closed = True
        except Exception as e:
            errors.append(f"force close: {e}")
        time.sleep(0.5)
        return {
            "success": bool(closed),
            "saved": saved,
            "closed": closed,
            "path": target,
            "errors": errors,
        }

    def _com_retry(self, fn, attempts: int = 10, base_sleep: float = 0.2, *, deadline_ts: Optional[float] = None):
        """Retry COM calls rejected while AutoCAD is busy or the user is interacting."""
        try:
            import pywintypes  # type: ignore
        except Exception:
            pywintypes = None
        last = None
        for k in range(max(1, int(attempts))):
            if deadline_ts is not None and time.time() >= float(deadline_ts):
                break
            try:
                return fn()
            except Exception as ex:
                last = ex
                # Any COM failure means AutoCAD may be mid-command or modal, so the
                # next quiesce must actually send ESC instead of trusting the window.
                self._quiesce_needed = True
                retryable = False
                try:
                    if pywintypes is not None and isinstance(ex, pywintypes.com_error):
                        hr = int(ex.hresult) if hasattr(ex, "hresult") else None
                        if hr == -2147418111:
                            retryable = True
                except Exception:
                    pass
                try:
                    if isinstance(ex, AttributeError) and any(
                        s in str(ex) for s in (".Open", ".Count", ".Item", ".SendCommand", ".Name", ".Activate")
                    ):
                        retryable = True
                except Exception:
                    pass
                if self._com_error_is_busy(ex) or self._com_error_is_broken_proxy(ex):
                    retryable = True
                # Busy/stale-proxy errors resolve with waiting, so they keep the
                # full attempt budget (capped per-sleep so a wedged AutoCAD cannot
                # burn minutes). Unclassified errors are usually permanent
                # (bad path, invalid argument): retry twice quickly, then fail
                # instead of sleeping through the whole budget.
                if not retryable and k + 1 >= 3:
                    break
                remaining = None
                if deadline_ts is not None:
                    remaining = max(0.0, float(deadline_ts) - time.time())
                sleep_s = min(base_sleep * (k + 1), 0.8) if retryable else min(base_sleep, 0.1)
                if remaining is not None:
                    sleep_s = min(sleep_s, max(0.05, remaining))
                    if remaining <= 0.0:
                        break
                time.sleep(sleep_s)
                continue
        raise last if last is not None else Exception("COM retry failed")

    def _safe_doc_name(self, doc: Any, *, deadline_ts: Optional[float] = None) -> str:
        """Read ``doc.Name`` with COM retry; classify Open.Name failures as stale proxies."""
        try:
            return str(
                self._com_retry(
                    lambda: getattr(doc, "Name"),
                    attempts=6,
                    base_sleep=0.12,
                    deadline_ts=deadline_ts,
                )
                or ""
            )
        except Exception as exc:
            if self._com_error_is_broken_proxy(exc):
                raise
            raise

    def _activate_document_by_path(
        self,
        file_path: Path,
        *,
        deadline_ts: Optional[float] = None,
    ) -> bool:
        """Activate an already-open drawing by full path / basename without Documents.Open."""
        want = file_path.resolve()
        want_name = want.name.lower()
        try:
            n = self._documents_count_safe()
            if not n:
                return False
            for i in range(int(n)):
                if deadline_ts is not None and time.time() >= float(deadline_ts):
                    return False
                try:
                    doc = self._com_retry(
                        lambda idx=i: self.acad.Documents.Item(idx),
                        attempts=4,
                        base_sleep=0.1,
                        deadline_ts=deadline_ts,
                    )
                    full = ""
                    try:
                        full = str(
                            self._com_retry(
                                lambda d=doc: getattr(d, "FullName", "") or "",
                                attempts=4,
                                base_sleep=0.1,
                                deadline_ts=deadline_ts,
                            )
                        )
                    except Exception:
                        full = ""
                    name = ""
                    try:
                        name = self._safe_doc_name(doc, deadline_ts=deadline_ts).lower()
                    except Exception:
                        name = Path(full).name.lower() if full else ""
                    match = False
                    if full:
                        try:
                            match = Path(full).resolve() == want
                        except Exception:
                            match = False
                    if not match and name == want_name:
                        match = True
                    if match:
                        self._com_retry(
                            lambda d=doc: d.Activate(),
                            attempts=6,
                            base_sleep=0.12,
                            deadline_ts=deadline_ts,
                        )
                        self.doc = doc
                        return True
                except Exception:
                    continue
        except Exception as exc:
            logger.debug("activate_document_by_path failed: %s", exc)
        return False

    def quiesce_autocad(self, *, esc_count: int = 2, force: bool = False) -> None:
        """Best-effort cancel of in-progress AutoCAD commands before automation.

        ``SendCommand`` is synchronous — it blocks until AutoCAD processes the ESC —
        so this is one of the most expensive calls available. Because it runs from
        ``ensure_workflow_document``, an unguarded quiesce fired before *every* entity
        write, costing hundreds of blocking round-trips per plot.

        Only a pending command needs cancelling, and SurvyAI's own property/method
        writes never leave one. So a quiesce is skipped when one succeeded recently
        and nothing has since suggested AutoCAD is busy. The window still re-arms on
        a timer so an operator typing in AutoCAD mid-plot is cancelled promptly, and
        any COM busy/rejected error re-arms it immediately.
        """
        if not self._connected or not self.acad:
            return
        now = time.time()
        if (
            not force
            and not self._quiesce_needed
            and (now - self._last_quiesce_ts) < self._QUIESCE_REARM_S
        ):
            try:
                from survyai.perf import incr

                incr("autocad_quiesce_skipped")
            except Exception:
                pass
            return
        sent = False
        for _ in range(max(1, int(esc_count))):
            try:
                doc = self.doc or getattr(self.acad, "ActiveDocument", None)
                if doc is not None:
                    self._com_retry(lambda: doc.SendCommand("\x1b"), attempts=3, base_sleep=0.08)
                    sent = True
            except Exception:
                pass
            time.sleep(0.05)
        self._last_quiesce_ts = time.time()
        # Only an ESC that actually reached AutoCAD may open the trust window.
        if sent:
            self._quiesce_needed = False
        try:
            from survyai.perf import incr

            incr("autocad_quiesce_sent")
        except Exception:
            pass

    def ensure_workflow_document(self, *, light: bool = False) -> bool:
        """
        Re-activate the pinned workflow drawing and quiesce stray commands.

        ``light=True`` (batch checkpoints): verify the active ModelSpace only —
        do not call ``open_drawing`` (avoids spam when Documents is wedged).

        Returns True when a usable document is ready for COM edits.
        """
        if not self._connected or not self.acad:
            if not self.connect():
                return False
        if not light:
            self.quiesce_autocad()
        if self._workflow_doc_path:
            try:
                want = Path(self._workflow_doc_path).resolve()
                cur = None
                try:
                    cur_raw = getattr(self.doc, "FullName", None) if self.doc else None
                    if cur_raw:
                        cur = Path(str(cur_raw)).resolve()
                except Exception:
                    cur = None
                if cur == want:
                    # Fast path: already on the pinned drawing and ModelSpace responds.
                    try:
                        if self.doc is not None:
                            _ = self._com_retry(lambda: self.doc.ModelSpace.Count, attempts=3, base_sleep=0.1)
                            return True
                    except Exception as e:
                        if not self._settle_busy_com(e) and self._com_error_is_broken_proxy(e):
                            self.recover_com_session(reason=str(e))
                        self.doc = None
                elif light:
                    # Batch checkpoints: try activate-by-handle only, never Open.
                    try:
                        n = self._documents_count_safe()
                        if n:
                            for i in range(int(n)):
                                try:
                                    doc = self.acad.Documents.Item(i)
                                    dp = Path(str(getattr(doc, "FullName", "") or "")).resolve()
                                    if dp == want:
                                        self._com_retry(lambda d=doc: d.Activate(), attempts=4)
                                        self.doc = doc
                                        return True
                                except Exception:
                                    continue
                    except Exception:
                        pass
                    return self._ensure_active_document()
                else:
                    # Prefer activate-by-full-path; only Open when the drawing is not loaded.
                    if self._activate_document_by_path(want):
                        self._workflow_open_fail_until = 0.0
                    else:
                        # Circuit breaker: do not hammer Open on a known-bad path.
                        if time.time() < float(getattr(self, "_workflow_open_fail_until", 0.0) or 0.0):
                            return self._ensure_active_document()
                        opened = self.open_drawing(str(want), read_only=False)
                        if not opened.get("success"):
                            self._workflow_open_fail_until = time.time() + 4.0
                            self._log_doc_warn_throttled(
                                f"ensure_workflow_document: could not activate {want}: {opened.get('error')}"
                            )
                            if self._com_error_is_broken_proxy(
                                Exception(str(opened.get("error") or ""))
                            ):
                                self.recover_com_session(reason="workflow open failed")
                        else:
                            self._workflow_open_fail_until = 0.0
                if not light:
                    self.quiesce_autocad()
            except Exception as e:
                logger.debug("ensure_workflow_document: %s", e)
        return self._ensure_active_document()

    # ==========================================================================
    # TEXT EXTRACTION
    # ==========================================================================
    
    def _ensure_active_document(self) -> bool:
        """
        Ensure we have a valid, active document to work with.
        
        This method verifies the document reference is valid and attempts
        to recover if it's stale. It also verifies the document is actually
        accessible by testing ModelSpace access. If connection is lost, it
        attempts to reconnect.
        
        Returns:
            bool: True if we have a valid document, False otherwise
        """
        # First, ensure we're connected to AutoCAD
        if not self._connected or not self.acad:
            logger.debug("Not connected, attempting to reconnect...")
            if not self.connect():
                self._log_doc_warn_throttled("Could not connect to AutoCAD")
                return False

        # Prefer the pinned workflow drawing when the user switched tabs mid-pipeline.
        if self._workflow_doc_path:
            try:
                want = Path(self._workflow_doc_path).resolve()
                active_ok = False
                try:
                    if self.doc:
                        ap = Path(str(getattr(self.doc, "FullName", "") or "")).resolve()
                        active_ok = ap == want
                except Exception:
                    active_ok = False
                if not active_ok:
                    n = self._documents_count_safe()
                    if n:
                        for i in range(int(n)):
                            try:
                                doc = self.acad.Documents.Item(i)
                                dp = Path(str(getattr(doc, "FullName", "") or "")).resolve()
                                if dp == want:
                                    self._com_retry(lambda d=doc: d.Activate(), attempts=6)
                                    self.doc = doc
                                    time.sleep(0.12)
                                    break
                            except Exception:
                                continue
            except Exception as e:
                logger.debug("Workflow document re-activation skipped: %s", e)
        
        # Verify connection is still valid
        try:
            _ = self._com_retry(lambda: self.acad.Name, attempts=3, base_sleep=0.1)
        except Exception:
            self._log_doc_warn_throttled("Connection lost, attempting to reconnect...")
            if not self.recover_com_session(force=True, reason="ActiveDocument probe"):
                return False
        
        # Check if current doc reference is valid and accessible
        try:
            if self.doc:
                # Test basic access
                _ = self._com_retry(lambda: self.doc.Name, attempts=3, base_sleep=0.1)
                # Test ModelSpace access - this is the real test
                _ = self._com_retry(lambda: self.doc.ModelSpace.Count, attempts=3, base_sleep=0.1)
                return True
        except Exception as e:
            logger.debug(f"Current doc reference is stale or inaccessible: {e}")
            if not self._settle_busy_com(e) and self._com_error_is_broken_proxy(e):
                self.recover_com_session(reason=str(e))
            self.doc = None  # Clear stale reference
        
        # Try to get the active document from AutoCAD
        try:
            # Check if there are any open documents
            doc_count = self._documents_count_safe()
            if doc_count is None:
                self._log_doc_warn_throttled(
                    "Could not access active document: Documents.Count unavailable"
                )
                return False
            if doc_count == 0:
                self._log_doc_warn_throttled("No documents are open in AutoCAD")
                return False
            
            # Get the active document
            try:
                self.doc = self._com_retry(lambda: self.acad.ActiveDocument, attempts=4)
            except Exception:
                # If ActiveDocument fails, try to get the first document
                if doc_count > 0:
                    self.doc = self._com_retry(lambda: self.acad.Documents.Item(0), attempts=4)
                    self._com_retry(lambda: self.doc.Activate(), attempts=4)
                    time.sleep(0.3)
                    self.doc = self._com_retry(lambda: self.acad.ActiveDocument, attempts=4)
            
            # Verify it's accessible
            _ = self._com_retry(lambda: self.doc.Name, attempts=3)
            _ = self._com_retry(lambda: self.doc.ModelSpace.Count, attempts=3)
            
            logger.info(f"Document ready: {self.doc.Name}")
            return True
        except Exception as e:
            self._log_doc_warn_throttled(f"Could not access active document: {e}")
            if not self._settle_busy_com(e) and self._com_error_is_broken_proxy(e):
                if self.recover_com_session(reason=str(e)):
                    try:
                        doc_count = self._documents_count_safe() or 0
                        if doc_count > 0:
                            self.doc = self.acad.Documents.Item(0)
                            self.doc.Activate()
                            time.sleep(0.25)
                            _ = self.doc.ModelSpace.Count
                            return True
                    except Exception:
                        pass
            # Try to find any open document as a fallback
            try:
                n = self._documents_count_safe() or 0
                if n > 0:
                    # Try to use the first document
                    self.doc = self.acad.Documents.Item(0)
                    self.doc.Activate()
                    time.sleep(0.3)
                    self.doc = self.acad.ActiveDocument
                    _ = self.doc.Name
                    _ = self.doc.ModelSpace.Count
                    logger.info(f"Activated document: {self.doc.Name}")
                    return True
            except Exception as fallback_error:
                self._log_doc_warn_throttled(
                    f"Fallback document activation failed: {fallback_error}"
                )
            
            return False
    
    def get_all_text(self) -> Dict[str, Any]:
        """
        Extract all text entities from the drawing.
        
        This method finds all TEXT and MTEXT entities and returns their
        content along with metadata (layer, color, position).
        
        Returns:
            Dict containing:
            - success: Boolean
            - text_count: Number of text entities found
            - texts: List of text entity dictionaries, each with:
              - type: "TEXT" or "MTEXT"
              - content: The text string
              - layer: Layer name
              - color: Color name
              - insertion_point: {x, y} coordinates
              
        Use Cases:
            - Finding property owner names
            - Extracting survey titles
            - Reading annotations and notes
        """
        if not self._ensure_active_document():
            return {"success": False, "error": "No active document. Please open a drawing first using autocad_open_drawing."}
        
        try:
            texts = []
            modelspace = self.doc.ModelSpace
            
            # Iterate through all entities in model space
            for i in range(modelspace.Count):
                entity = modelspace.Item(i)
                obj_name = entity.ObjectName
                
                # Check if it's a text entity
                if "Text" in obj_name or "MText" in obj_name:
                    text_data = {
                        "type": "MTEXT" if "MText" in obj_name else "TEXT",
                        "content": entity.TextString,
                        "layer": entity.Layer,
                        "color": self._get_color_name(entity.Color),
                    }
                    
                    # Get insertion point (where text is placed)
                    try:
                        text_data["insertion_point"] = {
                            "x": entity.InsertionPoint[0],
                            "y": entity.InsertionPoint[1],
                        }
                    except Exception:
                        pass
                    
                    # Only include non-empty text
                    if text_data["content"].strip():
                        texts.append(text_data)
            
            return {
                "success": True,
                "text_count": len(texts),
                "texts": texts
            }
            
        except Exception as e:
            logger.error(f"Failed to extract text: {e}")
            return {"success": False, "error": str(e)}
    
    def search_text(self, pattern: str, case_sensitive: bool = False) -> Dict[str, Any]:
        """
        Search for text matching a pattern.
        
        This method finds all text entities whose content matches the
        specified pattern. Supports regular expressions for flexible matching.
        
        Args:
            pattern: Text pattern to search for
                    - Simple string: "property of"
                    - Regex pattern: "property of \\w+"
            case_sensitive: If True, match case exactly
            
        Returns:
            Dict containing:
            - success: Boolean
            - pattern: The search pattern used
            - matches_found: Number of matches
            - matches: List of matching text entities
            
        Examples:
            >>> # Find owner name
            >>> result = acad.search_text("property of")
            >>> for match in result["matches"]:
            ...     print(match["content"])
            
            >>> # Find survey numbers (regex)
            >>> result = acad.search_text(r"SN[0-9]+")
        """
        # First, get all text from the drawing
        result = self.get_all_text()
        if not result.get("success"):
            return result
        
        import re
        
        # Set up regex flags
        flags = 0 if case_sensitive else re.IGNORECASE
        
        matches = []
        for text in result.get("texts", []):
            content = text.get("content", "")
            
            try:
                # Try regex search
                if re.search(pattern, content, flags):
                    matches.append(text)
            except re.error:
                # Invalid regex - fall back to simple substring search
                search_content = content if case_sensitive else content.lower()
                search_pattern = pattern if case_sensitive else pattern.lower()
                if search_pattern in search_content:
                    matches.append(text)
        
        return {
            "success": True,
            "pattern": pattern,
            "matches_found": len(matches),
            "matches": matches
        }
    
    # ==========================================================================
    # ATOMIC AI-DRIVEN METHODS (Return raw data for agent reasoning)
    # ==========================================================================
    
    def get_all_entities(self) -> Dict[str, Any]:
        """
        Get ALL entities from the drawing with complete properties.
        
        This is an atomic method designed for AI reasoning. It returns ALL entities
        with their full properties (type, layer, color, coordinates, etc.) without
        any filtering logic. The AI agent can then reason about which entities
        match specific criteria.
        
        Returns:
            Dict containing:
            - success: Boolean
            - entity_count: Total number of entities
            - entities: List of entity dictionaries, each with:
              - handle: Unique entity identifier
              - type: Entity type (LINE, LWPOLYLINE, CIRCLE, TEXT, etc.)
              - layer: Layer name
              - color: Color name (e.g., "red", "blue", "bylayer")
              - color_code: AutoCAD Color Index (1-256)
              - coordinates: Entity coordinates (varies by type)
              - properties: Type-specific properties (area, length, closed, etc.)
              - text_content: For TEXT/MTEXT entities
              
        AI Usage Pattern:
            The agent should call this method to get all entities, then reason
            about which ones match the criteria (e.g., "red", "closed", "polyline")
            based on the returned properties.
        """
        if not self._ensure_active_document():
            return {"success": False, "error": "No active document. Please open a drawing first using autocad_open_drawing."}
        
        try:
            entities = []
            modelspace = self.doc.ModelSpace
            
            for i in range(modelspace.Count):
                try:
                    entity = modelspace.Item(i)
                    obj_name = entity.ObjectName
                    ent_type = ENTITY_TYPES.get(obj_name, obj_name)
                    
                    # Extract complete entity data (already includes handle)
                    entity_data = self._extract_entity_data(entity, ent_type)
                    
                    # Ensure color information is complete
                    entity_data["color_code"] = entity.Color
                    if "color" not in entity_data:
                        entity_data["color"] = self._get_color_name(entity.Color)
                    
                    entities.append(entity_data)
                except Exception as e:
                    logger.debug(f"Error extracting entity {i}: {e}")
                    continue
            
            return {
                "success": True,
                "entity_count": len(entities),
                "entities": entities,
                "note": "All entities returned with complete properties. Use agent reasoning to filter by type, color, layer, or other properties."
            }
            
        except Exception as e:
            logger.error(f"Failed to get all entities: {e}")
            return {"success": False, "error": str(e)}

    def get_entities_on_layers(self, layers: List[str]) -> Dict[str, Any]:
        """
        Return entities whose layer is in ``layers`` (case-insensitive).

        Prefer this over ``get_all_entities`` for cadastral apply alignment so
        we do not walk/extract the entire ModelSpace twice per plan.
        """
        if not self._ensure_active_document():
            return {
                "success": False,
                "error": "No active document. Please open a drawing first using autocad_open_drawing.",
            }
        want = {str(L or "").strip().upper() for L in (layers or []) if str(L or "").strip()}
        if not want:
            return {"success": True, "entity_count": 0, "entities": []}
        try:
            entities: List[Dict[str, Any]] = []
            modelspace = self.doc.ModelSpace
            for i in range(modelspace.Count):
                try:
                    entity = modelspace.Item(i)
                    lyr = str(getattr(entity, "Layer", "") or "").upper()
                    if lyr not in want:
                        continue
                    obj_name = entity.ObjectName
                    ent_type = ENTITY_TYPES.get(obj_name, obj_name)
                    entity_data = self._extract_entity_data(entity, ent_type)
                    entity_data["color_code"] = entity.Color
                    if "color" not in entity_data:
                        entity_data["color"] = self._get_color_name(entity.Color)
                    entities.append(entity_data)
                except Exception:
                    continue
            return {"success": True, "entity_count": len(entities), "entities": entities}
        except Exception as e:
            logger.error(f"Failed to get entities on layers: {e}")
            return {"success": False, "error": str(e)}
    
    def get_entity_by_handle(self, handle: str) -> Dict[str, Any]:
        """
        Get a specific entity by its handle.
        
        Handles are unique identifiers for entities in AutoCAD. Use this to
        get detailed information about a specific entity that was identified
        from get_all_entities().
        
        Args:
            handle: Entity handle (unique identifier)
            
        Returns:
            Dict containing:
            - success: Boolean
            - entity: Entity data dictionary with complete properties
            - error: Error message if not found
        """
        if not self._ensure_active_document():
            return {"success": False, "error": "No active document. Please open a drawing first using autocad_open_drawing."}
        
        try:
            modelspace = self.doc.ModelSpace
            
            for i in range(modelspace.Count):
                entity = modelspace.Item(i)
                if entity.Handle == handle:
                    obj_name = entity.ObjectName
                    ent_type = ENTITY_TYPES.get(obj_name, obj_name)
                    entity_data = self._extract_entity_data(entity, ent_type)
                    entity_data["handle"] = handle
                    entity_data["color_code"] = entity.Color
                    entity_data["color"] = self._get_color_name(entity.Color)
                    
                    return {
                        "success": True,
                        "entity": entity_data
                    }
            
            return {"success": False, "error": f"Entity with handle {handle} not found"}
            
        except Exception as e:
            logger.error(f"Failed to get entity by handle: {e}")
            return {"success": False, "error": str(e)}
    
    def calculate_entity_area(self, handle: str) -> Dict[str, Any]:
        """
        Calculate the area of a specific entity by handle.
        
        This is an atomic method for calculating area of a single entity.
        The agent should first identify which entities are closed shapes
        (using get_all_entities), then call this method for each one.
        
        Args:
            handle: Entity handle (unique identifier)
            
        Returns:
            Dict containing:
            - success: Boolean
            - handle: Entity handle
            - area_sq_units: Area in drawing units
            - area_conversions: Area in various units (sq meters, hectares, etc.)
            - entity_type: Type of entity
            - error: Error message if calculation failed
        """
        if not self._ensure_active_document():
            return {"success": False, "error": "No active document. Please open a drawing first using autocad_open_drawing."}
        
        try:
            modelspace = self.doc.ModelSpace
            entity = None
            
            # Find entity by handle
            for i in range(modelspace.Count):
                if modelspace.Item(i).Handle == handle:
                    entity = modelspace.Item(i)
                    break
            
            if not entity:
                return {"success": False, "error": f"Entity with handle {handle} not found"}
            
            obj_name = entity.ObjectName
            ent_type = ENTITY_TYPES.get(obj_name, obj_name)
            
            # Check if entity can have area
            if not any(t in obj_name for t in ["Polyline", "Circle", "Hatch", "Region"]):
                return {
                    "success": False,
                    "error": f"Entity type {ent_type} does not have area property"
                }
            
            # Check if polyline is closed
            if "Polyline" in obj_name:
                try:
                    if not entity.Closed:
                        return {
                            "success": False,
                            "error": "Polyline is not closed. Only closed shapes have area."
                        }
                except Exception:
                    pass
            
            # Get area
            try:
                area = entity.Area
                if area <= 0:
                    return {
                        "success": False,
                        "error": "Entity area is zero or negative"
                    }
                
                units = self._get_units()
                conversions = self._calculate_area_conversions(area, units)
                
                return {
                    "success": True,
                    "handle": handle,
                    "entity_type": ent_type,
                    "area_sq_units": area,
                    "area_conversions": conversions,
                    "drawing_units": units
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Could not calculate area: {e}"
                }
                
        except Exception as e:
            logger.error(f"Failed to calculate entity area: {e}")
            return {"success": False, "error": str(e)}
    
    def get_entities_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all entities for quick analysis.
        
        This returns a lightweight summary (counts, types, colors, layers)
        that the agent can use to reason about what's in the drawing before
        calling get_all_entities() for detailed extraction.
        
        Returns:
            Dict containing:
            - success: Boolean
            - total_entities: Total count
            - by_type: Count of each entity type
            - by_color: Count of entities by color
            - by_layer: Count of entities by layer
            - color_codes: Mapping of color codes to names
        """
        if not self._ensure_active_document():
            return {"success": False, "error": "No active document. Please open a drawing first using autocad_open_drawing."}
        
        try:
            by_type = {}
            by_color = {}
            by_layer = {}
            modelspace = self.doc.ModelSpace
            
            for i in range(modelspace.Count):
                try:
                    entity = modelspace.Item(i)
                    obj_name = entity.ObjectName
                    ent_type = ENTITY_TYPES.get(obj_name, obj_name)
                    color_name = self._get_color_name(entity.Color)
                    layer = entity.Layer
                    
                    by_type[ent_type] = by_type.get(ent_type, 0) + 1
                    by_color[color_name] = by_color.get(color_name, 0) + 1
                    by_layer[layer] = by_layer.get(layer, 0) + 1
                except Exception:
                    continue
            
            return {
                "success": True,
                "total_entities": modelspace.Count,
                "by_type": by_type,
                "by_color": by_color,
                "by_layer": by_layer,
                "color_codes": ACI_COLORS,
                "note": "Use this summary to understand the drawing structure before detailed extraction."
            }
            
        except Exception as e:
            logger.error(f"Failed to get entities summary: {e}")
            return {"success": False, "error": str(e)}
    
    # ==========================================================================
    # HIGH-LEVEL METHODS (Backward compatibility - use atomic methods internally)
    # ==========================================================================
    
    def get_entities_by_type(
        self, 
        entity_type: Optional[str] = None,
        layer: Optional[str] = None,
        color: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get entities from the drawing with optional filters.
        
        [BACKWARD COMPATIBILITY METHOD]
        This method uses the atomic get_all_entities() internally and applies
        filters. For AI-driven extraction, prefer using get_all_entities() directly
        and let the agent reason about filtering.
        
        Args:
            entity_type: Filter by type (e.g., "LINE", "POLYLINE", "CIRCLE")
            layer: Filter by layer name (e.g., "Boundaries", "Survey")
            color: Filter by color name (e.g., "red", "blue")
            
        Returns:
            Dict containing:
            - success: Boolean
            - entity_count: Number of entities found
            - filters: The filters that were applied
            - entities: List of entity data dictionaries
            
        Entity Types:
            - LINE: Simple line segment
            - LWPOLYLINE: Lightweight polyline (most common for boundaries)
            - POLYLINE: 2D polyline
            - CIRCLE: Circle
            - ARC: Arc
            - TEXT/MTEXT: Text entities
            - HATCH: Hatched areas
            
        Example:
            >>> # Get all red polylines
            >>> result = acad.get_entities_by_type(
            ...     entity_type="LWPOLYLINE",
            ...     color="red"
            ... )
        """
        # Use atomic method and apply filters
        result = self.get_all_entities()
        if not result.get("success"):
            return result
        
        entities = result.get("entities", [])
        filtered_entities = []
        color_lower = color.lower() if color else None
        
        for entity in entities:
            # Apply type filter
            if entity_type and entity.get("type", "").upper() != entity_type.upper():
                continue
            
            # Apply layer filter
            if layer and entity.get("layer", "").lower() != layer.lower():
                continue
            
            # Apply color filter
            if color_lower:
                entity_color = entity.get("color", "").lower()
                if not entity_color or color_lower not in entity_color:
                    continue
            
            filtered_entities.append(entity)
        
        return {
            "success": True,
            "entity_count": len(filtered_entities),
            "filters": {"type": entity_type, "layer": layer, "color": color},
            "entities": filtered_entities
        }
    
    # ==========================================================================
    # AREA CALCULATION
    # ==========================================================================
    
    def calculate_area(
        self,
        entity_handle: Optional[str] = None,
        layer: Optional[str] = None,
        color: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate area of closed entities in the drawing.
        
        [BACKWARD COMPATIBILITY METHOD]
        This method uses atomic methods internally. For AI-driven extraction,
        prefer using get_all_entities() to identify closed shapes, then
        calculate_entity_area() for each one.
        
        Args:
            entity_handle: Specific entity handle to measure (optional)
            layer: Filter to entities on this layer
            color: Filter to entities of this color (e.g., "red" for boundaries)
            
        Returns:
            Dict containing:
            - success: Boolean
            - drawing_units: The units used in the drawing
            - shapes_found: Number of closed shapes found
            - total_area_sq_units: Total area in drawing units
            - area_conversions: Area in various units (sq meters, hectares, etc.)
            - individual_areas: List of individual shape areas
            
        Survey Usage:
            In surveying, boundaries are often drawn as closed polylines
            colored red ("verged in red"). To get the land area:
            
            >>> result = acad.calculate_area(color="red")
            >>> print(f"Area: {result['area_conversions']['hectares']:.4f} ha")
        """
        # If specific handle provided, use atomic method
        if entity_handle:
            result = self.calculate_entity_area(entity_handle)
            if result.get("success"):
                return {
                    "success": True,
                    "drawing_units": result.get("drawing_units"),
                    "shapes_found": 1,
                    "total_area_sq_units": result.get("area_sq_units", 0),
                    "area_conversions": result.get("area_conversions", {}),
                    "individual_areas": [{
                        "handle": entity_handle,
                        "entity_type": result.get("entity_type"),
                        "area_sq_units": result.get("area_sq_units", 0)
                    }]
                }
            return result
        
        # Otherwise, use get_all_entities and filter
        result = self.get_all_entities()
        if not result.get("success"):
            return result
        
        entities = result.get("entities", [])
        color_lower = color.lower() if color else None
        areas = []
        total_area = 0.0
        
        # Identify closed shapes that match criteria
        for entity in entities:
            ent_type = entity.get("type", "")
            
            # Only process entities that can have area
            if ent_type not in ["LWPOLYLINE", "POLYLINE", "CIRCLE", "HATCH"]:
                continue
            
            # Apply layer filter
            if layer and entity.get("layer", "").lower() != layer.lower():
                continue
            
            # Apply color filter
            if color_lower:
                entity_color = entity.get("color", "").lower()
                if not entity_color or color_lower not in entity_color:
                    continue
            
            # Check if closed (for polylines)
            if ent_type in ["LWPOLYLINE", "POLYLINE"]:
                if not entity.get("closed", False):
                    continue
            
            # Get area from entity properties
            area = entity.get("area")
            if area and area > 0:
                areas.append({
                    "handle": entity.get("handle"),
                    "type": ent_type,
                    "layer": entity.get("layer"),
                    "color": entity.get("color"),
                    "area_sq_units": area,
                })
                total_area += area
        
        # Get drawing units for conversions
        units = self._get_units()
        conversions = self._calculate_area_conversions(total_area, units)
        
        return {
            "success": True,
            "drawing_units": units,
            "shapes_found": len(areas),
            "total_area_sq_units": total_area,
            "area_conversions": conversions,
            "individual_areas": areas[:20],  # Limit to first 20
        }
    
    # ==========================================================================
    # DRAWING INFORMATION
    # ==========================================================================
    
    def get_drawing_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the current drawing.
        
        Returns:
            Dict containing:
            - success: Boolean
            - name: Drawing filename
            - path: Full file path
            - units: Drawing units (Meters, Feet, etc.)
            - layers: List of all layer names
            - entity_counts: Dict of entity type counts
        """
        if not self._ensure_active_document():
            return {"success": False, "error": "No active document. Please open a drawing first using autocad_open_drawing."}
        
        try:
            return {
                "success": True,
                "name": self.doc.Name,
                "path": self.doc.FullName,
                "units": self._get_units(),
                "layers": self._get_layers(),
                "entity_counts": self._count_entities_by_type(),
            }
        except Exception as e:
            logger.error(f"Failed to get drawing info: {e}")
            return {"success": False, "error": str(e)}
    
    # ==========================================================================
    # HELPER METHODS (Private)
    # ==========================================================================
    
    def _get_units(self) -> str:
        """
        Get the drawing units as a human-readable string.
        
        AutoCAD stores units as a numeric code in the INSUNITS system variable.
        This method converts that code to a readable name.
        
        Returns:
            str: Unit name (e.g., "Meters", "Feet", "Inches")
        """
        if not self.doc:
            return "unknown"
            
        try:
            # INSUNITS system variable stores the unit code
            unit_code = self.doc.GetVariable("INSUNITS")
            
            # Map codes to names
            unit_names = {
                0: "Unitless",
                1: "Inches",
                2: "Feet",
                3: "Miles",
                4: "Millimeters",
                5: "Centimeters",
                6: "Meters",
                7: "Kilometers",
                8: "Microinches",
                9: "Mils",
                10: "Yards",
                11: "Angstroms",
                12: "Nanometers",
                13: "Microns",
                14: "Decimeters",
                15: "Decameters",
            }
            
            return unit_names.get(unit_code, f"Unit_{unit_code}")
            
        except Exception:
            return "unknown"
    
    def _get_layers(self) -> List[str]:
        """
        Get a list of all layer names in the drawing.
        
        Returns:
            List[str]: Layer names
        """
        if not self.doc:
            return []
            
        try:
            layers = []
            for layer in self.doc.Layers:
                layers.append(layer.Name)
            return layers
        except Exception:
            return []
    
    def _count_entities(self) -> int:
        """
        Count total number of entities in model space.
        
        Returns:
            int: Entity count
        """
        if not self.doc:
            return 0
            
        try:
            return self.doc.ModelSpace.Count
        except Exception:
            return 0
    
    def _count_entities_by_type(self) -> Dict[str, int]:
        """
        Count entities grouped by type.
        
        Returns:
            Dict[str, int]: Mapping of entity type to count
        """
        if not self.doc:
            return {}
            
        try:
            counts = {}
            modelspace = self.doc.ModelSpace
            
            for i in range(modelspace.Count):
                entity = modelspace.Item(i)
                obj_name = entity.ObjectName
                ent_type = ENTITY_TYPES.get(obj_name, obj_name)
                counts[ent_type] = counts.get(ent_type, 0) + 1
                
            return counts
            
        except Exception:
            return {}
    
    def _get_color_name(self, color_code: int) -> Optional[str]:
        """
        Convert AutoCAD color index to a color name.
        
        Args:
            color_code: AutoCAD Color Index (ACI) number
            
        Returns:
            str: Color name or "aci_N" for custom colors
        """
        if color_code == 256:
            return "bylayer"
        if color_code == 0:
            return "byblock"
        return ACI_COLORS.get(color_code, f"aci_{color_code}")
    
    def _extract_entity_data(self, entity, ent_type: str) -> Dict[str, Any]:
        """
        Extract detailed data from an entity.
        
        This method reads various properties based on entity type:
        - Lines: start point, end point, length
        - Polylines: all vertices, closed status, area
        - Circles: center, radius, area
        - Text: content, insertion point
        
        Args:
            entity: AutoCAD entity COM object
            ent_type: Entity type string
            
        Returns:
            Dict with entity properties
        """
        data = {
            "type": ent_type,
            "layer": entity.Layer,
            "color": self._get_color_name(entity.Color),
            "handle": entity.Handle,
        }
        
        try:
            # ------------------------------------------------------------------
            # LINE entities
            # ------------------------------------------------------------------
            if ent_type in ["LINE"]:
                data["start"] = {
                    "x": entity.StartPoint[0], 
                    "y": entity.StartPoint[1]
                }
                data["end"] = {
                    "x": entity.EndPoint[0], 
                    "y": entity.EndPoint[1]
                }
                data["length"] = entity.Length
            
            # ------------------------------------------------------------------
            # POLYLINE entities
            # ------------------------------------------------------------------
            elif ent_type in ["LWPOLYLINE", "POLYLINE", "3DPOLYLINE"]:
                # Get coordinates (flat array: [x1,y1,x2,y2,...])
                coords = list(entity.Coordinates)
                points = []
                for i in range(0, len(coords), 2):
                    if i + 1 < len(coords):
                        points.append({"x": coords[i], "y": coords[i+1]})
                        
                data["points"] = points
                data["vertex_count"] = len(points)
                
                try:
                    data["closed"] = entity.Closed
                    if entity.Closed:
                        data["area"] = entity.Area
                    data["length"] = entity.Length
                except Exception:
                    pass
            
            # ------------------------------------------------------------------
            # CIRCLE entities
            # ------------------------------------------------------------------
            elif ent_type == "CIRCLE":
                data["center"] = {
                    "x": entity.Center[0], 
                    "y": entity.Center[1]
                }
                data["radius"] = entity.Radius
                data["area"] = entity.Area
                data["circumference"] = entity.Circumference
            
            # ------------------------------------------------------------------
            # TEXT entities
            # ------------------------------------------------------------------
            elif ent_type in ["TEXT", "MTEXT"]:
                data["content"] = entity.TextString
                try:
                    data["insertion_point"] = {
                        "x": entity.InsertionPoint[0],
                        "y": entity.InsertionPoint[1]
                    }
                except Exception:
                    pass

            # ------------------------------------------------------------------
            # INSERT (block reference) entities
            # ------------------------------------------------------------------
            elif ent_type == "INSERT":
                # Block name (best-effort; varies by AutoCAD flavor)
                block_name = None
                for attr in ("EffectiveName", "Name"):
                    try:
                        v = getattr(entity, attr, None)
                        if v:
                            block_name = str(v)
                            break
                    except Exception:
                        continue
                if block_name:
                    data["block_name"] = block_name

                try:
                    data["insertion_point"] = {
                        "x": float(entity.InsertionPoint[0]),
                        "y": float(entity.InsertionPoint[1]),
                    }
                except Exception:
                    pass

                for attr in ("Rotation", "XScaleFactor", "YScaleFactor", "ZScaleFactor"):
                    try:
                        v = getattr(entity, attr, None)
                        if v is not None:
                            data[attr.lower()] = float(v)
                    except Exception:
                        continue

            # ------------------------------------------------------------------
            # TABLE entities (AutoCAD tables)
            # ------------------------------------------------------------------
            elif ent_type in ["TABLE", "AcDbTable"]:
                try:
                    data["rows"] = int(entity.Rows)
                    data["cols"] = int(entity.Columns)
                except Exception:
                    pass
                # Table insertion point varies; try common properties
                for attr in ("InsertionPoint", "Position"):
                    try:
                        pt = getattr(entity, attr, None)
                        if pt is not None:
                            data["insertion_point"] = {"x": float(pt[0]), "y": float(pt[1])}
                            break
                    except Exception:
                        continue
            
            # ------------------------------------------------------------------
            # HATCH entities
            # ------------------------------------------------------------------
            elif ent_type == "HATCH":
                try:
                    data["area"] = entity.Area
                    data["pattern_name"] = entity.PatternName
                except Exception:
                    pass
        
        except Exception as e:
            data["extraction_error"] = str(e)
        
        return data

    # ==========================================================================
    # TABLE + BLOCK UTILITIES (For template-driven CAD automation)
    # ==========================================================================

    def list_tables(self, layer: Optional[str] = None) -> Dict[str, Any]:
        """
        List AutoCAD TABLE objects in the active drawing.
        """
        if not self._ensure_active_document():
            return {"success": False, "error": "No active document. Please open a drawing first using autocad_open_drawing."}

        try:
            tables = []
            want_layer = layer.upper() if layer else None
            for rec in self._modelspace_records():
                try:
                    if rec["name"] != "AcDbTable":
                        continue
                    if want_layer is not None and rec["layer"] != want_layer:
                        continue
                    e = rec["obj"]
                    t = {
                        "handle": getattr(e, "Handle", None),
                        "layer": getattr(e, "Layer", None),
                    }
                    try:
                        t["rows"] = int(e.Rows)
                        t["cols"] = int(e.Columns)
                    except Exception:
                        pass
                    for attr in ("InsertionPoint", "Position"):
                        try:
                            pt = getattr(e, attr, None)
                            if pt is not None:
                                t["insertion_point"] = {"x": float(pt[0]), "y": float(pt[1])}
                                break
                        except Exception:
                            continue
                    tables.append(t)
                except Exception:
                    continue
            return {"success": True, "count": len(tables), "tables": tables}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_table_cell_text(self, handle: str, row: int, col: int) -> Dict[str, Any]:
        """Get text from a TABLE cell by handle."""
        if not self._ensure_active_document():
            return {"success": False, "error": "No active document. Please open a drawing first using autocad_open_drawing."}
        try:
            target = self._get_entity_by_handle(str(handle), object_name="AcDbTable")
            if target is None:
                return {"success": False, "error": f"TABLE with handle {handle} not found"}

            try:
                text = target.GetText(int(row), int(col))
            except Exception as ex:
                return {"success": False, "error": f"Could not read cell ({row},{col}): {ex}"}

            return {"success": True, "handle": handle, "row": int(row), "col": int(col), "text": text}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def set_table_cell_text(self, handle: str, row: int, col: int, text: str) -> Dict[str, Any]:
        """Set text for a TABLE cell by handle."""
        if not self.ensure_workflow_document():
            return {"success": False, "error": "No active document. Please open a drawing first using autocad_open_drawing."}
        try:
            def _do() -> Dict[str, Any]:
                target = self._get_entity_by_handle(str(handle), object_name="AcDbTable")
                if target is None:
                    return {"success": False, "error": f"TABLE with handle {handle} not found"}
                target.SetText(int(row), int(col), str(text))
                return {"success": True, "handle": handle, "row": int(row), "col": int(col), "text": str(text)}

            return self._com_retry(_do, attempts=8)
        except Exception as e:
            return {"success": False, "error": str(e)}

    def set_table_cell_text_height(
        self, handle: str, row: int, col: int, height: float
    ) -> Dict[str, Any]:
        """Set the text height for one TABLE cell."""
        if not self.ensure_workflow_document():
            return {"success": False, "error": "No active document."}
        try:
            def _do() -> Dict[str, Any]:
                target = self._get_entity_by_handle(str(handle), object_name="AcDbTable")
                if target is None:
                    return {"success": False, "error": f"TABLE with handle {handle} not found"}
                err = None
                for method in ("SetCellTextHeight", "SetTextHeight"):
                    try:
                        getattr(target, method)(int(row), int(col), float(height))
                        err = None
                        break
                    except Exception as ex:
                        err = ex
                        continue
                if err is not None:
                    return {"success": False, "error": f"Could not set text height ({row},{col}): {err}"}
                return {
                    "success": True,
                    "handle": handle,
                    "row": int(row),
                    "col": int(col),
                    "height": float(height),
                }

            return self._com_retry(_do, attempts=8)
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_table_cell_text_height(
        self, handle: str, row: int, col: int
    ) -> Dict[str, Any]:
        """Read the text height for one TABLE cell."""
        if not self._ensure_active_document():
            return {"success": False, "error": "No active document.", "height": 0.0}
        try:
            target = self._get_entity_by_handle(str(handle), object_name="AcDbTable")
            if target is None:
                return {"success": False, "error": f"TABLE with handle {handle} not found", "height": 0.0}
            err = None
            height = 0.0
            for method in ("GetCellTextHeight", "GetTextHeight"):
                try:
                    height = float(getattr(target, method)(int(row), int(col)))
                    err = None
                    break
                except Exception as ex:
                    err = ex
                    continue
            if err is not None:
                return {"success": False, "error": f"Could not read text height ({row},{col}): {err}", "height": 0.0}
            return {
                "success": True,
                "handle": handle,
                "row": int(row),
                "col": int(col),
                "height": float(height),
            }
        except Exception as e:
            return {"success": False, "error": str(e), "height": 0.0}

    def get_table_cell_text_style(
        self, handle: str, row: int, col: int
    ) -> Dict[str, Any]:
        """Read the text style name for one TABLE cell."""
        if not self._ensure_active_document():
            return {"success": False, "error": "No active document.", "style": ""}
        try:
            target = self._get_entity_by_handle(str(handle), object_name="AcDbTable")
            if target is None:
                return {"success": False, "error": f"TABLE with handle {handle} not found", "style": ""}
            style = ""
            for method in ("GetCellTextStyle", "GetTextStyle"):
                try:
                    style = str(getattr(target, method)(int(row), int(col)) or "")
                    if style:
                        break
                except Exception:
                    continue
            return {
                "success": True,
                "handle": handle,
                "row": int(row),
                "col": int(col),
                "style": style,
            }
        except Exception as e:
            return {"success": False, "error": str(e), "style": ""}

    def set_table_cell_text_style(
        self, handle: str, row: int, col: int, style_name: str
    ) -> Dict[str, Any]:
        """Set the text style name for one TABLE cell."""
        if not self._ensure_active_document():
            return {"success": False, "error": "No active document."}
        style_name = str(style_name or "").strip()
        if not style_name:
            return {"success": False, "error": "style_name is empty"}
        try:
            target = self._get_entity_by_handle(str(handle), object_name="AcDbTable")
            if target is None:
                return {"success": False, "error": f"TABLE with handle {handle} not found"}
            err = None
            for method in ("SetCellTextStyle", "SetTextStyle"):
                try:
                    getattr(target, method)(int(row), int(col), style_name)
                    err = None
                    break
                except Exception as ex:
                    err = ex
                    continue
            if err is not None:
                return {"success": False, "error": f"Could not set text style ({row},{col}): {err}"}
            return {
                "success": True,
                "handle": handle,
                "row": int(row),
                "col": int(col),
                "style": style_name,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def recompute_table(self, handle: str) -> Dict[str, Any]:
        """Force a TABLE to recompute layout after cell edits."""
        if not self._ensure_active_document():
            return {"success": False, "error": "No active document."}
        try:
            target = self._get_entity_by_handle(str(handle), object_name="AcDbTable")
            if target is None:
                return {"success": False, "error": f"TABLE with handle {handle} not found"}
            try:
                target.RecomputeTableBlock(True)
            except Exception:
                try:
                    target.RecomputeTableBlock()
                except Exception as ex:
                    return {"success": False, "error": str(ex)}
            return {"success": True, "handle": handle}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_table_column_width(self, handle: str, col: int = 0) -> Dict[str, Any]:
        """Read one TABLE column width (drawing units)."""
        if not self._ensure_active_document():
            return {"success": False, "error": "No active document.", "width": 0.0}
        try:
            modelspace = self.doc.ModelSpace
            target = None
            for i in range(modelspace.Count):
                e = modelspace.Item(i)
                if getattr(e, "Handle", None) == handle and getattr(e, "ObjectName", "") == "AcDbTable":
                    target = e
                    break
            if target is None:
                return {"success": False, "error": f"TABLE with handle {handle} not found", "width": 0.0}
            width = 0.0
            err = None
            for method in ("GetColumnWidth", "GetColumnWidth2"):
                try:
                    width = float(getattr(target, method)(int(col)))
                    err = None
                    break
                except Exception as ex:
                    err = ex
                    continue
            if err is not None or width <= 0.0:
                # Fallback: outer bbox width for single-column tables.
                try:
                    bb = target.GetBoundingBox()
                    width = abs(float(bb[1][0]) - float(bb[0][0]))
                    err = None
                except Exception as ex:
                    err = ex
            if err is not None:
                return {
                    "success": False,
                    "error": f"Could not read column width ({col}): {err}",
                    "width": 0.0,
                }
            return {"success": True, "handle": handle, "col": int(col), "width": float(width)}
        except Exception as e:
            return {"success": False, "error": str(e), "width": 0.0}

    def set_table_column_width(self, handle: str, col: int, width: float) -> Dict[str, Any]:
        """Set one TABLE column width (drawing units)."""
        if not self.ensure_workflow_document():
            return {"success": False, "error": "No active document."}
        try:
            def _do() -> Dict[str, Any]:
                modelspace = self.doc.ModelSpace
                target = None
                for i in range(modelspace.Count):
                    e = modelspace.Item(i)
                    if getattr(e, "Handle", None) == handle and getattr(e, "ObjectName", "") == "AcDbTable":
                        target = e
                        break
                if target is None:
                    return {"success": False, "error": f"TABLE with handle {handle} not found"}
                w = float(width)
                if w <= 0.0:
                    return {"success": False, "error": "width must be positive"}
                err = None
                for method in ("SetColumnWidth", "SetColumnWidth2"):
                    try:
                        getattr(target, method)(int(col), w)
                        err = None
                        break
                    except Exception as ex:
                        err = ex
                        continue
                if err is not None:
                    return {"success": False, "error": f"Could not set column width ({col}): {err}"}
                try:
                    target.RecomputeTableBlock(True)
                except Exception:
                    try:
                        target.RecomputeTableBlock()
                    except Exception:
                        pass
                return {
                    "success": True,
                    "handle": handle,
                    "col": int(col),
                    "width": w,
                }

            return self._com_retry(_do, attempts=8)
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_table_cell_mtext_line_step(
        self,
        handle: str,
        row: int,
        col: int,
        cell_text_hint: str = "",
    ) -> Dict[str, Any]:
        """
        Return the vertical distance between MTEXT baselines in a TABLE cell (drawing units).

        Uses the cell's text height and line-spacing style/factor from AutoCAD when available.
        AutoCAD "At least" spacing uses text_height * (5/3) * line_spacing_factor; "Exactly"
        uses text_height * line_spacing_factor.
        """
        if not self._ensure_active_document():
            return {"success": False, "error": "No active document.", "line_step": 0.0}
        try:
            modelspace = self.doc.ModelSpace
            target = None
            for i in range(modelspace.Count):
                e = modelspace.Item(i)
                if getattr(e, "Handle", None) == handle and getattr(e, "ObjectName", "") == "AcDbTable":
                    target = e
                    break
            if target is None:
                return {"success": False, "error": f"TABLE with handle {handle} not found", "line_step": 0.0}

            r, c = int(row), int(col)
            err = None
            text_height = 0.0
            for method in ("GetCellTextHeight", "GetTextHeight"):
                try:
                    text_height = float(getattr(target, method)(r, c))
                    err = None
                    break
                except Exception as ex:
                    err = ex
                    continue
            if err is not None:
                return {"success": False, "error": f"Could not read text height: {err}", "line_step": 0.0}

            line_spacing_factor = 1.0
            for method in ("GetTextLineSpacingFactor",):
                try:
                    line_spacing_factor = float(getattr(target, method)(r, c))
                    break
                except Exception:
                    continue

            line_spacing_style = 0  # acLineSpacingStyleAtLeast
            for method in ("GetTextLineSpacingStyle",):
                try:
                    line_spacing_style = int(getattr(target, method)(r, c))
                    break
                except Exception:
                    continue

            hint = str(cell_text_hint or "")
            if not hint.strip():
                try:
                    hint = str(target.GetText(r, c) or "")
                except Exception:
                    hint = ""
            import re

            m_px = re.search(r"\\pxsm([\d.]+)", hint, re.IGNORECASE)
            if m_px:
                try:
                    line_spacing_factor = float(m_px.group(1))
                except (TypeError, ValueError):
                    pass

            if line_spacing_style == 1:  # acLineSpacingStyleExactly
                line_step = text_height * line_spacing_factor
            else:
                line_step = text_height * (5.0 / 3.0) * line_spacing_factor

            return {
                "success": True,
                "handle": handle,
                "row": r,
                "col": c,
                "text_height": float(text_height),
                "line_spacing_factor": float(line_spacing_factor),
                "line_spacing_style": int(line_spacing_style),
                "line_step": float(line_step),
            }
        except Exception as e:
            return {"success": False, "error": str(e), "line_step": 0.0}

    def get_table_cell_extents(
        self,
        handle: str,
        row: int,
        col: int,
        outer: bool = True,
    ) -> Dict[str, Any]:
        """Return axis-aligned extents of a TABLE cell (ModelSpace coordinates)."""
        if not self._ensure_active_document():
            return {"success": False, "error": "No active document."}
        try:
            modelspace = self.doc.ModelSpace
            target = None
            for i in range(modelspace.Count):
                e = modelspace.Item(i)
                if getattr(e, "Handle", None) == handle and getattr(e, "ObjectName", "") == "AcDbTable":
                    target = e
                    break
            if target is None:
                return {"success": False, "error": f"TABLE with handle {handle} not found"}

            r, c = int(row), int(col)
            corners = None
            try:
                corners = target.GetCellExtents(r, c, bool(outer))
            except Exception:
                try:
                    corners = target.GetCellExtents(r, c)
                except Exception as ex:
                    return {"success": False, "error": f"GetCellExtents failed: {ex}"}

            pts: List[Tuple[float, float]] = []
            if corners is None:
                return {"success": False, "error": "GetCellExtents returned no data"}
            if isinstance(corners, (list, tuple)):
                flat = list(corners)
                if flat and isinstance(flat[0], (list, tuple)):
                    for p in flat:
                        if p is not None and len(p) >= 2:
                            pts.append((float(p[0]), float(p[1])))
                else:
                    for i in range(0, len(flat) - 1, 3):
                        try:
                            pts.append((float(flat[i]), float(flat[i + 1])))
                        except (TypeError, ValueError, IndexError):
                            continue
                    if not pts and len(flat) >= 4:
                        for i in range(0, len(flat) - 1, 2):
                            try:
                                pts.append((float(flat[i]), float(flat[i + 1])))
                            except (TypeError, ValueError, IndexError):
                                continue

            if not pts:
                return {"success": False, "error": "Could not parse cell corner coordinates"}

            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            return {
                "success": True,
                "handle": handle,
                "row": r,
                "col": c,
                "minx": float(min(xs)),
                "miny": float(min(ys)),
                "maxx": float(max(xs)),
                "maxy": float(max(ys)),
                "corners": [{"x": x, "y": y} for x, y in pts],
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def adjust_scalebar_below_scale_label(
        self,
        title_table_handle: str,
        scale_label_row: int = 8,
        scale_label_col: int = 0,
        gap: Optional[float] = None,
        scalebar_layers: Optional[List[str]] = None,
        template_scale_label_bottom: Optional[float] = None,
        scale_base_y: Optional[float] = None,
        scale_k: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Move CADA_SCALEBAR down so the scale-bar graphic sits below the title-block
        \"SCALE:- 1:xxx\" cell with a small clearance gap.

        Uses measured cell extents after table recompute/regen.
        """
        if not self._ensure_active_document():
            return {"success": False, "error": "No active document."}
        layers = [str(l) for l in (scalebar_layers or ["CADA_SCALEBAR"])]
        try:
            modelspace = self.doc.ModelSpace
            target = None
            for i in range(modelspace.Count):
                e = modelspace.Item(i)
                if getattr(e, "Handle", None) == title_table_handle and getattr(e, "ObjectName", "") == "AcDbTable":
                    target = e
                    break
            if target is None:
                return {"success": False, "error": f"TABLE with handle {title_table_handle} not found"}

            try:
                target.RecomputeTableBlock(True)
            except Exception:
                try:
                    target.RecomputeTableBlock()
                except Exception:
                    pass
            try:
                self.execute_command("REGEN")
            except Exception:
                pass

            cell = self.get_table_cell_extents(
                title_table_handle, int(scale_label_row), int(scale_label_col), outer=True
            )
            if not cell.get("success"):
                return cell

            label_bottom = float(cell["miny"])

            sb = self.get_modelspace_bbox(layers=layers)
            if not sb.get("success"):
                return {"success": False, "error": sb.get("error") or "Could not read scalebar bbox"}

            scalebar_top = float(sb["maxy"])

            clearance = gap
            if clearance is None:
                try:
                    th_res = self.get_table_cell_text_height(
                        title_table_handle, int(scale_label_row), int(scale_label_col)
                    )
                    th = float(th_res.get("height") or 0.0) if th_res.get("success") else 0.0
                    clearance = max(0.25, 0.12 * th)
                except Exception:
                    clearance = 0.5

            # Scale-bar top must sit below the bottom of the SCALE:- cell (Y-up: lower maxy).
            dy_overlap = label_bottom - float(clearance) - scalebar_top
            dy = dy_overlap if dy_overlap < -1e-6 else 0.0

            if template_scale_label_bottom is not None:
                try:
                    base_y = float(scale_base_y or 0.0)
                    sk = float(scale_k or 1.0)
                    tpl_y = float(template_scale_label_bottom)
                    scaled_tpl = base_y + sk * (tpl_y - base_y)
                    drop = scaled_tpl - label_bottom
                    if drop > 1e-6:
                        dy_drop = -drop
                        if dy < 0:
                            dy = min(dy, dy_drop)
                        else:
                            dy = dy_drop
                except (TypeError, ValueError):
                    pass

            if dy >= -1e-6:
                return {
                    "success": True,
                    "moved": False,
                    "dy": 0.0,
                    "scale_label_bottom": label_bottom,
                    "scalebar_top": scalebar_top,
                    "clearance": float(clearance),
                }

            mv = self.move_modelspace_by_layers(0.0, dy, layers)
            if not mv.get("success"):
                return mv
            return {
                "success": True,
                "moved": True,
                "dy": float(dy),
                "scale_label_bottom": label_bottom,
                "scalebar_top_before": scalebar_top,
                "clearance": float(clearance),
                "moved_entities": int(mv.get("moved_entities", 0) or 0),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_sample_text_height(self, layers: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Sample text height from the first TEXT or MTEXT entity on the given layers.
        Used to match template styling for bearing/distance and road text.
        Returns default 1.2 if no suitable entity found.
        """
        if not self._ensure_active_document():
            return {"success": False, "error": "No active document.", "height": 1.2}
        want = {str(l).upper() for l in (layers or ["CADA_BEARING_DIST", "CADA_ROAD"])}
        try:
            for rec in self._modelspace_records():
                try:
                    obj = rec["name"]
                    if "Text" not in obj and "MText" not in obj:
                        continue
                    if rec["layer"] not in want:
                        continue
                    e = rec["obj"]
                    for attr in ("Height", "TextHeight"):
                        try:
                            h = getattr(e, attr, None)
                            if h is not None:
                                hf = float(h)
                                if 0.01 < hf < 1000.0:
                                    return {"success": True, "height": hf}
                        except (TypeError, ValueError):
                            continue
                except Exception:
                    continue
            return {"success": True, "height": 1.2}
        except Exception as e:
            return {"success": False, "error": str(e), "height": 1.2}

    def list_inserts(self, layer: Optional[str] = None) -> Dict[str, Any]:
        """List INSERT (block reference) entities."""
        if not self._ensure_active_document():
            return {"success": False, "error": "No active document. Please open a drawing first using autocad_open_drawing."}
        try:
            inserts = []
            want_layer = layer.upper() if layer else None
            for rec in self._modelspace_records():
                try:
                    obj = rec["name"]
                    if "BlockReference" not in obj and ENTITY_TYPES.get(obj) != "INSERT":
                        continue
                    if want_layer is not None and rec["layer"] != want_layer:
                        continue
                    e = rec["obj"]
                    item = {
                        "handle": getattr(e, "Handle", None),
                        "layer": getattr(e, "Layer", None),
                        "type": "INSERT",
                    }
                    for attr in ("EffectiveName", "Name"):
                        try:
                            v = getattr(e, attr, None)
                            if v:
                                item["block_name"] = str(v)
                                break
                        except Exception:
                            continue
                    try:
                        item["insertion_point"] = {
                            "x": float(e.InsertionPoint[0]),
                            "y": float(e.InsertionPoint[1]),
                        }
                    except Exception:
                        pass
                    inserts.append(item)
                except Exception:
                    continue
            return {"success": True, "count": len(inserts), "inserts": inserts}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_active_document_path(self) -> Optional[str]:
        """Return the full path of the active document, or None if no document."""
        if not self._ensure_active_document():
            return None
        try:
            return str(getattr(self.doc, "FullName", "") or "")
        except Exception:
            return None

    def save_active_drawing(self) -> Dict[str, Any]:
        """Save the active drawing (equivalent to QSAVE / doc.Save()).
        STRICT: Never saves read-only documents (e.g. survey plan template) to avoid corruption.
        """
        if not self._ensure_active_document():
            return {"success": False, "error": "No active document. Please open a drawing first using autocad_open_drawing."}
        try:
            if getattr(self.doc, "ReadOnly", False):
                logger.warning("Survey plan template (read-only) will not be written; save skipped to avoid corruption.")
                return {
                    "success": True,
                    "name": getattr(self.doc, "Name", None),
                    "path": getattr(self.doc, "FullName", None),
                    "skipped_readonly": True,
                    "message": "Document is read-only; save skipped to prevent template corruption.",
                }
            self.doc.Save()
            return {"success": True, "name": getattr(self.doc, "Name", None), "path": getattr(self.doc, "FullName", None)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ==========================================================================
    # WRITE OPERATIONS (Template-driven automation)
    # ==========================================================================

    def delete_entities(self, layer: str, entity_object_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Delete entities in ModelSpace on a given layer. Retries once on COM hiccups."""
        return self.delete_entities_on_layers([layer], entity_object_names=entity_object_names)

    def delete_entities_on_layers(
        self,
        layers: List[str],
        entity_object_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Delete entities on multiple layers in one reverse ModelSpace pass."""
        if not self.ensure_workflow_document():
            return {"success": False, "error": "No active document. Please open a drawing first using autocad_open_drawing."}
        layer_set = {str(l).lower() for l in (layers or []) if str(l).strip()}
        if not layer_set:
            return {"success": True, "deleted": 0, "by_layer": {}}

        def _do_delete() -> Dict[str, Any]:
            deleted = 0
            by_layer: Dict[str, int] = {l: 0 for l in layer_set}
            try:
                from survyai.perf import incr
                incr("autocad_batched_deletes")
            except Exception:
                pass
            for rec in reversed(self._modelspace_records()):
                try:
                    layer_name = rec["layer"].lower()
                    if layer_name not in layer_set:
                        continue
                    if entity_object_names:
                        if rec["name"] not in entity_object_names:
                            continue
                    rec["obj"].Delete()
                    deleted += 1
                    by_layer[layer_name] = int(by_layer.get(layer_name, 0) or 0) + 1
                except Exception:
                    continue
            self._invalidate_handle_cache()
            return {"success": True, "deleted": deleted, "by_layer": by_layer, "layers": list(layer_set)}

        try:
            return _do_delete()
        except Exception as e:
            try:
                time.sleep(0.3)
                return _do_delete()
            except Exception as e2:
                return {"success": False, "error": str(e2)}

    def create_lwpolyline(
        self,
        points_xy: List[Dict[str, float]],
        layer: str,
        closed: bool = True,
        linetype_scale: Optional[float] = None,
        assume_active: bool = False,
    ) -> Dict[str, Any]:
        """Create a lightweight polyline in ModelSpace."""
        if not assume_active and not self._ensure_active_document():
            return {"success": False, "error": "No active document. Please open a drawing first using autocad_open_drawing."}
        if assume_active and self.doc is None:
            return {"success": False, "error": "No active document. Please open a drawing first using autocad_open_drawing."}
        if not points_xy or len(points_xy) < 2:
            return {"success": False, "error": "At least 2 points required"}
        try:
            import pythoncom
            import win32com.client
            ms = self.doc.ModelSpace
            coords = []
            for p in points_xy:
                coords.extend([float(p["x"]), float(p["y"])])
            var = win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, tuple(coords))
            pl = None
            try:
                pl = ms.AddLightWeightPolyline(var)
            except Exception:
                coords3 = []
                for p in points_xy:
                    coords3.extend([float(p["x"]), float(p["y"]), 0.0])
                var3 = win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, tuple(coords3))
                pl = ms.AddPolyline(var3)
            try:
                pl.Layer = layer
            except Exception:
                pass
            # Ensure entity color is ByLayer (256) so it inherits the layer's red (ACI=1) for CADA_BOUNDARY.
            try:
                pl.Color = 256
            except Exception:
                pass
            if closed:
                try:
                    pl.Closed = True
                except Exception:
                    pass
            if linetype_scale is not None:
                try:
                    pl.LinetypeScale = float(linetype_scale)
                except Exception:
                    pass
            return {"success": True, "handle": getattr(pl, "Handle", None), "layer": getattr(pl, "Layer", None), "closed": bool(closed)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def delete_entity_by_handle(self, handle: str) -> Dict[str, Any]:
        """Delete a single entity in ModelSpace by handle."""
        if not self._ensure_active_document():
            return {"success": False, "error": "No active document. Please open a drawing first using autocad_open_drawing."}
        try:
            ms = self.doc.ModelSpace
            for i in range(ms.Count):
                e = ms.Item(i)
                if getattr(e, "Handle", None) == handle:
                    try:
                        e.Delete()
                        return {"success": True, "handle": handle}
                    except Exception as ex:
                        return {"success": False, "error": str(ex)}
            return {"success": False, "error": f"Entity with handle {handle} not found"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def move_entity_to_xy(self, handle: str, x: float, y: float) -> Dict[str, Any]:
        """Move an entity (INSERT/TABLE/etc) to a target XY."""
        if not self._ensure_active_document():
            return {"success": False, "error": "No active document. Please open a drawing first using autocad_open_drawing."}
        try:
            ms = self.doc.ModelSpace
            target = None
            for i in range(ms.Count):
                e = ms.Item(i)
                if getattr(e, "Handle", None) == handle:
                    target = e
                    break
            if target is None:
                return {"success": False, "error": f"Entity with handle {handle} not found"}

            for prop in ("InsertionPoint", "Position"):
                try:
                    _ = getattr(target, prop, None)
                    setattr(target, prop, (float(x), float(y), 0.0))
                    return {"success": True, "handle": handle, "moved_via": prop, "x": float(x), "y": float(y)}
                except Exception:
                    continue

            cur = None
            for prop in ("InsertionPoint", "Position"):
                try:
                    pt = getattr(target, prop, None)
                    if pt is not None:
                        cur = (float(pt[0]), float(pt[1]))
                        break
                except Exception:
                    continue
            if cur is None:
                return {"success": False, "error": "Could not determine current insertion/position point for move"}

            dx = float(x) - cur[0]
            dy = float(y) - cur[1]
            try:
                import pythoncom
                import win32com.client
                p_from = win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, (0.0, 0.0, 0.0))
                p_to = win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, (float(dx), float(dy), 0.0))
                target.Move(p_from, p_to)
                return {"success": True, "handle": handle, "moved_via": "Move(VARIANT)", "dx": dx, "dy": dy}
            except Exception as e:
                try:
                    target.Move((0.0, 0.0, 0.0), (float(dx), float(dy), 0.0))
                    return {"success": True, "handle": handle, "moved_via": "Move(tuple)", "dx": dx, "dy": dy}
                except Exception as e2:
                    return {"success": False, "error": str(e2)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def copy_entity_by_handle(self, handle: str, dx: float = 0.0, dy: float = 0.0, layer: Optional[str] = None) -> Dict[str, Any]:
        """
        Copy a single ModelSpace entity by handle and optionally move it by (dx, dy).
        Useful for cloning template-driven TABLE entities (e.g., CADA_PILLARNUMBERS).
        """
        if not self._ensure_active_document():
            return {"success": False, "error": "No active document. Please open a drawing first using autocad_open_drawing."}
        try:
            import pythoncom
            import win32com.client

            ms = self.doc.ModelSpace
            src = None
            for i in range(ms.Count):
                try:
                    e = ms.Item(i)
                    if str(getattr(e, "Handle", "")).upper() == str(handle).upper():
                        src = e
                        break
                except Exception:
                    continue
            if src is None:
                return {"success": False, "error": f"Entity with handle {handle} not found"}

            new_ent = None
            last_err = None
            for attempt in range(6):
                try:
                    new_ent = src.Copy()
                    break
                except Exception as ex:
                    last_err = ex
                    time.sleep(0.2 * (attempt + 1))
            if new_ent is None:
                return {"success": False, "error": str(last_err) if last_err else "Copy failed"}

            if layer:
                try:
                    new_ent.Layer = str(layer)
                except Exception:
                    pass

            dx_f = float(dx)
            dy_f = float(dy)
            if abs(dx_f) > 1e-12 or abs(dy_f) > 1e-12:
                try:
                    p_from = win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, (0.0, 0.0, 0.0))
                    p_to = win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, (dx_f, dy_f, 0.0))
                    new_ent.Move(p_from, p_to)
                except Exception:
                    try:
                        new_ent.Move((0.0, 0.0, 0.0), (dx_f, dy_f, 0.0))
                    except Exception:
                        pass

            out = {"success": True, "source_handle": str(handle), "handle": getattr(new_ent, "Handle", None), "layer": getattr(new_ent, "Layer", None)}
            try:
                for attr in ("InsertionPoint", "Position"):
                    pt = getattr(new_ent, attr, None)
                    if pt is not None:
                        out["insertion_point"] = {"x": float(pt[0]), "y": float(pt[1])}
                        break
            except Exception:
                pass
            return out
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_entity_bboxes(self, layers: Optional[List[str]] = None, object_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        List axis-aligned bounding boxes for matching ModelSpace entities.
        Returns individual bboxes (not union), best-effort.
        """
        if not self._ensure_active_document():
            return {"success": False, "error": "No active document. Please open a drawing first using autocad_open_drawing."}
        want_layers = {str(l).upper() for l in (layers or []) if str(l).strip()}
        want_objs = {str(o) for o in (object_names or []) if str(o).strip()}
        try:
            ms = self.doc.ModelSpace
            out = []
            for i in range(ms.Count):
                try:
                    e = ms.Item(i)
                    lyr = str(getattr(e, "Layer", "")).upper()
                    obj = str(getattr(e, "ObjectName", ""))
                    if want_layers and lyr not in want_layers:
                        continue
                    if want_objs and obj not in want_objs:
                        continue
                    try:
                        bb = e.GetBoundingBox()
                        pmin, pmax = bb[0], bb[1]
                        out.append(
                            {
                                "handle": getattr(e, "Handle", None),
                                "layer": getattr(e, "Layer", None),
                                "object_name": obj,
                                "min": {"x": float(pmin[0]), "y": float(pmin[1])},
                                "max": {"x": float(pmax[0]), "y": float(pmax[1])},
                            }
                        )
                    except Exception:
                        continue
                except Exception:
                    continue
            return {"success": True, "count": len(out), "bboxes": out}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def set_layer_color(self, layer: str, color_code: int) -> Dict[str, Any]:
        """Set a layer's color (ACI). Example: color_code=1 for red."""
        if not self._ensure_active_document():
            return {"success": False, "error": "No active document. Please open a drawing first using autocad_open_drawing."}
        try:
            lyr = self.doc.Layers.Item(layer)
            lyr.Color = int(color_code)
            return {"success": True, "layer": layer, "color": int(color_code)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def insert_block(
        self,
        block_name: str,
        x: float,
        y: float,
        layer: Optional[str] = None,
        xscale: float = 1.0,
        yscale: float = 1.0,
        zscale: float = 1.0,
        rotation_rad: float = 0.0,
        assume_active: bool = False,
    ) -> Dict[str, Any]:
        """Insert a block reference into ModelSpace."""
        if not assume_active and not self.ensure_workflow_document():
            return {"success": False, "error": "No active document. Please open a drawing first using autocad_open_drawing."}
        if assume_active and self.doc is None:
            return {"success": False, "error": "No active document. Please open a drawing first using autocad_open_drawing."}
        try:
            import pythoncom
            import win32com.client
            ms = self.doc.ModelSpace
            ip = win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, (float(x), float(y), 0.0))
            last_err = None
            for attempt in range(3):
                try:
                    ins = ms.InsertBlock(ip, str(block_name), float(xscale), float(yscale), float(zscale), float(rotation_rad))
                    if layer:
                        try:
                            ins.Layer = str(layer)
                        except Exception:
                            pass
                    return {
                        "success": True,
                        "handle": getattr(ins, "Handle", None),
                        "block_name": str(block_name),
                        "layer": getattr(ins, "Layer", None),
                        "insertion_point": {"x": float(x), "y": float(y)},
                    }
                except Exception as e:
                    last_err = e
                    time.sleep(0.2 * (attempt + 1))
            return {"success": False, "error": str(last_err)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def move_entities_by_handles(self, dx: float, dy: float, handles: List[str]) -> Dict[str, Any]:
        """
        Move a specific set of ModelSpace entities by (dx, dy) using their handles.
        This is essential for moving guide/annotation geometry (LINE/LWPOLYLINE/etc) that
        does not have an InsertionPoint/Position and can't be moved via move_entity_to_xy.
        """
        if not self._ensure_active_document():
            return {"success": False, "error": "No active document. Please open a drawing first using autocad_open_drawing."}
        try:
            import pythoncom
            import win32com.client
            ms = self.doc.ModelSpace
            want = {str(h).upper() for h in (handles or []) if str(h).strip()}
            if not want:
                return {"success": True, "dx": float(dx), "dy": float(dy), "moved": 0, "requested": 0}
            p_from = win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, (0.0, 0.0, 0.0))
            p_to = win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, (float(dx), float(dy), 0.0))
            moved = 0
            for i in range(ms.Count):
                try:
                    e = ms.Item(i)
                    h = str(getattr(e, "Handle", "")).upper()
                    if h not in want:
                        continue
                    try:
                        e.Move(p_from, p_to)
                    except Exception:
                        e.Move((0.0, 0.0, 0.0), (float(dx), float(dy), 0.0))
                    moved += 1
                except Exception:
                    continue
            return {"success": True, "dx": float(dx), "dy": float(dy), "moved": moved, "requested": len(want)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def add_text(
        self,
        text: str,
        x: float,
        y: float,
        layer: Optional[str] = None,
        rotation_rad: float = 0.0,
        height: float = 1.2,
        alignment: int = 10,  # acAlignmentMiddleCenter (best-effort)
    ) -> Dict[str, Any]:
        """
        Add a single-line TEXT entity to ModelSpace. Best-effort center alignment.
        This is more stable than MTEXT for "stick to point" labeling.
        """
        if not self._ensure_active_document():
            return {"success": False, "error": "No active document. Please open a drawing first using autocad_open_drawing."}
        try:
            import pythoncom
            import win32com.client
            ms = self.doc.ModelSpace
            ip = win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, (float(x), float(y), 0.0))
            tx = ms.AddText(str(text), ip, float(height))
            if layer:
                try:
                    tx.Layer = str(layer)
                except Exception:
                    pass
            # Prefer ByLayer color
            try:
                tx.Color = 256
            except Exception:
                pass
            # Rotation
            try:
                tx.Rotation = float(rotation_rad)
            except Exception:
                pass
            # Center align if supported
            try:
                tx.Alignment = int(alignment)
                tx.TextAlignmentPoint = (float(x), float(y), 0.0)
            except Exception:
                pass
            # Re-anchor: some combinations of alignment/rotation shift the insertion point
            try:
                cur = tx.InsertionPoint
                dx = float(x) - float(cur[0])
                dy = float(y) - float(cur[1])
                if abs(dx) > 1e-9 or abs(dy) > 1e-9:
                    p_from = win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, (0.0, 0.0, 0.0))
                    p_to = win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, (dx, dy, 0.0))
                    tx.Move(p_from, p_to)
            except Exception:
                pass
            return {"success": True, "handle": getattr(tx, "Handle", None), "layer": getattr(tx, "Layer", None)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def purge_all_modelspace(self) -> Dict[str, Any]:
        """
        Best-effort: delete every entity in ModelSpace.
        Useful for reset workflows where template border/title live in PaperSpace.
        """
        if not self._ensure_active_document():
            return {"success": False, "error": "No active document. Please open a drawing first using autocad_open_drawing."}
        try:
            deleted = 0
            ms = self.doc.ModelSpace
            for i in range(ms.Count - 1, -1, -1):
                try:
                    ms.Item(i).Delete()
                    deleted += 1
                except Exception:
                    continue
            return {"success": True, "deleted": deleted}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_modelspace_bbox(
        self,
        layers: Optional[List[str]] = None,
        object_names: Optional[List[str]] = None,
        block_name_contains: Optional[str] = None,
        prefer_largest: bool = False,
    ) -> Dict[str, Any]:
        """
        Compute a bounding box for entities in ModelSpace.
        - If prefer_largest=True, returns bbox of the single largest matching entity.
        - Otherwise returns the union bbox.
        """
        if not self._ensure_active_document():
            return {"success": False, "error": "No active document. Please open a drawing first using autocad_open_drawing."}
        try:
            want_layers = {str(l).upper() for l in (layers or [])} if layers else None
            want_objs = set(object_names or []) if object_names else None
            bn_sub = (block_name_contains or "").upper().strip() or None

            def _scan(records: List[Dict[str, Any]]):
                best = None  # (area, minx, miny, maxx, maxy, handle)
                agg = None   # (minx, miny, maxx, maxy)
                matched = 0
                for rec in records:
                    try:
                        if want_layers is not None and rec["layer"] not in want_layers:
                            continue
                        on = rec["name"]
                        if want_objs is not None and on not in want_objs:
                            continue
                        e = rec["obj"]
                        if bn_sub:
                            # Only meaningful for block refs
                            if "BlockReference" not in on:
                                continue
                            nm = str(getattr(e, "EffectiveName", "") or getattr(e, "Name", "") or "").upper()
                            if bn_sub not in nm:
                                continue
                        try:
                            bb = e.GetBoundingBox()
                            pmin, pmax = bb[0], bb[1]
                            minx, miny = float(pmin[0]), float(pmin[1])
                            maxx, maxy = float(pmax[0]), float(pmax[1])
                        except Exception:
                            continue

                        matched += 1
                        if agg is None:
                            agg = (minx, miny, maxx, maxy)
                        else:
                            agg = (min(agg[0], minx), min(agg[1], miny), max(agg[2], maxx), max(agg[3], maxy))

                        area = (maxx - minx) * (maxy - miny)
                        if best is None or area > best[0]:
                            best = (area, minx, miny, maxx, maxy, rec["handle"])
                    except Exception:
                        continue
                return best, agg, matched

            reused = self._ms_snapshot is not None
            best, agg, matched = _scan(self._modelspace_records())
            if matched == 0 and reused:
                # A cached proxy set can be invalidated by AutoCAD (regen/undo); never
                # report an empty bbox until a fresh pass agrees.
                best, agg, matched = _scan(self._modelspace_records(force=True))

            if matched == 0 or (prefer_largest and best is None) or (not prefer_largest and agg is None):
                return {"success": False, "error": "No matching entities for bbox"}

            if prefer_largest:
                _, minx, miny, maxx, maxy, h = best
            else:
                minx, miny, maxx, maxy = agg
                h = None
            return {
                "success": True,
                "matched": matched,
                "handle": h,
                "minx": float(minx),
                "miny": float(miny),
                "maxx": float(maxx),
                "maxy": float(maxy),
                "center": {"x": float((minx + maxx) / 2.0), "y": float((miny + maxy) / 2.0)},
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def move_modelspace_by_layers(self, dx: float, dy: float, layers: List[str]) -> Dict[str, Any]:
        """Move all ModelSpace entities whose Layer is in layers by (dx, dy)."""
        if not self._ensure_active_document():
            return {"success": False, "error": "No active document. Please open a drawing first using autocad_open_drawing."}
        try:
            import pythoncom
            import win32com.client
            want = {str(l).upper() for l in (layers or [])}
            p_from = win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, (0.0, 0.0, 0.0))
            p_to = win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, (float(dx), float(dy), 0.0))
            moved = 0
            for rec in self._modelspace_records():
                try:
                    if rec["layer"] not in want:
                        continue
                    e = rec["obj"]
                    try:
                        e.Move(p_from, p_to)
                    except Exception:
                        e.Move((0.0, 0.0, 0.0), (float(dx), float(dy), 0.0))
                    moved += 1
                except Exception:
                    continue
            return {"success": True, "dx": float(dx), "dy": float(dy), "layers": list(want), "moved_entities": moved}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def scale_modelspace_by_layers(
        self,
        base_x: float,
        base_y: float,
        scale_factor: float,
        layers: List[str],
    ) -> Dict[str, Any]:
        """
        Scale all ModelSpace entities on the given layers by scale_factor about (base_x, base_y).
        Uses AutoCAD COM ScaleEntity; best-effort across entity types.
        """
        if not self.ensure_workflow_document():
            return {"success": False, "error": "No active document. Please open a drawing first using autocad_open_drawing."}
        if not layers or float(scale_factor) <= 0.0:
            return {"success": False, "error": "layers must be non-empty and scale_factor must be positive"}
        try:
            import pythoncom
            import win32com.client

            def _do() -> Dict[str, Any]:
                want = {str(l).upper() for l in layers}
                base_pt = win32com.client.VARIANT(
                    pythoncom.VT_ARRAY | pythoncom.VT_R8,
                    (float(base_x), float(base_y), 0.0),
                )
                sf = float(scale_factor)
                scaled = 0
                for rec in self._modelspace_records():
                    try:
                        if rec["layer"] not in want:
                            continue
                        e = rec["obj"]
                        try:
                            e.ScaleEntity(base_pt, sf)
                            scaled += 1
                        except Exception:
                            try:
                                e.ScaleEntity((float(base_x), float(base_y), 0.0), sf)
                                scaled += 1
                            except Exception:
                                continue
                    except Exception:
                        continue
                return {
                    "success": True,
                    "base_x": float(base_x),
                    "base_y": float(base_y),
                    "scale_factor": sf,
                    "layers": list(want),
                    "scaled_entities": scaled,
                }

            return self._com_retry(_do, attempts=6, base_sleep=0.25)
        except Exception as e:
            return {"success": False, "error": str(e)}

    def scale_scalebar_text_values(
        self,
        scale_factor: float,
        layers: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Multiply numeric values in TEXT/MTEXT entities on scalebar layers by scale_factor.

        This is used to keep scale bar labels correct when a template sheet is scaled up/down.
        Example: template 1:500 -> output 1:250 => factor = 250/500 = 0.5, so "10m" becomes "5m".

        Notes:
        - Only modifies entities whose Layer matches provided layers (case-insensitive).
        - Attempts to update both ModelSpace entities and entities inside Block definitions.
        """
        if not self._ensure_active_document():
            return {"success": False, "error": "No active document. Please open a drawing first using autocad_open_drawing."}
        try:
            import re

            sf = float(scale_factor)
            if sf <= 0:
                return {"success": False, "error": "scale_factor must be positive"}

            want = {str(l).upper() for l in (layers or ["SCALEBAR", "CADA_SCALEBAR"])}

            num_re = re.compile(r"-?\d+(?:\.\d+)?")

            def _fmt(x: float) -> str:
                # Avoid -0, keep clean numeric formatting
                if abs(x) < 5e-10:
                    x = 0.0
                xr = round(x, 6)
                if abs(xr - round(xr)) < 1e-9:
                    return str(int(round(xr)))
                s = ("{:.6f}".format(xr)).rstrip("0").rstrip(".")
                return s if s else "0"

            def _should_update(text: str) -> bool:
                # On scalebar layers, *all* numeric labels should scale (e.g. "10m" and "5").
                # Only skip scale-ratio patterns if they ever appear (e.g. "1:250").
                t = (text or "")
                if re.search(r"\b1\s*:\s*\d+\b", t):
                    return False
                return bool(num_re.search(t))

            def _scale_numbers(text: str) -> str:
                if not text:
                    return text
                if not _should_update(text):
                    return text

                def repl(m: re.Match) -> str:
                    try:
                        v = float(m.group(0))
                    except Exception:
                        return m.group(0)
                    return _fmt(v * sf)

                return num_re.sub(repl, text)

            def _get_text(ent) -> Optional[str]:
                for prop in ("TextString", "Contents", "Text"):
                    try:
                        v = getattr(ent, prop, None)
                        if v is not None:
                            return str(v)
                    except Exception:
                        continue
                return None

            def _set_text(ent, new_text: str) -> bool:
                for prop in ("TextString", "Contents", "Text"):
                    try:
                        if hasattr(ent, prop):
                            setattr(ent, prop, str(new_text))
                            return True
                    except Exception:
                        continue
                return False

            def _is_text_entity(ent) -> bool:
                try:
                    obj = str(getattr(ent, "ObjectName", "") or "")
                except Exception:
                    obj = ""
                if obj in ("AcDbText", "AcDbMText"):
                    return True
                # Some AutoCAD variants expose attribute references as text-like
                if "AcDbAttribute" in obj:
                    return True
                return False

            def _layer_ok(ent) -> bool:
                try:
                    lyr = str(getattr(ent, "Layer", "") or "").upper()
                    return lyr in want
                except Exception:
                    return False

            updated = 0
            scanned = 0

            # 1) ModelSpace
            ms = self.doc.ModelSpace
            for i in range(ms.Count):
                try:
                    e = ms.Item(i)
                    if not _is_text_entity(e) or not _layer_ok(e):
                        continue
                    scanned += 1
                    old = _get_text(e)
                    if old is None:
                        continue
                    new = _scale_numbers(old)
                    if new != old and _set_text(e, new):
                        updated += 1
                except Exception:
                    continue

            # 2) Block definitions (covers scalebar text nested in blocks)
            blocks_updated = 0
            blocks_scanned = 0
            blocks = getattr(self.doc, "Blocks", None)
            if blocks is not None:
                # Try index-based access first; if it fails, fallback to enumeration.
                try:
                    bcount = int(blocks.Count)
                    block_iter = (blocks.Item(i) for i in range(bcount))
                except Exception:
                    try:
                        block_iter = iter(blocks)
                    except Exception:
                        block_iter = []

                for b in block_iter:
                    try:
                        # IMPORTANT: Blocks collection includes special blocks like "*Model_Space" and "*Paper_Space"
                        # that reference the live contents of ModelSpace/PaperSpace. We already processed ModelSpace above,
                        # so skipping these prevents applying the scale factor twice (e.g. 0.5 -> 0.25).
                        try:
                            bname = str(getattr(b, "Name", "") or "").upper()
                            if bname in ("*MODEL_SPACE", "*PAPER_SPACE"):
                                continue
                        except Exception:
                            pass
                        blocks_scanned += 1
                        try:
                            ec = int(getattr(b, "Count", 0))
                        except Exception:
                            ec = 0
                        for j in range(ec):
                            try:
                                e = b.Item(j)
                                if not _is_text_entity(e) or not _layer_ok(e):
                                    continue
                                scanned += 1
                                old = _get_text(e)
                                if old is None:
                                    continue
                                new = _scale_numbers(old)
                                if new != old and _set_text(e, new):
                                    updated += 1
                                    blocks_updated += 1
                            except Exception:
                                continue
                    except Exception:
                        continue

            return {
                "success": True,
                "scale_factor": sf,
                "layers": list(want),
                "text_entities_scanned": scanned,
                "text_entities_updated": updated,
                "blocks_scanned": blocks_scanned,
                "block_text_entities_updated": blocks_updated,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def scale_hatch_pattern_scale_by_layers(
        self,
        scale_factor: float,
        layers: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Multiply Hatch PatternScale for hatches on the specified layers by scale_factor.

        Used so the scale bar hatching scales with the plan: e.g. template at 1:500 with
        hatch PatternScale 4, output at 1:1000 → pass factor 2 so hatch becomes 8.

        Notes:
        - In ModelSpace, modifies hatch entities whose Layer matches provided layers (case-insensitive).
        - In Block definitions, also modifies hatch entities in any block that is INSERTed on the provided
          layers (even if the hatch's internal layer is "0"/ByBlock), since scalebars are often block-based.
        """
        if not self._ensure_active_document():
            return {"success": False, "error": "No active document. Please open a drawing first using autocad_open_drawing."}
        try:
            sf = float(scale_factor)
            if sf <= 0:
                return {"success": False, "error": "scale_factor must be positive"}

            want = {str(l).upper() for l in (layers or ["SCALEBAR", "CADA_SCALEBAR"])}

            def _layer_ok(ent) -> bool:
                try:
                    lyr = str(getattr(ent, "Layer", "") or "").upper()
                    return lyr in want
                except Exception:
                    return False

            def _is_hatch_like(ent) -> bool:
                try:
                    obj = str(getattr(ent, "ObjectName", "") or "")
                except Exception:
                    obj = ""
                if obj == "AcDbHatch":
                    return True
                # Best-effort: some variants expose hatch-like entities with PatternScale
                try:
                    return hasattr(ent, "PatternScale")
                except Exception:
                    return False

            def _is_insert(ent) -> bool:
                try:
                    obj = str(getattr(ent, "ObjectName", "") or "")
                except Exception:
                    obj = ""
                return obj in ("AcDbBlockReference", "AcDbMInsertBlock") or "BlockReference" in obj

            def _get_block_name(ent) -> Optional[str]:
                for prop in ("EffectiveName", "Name", "BlockName"):
                    try:
                        v = getattr(ent, prop, None)
                        if v:
                            return str(v)
                    except Exception:
                        continue
                return None

            updated = 0
            scanned = 0
            blocks_updated = 0
            blocks_scanned = 0
            blocks_targeted = 0
            targeted_block_names: set[str] = set()

            def _adjust(ent, *, ignore_layer: bool = False) -> bool:
                try:
                    if (not ignore_layer) and (not _layer_ok(ent)):
                        return False
                    if not _is_hatch_like(ent):
                        return False
                    if not hasattr(ent, "PatternScale"):
                        return False
                    old = getattr(ent, "PatternScale", None)
                    if old is None:
                        return False
                    new = float(old) * sf
                    if new <= 0:
                        return False
                    setattr(ent, "PatternScale", new)
                    return True
                except Exception:
                    return False

            def _scan_space_for_targets(space) -> None:
                nonlocal targeted_block_names
                try:
                    for i in range(space.Count):
                        try:
                            e = space.Item(i)
                            if not _is_insert(e) or not _layer_ok(e):
                                continue
                            bn = _get_block_name(e)
                            if bn:
                                targeted_block_names.add(str(bn).upper())
                        except Exception:
                            continue
                except Exception:
                    return

            def _scale_space_hatches(space) -> None:
                nonlocal scanned, updated
                try:
                    for i in range(space.Count):
                        try:
                            e = space.Item(i)
                            if not _layer_ok(e) or not _is_hatch_like(e):
                                continue
                            scanned += 1
                            if _adjust(e, ignore_layer=False):
                                updated += 1
                        except Exception:
                            continue
                except Exception:
                    return

            # 1) ModelSpace (+ PaperSpace if available)
            ms = self.doc.ModelSpace
            _scan_space_for_targets(ms)
            _scale_space_hatches(ms)
            try:
                ps = getattr(self.doc, "PaperSpace", None)
                if ps is not None:
                    _scan_space_for_targets(ps)
                    _scale_space_hatches(ps)
            except Exception:
                pass

            # 2) Block definitions (covers scalebar hatch nested in blocks)
            blocks = getattr(self.doc, "Blocks", None)
            if blocks is not None:
                try:
                    bcount = int(blocks.Count)
                    block_iter = (blocks.Item(i) for i in range(bcount))
                except Exception:
                    try:
                        block_iter = iter(blocks)
                    except Exception:
                        block_iter = []

                for b in block_iter:
                    try:
                        try:
                            bname = str(getattr(b, "Name", "") or "").upper()
                            if bname in ("*MODEL_SPACE", "*PAPER_SPACE"):
                                continue
                        except Exception:
                            pass
                        blocks_scanned += 1
                        is_target_block = False
                        try:
                            is_target_block = bool(bname) and (bname in targeted_block_names)
                        except Exception:
                            is_target_block = False
                        if is_target_block:
                            blocks_targeted += 1
                        try:
                            ec = int(getattr(b, "Count", 0))
                        except Exception:
                            ec = 0
                        for j in range(ec):
                            try:
                                e = b.Item(j)
                                # In target blocks, allow hatch entities on layer "0"/ByBlock, etc.
                                if (not is_target_block) and (not _layer_ok(e)):
                                    continue
                                if not _is_hatch_like(e):
                                    continue
                                scanned += 1
                                if _adjust(e, ignore_layer=is_target_block):
                                    updated += 1
                                    blocks_updated += 1
                            except Exception:
                                continue
                    except Exception:
                        continue

            return {
                "success": True,
                "scale_factor": sf,
                "layers": list(want),
                "target_block_names": sorted(list(targeted_block_names))[:25],
                "target_blocks_count": int(blocks_targeted),
                "hatches_scanned": scanned,
                "hatches_updated": updated,
                "blocks_scanned": blocks_scanned,
                "block_hatches_updated": blocks_updated,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_entity_insertion_xy(self, handle: str) -> Dict[str, Any]:
        """Get an entity's (x,y) from InsertionPoint/Position by handle (best-effort)."""
        if not self._ensure_active_document():
            return {"success": False, "error": "No active document. Please open a drawing first using autocad_open_drawing."}
        try:
            ms = self.doc.ModelSpace
            target = None
            for i in range(ms.Count):
                e = ms.Item(i)
                if getattr(e, "Handle", None) == handle:
                    target = e
                    break
            if target is None:
                return {"success": False, "error": f"Entity with handle {handle} not found"}
            for prop in ("InsertionPoint", "Position"):
                try:
                    pt = getattr(target, prop, None)
                    if pt is not None:
                        return {"success": True, "handle": handle, "prop": prop, "x": float(pt[0]), "y": float(pt[1])}
                except Exception:
                    continue
            return {"success": False, "error": "Entity has no InsertionPoint/Position"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def add_mtext(
        self,
        text: str,
        x: float,
        y: float,
        layer: Optional[str] = None,
        rotation_rad: float = 0.0,
        height: Optional[float] = None,
        width: float = 0.0,
        attachment_point: int = 5,
        assume_active: bool = False,
    ) -> Dict[str, Any]:
        """Add an MTEXT entity to ModelSpace."""
        if not assume_active and not self._ensure_active_document():
            return {"success": False, "error": "No active document. Please open a drawing first using autocad_open_drawing."}
        if assume_active and self.doc is None:
            return {"success": False, "error": "No active document. Please open a drawing first using autocad_open_drawing."}
        try:
            import pythoncom
            import win32com.client
            ms = self.doc.ModelSpace
            w = float(width)
            if w <= 0.0:
                # AutoCAD can reject width=0 for MTEXT; use a small positive width.
                w = 10.0
            ip = win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, (float(x), float(y), 0.0))
            mt = None
            last_err = None
            # AutoCAD can intermittently reject COM calls ("Call was rejected by callee").
            # Retry with backoff for stability under load or glitches.
            for attempt in range(6):
                try:
                    mt = ms.AddMText(ip, w, str(text))
                    break
                except Exception as ex:
                    last_err = ex
                    time.sleep(0.2 * (attempt + 1))
            if mt is None:
                raise Exception(last_err)
            if layer:
                try:
                    mt.Layer = str(layer)
                except Exception:
                    pass
            # Set height FIRST so the entity is never left at AutoCAD's default (often huge) if COM glitches.
            req_height = float(height) if height is not None else None
            if req_height is not None:
                for prop in ("Height", "TextHeight"):
                    try:
                        setattr(mt, prop, req_height)
                        break
                    except Exception:
                        continue
            # Middle-center by default for segment-centered labels
            try:
                mt.AttachmentPoint = int(attachment_point)
            except Exception:
                pass
            try:
                mt.Rotation = float(rotation_rad)
            except Exception:
                pass
            # Verify/correct height after AttachmentPoint/Rotation (COM can sometimes drop or wrong-foot height)
            if req_height is not None:
                try:
                    actual = getattr(mt, "Height", None) or getattr(mt, "TextHeight", None)
                    if actual is not None:
                        actual = float(actual)
                        if actual < 0.25 * req_height or actual > 4.0 * req_height:
                            for prop in ("Height", "TextHeight"):
                                try:
                                    setattr(mt, prop, req_height)
                                    break
                                except Exception:
                                    continue
                except Exception:
                    pass

            # IMPORTANT: AutoCAD can shift the InsertionPoint when AttachmentPoint/Rotation are set.
            # Re-anchor the MTEXT so its InsertionPoint ends up exactly at (x, y).
            try:
                cur = mt.InsertionPoint
                dx = float(x) - float(cur[0])
                dy = float(y) - float(cur[1])
                if abs(dx) > 1e-9 or abs(dy) > 1e-9:
                    p_from = win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, (0.0, 0.0, 0.0))
                    p_to = win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, (dx, dy, 0.0))
                    mt.Move(p_from, p_to)
            except Exception:
                pass
            return {"success": True, "handle": getattr(mt, "Handle", None), "layer": getattr(mt, "Layer", None)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def move_all_modelspace(self, dx: float, dy: float) -> Dict[str, Any]:
        """
        Move ALL ModelSpace entities by a delta (dx, dy).
        Useful when you want the entire plan to be anchored to a specific coordinate.
        """
        if not self._ensure_active_document():
            return {"success": False, "error": "No active document. Please open a drawing first using autocad_open_drawing."}
        try:
            import pythoncom
            import win32com.client
            ms = self.doc.ModelSpace
            p_from = win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, (0.0, 0.0, 0.0))
            p_to = win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, (float(dx), float(dy), 0.0))
            moved = 0
            for i in range(ms.Count):
                try:
                    e = ms.Item(i)
                    try:
                        e.Move(p_from, p_to)
                        moved += 1
                        continue
                    except Exception:
                        # Some entity types behave better with plain tuples
                        e.Move((0.0, 0.0, 0.0), (float(dx), float(dy), 0.0))
                        moved += 1
                except Exception:
                    continue
            return {"success": True, "dx": float(dx), "dy": float(dy), "moved_entities": moved}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def move_all_entities_by(self, dx: float, dy: float) -> Dict[str, Any]:
        """Move all ModelSpace entities by a delta. Best-effort (skips entities that error)."""
        if not self._ensure_active_document():
            return {"success": False, "error": "No active document. Please open a drawing first using autocad_open_drawing."}
        try:
            ms = self.doc.ModelSpace
            moved = 0
            skipped = 0
            for i in range(ms.Count):
                try:
                    e = ms.Item(i)
                    e.Move((0.0, 0.0, 0.0), (float(dx), float(dy), 0.0))
                    moved += 1
                except Exception:
                    skipped += 1
                    continue
            return {"success": True, "dx": float(dx), "dy": float(dy), "moved": moved, "skipped": skipped}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # ------------------------------------------------------------------
    # TABLE READING UTILITIES
    # ------------------------------------------------------------------

    def read_full_table(self, handle: str) -> Dict[str, Any]:
        """
        Read all cell content from an AutoCAD TABLE object identified by handle.

        Iterates every row and column and returns the text found in each cell.
        Returns a 2-D list of strings (rows × cols).
        """
        if not self._ensure_active_document():
            return {"success": False, "error": "No active document."}
        try:
            ms = self.doc.ModelSpace
            target = None
            for i in range(ms.Count):
                e = ms.Item(i)
                if getattr(e, "Handle", None) == handle and getattr(e, "ObjectName", "") == "AcDbTable":
                    target = e
                    break
            if target is None:
                return {"success": False, "error": f"TABLE with handle {handle} not found"}

            rows = int(target.Rows)
            cols = int(target.Columns)
            grid: list = []
            for r in range(rows):
                row_data: list = []
                for c in range(cols):
                    try:
                        text = target.GetText(r, c)
                    except Exception:
                        text = ""
                    row_data.append(str(text).strip())
                grid.append(row_data)
            return {
                "success": True,
                "handle": handle,
                "rows": rows,
                "cols": cols,
                "grid": grid,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def dump_all_tables(self) -> Dict[str, Any]:
        """
        Read every TABLE object in ModelSpace and return all cell text.

        Returns a list of table dicts, each with:
          - handle, layer, rows, cols, insertion_point, grid (2-D list of cell strings)

        This is the primary tool for reading title-block metadata (owner name,
        plan number, surveyor, certification date, CRS, etc.) stored as TABLE objects.
        """
        if not self._ensure_active_document():
            return {"success": False, "error": "No active document."}
        try:
            ms = self.doc.ModelSpace
            tables = []
            for i in range(ms.Count):
                try:
                    e = ms.Item(i)
                    if getattr(e, "ObjectName", "") != "AcDbTable":
                        continue
                    handle = getattr(e, "Handle", None)
                    layer = getattr(e, "Layer", "")
                    rows = int(e.Rows)
                    cols = int(e.Columns)
                    ins = {}
                    for attr in ("InsertionPoint", "Position"):
                        try:
                            pt = getattr(e, attr, None)
                            if pt is not None:
                                ins = {"x": float(pt[0]), "y": float(pt[1])}
                                break
                        except Exception:
                            continue
                    grid: list = []
                    for r in range(rows):
                        row_data: list = []
                        for c in range(cols):
                            try:
                                text = e.GetText(r, c)
                            except Exception:
                                text = ""
                            row_data.append(str(text).strip())
                        grid.append(row_data)
                    tables.append({
                        "handle": handle,
                        "layer": layer,
                        "rows": rows,
                        "cols": cols,
                        "insertion_point": ins,
                        "grid": grid,
                    })
                except Exception:
                    continue
            return {"success": True, "count": len(tables), "tables": tables}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------
    # SMART BOUNDARY AREA (avoids mistaking border frames for plot area)
    # ------------------------------------------------------------------

    @staticmethod
    def _is_axis_aligned_rect(pts: list, tol: float = 0.01) -> bool:
        """Return True if the point list approximates an axis-aligned rectangle."""
        if len(pts) < 4:
            return False
        cleaned = [p for p in pts if len(p) >= 2]
        # Remove duplicate closing point
        if len(cleaned) >= 2 and abs(cleaned[0][0] - cleaned[-1][0]) < tol and abs(cleaned[0][1] - cleaned[-1][1]) < tol:
            cleaned = cleaned[:-1]
        if len(cleaned) != 4:
            return False
        xs = sorted(set(round(p[0], 3) for p in cleaned))
        ys = sorted(set(round(p[1], 3) for p in cleaned))
        return len(xs) == 2 and len(ys) == 2

    def calculate_boundary_area(self) -> Dict[str, Any]:
        """
        Intelligently identify and calculate the actual survey plot boundary area.

        Strategy (in order of preference):
        1. If a closed polyline exists on a layer whose name contains 'BOUNDARY'
           (case-insensitive), use that — it is almost certainly the plot outline.
        2. Otherwise look for the closed polyline coloured red (ACI colour 1),
           which is the surveying convention for a boundary 'verged in red'.
        3. Otherwise, from all remaining closed polylines, exclude those that
           form axis-aligned rectangles (those are sheet borders / interior borders)
           and pick the one with the SMALLEST area among the irregular shapes,
           which is almost always the actual land parcel on a cadastral plan.

        Returns the area and which strategy was used, so the agent can report
        the reasoning transparently.
        """
        if not self._ensure_active_document():
            return {"success": False, "error": "No active document."}

        result = self.get_all_entities()
        if not result.get("success"):
            return result

        entities = result.get("entities", [])
        candidates = []

        for ent in entities:
            etype = ent.get("type", "")
            if etype not in ("LWPOLYLINE", "POLYLINE"):
                continue
            if not ent.get("closed", False):
                continue
            area = ent.get("area")
            if not area or area <= 0:
                continue
            candidates.append(ent)

        if not candidates:
            return {"success": False, "error": "No closed polylines found in the drawing."}

        units = self._get_units()

        # Strategy 1 – layer name contains 'boundary'
        boundary_layer = [
            c for c in candidates
            if "boundary" in str(c.get("layer", "")).lower()
            and "interior" not in str(c.get("layer", "")).lower()
            and "border" not in str(c.get("layer", "")).lower()
        ]
        if boundary_layer:
            chosen = min(boundary_layer, key=lambda c: c.get("area", float("inf")))
            strategy = "layer-name contains 'BOUNDARY'"
            return self._boundary_area_result(chosen, strategy, units)

        # Strategy 2 – red polyline (colour 1 / 'red')
        red_polys = [c for c in candidates if str(c.get("color", "")).lower() in ("red", "1")]
        if red_polys:
            chosen = min(red_polys, key=lambda c: c.get("area", float("inf")))
            strategy = "red polyline (survey convention: boundary verged in red)"
            return self._boundary_area_result(chosen, strategy, units)

        # Strategy 3 – smallest non-rectangular closed polyline
        # Fetch coordinate data to test rectangularity
        non_rect = []
        for ent in candidates:
            # Exclude obvious border layers
            lyr = str(ent.get("layer", "")).upper()
            if any(kw in lyr for kw in ("BORDER", "FRAME", "INTERIOR", "SHEET", "TITLEBLOCK")):
                continue
            # Try to get coordinates for rectangularity check
            coords = ent.get("coordinates") or ent.get("vertices") or []
            is_rect = self._is_axis_aligned_rect(coords) if coords else False
            if not is_rect:
                non_rect.append(ent)

        pool = non_rect if non_rect else candidates
        # Exclude very large entities that are likely border frames
        areas = [c.get("area", 0) for c in pool]
        if areas:
            median_area = sorted(areas)[len(areas) // 2]
            pool_filtered = [c for c in pool if c.get("area", 0) <= median_area * 10]
            if pool_filtered:
                pool = pool_filtered

        chosen = min(pool, key=lambda c: c.get("area", float("inf")))
        strategy = "smallest non-rectangular closed polyline (border layers excluded)"
        return self._boundary_area_result(chosen, strategy, units)

    def _boundary_area_result(self, ent: dict, strategy: str, units: str) -> Dict[str, Any]:
        area = ent.get("area", 0.0)
        conversions = self._calculate_area_conversions(area, units)
        return {
            "success": True,
            "strategy_used": strategy,
            "layer": ent.get("layer"),
            "color": ent.get("color"),
            "handle": ent.get("handle"),
            "area_sq_units": area,
            "drawing_units": units,
            "area_sq_meters": conversions.get("sq_meters", area),
            "area_hectares": conversions.get("hectares"),
            "area_acres": conversions.get("acres"),
            "area_sq_feet": conversions.get("sq_feet"),
            "note": (
                f"Boundary identified by: {strategy}. "
                "If this is incorrect, call autocad_calculate_area(layer='<layer_name>') "
                "with the specific layer that contains the plot outline."
            ),
        }

    def _calculate_area_conversions(self, area: float, units: str) -> Dict[str, float]:
        """
        Convert area to various units.
        
        Takes the raw area in drawing units and converts to:
        - Square meters
        - Square feet
        - Hectares
        - Acres
        - Square kilometers
        
        Args:
            area: Area value in drawing units
            units: Drawing unit name (e.g., "Meters", "Feet")
            
        Returns:
            Dict with area in various units
        """
        # Conversion factors to meters
        unit_to_meters = {
            "Meters": 1.0,
            "Centimeters": 0.01,
            "Millimeters": 0.001,
            "Feet": 0.3048,
            "Inches": 0.0254,
            "Yards": 0.9144,
            "Kilometers": 1000.0,
            "Miles": 1609.34,
        }
        
        # Get the conversion factor (default to 1.0 if unknown)
        factor = unit_to_meters.get(units, 1.0)
        
        # Convert to square meters first
        # (factor^2 because area is in square units)
        sq_meters = area * (factor ** 2)
        
        # Then convert to other units
        return {
            "sq_meters": sq_meters,
            "sq_feet": sq_meters * 10.7639,
            "hectares": sq_meters / 10000,
            "acres": sq_meters / 4046.86,
            "sq_kilometers": sq_meters / 1000000,
        }


# ==============================================================================
# FALLBACK DXF PROCESSOR (using ezdxf - works without AutoCAD)
# ==============================================================================

def _locate_oda_file_converter() -> Optional[str]:
    """
    Find ODAFileConverter.exe for ezdxf DWG fallback reads.

    Checks (in order): configured ezdxf option, ODAFC / ODA_FILE_CONVERTER env,
    PATH, then common Windows install directories. Returns None when not found.
    """
    candidates: List[str] = []

    try:
        import ezdxf
        from ezdxf.addons import odafc

        configured = str(odafc.get_win_exec_path() or "").strip().strip('"')
        if configured:
            candidates.append(configured)
        # Also accept a direct options key if present.
        try:
            opt = str(ezdxf.options.get("odafc-addon", "win_exec_path") or "").strip().strip('"')
            if opt:
                candidates.append(opt)
        except Exception:
            pass
    except Exception:
        pass

    for env_key in ("ODAFC", "ODA_FILE_CONVERTER", "ODA_FILE_CONVERTER_EXE"):
        val = str(os.environ.get(env_key) or "").strip().strip('"')
        if val:
            candidates.append(val)

    try:
        import shutil

        which = shutil.which("ODAFileConverter") or shutil.which("ODAFileConverter.exe")
        if which:
            candidates.append(which)
    except Exception:
        pass

    # Common Windows install roots (versioned folders under ODA\).
    roots = []
    for env_key in ("ProgramFiles", "ProgramFiles(x86)", "ProgramW6432"):
        root = os.environ.get(env_key)
        if root:
            roots.append(Path(root) / "ODA")
            roots.append(Path(root))
    for root in roots:
        try:
            if not root.is_dir():
                continue
            # Exact well-known path patterns
            for pattern in (
                "ODAFileConverter*/ODAFileConverter.exe",
                "ODA/ODAFileConverter*/ODAFileConverter.exe",
            ):
                for hit in root.glob(pattern):
                    candidates.append(str(hit))
            direct = root / "ODAFileConverter.exe"
            if direct.is_file():
                candidates.append(str(direct))
        except Exception:
            continue

    seen: set[str] = set()
    for raw in candidates:
        try:
            p = Path(raw).expanduser().resolve()
        except Exception:
            continue
        key = str(p).lower()
        if key in seen:
            continue
        seen.add(key)
        if p.is_file():
            return str(p)
    return None


class DXFProcessor:
    """
    Fallback DXF/DWG processor using ezdxf library.
    
    This processor works WITHOUT AutoCAD installed. It can read DXF files
    directly and extract entities, text, and calculate areas.
    
    Note: DWG files may have limited support depending on the version.
    For full DWG support, AutoCAD is required.
    
    Usage:
        >>> processor = DXFProcessor()
        >>> result = processor.open_drawing("survey.dxf")
        >>> texts = processor.get_all_text()
    """
    
    def __init__(self):
        """Initialize the DXF processor."""
        self.doc = None
        self.modelspace = None
        self._ezdxf_available = False
        
        try:
            import ezdxf
            self._ezdxf_available = True
            logger.info("ezdxf fallback processor available")
        except ImportError:
            logger.warning("ezdxf not installed. Install with: pip install ezdxf")
    
    @property
    def is_available(self) -> bool:
        """Check if ezdxf is available."""
        return self._ezdxf_available
    
    def open_drawing(self, file_path: str) -> Dict[str, Any]:
        """
        Open a DXF file using ezdxf.
        
        Args:
            file_path: Path to .dxf file
            
        Returns:
            Dict with success status and file info
        """
        if not self._ezdxf_available:
            return {"success": False, "error": "ezdxf not installed"}
        
        import ezdxf
        
        file_path = Path(file_path).resolve()
        if not file_path.exists():
            return {"success": False, "error": f"File not found: {file_path}"}
        
        # Check file extension
        ext = file_path.suffix.lower()
        if ext not in ['.dxf', '.dwg']:
            return {"success": False, "error": f"Unsupported file type: {ext}"}
        
        try:
            if ext == '.dwg':
                # ezdxf DWG support requires ODA File Converter on PATH or a known install dir.
                try:
                    from ezdxf.addons import odafc
                except ImportError:
                    return {
                        "success": False,
                        "error": "DWG files require ODA File Converter. Please use DXF format or install AutoCAD.",
                    }
                oda_exe = _locate_oda_file_converter()
                if oda_exe:
                    try:
                        ezdxf.options.set("odafc-addon", "win_exec_path", oda_exe)
                        logger.info("Configured ODA File Converter: %s", oda_exe)
                    except Exception as cfg_exc:
                        logger.debug("Could not set odafc win_exec_path: %s", cfg_exc)
                if not odafc.is_installed():
                    return {
                        "success": False,
                        "error": (
                            "Could not find ODAFileConverter. Install from "
                            "https://www.opendesign.com/guestfiles/oda_file_converter "
                            "or open the DWG with AutoCAD (preferred)."
                        ),
                    }
                self.doc = odafc.readfile(str(file_path))
                logger.info("Opened DWG file via ODA File Converter")
            else:
                self.doc = ezdxf.readfile(str(file_path))
            
            self.modelspace = self.doc.modelspace()
            
            # Gather info
            layers = [layer.dxf.name for layer in self.doc.layers]
            entity_count = len(list(self.modelspace))
            
            return {
                "success": True,
                "file_path": str(file_path),
                "drawing_name": file_path.name,
                "layers": layers,
                "entity_count": entity_count,
                "processor": "ezdxf (fallback - AutoCAD not available)"
            }
            
        except Exception as e:
            logger.error(f"Failed to open file with ezdxf: {e}")
            return {"success": False, "error": str(e)}
    
    def get_all_text(self) -> Dict[str, Any]:
        """Extract all text entities from the drawing."""
        if not self.doc or not self.modelspace:
            return {"success": False, "error": "No drawing open"}
        
        texts = []
        for entity in self.modelspace:
            if entity.dxftype() in ['TEXT', 'MTEXT']:
                try:
                    content = ""
                    if entity.dxftype() == 'TEXT':
                        content = entity.dxf.text
                    elif entity.dxftype() == 'MTEXT':
                        content = entity.text
                    
                    if content and content.strip():
                        texts.append({
                            "type": entity.dxftype(),
                            "content": content.strip(),
                            "layer": entity.dxf.layer,
                            "color": entity.dxf.color,
                        })
                except Exception as e:
                    logger.debug(f"Error reading text entity: {e}")
                    continue
        
        return {
            "success": True,
            "text_count": len(texts),
            "texts": texts
        }
    
    def calculate_area(
        self, 
        layer: Optional[str] = None, 
        color: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate area of closed polylines.
        
        Args:
            layer: Filter by layer name
            color: Filter by color (limited support in DXF)
            
        Returns:
            Dict with area calculations
        """
        if not self.doc or not self.modelspace:
            return {"success": False, "error": "No drawing open"}
        
        areas = []
        total_area = 0.0
        
        # Color name to ACI mapping
        color_to_aci = {
            "red": 1, "yellow": 2, "green": 3, "cyan": 4,
            "blue": 5, "magenta": 6, "white": 7
        }
        target_color = color_to_aci.get(color.lower()) if color else None
        
        for entity in self.modelspace:
            try:
                # Only process closed polylines
                if entity.dxftype() not in ['LWPOLYLINE', 'POLYLINE', 'CIRCLE']:
                    continue
                
                # Layer filter
                if layer and entity.dxf.layer.lower() != layer.lower():
                    continue
                
                # Color filter
                if target_color and entity.dxf.color != target_color:
                    continue
                
                # Check if closed (for polylines)
                if entity.dxftype() in ['LWPOLYLINE', 'POLYLINE']:
                    if not entity.is_closed:
                        continue
                
                # Calculate area
                area = 0.0
                if entity.dxftype() == 'CIRCLE':
                    area = math.pi * (entity.dxf.radius ** 2)
                elif entity.dxftype() in ['LWPOLYLINE', 'POLYLINE']:
                    # Use shoelace formula for polygon area
                    points = list(entity.get_points())
                    n = len(points)
                    if n >= 3:
                        area = 0.0
                        for i in range(n):
                            j = (i + 1) % n
                            area += points[i][0] * points[j][1]
                            area -= points[j][0] * points[i][1]
                        area = abs(area) / 2.0
                
                if area > 0:
                    areas.append({
                        "type": entity.dxftype(),
                        "layer": entity.dxf.layer,
                        "area_sq_units": area,
                    })
                    total_area += area
                    
            except Exception as e:
                logger.debug(f"Error processing entity for area: {e}")
                continue
        
        return {
            "success": True,
            "shapes_found": len(areas),
            "total_area_sq_units": total_area,
            "individual_areas": areas,
            "note": "Areas calculated using ezdxf (fallback). For precise results, use AutoCAD."
        }


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = ["AutoCADProcessor", "DXFProcessor", "ENTITY_TYPES", "ACI_COLORS"]
