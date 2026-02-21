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
        
        # Optionally connect immediately
        if auto_connect:
            self.connect()
    
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
            
            # List of AutoCAD ProgIDs to try (different versions)
            # Order: Generic first, then specific versions (newer to older)
            autocad_progids = [
                "AutoCAD.Application",           # Generic - uses default version
                "AutoCAD.Application.25",        # AutoCAD 2025
                "AutoCAD.Application.24",        # AutoCAD 2024
                "AutoCAD.Application.23",        # AutoCAD 2022/2023
                "AutoCAD.Application.22",        # AutoCAD 2021
                "AutoCAD.Application.21",        # AutoCAD 2020
                "AutoCAD.Application.20",        # AutoCAD 2019
                "AutoCAD.Application.19",        # AutoCAD 2018
                "AutoCAD.Application.18",        # AutoCAD 2017
            ]
            
            # First, try to connect to an already running instance
            # GetActiveObject finds a running COM server
            connected = False
            for progid in autocad_progids:
                try:
                    self.acad = win32com.client.GetActiveObject(progid)
                    logger.info(f"Connected to running AutoCAD instance via {progid}")
                    connected = True
                    break
                except Exception:
                    continue
            
            if not connected:
                # No running instance - try to start one
                logger.info("No running AutoCAD instance found. Attempting to start...")
                
                for progid in autocad_progids:
                    try:
                        # Try DispatchEx first (creates new instance)
                        try:
                            self.acad = win32com.client.DispatchEx(progid)
                            logger.info(f"Started new AutoCAD instance via DispatchEx ({progid})")
                        except Exception:
                            self.acad = win32com.client.Dispatch(progid)
                            logger.info(f"Started new AutoCAD instance via Dispatch ({progid})")
                            
                        # Make AutoCAD visible
                        self.acad.Visible = True
                        
                        # Give AutoCAD time to initialize
                        logger.info("Waiting for AutoCAD to initialize...")
                        time.sleep(3)
                        
                        connected = True
                        break
                        
                    except Exception as e:
                        logger.debug(f"Could not connect via {progid}: {e}")
                        continue
                
                if not connected:
                    # Provide detailed troubleshooting info
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
        
        try:
            # COM calls can be intermittently rejected when AutoCAD is busy/modal.
            # Wrap critical COM operations with a small retry.
            def _com_retry(fn, attempts: int = 10, base_sleep: float = 0.2):
                try:
                    import pywintypes  # type: ignore
                except Exception:
                    pywintypes = None
                last = None
                for k in range(attempts):
                    try:
                        return fn()
                    except Exception as ex:
                        last = ex
                        # Retry specifically for "Call was rejected by callee"
                        try:
                            if pywintypes is not None and isinstance(ex, pywintypes.com_error):
                                hr = int(ex.hresult) if hasattr(ex, "hresult") else None
                                if hr == -2147418111:
                                    time.sleep(base_sleep * (k + 1))
                                    continue
                        except Exception:
                            pass
                        # When AutoCAD is busy, attribute resolution can fail with AttributeError like "<unknown>.Open".
                        # Treat that as retryable too.
                        try:
                            if isinstance(ex, AttributeError) and any(s in str(ex) for s in [".Open", ".Count", ".Item"]):
                                time.sleep(base_sleep * (k + 1))
                                continue
                        except Exception:
                            pass
                        # Best-effort retry on generic COM hiccups as well
                        time.sleep(base_sleep * (k + 1))
                        continue
                raise last if last is not None else Exception("COM retry failed")

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
            # This prevents AutoCAD from opening a read-only copy
            existing_doc = None
            try:
                for doc in _iter_docs():
                    try:
                        doc_path = Path(doc.FullName).resolve()
                        if doc_path == file_path or doc.Name.lower() == file_name:
                            existing_doc = doc
                            logger.info(f"Drawing already open: {doc.Name}")
                            break
                    except Exception:
                        continue
            except Exception as e:
                logger.debug(f"Could not enumerate documents: {e}")
            
            # ------------------------------------------------------------------
            # Step 2: Activate existing or open new
            # ------------------------------------------------------------------
            if existing_doc:
                # Activate the existing document
                _com_retry(lambda: existing_doc.Activate(), attempts=8)
                self.doc = existing_doc
                logger.info(f"Activated existing document: {self.doc.Name}")
            else:
                # Close any existing read-only copies of the same file first
                try:
                    for doc in _iter_docs():
                        try:
                            if doc.Name.lower() == file_name and doc.ReadOnly:
                                logger.info(f"Closing read-only copy: {doc.Name}")
                                _com_retry(lambda: doc.Close(False), attempts=6)  # False = don't save
                        except Exception:
                            continue
                except Exception:
                    pass
                
                # Open the drawing fresh
                # The second parameter (False) means "don't open read-only"
                try:
                    docs = _com_retry(lambda: self.acad.Documents, attempts=12)
                    self.doc = _com_retry(lambda: getattr(docs, "Open")(full_path_str, read_only), attempts=12)
                except Exception:
                    # Fallback: try opening without the read-only parameter
                    docs = _com_retry(lambda: self.acad.Documents, attempts=12)
                    self.doc = _com_retry(lambda: getattr(docs, "Open")(full_path_str), attempts=12)
                
                logger.info(f"Opened new document: {self.doc.Name}")
            
            # ------------------------------------------------------------------
            # Step 3: Ensure document is activated
            # ------------------------------------------------------------------
            # Explicitly activate the document to ensure it's the active one
            try:
                if self.doc.Name.lower() != file_name:
                    # Find and activate the correct document
                    for doc in _iter_docs():
                        try:
                            if doc.Name.lower() == file_name:
                                _com_retry(lambda: doc.Activate(), attempts=8)
                                self.doc = doc
                                logger.info(f"Activated document: {self.doc.Name}")
                                break
                        except Exception:
                            continue
                else:
                    # Document is already correct, but ensure it's active
                    _com_retry(lambda: self.doc.Activate(), attempts=8)
                    time.sleep(0.2)  # Brief pause for activation
            except Exception as e:
                logger.warning(f"Could not activate document: {e}")
            
            # ------------------------------------------------------------------
            # Step 4: Wait for document to fully load and verify it's accessible
            # ------------------------------------------------------------------
            # AutoCAD needs time to parse and render complex drawings
            max_wait = 25
            wait_interval = 0.5
            waited = 0
            doc_ready = False
            
            while waited < max_wait:
                try:
                    # Refresh document reference to ensure we have the active one
                    self.doc = _com_retry(lambda: self.acad.ActiveDocument, attempts=8)
                    
                    # Verify it's the correct document
                    if self.doc.Name.lower() != file_name:
                        # Try to find and activate the correct document
                        for doc in _iter_docs():
                            try:
                                if doc.Name.lower() == file_name:
                                    _com_retry(lambda: doc.Activate(), attempts=8)
                                    time.sleep(0.3)
                                    self.doc = _com_retry(lambda: self.acad.ActiveDocument, attempts=8)
                                    if self.doc.Name.lower() == file_name:
                                        break
                            except Exception:
                                continue
                    
                    # Try to access modelspace - this will fail if doc isn't ready
                    _ = _com_retry(lambda: self.doc.ModelSpace.Count, attempts=8)
                    
                    # Try to access document name to ensure it's fully loaded
                    _ = self.doc.Name
                    
                    # If we get here, document is ready
                    doc_ready = True
                    break
                except Exception as e:
                    logger.debug(f"Waiting for document to load... ({waited:.1f}s) - {e}")
                    time.sleep(wait_interval)
                    waited += wait_interval
            
            if not doc_ready:
                logger.error(f"Document did not become ready after {max_wait} seconds")
                return {
                    "success": False,
                    "error": f"Document opened but did not become ready after {max_wait} seconds. The file may be corrupted or AutoCAD may need more time."
                }
            
            # Give a bit more time for the UI to settle
            time.sleep(0.5)
            
            # ------------------------------------------------------------------
            # Step 5: Final verification that document is accessible
            # ------------------------------------------------------------------
            try:
                # Final refresh of document reference
                self.doc = self.acad.ActiveDocument
                
                # Verify it's still the correct document
                if self.doc.Name.lower() != file_name:
                    logger.warning(f"Active document mismatch: expected {file_name}, got {self.doc.Name}")
                    # Try one more time to find and activate
                    for doc in self.acad.Documents:
                        try:
                            if doc.Name.lower() == file_name:
                                doc.Activate()
                                time.sleep(0.3)
                                self.doc = self.acad.ActiveDocument
                                break
                        except Exception:
                            continue
                
                # Final verification - try to access document properties
                doc_name = self.doc.Name
                entity_count = self._count_entities()
                is_readonly = getattr(self.doc, 'ReadOnly', False)
                
                if is_readonly:
                    logger.warning("Document opened in READ-ONLY mode. Some operations may be limited.")
                
                result = {
                    "success": True,
                    "file_path": str(file_path),
                    "drawing_name": doc_name,
                    "units": self._get_units(),
                    "layers": self._get_layers(),
                    "entity_count": entity_count,
                    "read_only": is_readonly,
                }
                
                logger.info(f"Document ready: {doc_name} ({entity_count} entities)")
                return result
                
            except Exception as e:
                logger.error(f"Document opened but not accessible: {e}")
                return {
                    "success": False, 
                    "error": f"Document opened but not accessible: {e}"
                }
            
        except Exception as e:
            logger.error(f"Failed to open drawing: {e}")
            return {"success": False, "error": str(e)}
    
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
        if not self._ensure_active_document():
            return {"success": False, "error": "No active document. Please open a drawing first using autocad_open_drawing."}
        
        try:
            # SendCommand sends the command string to AutoCAD
            # The newline (\n) is like pressing Enter
            self.doc.SendCommand(command + "\n")
            
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
                logger.warning("Could not connect to AutoCAD")
                return False
        
        # Verify connection is still valid
        try:
            _ = self.acad.Name
        except Exception:
            logger.warning("Connection lost, attempting to reconnect...")
            if not self.connect():
                return False
        
        # Check if current doc reference is valid and accessible
        try:
            if self.doc:
                # Test basic access
                _ = self.doc.Name
                # Test ModelSpace access - this is the real test
                _ = self.doc.ModelSpace.Count
                return True
        except Exception as e:
            logger.debug(f"Current doc reference is stale or inaccessible: {e}")
            self.doc = None  # Clear stale reference
        
        # Try to get the active document from AutoCAD
        try:
            # Check if there are any open documents
            doc_count = self.acad.Documents.Count
            if doc_count == 0:
                logger.warning("No documents are open in AutoCAD")
                return False
            
            # Get the active document
            try:
                self.doc = self.acad.ActiveDocument
            except Exception:
                # If ActiveDocument fails, try to get the first document
                if doc_count > 0:
                    self.doc = self.acad.Documents.Item(0)
                    self.doc.Activate()
                    time.sleep(0.3)
                    self.doc = self.acad.ActiveDocument
            
            # Verify it's accessible
            _ = self.doc.Name
            _ = self.doc.ModelSpace.Count
            
            logger.info(f"Document ready: {self.doc.Name}")
            return True
        except Exception as e:
            logger.warning(f"Could not access active document: {e}")
            # Try to find any open document as a fallback
            try:
                if self.acad.Documents.Count > 0:
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
                logger.warning(f"Fallback document activation failed: {fallback_error}")
            
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
            modelspace = self.doc.ModelSpace
            for i in range(modelspace.Count):
                try:
                    e = modelspace.Item(i)
                    if getattr(e, "ObjectName", "") != "AcDbTable":
                        continue
                    if layer and str(getattr(e, "Layer", "")).lower() != layer.lower():
                        continue
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
            modelspace = self.doc.ModelSpace
            target = None
            for i in range(modelspace.Count):
                e = modelspace.Item(i)
                if getattr(e, "Handle", None) == handle and getattr(e, "ObjectName", "") == "AcDbTable":
                    target = e
                    break
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
        if not self._ensure_active_document():
            return {"success": False, "error": "No active document. Please open a drawing first using autocad_open_drawing."}
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

            try:
                target.SetText(int(row), int(col), str(text))
            except Exception as ex:
                return {"success": False, "error": f"Could not set cell ({row},{col}): {ex}"}

            return {"success": True, "handle": handle, "row": int(row), "col": int(col), "text": str(text)}
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
            ms = self.doc.ModelSpace
            for i in range(ms.Count):
                try:
                    e = ms.Item(i)
                    obj = getattr(e, "ObjectName", "")
                    if "Text" not in obj and "MText" not in obj:
                        continue
                    lyr = str(getattr(e, "Layer", "")).upper()
                    if lyr not in want:
                        continue
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
            modelspace = self.doc.ModelSpace
            for i in range(modelspace.Count):
                try:
                    e = modelspace.Item(i)
                    obj = getattr(e, "ObjectName", "")
                    if "BlockReference" not in obj and ENTITY_TYPES.get(obj) != "INSERT":
                        continue
                    if layer and str(getattr(e, "Layer", "")).lower() != layer.lower():
                        continue
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
        if not self._ensure_active_document():
            return {"success": False, "error": "No active document. Please open a drawing first using autocad_open_drawing."}

        def _do_delete() -> Dict[str, Any]:
            deleted = 0
            ms = self.doc.ModelSpace
            for i in range(ms.Count - 1, -1, -1):
                try:
                    e = ms.Item(i)
                    if str(getattr(e, "Layer", "")).lower() != str(layer).lower():
                        continue
                    if entity_object_names:
                        if str(getattr(e, "ObjectName", "")) not in entity_object_names:
                            continue
                    e.Delete()
                    deleted += 1
                except Exception:
                    continue
            return {"success": True, "layer": layer, "deleted": deleted}

        try:
            return _do_delete()
        except Exception as e:
            try:
                time.sleep(0.3)
                return _do_delete()
            except Exception as e2:
                return {"success": False, "error": str(e2)}

    def create_lwpolyline(self, points_xy: List[Dict[str, float]], layer: str, closed: bool = True, linetype_scale: Optional[float] = None) -> Dict[str, Any]:
        """Create a lightweight polyline in ModelSpace."""
        if not self._ensure_active_document():
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
    ) -> Dict[str, Any]:
        """Insert a block reference into ModelSpace."""
        if not self._ensure_active_document():
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
            ms = self.doc.ModelSpace
            want_layers = {str(l).upper() for l in (layers or [])} if layers else None
            want_objs = set(object_names or []) if object_names else None
            bn_sub = (block_name_contains or "").upper().strip() or None

            best = None  # (area, minx, miny, maxx, maxy, handle)
            agg = None   # (minx, miny, maxx, maxy)
            matched = 0

            for i in range(ms.Count):
                try:
                    e = ms.Item(i)
                    lyr = str(getattr(e, "Layer", "")).upper()
                    if want_layers is not None and lyr not in want_layers:
                        continue
                    on = str(getattr(e, "ObjectName", ""))
                    if want_objs is not None and on not in want_objs:
                        continue
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
                        best = (area, minx, miny, maxx, maxy, str(getattr(e, "Handle", "")))
                except Exception:
                    continue

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
            ms = self.doc.ModelSpace
            want = {str(l).upper() for l in (layers or [])}
            p_from = win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, (0.0, 0.0, 0.0))
            p_to = win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, (float(dx), float(dy), 0.0))
            moved = 0
            for i in range(ms.Count):
                try:
                    e = ms.Item(i)
                    if str(getattr(e, "Layer", "")).upper() not in want:
                        continue
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
        if not self._ensure_active_document():
            return {"success": False, "error": "No active document. Please open a drawing first using autocad_open_drawing."}
        if not layers or float(scale_factor) <= 0.0:
            return {"success": False, "error": "layers must be non-empty and scale_factor must be positive"}
        try:
            import pythoncom
            import win32com.client

            ms = self.doc.ModelSpace
            want = {str(l).upper() for l in layers}
            base_pt = win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, (float(base_x), float(base_y), 0.0))
            sf = float(scale_factor)
            scaled = 0
            for i in range(ms.Count):
                try:
                    e = ms.Item(i)
                    if str(getattr(e, "Layer", "")).upper() not in want:
                        continue
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
    ) -> Dict[str, Any]:
        """Add an MTEXT entity to ModelSpace."""
        if not self._ensure_active_document():
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
                # ezdxf has limited DWG support via odafc (if installed)
                try:
                    from ezdxf.addons import odafc
                    self.doc = odafc.readfile(str(file_path))
                    logger.info("Opened DWG file via ODA File Converter")
                except ImportError:
                    return {
                        "success": False, 
                        "error": "DWG files require ODA File Converter. Please use DXF format or install AutoCAD."
                    }
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
