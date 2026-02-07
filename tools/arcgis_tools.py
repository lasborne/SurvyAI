"""
================================================================================
ArcGIS Pro Interface for Complex Geospatial Tasks
================================================================================

This module provides an interface to control ArcGIS Pro programmatically,
enabling project management, coordinate system configuration, and advanced
geospatial analysis operations.

CAPABILITIES:
-------------
1. Project Management:
   - Launch ArcGIS Pro
   - Create new projects with specified coordinate systems
   - Open existing projects
   - Save and close projects

2. Coordinate System Management:
   - Set project coordinate systems (geographic or projected)
   - Support for common CRS like WGS 84, UTM zones, State Plane

3. Analysis Operations (requires arcpy):
   - IDW volume calculation
   - Cut/fill analysis
   - Surface analysis

USAGE MODES:
------------
1. Direct arcpy mode: When running from ArcGIS Pro's Python environment
2. Subprocess mode: Launch ArcGIS Pro and execute scripts via subprocess
3. ArcGIS Python API mode: Using the ArcGIS API for Python (arcgis package)

Author: SurvyAI Team
License: MIT
================================================================================
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import sys
import os
import subprocess
import json
import tempfile
import time
import datetime
from config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


# ==============================================================================
# COMMON COORDINATE SYSTEMS
# ==============================================================================

# Dictionary of common coordinate systems with their WKID (Well-Known ID) codes
# These are used by ArcGIS Pro to define spatial references
COORDINATE_SYSTEMS = {
    # Geographic Coordinate Systems (lat/lon)
    "WGS84": {"wkid": 4326, "name": "WGS 1984", "type": "geographic"},
    "WGS 84": {"wkid": 4326, "name": "WGS 1984", "type": "geographic"},
    "NAD83": {"wkid": 4269, "name": "NAD 1983", "type": "geographic"},
    "NAD 83": {"wkid": 4269, "name": "NAD 1983", "type": "geographic"},
    "NAD27": {"wkid": 4267, "name": "NAD 1927", "type": "geographic"},
    
    # UTM Zones - WGS 84 (Northern Hemisphere)
    "UTM Zone 1N": {"wkid": 32601, "name": "WGS 84 / UTM zone 1N", "type": "projected"},
    "UTM Zone 10N": {"wkid": 32610, "name": "WGS 84 / UTM zone 10N", "type": "projected"},
    "UTM Zone 11N": {"wkid": 32611, "name": "WGS 84 / UTM zone 11N", "type": "projected"},
    "UTM Zone 12N": {"wkid": 32612, "name": "WGS 84 / UTM zone 12N", "type": "projected"},
    "UTM Zone 13N": {"wkid": 32613, "name": "WGS 84 / UTM zone 13N", "type": "projected"},
    "UTM Zone 14N": {"wkid": 32614, "name": "WGS 84 / UTM zone 14N", "type": "projected"},
    "UTM Zone 15N": {"wkid": 32615, "name": "WGS 84 / UTM zone 15N", "type": "projected"},
    "UTM Zone 16N": {"wkid": 32616, "name": "WGS 84 / UTM zone 16N", "type": "projected"},
    "UTM Zone 17N": {"wkid": 32617, "name": "WGS 84 / UTM zone 17N", "type": "projected"},
    "UTM Zone 18N": {"wkid": 32618, "name": "WGS 84 / UTM zone 18N", "type": "projected"},
    "UTM Zone 19N": {"wkid": 32619, "name": "WGS 84 / UTM zone 19N", "type": "projected"},
    "UTM Zone 20N": {"wkid": 32620, "name": "WGS 84 / UTM zone 20N", "type": "projected"},
    "UTM Zone 29N": {"wkid": 32629, "name": "WGS 84 / UTM zone 29N", "type": "projected"},
    "UTM Zone 30N": {"wkid": 32630, "name": "WGS 84 / UTM zone 30N", "type": "projected"},
    "UTM Zone 31N": {"wkid": 32631, "name": "WGS 84 / UTM zone 31N", "type": "projected"},
    "UTM Zone 32N": {"wkid": 32632, "name": "WGS 84 / UTM zone 32N", "type": "projected"},
    "UTM Zone 33N": {"wkid": 32633, "name": "WGS 84 / UTM zone 33N", "type": "projected"},
    "UTM Zone 34N": {"wkid": 32634, "name": "WGS 84 / UTM zone 34N", "type": "projected"},
    "UTM Zone 35N": {"wkid": 32635, "name": "WGS 84 / UTM zone 35N", "type": "projected"},
    "UTM Zone 36N": {"wkid": 32636, "name": "WGS 84 / UTM zone 36N", "type": "projected"},
    
    # UTM Zones - WGS 84 (Southern Hemisphere)
    "UTM Zone 32S": {"wkid": 32732, "name": "WGS 84 / UTM zone 32S", "type": "projected"},
    "UTM Zone 33S": {"wkid": 32733, "name": "WGS 84 / UTM zone 33S", "type": "projected"},
    "UTM Zone 34S": {"wkid": 32734, "name": "WGS 84 / UTM zone 34S", "type": "projected"},
    "UTM Zone 35S": {"wkid": 32735, "name": "WGS 84 / UTM zone 35S", "type": "projected"},
    "UTM Zone 36S": {"wkid": 32736, "name": "WGS 84 / UTM zone 36S", "type": "projected"},
    
    # Web Mercator (used by web maps)
    "Web Mercator": {"wkid": 3857, "name": "WGS 84 / Pseudo-Mercator", "type": "projected"},
    
    # British National Grid
    "British National Grid": {"wkid": 27700, "name": "OSGB 1936 / British National Grid", "type": "projected"},
    "OSGB36": {"wkid": 27700, "name": "OSGB 1936 / British National Grid", "type": "projected"},

    # Nigeria (Minna Datum) - National Transverse Mercator belts (EPSG)
    # NOTE: ArcGIS uses "WKID" factory codes which commonly match EPSG codes for these CRSs.
    "Minna / Nigeria West Belt": {"wkid": 26391, "name": "Minna / Nigeria West Belt", "type": "projected"},
    "Minna / Nigeria Mid Belt": {"wkid": 26392, "name": "Minna / Nigeria Mid Belt", "type": "projected"},
    "Minna / Nigeria East Belt": {"wkid": 26393, "name": "Minna / Nigeria East Belt", "type": "projected"},

    # Common aliases users type
    "Nigeria Minna West Belt": {"wkid": 26391, "name": "Minna / Nigeria West Belt", "type": "projected"},
    "Nigeria Minna Mid Belt": {"wkid": 26392, "name": "Minna / Nigeria Mid Belt", "type": "projected"},
    "Nigeria Minna Mid-Belt": {"wkid": 26392, "name": "Minna / Nigeria Mid Belt", "type": "projected"},
    "Nigeria Minna East Belt": {"wkid": 26393, "name": "Minna / Nigeria East Belt", "type": "projected"},
    "Minna NTM West Belt": {"wkid": 26391, "name": "Minna / Nigeria West Belt", "type": "projected"},
    "Minna NTM Mid Belt": {"wkid": 26392, "name": "Minna / Nigeria Mid Belt", "type": "projected"},
    "Minna NTM Mid-Belt": {"wkid": 26392, "name": "Minna / Nigeria Mid Belt", "type": "projected"},
    "Minna NTM East Belt": {"wkid": 26393, "name": "Minna / Nigeria East Belt", "type": "projected"},
}


def parse_coordinate_system(crs_input: str) -> Dict[str, Any]:
    """
    Parse a coordinate system string and return its properties.
    
    Supports:
    - Named systems: "WGS84", "UTM Zone 32N", "British National Grid"
    - EPSG codes: "EPSG:4326", "EPSG:32632"
    - WKID numbers: "4326", "32632"
    
    Args:
        crs_input: Coordinate system identifier
        
    Returns:
        Dictionary with wkid, name, and type
    """
    crs_input = crs_input.strip()
    
    # Check for direct WKID lookup
    for key, value in COORDINATE_SYSTEMS.items():
        if key.lower() == crs_input.lower():
            return value
    
    # Check for EPSG: prefix
    if crs_input.upper().startswith("EPSG:"):
        try:
            wkid = int(crs_input.split(":")[1])
            return {"wkid": wkid, "name": f"EPSG:{wkid}", "type": "unknown"}
        except ValueError:
            pass
    
    # Check for numeric WKID
    try:
        wkid = int(crs_input)
        return {"wkid": wkid, "name": f"WKID:{wkid}", "type": "unknown"}
    except ValueError:
        pass
    
    # Try to parse UTM zone format (e.g., "UTM 32N", "UTM32N", "32N")
    import re
    utm_match = re.match(r"(?:UTM\s*)?(\d{1,2})([NS])", crs_input.upper())
    if utm_match:
        zone = int(utm_match.group(1))
        hemisphere = utm_match.group(2)
        if hemisphere == 'N':
            wkid = 32600 + zone
        else:
            wkid = 32700 + zone
        return {
            "wkid": wkid, 
            "name": f"WGS 84 / UTM zone {zone}{hemisphere}", 
            "type": "projected"
        }
    
    # Try pyproj as a last-resort resolver for named CRSs (e.g., "Minna / Nigeria Mid Belt")
    # This helps convert an authoritative EPSG code into a WKID we can pass into arcpy.SpatialReference.
    try:
        from pyproj import CRS

        crs_obj = CRS.from_user_input(crs_input)
        epsg = crs_obj.to_epsg()
        if epsg:
            return {"wkid": int(epsg), "name": crs_obj.name or f"EPSG:{epsg}", "type": "unknown"}
        auth = crs_obj.to_authority()
        if auth and auth[0].upper() == "EPSG":
            return {"wkid": int(auth[1]), "name": crs_obj.name or f"EPSG:{auth[1]}", "type": "unknown"}
    except Exception:
        pass

    # Default: return as-is with unknown type
    return {"wkid": None, "name": crs_input, "type": "unknown"}


class ArcGISProcessor:
    """
    Interface with ArcGIS Pro for advanced geospatial analysis and project management.
    
    This class provides multiple methods to interact with ArcGIS Pro:
    1. Direct arcpy access (when running in ArcGIS Pro's Python environment)
    2. Subprocess-based execution (for launching ArcGIS Pro and running scripts)
    3. ArcGIS API for Python (arcgis package)
    
    Capabilities:
    - Create and manage ArcGIS Pro projects
    - Set coordinate systems for maps
    - Perform geospatial analysis (IDW, cut/fill, etc.)
    - Execute geoprocessing tools
    """
    
    # Common ArcGIS Pro installation paths
    ARCGIS_COMMON_PATHS = [
        r"C:\Program Files\ArcGIS\Pro",
        r"C:\ArcGIS\Pro",
        r"C:\Program Files (x86)\ArcGIS\Pro",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\ArcGIS\Pro"),
        os.path.expandvars(r"%PROGRAMFILES%\ArcGIS\Pro"),
    ]
    
    def __init__(self):
        """Initialize the ArcGIS processor."""
        self.settings = get_settings()
        self.arcpy = None
        self.arcgis_pro_path = None
        self.arcgis_python_path = None
        self.current_project = None
        # Prevent runaway duplicate ArcGIS Pro launches (common when multiple tools chain together)
        self._last_launch_project: Optional[str] = None
        self._last_launch_ts: float = 0.0
        self._initialize_arcpy()
    
    def _find_arcgis_pro(self) -> Optional[Path]:
        """
        Find ArcGIS Pro installation path.
        
        Checks:
        1. User-configured path in settings (handles both Pro root and Python env paths)
        2. Registry entries
        3. Common installation locations
        """
        # Helper to find the Pro root from a given path
        def find_pro_root(path: Path) -> Optional[Path]:
            """
            Given a path, try to find the ArcGIS Pro root directory.
            Handles cases where the path points to:
            - The Pro root directly (C:\...\ArcGIS\Pro)
            - The Python environment (C:\...\ArcGIS\Pro\bin\Python\envs\arcgispro-py3)
            - Some other subdirectory
            """
            # Check if this is already the Pro root (has bin/ArcGISPro.exe)
            if (path / "bin" / "ArcGISPro.exe").exists():
                return path
            
            # Walk up the directory tree to find Pro root
            current = path
            for _ in range(10):  # Limit depth to avoid infinite loops
                parent = current.parent
                if parent == current:  # Reached filesystem root
                    break
                if (parent / "bin" / "ArcGISPro.exe").exists():
                    return parent
                # Check if "Pro" directory exists at parent level
                pro_dir = parent / "Pro"
                if pro_dir.exists() and (pro_dir / "bin" / "ArcGISPro.exe").exists():
                    return pro_dir
                current = parent
            
            return None
        
        # 1. Check user-configured path
        if self.settings.arcgis_pro_path:
            user_path = Path(self.settings.arcgis_pro_path)
            if user_path.exists():
                # Try to find the actual Pro root from this path
                pro_root = find_pro_root(user_path)
                if pro_root:
                    logger.debug(f"Found ArcGIS Pro root from user path: {pro_root}")
                    return pro_root
                # If we can't find the exe, still return the path for other uses
                logger.debug(f"Using user-configured ArcGIS Pro path: {user_path}")
                return user_path
        
        # 2. Try registry lookup (Windows)
        try:
            import winreg
            
            # Check multiple registry locations
            registry_paths = [
                (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\ESRI\ArcGISPro"),
                (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\ESRI\ArcGISPro"),
                (winreg.HKEY_CURRENT_USER, r"SOFTWARE\ESRI\ArcGISPro"),
            ]
            
            for hkey, subkey in registry_paths:
                try:
                    with winreg.OpenKey(hkey, subkey) as key:
                        install_dir, _ = winreg.QueryValueEx(key, "InstallDir")
                        if install_dir and Path(install_dir).exists():
                            logger.debug(f"Found ArcGIS Pro in registry: {install_dir}")
                            return Path(install_dir)
                except (FileNotFoundError, OSError):
                    continue
        except ImportError:
            pass  # Not on Windows
        
        # 3. Check common paths
        for path_str in self.ARCGIS_COMMON_PATHS:
            try:
                path = Path(path_str)
                if path.exists():
                    logger.debug(f"Found ArcGIS Pro at common path: {path}")
                    return path
            except Exception:
                continue
        
        return None
    
    def _initialize_arcpy(self):
        """Initialize arcpy module from ArcGIS Pro."""
        # First, try direct import (works if running from ArcGIS Pro Python)
        try:
            import arcpy
            self.arcpy = arcpy
            logger.info("✓ arcpy imported directly (running in ArcGIS Pro Python environment)")
            return
        except ImportError:
            pass
        
        # Find ArcGIS Pro installation
        arcgis_path = self._find_arcgis_pro()
        if arcgis_path:
            self.arcgis_pro_path = arcgis_path
            
            # Find the Python executable
            python_paths = [
                arcgis_path / "bin" / "Python" / "envs" / "arcgispro-py3" / "python.exe",
                arcgis_path / "bin" / "Python" / "python.exe",
            ]
            
            for py_path in python_paths:
                if py_path.exists():
                    self.arcgis_python_path = py_path
                    break
            
            logger.info(f"✓ ArcGIS Pro found at: {arcgis_path}")
            if self.arcgis_python_path:
                logger.info(f"✓ ArcGIS Python found at: {self.arcgis_python_path}")
            logger.info("  Note: For full arcpy access, run SurvyAI from ArcGIS Pro's Python environment")
        else:
            logger.info("⚠ ArcGIS Pro not detected; arcpy features unavailable.")
            logger.info("  Project management via subprocess will still be attempted.")
    
    @property
    def is_available(self) -> bool:
        """Check if arcpy is available for direct use."""
        return self.arcpy is not None
    
    @property
    def is_installed(self) -> bool:
        """Check if ArcGIS Pro is installed (even if arcpy isn't accessible)."""
        return self.arcgis_pro_path is not None
    
    # ==========================================================================
    # PROJECT MANAGEMENT METHODS
    # ==========================================================================
    
    def launch_arcgis_pro(self, project_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Launch ArcGIS Pro application, optionally opening a project.
        
        If a project path is provided but doesn't exist, this method will:
        1. Try to find the project in common locations
        2. If not found and the input looks like a missing *.aprx path, attempt to create it
        3. If still not found, launch ArcGIS Pro without opening a project
        
        Args:
            project_path: Optional path to .aprx project file to open, or project name
            
        Returns:
            Dictionary with success status and message
        """
        if not self.arcgis_pro_path:
            return {
                "success": False,
                "error": "ArcGIS Pro installation not found",
                "suggestion": "Please install ArcGIS Pro or set ARCGIS_PRO_PATH in settings"
            }
        
        try:
            # Guard against opening many windows of the same project in quick succession.
            # This can happen when a workflow calls multiple ArcGIS actions (run -> finalize -> open).
            try:
                import time as _time
                now = _time.time()
                proj_norm = str(Path(project_path).resolve()) if project_path else None
                if proj_norm and self._last_launch_project == proj_norm and (now - self._last_launch_ts) < 90:
                    return {
                        "success": True,
                        "message": "ArcGIS Pro launch skipped (duplicate launch guard)",
                        "project_path": proj_norm,
                        "note": "ArcGIS Pro was launched recently for this project; avoiding multiple windows.",
                    }
            except Exception:
                pass

            # Find the ArcGIS Pro executable
            exe_path = self.arcgis_pro_path / "bin" / "ArcGISPro.exe"
            
            if not exe_path.exists():
                return {
                    "success": False,
                    "error": f"ArcGIS Pro executable not found at: {exe_path}"
                }
            
            # Build command to launch ArcGIS Pro
            cmd = [str(exe_path)]
            resolved_project_path = None
            auto_created = False
            
            if project_path:
                project_file = Path(project_path).resolve()

                # If the provided path exists, use it
                if project_file.exists() and project_file.suffix.lower() == ".aprx":
                    resolved_project_path = project_file
                    cmd.append(str(resolved_project_path))
                    logger.info(f"Launching ArcGIS Pro with project: {resolved_project_path}")
                else:
                    # Determine intended project name (handle both name-only and *.aprx inputs)
                    input_looks_like_aprx = project_file.suffix.lower() == ".aprx"
                    project_name = project_file.stem if input_looks_like_aprx else project_file.name

                    # 1) Prefer current_project if it matches
                    if self.current_project:
                        current_proj = Path(self.current_project)
                        if current_proj.exists() and current_proj.stem.lower() == project_name.lower():
                            resolved_project_path = current_proj
                            cmd.append(str(resolved_project_path))
                            logger.info(f"Launching ArcGIS Pro with current project: {resolved_project_path}")

                    # 2) If not resolved, search common locations (ALWAYS search even if current_project is None)
                    if resolved_project_path is None:
                        search_paths = [
                            project_file.parent,  # Original location
                            Path.cwd(),  # Current working directory
                            Path.cwd() / project_name,  # Subdirectory in current dir
                            Path(os.path.expanduser("~")) / "Documents" / "ArcGIS" / "Projects",
                            Path(os.path.expanduser("~")) / "Documents" / "ArcGIS" / "Projects" / project_name,
                        ]

                        for search_base in search_paths:
                            # Try direct .aprx file
                            if search_base.is_dir():
                                potential_file = search_base / f"{project_name}.aprx"
                                if potential_file.exists():
                                    resolved_project_path = potential_file
                                    cmd.append(str(resolved_project_path))
                                    logger.info(f"Found project at: {resolved_project_path}")
                                    break

                                # Try in subdirectory (ArcGIS Pro creates projects in subdirectories)
                                subdir_file = search_base / project_name / f"{project_name}.aprx"
                                if subdir_file.exists():
                                    resolved_project_path = subdir_file
                                    cmd.append(str(resolved_project_path))
                                    logger.info(f"Found project in subdirectory: {resolved_project_path}")
                                    break
                            else:
                                # If a file path was provided (non-dir), try it directly
                                if search_base.exists() and search_base.suffix.lower() == ".aprx":
                                    resolved_project_path = search_base
                                    cmd.append(str(resolved_project_path))
                                    logger.info(f"Found project at: {resolved_project_path}")
                                    break

                    # 3) If still not resolved and user gave a missing *.aprx path, auto-create then open
                    if resolved_project_path is None and input_looks_like_aprx:
                        default_cs = getattr(self.settings, "arcgis_default_coordinate_system", None)
                        try:
                            logger.info(
                                "Project file not found (%s). Attempting auto-create project '%s' in %s",
                                project_file,
                                project_name,
                                project_file.parent,
                            )
                            create_result = self.create_project(
                                project_name=project_name,
                                project_path=str(project_file.parent),
                                coordinate_system=default_cs,
                                template="MAP",
                            )
                            if create_result.get("success") and create_result.get("project_path"):
                                created_path = Path(create_result["project_path"]).resolve()
                                if created_path.exists():
                                    resolved_project_path = created_path
                                    cmd.append(str(resolved_project_path))
                                    auto_created = True
                                    logger.info(f"Auto-created and will open project: {resolved_project_path}")
                        except Exception as e:
                            logger.warning(f"Auto-create project attempt failed: {e}")

                    if resolved_project_path is None:
                        logger.warning(
                            f"Project file not found: {project_file}. "
                            f"Launching ArcGIS Pro without opening a project."
                        )
            else:
                logger.info(f"Launching ArcGIS Pro from: {exe_path}")
            
            # Launch ArcGIS Pro
            subprocess.Popen(cmd, shell=False)
            try:
                import time as _time
                self._last_launch_ts = _time.time()
                self._last_launch_project = str(Path(resolved_project_path).resolve()) if resolved_project_path else (str(Path(project_path).resolve()) if project_path else None)
            except Exception:
                pass
            
            # Give it a moment to start
            time.sleep(2)
            
            message = "ArcGIS Pro launched successfully"
            if resolved_project_path:
                message += f" with project: {resolved_project_path.name}"
                # Update current_project if we found and opened one
                self.current_project = str(resolved_project_path)
            elif project_path:
                message += " (project file not found - launched without opening project)"
            
            return {
                "success": True,
                "message": message,
                "path": str(exe_path),
                "project_path": str(resolved_project_path) if resolved_project_path else None,
                "auto_created": auto_created,
                "note": (
                    "Project was missing; SurvyAI auto-created it before launching"
                    if auto_created
                    else ("Project not found - launched without opening project" if project_path and not resolved_project_path else None)
                ),
            }
            
        except Exception as e:
            logger.error(f"Error launching ArcGIS Pro: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def create_project(
        self,
        project_name: str,
        project_path: Optional[str] = None,
        coordinate_system: Optional[str] = None,
        template: str = "MAP",
        clean_layers: bool = True,
    ) -> Dict[str, Any]:
        """
        Create a new ArcGIS Pro project.
        
        Args:
            project_name: Name of the project (without .aprx extension)
            project_path: Directory to save the project (default: Documents/ArcGIS/Projects)
            coordinate_system: Coordinate system to use (e.g., "UTM Zone 32N", "WGS84")
            template: Project template - "MAP", "CATALOG", "GLOBAL_SCENE", "LOCAL_SCENE"
            clean_layers: If True (default), remove any template/sample layers & tables so the project starts empty.
            
        Returns:
            Dictionary with success status and project details
        """
        # Determine project path
        if project_path:
            base_path = Path(project_path)
            # If the provided path is a file (ends with .aprx), use its parent directory
            if base_path.suffix.lower() == '.aprx':
                logger.warning(f"Project path provided is a file path, using parent directory: {base_path.parent}")
                base_path = base_path.parent
        else:
            # Default to Documents/ArcGIS/Projects
            documents = Path(os.path.expanduser("~")) / "Documents"
            base_path = documents / "ArcGIS" / "Projects"
        
        # Ensure base_path is a directory (create if needed)
        if not base_path.exists():
            try:
                base_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created project base directory: {base_path}")
            except Exception as e:
                logger.error(f"Failed to create project base directory {base_path}: {e}")
                return {
                    "success": False,
                    "error": f"Cannot create project directory: {e}",
                    "suggested_path": str(base_path)
                }
        
        # Create project directory
        project_dir = base_path / project_name
        project_file = project_dir / f"{project_name}.aprx"
        
        # Check if project already exists
        if project_file.exists():
            logger.info(f"Project already exists at: {project_file}")
            self.current_project = str(project_file)
            return {
                "success": True,
                "message": f"Project '{project_name}' already exists",
                "project_path": str(project_file),
                "project_directory": str(project_dir),
                "already_exists": True,
                "note": "Using existing project"
            }
        
        logger.info(f"Creating project '{project_name}' at: {project_file}")
        
        # Parse coordinate system
        crs_info = None
        if coordinate_system:
            crs_info = parse_coordinate_system(coordinate_system)
            logger.info(f"Coordinate system parsed: {crs_info}")
        
        # Try direct arcpy method first
        if self.arcpy:
            return self._create_project_arcpy(
                project_name, project_dir, project_file, crs_info, template, clean_layers
            )
        
        # Fall back to subprocess method
        if self.arcgis_python_path:
            return self._create_project_subprocess(
                project_name, project_dir, project_file, crs_info, template, clean_layers
            )
        
        # Last resort: provide manual instructions
        return self._create_project_instructions(
            project_name, project_dir, project_file, crs_info, template
        )
    
    def _create_project_arcpy(
        self,
        project_name: str,
        project_dir: Path,
        project_file: Path,
        crs_info: Optional[Dict],
        template: str,
        clean_layers: bool = True,
    ) -> Dict[str, Any]:
        """Create project using direct arcpy access."""
        try:
            import arcpy
            
            # Create project directory
            project_dir.mkdir(parents=True, exist_ok=True)
            
            # Find a suitable project template
            install_dir = arcpy.GetInstallInfo()["InstallDir"]
            template_paths = [
                Path(install_dir) / "Resources" / "ProjectTemplates" / "BlankProject.aptx",
                Path(install_dir) / "Resources" / "ProjectTemplates" / "Blank.aptx",
            ]
            
            template_file = None
            for tp in template_paths:
                if tp.exists():
                    template_file = str(tp)
                    break
            
            if template_file:
                # Create project from template file
                aprx = arcpy.mp.ArcGISProject(template_file)
            else:
                # Fallback: try to use CURRENT (only works if ArcGIS Pro is open)
                try:
                    aprx = arcpy.mp.ArcGISProject("CURRENT")
                except Exception:
                    return {
                        "success": False,
                        "error": "No project template found and ArcGIS Pro is not open",
                        "suggestion": "Please open ArcGIS Pro first or use the subprocess method"
                    }
            
            # Set coordinate system if specified
            if crs_info and crs_info.get("wkid"):
                maps = aprx.listMaps()
                if maps:
                    m = maps[0]
                    sr = arcpy.SpatialReference(crs_info["wkid"])
                    m.spatialReference = sr
                    logger.info(f"Set map coordinate system to: {crs_info['name']}")

            # Optionally clean template/sample layers and tables (prevents "mystery points" showing up)
            if clean_layers:
                try:
                    for m in aprx.listMaps():
                        for lyr in m.listLayers():
                            try:
                                m.removeLayer(lyr)
                            except Exception:
                                pass
                        # Tables (best-effort; removeTable may not exist in older versions)
                        try:
                            for tbl in getattr(m, "listTables", lambda: [])():
                                try:
                                    m.removeTable(tbl)  # type: ignore[attr-defined]
                                except Exception:
                                    pass
                        except Exception:
                            pass
                except Exception:
                    pass
            
            # Save the project
            aprx.saveACopy(str(project_file))
            
            # Verify the file was created
            if not project_file.exists():
                return {
                    "success": False,
                    "error": f"Project file was not created at: {project_file}",
                    "project_path": str(project_file),
                    "project_directory": str(project_dir)
                }
            
            # Set current project
            self.current_project = str(project_file)
            logger.info(f"✓ Project created and set as current: {project_file}")
            
            return {
                "success": True,
                "message": f"Project '{project_name}' created successfully",
                "project_path": str(project_file),
                "project_directory": str(project_dir),
                "coordinate_system": crs_info,
                "template": template
            }
            
        except Exception as e:
            logger.error(f"Error creating project with arcpy: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _create_project_subprocess(
        self,
        project_name: str,
        project_dir: Path,
        project_file: Path,
        crs_info: Optional[Dict],
        template: str,
        clean_layers: bool = True,
    ) -> Dict[str, Any]:
        """
        Create project by generating a script and providing instructions.
        
        Since arcpy requires the full ArcGIS Pro environment which is difficult
        to initialize via subprocess, we create a script file and provide
        instructions for the user to run it.
        """
        try:
            # Create the project directory
            project_dir.mkdir(parents=True, exist_ok=True)
            
            # Build the Python script to create the project
            wkid = crs_info.get("wkid") if crs_info else None
            crs_name = crs_info.get("name", "Unknown") if crs_info else "Unknown"
            
            # Escape paths for use in Python script (use forward slashes)
            project_file_escaped = str(project_file).replace("\\", "/")
            project_dir_escaped = str(project_dir).replace("\\", "/")
            
            # Build script that searches thoroughly for templates and creates project
            script_content = '''# ArcGIS Pro Project Creation Script
# Generated by SurvyAI
# Project: ''' + project_name + '''
# Coordinate System: ''' + crs_name + ''' (WKID: ''' + str(wkid) + ''')

import arcpy
import os
import glob

print("=" * 60)
print("SurvyAI - Creating ArcGIS Pro Project")
print("=" * 60)

# Configuration
project_name = "''' + project_name + '''"
project_dir = "''' + project_dir_escaped + '''"
project_path = "''' + project_file_escaped + '''"
template_type = "''' + template.upper() + '''"
wkid = ''' + str(wkid if wkid else 0) + '''
clean_layers = ''' + ("True" if clean_layers else "False") + '''

# Create project directory
os.makedirs(project_dir, exist_ok=True)
print("Project directory:", project_dir)

try:
    # Get ArcGIS Pro version info
    install_info = arcpy.GetInstallInfo()
    version = install_info.get("Version", "unknown")
    install_dir = install_info["InstallDir"]
    print("ArcGIS Pro version:", version)
    print("Install directory:", install_dir)
    
    # Search for template files in multiple locations
    template_paths_to_check = [
        os.path.join(install_dir, "Resources", "ProjectTemplates"),
        os.path.join(install_dir, "Resources", "Core", "ProjectTemplates"),
        os.path.join(os.path.expanduser("~"), "AppData", "Local", "ESRI", "ArcGISPro", "ProjectTemplates"),
        os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "ESRI", "ArcGISPro", "ProjectTemplates"),
        os.path.join(os.path.expanduser("~"), "Documents", "ArcGIS", "ProjectTemplates"),
    ]
    
    template_found = None
    all_templates = []
    
    # Also search for .aprx files in template directories (some versions use .aprx as templates)
    for template_dir in template_paths_to_check:
        if os.path.exists(template_dir):
            templates_aptx = glob.glob(os.path.join(template_dir, "*.aptx"))
            templates_aprx = glob.glob(os.path.join(template_dir, "*.aprx"))
            all_templates.extend(templates_aptx)
            all_templates.extend(templates_aprx)
            total = len(templates_aptx) + len(templates_aprx)
            if total > 0:
                print("Found", total, "templates in:", template_dir)
    
    # Prefer templates matching the requested type
    template_preferences = {
        "MAP": ["map", "blank"],
        "CATALOG": ["catalog"],
        "GLOBAL_SCENE": ["global", "scene"],
        "LOCAL_SCENE": ["local", "scene"]
    }
    
    preferred_names = template_preferences.get(template_type, ["map", "blank"])
    
    for template_file in all_templates:
        tname_lower = os.path.basename(template_file).lower()
        if any(pref in tname_lower for pref in preferred_names):
            template_found = template_file
            print("Selected matching template:", template_file)
            break
    
    # If no matching template, use first available
    if not template_found and all_templates:
        template_found = all_templates[0]
        print("Using first available template:", template_found)
    
    if template_found:
        print("Creating project from template:", template_found)
        aprx = arcpy.mp.ArcGISProject(template_found)
        print("Project object created successfully")
    else:
        # No templates found - try to find any existing .aprx project to use as template
        print("No .aptx templates found. Searching for existing projects to use as template...")
        existing_project_paths = [
            os.path.join(os.path.expanduser("~"), "Documents", "ArcGIS", "Projects"),
        ]
        
        existing_project = None
        for proj_dir in existing_project_paths:
            if os.path.exists(proj_dir):
                # Look for any .aprx file in subdirectories
                for root, dirs, files in os.walk(proj_dir):
                    for file in files:
                        if file.endswith(".aprx") and file != os.path.basename(project_path):
                            existing_project = os.path.join(root, file)
                            print("Found existing project to use as template:", existing_project)
                            break
                    if existing_project:
                        break
            if existing_project:
                break
        
        if existing_project:
            try:
                print("Using existing project as template:", existing_project)
                aprx = arcpy.mp.ArcGISProject(existing_project)
                print("Opened existing project as template")
            except Exception as e:
                print("Failed to open existing project:", str(e))
                raise Exception("Cannot create project - found existing project but failed to open it. Please create manually in ArcGIS Pro.")
        else:
            # Last resort: try CURRENT (only works if ArcGIS Pro is running with a project open)
            try:
                print("Trying CURRENT project (requires ArcGIS Pro to be running)...")
                aprx = arcpy.mp.ArcGISProject("CURRENT")
                print("Using CURRENT project as base")
            except Exception as e:
                print("CURRENT method failed:", str(e))
                raise Exception("Cannot create project automatically - no templates found. Please create manually: 1) Open ArcGIS Pro, 2) Click 'Map' under 'New Project', 3) Save to: " + project_dir)
    
    # Set coordinate system if specified
    if wkid and wkid > 0:
        maps = aprx.listMaps()
        print("Maps in project:", [m.name for m in maps])
        if maps:
            sr = arcpy.SpatialReference(wkid)
            for m in maps:
                m.spatialReference = sr
            print("Set coordinate system to WKID:", wkid, "(" + ''' + repr(crs_name) + ''' + ")")
        else:
            print("Warning: No maps found in project")

    # Remove any template/sample layers & tables unless explicitly disabled
    if clean_layers:
        try:
            for m in aprx.listMaps():
                for lyr in m.listLayers():
                    try:
                        m.removeLayer(lyr)
                    except Exception:
                        pass
                try:
                    for tbl in m.listTables():
                        try:
                            m.removeTable(tbl)
                        except Exception:
                            pass
                except Exception:
                    pass
            print("Cleaned template layers/tables")
        except Exception as e:
            print("Warning: failed cleaning template layers:", str(e))
    
    # Save the project to the specified path
    print("Saving project to:", project_path)
    aprx.saveACopy(project_path)
    print("Project saved successfully")
    
    # Verify the file was created
    if os.path.exists(project_path):
        file_size = os.path.getsize(project_path)
        print("")
        print("=" * 60)
        print("SUCCESS! Project created:")
        print("  Path:", project_path)
        print("  Size:", file_size, "bytes")
        print("  Coordinate System:", ''' + repr(crs_name) + ''', "WKID:", wkid)
        print("=" * 60)
        exit(0)  # Success exit code
    else:
        print("ERROR: Project file was not created at:", project_path)
        exit(1)

except Exception as e:
    import traceback
    print("")
    print("=" * 60)
    print("ERROR creating project:")
    print("  ", str(e))
    print("")
    print("Full traceback:")
    traceback.print_exc()
    print("=" * 60)
    exit(1)
'''
            
            # Save the script to a permanent location (not temp)
            scripts_dir = project_dir / "scripts"
            scripts_dir.mkdir(parents=True, exist_ok=True)
            script_file = scripts_dir / f"create_{project_name}.py"
            
            with open(script_file, 'w', encoding='utf-8') as f:
                f.write(script_content)
            
            logger.info(f"Created project creation script: {script_file}")
            
            # Execute the script using propy.bat to create the project
            propy_bat = self.arcgis_pro_path / "bin" / "Python" / "Scripts" / "propy.bat"
            
            if propy_bat.exists():
                logger.info(f"Executing project creation script via propy.bat...")
                logger.info(f"Script: {script_file}")
                
                try:
                    # Run the script using propy.bat (this initializes ArcGIS Python environment)
                    # Use shell=True on Windows for .bat files
                    cmd = f'"{propy_bat}" "{script_file}"'
                    result = subprocess.run(
                        cmd,
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=300,  # 5 minutes timeout (ArcGIS Pro initialization takes time)
                        cwd=str(project_dir)
                    )
                    
                    script_output = result.stdout
                    script_error = result.stderr
                    
                    logger.info(f"propy.bat return code: {result.returncode}")
                    if script_output:
                        logger.info(f"Script output: {script_output[:1000]}")
                    if script_error:
                        logger.warning(f"Script stderr: {script_error[:500]}")
                    
                    # Check if project file was created
                    if project_file.exists() and project_file.stat().st_size > 0:
                        self.current_project = str(project_file)
                        logger.info(f"✓ Project created successfully: {project_file}")
                        logger.info(f"✓ Project set as current project for subsequent operations")
                        return {
                            "success": True,
                            "message": f"Project '{project_name}' created successfully with coordinate system {crs_name}!",
                            "project_path": str(project_file),
                            "project_directory": str(project_dir),
                            "coordinate_system": crs_info,
                            "template": template,
                            "output": script_output
                        }
                    elif result.returncode == 0:
                        # Script completed but file not found - might be a warning
                        logger.warning(f"Script completed (return code 0) but project file not found at: {project_file}")
                        logger.warning(f"Script output: {script_output[:500] if script_output else 'No output'}")
                        logger.warning(f"Script error: {script_error[:500] if script_error else 'No errors'}")
                        return {
                            "success": False,
                            "message": "Script executed but project file was not created. Check output for errors.",
                            "script_path": str(script_file),
                            "project_path": str(project_file),
                            "project_directory": str(project_dir),
                            "coordinate_system": crs_info,
                            "output": script_output,
                            "error": script_error,
                            "suggestion": (
                                f"The script completed but the project file was not created. "
                                f"Please check the script output above for errors. "
                                f"You can also try running the script manually: {script_file}"
                            )
                        }
                    else:
                        # Script failed
                        logger.error(f"Script execution failed with return code {result.returncode}")
                        logger.error(f"Script output (last 1000 chars): {script_output[-1000:] if script_output else 'No output'}")
                        logger.error(f"Script error (last 1000 chars): {script_error[-1000:] if script_error else 'No errors'}")
                        return {
                            "success": False,
                            "message": f"Failed to create project automatically (return code: {result.returncode}). See output for details.",
                            "script_path": str(script_file),
                            "project_path": str(project_file),
                            "project_directory": str(project_dir),
                            "coordinate_system": crs_info,
                            "output": script_output,
                            "error": script_error,
                            "return_code": result.returncode,
                            "instructions": (
                                f"**Manual Creation:**\n"
                                f"1. Open ArcGIS Pro\n"
                                f"2. Click 'Map' under 'New Project'\n"
                                f"3. Name: {project_name}\n"
                                f"4. Location: {project_dir}\n"
                                f"5. Map tab > Properties > Coordinate Systems > Search: {wkid} ({crs_name})\n\n"
                                f"**Or try running the script manually:**\n"
                                f"Open ArcGIS Pro Python window and run: exec(open(r'{script_file}').read())"
                            )
                        }
                        
                except subprocess.TimeoutExpired:
                    logger.warning("Script execution timed out after 5 minutes")
                    return {
                        "success": False,
                        "message": "Project creation timed out. This may happen if ArcGIS Pro is initializing. Please try again or create manually.",
                        "script_path": str(script_file),
                        "project_path": str(project_file),
                        "project_directory": str(project_dir),
                        "coordinate_system": crs_info,
                        "instructions": (
                            f"**Manual Creation:**\n"
                            f"1. Open ArcGIS Pro\n"
                            f"2. Click 'Map' under 'New Project'\n"
                            f"3. Name: {project_name}\n"
                            f"4. Location: {project_dir}\n"
                            f"5. Map tab > Properties > Coordinate Systems > Search: {wkid} ({crs_name})"
                        )
                    }
                except Exception as e:
                    logger.error(f"Error executing script: {e}")
                    return {
                        "success": False,
                        "message": f"Error executing project creation script: {str(e)}",
                        "script_path": str(script_file),
                        "project_path": str(project_file),
                        "project_directory": str(project_dir),
                        "coordinate_system": crs_info,
                        "error": str(e)
                    }
            else:
                logger.warning("propy.bat not found, cannot execute script automatically")
                return {
                    "success": False,
                    "message": "Cannot execute script automatically - propy.bat not found",
                    "script_path": str(script_file),
                    "project_path": str(project_file),
                    "project_directory": str(project_dir),
                    "coordinate_system": crs_info
                }
                
        except Exception as e:
            logger.error(f"Error creating project script: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _create_project_instructions(
        self,
        project_name: str,
        project_dir: Path,
        project_file: Path,
        crs_info: Optional[Dict],
        template: str
    ) -> Dict[str, Any]:
        """Return manual instructions when automation isn't available."""
        crs_name = crs_info.get("name", "Not specified") if crs_info else "Not specified"
        wkid = crs_info.get("wkid", "N/A") if crs_info else "N/A"
        
        instructions = f"""
**ArcGIS Pro Project Creation Instructions**

Since direct ArcGIS Pro control is not available, please follow these steps manually:

1. **Open ArcGIS Pro**
   - Launch ArcGIS Pro from your Start Menu or desktop

2. **Create New Project**
   - Click "New" on the start page
   - Select "{template}" template
   - Name: {project_name}
   - Location: {project_dir}
   - Click "OK" to create

3. **Set Coordinate System**
   - In the Contents pane, right-click on "Map"
   - Select "Properties"
   - Go to "Coordinate Systems" tab
   - Search for: {crs_name}
   - WKID: {wkid}
   - Click "OK" to apply

4. **Save Project**
   - Press Ctrl+S or File > Save

**Expected Output:**
- Project file: {project_file}
"""
        
        return {
            "success": False,
            "requires_manual": True,
            "message": "Automated project creation not available - please follow manual instructions",
            "instructions": instructions,
            "project_name": project_name,
            "suggested_path": str(project_file),
            "coordinate_system": crs_info
        }
    
    def open_project(self, project_path: str) -> Dict[str, Any]:
        """
        Open an existing ArcGIS Pro project.
        
        Args:
            project_path: Path to the .aprx project file
            
        Returns:
            Dictionary with success status and project details
        """
        project_file = Path(project_path)
        
        if not project_file.exists():
            return {
                "success": False,
                "error": f"Project file not found: {project_path}"
            }
        
        if not project_file.suffix.lower() == '.aprx':
            return {
                "success": False,
                "error": "File must be an ArcGIS Pro project (.aprx)"
            }
        
        # Try direct arcpy method
        if self.arcpy:
            try:
                aprx = self.arcpy.mp.ArcGISProject(str(project_file))
                self.current_project = str(project_file)
                
                # Get project info
                maps = aprx.listMaps()
                
                return {
                    "success": True,
                    "message": f"Project opened: {project_file.name}",
                    "project_path": str(project_file),
                    "maps": [m.name for m in maps],
                    "map_count": len(maps)
                }
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        # Fall back to launching ArcGIS Pro with the project
        if self.arcgis_pro_path:
            try:
                exe_path = self.arcgis_pro_path / "bin" / "ArcGISPro.exe"
                subprocess.Popen([str(exe_path), str(project_file)], shell=False)
                self.current_project = str(project_file)
                
                return {
                    "success": True,
                    "message": f"Opening project in ArcGIS Pro: {project_file.name}",
                    "project_path": str(project_file)
                }
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        return {
            "success": False,
            "error": "Cannot open project - ArcGIS Pro not available"
        }
    
    def set_map_coordinate_system(
        self,
        coordinate_system: str,
        map_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Set the coordinate system for a map in the current project.
        
        Args:
            coordinate_system: Coordinate system identifier (name, EPSG code, or WKID)
            map_name: Name of the map to modify (default: first map)
            
        Returns:
            Dictionary with success status
        """
        if not self.arcpy:
            # Generate script, save it, launch ArcGIS Pro, and provide instructions
            if self.arcgis_pro_path and self.current_project:
                try:
                    crs_info = parse_coordinate_system(coordinate_system)
                    wkid = crs_info.get("wkid")
                    if not wkid:
                        return {
                            "success": False,
                            "error": f"Could not determine WKID for coordinate system: {coordinate_system}",
                            "coordinate_system": crs_info,
                        }

                    project_file = Path(self.current_project).resolve()
                    project_path_escaped = str(project_file).replace("\\", "/")
                    map_name_escaped = (map_name or "").replace("'", "\\'")

                    script_content = f'''# Generated by SurvyAI: set map coordinate system
# Run this script in ArcGIS Pro's Python Window (Analysis > Python > Python Window)
# Or copy-paste the code below into the Python window

import arcpy

project_path = r"{project_path_escaped}"
wkid = {int(wkid)}
map_name = {repr(map_name) if map_name else "None"}

print("Opening project:", project_path)
aprx = arcpy.mp.ArcGISProject(project_path)
maps = aprx.listMaps(map_name) if map_name else aprx.listMaps()
if not maps:
    raise RuntimeError("No maps found" if not map_name else f"Map not found: {{map_name}}")

print(f"Found {{len(maps)}} map(s). Setting coordinate system to WKID {{wkid}}...")
sr = arcpy.SpatialReference(wkid)
for m in maps:
    m.spatialReference = sr
    print(f"  Set CRS for map: {{m.name}}")

aprx.save()
print("OK: Coordinate system set to", sr.name, "(WKID", wkid, ")")
print("Project saved successfully.")
'''
                    # Save script to project's scripts folder
                    scripts_dir = project_file.parent / "scripts"
                    scripts_dir.mkdir(parents=True, exist_ok=True)
                    script_file = scripts_dir / "set_map_coordinate_system.py"
                    script_file.write_text(script_content, encoding="utf-8")
                    logger.info(f"Saved coordinate system script to: {script_file}")

                    # Launch ArcGIS Pro with the project
                    launch_result = self.launch_arcgis_pro(project_path=str(project_file))
                    
                    instructions = (
                        f"I've generated a script to set the coordinate system and launched ArcGIS Pro.\n\n"
                        f"To run the script:\n"
                        f"1. In ArcGIS Pro, go to the Analysis tab\n"
                        f"2. Click Python > Python Window (or press Ctrl+Alt+P)\n"
                        f"3. In the Python window, type: exec(open(r'{script_file}').read())\n"
                        f"4. Press Enter to execute\n\n"
                        f"Alternatively, open the script file in a text editor and copy-paste its contents into the Python window.\n\n"
                        f"Script location: {script_file}\n"
                        f"Coordinate system: {crs_info['name']} (WKID: {wkid})"
                    )

                    return {
                        "success": True,
                        "message": f"Script generated and ArcGIS Pro launched. Please run the script in ArcGIS Pro's Python window.",
                        "coordinate_system": crs_info,
                        "project_path": str(project_file),
                        "script_path": str(script_file),
                        "instructions": instructions,
                        "arcgis_launched": launch_result.get("success", False),
                    }
                except Exception as e:
                    logger.error(f"Error generating coordinate system script: {e}")
                    return {"success": False, "error": str(e)}

            # Otherwise, fall back to manual instructions
            crs_info = parse_coordinate_system(coordinate_system)
            return {
                "success": False,
                "requires_manual": True,
                "message": "Please set the coordinate system manually in ArcGIS Pro",
                "instructions": (
                    f"1. Right-click on '{map_name or 'Map'}' in Contents pane\n"
                    f"2. Select 'Properties'\n"
                    f"3. Go to 'Coordinate Systems' tab\n"
                    f"4. Search for: {crs_info['name']}\n"
                    f"5. WKID: {crs_info.get('wkid', 'N/A')}\n"
                    f"6. Click 'OK'"
                ),
                "coordinate_system": crs_info
            }
        
        try:
            crs_info = parse_coordinate_system(coordinate_system)
            
            if not crs_info.get("wkid"):
                return {
                    "success": False,
                    "error": f"Could not determine WKID for coordinate system: {coordinate_system}"
                }
            
            if not self.current_project:
                return {
                    "success": False,
                    "error": "No project is currently open"
                }
            
            aprx = self.arcpy.mp.ArcGISProject(self.current_project)
            maps = aprx.listMaps(map_name) if map_name else aprx.listMaps()
            
            if not maps:
                return {
                    "success": False,
                    "error": f"Map not found: {map_name}" if map_name else "No maps in project"
                }
            
            target_map = maps[0]
            sr = self.arcpy.SpatialReference(crs_info["wkid"])
            target_map.spatialReference = sr
            aprx.save()
            
            return {
                "success": True,
                "message": f"Coordinate system set to {crs_info['name']}",
                "map_name": target_map.name,
                "coordinate_system": crs_info
            }
            
        except Exception as e:
            logger.error(f"Error setting coordinate system: {e}")
            return {"success": False, "error": str(e)}

    def import_xy_points_from_excel(
        self,
        project_path: str,
        excel_path: str,
        x_field: str,
        y_field: str,
        coordinate_system: str,
        sheet_name: Optional[str] = None,
        layer_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Import XY points from an Excel file into the given ArcGIS Pro project.

        This creates a point feature class in a file geodatabase inside the project folder,
        adds it to the first map, and saves the project.

        Notes:
        - When run headlessly via propy.bat, ArcGIS Pro's GUI can't be "zoomed".
          We still add the layer so you can open the project and zoom-to-layer instantly.
        - Excel headers can contain leading tabs/spaces; this method writes a clean CSV
          (with safe field names) before importing.
        """
        project_file = Path(project_path).resolve()
        if not project_file.exists() or project_file.suffix.lower() != ".aprx":
            return {"success": False, "error": f"Invalid project path: {project_path}"}

        excel_file = Path(excel_path).resolve()
        if not excel_file.exists():
            return {"success": False, "error": f"Excel file not found: {excel_path}"}

        crs_info = parse_coordinate_system(coordinate_system)
        wkid = crs_info.get("wkid")
        if not wkid:
            return {
                "success": False,
                "error": f"Could not determine WKID for coordinate system: {coordinate_system}",
                "coordinate_system": crs_info,
            }

        # Prepare a clean CSV with safe column names so arcpy import is reliable.
        try:
            import pandas as pd

            df = pd.read_excel(excel_file, sheet_name=sheet_name or 0)

            def _norm_col(c: str) -> str:
                return " ".join(str(c).replace("\t", " ").split()).strip().lower()

            norm_map = {_norm_col(c): c for c in df.columns}

            def _resolve_col(requested: str) -> Optional[str]:
                if requested in df.columns:
                    return requested
                return norm_map.get(_norm_col(requested))

            x_res = _resolve_col(x_field)
            y_res = _resolve_col(y_field)
            if x_res is None or y_res is None:
                return {
                    "success": False,
                    "error": f"Could not find x/y fields in Excel. Requested x='{x_field}', y='{y_field}'.",
                    "available_columns": list(df.columns),
                }

            safe_df = df.copy()
            # ---- Numeric hygiene for ArcGIS XY import (critical) ----
            # ArcGIS often reads Excel columns as TEXT; XYTableToPoint requires numeric X/Y.
            # We always coerce to numeric *here* before writing CSV.
            def _coerce_numeric(series: "pd.Series") -> "pd.Series":
                s = series
                # If the series is already numeric dtype, keep it.
                try:
                    if pd.api.types.is_numeric_dtype(s):
                        return s
                except Exception:
                    pass

                # Normalize common numeric string issues:
                # - thousands separators: "1,234.56"
                # - decimal comma: "1234,56"
                s_str = s.astype(str).str.strip()
                s_str = s_str.replace({"": None, "nan": None, "None": None})

                has_comma = s_str.str.contains(",", na=False)
                has_dot = s_str.str.contains(r"\.", na=False)

                # Decide per-value:
                # - If a value has comma but NO dot => treat comma as decimal separator
                # - Else treat comma as thousands separator
                decimal_comma_mask = has_comma & (~has_dot)
                if bool(decimal_comma_mask.any()):
                    s_str.loc[decimal_comma_mask] = s_str.loc[decimal_comma_mask].str.replace(",", ".", regex=False)
                if bool((~decimal_comma_mask).any()):
                    s_str.loc[~decimal_comma_mask] = s_str.loc[~decimal_comma_mask].str.replace(",", "", regex=False)

                # Remove spaces inside numbers (e.g., "1 234.56")
                s_str = s_str.str.replace(" ", "", regex=False)

                return pd.to_numeric(s_str, errors="coerce")

            x_num = _coerce_numeric(safe_df[x_res])
            y_num = _coerce_numeric(safe_df[y_res])

            # Drop rows where X/Y cannot be coerced to numbers
            invalid_mask = x_num.isna() | y_num.isna()
            invalid_count = int(invalid_mask.sum())
            if invalid_count:
                logger.warning(
                    "ArcGIS XY import: dropping %s rows with non-numeric coordinates (x='%s', y='%s')",
                    invalid_count, x_res, y_res
                )
            safe_df = safe_df.loc[~invalid_mask].copy()
            safe_df["X"] = x_num.loc[~invalid_mask].astype("float64")
            safe_df["Y"] = y_num.loc[~invalid_mask].astype("float64")

            if len(safe_df) == 0:
                return {
                    "success": False,
                    "error": "No valid numeric coordinates found for ArcGIS import after coercion",
                    "details": {
                        "x_field": x_res,
                        "y_field": y_res,
                        "source_rows": int(len(df)),
                        "dropped_rows": invalid_count,
                    },
                }

            # Keep all attributes; just ensure X/Y exist with safe names.
            csv_path = excel_file.with_suffix("").with_name(excel_file.stem + "_xy.csv")
            # Use fixed float formatting so ArcGIS reads as numeric with '.' decimal separator.
            safe_df.to_csv(csv_path, index=False, encoding="utf-8", float_format="%.6f")
            logger.info("Prepared ArcGIS XY CSV with %s valid rows: %s", len(safe_df), csv_path)
        except Exception as e:
            return {"success": False, "error": f"Failed to prepare CSV from Excel: {e}"}

        # Execute import using arcpy directly if available, otherwise via propy.bat
        if self.arcpy:
            try:
                import arcpy

                aprx = arcpy.mp.ArcGISProject(str(project_file))
                maps = aprx.listMaps()
                if not maps:
                    return {"success": False, "error": "No maps found in project"}

                m = maps[0]
                sr = arcpy.SpatialReference(int(wkid))
                m.spatialReference = sr

                project_dir = project_file.parent
                gdb_path = project_dir / f"{project_file.stem}.gdb"
                if not gdb_path.exists():
                    arcpy.management.CreateFileGDB(str(project_dir), gdb_path.name)

                out_fc = gdb_path / (layer_name or f"{project_file.stem}_points")
                arcpy.management.XYTableToPoint(
                    in_table=str(csv_path),
                    out_feature_class=str(out_fc),
                    x_field="X",
                    y_field="Y",
                    coordinate_system=sr,
                )
                m.addDataFromPath(str(out_fc))
                aprx.save()

                self.current_project = str(project_file)
                return {
                    "success": True,
                    "message": "Imported XY points and added to map",
                    "project_path": str(project_file),
                    "coordinate_system": crs_info,
                    "source_table": str(csv_path),
                    "output_feature_class": str(out_fc),
                }
            except Exception as e:
                return {"success": False, "error": str(e)}

        # Generate script, save it, launch ArcGIS Pro, and provide instructions
        if not self.arcgis_pro_path:
            return {
                "success": False,
                "error": "ArcGIS Pro installation not found. Please install ArcGIS Pro.",
            }

        project_path_escaped = str(project_file.resolve()).replace("\\", "/")
        csv_path_escaped = str(Path(csv_path).resolve()).replace("\\", "/")
        gdb_path_escaped = str((project_file.parent / f"{project_file.stem}.gdb").resolve()).replace("\\", "/")
        out_fc_name = (layer_name or f"{project_file.stem}_points")

        script_content = f'''# Generated by SurvyAI: import XY points
# Run this script in ArcGIS Pro's Python Window (Analysis > Python > Python Window)
# Or copy-paste the code below into the Python window

import arcpy
import os

project_path = r"{project_path_escaped}"
csv_path = r"{csv_path_escaped}"
gdb_path = r"{gdb_path_escaped}"
out_fc_name = r"{out_fc_name}"
wkid = {int(wkid)}

print("Opening project:", project_path)
aprx = arcpy.mp.ArcGISProject(project_path)
maps = aprx.listMaps()
if not maps:
    raise RuntimeError("No maps found in project")
m = maps[0]
print(f"Using map: {{m.name}}")

print(f"Setting coordinate system to WKID {{wkid}}...")
sr = arcpy.SpatialReference(wkid)
m.spatialReference = sr

project_dir = os.path.dirname(project_path)
print(f"Creating file geodatabase if needed: {{gdb_path}}")
if not arcpy.Exists(gdb_path):
    arcpy.management.CreateFileGDB(project_dir, os.path.basename(gdb_path))
    print("  File geodatabase created")
else:
    print("  File geodatabase already exists")

out_fc = os.path.join(gdb_path, out_fc_name)
if arcpy.Exists(out_fc):
    print(f"Deleting existing feature class: {{out_fc}}")
    arcpy.management.Delete(out_fc)

print(f"Importing XY points from CSV: {{csv_path}}")
print(f"  X field: X")
print(f"  Y field: Y")
print(f"  Output: {{out_fc}}")
arcpy.management.XYTableToPoint(
    in_table=csv_path,
    out_feature_class=out_fc,
    x_field="X",
    y_field="Y",
    coordinate_system=sr,
)
print("  Points imported successfully")

print(f"Adding layer to map: {{m.name}}")
m.addDataFromPath(out_fc)
print("  Layer added to map")

aprx.save()
print("OK: Project saved successfully")
print(f"Feature class: {{out_fc}}")
print(f"Coordinate system: {{sr.name}} (WKID {{wkid}})")
print("\\nTo zoom to the points, right-click the layer in the Contents pane and select 'Zoom To Layer'")
'''

        # Save script to project's scripts folder
        scripts_dir = project_file.parent / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        script_file = scripts_dir / "import_xy_points.py"
        script_file.write_text(script_content, encoding="utf-8")
        logger.info(f"Saved XY import script to: {script_file}")

        # Launch ArcGIS Pro with the project
        launch_result = self.launch_arcgis_pro(project_path=str(project_file))

        instructions = (
            f"I've generated a script to import XY points and launched ArcGIS Pro.\n\n"
            f"To run the script:\n"
            f"1. In ArcGIS Pro, go to the Analysis tab\n"
            f"2. Click Python > Python Window (or press Ctrl+Alt+P)\n"
            f"3. In the Python window, type: exec(open(r'{script_file}').read())\n"
            f"4. Press Enter to execute\n\n"
            f"Alternatively, open the script file in a text editor and copy-paste its contents into the Python window.\n\n"
            f"Script location: {script_file}\n"
            f"Source CSV: {csv_path}\n"
            f"Output feature class: {project_file.stem}.gdb\\{out_fc_name}\n"
            f"Coordinate system: {crs_info['name']} (WKID: {wkid})\n\n"
            f"After running the script, right-click the layer in Contents and select 'Zoom To Layer' to view the points."
        )

        self.current_project = str(project_file)

        return {
            "success": True,
            "message": "Script generated and ArcGIS Pro launched. Please run the script in ArcGIS Pro's Python window.",
            "project_path": str(project_file),
            "coordinate_system": crs_info,
            "source_table": str(csv_path),
            "output_feature_class": str(project_file.parent / f"{project_file.stem}.gdb" / out_fc_name),
            "script_path": str(script_file),
            "instructions": instructions,
            "arcgis_launched": launch_result.get("success", False),
        }

    def execute_python_code(
        self,
        python_code: str,
        project_path: Optional[str] = None,
        script_name: Optional[str] = None,
        execute_automatically: bool = True,
        # New controls to prevent runaway launches / recursion
        launch_arcgis_pro: bool = True,
        prelaunch_arcgis_pro: bool = False,
        skip_finalization: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute dynamically generated Python/arcpy code.
        
        This method allows the SurvyAI agent to generate arcpy code on-the-fly based on user requests
        and execute it automatically. The code is saved to the project's scripts folder for reference.
        
        Args:
            python_code: The Python/arcpy code to execute (dynamically generated by LLM)
            project_path: Optional path to .aprx project (used to determine script save location)
            script_name: Optional script filename (default: auto-generated timestamp-based name)
            execute_automatically: If True, execute via propy.bat. If False, save and provide instructions.
            
        Returns:
            Dictionary with execution results, script path, and instructions if applicable
        """
        if not self.arcgis_pro_path:
            return {
                "success": False,
                "error": "ArcGIS Pro installation not found. Please install ArcGIS Pro.",
            }

        # ------------------------------------------------------------------
        # Headless safety rewrites
        # ------------------------------------------------------------------
        # IMPORTANT:
        # - arcpy.mp.ArcGISProject("CURRENT") only works inside the ArcGIS Pro UI Python window.
        # - propy.bat runs in a separate, headless Python process and can never access "CURRENT".
        # Therefore, when the caller provides a project_path (or we have a known current project),
        # we automatically rewrite "CURRENT" to the explicit .aprx path to keep automation scalable.
        def _rewrite_current_aprx(code: str, aprx_path: Optional[str]) -> str:
            if not code or not aprx_path:
                # Still strip CURRENT usages to prevent OSError: CURRENT in headless propy.bat runs.
                # (Even if we don't know the target .aprx, we can safely remove standalone
                # "initialization" calls like: arcpy.mp.ArcGISProject('CURRENT') )
                try:
                    import re
                    pattern = r"^\s*arcpy\.mp\.ArcGISProject\(\s*['\"]CURRENT['\"]\s*\)\s*(#.*)?$"
                    return re.sub(pattern, "# NOTE: removed ArcGISProject('CURRENT') (not available in headless execution)", code, flags=re.IGNORECASE | re.MULTILINE)
                except Exception:
                    return code
            try:
                import re
                # Replace arcpy.mp.ArcGISProject("CURRENT") (any quoting) with explicit path
                pattern = r"arcpy\.mp\.ArcGISProject\(\s*['\"]CURRENT['\"]\s*\)"
                replacement = f"arcpy.mp.ArcGISProject(r\"{aprx_path}\")"
                return re.sub(pattern, replacement, code, flags=re.IGNORECASE)
            except Exception:
                return code

        def _rewrite_layer_getextent(code: str) -> str:
            """
            Replace fragile Layer.getExtent() usage with Describe(...).extent.

            ArcGIS Pro's arcpy.mp Layer.getExtent is not consistently available in
            headless/propy executions (and sometimes differs by object type).
            We rewrite the most common patterns to avoid hard failures.
            """
            try:
                import re
                if not code:
                    return code

                # Pattern 1: mp.defaultCamera.setExtent(points_lyr.getExtent())
                pat_set = re.compile(
                    r"(?m)^(?P<indent>\s*)(?P<mapvar>\w+)\.defaultCamera\.setExtent\(\s*(?P<lyr>\w+)\.getExtent\(\)\s*\)\s*$"
                )

                def _repl_set(m: "re.Match") -> str:
                    indent = m.group("indent")
                    mapvar = m.group("mapvar")
                    lyr = m.group("lyr")
                    return (
                        f"{indent}# NOTE: rewritten by SurvyAI (Layer.getExtent not reliable in headless runs)\n"
                        f"{indent}try:\n"
                        f"{indent}    _ext = arcpy.Describe({lyr}).extent\n"
                        f"{indent}    if _ext:\n"
                        f"{indent}        {mapvar}.defaultCamera.setExtent(_ext)\n"
                        f"{indent}except Exception:\n"
                        f"{indent}    pass"
                    )

                code2 = pat_set.sub(_repl_set, code)

                # Pattern 2: extent = lyr.getExtent()
                pat_assign = re.compile(
                    r"(?m)^(?P<indent>\s*)(?P<lhs>\w+)\s*=\s*(?P<lyr>\w+)\.getExtent\(\)\s*$"
                )

                def _repl_assign(m: "re.Match") -> str:
                    indent = m.group("indent")
                    lhs = m.group("lhs")
                    lyr = m.group("lyr")
                    return (
                        f"{indent}# NOTE: rewritten by SurvyAI (Layer.getExtent not reliable in headless runs)\n"
                        f"{indent}{lhs} = arcpy.Describe({lyr}).extent"
                    )

                code2 = pat_assign.sub(_repl_assign, code2)
                return code2
            except Exception:
                return code

        def _rewrite_xy_field_literals(code: str) -> str:
            """
            Fix a very common LLM mistake: using lowercase 'x'/'y' field names.

            In ExcelToTable outputs, field names preserve case, and most cleaned
            coordinate exports use 'X'/'Y'. If a generated script sets x_field/y_field
            to literal 'x'/'y', it will often insert zero points or fail.
            """
            if not code:
                return code
            try:
                import re
                code2 = re.sub(r"(?m)^(\s*x_field\s*=\s*['\"])x(['\"]\s*)$", r"\1X\2", code)
                code2 = re.sub(r"(?m)^(\s*y_field\s*=\s*['\"])y(['\"]\s*)$", r"\1Y\2", code2)
                return code2
            except Exception:
                return code
        
        # Determine script save location
        if project_path:
            project_file = Path(project_path).resolve()
            if project_file.exists() and project_file.suffix.lower() == ".aprx":
                scripts_dir = project_file.parent / "scripts"
                scripts_dir.mkdir(parents=True, exist_ok=True)
                working_dir = scripts_dir
            else:
                working_dir = Path.cwd() / "scripts"
                working_dir.mkdir(parents=True, exist_ok=True)
        else:
            working_dir = Path.cwd() / "scripts"
            working_dir.mkdir(parents=True, exist_ok=True)

        # Apply rewrite using the best available aprx path
        aprx_for_rewrite = None
        if project_path:
            aprx_for_rewrite = str(Path(project_path).resolve())
        elif getattr(self, "current_project", None):
            try:
                cp = str(Path(self.current_project).resolve())
                if cp.lower().endswith(".aprx") and Path(cp).exists():
                    aprx_for_rewrite = cp
            except Exception:
                pass
        python_code = _rewrite_current_aprx(python_code, aprx_for_rewrite)
        python_code = _rewrite_layer_getextent(python_code)
        python_code = _rewrite_xy_field_literals(python_code)
        
        # Generate script filename
        if not script_name:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            script_name = f"survyai_script_{timestamp}.py"
        if not script_name.endswith(".py"):
            script_name += ".py"
        
        script_path = (working_dir / script_name).resolve()
        
        # Add header comment to the code
        script_content = f"""# Generated by SurvyAI
# This script was dynamically generated based on user request
# Generated: {datetime.datetime.now().isoformat()}
# 
# Execute this script in ArcGIS Pro's Python Window if needed:
# exec(open(r'{script_path}').read())

{python_code}
"""
        
        # Save the script
        script_path.write_text(script_content, encoding="utf-8")
        logger.info(f"Saved dynamically generated script to: {script_path}")
        
        # Execute automatically if requested
        if execute_automatically:
            # Optional: prelaunch ArcGIS Pro before headless execution (off by default).
            prelaunch = None
            if prelaunch_arcgis_pro and launch_arcgis_pro:
                try:
                    if project_path:
                        prelaunch = self.launch_arcgis_pro(project_path=project_path)
                    else:
                        prelaunch = self.launch_arcgis_pro()
                    time.sleep(3)
                except Exception as e:
                    logger.debug("Pre-launch ArcGIS Pro failed (continuing headless): %s", e)

            run_result = self._run_propy_script(
                script_content=script_content,
                working_dir=working_dir,
                script_name=script_name,
            )
            
            if run_result.get("success"):
                # Parse computational results from stdout
                stdout = run_result.get("stdout", "")
                results = self._parse_arcgis_results(stdout)
                
                # Finalize project visualization AFTER user operations complete (optional, non-recursive)
                finalize_result = None
                if project_path and (not skip_finalization):
                    try:
                        finalize_result = self.finalize_project_visualization(
                            project_path=project_path,
                            load_basemap=True,
                            basemap_name="Imagery Hybrid",
                            load_geodatabase=True,
                            # prevent recursion / duplicate launches
                            launch_arcgis_pro=False,
                        )
                    except Exception as e:
                        logger.warning("Failed to finalize project visualization: %s", e)
                        finalize_result = {"success": False, "error": str(e)}
                
                # If execution succeeded, also launch ArcGIS Pro with project if provided
                # If we already prelaunched, prefer that; otherwise launch now.
                launch_result = None
                if launch_arcgis_pro:
                    if prelaunch is not None:
                        launch_result = prelaunch
                    else:
                        if project_path:
                            launch_result = self.launch_arcgis_pro(project_path=project_path)
                        else:
                            launch_result = self.launch_arcgis_pro()
                
                return {
                    "success": True,
                    "message": "ArcGIS workflow executed successfully - results extracted automatically",
                    "script_path": str(script_path),
                    "stdout": stdout,
                    "stderr": run_result.get("stderr", ""),
                    "returncode": run_result.get("returncode", 0),
                    "arcgis_launched": launch_result.get("success", False) if isinstance(launch_result, dict) else False,
                    "results": results,
                    "visualization_finalized": finalize_result.get("success", False) if finalize_result else None,
                    "visualization_details": finalize_result,
                    "note": "Complete workflow executed automatically. Results parsed from output. No manual steps required.",
                }
            else:
                # Heuristic auto-retry for common ArcGIS GP parameter pitfalls.
                # Example: ERROR 001017 for MinimumBoundingGeometry group_option='NONE' on point inputs.
                stdout = run_result.get("stdout", "") or ""
                stderr = run_result.get("stderr", "") or ""
                combined = f"{stdout}\n{stderr}"

                if "ERROR 001017" in combined and "MinimumBoundingGeometry" in combined and "NONE" in combined.upper():
                    try:
                        import re

                        def _rewrite_mbg_group_option_to_all(src: str) -> str:
                            # Replace group_option='NONE' / "NONE" with 'ALL' for MBG calls.
                            out = re.sub(
                                r"(MinimumBoundingGeometry\s*\([\s\S]*?group_option\s*=\s*)(['\"])NONE\2",
                                r"\1'ALL'",
                                src,
                                flags=re.IGNORECASE,
                            )
                            # Also handle positional 4th argument: MinimumBoundingGeometry(..., 'CONVEX_HULL', 'NONE')
                            out = re.sub(
                                r"(MinimumBoundingGeometry\s*\([\s\S]*?['\"])CONVEX_HULL(['\"]\s*,\s*)(['\"])NONE\3",
                                r"\1CONVEX_HULL\2'ALL'",
                                out,
                                flags=re.IGNORECASE,
                            )
                            return out

                        rewritten = _rewrite_mbg_group_option_to_all(script_content)
                        if rewritten != script_content:
                            retry_name = script_name.replace(".py", "_retry.py")
                            retry_path = (working_dir / retry_name).resolve()
                            retry_path.write_text(rewritten, encoding="utf-8")
                            logger.info("Retrying ArcGIS script after rewriting MBG group_option NONE->ALL: %s", retry_path)

                            retry_result = self._run_propy_script(
                                script_content=rewritten,
                                working_dir=working_dir,
                                script_name=retry_name,
                            )
                            if retry_result.get("success"):
                                retry_stdout = retry_result.get("stdout", "") or ""
                                results = self._parse_arcgis_results(retry_stdout)
                                return {
                                    "success": True,
                                    "message": "ArcGIS workflow succeeded after auto-fix (MBG group_option NONE->ALL)",
                                    "script_path": str(retry_path),
                                    "stdout": retry_stdout,
                                    "stderr": retry_result.get("stderr", ""),
                                    "returncode": retry_result.get("returncode", 0),
                                    "arcgis_launched": (prelaunch or {}).get("success", False),
                                    "results": results,
                                    "note": "Initial run failed with ERROR 001017; auto-rewritten and retried successfully.",
                                }
                    except Exception as e:
                        logger.debug("Auto-retry for MBG group_option failed: %s", e)

                # Execution failed - return error but still provide script location
                return {
                    "success": False,
                    "error": f"Script execution failed: {run_result.get('error', 'Unknown error')}",
                    "script_path": str(script_path),
                    "stdout": stdout,
                    "stderr": stderr,
                    "returncode": run_result.get("returncode", -1),
                    "instructions": (
                        f"Script execution failed automatically. You can try running it manually:\n"
                        f"1. Open ArcGIS Pro\n"
                        f"2. Go to Analysis > Python > Python Window\n"
                        f"3. Type: exec(open(r'{script_path}').read())\n"
                        f"4. Press Enter"
                    ),
                }
        else:
            # Don't execute automatically - just save and provide instructions
            if project_path:
                launch_result = self.launch_arcgis_pro(project_path=project_path)
            else:
                launch_result = self.launch_arcgis_pro()
            
            instructions = (
                f"I've saved the dynamically generated Python code to a script file.\n\n"
                f"To run the script:\n"
                f"1. In ArcGIS Pro, go to the Analysis tab\n"
                f"2. Click Python > Python Window (or press Ctrl+Alt+P)\n"
                f"3. In the Python window, type: exec(open(r'{script_path}').read())\n"
                f"4. Press Enter to execute\n\n"
                f"Script location: {script_path}"
            )
            
            return {
                "success": True,
                "message": "Python code saved to script file. Please run it in ArcGIS Pro's Python window.",
                "script_path": str(script_path),
                "instructions": instructions,
                "arcgis_launched": launch_result.get("success", False),
            }

    # --------------------------------------------------------------------------
    # High-level, data-driven workflow (preferred over free-form code generation)
    # --------------------------------------------------------------------------
    def excel_points_convex_hull_traverse(
        self,
        excel_path: str,
        project_name: str,
        project_folder: Optional[str],
        coordinate_system: str,
        output_csv: str,
        sheet_name: Optional[str] = None,
        close_traverse: bool = True,
        clean_project_layers: bool = True,
    ) -> Dict[str, Any]:
        """
        End-to-end, verified workflow:
        - Create/open an ArcGIS Pro project in a user-specified folder
        - Import points from Excel (expects canonical X/Y columns; auto-detects if needed)
        - Create convex hull polygon, compute area
        - Compute traverse distances & bearings (north-clockwise, 0-360)
        - Export results to CSV (verified on disk)

        IMPORTANT: This method is deterministic and verifies outputs. It avoids
        hallucinated "done" claims by checking inserted point counts and file existence.
        """
        from pathlib import Path
        import json
        import tempfile

        excel_file = Path(excel_path)
        if not excel_file.exists():
            return {"success": False, "error": f"Excel file not found: {excel_path}"}

        out_csv = Path(output_csv)
        if not out_csv.is_absolute():
            out_csv = (excel_file.parent / out_csv.name).resolve()

        # Create project in the requested folder (default: same folder as Excel file)
        proj_base = Path(project_folder).resolve() if project_folder else excel_file.parent.resolve()
        create = self.create_project(
            project_name=project_name,
            project_path=str(proj_base),
            coordinate_system=coordinate_system,
            template="MAP",
            clean_layers=clean_project_layers,
        )
        if not create.get("success"):
            return {"success": False, "error": create.get("error", "Failed to create project"), "details": create}

        project_path = create.get("project_path")
        if not project_path:
            return {"success": False, "error": "Project creation did not return project_path", "details": create}

        # Build robust arcpy workflow as code string to run via propy.bat
        # NOTE: keep it generic; avoid relying on map UI-only methods.
        sheet_clause = f", sheet_name={json.dumps(sheet_name)}" if sheet_name else ""
        code = f"""
import os, math, csv
import arcpy

arcpy.env.overwriteOutput = True

project_path = r\"{project_path}\"
excel_path = r\"{str(excel_file)}\"
out_csv = r\"{str(out_csv)}\"

sr_info = {json.dumps(parse_coordinate_system(coordinate_system))}
wkid = sr_info.get("wkid") or None
sr = arcpy.SpatialReference(int(wkid)) if wkid else None

aprx = arcpy.mp.ArcGISProject(project_path)
maps = aprx.listMaps()
mp = maps[0] if maps else None
if mp and sr:
    mp.spatialReference = sr

project_dir = os.path.dirname(project_path)
gdb_path = os.path.join(project_dir, os.path.splitext(os.path.basename(project_path))[0] + ".gdb")
if not arcpy.Exists(gdb_path):
    arcpy.management.CreateFileGDB(project_dir, os.path.basename(gdb_path))

# Excel -> table
excel_table = os.path.join(gdb_path, "points_table")
if arcpy.Exists(excel_table):
    arcpy.management.Delete(excel_table)
arcpy.conversion.ExcelToTable(excel_path, excel_table{sheet_clause})

fields = [f.name for f in arcpy.ListFields(excel_table)]

def _pick_xy(fields):
    # Prefer canonical X/Y (from SurvyAI clean conversion)
    if "X" in fields and "Y" in fields:
        return "X", "Y"
    # Otherwise, attempt a simple heuristic
    x = next((f for f in fields if "east" in f.lower() or "lon" in f.lower() or f.lower().endswith("x")), None)
    y = next((f for f in fields if "north" in f.lower() or "lat" in f.lower() or f.lower().endswith("y")), None)
    return x, y

x_field, y_field = _pick_xy(fields)
if not x_field or not y_field:
    raise RuntimeError("Could not determine X/Y fields. Fields found: " + ",".join(fields))

points_fc = os.path.join(gdb_path, "points")
if arcpy.Exists(points_fc):
    arcpy.management.Delete(points_fc)

# Preserve Excel attributes:
# Create the point feature class using the Excel table as a template so all fields are copied over.
arcpy.management.CreateFeatureclass(gdb_path, "points", "POINT", template=excel_table, spatial_reference=sr)
# Add SrcOID for stable ordering (if it doesn't already exist)
existing_fc_fields = [f.name for f in arcpy.ListFields(points_fc)]
if "SrcOID" not in existing_fc_fields:
    arcpy.management.AddField(points_fc, "SrcOID", "LONG")

# Determine attribute fields we will copy from the Excel table
attr_fields = [f.name for f in arcpy.ListFields(excel_table) if f.type not in ("OID", "Geometry")]
# Avoid duplicates in cursors; we handle X/Y separately
attr_fields_no_xy = [f for f in attr_fields if f not in (x_field, y_field)]

inserted = 0
search_fields = ["OID@", x_field, y_field] + attr_fields_no_xy
insert_fields = ["SrcOID", "SHAPE@XY", x_field, y_field] + attr_fields_no_xy
with arcpy.da.SearchCursor(excel_table, search_fields) as sc, \
     arcpy.da.InsertCursor(points_fc, insert_fields) as ic:
    for row in sc:
        oid = row[0]
        x_raw = row[1]
        y_raw = row[2]
        other_vals = list(row[3:]) if len(row) > 3 else []
        if x_raw is None or y_raw is None:
            continue
        try:
            x = float(str(x_raw).replace(",", "").strip())
            y = float(str(y_raw).replace(",", "").strip())
        except Exception:
            continue
        # Write geometry + keep X/Y attributes + copy the rest of the attributes
        ic.insertRow((int(oid), (x, y), x, y, *other_vals))
        inserted += 1

print("RESULT_POINTS_INSERTED:", inserted)
print("RESULT_X_FIELD:", x_field)
print("RESULT_Y_FIELD:", y_field)
print("RESULT_GDB:", gdb_path)
print("RESULT_POINTS_FC:", points_fc)

if inserted < 2:
    raise RuntimeError("No usable points were inserted; cannot compute traverse/hull.")

# Add layer to map (best-effort) and set extent (headless-safe)
if mp:
    try:
        mp.addDataFromPath(points_fc)
    except Exception:
        pass
    try:
        ext = arcpy.Describe(points_fc).extent
        if ext:
            mp.defaultCamera.setExtent(ext)
    except Exception:
        pass

# Convex hull & area (requires 3+ points)
area_m2 = None
polygon_fc = os.path.join(gdb_path, "points_hull")
if arcpy.Exists(polygon_fc):
    arcpy.management.Delete(polygon_fc)

if inserted >= 3:
    arcpy.management.MinimumBoundingGeometry(points_fc, polygon_fc, "CONVEX_HULL", group_option="ALL")
    with arcpy.da.SearchCursor(polygon_fc, ["SHAPE@"]) as cur:
        for (geom,) in cur:
            if geom:
                try:
                    area_m2 = geom.getArea("PLANAR", "SQUAREMETERS")
                except Exception:
                    area_m2 = geom.area
            break

print("RESULT_AREA_M2:", area_m2 if area_m2 is not None else "NA")
print("RESULT_HULL_FC:", polygon_fc if arcpy.Exists(polygon_fc) else "NA")

# Add hull layer to map (best-effort)
if mp and arcpy.Exists(polygon_fc):
    try:
        mp.addDataFromPath(polygon_fc)
    except Exception:
        pass

# Traverse legs
order_field = "SrcOID" if any(f.name == "SrcOID" for f in arcpy.ListFields(points_fc)) else "OBJECTID"
pts = []
_sql = (None, "ORDER BY " + str(order_field))
with arcpy.da.SearchCursor(points_fc, [order_field, "SHAPE@XY"], sql_clause=_sql) as cur:
    for oid, xy in cur:
        if not xy:
            continue
        pts.append((int(oid), float(xy[0]), float(xy[1])))

rows = []
if len(pts) >= 2:
    n = len(pts)
    last_index = n if {str(close_traverse)} else n - 1
    for i in range(last_index):
        oid1, x1, y1 = pts[i]
        oid2, x2, y2 = pts[(i + 1) % n]
        dx = x2 - x1
        dy = y2 - y1
        dist = math.hypot(dx, dy)
        bearing = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0
        rows.append((i + 1, oid1, oid2, dist, bearing, area_m2))

# Create a traverse polyline feature class for visualization (best-effort)
traverse_fc = os.path.join(gdb_path, "traverse_lines")
try:
    if arcpy.Exists(traverse_fc):
        arcpy.management.Delete(traverse_fc)
    arcpy.management.CreateFeatureclass(gdb_path, "traverse_lines", "POLYLINE", spatial_reference=sr)
    for fname, ftype in [("LegID", "LONG"), ("FromOID", "LONG"), ("ToOID", "LONG"), ("Distance_m", "DOUBLE"), ("Bearing_deg", "DOUBLE")]:
        if fname not in [f.name for f in arcpy.ListFields(traverse_fc)]:
            arcpy.management.AddField(traverse_fc, fname, ftype)
    with arcpy.da.InsertCursor(traverse_fc, ["SHAPE@", "LegID", "FromOID", "ToOID", "Distance_m", "Bearing_deg"]) as ic:
        for leg_id, from_oid, to_oid, dist, bearing, _ in rows:
            # Build line geometry
            p1 = next((p for p in pts if p[0] == from_oid), None)
            p2 = next((p for p in pts if p[0] == to_oid), None)
            if not p1 or not p2:
                continue
            arr = arcpy.Array([arcpy.Point(p1[1], p1[2]), arcpy.Point(p2[1], p2[2])])
            geom = arcpy.Polyline(arr, sr)
            ic.insertRow((geom, int(leg_id), int(from_oid), int(to_oid), float(dist), float(bearing)))
    print("RESULT_TRAVERSE_FC:", traverse_fc)
    if mp:
        try:
            mp.addDataFromPath(traverse_fc)
        except Exception:
            pass
except Exception as e:
    print("RESULT_TRAVERSE_WARNING:", str(e))

with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["LegID", "FromOID", "ToOID", "Distance_m", "Bearing_deg", "HullArea_m2"])
    for r in rows:
        w.writerow(list(r))

print("RESULT_OUTPUT_CSV:", out_csv)
aprx.save()
"""

        run = self.execute_python_code(
            python_code=code,
            project_path=str(project_path),
            script_name=f"{project_name}_points_hull_traverse.py",
            execute_automatically=True,
            # Avoid opening ArcGIS Pro multiple times (we will open once after finalization)
            launch_arcgis_pro=False,
            prelaunch_arcgis_pro=False,
            skip_finalization=True,
        )

        if not run.get("success"):
            return {"success": False, "error": run.get("error", "ArcGIS execution failed"), "details": run}

        # Verified outputs
        csv_exists = out_csv.exists() and out_csv.stat().st_size > 0
        
        # Finalize project visualization AFTER user operations complete
        # This ensures basemap and geodatabase are loaded so user can visually verify results
        finalize_result = self.finalize_project_visualization(
            project_path=project_path,
            load_basemap=True,
            basemap_name="Imagery Hybrid",
            load_geodatabase=True,
        )

        # Open ArcGIS Pro exactly once for visual verification
        launch_result = self.launch_arcgis_pro(project_path=project_path)
        
        return {
            "success": True,
            "project_path": project_path,
            "output_csv": str(out_csv),
            "output_csv_exists": csv_exists,
            "arcgis_results": run.get("results", {}),
            "stdout": run.get("stdout", ""),
            "visualization_finalized": finalize_result.get("success", False),
            "visualization_details": finalize_result,
            "arcgis_launched": launch_result.get("success", False),
            "arcgis_launch_details": launch_result,
            "note": "Workflow executed and outputs were verified on disk." if csv_exists else "ArcGIS run succeeded but CSV was not found or empty.",
        }
    
    def _run_propy_script(self, script_content: str, working_dir: Path, script_name: str) -> Dict[str, Any]:
        """
        Run a small arcpy script using ArcGIS Pro's propy.bat (headless geoprocessing).
        """
        try:
            if not self.arcgis_pro_path:
                return {"success": False, "error": "ArcGIS Pro path not set/detected."}

            propy_bat = self.arcgis_pro_path / "bin" / "Python" / "Scripts" / "propy.bat"
            if not propy_bat.exists():
                return {"success": False, "error": f"propy.bat not found at: {propy_bat}"}

            working_dir.mkdir(parents=True, exist_ok=True)
            script_path = (working_dir / script_name).resolve()
            script_path.write_text(script_content, encoding="utf-8")

            # IMPORTANT (Windows):
            # Running a .bat via subprocess with shell=True is fragile and can silently drop arguments,
            # especially under different shells. Use cmd.exe /c for deterministic behavior and reliable
            # stdout/stderr capture.
            # Windows reliability note:
            # - Executing .bat files with arguments through different shells is surprisingly brittle.
            # - ArcGIS' propy.bat lives under a path with spaces, and we've observed cmd.exe quoting
            #   edge-cases leading to "silent" no-op runs.
            # - PowerShell's call operator (&) handles this robustly.
            cmd = [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                f"& '{str(propy_bat)}' '{str(script_path)}'",
            ]
            result = subprocess.run(
                cmd,
                shell=False,
                cwd=str(working_dir),
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout for complex operations
            )
            ok = result.returncode == 0
            return {
                "success": ok,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "script": str(script_path),
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Script execution timed out after 10 minutes",
                "script": str(working_dir / script_name),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _parse_arcgis_results(self, stdout: str) -> Dict[str, Any]:
        """
        Parse computational results from ArcGIS script stdout.
        Looks for lines starting with 'RESULT_' and extracts structured data.
        """
        results = {}
        if not stdout:
            return results
        
        lines = stdout.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('RESULT_'):
                try:
                    # Expected format: RESULT_AREA: 12345.67 square_meters
                    # or: RESULT_BEARING_P1_P2: 45.5 degrees
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        key = parts[0].replace('RESULT_', '').lower()
                        value_part = parts[1].strip()
                        
                        # Try to extract numeric value and unit
                        value_tokens = value_part.split()
                        if value_tokens:
                            try:
                                numeric_value = float(value_tokens[0])
                                unit = value_tokens[1] if len(value_tokens) > 1 else ""
                                results[key] = {
                                    "value": numeric_value,
                                    "unit": unit,
                                    "formatted": value_part
                                }
                            except ValueError:
                                # If not numeric, store as text
                                results[key] = {"text": value_part}
                except Exception as e:
                    logger.debug(f"Error parsing result line '{line}': {e}")
                    continue
        
        return results
    
    def finalize_project_visualization(
        self,
        project_path: str,
        load_basemap: bool = True,
        basemap_name: str = "Imagery Hybrid",
        load_geodatabase: bool = True,
        launch_arcgis_pro: bool = False,
    ) -> Dict[str, Any]:
        """
        Finalize ArcGIS Pro project visualization after user operations complete.
        
        This function is called AFTER all user-requested operations have been executed
        to ensure the project is visually ready for inspection:
        - Adds 'Imagery Hybrid' basemap to all maps
        - Loads the native geodatabase (project_dir/project_name.gdb) and all its feature classes
        
        IMPORTANT: This should be called AFTER user operations complete, not during project creation,
        so users can visually verify that their instructions were properly carried out.
        
        Args:
            project_path: Path to the .aprx project file
            load_basemap: If True, add basemap to all maps (default: True)
            basemap_name: Name of basemap to add (default: "Imagery Hybrid")
            load_geodatabase: If True, load native geodatabase and all feature classes (default: True)
            
        Returns:
            Dictionary with success status and details about what was loaded
        """
        if not self.arcgis_pro_path:
            return {
                "success": False,
                "error": "ArcGIS Pro installation not found",
            }
        
        project_file = Path(project_path)
        if not project_file.exists() or project_file.suffix.lower() != ".aprx":
            return {
                "success": False,
                "error": f"Project file not found or invalid: {project_path}",
            }
        
        # Build arcpy script to finalize visualization
        project_dir = project_file.parent
        project_name = project_file.stem
        gdb_path = project_dir / f"{project_name}.gdb"
        
        load_basemap_py = bool(load_basemap)
        load_gdb_py = bool(load_geodatabase)

        code = f"""
import os
import arcpy

project_path = r\"{str(project_file)}\"
project_dir = r\"{str(project_dir)}\"
gdb_path = r\"{str(gdb_path)}\"

aprx = arcpy.mp.ArcGISProject(project_path)
maps = aprx.listMaps()

basemap_loaded = False
gdb_loaded = False
feature_classes_loaded = []

# Prefer the project's own geodatabase as the default, so it appears under Databases
try:
    if os.path.exists(gdb_path):
        aprx.defaultGeodatabase = gdb_path
except Exception:
    pass

# Add a folder connection to the project directory (so the .gdb is easy to find in Catalog)
try:
    aprx.addFolderConnection(project_dir)
except Exception:
    pass

# Load basemap
if {load_basemap_py}:
    for m in maps:
        try:
            # Try common basemap name variations
            basemap_names = [
                "{basemap_name}",
                "Imagery Hybrid",
                "ImageryHybrid",
                "Imagery_Hybrid",
                "ImageHybrid",
                "Image_Hybrid",
            ]
            for bm_name in basemap_names:
                try:
                    m.addBasemap(bm_name)
                    basemap_loaded = True
                    print("RESULT_BASEMAP_LOADED:", bm_name, "on map:", m.name)
                    break
                except Exception:
                    continue
            if not basemap_loaded:
                # Fallback: try to list available basemaps and use first imagery hybrid one
                try:
                    available = m.listBasemaps()
                    for bm in available:
                        if "imagery" in bm.name.lower() and "hybrid" in bm.name.lower():
                            m.addBasemap(bm.name)
                            basemap_loaded = True
                            print("RESULT_BASEMAP_LOADED:", bm.name, "on map:", m.name)
                            break
                except Exception as e:
                    print("RESULT_BASEMAP_WARNING:", str(e))
        except Exception as e:
            print("RESULT_BASEMAP_ERROR:", str(e))

# Load native geodatabase and all feature classes
if {load_gdb_py} and os.path.exists(gdb_path):
    for m in maps:
        try:
            # List all feature classes in the geodatabase
            fcs = []
            for dirpath, dirnames, filenames in arcpy.da.Walk(gdb_path, datatype="FeatureClass"):
                for filename in filenames:
                    fc_path = os.path.join(dirpath, filename)
                    if arcpy.Exists(fc_path):
                        fcs.append(fc_path)
            
            # Add each feature class to the map (if not already present)
            for fc_path in fcs:
                try:
                    # Check if layer already exists
                    fc_name = os.path.basename(fc_path)
                    existing = [lyr.name for lyr in m.listLayers() if lyr.name == fc_name]
                    if not existing:
                        m.addDataFromPath(fc_path)
                        feature_classes_loaded.append(fc_name)
                        print("RESULT_FC_LOADED:", fc_name, "on map:", m.name)
                except Exception as e:
                    print("RESULT_FC_WARNING:", fc_name, ":", str(e))
            
            if fcs:
                gdb_loaded = True
                print("RESULT_GDB_LOADED:", gdb_path, "with", len(fcs), "feature classes")
        except Exception as e:
            print("RESULT_GDB_ERROR:", str(e))

aprx.save()
print("RESULT_FINALIZATION_COMPLETE")
"""
        
        # Execute the finalization script
        result = self.execute_python_code(
            python_code=code,
            project_path=str(project_file),
            script_name=f"{project_name}_finalize_visualization.py",
            execute_automatically=True,
            # Prevent recursion and extra windows; caller decides whether to open ArcGIS Pro
            launch_arcgis_pro=bool(launch_arcgis_pro),
            prelaunch_arcgis_pro=False,
            skip_finalization=True,
        )
        
        if not result.get("success"):
            return {
                "success": False,
                "error": result.get("error", "Failed to finalize project visualization"),
                "details": result,
            }
        
        # Parse results from stdout
        stdout = result.get("stdout", "")
        basemap_status = "loaded" if "RESULT_BASEMAP_LOADED" in stdout else "not loaded"
        gdb_status = "loaded" if "RESULT_GDB_LOADED" in stdout else "not found or empty"
        
        # Extract feature class names
        fc_names = []
        for line in stdout.split("\n"):
            if "RESULT_FC_LOADED:" in line:
                try:
                    parts = line.split("RESULT_FC_LOADED:")[1].strip().split()
                    if parts:
                        fc_names.append(parts[0])
                except Exception:
                    pass
        
        return {
            "success": True,
            "message": "Project visualization finalized",
            "project_path": str(project_file),
            "basemap_loaded": basemap_status == "loaded",
            "basemap_name": basemap_name,
            "geodatabase_loaded": gdb_status == "loaded",
            "geodatabase_path": str(gdb_path),
            "feature_classes_loaded": fc_names,
            "feature_class_count": len(fc_names),
            "stdout": stdout,
        }
    
    def get_project_info(self) -> Dict[str, Any]:
        """
        Get information about the current project.
        
        Returns:
            Dictionary with project details
        """
        if not self.arcpy:
            return {
                "success": False,
                "error": "arcpy not available - cannot get project info",
                "arcgis_installed": self.is_installed,
                "arcgis_path": str(self.arcgis_pro_path) if self.arcgis_pro_path else None
            }
        
        if not self.current_project:
            return {
                "success": False,
                "error": "No project is currently open"
            }
        
        try:
            aprx = self.arcpy.mp.ArcGISProject(self.current_project)
            
            maps_info = []
            for m in aprx.listMaps():
                sr = m.spatialReference
                maps_info.append({
                    "name": m.name,
                    "type": m.mapType,
                    "coordinate_system": {
                        "name": sr.name if sr else "Unknown",
                        "wkid": sr.factoryCode if sr else None,
                        "type": "geographic" if (sr and sr.type == "Geographic") else "projected"
                    } if sr else None
                })
            
            return {
                "success": True,
                "project_path": self.current_project,
                "maps": maps_info,
                "map_count": len(maps_info)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _parse_arcgis_results(self, stdout: str) -> Dict[str, Any]:
        """
        Parse computational results from ArcGIS script stdout.
        Looks for lines starting with 'RESULT_' and extracts structured data.
        """
        results = {}
        if not stdout:
            return results
        
        lines = stdout.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('RESULT_'):
                try:
                    # Expected format: RESULT_AREA: 12345.67 square_meters
                    # or: RESULT_BEARING_P1_P2: 45.5 degrees
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        key = parts[0].replace('RESULT_', '').lower()
                        value_part = parts[1].strip()
                        
                        # Try to extract numeric value and unit
                        value_tokens = value_part.split()
                        if value_tokens:
                            try:
                                numeric_value = float(value_tokens[0])
                                unit = value_tokens[1] if len(value_tokens) > 1 else ""
                                results[key] = {
                                    "value": numeric_value,
                                    "unit": unit,
                                    "formatted": value_part
                                }
                            except ValueError:
                                # If not numeric, store as text
                                results[key] = {"text": value_part}
                except Exception as e:
                    logger.debug(f"Error parsing result line '{line}': {e}")
                    continue
        
        return results
    
    def list_coordinate_systems(self, filter_text: Optional[str] = None) -> Dict[str, Any]:
        """
        List available coordinate systems.
        
        Args:
            filter_text: Optional text to filter coordinate systems
            
        Returns:
            Dictionary with available coordinate systems
        """
        systems = []
        for name, info in COORDINATE_SYSTEMS.items():
            if filter_text:
                if filter_text.lower() not in name.lower() and filter_text.lower() not in info["name"].lower():
                    continue
            systems.append({
                "short_name": name,
                "full_name": info["name"],
                "wkid": info["wkid"],
                "type": info["type"]
            })
        
        return {
            "success": True,
            "count": len(systems),
            "coordinate_systems": systems
        }
    
    # ==========================================================================
    # ANALYSIS METHODS (require arcpy)
    # ==========================================================================
    
    def compute_volume_idw(
        self,
        point_features: str,
        elevation_field: str,
        output_raster: str,
        cell_size: float = 1.0,
        power: float = 2.0,
        search_radius: Optional[float] = None,
        base_elevation: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Compute volume using Inverse Distance Weighting (IDW) interpolation.
        
        Args:
            point_features: Path to point feature class or shapefile
            elevation_field: Field name containing elevation values
            output_raster: Path for output raster
            cell_size: Cell size for output raster
            power: Power parameter for IDW (default 2.0)
            search_radius: Search radius for interpolation
            base_elevation: Base elevation for volume calculation (cut/fill reference)
            
        Returns:
            Dictionary with volume calculation results
        """
        if not self.arcpy:
            return {
                "success": False,
                "error": "arcpy is not available. Run from ArcGIS Pro Python environment for this operation."
            }
        
        try:
            # Check if point features exist
            if not self.arcpy.Exists(point_features):
                raise FileNotFoundError(f"Point features not found: {point_features}")
            
            # Create IDW raster
            logger.info(f"Creating IDW raster from {point_features}")
            
            # Set up IDW parameters
            idw_params = {
                "in_point_features": point_features,
                "z_field": elevation_field,
                "out_raster": output_raster,
                "cell_size": cell_size,
                "power": power
            }
            
            if search_radius:
                idw_params["search_radius"] = search_radius
            
            # Execute IDW
            idw_raster = self.arcpy.sa.Idw(**idw_params)
            idw_raster.save(output_raster)
            
            logger.info(f"IDW raster created: {output_raster}")
            
            # Calculate volume if base elevation is provided
            volume_result = None
            if base_elevation is not None:
                # Create base elevation raster
                base_raster = self.arcpy.sa.CreateConstantRaster(
                    base_elevation,
                    "FLOAT",
                    cell_size,
                    idw_raster.extent
                )
                
                # Calculate difference (cut/fill)
                diff_raster = idw_raster - base_raster
                
                # Calculate volume
                cell_area = cell_size * cell_size
                volume_raster = diff_raster * cell_area
                
                # Get statistics
                volume_stats = self.arcpy.GetRasterProperties_management(
                    volume_raster,
                    "SUM"
                )
                total_volume = float(volume_stats.getOutput(0))
                
                # Calculate cut and fill separately
                cut_raster = self.arcpy.sa.Con(diff_raster < 0, abs(diff_raster) * cell_area, 0)
                fill_raster = self.arcpy.sa.Con(diff_raster > 0, diff_raster * cell_area, 0)
                
                cut_stats = self.arcpy.GetRasterProperties_management(cut_raster, "SUM")
                fill_stats = self.arcpy.GetRasterProperties_management(fill_raster, "SUM")
                
                cut_volume = float(cut_stats.getOutput(0))
                fill_volume = float(fill_stats.getOutput(0))
                
                volume_result = {
                    "total_volume": total_volume,
                    "cut_volume": cut_volume,
                    "fill_volume": fill_volume,
                    "net_volume": fill_volume - cut_volume,
                    "base_elevation": base_elevation,
                    "unit": "cubic units"
                }
            
            # Get raster statistics
            raster_stats = self.arcpy.GetRasterProperties_management(
                idw_raster,
                ["MINIMUM", "MAXIMUM", "MEAN", "STD"]
            )
            
            result = {
                "success": True,
                "output_raster": output_raster,
                "raster_statistics": {
                    "minimum": float(raster_stats.getOutput(0)),
                    "maximum": float(raster_stats.getOutput(1)),
                    "mean": float(raster_stats.getOutput(2)),
                    "std_dev": float(raster_stats.getOutput(3))
                },
                "parameters": {
                    "cell_size": cell_size,
                    "power": power,
                    "search_radius": search_radius
                }
            }
            
            if volume_result:
                result["volume"] = volume_result
            
            logger.info("Volume computation completed successfully")
            return result
        
        except Exception as e:
            logger.error(f"Error computing volume with IDW: {e}")
            return {"success": False, "error": str(e)}
    
    def compute_cutfill(
        self,
        before_surface: str,
        after_surface: str,
        output_raster: Optional[str] = None,
        cell_size: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Compute cut/fill analysis between two surfaces.
        
        Args:
            before_surface: Path to before surface raster or feature class
            after_surface: Path to after surface raster or feature class
            output_raster: Path for output cut/fill raster (optional)
            cell_size: Cell size for output (optional, uses input if not specified)
            
        Returns:
            Dictionary with cut/fill analysis results
        """
        if not self.arcpy:
            return {
                "success": False,
                "error": "arcpy is not available. Run from ArcGIS Pro Python environment for this operation."
            }
        
        try:
            # Check if surfaces exist
            if not self.arcpy.Exists(before_surface):
                raise FileNotFoundError(f"Before surface not found: {before_surface}")
            if not self.arcpy.Exists(after_surface):
                raise FileNotFoundError(f"After surface not found: {after_surface}")
            
            logger.info("Computing cut/fill analysis")
            
            # Convert to rasters if needed
            before_raster = self._ensure_raster(before_surface)
            after_raster = self._ensure_raster(after_surface)
            
            # Calculate difference
            diff_raster = after_raster - before_raster
            
            # Set cell size if specified
            if cell_size:
                diff_raster = self.arcpy.sa.Resample(
                    diff_raster,
                    cell_size,
                    "NEAREST"
                )
            
            # Save output if specified
            if output_raster:
                diff_raster.save(output_raster)
                logger.info(f"Cut/fill raster saved: {output_raster}")
            
            # Calculate volumes
            cell_x = self.arcpy.GetRasterProperties_management(diff_raster, "CELLSIZEX")
            cell_y = self.arcpy.GetRasterProperties_management(diff_raster, "CELLSIZEY")
            cell_size_actual = float(cell_x.getOutput(0))
            cell_area = cell_size_actual * cell_size_actual
            
            # Calculate cut (negative values)
            cut_raster = self.arcpy.sa.Con(diff_raster < 0, abs(diff_raster) * cell_area, 0)
            fill_raster = self.arcpy.sa.Con(diff_raster > 0, diff_raster * cell_area, 0)
            
            # Get statistics
            cut_stats = self.arcpy.GetRasterProperties_management(cut_raster, "SUM")
            fill_stats = self.arcpy.GetRasterProperties_management(fill_raster, "SUM")
            
            cut_volume = float(cut_stats.getOutput(0))
            fill_volume = float(fill_stats.getOutput(0))
            
            # Get difference statistics
            diff_stats = self.arcpy.GetRasterProperties_management(
                diff_raster,
                ["MINIMUM", "MAXIMUM", "MEAN", "STD"]
            )
            
            result = {
                "success": True,
                "cut_volume": cut_volume,
                "fill_volume": fill_volume,
                "net_volume": fill_volume - cut_volume,
                "difference_statistics": {
                    "minimum": float(diff_stats.getOutput(0)),
                    "maximum": float(diff_stats.getOutput(1)),
                    "mean": float(diff_stats.getOutput(2)),
                    "std_dev": float(diff_stats.getOutput(3))
                },
                "cell_size": cell_size_actual,
                "unit": "cubic units"
            }
            
            if output_raster:
                result["output_raster"] = output_raster
            
            logger.info("Cut/fill analysis completed successfully")
            return result
        
        except Exception as e:
            logger.error(f"Error computing cut/fill: {e}")
            return {"success": False, "error": str(e)}
    
    def _ensure_raster(self, input_data: str):
        """Ensure input is a raster, converting if necessary."""
        desc = self.arcpy.Describe(input_data)
        
        if desc.dataType == "RasterDataset":
            return self.arcpy.Raster(input_data)
        elif desc.dataType in ["FeatureClass", "ShapeFile"]:
            logger.warning("Feature class to raster conversion requires elevation field specification")
            raise ValueError("Feature class input requires additional parameters for raster conversion")
        else:
            raise ValueError(f"Unsupported input type: {desc.dataType}")
