"""
Blue Marble Geographic Calculator Interface

Provides COM and CLI interfaces for coordinate conversions and geodetic operations.
"""

from __future__ import annotations

import os
import subprocess
import shutil
import sys
import platform
import re
import difflib
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from utils.coordinate_parsing import parse_angle

try:
    import pythoncom
    import win32com.client
    from win32com.client import DispatchEx, Dispatch
    COM_AVAILABLE = True
except ImportError:
    COM_AVAILABLE = False

from config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)

# ProgIDs for COM interface
DEFAULT_PROGIDS: Tuple[str, ...] = (
    "GeographicCalculator.Application", "GeographicCalculator.Application.1",
    "GeographicCalculator.Application.2", "GeographicCalculator.Application.3",
    "BlueMarble.GeoCalc.Application", "BlueMarble.GeoCalc.Application.1",
    "BlueMarble.GeographicCalculator", "BlueMarble.GeographicCalculator.1",
    "BlueMarble.GeoCalc", "BlueMarble.GeoCalc.1",
    "GeoCalc.Application", "GeoCalc.Application.1", "GeoCalc", "GeoCalc.1",
    "BMG.GeoCalc.Application", "BMG.GeoCalc.Application.1",
    "BMG.GeographicCalculator", "BMG.GeographicCalculator.1",
    "GeoCalcPro.Application", "GeoCalcPro.Application.1",
)

# Common installation paths
GEOCALC_COMMON_PATHS: Tuple[str, ...] = (
    r"C:\Program Files\Blue Marble Geo\Geographic Calculator",
    r"C:\Program Files (x86)\Blue Marble Geo\Geographic Calculator",
    r"C:\Program Files\Blue Marble Geographics\Geographic Calculator",
    r"C:\Program Files (x86)\Blue Marble Geographics\Geographic Calculator",
    r"C:\Program Files\Blue Marble\Geographic Calculator",
    r"C:\Program Files (x86)\Blue Marble\Geographic Calculator",
    r"C:\Program Files\Geographic Calculator",
    r"C:\Program Files (x86)\Geographic Calculator",
)

GEOCALC_CMD_COMMON_PATHS: Tuple[str, ...] = (
    r"C:\Program Files\Blue Marble Geographics\Geographic Calculator 2025\GeographicCalculatorCMD.exe",
    r"C:\Program Files\Blue Marble Geographics\Geographic Calculator 2024\GeographicCalculatorCMD.exe",
    r"C:\Program Files\Blue Marble Geographics\Geographic Calculator 2023\GeographicCalculatorCMD.exe",
    r"C:\Program Files (x86)\Blue Marble Geographics\Geographic Calculator 2025\GeographicCalculatorCMD.exe",
    r"C:\Program Files (x86)\Blue Marble Geographics\Geographic Calculator 2024\GeographicCalculatorCMD.exe",
    r"C:\Program Files (x86)\Blue Marble Geographics\Geographic Calculator 2023\GeographicCalculatorCMD.exe",
    r"C:\Program Files\Blue Marble Geo\Geographic Calculator\GeographicCalculatorCMD.exe",
    r"C:\Program Files (x86)\Blue Marble Geo\Geographic Calculator\GeographicCalculatorCMD.exe",
    r"C:\Program Files\Geographic Calculator\GeographicCalculatorCMD.exe",
    r"C:\Program Files (x86)\Geographic Calculator\GeographicCalculatorCMD.exe",
)

REGISTRY_PATHS = [
    r"SOFTWARE\Blue Marble Geographics\Geographic Calculator",
    r"SOFTWARE\WOW6432Node\Blue Marble Geographics\Geographic Calculator",
    r"SOFTWARE\Blue Marble\Geographic Calculator",
    r"SOFTWARE\Blue Marble Geographics\GeoCalc Pro",
    r"SOFTWARE\WOW6432Node\Blue Marble Geographics\GeoCalc Pro",
]

SEARCH_PATTERNS = ["Blue Marble Geographics", "Blue Marble Geo", "Geographic Calculator"]


class GeographicCalculatorScanner:
    """Shared scanning logic for finding Geographic Calculator installations."""
    
    @staticmethod
    def _get_program_files_dirs() -> List[Path]:
        """Get Program Files directories."""
        return [Path(p) for env_var in ["ProgramFiles", "ProgramFiles(x86)"] 
                if os.path.exists(p := os.environ.get(env_var, rf"C:\{env_var.replace('(x86)', ' (x86)')}"))]
    
    @staticmethod
    def _query_registry(hkey, subkey: str, value_names: List[str]) -> Optional[str]:
        """Query registry for a value."""
        try:
            import winreg
            with winreg.OpenKey(hkey, subkey) as key:
                for value_name in value_names:
                    try:
                        value, _ = winreg.QueryValueEx(key, value_name if value_name else None)
                        if value:
                            return str(value)
                    except (FileNotFoundError, OSError):
                        continue
        except (FileNotFoundError, OSError, ImportError):
            pass
        return None
    
    @staticmethod
    def _scan_directory(directory: Path, target: str, max_depth: int = 3) -> Optional[Path]:
        """Recursively scan directory for target file."""
        if max_depth <= 0:
            return None
        try:
            if (file_path := directory / target).exists() and file_path.is_file():
                return file_path
            for item in directory.iterdir():
                if item.is_dir() and (found := GeographicCalculatorScanner._scan_directory(item, target, max_depth - 1)):
                    return found
        except (PermissionError, OSError):
            pass
        return None

    @staticmethod
    def is_com_progid_registered(progid: str) -> bool:
        """Return True if a COM ProgID is registered on this machine/user."""
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, progid + r"\CLSID"):
                return True
        except Exception:
            return False

    @staticmethod
    def get_gui_executable() -> Optional[Path]:
        """Return full path to Geographic Calculator GUI executable if present."""
        install_dir = GeographicCalculatorScanner.find_installation_path()
        if not install_dir:
            return None
        gui_candidates = (
            install_dir / "Geographic Calculator.exe",
            install_dir / "GeographicCalculator.exe",
        )
        for p in gui_candidates:
            if p.exists() and p.is_file():
                return p
        return None

    @staticmethod
    def attempt_register_com(gui_exe: Optional[Path] = None, timeout: int = 20) -> bool:
        """
        Best-effort attempt to register Geographic Calculator COM automation.

        Notes:
        - Many installs do NOT ship COM automation; in that case this won't help.
        - Registration may require Administrator privileges (HKLM writes).
        - We intentionally keep this best-effort and non-fatal.
        """
        gui_exe = gui_exe or GeographicCalculatorScanner.get_gui_executable()
        if not gui_exe or not gui_exe.exists():
            return False

        switches = ("/regserver", "-regserver", "/RegServer", "-RegServer", "/register", "-register")
        for sw in switches:
            try:
                # Some apps register COM and exit immediately; others may briefly show UI.
                subprocess.run([str(gui_exe), sw], capture_output=True, text=True, timeout=timeout)
                return True
            except Exception:
                continue
        return False
    
    @staticmethod
    def _find_in_registry(target: str = "dir") -> Optional[Path]:
        """Check Windows registry for installation path or CLI executable."""
        try:
            import winreg
            value_names = ["InstallPath", "InstallDir", "Path", ""]
            
            for hkey in [winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER]:
                for subkey in REGISTRY_PATHS:
                    if not (install_dir := GeographicCalculatorScanner._query_registry(hkey, subkey, value_names)):
                        continue
                    
                    install_path = Path(install_dir)
                    if not install_path.exists():
                        continue
                    
                    if target == "cli":
                        if (cmd_path := install_path / "GeographicCalculatorCMD.exe").exists():
                            return cmd_path
                        # Check subdirectories (max depth 2)
                        for subdir in install_path.iterdir():
                            if subdir.is_dir():
                                if (cmd_path := subdir / "GeographicCalculatorCMD.exe").exists():
                                    return cmd_path
                                for subsubdir in subdir.iterdir():
                                    if subsubdir.is_dir() and (cmd_path := subsubdir / "GeographicCalculatorCMD.exe").exists():
                                        return cmd_path
                    else:
                        return install_path
        except ImportError:
            pass
        return None
    
    @staticmethod
    def _find_in_app_paths() -> Optional[Path]:
        """Check Windows App Paths registry."""
        try:
            import winreg
            candidates = [
                r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\GeographicCalculatorCMD.exe",
                r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\GeographicCalculatorCmd.exe",
                r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\GeographicCalculatorCMD",
            ]
            for root in [winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER]:
                for subkey in candidates:
                    if exe_path := GeographicCalculatorScanner._query_registry(root, subkey, [""]):
                        if (p := Path(exe_path.strip().strip('"'))).exists() and p.is_file():
                            return p
        except ImportError:
            pass
        return None
    
    @staticmethod
    def _find_in_uninstall_registry() -> Optional[Path]:
        """Search uninstall registry for installation location."""
        try:
            import winreg
            uninstall_roots = [
                (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
                (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
                (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
            ]
            keywords = ("geographic calculator", "geocalc", "blue marble", "geocalc pro")
            value_candidates = ("InstallLocation", "InstallSource", "DisplayIcon", "UninstallString")
            
            for root, base in uninstall_roots:
                try:
                    with winreg.OpenKey(root, base) as uninstall_key:
                        for i in range(winreg.QueryInfoKey(uninstall_key)[0]):
                            try:
                                with winreg.OpenKey(uninstall_key, winreg.EnumKey(uninstall_key, i)) as app_key:
                                    try:
                                        display_name, _ = winreg.QueryValueEx(app_key, "DisplayName")
                                        if not any(k in str(display_name or "").lower() for k in keywords):
                                            continue
                                    except Exception:
                                        continue
                                    
                                    for value_name in value_candidates:
                                        try:
                                            raw, _ = winreg.QueryValueEx(app_key, value_name)
                                            if not raw:
                                                continue
                                            raw_s = str(raw).strip().split(",")[0].strip().strip('"')
                                            p = Path(raw_s)
                                            
                                            if p.exists():
                                                if p.is_file():
                                                    if (candidate := p.parent / "GeographicCalculatorCMD.exe").exists():
                                                        return candidate
                                                    if found := GeographicCalculatorScanner._scan_directory(p.parent, "GeographicCalculatorCMD.exe", 4):
                                                        return found
                                                elif p.is_dir():
                                                    if found := GeographicCalculatorScanner._scan_directory(p, "GeographicCalculatorCMD.exe", 6):
                                                        return found
                                        except Exception:
                                            continue
                            except (FileNotFoundError, OSError):
                                continue
                except (FileNotFoundError, OSError):
                    continue
        except ImportError:
            pass
        return None
    
    @staticmethod
    def _find_cli_via_gui() -> Optional[Path]:
        """Find CLI by locating GUI executable."""
        user_path = Path(r"C:\Program Files\Blue Marble Geo\Geographic Calculator")
        if user_path.exists() and (gui_path := user_path / "Geographic Calculator.exe").exists():
            if (cli_path := user_path / "GeographicCalculatorCMD.exe").exists():
                return cli_path
            if found := GeographicCalculatorScanner._scan_directory(user_path, "GeographicCalculatorCMD.exe", 5):
                return found
        
        gui_names = ["Geographic Calculator.exe", "GeographicCalculator.exe", "GeoCalc.exe", "GeoCalcPro.exe"]
        for pf_dir in GeographicCalculatorScanner._get_program_files_dirs():
            for pattern in SEARCH_PATTERNS:
                potential_dir = pf_dir / pattern
                if potential_dir.exists():
                    for gui_name in gui_names:
                        if (gui_path := potential_dir / gui_name).exists():
                            if (cli_path := gui_path.parent / "GeographicCalculatorCMD.exe").exists():
                                return cli_path
                            if found := GeographicCalculatorScanner._scan_directory(gui_path.parent, "GeographicCalculatorCMD.exe", 3):
                                return found
        return None
    
    @staticmethod
    def _find_with_where() -> Optional[Path]:
        """Locate CLI using shutil.which or where.exe."""
        if p := shutil.which("GeographicCalculatorCMD.exe"):
            if (path := Path(p)).exists():
                return path
        
        try:
            result = subprocess.run(
                ["where", "GeographicCalculatorCMD.exe"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for line in (result.stdout or "").splitlines():
                    if (candidate := Path(line.strip())).exists() and candidate.is_file():
                        return candidate
        except Exception:
            pass
        return None
    
    @staticmethod
    def _comprehensive_scan(target: str = "dir") -> Optional[Path]:
        """Comprehensive scan of Program Files."""
        for pf_dir in GeographicCalculatorScanner._get_program_files_dirs():
            if not pf_dir.exists():
                continue
            for pattern in SEARCH_PATTERNS:
                try:
                    if (potential_dir := pf_dir / pattern).exists():
                        if target == "cli":
                            if found := GeographicCalculatorScanner._scan_directory(potential_dir, "GeographicCalculatorCMD.exe", 6):
                                return found
                        else:
                            return potential_dir
                    for item in pf_dir.iterdir():
                        if item.is_dir() and pattern.lower() in item.name.lower():
                            if target == "cli":
                                if found := GeographicCalculatorScanner._scan_directory(item, "GeographicCalculatorCMD.exe", 6):
                                    return found
                            else:
                                return item
                except (PermissionError, OSError):
                    continue
        return None
    
    @staticmethod
    def _save_path_to_env(path: Path, is_cli: bool = False) -> bool:
        """Save found path to .env file."""
        try:
            path_str = str(path.resolve())
            lowered = path_str.lower()
            if not any(m in lowered for m in (
                r"\program files\\", r"\program files (x86)\\", r"\blue marble",
                r"\geographic calculator", r"\geocalc"
            )):
                logger.debug("Skipping .env persistence (non-install path): %s", path_str)
                return True
            
            project_root = Path(__file__).parent.parent
            env_file = project_root / ".env"
            env_content = env_file.read_text(encoding='utf-8') if env_file.exists() else ""
            
            env_var = "GEOGRAPHIC_CALCULATOR_CMD_PATH" if is_cli else "BLUE_MARBLE_PATH"
            comment = f"# {'Geographic Calculator CLI' if is_cli else 'Blue Marble'} Path (auto-detected)"
            
            lines = env_content.split('\n')
            new_lines = []
            found_existing = False
            
            for line in lines:
                if line.strip().startswith(env_var + '='):
                    new_lines.append(env_var + '=' + path_str)
                    found_existing = True
                else:
                    new_lines.append(line)
            
            if not found_existing:
                if env_content and not env_content.endswith('\n'):
                    new_lines.append('')
                new_lines.extend([comment, env_var + '=' + path_str])
            
            env_file.write_text('\n'.join(new_lines), encoding='utf-8')
            logger.info(f"✓ {'Updated' if found_existing else 'Saved'} {env_var} in .env file: {path_str}")
            return True
        except Exception as e:
            logger.warning(f"Could not save path to .env file: {e}")
            return False
    
    @staticmethod
    def find_installation_path() -> Optional[Path]:
        """Find Geographic Calculator installation directory."""
        settings = get_settings()
        if settings.blue_marble_path and (path := Path(settings.blue_marble_path)).exists():
            return path
        
        if found := GeographicCalculatorScanner._find_in_registry("dir"):
            return found
        
        for path_str in GEOCALC_COMMON_PATHS:
            if (path := Path(path_str)).exists():
                return path
        
        return GeographicCalculatorScanner._comprehensive_scan("dir")
    
    @staticmethod
    def find_cli_executable() -> Optional[Path]:
        """Find GeographicCalculatorCMD.exe using comprehensive scanning."""
        # Check env var and settings first
        if env_path := os.environ.get("GEOGRAPHIC_CALCULATOR_CMD_PATH", "").strip():
            if (path := Path(env_path)).exists() and path.is_file():
                return path
        
        settings = get_settings()
        if user_path := getattr(settings, 'geographic_calculator_cmd_path', ''):
            if (path := Path(user_path.strip())).exists() and path.is_file():
                return path
        
        # Try various detection methods
        search_methods = [
            ("App Paths", GeographicCalculatorScanner._find_in_app_paths),
            ("Registry", lambda: GeographicCalculatorScanner._find_in_registry("cli")),
            ("Uninstall Registry", GeographicCalculatorScanner._find_in_uninstall_registry),
            ("Common Paths", lambda: next(
                (Path(p) for p in GEOCALC_CMD_COMMON_PATHS if Path(p).exists() and Path(p).is_file()),
                None
            )),
            ("GUI Executable", GeographicCalculatorScanner._find_cli_via_gui),
            ("PATH", lambda: next(
                (Path(p) / "GeographicCalculatorCMD.exe" for p in os.environ.get("PATH", "").split(os.pathsep)
                 if (Path(p) / "GeographicCalculatorCMD.exe").exists()),
                None
            )),
            ("where.exe", GeographicCalculatorScanner._find_with_where),
            ("Install Dir", lambda: GeographicCalculatorScanner._scan_directory(
                GeographicCalculatorScanner.find_installation_path() or Path(""),
                "GeographicCalculatorCMD.exe", 6
            ) if GeographicCalculatorScanner.find_installation_path() else None),
            ("Comprehensive Scan", lambda: GeographicCalculatorScanner._comprehensive_scan("cli")),
        ]
        
        for method_name, method_func in search_methods:
            try:
                if (found := method_func()) and found.exists() and found.is_file():
                    GeographicCalculatorScanner._save_path_to_env(found, is_cli=True)
                    return found
            except Exception:
                continue
        
        return None


class BlueMarbleConverter:
    """Interface with Blue Marble Geographic Calculator using COM with pyproj fallback."""
    
    def __init__(self, progids: Optional[Iterable[str]] = None, auto_connect: bool = False) -> None:
        self.settings = get_settings()
        self._progids = self._prepare_progids(progids)
        self.geocalc = None
        self._geocalc_path: Optional[Path] = None
        self._connected_progid: Optional[str] = None
        # IMPORTANT: Do not auto-connect by default.
        # COM scanning/registration attempts are noisy and slow, and most user queries
        # (documents, Excel parsing, etc.) do not require Geographic Calculator.
        # We only connect when explicitly requested (use_geographic_calculator=True).
        if auto_connect:
            self.geocalc = self._connect_via_com()
    
    @staticmethod
    def _prepare_progids(extra: Optional[Iterable[str]]) -> Tuple[str, ...]:
        seen = set()
        candidates = (
            os.getenv("BLUE_MARBLE_PROGID"), os.getenv("GEOCALC_PROGID"),
            *(extra or ()), *DEFAULT_PROGIDS
        )
        return tuple(p for p in (c.strip() if c else None for c in candidates)
                    if p and p not in seen and not seen.add(p))
    
    @property
    def is_available(self) -> bool:
        return self.geocalc is not None
    
    def refresh_connection(self) -> None:
        self.geocalc = self._connect_via_com()
    
    def _connect_via_com(self) -> Optional[Any]:
        """Connect to Geographic Calculator via COM interface."""
        if not COM_AVAILABLE:
            logger.debug("COM libraries (win32com) not available")
            return None
        
        # Initialize COM
        try:
            pythoncom.CoInitialize()
        except Exception as e:
            logger.debug(f"COM initialization note: {e}")
            # May already be initialized, continue
        
        # Find installation path for diagnostics
        self._geocalc_path = GeographicCalculatorScanner.find_installation_path()
        if self._geocalc_path:
            GeographicCalculatorScanner._save_path_to_env(self._geocalc_path, is_cli=False)
            logger.debug(f"Geographic Calculator installation found: {self._geocalc_path}")
        else:
            logger.debug("Geographic Calculator installation path not found")
        
        # Before attempting Dispatch, verify that at least one ProgID is registered.
        registered_progids = [p for p in self._progids if GeographicCalculatorScanner.is_com_progid_registered(p)]
        if not registered_progids:
            gui_exe = GeographicCalculatorScanner.get_gui_executable()
            logger.warning(
                "Geographic Calculator is installed, but no COM ProgIDs are registered. "
                "This usually means the COM/Automation component is not installed or not registered."
            )

            # Best-effort: try to self-register COM via the GUI executable.
            if GeographicCalculatorScanner.attempt_register_com(gui_exe):
                registered_progids = [p for p in self._progids if GeographicCalculatorScanner.is_com_progid_registered(p)]

        if not registered_progids:
            # Give the most actionable explanation we can.
            gui_exe = GeographicCalculatorScanner.get_gui_executable()
            python_bits = "64-bit" if sys.maxsize > 2**32 else "32-bit"
            logger.warning(
                "COM automation is not available for Geographic Calculator on this machine.\n"
                f"- Python: {python_bits} ({platform.python_version()})\n"
                f"- GUI: {'found at ' + str(gui_exe) if gui_exe else 'not found'}\n"
                "- COM ProgIDs: not registered\n"
                "Fix options:\n"
                "  1) Re-run the Geographic Calculator installer and enable the Automation/COM/SDK component (if available).\n"
                "  2) Run the installer/registration as Administrator (COM registration often writes to HKLM).\n"
                "  3) If your license does not include automation, contact Blue Marble support for the automation installer/license.\n"
                "Falling back to pyproj for coordinate conversions."
            )
            return None

        # Try to connect using various ProgIDs
        logger.debug(f"Attempting COM connection with {len(self._progids)} ProgID(s)")
        for progid in registered_progids:
            try:
                if instance := self._try_connect(progid):
                    self._connected_progid = progid
                    logger.info(f"✓ Connected to Geographic Calculator via COM (ProgID: {progid})")
                    return instance
            except Exception as e:
                logger.debug(f"Failed to connect with ProgID {progid}: {e}")
                continue
        
        # Connection failed - provide helpful diagnostics
        if self._geocalc_path:
            logger.warning(
                f"Geographic Calculator GUI found at: {self._geocalc_path}\n"
                f"But COM automation failed. This may be due to:\n"
                f"  1. Geographic Calculator not running (COM may require GUI to be running)\n"
                f"  2. COM registration issues (try running Geographic Calculator GUI once)\n"
                f"  3. License restrictions (some versions require specific licenses for automation)\n"
                f"Falling back to pyproj for coordinate conversions."
            )
        else:
            logger.info("Geographic Calculator not found; using pyproj for coordinate conversions")
        
        return None
    
    def _try_connect(self, progid: str) -> Optional[Any]:
        """Try to connect to Geographic Calculator using various COM methods."""
        if not COM_AVAILABLE:
            return None
        
        # Method 1: GetActiveObject - connects to running instance
        try:
            instance = win32com.client.GetActiveObject(progid)
            logger.debug(f"Connected via GetActiveObject with {progid}")
            return instance
        except Exception as e:
            logger.debug(f"GetActiveObject failed for {progid}: {type(e).__name__}")
        
        # Method 2: Dispatch - creates new instance
        try:
            instance = Dispatch(progid)
            # Verify it's actually working by checking if it has CreatePoint method
            if hasattr(instance, 'CreatePoint'):
                logger.debug(f"Connected via Dispatch with {progid}")
                return instance
        except Exception as e:
            logger.debug(f"Dispatch failed for {progid}: {type(e).__name__}")
        
        # Method 3: DispatchEx - creates new instance with specific flags
        try:
            instance = DispatchEx(progid)
            if hasattr(instance, 'CreatePoint'):
                logger.debug(f"Connected via DispatchEx with {progid}")
                return instance
        except Exception as e:
            logger.debug(f"DispatchEx failed for {progid}: {type(e).__name__}")
        
        return None
    
    def convert_coordinate(
        self, x: float, y: float, z: Optional[float] = None,
        source_crs: str = "WGS84", target_crs: str = "WGS84",
        source_zone: Optional[int] = None, target_zone: Optional[int] = None,
        use_geographic_calculator: bool = False,
    ) -> Dict[str, Any]:
        """
        Convert a single coordinate using pyproj (default) or Geographic Calculator COM (if requested).
        
        Args:
            x, y, z: Coordinate values
            source_crs, target_crs: Coordinate reference system names
            source_zone, target_zone: UTM zones if applicable
            use_geographic_calculator: If True, attempt to use Geographic Calculator COM interface.
                                      If False (default), use pyproj. Always falls back to pyproj if COM fails.
        
        Returns:
            Dictionary with conversion results including method used and resolved CRS information.
        """
        # Default: Use pyproj (fast, reliable, no external dependencies)
        if not use_geographic_calculator:
            return self._convert_with_pyproj(x, y, z, source_crs, target_crs, source_zone, target_zone)
        
        # User explicitly requested Geographic Calculator - try COM if available
        # Lazy-connect so COM isn't triggered during unrelated workflows.
        if use_geographic_calculator and (self.geocalc is None):
            self.geocalc = self._connect_via_com()

        if self.geocalc:
            try:
                result = self._convert_with_com(x, y, z, source_crs, target_crs, source_zone, target_zone)
                logger.debug("Used Geographic Calculator COM as requested")
                return result
            except Exception as exc:
                logger.warning("Geographic Calculator COM conversion failed; falling back to pyproj: %s", exc)
                # Fall through to pyproj fallback
        else:
            logger.debug("Geographic Calculator COM not available; using pyproj as requested")
        
        # Always fallback to pyproj (even if user requested COM but it's unavailable/failed)
        return self._convert_with_pyproj(x, y, z, source_crs, target_crs, source_zone, target_zone)
    
    def _normalize_crs_name(self, crs: str, zone: Optional[int] = None) -> str:
        """Normalize CRS name for Geographic Calculator COM interface."""
        crs_clean = crs.strip()
        
        # Geographic Calculator can handle many formats directly, but we normalize common patterns
        crs_upper = crs_clean.upper()
        
        # Handle UTM zones
        if zone is not None:
            if "UTM" in crs_upper:
                # Let COM handle UTM with zone
                return crs_clean
        
        # Handle EPSG codes
        if crs_clean.startswith("EPSG:") or crs_clean.startswith("epsg:"):
            return crs_clean
        
        # Handle numeric EPSG
        if crs_clean.isdigit():
            return f"EPSG:{crs_clean}"
        
        # Common mappings for better compatibility
        crs_mappings = {
            "WGS84": "WGS84",
            "WGS 84": "WGS84",
            "NAD83": "NAD83",
            "NAD 83": "NAD83",
            "NAD27": "NAD27",
            "NAD 27": "NAD27",
        }
        
        if crs_upper in crs_mappings:
            return crs_mappings[crs_upper]
        
        # For complex names like "Minna Nigerian NTM MidBelt", let Geographic Calculator resolve it
        # GC has extensive CRS database and can handle many named coordinate systems
        return crs_clean
    
    def _convert_with_com(
        self, x: float, y: float, z: Optional[float],
        source_crs: str, target_crs: str, source_zone: Optional[int], target_zone: Optional[int],
    ) -> Dict[str, Any]:
        """Convert coordinates using Geographic Calculator COM interface."""
        if not COM_AVAILABLE or not self.geocalc:
            raise RuntimeError("COM interface not available")
        
        try:
            # Normalize CRS names - Geographic Calculator has extensive CRS database
            source_crs_normalized = self._normalize_crs_name(source_crs, source_zone)
            target_crs_normalized = self._normalize_crs_name(target_crs, target_zone)
            
            # Create point
            point = self.geocalc.CreatePoint()
            point.X, point.Y = x, y
            if z is not None:
                point.Z = z
            
            # Set source coordinate system
            # Geographic Calculator can resolve CRS names from its database
            try:
                source_system = self.geocalc.CreateCoordinateSystem(source_crs_normalized)
                if source_zone is not None:
                    source_system.Zone = source_zone
                point.CoordinateSystem = source_system
            except Exception as e:
                # Try with original name if normalized fails
                logger.debug(f"Failed to create source CRS '{source_crs_normalized}', trying original: {e}")
                source_system = self.geocalc.CreateCoordinateSystem(source_crs)
                if source_zone is not None:
                    source_system.Zone = source_zone
                point.CoordinateSystem = source_system
            
            # Set target coordinate system
            try:
                target_system = self.geocalc.CreateCoordinateSystem(target_crs_normalized)
                if target_zone is not None:
                    target_system.Zone = target_zone
            except Exception as e:
                # Try with original name if normalized fails
                logger.debug(f"Failed to create target CRS '{target_crs_normalized}', trying original: {e}")
                target_system = self.geocalc.CreateCoordinateSystem(target_crs)
                if target_zone is not None:
                    target_system.Zone = target_zone
            
            # Perform conversion
            converted = point.ConvertTo(target_system)
            
            result = {
                "source": {"x": x, "y": y, "crs": source_crs},
                "target": {"x": float(converted.X), "y": float(converted.Y), "crs": target_crs},
                "method": "COM",
            }
            if z is not None:
                result["source"]["z"] = z
                result["target"]["z"] = float(getattr(converted, "Z", z))
            
            logger.debug(f"COM conversion successful: {source_crs} -> {target_crs}")
            return result
            
        except Exception as e:
            error_msg = f"COM conversion failed: {str(e)}"
            logger.error(error_msg)
            # Re-raise with more context
            raise RuntimeError(f"{error_msg}. Source CRS: {source_crs}, Target CRS: {target_crs}") from e
    
    def _convert_with_pyproj(
        self, x: float, y: float, z: Optional[float],
        source_crs: str, target_crs: str, source_zone: Optional[int], target_zone: Optional[int],
    ) -> Dict[str, Any]:
        from pyproj import CRS, Transformer

        source = self._resolve_crs_pyproj(source_crs, source_zone, role="source")
        target = self._resolve_crs_pyproj(target_crs, target_zone, role="target")
        transformer = Transformer.from_crs(source, target, always_xy=True)
        
        if z is not None:
            x_new, y_new, z_new = transformer.transform(x, y, z)
        else:
            x_new, y_new = transformer.transform(x, y)
            z_new = z
        
        result = {
            "source": {"x": x, "y": y, "crs": source_crs},
            "target": {"x": float(x_new), "y": float(y_new), "crs": target_crs},
            "method": "pyproj",
            "resolved": {
                "source": self._crs_summary(source),
                "target": self._crs_summary(target),
            },
        }
        if z is not None:
            result["source"]["z"] = z
            result["target"]["z"] = float(z_new)
        logger.debug("Converted using pyproj: %s -> %s", source_crs, target_crs)
        return result
    
    @staticmethod
    def _crs_summary(crs_obj: Any) -> Dict[str, Any]:
        """Small, serializable CRS summary (authority, name)."""
        try:
            auth = crs_obj.to_authority()
        except Exception:
            auth = None
        try:
            name = getattr(crs_obj, "name", None) or str(crs_obj)
        except Exception:
            name = str(crs_obj)
        return {"authority": auth, "name": name}

    @staticmethod
    def _normalize_user_crs_text(text: str) -> str:
        """Normalize user-provided CRS strings to improve pyproj parsing."""
        t = (text or "").strip()
        if not t:
            return t
        # Common user formats: MINNA_NTM_MIDBELT, "UTM zone 32", etc.
        t = t.replace("_", " ")
        t = re.sub(r"\s+", " ", t)
        # Make some common concatenations searchable
        t = re.sub(r"\bmidbelt\b", "mid belt", t, flags=re.IGNORECASE)
        t = re.sub(r"\bmid-belt\b", "mid belt", t, flags=re.IGNORECASE)
        return t.strip()

    @staticmethod
    def _parse_utm_zone(text: str) -> Tuple[Optional[int], Optional[str]]:
        """
        Extract UTM zone and hemisphere from a free-form string.
        Returns (zone, hemisphere) where hemisphere is 'N'/'S'/None.
        """
        t = (text or "").upper().replace("_", " ")
        # Patterns like: "UTM zone 32", "UTM 32N", "UTM32 N", "UTM32N"
        m = re.search(r"\bUTM\b\s*(?:ZONE\s*)?(\d{1,2})\s*([NS])?\b", t)
        if not m:
            m = re.search(r"\bUTM\s*(\d{1,2})\s*([NS])\b", t)
        if not m:
            return None, None
        zone = int(m.group(1))
        hemi = (m.group(2) or "").strip() or None
        if zone < 1 or zone > 60:
            return None, None
        return zone, hemi

    @staticmethod
    @lru_cache(maxsize=512)
    def _fuzzy_find_epsg_crs(normalized_query: str) -> Optional[str]:
        """
        Fuzzy match an EPSG CRS name and return an 'EPSG:####' string.
        Keeps this intentionally conservative: only returns a best guess when score is strong.
        """
        try:
            from pyproj.database import query_crs_info
            from pyproj.enums import PJType
        except Exception:
            return None

        q = (normalized_query or "").strip()
        if not q:
            return None

        q_lower = q.lower()
        # Quick aliases for common user tokens seen in this project
        alias_map = {
            "minna ntm midbelt": "Minna / Nigeria Mid Belt",
            "minna ntm mid belt": "Minna / Nigeria Mid Belt",
            "minna nigeria ntm midbelt": "Minna / Nigeria Mid Belt",
            "minna nigeria ntm mid belt": "Minna / Nigeria Mid Belt",
        }
        if q_lower in alias_map:
            try:
                from pyproj import CRS
                return CRS.from_user_input(alias_map[q_lower]).to_string()
            except Exception:
                pass

        # Candidate filtering: require at least one meaningful keyword to reduce scan.
        keywords = [k for k in re.split(r"[^a-z0-9]+", q_lower) if len(k) >= 4]
        keywords = keywords[:6]
        if not keywords:
            return None

        # Query projected CRSs first (what the user usually means for grids like NTM/UTM).
        candidates = query_crs_info(
            auth_name="EPSG",
            pj_types=[PJType.PROJECTED_CRS, PJType.GEOGRAPHIC_2D_CRS, PJType.GEOGRAPHIC_3D_CRS],
        )

        filtered = []
        for info in candidates:
            name = (info.name or "")
            nlow = name.lower()
            if any(k in nlow for k in keywords):
                filtered.append(info)
            if len(filtered) >= 800:
                break

        if not filtered:
            return None

        def score(info) -> float:
            name = info.name or ""
            n = name.lower()
            # token overlap
            overlap = sum(1 for k in keywords if k in n)
            # sequence similarity (after stripping punctuation)
            a = re.sub(r"[^a-z0-9 ]+", " ", q_lower)
            b = re.sub(r"[^a-z0-9 ]+", " ", n)
            ratio = difflib.SequenceMatcher(None, a, b).ratio()
            return overlap * 2.0 + ratio

        best = max(filtered, key=score)
        best_score = score(best)

        # Conservative threshold to avoid surprising mismatches.
        if best_score < 3.4:
            return None

        return f"EPSG:{best.code}"

    def _resolve_crs_pyproj(self, crs: str, zone: Optional[int], role: str = "crs"):
        """
        Resolve a user-provided CRS into a pyproj.CRS object.

        Strategy:
        - Handle EPSG:#### and numeric EPSG directly
        - Handle common names (WGS84) and UTM zone patterns
        - Try pyproj parsing as-is
        - Normalize text and retry
        - Fuzzy match to closest EPSG CRS name and retry
        """
        from pyproj import CRS

        raw = (crs or "").strip()
        if not raw:
            # Default to WGS84 if user passes empty
            return CRS.from_epsg(4326)

        # EPSG direct
        if re.match(r"^epsg:\d+$", raw, flags=re.IGNORECASE):
            return CRS.from_user_input(raw)
        if raw.isdigit():
            return CRS.from_epsg(int(raw))

        # Quick common names
        common = {
            "WGS84": "EPSG:4326",
            "WGS 84": "EPSG:4326",
            "NAD83": "EPSG:4269",
            "NAD 83": "EPSG:4269",
            "NAD27": "EPSG:4267",
            "NAD 27": "EPSG:4267",
        }
        if raw.upper() in common:
            return CRS.from_user_input(common[raw.upper()])

        # UTM detection (from explicit zone arg or embedded in string)
        utm_zone, utm_hemi = self._parse_utm_zone(raw)
        z = zone or utm_zone
        if z:
            # Determine hemisphere: explicit in string wins; else assume north unless zone passed negative
            hemi = utm_hemi
            if hemi is None and zone is not None and zone < 0:
                hemi = "S"
            if hemi is None:
                hemi = "N"
            epsg = 32600 + z if hemi == "N" else 32700 + z
            # Only use this shortcut if the user *actually* meant UTM
            if ("UTM" in raw.upper()) or (zone is not None):
                return CRS.from_epsg(epsg)

        # Attempt direct parse
        try:
            return CRS.from_user_input(raw)
        except Exception:
            pass

        # Normalize and retry
        normalized = self._normalize_user_crs_text(raw)
        if normalized and normalized != raw:
            try:
                return CRS.from_user_input(normalized)
            except Exception:
                pass

        # Fuzzy: pick best EPSG match from database
        epsg_guess = self._fuzzy_find_epsg_crs(normalized or raw)
        if epsg_guess:
            try:
                guessed = CRS.from_user_input(epsg_guess)
                logger.info("Resolved %s CRS '%s' -> %s (%s)", role, raw, epsg_guess, getattr(guessed, "name", epsg_guess))
                return guessed
            except Exception:
                pass

        # Give a clearer error than "unknown name"
        raise ValueError(
            f"Unable to resolve {role} CRS '{raw}'. "
            f"Try an EPSG code (e.g., 'EPSG:4326') or a standard name like 'Minna / Nigeria Mid Belt'."
        )
    
    def batch_convert(
        self, coordinates: List[Tuple[float, float]],
        source_crs: str = "WGS84", target_crs: str = "WGS84",
        source_zone: Optional[int] = None, target_zone: Optional[int] = None,
        use_geographic_calculator: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Batch convert coordinates using pyproj (default) or Geographic Calculator COM (if requested).
        
        Args:
            coordinates: List of (x, y) tuples to convert
            source_crs, target_crs: Coordinate reference system names
            source_zone, target_zone: UTM zones if applicable
            use_geographic_calculator: If True, attempt to use Geographic Calculator COM interface.
                                      If False (default), use pyproj. Always falls back to pyproj if COM fails.
        
        Returns:
            List of conversion result dictionaries.
        """
        results = []
        total = len(coordinates)
        method = "Geographic Calculator COM" if use_geographic_calculator else "pyproj"
        logger.info(f"Starting batch conversion of {total} coordinates using {method}: {source_crs} -> {target_crs}")
        
        for idx, (x, y) in enumerate(coordinates, 1):
            try:
                result = self.convert_coordinate(
                    x, y, None, source_crs, target_crs, source_zone, target_zone,
                    use_geographic_calculator=use_geographic_calculator
                )
                results.append(result)
                if idx % 100 == 0:
                    logger.debug(f"Converted {idx}/{total} coordinates")
            except Exception as exc:
                logger.warning("Failed to convert (%s, %s): %s", x, y, exc)
                results.append({"source": {"x": x, "y": y, "crs": source_crs}, "error": str(exc)})
        
        successful = sum(1 for r in results if "error" not in r)
        logger.info(f"Batch conversion complete: {successful}/{total} successful")
        return results
    
    def convert_excel_file(
        self,
        excel_path: str,
        x_column: str = "X",
        y_column: str = "Y",
        source_crs: str = "WGS84",
        target_crs: str = "WGS84",
        source_zone: Optional[int] = None,
        target_zone: Optional[int] = None,
        output_path: Optional[str] = None,
        sheet_name: Optional[str] = None,
        use_geographic_calculator: bool = False,
        # ------------------------------------------------------------------
        # Output shaping (NEW)
        # ------------------------------------------------------------------
        output_schema: str = "clean",
        output_x_column: str = "X",
        output_y_column: str = "Y",
        include_crs_metadata: bool = True,
        include_error_column: bool = True,
    ) -> Dict[str, Any]:
        """
        Convert coordinates in an Excel file using pyproj (default) or Geographic Calculator COM (if requested).
        
        Args:
            excel_path: Path to Excel file
            x_column: Name of column containing X/Easting coordinates
            y_column: Name of column containing Y/Northing coordinates
            source_crs: Source coordinate reference system
            target_crs: Target coordinate reference system
            source_zone: Source UTM zone (if applicable)
            target_zone: Target UTM zone (if applicable)
            output_path: Optional output path (default: adds '_converted' to filename)
            sheet_name: Optional sheet name (default: first sheet)
            use_geographic_calculator: If True, attempt to use Geographic Calculator COM interface.
                                      If False (default), use pyproj. Always falls back to pyproj if COM fails.
        
        Returns:
            Dictionary with conversion results and statistics
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for Excel conversion. Install with: pip install pandas openpyxl")
        
        excel_file = Path(excel_path)
        if not excel_file.exists():
            raise FileNotFoundError(f"Excel file not found: {excel_path}")
        
        # Read Excel file
        try:
            if sheet_name:
                df = pd.read_excel(excel_path, sheet_name=sheet_name)
            else:
                df = pd.read_excel(excel_path, sheet_name=0)  # First sheet
        except Exception as e:
            raise ValueError(f"Failed to read Excel file: {e}")
        
        # Normalize/repair column-name matching (handles tabs/extra whitespace like "\tLong.")
        def _norm_col(c: str) -> str:
            return " ".join(str(c).replace("\t", " ").split()).strip().lower()

        norm_map = {_norm_col(c): c for c in df.columns}

        def _resolve_col(requested: str) -> Optional[str]:
            if requested in df.columns:
                return requested
            key = _norm_col(requested)
            return norm_map.get(key)

        x_column_resolved = _resolve_col(x_column)
        y_column_resolved = _resolve_col(y_column)
        if x_column_resolved is None:
            raise ValueError(f"Column '{x_column}' not found. Available columns: {list(df.columns)}")
        if y_column_resolved is None:
            raise ValueError(f"Column '{y_column}' not found. Available columns: {list(df.columns)}")
        x_column = x_column_resolved
        y_column = y_column_resolved

        # Detect geodetic-like columns for DMS parsing
        x_lower = _norm_col(x_column)
        y_lower = _norm_col(y_column)
        src_lower = (source_crs or "").lower()
        is_geodetic_input = (
            any(k in src_lower for k in ("wgs84", "wgs 84", "epsg:4326", "wkid 4326", "geographic"))
            or ("lat" in y_lower)
            or ("lon" in x_lower or "long" in x_lower)
        )
        
        # Extract coordinates
        coordinates = []
        valid_indices = []
        for idx, row in df.iterrows():
            try:
                raw_x = row[x_column]
                raw_y = row[y_column]

                if pd.isna(raw_x) or pd.isna(raw_y):
                    continue

                # Parse DMS/DM strings for geodetic columns; otherwise require numeric
                if is_geodetic_input:
                    x_parsed = parse_angle(str(raw_x))
                    y_parsed = parse_angle(str(raw_y))
                    if x_parsed is None or y_parsed is None:
                        continue
                    x_val = float(x_parsed)
                    y_val = float(y_parsed)
                else:
                    x_val = float(raw_x)
                    y_val = float(raw_y)

                if pd.notna(x_val) and pd.notna(y_val):
                    coordinates.append((x_val, y_val))
                    valid_indices.append(idx)
            except (ValueError, TypeError):
                continue
        
        if not coordinates:
            raise ValueError("No valid coordinates found in the specified columns")
        
        logger.info(f"Found {len(coordinates)} valid coordinates in Excel file")
        
        # Perform batch conversion
        results = self.batch_convert(
            coordinates, source_crs, target_crs, source_zone, target_zone,
            use_geographic_calculator=use_geographic_calculator
        )
        
        # Add converted coordinates to dataframe
        # NOTE: initialize with NaN (not None) so pandas keeps the column numeric-friendly.
        try:
            import numpy as np
        except Exception:
            np = None  # type: ignore

        # Default new columns
        x_conv_col = f"{x_column}_converted"
        y_conv_col = f"{y_column}_converted"
        df[x_conv_col] = np.nan if np is not None else None
        df[y_conv_col] = np.nan if np is not None else None
        df["conversion_method"] = None
        df["conversion_error"] = None
        
        result_idx = 0
        for idx in valid_indices:
            if result_idx < len(results):
                result = results[result_idx]
                if "error" not in result:
                    df.at[idx, x_conv_col] = result["target"]["x"]
                    df.at[idx, y_conv_col] = result["target"]["y"]
                    df.at[idx, "conversion_method"] = result["method"]
                else:
                    df.at[idx, "conversion_error"] = result["error"]
                result_idx += 1

        # ---- Numeric hygiene (critical for ArcGIS & Excel interoperability) ----
        # Excel and ArcGIS can interpret numeric-looking fields as TEXT if the dtype is object.
        # We therefore coerce converted columns to real numeric dtype before writing.
        # Exception: if user explicitly requests DMS-style output, numbers may contain symbols.
        target_lower = (target_crs or "").lower()
        user_wants_dms = any(k in target_lower for k in ("dms", "degree minute second", "degrees minutes seconds"))

        if not user_wants_dms:
            try:
                df[x_conv_col] = pd.to_numeric(df[x_conv_col], errors="coerce")
                df[y_conv_col] = pd.to_numeric(df[y_conv_col], errors="coerce")
            except Exception as e:
                logger.warning("Failed numeric coercion for converted columns (%s, %s): %s", x_conv_col, y_conv_col, e)

        # ------------------------------------------------------------------
        # Output shaping (clean vs debug)
        #
        # Problem:
        # - The previous implementation always appended multiple helper columns
        #   (CRS name/epsg, CRS-labeled XY, canonical XY, method/error), which
        #   produced "noisy" spreadsheets with many repeated-value columns.
        #
        # Fix:
        # - Default to a CLEAN output: keep the original columns, and add ONLY
        #   one converted XY pair (defaults to X/Y) plus optional minimal metadata.
        # - Preserve the old (debug) behavior via output_schema="debug".
        # ------------------------------------------------------------------
        schema = (output_schema or "clean").strip().lower()

        # Decide final output XY column names without clobbering existing data
        out_x = (output_x_column or "X").strip() or "X"
        out_y = (output_y_column or "Y").strip() or "Y"
        if out_x in df.columns and out_x not in (x_conv_col,):
            # Avoid overwriting an existing user column.
            out_x = f"{out_x}_converted"
        if out_y in df.columns and out_y not in (y_conv_col,):
            out_y = f"{out_y}_converted"

        if schema == "clean":
            # Build a clean output frame: original columns + one converted XY pair + minimal metadata.
            df_out = df.copy()
            df_out[out_x] = df_out[x_conv_col]
            df_out[out_y] = df_out[y_conv_col]

            if include_error_column:
                # Keep error information only if needed.
                if df_out["conversion_error"].notna().any():
                    df_out["conversion_error"] = df_out["conversion_error"]
                else:
                    # Drop if it's entirely empty
                    df_out = df_out.drop(columns=["conversion_error"], errors="ignore")

            # Drop verbose/debug columns
            df_out = df_out.drop(columns=["conversion_method"], errors="ignore")
            df_out = df_out.drop(columns=[x_conv_col, y_conv_col], errors="ignore")

            if include_crs_metadata:
                try:
                    target_obj = self._resolve_crs_pyproj(target_crs, target_zone, role="target_crs")
                    target_epsg = target_obj.to_epsg()
                    target_name = getattr(target_obj, "name", str(target_crs))
                    # Keep just 1-2 metadata columns (optional), rather than many CRS-labeled XY duplicates.
                    df_out["target_crs_name"] = target_name
                    df_out["target_crs_epsg"] = int(target_epsg) if target_epsg else None
                except Exception:
                    pass

            df_to_write = df_out
            written_xy_columns = [out_x, out_y]
        else:
            # Debug/legacy behavior: keep *_converted + method/error, plus optional metadata.
            df_to_write = df
            written_xy_columns = [x_conv_col, y_conv_col]

            if include_crs_metadata:
                try:
                    target_obj = self._resolve_crs_pyproj(target_crs, target_zone, role="target_crs")
                    target_epsg = target_obj.to_epsg()
                    target_name = getattr(target_obj, "name", str(target_crs))
                    df_to_write["converted_crs_name"] = target_name
                    df_to_write["converted_crs_epsg"] = int(target_epsg) if target_epsg else None
                except Exception:
                    pass
        
        # Determine output path
        if output_path is None:
            output_path = excel_file.parent / f"{excel_file.stem}_converted{excel_file.suffix}"
        else:
            out_p = Path(output_path)
            # If user provided only a filename, resolve it into the same folder as the input file
            output_path = (excel_file.parent / out_p.name) if not out_p.is_absolute() else out_p
        
        # Save results
        try:
            # Write with numeric-friendly engine. openpyxl is typical; if missing, pandas will raise clearly.
            df_to_write.to_excel(output_path, index=False)
            logger.info(f"Converted coordinates saved to: {output_path}")
        except Exception as e:
            raise IOError(f"Failed to save converted Excel file: {e}")
        
        # Calculate statistics
        successful = sum(1 for r in results if "error" not in r)
        failed = len(results) - successful
        
        # Determine actual method used from results
        actual_method = "pyproj"
        if results and "error" not in results[0]:
            actual_method = results[0].get("method", "pyproj")
        
        return {
            "success": True,
            "input_file": str(excel_file),
            "output_file": str(output_path),
            "total_coordinates": len(coordinates),
            "successful_conversions": successful,
            "failed_conversions": failed,
            "source_crs": source_crs,
            "target_crs": target_crs,
            "method": actual_method,
            "x_column": x_column,
            "y_column": y_column,
            "output_schema": schema,
            "output_x_column": out_x if schema == "clean" else x_conv_col,
            "output_y_column": out_y if schema == "clean" else y_conv_col,
            "output_columns": written_xy_columns,
        }


class GeographicCalculatorCLI:
    """Interface to Blue Marble Geographic Calculator via command-line."""
    
    def __init__(self, cmd_path: Optional[str] = None, auto_detect: bool = False):
        self.settings = get_settings()
        self._cmd_path: Optional[Path] = None
        
        if cmd_path and (path := Path(cmd_path)).exists() and path.is_file():
            self._cmd_path = path
        elif cmd_path:
            logger.warning(f"Provided path does not exist: {cmd_path}")
        
        # IMPORTANT: Don't auto-detect at startup by default.
        # CLI scanning is unnecessary for most workflows; tools that need it should call refresh().
        if auto_detect and self._cmd_path is None:
            self._cmd_path = GeographicCalculatorScanner.find_cli_executable()
        
        if self._cmd_path:
            logger.info(f"✓ Geographic Calculator CLI found: {self._cmd_path}")
        else:
            logger.info("⚠ Geographic Calculator CLI not found on system")
    
    def refresh(self) -> None:
        self._cmd_path = GeographicCalculatorScanner.find_cli_executable()
    
    @property
    def is_available(self) -> bool:
        return self._cmd_path is not None and self._cmd_path.exists()
    
    @property
    def cmd_path(self) -> Optional[Path]:
        return self._cmd_path
    
    def execute_job(
        self, job_path: str, close_after_done: bool = True,
        continue_after_error: bool = False, timeout: int = 300
    ) -> Dict[str, Any]:
        if not self.is_available:
            return {"success": False, "error": "Geographic Calculator CLI not found", "output": "", "exit_code": -1}
        
        job_file = Path(job_path)
        if not job_file.exists():
            return {"success": False, "error": f"Job file not found: {job_path}", "output": "", "exit_code": -1}
        
        try:
            cmd = [
                str(self._cmd_path), str(job_file.absolute()),
                "true" if close_after_done else "false",
                "true" if continue_after_error else "false"
            ]
            logger.info(f"Executing Geographic Calculator job: {job_file.name}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(job_file.parent))
            return {
                "success": result.returncode == 0,
                "output": result.stdout + result.stderr,
                "error": result.stderr if result.returncode != 0 else None,
                "exit_code": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": f"Job execution timed out after {timeout} seconds", "output": "", "exit_code": -1}
        except Exception as e:
            logger.error(f"Error executing job: {e}", exc_info=True)
            return {"success": False, "error": str(e), "output": "", "exit_code": -1}
    
    def get_version(self) -> Optional[str]:
        if not self.is_available:
            return None
        try:
            result = subprocess.run([str(self._cmd_path), "--version"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        try:
            import win32api
            version_info = win32api.GetFileVersionInfo(str(self._cmd_path), "\\")
            return f"{version_info['FileVersionMS'] >> 16}.{version_info['FileVersionMS'] & 0xFFFF}"
        except Exception:
            pass
        return None
