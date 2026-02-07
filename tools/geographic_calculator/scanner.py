"""
Geographic Calculator Scanner

Handles detection and scanning for Geographic Calculator installations.
"""

from __future__ import annotations

import os
import subprocess
import shutil
from pathlib import Path
from typing import List, Optional

from config import get_settings
from utils.logger import get_logger
from tools.geographic_calculator.constants import (
    GEOCALC_COMMON_PATHS,
    GEOCALC_CMD_COMMON_PATHS,
    REGISTRY_PATHS,
    SEARCH_PATTERNS,
)

logger = get_logger(__name__)


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
            
            project_root = Path(__file__).parent.parent.parent
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

