"""
Lightweight machine capability scan for Windows integrations.

Does not start AutoCAD or open COM connections to Blue Marble by default;
suitable for splash/settings screens and installers.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MachineCapabilities:
    """What this PC appears to support (best-effort, no COM connect to AutoCAD)."""

    platform: str
    pywin32_available: bool
    autocad_registry_detected: bool
    dxf_ezdxf_available: bool
    arcgis_pro_installed: bool
    arcgis_arcpy_available: bool
    geographic_calculator_cli: Optional[str]
    python_version: str


def _pywin32_available() -> bool:
    try:
        import win32com.client  # noqa: F401

        return True
    except ImportError:
        return False


def _autocad_registry_present() -> bool:
    if sys.platform != "win32":
        return False
    try:
        import winreg

        for hive, path in (
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Autodesk\AutoCAD"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Autodesk\AutoCAD"),
        ):
            try:
                key = winreg.OpenKey(hive, path)
                winreg.CloseKey(key)
                return True
            except OSError:
                continue
    except ImportError:
        pass
    return False


def _ezdxf_available() -> bool:
    return importlib.util.find_spec("ezdxf") is not None


def scan_machine_capabilities() -> MachineCapabilities:
    """Probe integrations without initializing the full agent."""
    geo_cli: Optional[str] = None
    try:
        from tools.geographic_calculator import GeographicCalculatorScanner

        found = GeographicCalculatorScanner.find_cli_executable()
        if found is not None:
            geo_cli = str(found)
    except Exception:
        geo_cli = None

    arcgis_installed = False
    arcpy_available = False
    try:
        from tools.arcgis_tools import ArcGISProcessor

        p = ArcGISProcessor()
        arcgis_installed = bool(p.is_installed)
        arcpy_available = bool(p.is_available)
    except Exception:
        pass

    return MachineCapabilities(
        platform=sys.platform,
        pywin32_available=_pywin32_available(),
        autocad_registry_detected=_autocad_registry_present(),
        dxf_ezdxf_available=_ezdxf_available(),
        arcgis_pro_installed=arcgis_installed,
        arcgis_arcpy_available=arcpy_available,
        geographic_calculator_cli=geo_cli,
        python_version=sys.version.split()[0],
    )


def format_capabilities_summary(caps: MachineCapabilities) -> str:
    """Human-readable block for CLI `test` or support bundles."""
    lines = [
        f"Platform: {caps.platform} (Python {caps.python_version})",
        f"pywin32 (COM): {'yes' if caps.pywin32_available else 'no'}",
        f"AutoCAD (registry hint): {'yes' if caps.autocad_registry_detected else 'no'}",
        f"ezdxf (DXF fallback): {'yes' if caps.dxf_ezdxf_available else 'no'}",
        f"ArcGIS Pro installed: {'yes' if caps.arcgis_pro_installed else 'no'}",
        f"arcpy available: {'yes' if caps.arcgis_arcpy_available else 'no'}",
        f"Geographic Calculator CLI: {caps.geographic_calculator_cli or 'not found'}",
    ]
    return "\n".join(lines)


__all__ = ["MachineCapabilities", "scan_machine_capabilities", "format_capabilities_summary"]
