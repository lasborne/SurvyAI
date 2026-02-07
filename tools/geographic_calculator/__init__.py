"""
Blue Marble Geographic Calculator Interface

Provides COM and CLI interfaces for coordinate conversions and geodetic operations.

Note: This directory contains split modules (scanner, constants), but the main
classes are in the parent module tools.geographic_calculator (the .py file).
This __init__.py re-exports from the parent module for compatibility.
"""

# Import from the parent module (geographic_calculator.py file)
# We need to use importlib to avoid circular imports since Python resolves
# tools.geographic_calculator to this directory first
import sys
import importlib.util
from pathlib import Path

# Load the parent geographic_calculator.py module
_parent_module_path = (Path(__file__).parent.parent / "geographic_calculator.py")
if _parent_module_path.exists():
    spec = importlib.util.spec_from_file_location("geographic_calculator_module", _parent_module_path)
    _parent_module = importlib.util.module_from_spec(spec)
    sys.modules["geographic_calculator_module"] = _parent_module
    spec.loader.exec_module(_parent_module)
    
    GeographicCalculatorScanner = _parent_module.GeographicCalculatorScanner
    BlueMarbleConverter = _parent_module.BlueMarbleConverter
    GeographicCalculatorCLI = _parent_module.GeographicCalculatorCLI
else:
    # Fallback: try importing from scanner if parent doesn't exist
    from tools.geographic_calculator.scanner import GeographicCalculatorScanner
    # These won't work but at least we tried
    BlueMarbleConverter = None
    GeographicCalculatorCLI = None

__all__ = [
    "GeographicCalculatorScanner",
    "BlueMarbleConverter",
    "GeographicCalculatorCLI",
]

