"""
Constants for Geographic Calculator interface.
"""

from typing import Tuple, List

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

REGISTRY_PATHS: List[str] = [
    r"SOFTWARE\Blue Marble Geographics\Geographic Calculator",
    r"SOFTWARE\WOW6432Node\Blue Marble Geographics\Geographic Calculator",
    r"SOFTWARE\Blue Marble\Geographic Calculator",
    r"SOFTWARE\Blue Marble Geographics\GeoCalc Pro",
    r"SOFTWARE\WOW6432Node\Blue Marble Geographics\GeoCalc Pro",
]

SEARCH_PATTERNS: List[str] = ["Blue Marble Geographics", "Blue Marble Geo", "Geographic Calculator"]

