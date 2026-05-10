"""Excel file processor for extracting coordinates and geospatial data."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from utils.logger import get_logger

logger = get_logger(__name__)


class ExcelProcessor:
    """Process Excel files to extract coordinates and geospatial data."""
    
    def __init__(self):
        """Initialize the Excel processor."""
        self.supported_formats = ['.xlsx', '.xls', '.xlsm']

    def list_sheets(self, file_path: str) -> Dict[str, Any]:
        """
        List all worksheet names in an Excel workbook.
        Use this FIRST when the user refers to named data (e.g. "Pre-fill", "Post-fill")
        so you can map their terms to actual sheet names.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            return {"success": False, "error": f"File not found: {file_path}", "sheets": []}
        if file_path.suffix not in self.supported_formats:
            return {"success": False, "error": f"Unsupported format: {file_path.suffix}", "sheets": []}
        try:
            xl = pd.ExcelFile(file_path, engine="openpyxl")
            sheets = xl.sheet_names
            xl.close()
            logger.info(f"Listed {len(sheets)} sheet(s) in {file_path}")
            return {"success": True, "sheets": sheets, "file_path": str(file_path)}
        except Exception as e:
            logger.error(f"Error listing sheets: {e}")
            return {"success": False, "error": str(e), "sheets": []}

    def inspect_workbook(self, file_path: str) -> Dict[str, Any]:
        """
        Inspect workbook structure: list all sheet names and each sheet's column headers
        (and optionally first row of data). Use this to discover actual sheet and column
        names before calling ArcGIS/Excel tools—match user terms (e.g. "Pre-fill", "X/Y/Z")
        to real names (e.g. "Pre_Fill_2024", "EASTING", "NORTHING", "RL").
        """
        file_path = Path(file_path)
        if not file_path.exists():
            return {"success": False, "error": f"File not found: {file_path}", "workbook": {}}
        if file_path.suffix not in self.supported_formats:
            return {"success": False, "error": f"Unsupported format: {file_path.suffix}", "workbook": {}}
        try:
            xl = pd.ExcelFile(file_path, engine="openpyxl")
            workbook: Dict[str, Any] = {"file_path": str(file_path), "sheets": {}}
            for name in xl.sheet_names:
                try:
                    df = pd.read_excel(xl, sheet_name=name, nrows=1, header=0)
                    headers = [str(c) for c in df.columns]
                    workbook["sheets"][name] = {"columns": headers}
                except Exception as e:
                    workbook["sheets"][name] = {"columns": [], "error": str(e)}
            xl.close()
            logger.info(f"Inspected workbook {file_path}: {list(workbook['sheets'].keys())}")
            return {"success": True, "workbook": workbook}
        except Exception as e:
            logger.error(f"Error inspecting workbook: {e}")
            return {"success": False, "error": str(e), "workbook": {}}

    def read_file(self, file_path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
        """
        Read an Excel file and return as DataFrame.

        Args:
            file_path: Path to the Excel file
            sheet_name: Optional sheet name or 0-based index. If None, reads first sheet.

        Returns:
            DataFrame containing the Excel data
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Excel file not found: {file_path}")

        if file_path.suffix not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        try:
            kwargs = {"engine": "openpyxl"}
            if sheet_name is not None:
                kwargs["sheet_name"] = sheet_name
            df = pd.read_excel(file_path, **kwargs)
            logger.info(f"Successfully read Excel file: {file_path}" + (f" sheet={sheet_name}" if sheet_name is not None else ""))
            return df
        except Exception as e:
            logger.error(f"Error reading Excel file: {e}")
            raise
    
    def extract_coordinates(
        self,
        df: pd.DataFrame,
        x_column: Optional[str] = None,
        y_column: Optional[str] = None,
        z_column: Optional[str] = None,
        coordinate_system: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract coordinates from DataFrame.
        
        Args:
            df: DataFrame containing coordinate data
            x_column: Name of X/Easting column (auto-detected if None)
            y_column: Name of Y/Northing column (auto-detected if None)
            z_column: Name of Z/Elevation column (optional)
            coordinate_system: Coordinate system identifier (optional)
            
        Returns:
            Dictionary containing extracted coordinates and metadata
        """
        result = {
            "coordinates": [],
            "count": 0,
            "coordinate_system": coordinate_system,
            "columns_used": {}
        }
        
        # Auto-detect coordinate columns if not specified
        if x_column is None or y_column is None:
            x_column, y_column = self._detect_coordinate_columns(df)
        
        if x_column is None or y_column is None:
            raise ValueError("Could not detect coordinate columns. Please specify x_column and y_column.")
        
        result["columns_used"]["x"] = x_column
        result["columns_used"]["y"] = y_column
        
        # Extract coordinates
        for idx, row in df.iterrows():
            try:
                x = float(row[x_column]) if pd.notna(row[x_column]) else None
                y = float(row[y_column]) if pd.notna(row[y_column]) else None
                z = float(row[z_column]) if z_column and pd.notna(row.get(z_column)) else None
                
                if x is not None and y is not None:
                    coord = {"x": x, "y": y, "index": int(idx)}
                    if z is not None:
                        coord["z"] = z
                    result["coordinates"].append(coord)
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping row {idx} due to invalid coordinate data: {e}")
                continue
        
        result["count"] = len(result["coordinates"])
        
        # Calculate statistics
        if result["count"] > 0:
            x_values = [c["x"] for c in result["coordinates"]]
            y_values = [c["y"] for c in result["coordinates"]]
            result["statistics"] = {
                "x_range": [min(x_values), max(x_values)],
                "y_range": [min(y_values), max(y_values)],
                "x_mean": np.mean(x_values),
                "y_mean": np.mean(y_values)
            }
            if any("z" in c for c in result["coordinates"]):
                z_values = [c["z"] for c in result["coordinates"] if "z" in c]
                result["statistics"]["z_range"] = [min(z_values), max(z_values)]
                result["statistics"]["z_mean"] = np.mean(z_values)
        
        logger.info(f"Extracted {result['count']} coordinates from Excel file")
        return result
    
    def _detect_coordinate_columns(self, df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
        """
        Auto-detect coordinate column names.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Tuple of (x_column, y_column) names
        """
        # NOTE:
        # Header-only heuristics are brittle (users often have many "X/Y" columns,
        # converted columns, or non-standard names). We therefore:
        # 1) Try a value-based inference (preferred)
        # 2) Fall back to header heuristics (legacy)
        x_by_value, y_by_value = self._infer_coordinate_columns_by_values(df)
        if x_by_value and y_by_value:
            return x_by_value, y_by_value

        # Legacy fallback: common coordinate column name patterns
        x_patterns = ['x', 'easting', 'east', 'lon', 'longitude', 'long']
        y_patterns = ['y', 'northing', 'north', 'lat', 'latitude']

        x_column = None
        y_column = None

        for col in df.columns:
            col_lower = str(col).lower()
            if any(pattern in col_lower for pattern in x_patterns) and x_column is None:
                x_column = col
            if any(pattern in col_lower for pattern in y_patterns) and y_column is None:
                y_column = col

        return x_column, y_column

    def _infer_coordinate_columns_by_values(self, df: pd.DataFrame, sample_size: int = 50) -> Tuple[Optional[str], Optional[str]]:
        """
        Infer coordinate columns by sampling values (more robust than name matching).

        Strategy:
        - Parse numeric values (handles commas) and DMS/DM strings
        - Prefer geodetic (lon/lat) pair when values fall within [-180,180]/[-90,90]
        - Otherwise prefer projected (easting/northing) pair based on magnitudes
        """
        try:
            from utils.coordinate_parsing import parse_angle
        except Exception:
            parse_angle = None  # type: ignore

        def _to_float(v):
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return None
            # DMS/DM parsing if available
            if parse_angle is not None and isinstance(v, str):
                if any(ch in v for ch in ("°", "º", "'", '"')) or any(h in v.upper() for h in ("N", "S", "E", "W")):
                    ang = parse_angle(v)
                    if ang is not None:
                        return float(ang)
            # Numeric parse
            try:
                s = str(v).replace(",", "").strip()
                if s == "":
                    return None
                return float(s)
            except Exception:
                return None

        stats = {}
        for col in df.columns:
            vals = []
            for v in df[col].dropna().head(sample_size).tolist():
                fv = _to_float(v)
                if fv is not None:
                    vals.append(fv)
            if not vals:
                continue
            stats[col] = {
                "count": len(vals),
                "min": min(vals),
                "max": max(vals),
                "mean_abs": float(np.mean([abs(x) for x in vals])),
            }

        if not stats:
            return None, None

        # Candidate lists
        lon_cands = []
        lat_cands = []
        east_cands = []
        north_cands = []

        for col, s in stats.items():
            mn, mx = s["min"], s["max"]
            mean_abs = s["mean_abs"]
            # Geodetic
            if -180.0 <= mn <= 180.0 and -180.0 <= mx <= 180.0:
                lon_cands.append((col, s["count"]))
            if -90.0 <= mn <= 90.0 and -90.0 <= mx <= 90.0:
                lat_cands.append((col, s["count"]))
            # Projected (loose magnitude gates)
            if mean_abs >= 1000.0:
                # Easting often in 100k-900k range; northing can be broader, but both are large.
                east_cands.append((col, s["count"], mean_abs))
                north_cands.append((col, s["count"], mean_abs))

        # Prefer geodetic if we can find a plausible lon/lat pair.
        if lon_cands and lat_cands:
            # Break ties using header hints (but not required)
            def _hint_score(c: str) -> int:
                cl = str(c).lower()
                score = 0
                if any(k in cl for k in ("lon", "long", "longitude")):
                    score += 3
                if "lat" in cl or "latitude" in cl:
                    score += 3
                if cl.strip() in ("x", "y"):
                    score += 1
                return score

            lon = sorted(lon_cands, key=lambda t: (t[1], _hint_score(t[0])), reverse=True)[0][0]
            lat = sorted(lat_cands, key=lambda t: (t[1], _hint_score(t[0])), reverse=True)[0][0]
            # Ensure we didn't pick the same column twice
            if lon != lat:
                return lon, lat

        # Otherwise, pick the two strongest projected-like numeric columns and apply header hints.
        proj = sorted(east_cands, key=lambda t: (t[1], t[2]), reverse=True)
        if len(proj) >= 2:
            # Try to label easting/northing using name hints
            def _is_easting(name: str) -> bool:
                nl = str(name).lower()
                return any(k in nl for k in ("east", "easting", "x"))

            def _is_northing(name: str) -> bool:
                nl = str(name).lower()
                return any(k in nl for k in ("north", "northing", "y"))

            c1, c2 = proj[0][0], proj[1][0]
            if _is_easting(c1) and _is_northing(c2):
                return c1, c2
            if _is_easting(c2) and _is_northing(c1):
                return c2, c1
            # Fallback: treat first as X, second as Y
            return c1, c2

        return None, None
    
    def process_file(
        self,
        file_path: str,
        x_column: Optional[str] = None,
        y_column: Optional[str] = None,
        z_column: Optional[str] = None,
        coordinate_system: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Complete processing pipeline: read and extract coordinates.
        
        Args:
            file_path: Path to Excel file
            x_column: X coordinate column name (optional)
            y_column: Y coordinate column name (optional)
            z_column: Z coordinate column name (optional)
            coordinate_system: Coordinate system identifier (optional)
            
        Returns:
            Dictionary with extracted data and metadata
        """
        df = self.read_file(file_path)
        coordinates = self.extract_coordinates(df, x_column, y_column, z_column, coordinate_system)
        
        return {
            "file_path": str(file_path),
            "total_rows": len(df),
            "columns": list(df.columns),
            "coordinates": coordinates
        }

    def csv_to_excel(
        self,
        csv_path: str,
        output_excel_path: Optional[str] = None,
        encoding: str = "utf-8",
    ) -> Dict[str, Any]:
        """
        Convert a CSV file to an Excel workbook (.xlsx).

        Use this when a downstream tool (e.g. ArcGIS ExcelToTable, excel_coordinate_convert,
        or arcgis_import_xy_points_from_excel) requires .xlsx/.xls and the user provides a .csv.
        Output defaults to the same folder as the CSV with the same base name and .xlsx extension.

        Args:
            csv_path: Path to the CSV file.
            output_excel_path: Path for the output Excel file. If None, uses same folder as CSV, same stem, .xlsx.
            encoding: CSV encoding (default utf-8). Tries utf-8 first, then latin-1 if needed.

        Returns:
            Dict with success, file_path, output_path, total_rows, columns, error.
        """
        csv_p = Path(csv_path).resolve()
        if not csv_p.exists():
            return {"success": False, "error": f"CSV file not found: {csv_path}"}
        if csv_p.suffix.lower() != ".csv":
            return {"success": False, "error": f"File is not a CSV: {csv_p.suffix}"}

        out_p = Path(output_excel_path).resolve() if output_excel_path else csv_p.with_suffix(".xlsx")
        out_p.parent.mkdir(parents=True, exist_ok=True)

        try:
            df = pd.read_csv(csv_p, encoding=encoding)
        except Exception:
            try:
                df = pd.read_csv(csv_p, encoding="latin-1")
            except Exception as e:
                return {"success": False, "error": f"Failed to read CSV: {e}"}

        try:
            df.to_excel(out_p, index=False, engine="openpyxl")
        except Exception as e:
            return {"success": False, "error": f"Failed to write Excel: {e}"}

        logger.info(f"Converted CSV to Excel: {csv_p} -> {out_p}, rows={len(df)}")
        return {
            "success": True,
            "file_path": str(csv_p),
            "output_path": str(out_p),
            "total_rows": len(df),
            "columns": list(df.columns),
        }

