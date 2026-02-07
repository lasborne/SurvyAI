"""
================================================================================
SurvyAI Tools Package
================================================================================

This package contains all the tool modules that enable the SurvyAI agent to
interact with external software and process various file types.

TOOL OVERVIEW:
--------------
Each tool provides specific capabilities that the LLM can invoke to perform
real-world operations:

1. ExcelProcessor
   - Read Excel files (.xlsx, .xls)
   - Extract coordinate data
   - Process survey point lists
   - Handle tabular data

2. DocumentProcessor
   - Extract text from PDF documents
   - Read Word documents (.docx, .doc)
   - Parse survey reports and legal documents

3. AutoCADProcessor
   - Connect to AutoCAD via COM API
   - Open DWG and DXF drawings
   - Extract text, entities, and properties
   - Calculate areas using AutoCAD's native engine
   - Execute AutoCAD commands

4. BlueMarbleConverter
   - Convert between coordinate reference systems
   - Support for WGS84, UTM, State Plane, etc.
   - Falls back to pyproj if Blue Marble not available

5. GeographicCalculatorCLI
   - Execute Geographic Calculator jobs/projects/workspaces via command-line
   - Perform batch coordinate conversions using pre-configured jobs
   - Interface with GeographicCalculatorCMD.exe

6. ArcGISProcessor
   - Perform advanced GIS operations (if arcpy available)
   - Spatial analysis and data processing
   - Work with geodatabases and feature classes

7. VectorStore
   - Semantic search using vector embeddings
   - Local storage with ChromaDB
   - Multiple embedding providers (local Sentence Transformers, OpenAI)
   - Collections for documents, drawings, coordinates, conversations
   - Persistent storage for long-term memory

HOW TOOLS WORK WITH LANGGRAPH:
------------------------------
1. Each tool class provides methods that perform specific operations
2. In agent.py, these methods are wrapped as LangChain StructuredTools
3. The LLM is given descriptions of each tool's capabilities
4. When processing a query, the LLM decides which tools to call
5. Tool results are fed back to the LLM for final response

ADDING NEW TOOLS:
-----------------
To add a new tool:
1. Create a new module in this directory (e.g., gps_processor.py)
2. Implement a class with methods for each operation
3. Import it in this __init__.py file
4. Add it to __all__
5. Create corresponding StructuredTools in agent.py

Example:
    ```python
    # tools/gps_processor.py
    class GPSProcessor:
        def read_gpx_file(self, file_path: str) -> Dict[str, Any]:
            \"\"\"Read a GPX file and extract waypoints.\"\"\"
            ...
    
    # tools/__init__.py
    from tools.gps_processor import GPSProcessor
    __all__ = [..., "GPSProcessor"]
    ```

Author: SurvyAI Team
License: MIT
================================================================================
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

# Each processor handles a specific type of data or software interface

# Excel file processing (coordinates, point lists, tabular data)
from tools.excel_processor import ExcelProcessor

# Document processing (PDF, Word documents)
from tools.document_processor import DocumentProcessor

# AutoCAD COM interface (native CAD operations)
from tools.autocad_processor import AutoCADProcessor

# Geographic Calculator interfaces (COM and CLI) - unified module
from tools.geographic_calculator import BlueMarbleConverter, GeographicCalculatorCLI

# ArcGIS Pro integration (advanced GIS operations)
from tools.arcgis_tools import ArcGISProcessor

# Vector database for semantic search
from tools.vector_store import (
    VectorStore,
    create_vector_store,
    COLLECTION_DOCUMENTS,
    COLLECTION_DRAWINGS,
    COLLECTION_COORDINATES,
    COLLECTION_CONVERSATIONS,
)


# ==============================================================================
# PUBLIC API
# ==============================================================================

# __all__ defines what gets exported when someone does:
# from tools import *
# 
# It's also used by IDEs for autocompletion and static analysis

__all__ = [
    "ExcelProcessor",           # Read Excel files with coordinate data
    "DocumentProcessor",        # Extract text from PDF/Word documents  
    "AutoCADProcessor",         # Control AutoCAD via COM API
    "BlueMarbleConverter",      # Coordinate reference system conversions
    "GeographicCalculatorCLI",  # Geographic Calculator command-line interface
    "ArcGISProcessor",          # ArcGIS Pro operations (requires arcpy)
    "VectorStore",              # Vector database for semantic search
    "create_vector_store",      # Convenience function for creating VectorStore
    "COLLECTION_DOCUMENTS",     # Collection name constant
    "COLLECTION_DRAWINGS",      # Collection name constant
    "COLLECTION_COORDINATES",   # Collection name constant
    "COLLECTION_CONVERSATIONS", # Collection name constant
]
