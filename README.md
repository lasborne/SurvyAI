# SurvyAI - AI Agent for Surveyors

SurvyAI is an intelligent AI agent that controls native surveying and CAD software on your computer. Using LLM reasoning (Gemini, DeepSeek, Claude, or OpenAI), it interprets your natural language requests and executes them through application APIs.

## Key Features

- **AutoCAD Integration**: Opens drawings, extracts data, calculates areas using AutoCAD's native engine
- **Natural Language Control**: Ask questions like "What is the area of the red-verged boundary?"
- **Text Extraction**: Find owner names, titles, and annotations from drawings
- **Coordinate Conversion**: Transform coordinates between reference systems
- **Document Analysis**: Extract information from PDF and Word documents
- **Excel Processing**: Read coordinate data from spreadsheets

## Requirements

### Software
- **AutoCAD** (any recent version) - Required for CAD operations
- **Python 3.10+**
- **Windows** (for COM automation)

### API Keys
- At least one of the following API keys is required:
  - **Google Gemini** API key (for Gemini models)
  - **DeepSeek** API key (for DeepSeek models)
  - **Anthropic** API key (for Claude Opus/Sonnet/Haiku models)
  - **OpenAI** API key (for GPT-4, GPT-4o, GPT-4o-Turbo, GPT-5 models)

## Installation

```bash
# Clone and install
git clone <repository-url>
cd SurvyAI
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your API keys
```

## Usage

### Command Line

```bash
# Simple query
python -m cli query "What is the history of surveying?"

# AutoCAD operations (AutoCAD must be running)
python -m cli query "Open survey_data.dwg and calculate the area of red boundaries"

# Find property owner
python -m cli query "Search for 'property of' in the drawing and tell me the owner's name"
```

### Python API

```python
from agent import SurvyAIAgent

agent = SurvyAIAgent()

# The LLM reasons about your request and controls AutoCAD
result = agent.process_query("""
    Open survey_data.dwg and:
    1. Find the property owner's name
    2. Calculate the total area of land verged in red
    3. Report in both metric and imperial units
""")

print(result["response"])
```

## How It Works

1. **You ask a question** in natural language
2. **The LLM reasons** about what operations are needed
3. **Tools execute** via AutoCAD's COM API, Excel, etc.
4. **Results are analyzed** and returned in a clear response

## Available Tools

| Tool | Description |
|------|-------------|
| `autocad_open_drawing` | Open DWG/DXF files in AutoCAD |
| `autocad_calculate_area` | Calculate areas of closed shapes |
| `autocad_search_text` | Find text matching patterns |
| `autocad_get_entities` | Get entities by type, layer, or color |
| `excel_processor` | Extract data from Excel files |
| `document_processor` | Parse PDF/Word documents |
| `coordinate_converter` | Convert between coordinate systems |
| `geographic_calculator_check` | Check if Geographic Calculator is installed |
| `geographic_calculator_execute_job` | Execute Geographic Calculator jobs |

## Interactive Permissions

SurvyAI supports interactive permission requests for file operations. See [docs/PERMISSIONS.md](docs/PERMISSIONS.md) for complete documentation on:
- How to enable interactive mode
- When permissions are needed
- How to respond to permission requests
- Security and privacy information

## Project Structure

```
SurvyAI/
├── agent/agent.py          # Main AI agent with LLM integration
├── tools/
│   ├── autocad_processor.py  # AutoCAD COM automation
│   ├── excel_processor.py    # Excel file handling
│   └── ...
├── config/settings.py      # Configuration management
├── cli.py                  # Command-line interface
└── requirements.txt
```

## License

MIT License
