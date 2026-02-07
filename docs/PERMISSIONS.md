# Interactive Permissions Guide

## Overview

SurvyAI is designed to be transparent and respectful of your privacy. When the agent needs to access system information or perform operations that might require permission, it will ask you interactively with a clear explanation of **why** it needs access and **what** it will do.

---

## Quick Start

### Enable Interactive Mode

Add the `--interactive` (or `-i`) flag to enable permission requests:

```bash
python -m cli query "Read coordinates from survey.xlsx" --interactive
```

### Respond to Permission Prompts

When the agent asks for permission, you'll see:

```
⚠️  PERMISSION REQUEST DETECTED

Your response [yes]: 
```

**Type your response:**
- `yes` or press Enter → Grant permission
- `no` → Deny permission
- `exit` → End conversation

---

## How It Works

### Step-by-Step Process

1. **Agent processes your query** and may ask for permission
2. **Agent pauses** when it needs your input
3. **You respond** with yes, no, or a custom message
4. **Agent continues** based on your response
5. **Conversation continues** until you type 'exit'

### Complete Example Session

```bash
$ python -m cli query "Read coordinates from survey.xlsx" --interactive

🔍 Processing query: Read coordinates from survey.xlsx
⏳ Initializing SurvyAI agent...
🤖 Agent ready. Processing your query...

============================================================
💬 INTERACTIVE MODE
============================================================
Type 'exit' or 'quit' to end the conversation

============================================================
🤖 AGENT RESPONSE:
============================================================
To read the coordinates from survey.xlsx, I need to access the file contents.
I will:
- Open the Excel file
- Read coordinate data from the specified sheets
- Extract X, Y, and optionally Z coordinates
- Not modify or save anything

May I proceed with reading survey.xlsx? (yes/no)
============================================================

────────────────────────────────────────────────────────────
⚠️  PERMISSION REQUEST DETECTED
────────────────────────────────────────────────────────────

Your response [yes]: yes
✅ Permission granted. Continuing...

============================================================
🤖 AGENT RESPONSE:
============================================================
✓ Successfully read coordinates from survey.xlsx
Found 150 coordinate points:
- X coordinates range: 123456.78 to 234567.89
- Y coordinates range: 987654.32 to 876543.21
- Coordinate system: WGS84
============================================================

────────────────────────────────────────────────────────────
Continue conversation? (type your next question, or 'exit' to quit): exit

👋 Ending conversation. Goodbye!
```

---

## Permission Requirements

### ✅ Automatic Checks (No Permission Needed)

These operations are **read-only** and work automatically:

- Checking if software is installed (e.g., Geographic Calculator, AutoCAD)
- Checking file paths and directory structures
- Reading file metadata (size, date, etc.)
- Checking system registry for installation paths

**Example:**
```bash
# No --interactive needed
python -m cli query "Check if Geographic Calculator is available"
```

### 📝 Operations That Require Permission

The agent will ask for permission before:

- Reading file contents (CAD files, documents, Excel files)
- Executing commands that modify files
- Writing or modifying files
- Accessing network resources
- Accessing sensitive system information

**Example:**
```bash
# --interactive required
python -m cli query "Read coordinates from survey.xlsx" --interactive
```

---

## Response Options

### Granting Permission

When asked for permission, you can respond with:
- `yes` or `y`
- `ok` or `okay`
- `sure`
- `proceed`
- `go ahead`
- Just press Enter (defaults to `yes`)

### Denying Permission

To deny permission:
- `no` or `n`
- `deny`
- `cancel`
- `abort`

The agent will suggest alternative approaches.

### Custom Responses

You can type any custom message:
```
"Only read the first 10 rows"
"Skip the header row"
"Use sheet 2 instead"
```

### Ending the Conversation

To exit at any time:
- `exit`
- `quit`
- `q`
- Just press Enter when prompted to continue (if you don't want to continue)

---

## Common Scenarios

### Scenario 1: Checking Software Availability

**Command:**
```bash
python -m cli query "Check if Geographic Calculator is available"
```

**Result:**
```
✓ Geographic Calculator CLI is available on your system!
Executable Path: C:\Program Files\Blue Marble Geographics\Geographic Calculator 2025\GeographicCalculatorCMD.exe
Version: 2025.1
```

**Note:** No `--interactive` needed - this is a read-only check.

### Scenario 2: Reading Files

**Command:**
```bash
python -m cli query "Read coordinates from survey.xlsx" --interactive
```

**Flow:**
1. Agent asks permission → You type `yes`
2. Agent reads file and returns results
3. You can continue or type `exit`

### Scenario 3: Multiple Operations

**Command:**
```bash
python -m cli query "Read survey.xlsx and calculate areas" --interactive
```

**Flow:**
1. Agent asks permission to read Excel file → You type `yes`
2. Agent asks permission to access AutoCAD → You type `yes`
3. Agent performs operations and reports results
4. You type `exit` to finish

### Scenario 4: Denying Permission

**Command:**
```bash
python -m cli query "Read my private file.xlsx" --interactive
```

**Permission Request:**
```
May I proceed with reading file.xlsx? (yes/no)
```

**Your Response:**
```
no
```

**Agent Response:**
```
❌ Permission denied. The agent will suggest alternatives.

I understand you don't want me to access that file. Here are some alternatives:
1. You can manually extract the data and paste it here
2. I can help you with a different file
3. I can guide you through the process without accessing the file
```

---

## Available Tools

### Geographic Calculator Tools

1. **geographic_calculator_check**
   - Checks if Geographic Calculator CLI is installed
   - Returns version and executable path
   - **No permission needed** - read-only check

2. **geographic_calculator_execute_job**
   - Executes pre-configured job files (.gpj, .gpp, .gpw)
   - Requires a job file created in Geographic Calculator GUI
   - **May require permission** if the job modifies files

### Other Available Tools

- **autocad_*** tools: AutoCAD operations (may require permission for file access)
- **excel_processor**: Read Excel files (requires permission)
- **document_processor**: Read PDF/Word files (requires permission)
- **coordinate_converter**: Convert coordinates (no permission needed)
- **arcgis_*** tools: ArcGIS Pro operations (may require permission)

---

## Best Practices

### 1. Be Specific in Your Queries

**✅ Good:**
```
"Check if Geographic Calculator is installed and show me the path"
"Is Geographic Calculator available?"
"Where is Geographic Calculator installed?"
```

**❌ Less Effective:**
```
"Check my system"
"What software do I have?"
```

### 2. Use Interactive Mode for File Operations

```bash
# ✅ Good - Interactive mode for file access
python -m cli query "Read coordinates from data.xlsx" --interactive

# ❌ Less effective - No way to grant permission
python -m cli query "Read coordinates from data.xlsx"
```

### 3. Grant Permission When Appropriate

When the agent asks for permission with a clear reason:
- ✅ **Grant permission** if you understand and trust the operation
- ❌ **Deny permission** if you're unsure or don't want the operation
- ❓ **Ask for clarification** if the reason isn't clear

### 4. Follow Up Naturally

After checking availability or completing operations:
```
"Now execute the job at C:\Jobs\convert.gpj"
"Run my saved conversion project"
"Check the version number"
```

---

## Security and Privacy

### What SurvyAI Does NOT Do

- ❌ Access files without asking (when permission is needed)
- ❌ Modify files without explicit user request
- ❌ Access network resources without permission
- ❌ Store sensitive information without your knowledge
- ❌ Share data with external services (except LLM APIs for processing)

### What SurvyAI Does

- ✅ Asks for permission before accessing files
- ✅ Explains why access is needed
- ✅ Only performs requested operations
- ✅ Logs operations for transparency
- ✅ Respects your privacy choices

---

## Troubleshooting

### "The conversation closes before I can respond"

**Solution:** You forgot the `--interactive` flag. Add it:
```bash
python -m cli query "Your question" --interactive
```

### "I can't type my response"

**Solution:** Make sure:
1. You're in the correct terminal window
2. The cursor is at the prompt
3. You type your response and press **Enter**

### "The agent doesn't ask for permission"

**Possible reasons:**
1. The operation doesn't require permission (e.g., software checks)
2. The agent detected it can proceed automatically
3. You're not in interactive mode

**Solution:** Use `--interactive` flag and be explicit about file operations.

### "Agent doesn't use tools when it should"

**Solution:**
1. **Be more specific**: Include keywords like "check", "verify", "find", "locate"
2. **Mention the software name**: "Geographic Calculator", "AutoCAD", etc.
3. **Ask directly**: "Use the geographic_calculator_check tool"

### "Permission denied - what now?"

If you deny permission, the agent will:
- Suggest alternative approaches
- Explain what can be done without that permission
- Offer to help in other ways

---

## Common Commands

### Check Software (No Interactive Needed)
```bash
python -m cli query "Check if Geographic Calculator is available"
```

### Read File (Needs Interactive)
```bash
python -m cli query "Read coordinates from survey.xlsx" --interactive
```

### Execute Job (Needs Interactive if File Access Required)
```bash
python -m cli query "Execute the Geographic Calculator job at C:\Jobs\convert.gpj" --interactive
```

---

## Examples of Effective Queries

### Checking Software Availability
```
✅ "Check if Geographic Calculator is installed"
✅ "Is Geographic Calculator available on my system?"
✅ "Find the Geographic Calculator installation path"
✅ "What version of Geographic Calculator do I have?"
```

### Executing Operations
```
✅ "Execute the Geographic Calculator job at C:\Jobs\convert.gpj"
✅ "Run my coordinate conversion job"
✅ "Process the batch conversion using my saved project file"
```

### Getting Help
```
✅ "What can Geographic Calculator do?"
✅ "How do I create a Geographic Calculator job?"
✅ "Show me available Geographic Calculator tools"
```

---

## Summary

1. **Add `--interactive`** to your command for file operations
2. **Wait** for the permission request prompt
3. **Type `yes`** (or press Enter) to grant permission
4. **Type `no`** to deny permission
5. **Type `exit`** to end the conversation

**Remember:**
- Automatic checks (software availability, paths) don't need permission
- File access and modifications require permission
- Be specific in your queries for best results
- Grant permission when you understand and trust the operation

The agent is designed to be helpful, transparent, and respectful of your privacy!

