"""
Restore survey_plan_template2.dwg from the last saved backup.

AutoCAD creates a .bak file when saving. This script copies
survey_plan_template2.bak -> survey_plan_template2.dwg so the template
is restored to the last saved state (e.g. after accidental edits).

Usage:
  python -m scripts.restore_survey_plan_template2
  # or from project root:
  python scripts/restore_survey_plan_template2.py

Ensure AutoCAD does not have the template open when restoring.
"""
from pathlib import Path
import shutil

def main() -> None:
    root = Path(__file__).resolve().parent.parent
    bak = root / "survey_plan_template2.bak"
    dwg = root / "survey_plan_template2.dwg"
    if not bak.exists():
        print(f"Backup not found: {bak}")
        return
    shutil.copy2(bak, dwg)
    print(f"Restored: {dwg} from {bak}")

if __name__ == "__main__":
    main()
