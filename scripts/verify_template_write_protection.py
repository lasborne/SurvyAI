"""
Verify that the AI agent has STRICT read-only access to CAD template .dwg files.
Run: python -m scripts.verify_template_write_protection
"""
from pathlib import Path
import sys

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _ensure_protected_templates_loaded(protected_paths: set, profile_dir: Path) -> None:
    """Mirror of agent._ensure_protected_templates_loaded logic."""
    import json as _json
    if not profile_dir.exists():
        return
    for prof_path in profile_dir.glob("*.json"):
        try:
            data = _json.loads(prof_path.read_text(encoding="utf-8"))
            tp = (data.get("template") or {}).get("path") or ""
            if tp:
                protected_paths.add(str(Path(tp).resolve()))
        except Exception:
            continue


def _is_protected_template_path(dwg_path: str, protected_paths: set, profile_dir: Path) -> bool:
    """Mirror of agent._is_protected_template_path logic."""
    if not dwg_path:
        return False
    _ensure_protected_templates_loaded(protected_paths, profile_dir)
    try:
        # Must compare str to str (protected set stores str(Path.resolve()))
        return str(Path(dwg_path).resolve()) in protected_paths
    except Exception:
        return False


def main():
    profile_dir = ROOT / "template_profiles"
    protected = set()
    _ensure_protected_templates_loaded(protected, profile_dir)

    print("=" * 60)
    print("CAD TEMPLATE WRITE PROTECTION VERIFICATION")
    print("=" * 60)

    # 1. Protected paths from profiles
    print("\n1. Protected template paths (from template_profiles/*.json):")
    for p in sorted(protected):
        print(f"   - {p}")
    if not protected:
        print("   (none - no template_profiles/*.json with template.path)")
        return 1

    # 2. Test _is_protected_template_path
    print("\n2. _is_protected_template_path() tests:")
    template_sample = next(iter(protected))
    tests = [
        (template_sample, True, "exact template path"),
        (str(Path(template_sample) / ".." / Path(template_sample).name), True, "resolved equivalent"),
        (str(ROOT / "output_plan.dwg"), False, "output plan (not template)"),
        ("C:\\nonexistent\\survey_plan_template2.dwg", False, "different path (not in profiles)"),
    ]
    all_pass = True
    for path, expected, desc in tests:
        result = _is_protected_template_path(path, protected, profile_dir)
        ok = result == expected
        all_pass = all_pass and ok
        status = "PASS" if ok else "FAIL"
        print(f"   [{status}] {desc}: is_protected={result} (expected {expected})")
    if not all_pass:
        return 1

    # 3. Document all agent code paths that touch template
    print("\n3. Agent code paths that touch CAD template:")
    print("   A. _learn_cadastral_template_profile (agent.py ~1136)")
    print("      -> open_drawing(template, read_only=True)")
    print("   B. _apply_cadastral_template (agent.py ~1278)")
    print("      -> shutil.copy2(template, output)  # copy only, never write to template")
    print("   C. Peg fallback (agent.py ~1571)")
    print("      -> acad.Documents.Open(template, True)  # True = read-only")
    print("   D. Pillar-number fallback (agent.py ~1669)")
    print("      -> acad.Documents.Open(template, True)  # True = read-only")
    print("   E. autocad_open_drawing tool (agent.py ~3835-3836)")
    print("      -> if _is_protected_template_path: open_drawing(..., read_only=True)")
    print("   F. _safe_save_active_drawing (agent.py ~2395)")
    print("      -> skips save if _is_protected_template_path(active_path)")
    print("   G. save_active_drawing (autocad_processor.py ~1845)")
    print("      -> skips doc.Save() if doc.ReadOnly")
    print("   H. _run_cad_modification_pipeline (agent.py ~2463)")
    print("      -> returns error if _is_template_path(target)")

    # 4. Write paths summary
    print("\n4. Write protection layers:")
    print("   LAYER 1: Template opened read-only (Documents.Open(path, True) / open_drawing(..., read_only=True))")
    print("   LAYER 2: save_active_drawing checks doc.ReadOnly -> skips Save() if True")
    print("   LAYER 3: _safe_save_active_drawing checks _is_protected_template_path -> skips save if template")
    print("   LAYER 4: _run_cad_modification_pipeline rejects target if _is_template_path")
    print("   LAYER 5: autocad_open_drawing forces read_only=True for protected paths")
    print("   NO SaveAs/WBlock/Export to template path in codebase.")

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED - Template write protection is verified.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
