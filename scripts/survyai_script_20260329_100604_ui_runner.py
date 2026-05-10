# SurvyAI ArcGIS Pro UI runner
import json, traceback
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

_status_path = r"C:\Users\UZOR\PycharmProjects\untitled\venv\SurvyAI\scripts\survyai_script_20260329_100604_ui_status.json"
_log_path = r"C:\Users\UZOR\PycharmProjects\untitled\venv\SurvyAI\scripts\survyai_script_20260329_100604_ui_log.txt"
_target_path = r"C:\Users\UZOR\PycharmProjects\untitled\venv\SurvyAI\scripts\survyai_script_20260329_100604.py"

def _write_status(payload):
    with open(_status_path, "w", encoding="utf-8") as _f:
        json.dump(payload, _f, ensure_ascii=True, indent=2)

_buf = StringIO()
_write_status({"state": "running", "target_script": _target_path})

try:
    with redirect_stdout(_buf), redirect_stderr(_buf):
        _g = {"__name__": "__main__"}
        exec(compile(open(_target_path, "r", encoding="utf-8").read(), _target_path, "exec"), _g, _g)
    _log = _buf.getvalue()
    with open(_log_path, "w", encoding="utf-8") as _lf:
        _lf.write(_log)
    _write_status({"state": "completed", "success": True, "log_path": _log_path})
except Exception:
    _err = traceback.format_exc()
    _log = _buf.getvalue()
    with open(_log_path, "w", encoding="utf-8") as _lf:
        _lf.write(_log)
        if _log and not _log.endswith("\n"):
            _lf.write("\n")
        _lf.write(_err)
    _write_status({
        "state": "completed",
        "success": False,
        "error": _err,
        "log_path": _log_path
    })
