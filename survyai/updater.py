from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import requests

from runtime_paths import user_data_path

# Default interval between automatic (consent-gated) update checks.
UPDATE_CHECK_INTERVAL_HOURS = 12


def _version_tuple(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for part in str(value or "").strip().split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        parts.append(int(digits or 0))
    return tuple(parts or [0])


def update_check_due(
    last_check_iso: str,
    *,
    interval_hours: float = UPDATE_CHECK_INTERVAL_HOURS,
) -> bool:
    """Return True when no prior check exists or the interval has elapsed."""
    raw = str(last_check_iso or "").strip()
    if not raw:
        return True
    try:
        last = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
    except Exception:
        return True
    return datetime.now(timezone.utc) >= last + timedelta(hours=float(interval_hours))


def launch_staged_installer(installer_path: Path) -> None:
    """Open a staged Windows installer with the OS shell."""
    target = Path(installer_path)
    if not target.is_file():
        raise FileNotFoundError(f"Staged installer not found: {target}")
    if os.name == "nt":
        os.startfile(str(target))  # type: ignore[attr-defined]
        return
    raise RuntimeError("Automatic installer launch is only supported on Windows.")


@dataclass(frozen=True)
class UpdateManifest:
    channel: str
    platform: str
    latest_version: str
    min_supported_version: Optional[str] = None
    download_url: Optional[str] = None
    sha256: Optional[str] = None
    artifact_kind: str = "full-installer"
    signature: Optional[str] = None
    signing_scheme: Optional[str] = None
    rollback_version: Optional[str] = None
    release_notes_url: Optional[str] = None
    mandatory: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UpdateManifest":
        return cls(
            channel=str(data.get("channel") or "stable"),
            platform=str(data.get("platform") or "windows-x64"),
            latest_version=str(data.get("latest_version") or "0.0.0"),
            min_supported_version=_opt_str(data.get("min_supported_version")),
            download_url=_opt_str(data.get("download_url")),
            sha256=_opt_str(data.get("sha256")),
            artifact_kind=str(data.get("artifact_kind") or "full-installer"),
            signature=_opt_str(data.get("signature")),
            signing_scheme=_opt_str(data.get("signing_scheme")),
            rollback_version=_opt_str(data.get("rollback_version")),
            release_notes_url=_opt_str(data.get("release_notes_url")),
            mandatory=bool(data.get("mandatory")),
        )

    def is_newer_than(self, current_version: str) -> bool:
        return _version_tuple(self.latest_version) > _version_tuple(current_version)

    def requires_upgrade_from(self, current_version: str) -> bool:
        if not self.min_supported_version:
            return False
        return _version_tuple(current_version) < _version_tuple(self.min_supported_version)

    def is_required_for(self, current_version: str) -> bool:
        """True when the update is mandatory or below the supported floor."""
        return bool(self.mandatory) or self.requires_upgrade_from(current_version)


def _opt_str(value: Any) -> Optional[str]:
    raw = str(value or "").strip()
    return raw or None


class UpdateManager:
    def __init__(self, app_dir: Optional[Path] = None) -> None:
        base = Path(app_dir) if app_dir is not None else user_data_path("updates")
        self.base_dir = base
        self.download_dir = base / "downloads"
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def stage_update(
        self,
        manifest: UpdateManifest,
        *,
        current_version: str,
        current_executable: str,
        progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
    ) -> Path:
        if not manifest.download_url or not manifest.sha256:
            raise RuntimeError("Update manifest is missing download_url or sha256.")
        target = self.download_dir / self._download_filename(manifest)
        self._download_with_sha256(
            manifest.download_url,
            target,
            manifest.sha256,
            progress_callback=progress_callback,
        )
        self._verify_signature_if_configured(target, manifest)
        rollback_plan = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "current_version": current_version,
            "target_version": manifest.latest_version,
            "rollback_version": manifest.rollback_version or current_version,
            "current_executable": current_executable,
            "staged_installer": str(target),
            "artifact_kind": manifest.artifact_kind,
            "manual_rollback_required": True,
            "rollback_instructions": (
                "Keep the previous installer/build artifact available. "
                "If the staged update fails, reinstall the rollback_version package."
            ),
        }
        (self.base_dir / "rollback_plan.json").write_text(
            json.dumps(rollback_plan, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        (self.base_dir / "last_manifest.json").write_text(
            json.dumps(manifest.__dict__, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        return target

    def _download_filename(self, manifest: UpdateManifest) -> str:
        suffix = Path((manifest.download_url or "").split("?")[0]).suffix or ".bin"
        return f"SurvyAI-{manifest.latest_version}-{manifest.platform}{suffix}"

    def _download_with_sha256(
        self,
        url: str,
        target: Path,
        expected_sha256: str,
        *,
        progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
    ) -> None:
        digest = hashlib.sha256()
        downloaded = 0
        with requests.get(url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            total_header = resp.headers.get("Content-Length")
            total: Optional[int]
            try:
                total = int(total_header) if total_header else None
            except (TypeError, ValueError):
                total = None
            with target.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=1024 * 256):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    digest.update(chunk)
                    downloaded += len(chunk)
                    if progress_callback is not None:
                        progress_callback(downloaded, total)
        actual = digest.hexdigest().lower()
        if actual != expected_sha256.strip().lower():
            target.unlink(missing_ok=True)
            raise RuntimeError(
                f"Downloaded update hash mismatch. Expected {expected_sha256}, got {actual}."
            )

    def _verify_signature_if_configured(self, target: Path, manifest: UpdateManifest) -> None:
        scheme = str(manifest.signing_scheme or "").strip().lower()
        if not scheme:
            return
        if scheme == "authenticode":
            self._verify_windows_authenticode(target, expected_identity=manifest.signature)
            return
        raise RuntimeError(f"Unsupported update signing scheme: {manifest.signing_scheme}")

    def _verify_windows_authenticode(self, target: Path, expected_identity: Optional[str]) -> None:
        if os.name != "nt":
            raise RuntimeError("Authenticode verification is only supported on Windows desktops.")
        quoted = str(target).replace("'", "''")
        script = (
            "$sig = Get-AuthenticodeSignature -FilePath '" + quoted + "'; "
            "$thumb = ''; "
            "if ($sig.SignerCertificate) { $thumb = $sig.SignerCertificate.Thumbprint }; "
            "Write-Output ($sig.Status.ToString()); "
            "Write-Output $thumb"
        )
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Authenticode verification failed to execute: {(proc.stderr or proc.stdout).strip()}"
            )
        lines = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]
        status = lines[0] if lines else ""
        thumbprint = lines[1] if len(lines) > 1 else ""
        if status.lower() != "valid":
            raise RuntimeError(
                f"Downloaded installer is not Authenticode-valid (status={status or 'unknown'})."
            )
        expected = str(expected_identity or "").strip().lower().replace(" ", "")
        if expected and expected not in {thumbprint.lower().replace(" ", ""), ""}:
            raise RuntimeError(
                "Downloaded installer signature thumbprint does not match the manifest identity."
            )


__all__ = [
    "UPDATE_CHECK_INTERVAL_HOURS",
    "UpdateManager",
    "UpdateManifest",
    "launch_staged_installer",
    "update_check_due",
]
