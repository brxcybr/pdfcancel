"""install.sh records what it installed in the profile the wrapper reads."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

INSTALL = Path(__file__).resolve().parents[1] / "skills" / "pdfcancel" / "install.sh"


def _install(tmp_path: Path, *args: str) -> tuple[subprocess.CompletedProcess, Path]:
    home = tmp_path / "home"
    home.mkdir()
    env = {"HOME": str(home), "PATH": os.environ["PATH"]}
    proc = subprocess.run(["bash", str(INSTALL), "--no-venv", *args],
                          env=env, capture_output=True, text=True, timeout=60)
    return proc, home / ".config" / "pdfcancel" / "skill.env"


def _sourced(profile: Path, var: str, home: Path) -> str:
    out = subprocess.run(["bash", "-c", f'set -a; . "$1"; printf %s "${{{var}:-}}"', "_",
                          str(profile)], env={"HOME": str(home)}, capture_output=True,
                         text=True, check=True)
    return out.stdout


def test_venv_and_ca_are_written_to_the_profile(tmp_path):
    sender_ca = tmp_path / "builder-ca.pem"
    sender_ca.write_text("old\n")
    profile = tmp_path / "team.env"
    profile.write_text(
        'PDFCANCEL_VENV="$HOME/.local/share/pdfcancel/venv"\n'
        "PDFCANCEL_API_BASE=http://127.0.0.1:9/v1\n"
        "PDFCANCEL_VLM_MODEL=m\n"
        f"export PDFCANCEL_CA_BUNDLE={sender_ca}\n"
    )
    ca = tmp_path / "root.crt"
    ca.write_text("-----BEGIN CERTIFICATE-----\nx\n-----END CERTIFICATE-----\n")
    venv = tmp_path / "my venv's dir"

    proc, installed = _install(tmp_path, "--profile", str(profile), "--ca", str(ca),
                               "--venv", str(venv))
    home = tmp_path / "home"
    assert installed.exists(), proc.stderr
    assert _sourced(installed, "PDFCANCEL_VENV", home) == str(venv)
    want_ca = home / ".config" / "pdfcancel" / "skill-ca.pem"
    assert _sourced(installed, "PDFCANCEL_CA_BUNDLE", home) == str(want_ca)
    assert want_ca.read_text() == ca.read_text()
    assert installed.read_text().count("PDFCANCEL_CA_BUNDLE") == 1
    assert _sourced(installed, "PDFCANCEL_VLM_MODEL", home) == "m"  # rest kept
