"""macOS Keychain-backed application credentials."""

from __future__ import annotations

import subprocess


SECURITY = "/usr/bin/security"


def _run_security(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [SECURITY, *arguments],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def load_secret(service: str, account: str) -> str:
    result = _run_security(
        ["find-generic-password", "-s", service, "-a", account, "-w"]
    )
    if result.returncode == 0:
        return result.stdout.strip()
    if result.returncode == 44 or "could not be found" in result.stderr.lower():
        return ""
    raise RuntimeError(result.stderr.strip() or "Keychain read failed")


def save_secret(service: str, account: str, value: str) -> None:
    if not value:
        raise ValueError("Secret value must not be empty")
    result = _run_security(
        [
            "add-generic-password",
            "-U",
            "-s",
            service,
            "-a",
            account,
            "-w",
            value,
        ]
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Keychain write failed")


def delete_secret(service: str, account: str) -> None:
    result = _run_security(
        ["delete-generic-password", "-s", service, "-a", account]
    )
    if result.returncode == 0:
        return
    if result.returncode == 44 or "could not be found" in result.stderr.lower():
        return
    raise RuntimeError(result.stderr.strip() or "Keychain delete failed")
