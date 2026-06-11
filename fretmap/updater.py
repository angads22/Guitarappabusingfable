"""Self-update support: check GitHub Releases for a newer build and swap
the running executable for the downloaded one.

Uses only the standard library so it adds nothing to the bundle. All
network calls are best-effort: any failure is reported as "no update
information", never an exception to the caller.
"""

import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from fretmap import __version__

REPO = "angads22/Guitarappabusingfable"
LATEST_RELEASE_API = f"https://api.github.com/repos/{REPO}/releases/latest"
RELEASES_PAGE = f"https://github.com/{REPO}/releases/latest"
_UA = {"User-Agent": f"fretmap/{__version__}"}


@dataclass
class UpdateInfo:
    version: str          # e.g. "0.2.0"
    tag: str              # e.g. "v0.2.0"
    asset_name: str
    download_url: str
    release_url: str


def parse_version(s: str) -> tuple[int, ...]:
    s = s.strip().lstrip("vV")
    parts = []
    for chunk in s.split("."):
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts) or (0,)


def is_newer(remote: str, local: str) -> bool:
    return parse_version(remote) > parse_version(local)


def platform_key() -> str:
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


def asset_name(gui: bool = True) -> str:
    base = "fretmap-gui" if gui else "fretmap"
    suffix = ".exe" if platform_key() == "windows" else ""
    return f"{base}-{platform_key()}{suffix}"


def fetch_latest_release(timeout: float = 10.0) -> dict | None:
    try:
        req = urllib.request.Request(LATEST_RELEASE_API, headers=_UA)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except Exception:
        return None


def find_update(release: dict | None, gui: bool = True, current: str = __version__) -> UpdateInfo | None:
    """Extract an applicable update from a release API payload, if any."""
    if not release:
        return None
    tag = release.get("tag_name") or ""
    if not is_newer(tag, current):
        return None
    wanted = asset_name(gui)
    url = ""
    for asset in release.get("assets", []):
        if asset.get("name") == wanted:
            url = asset.get("browser_download_url", "")
            break
    if not url:
        return None
    return UpdateInfo(
        version=tag.lstrip("vV"),
        tag=tag,
        asset_name=wanted,
        download_url=url,
        release_url=release.get("html_url", RELEASES_PAGE),
    )


def check_for_update(gui: bool = True, current: str = __version__) -> UpdateInfo | None:
    return find_update(fetch_latest_release(), gui=gui, current=current)


def download(url: str, dest: Path, progress=None, timeout: float = 30.0) -> None:
    """Download url to dest; progress(done_bytes, total_bytes) if given."""
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(256 * 1024)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if progress:
                progress(done, total)


def can_self_update() -> bool:
    return bool(getattr(sys, "frozen", False))


def download_update(info: UpdateInfo, progress=None) -> Path:
    dest = Path(tempfile.gettempdir()) / f"{info.asset_name}.new"
    download(info.download_url, dest, progress=progress)
    return dest


def apply_and_restart(new_file: Path) -> None:
    """Replace the running executable with new_file and relaunch.

    On Windows the running exe is locked, so a detached batch script waits
    for this process to exit, swaps the files, and restarts. On POSIX the
    file can be replaced in place and re-exec'd directly.
    """
    if not can_self_update():
        raise RuntimeError("self-update only works in the packaged executable")
    current = Path(sys.executable).resolve()

    if platform_key() == "windows":
        bat = Path(tempfile.gettempdir()) / "fretmap_update.bat"
        bat.write_text(
            "@echo off\r\n"
            ":wait\r\n"
            "ping -n 2 127.0.0.1 >nul\r\n"
            f'del "{current}" 2>nul\r\n'
            f'if exist "{current}" goto wait\r\n'
            f'move /y "{new_file}" "{current}" >nul\r\n'
            f'start "" "{current}"\r\n'
            'del "%~f0"\r\n'
        )
        subprocess.Popen(
            ["cmd", "/c", str(bat)],
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "DETACHED_PROCESS", 0),
            close_fds=True,
        )
        os._exit(0)
    else:
        os.chmod(new_file, 0o755)
        os.replace(new_file, current)
        os.execv(str(current), [str(current)])
