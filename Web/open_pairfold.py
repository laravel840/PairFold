#!/usr/bin/env python3
"""One-click PairFold web app launcher (lives in Web/).

- If the server is already up → open the browser immediately.
- Otherwise start it in the background, wait until the UI responds, then open.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import Optional

# Repo root is the parent of Web/
ROOT = Path(__file__).resolve().parent.parent
URL = "http://127.0.0.1:8000/"
HOST = "127.0.0.1"
PORT = 8000
READY_TIMEOUT_S = 90.0


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.4):
            return True
    except OSError:
        return False


def _ui_ready() -> bool:
    try:
        with urllib.request.urlopen(URL, timeout=1.5) as res:
            return 200 <= res.status < 500
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _source_newer_than_dist() -> bool:
    """True if frontend sources changed after the last production build."""
    dist_index = ROOT / "dist" / "index.html"
    if not dist_index.is_file():
        return True
    dist_mtime = dist_index.stat().st_mtime
    watch = [
        ROOT / "index.html",
        ROOT / "viewer.html",
        ROOT / "vite.config.js",
        ROOT / "package.json",
        ROOT / "src",
    ]
    for path in watch:
        if not path.exists():
            continue
        if path.is_file():
            if path.stat().st_mtime > dist_mtime:
                return True
            continue
        for child in path.rglob("*"):
            if child.is_file() and child.stat().st_mtime > dist_mtime:
                return True
    return False


def _ensure_dist() -> None:
    npm = "npm.cmd" if os.name == "nt" else "npm"
    need_build = _source_newer_than_dist()
    if not need_build:
        return
    if not (ROOT / "node_modules").is_dir():
        subprocess.check_call([npm, "install"], cwd=str(ROOT))
    subprocess.check_call([npm, "run", "build"], cwd=str(ROOT))
    if not (ROOT / "dist" / "index.html").is_file():
        raise RuntimeError("Frontend build failed — dist/index.html missing.")


def _start_server() -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    kwargs = {
        "cwd": str(ROOT),
        "env": env,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)

    return subprocess.Popen(
        [sys.executable, "-m", "pairfold.server", "--no-browser", "--host", HOST, "--port", str(PORT)],
        **kwargs,
    )


def _wait_ready(proc: Optional[subprocess.Popen]) -> None:
    deadline = time.time() + READY_TIMEOUT_S
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(
                f"PairFold server exited early (code {proc.returncode}). "
                "Run: python -m pairfold.server"
            )
        if _ui_ready():
            return
        time.sleep(0.25)
    raise RuntimeError(f"Timed out waiting for {URL}")


def main() -> int:
    os.chdir(ROOT)
    try:
        _ensure_dist()
    except Exception as exc:  # noqa: BLE001 — show plain message to end user
        _fail(f"Could not prepare the web UI.\n{exc}")
        return 1

    already = _port_open(HOST, PORT)
    proc = None
    if already:
        if not _ui_ready():
            _fail(
                f"Port {PORT} is busy but PairFold UI is not responding.\n"
                "Close the other app, then try again."
            )
            return 1
    else:
        try:
            proc = _start_server()
        except Exception as exc:  # noqa: BLE001
            _fail(f"Could not start PairFold server.\n{exc}")
            return 1
        try:
            _wait_ready(proc)
        except Exception as exc:  # noqa: BLE001
            _fail(str(exc))
            return 1

    # Cache-bust so the browser does not keep a stale UI shell
    webbrowser.open(f"{URL}?t={int(time.time())}")
    return 0


def _fail(message: str) -> None:
    if os.name == "nt":
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(0, message, "PairFold", 0x10)
            return
        except Exception:
            pass
    sys.stderr.write(message + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
