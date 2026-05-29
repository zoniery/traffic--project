#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENV_PYTHON = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

COMMANDS = {
    "check": ["src/01_check_gpu.py"],
    "track": ["src/02_track_video.py"],
    "count": ["src/03_count_line.py"],
    "status": ["src/04_traffic_status.py"],
    "charts": ["src/05_export_charts.py"],
}


def ensure_venv() -> None:
    if VENV_PYTHON.exists():
        return
    print(
        "Project virtualenv was not found. Create it first:\n"
        "  python3 -m venv .venv\n"
        "  ./.venv/bin/python -m pip install -U pip\n"
        "  ./.venv/bin/python -m pip install streamlit ultralytics supervision opencv-python pandas matplotlib torch torchvision torchaudio",
        file=sys.stderr,
    )
    raise SystemExit(1)


def run_python(args: list[str]) -> int:
    ensure_venv()
    return subprocess.run([str(VENV_PYTHON), *args], cwd=ROOT).returncode


def run_script(script: str, extra_args: list[str]) -> int:
    return run_python([script, *extra_args])


def run_test_chain() -> int:
    for name in ("check", "track", "count", "status", "charts"):
        print(f"\n== {name}: {COMMANDS[name][0]} ==")
        code = run_script(COMMANDS[name][0], [])
        if code != 0:
            return code
    return 0


def print_usage() -> None:
    commands = ", ".join([*COMMANDS, "app", "test", "python"])
    print(f"Usage: python3 run.py <command> [args]\nCommands: {commands}")


def main(argv: list[str]) -> int:
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print_usage()
        return 0

    command, *extra_args = argv
    if command in COMMANDS:
        return run_script(COMMANDS[command][0], extra_args)
    if command == "test":
        return run_test_chain()
    if command == "app":
        return run_python(["-m", "streamlit", "run", "src/app.py", *extra_args])
    if command == "python":
        return run_python(extra_args)

    script_path = ROOT / command
    if script_path.is_file() and script_path.suffix == ".py":
        return run_script(command, extra_args)

    print(f"Unknown command: {command}", file=sys.stderr)
    print_usage()
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
