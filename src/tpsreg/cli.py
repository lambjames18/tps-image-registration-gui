"""Console entry point for the ``tpsreg`` command.

Kept separate from :mod:`tpsreg.GUI` so that a missing Tk installation is
reported as an actionable message rather than an import traceback. ``import
tkinter`` fails at module import time, which is before any error handling
inside ``GUI.main`` could run.
"""

from __future__ import annotations

import sys

_TK_HELP = {
    "linux": (
        "Install Tk with your system package manager, for example:\n"
        "    sudo apt install python3-tk        # Debian, Ubuntu\n"
        "    sudo dnf install python3-tkinter   # Fedora, RHEL\n"
        "    sudo pacman -S tk                  # Arch"
    ),
    "darwin": (
        "Install a Python build that includes Tk, for example:\n"
        "    brew install python-tk"
    ),
    "win32": (
        "Reinstall Python from python.org and make sure the\n"
        "'tcl/tk and IDLE' component is selected."
    ),
}


def _tk_platform_help() -> str:
    """Return platform-appropriate advice for installing Tk."""
    if sys.platform.startswith("linux"):
        return _TK_HELP["linux"]
    return _TK_HELP.get(sys.platform, _TK_HELP["linux"])


def main() -> None:
    """Launch the GUI, explaining clearly if Tk is unavailable."""
    try:
        import tkinter  # noqa: F401
    except ImportError:
        print(
            "tpsreg needs Tk, which is not available in this Python "
            "installation.\n\n" + _tk_platform_help(),
            file=sys.stderr,
        )
        raise SystemExit(1) from None

    from tpsreg.GUI import main as gui_main

    gui_main()


if __name__ == "__main__":
    main()
