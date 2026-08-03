"""Locating packaged resources (Tk theme files, application icon).

Resources live inside the installed package rather than beside the repository
checkout, so the GUI works identically from a wheel, an editable install and a
source tree.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

#: Root of the packaged resource tree.
RESOURCES_PATH = Path(__file__).parent / "resources"

#: Application icon. Only usable as a window icon on Windows.
ICON_PATH = RESOURCES_PATH / "EBSD-Correction.ico"


def theme_path(style: str) -> Path:
    """Return the path to a Tk theme definition.

    Parameters
    ----------
    style:
        Either ``"dark"`` or ``"light"``.

    Raises
    ------
    ValueError
        If ``style`` is not a known theme.
    FileNotFoundError
        If the packaged theme file is missing.
    """
    if style not in ("dark", "light"):
        raise ValueError(f"Unknown theme style: {style!r}. Expected 'dark' or 'light'.")

    path = RESOURCES_PATH / "theme" / f"{style}.tcl"
    if not path.exists():
        raise FileNotFoundError(
            f"Packaged theme file not found: {path}. The tpsreg installation "
            "appears to be incomplete."
        )
    return path


def apply_window_icon(window) -> None:
    """Set the application icon on a Tk window, where the platform supports it.

    ``iconbitmap`` only accepts ``.ico`` files on Windows; elsewhere it raises
    a ``TclError``. A missing icon is cosmetic, so failures are logged and
    swallowed rather than blocking startup.
    """
    if sys.platform != "win32":
        logger.debug("Window icon skipped: .ico is Windows-only")
        return

    try:
        window.iconbitmap(str(ICON_PATH))
    except Exception:  # pragma: no cover - depends on the Tk build
        logger.debug("Could not set window icon", exc_info=True)
