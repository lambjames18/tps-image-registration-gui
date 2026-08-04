"""The application colour palette.

Every colour the GUI draws lives here. Popup windows are separate ``Toplevel``
widgets, and ttk styling only reaches ttk widgets, so plain Tk widgets
(canvases, and the windows themselves) have to be coloured explicitly. Keeping
the values in one place is what stops those windows drifting apart from the
main one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Palette:
    """Colours for one theme.

    Attributes
    ----------
    name:
        Theme name, matching the packaged ttk theme suffix ("dark"/"light").
    background:
        Window and frame background.
    foreground:
        Primary text.
    accent:
        Focus rings and highlights.
    success:
        Positive emphasis, such as matched-point markers.
    canvas:
        Backdrop behind displayed images. Distinct from ``background`` only if
        a theme wants the image area to stand out.
    muted_foreground:
        Secondary text such as hints and tooltips. Chosen to stay legible
        against ``background``, which plain "gray" is not on a dark theme.
    """

    name: str
    background: str
    foreground: str
    accent: str
    success: str
    canvas: str
    muted_foreground: str

    @property
    def ttk_theme(self) -> str:
        """Name of the packaged ttk theme that matches this palette."""
        return f"azure-{self.name}"


DARK = Palette(
    name="dark",
    background="#333333",
    foreground="#ffffff",
    accent="#229fff",
    success="#00bb00",
    canvas="#333333",
    muted_foreground="#9a9a9a",
)

LIGHT = Palette(
    name="light",
    background="#ffffff",
    foreground="#000000",
    accent="#007fff",
    success="#00bb00",
    canvas="#ffffff",
    muted_foreground="#666666",
)

PALETTES: dict[str, Palette] = {DARK.name: DARK, LIGHT.name: LIGHT}

#: Theme used when nothing else is specified.
DEFAULT_PALETTE = DARK


def get_palette(name: str) -> Palette:
    """Look up a palette by theme name.

    Raises
    ------
    ValueError
        If the name is not a known theme.
    """
    try:
        return PALETTES[name]
    except KeyError:
        raise ValueError(
            f"Unknown theme: {name!r}. Expected one of {sorted(PALETTES)}."
        ) from None


def palette_of(widget: object) -> Palette:
    """Return the palette a widget was built with.

    Popup windows are handed their parent so they can match it. A parent that
    predates this module, or a bare Tk root in a test, simply gets the default
    rather than an error.
    """
    palette = getattr(widget, "palette", None)
    if isinstance(palette, Palette):
        return palette

    master = getattr(widget, "master", None)
    if master is not None and master is not widget:
        parent_palette = getattr(master, "palette", None)
        if isinstance(parent_palette, Palette):
            return parent_palette

    logger.debug(
        "No palette on %r; using %s", type(widget).__name__, DEFAULT_PALETTE.name
    )
    return DEFAULT_PALETTE


def apply_to_window(window: object, palette: Palette) -> None:
    """Set a window's own background.

    ttk styling never reaches the ``Tk``/``Toplevel`` itself, so without this
    the frame around themed widgets keeps the platform default, which is what
    made the popups look out of place.
    """
    try:
        window.configure(background=palette.background)
    except Exception:  # pragma: no cover - depends on the Tk build
        logger.debug("Could not set window background", exc_info=True)
