"""Tests for the colour palette.

No Tk needed: these cover the palette values and lookup helpers. The tests that
check real windows actually pick the colours up live in test_gui.py.
"""

from __future__ import annotations

import dataclasses
import re

import pytest

from tpsreg.theme import (
    DARK,
    DEFAULT_PALETTE,
    LIGHT,
    PALETTES,
    apply_to_window,
    get_palette,
    palette_of,
)

HEX_COLOUR = re.compile(r"^#[0-9a-fA-F]{6}$")

COLOUR_FIELDS = (
    "background",
    "foreground",
    "accent",
    "success",
    "canvas",
    "muted_foreground",
    "warning",
)


class TestPalettes:
    """The shipped palettes."""

    @pytest.mark.parametrize("palette", [DARK, LIGHT])
    @pytest.mark.parametrize("field", COLOUR_FIELDS)
    def test_every_colour_is_a_hex_string(self, palette, field):
        value = getattr(palette, field)
        assert HEX_COLOUR.match(value), f"{palette.name}.{field} = {value!r}"

    @pytest.mark.parametrize("palette", [DARK, LIGHT])
    def test_every_field_is_populated(self, palette):
        for field in COLOUR_FIELDS:
            assert getattr(palette, field)

    def test_dark_and_light_actually_differ(self):
        assert DARK.background != LIGHT.background
        assert DARK.foreground != LIGHT.foreground

    def test_ttk_theme_name_matches_the_packaged_theme(self):
        """The name has to line up with the .tcl files that ship in the package."""
        assert DARK.ttk_theme == "azure-dark"
        assert LIGHT.ttk_theme == "azure-light"

    @pytest.mark.parametrize("palette", [DARK, LIGHT])
    def test_text_is_not_the_same_colour_as_its_background(self, palette):
        assert palette.foreground != palette.background
        assert palette.muted_foreground != palette.background

    @pytest.mark.parametrize("palette", [DARK, LIGHT])
    def test_muted_text_is_between_background_and_foreground(self, palette):
        """Muted text must be dimmer than body text but still readable.

        Plain "gray" failed this on the dark theme, which is why hints were
        hard to read.
        """

        def luminance(colour: str) -> float:
            r, g, b = (int(colour[i : i + 2], 16) for i in (1, 3, 5))
            return 0.2126 * r + 0.7152 * g + 0.0722 * b

        background = luminance(palette.background)
        foreground = luminance(palette.foreground)
        muted = luminance(palette.muted_foreground)

        low, high = sorted((background, foreground))
        assert low < muted < high, (
            f"{palette.name}: muted text ({muted:.0f}) is not between the "
            f"background ({background:.0f}) and foreground ({foreground:.0f})"
        )

    @pytest.mark.parametrize("palette", [DARK, LIGHT])
    def test_muted_text_has_usable_contrast(self, palette):
        """A hint nobody can read is not a hint."""

        def channels(colour: str) -> tuple[float, ...]:
            return tuple(int(colour[i : i + 2], 16) / 255 for i in (1, 3, 5))

        def relative_luminance(colour: str) -> float:
            def linear(c: float) -> float:
                return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

            r, g, b = (linear(c) for c in channels(colour))
            return 0.2126 * r + 0.7152 * g + 0.0722 * b

        lighter, darker = sorted(
            (
                relative_luminance(palette.muted_foreground),
                relative_luminance(palette.background),
            ),
            reverse=True,
        )
        ratio = (lighter + 0.05) / (darker + 0.05)

        # WCAG AA for large text. Hints are secondary, so this is the floor
        # rather than the 4.5 required for body text.
        assert ratio >= 3.0, f"{palette.name}: contrast ratio only {ratio:.2f}"

    def test_palettes_are_immutable(self):
        """A popup must not be able to recolour the shared palette."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            DARK.background = "#ff0000"

    @pytest.mark.parametrize("palette", [DARK, LIGHT])
    def test_warning_is_distinguishable_from_success(self, palette):
        """A flagged point must not read as a healthy one."""
        assert palette.warning != palette.success

    @pytest.mark.parametrize("palette", [DARK, LIGHT])
    def test_warning_stands_out_against_the_canvas(self, palette):
        """It is drawn on the image backdrop, not the window background."""
        assert palette.warning != palette.canvas

    def test_registry_is_complete(self):
        assert set(PALETTES) == {"dark", "light"}
        assert PALETTES["dark"] is DARK
        assert PALETTES["light"] is LIGHT

    def test_default_is_one_of_the_registered_palettes(self):
        assert DEFAULT_PALETTE in PALETTES.values()


class TestGetPalette:
    """Lookup by name."""

    @pytest.mark.parametrize("name", ["dark", "light"])
    def test_known_names(self, name):
        assert get_palette(name).name == name

    def test_unknown_name_lists_the_options(self):
        with pytest.raises(ValueError, match="Unknown theme"):
            get_palette("solarized")


class TestPaletteOf:
    """Finding the palette a widget should use."""

    def test_reads_it_from_the_widget(self):
        class Widget:
            palette = LIGHT

        assert palette_of(Widget()) is LIGHT

    def test_falls_back_to_the_parent(self):
        """Popups are handed their parent so they can match it."""

        class Parent:
            palette = LIGHT

        class Popup:
            master = Parent()

        assert palette_of(Popup()) is LIGHT

    def test_defaults_when_nothing_declares_one(self):
        class Bare:
            pass

        assert palette_of(Bare()) is DEFAULT_PALETTE

    def test_ignores_a_non_palette_attribute(self):
        class Confusing:
            palette = "dark"  # a string, not a Palette

        assert palette_of(Confusing()) is DEFAULT_PALETTE

    def test_self_referential_master_does_not_recurse(self):
        class Loop:
            pass

        widget = Loop()
        widget.master = widget
        assert palette_of(widget) is DEFAULT_PALETTE


class TestApplyToWindow:
    """Setting a window background."""

    def test_sets_the_background(self):
        class FakeWindow:
            def __init__(self):
                self.kwargs = {}

            def configure(self, **kwargs):
                self.kwargs.update(kwargs)

        window = FakeWindow()
        apply_to_window(window, DARK)
        assert window.kwargs["background"] == DARK.background

    def test_a_window_that_refuses_is_not_fatal(self):
        """Cosmetics must never stop a window opening."""

        class Awkward:
            def configure(self, **kwargs):
                raise RuntimeError("no")

        apply_to_window(Awkward(), DARK)


class TestNoStrayColours:
    """Colours belong in this module, not scattered through the view."""

    def test_gui_module_defines_no_hex_colours(self):
        from pathlib import Path

        gui = Path(__file__).resolve().parents[1] / "src" / "tpsreg" / "GUI.py"
        source = gui.read_text(encoding="utf-8")

        strays = sorted(set(re.findall(r"#[0-9a-fA-F]{6}\b", source)))
        assert not strays, (
            f"GUI.py hardcodes colours {strays}; add them to tpsreg.theme so "
            "every window stays consistent"
        )

    def test_gui_module_uses_no_named_colours_for_widgets(self):
        """'gray' in three popups is what made them not match the main window."""
        from pathlib import Path

        gui = Path(__file__).resolve().parents[1] / "src" / "tpsreg" / "GUI.py"
        source = gui.read_text(encoding="utf-8")

        strays = re.findall(
            r"(?:bg|fg|background|foreground)\s*=\s*[\"'](\w+)[\"']", source
        )
        assert not strays, (
            f"GUI.py sets widget colours by name {sorted(set(strays))}; use the "
            "palette instead"
        )
