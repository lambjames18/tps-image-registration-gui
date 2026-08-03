"""Smoke tests that construct the real Tk window.

These need a display. On CI they run under Xvfb; locally they run against your
desktop session. They skip cleanly when neither Tk nor a display is available,
so the rest of the suite stays headless.

Deselect them with: pytest -m "not gui"

One root window is created for the whole module and reset between tests. Do not
change this to a function-scoped fixture: repeatedly creating and destroying
tk.Tk() roots in a single process segfaults Aqua Tk on macOS (it crashed on the
seventh window under Python 3.11/3.12 in CI). Reusing one root also makes the
module noticeably faster everywhere.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pytest

tk = pytest.importorskip("tkinter", reason="Tk is not installed")

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def _root():
    """The single Tk window shared by every test in this module."""
    from tpsreg.GUI import ModernDistortionCorrectionView

    try:
        window = ModernDistortionCorrectionView()
    except tk.TclError as exc:
        pytest.skip(f"no usable display: {exc}")

    window.update_idletasks()
    try:
        yield window
    finally:
        # Already torn down is fine; nothing useful left to do.
        with contextlib.suppress(tk.TclError):
            window.destroy()


@pytest.fixture
def app(_root):
    """The shared window, reset to a clean project for each test."""
    _root.presenter.new_project()
    _root.update_idletasks()
    return _root


class TestMainWindow:
    """The main window comes up and is wired to a presenter."""

    def test_window_is_titled(self, app):
        assert app.title()

    def test_presenter_is_bidirectionally_wired(self, app):
        assert app.presenter is not None
        assert app.presenter.view is app

    def test_packaged_theme_is_applied(self, app):
        """Proves the theme resolved from inside the package, not the repo.

        A failure to source the .tcl leaves ttk on its default theme, so this
        is the end-to-end check that moving resources into the package worked.
        """
        from tkinter import ttk

        assert ttk.Style(app).theme_use().startswith("azure")

    def test_starts_with_no_data(self, app):
        assert app.presenter.source_image is None
        assert app.presenter.destination_image is None


class TestViewCallbacks:
    """The presenter's notification callbacks are all implemented."""

    @pytest.mark.parametrize(
        "callback",
        [
            "on_data_loaded",
            "on_display_update_needed",
            "on_error",
            "on_points_changed",
            "on_project_loaded",
            "on_project_reset",
            "on_request_corresponding_point",
            "on_show_matched_points",
        ],
    )
    def test_callback_exists(self, app, callback):
        assert callable(getattr(app, callback, None))

    def test_status_bar_accepts_a_message(self, app):
        app.set_status("hello")
        app.update_idletasks()


class TestLoadingThroughTheRealView:
    """Driving the real window with real files."""

    def test_loads_an_image_pair(self, app, tmp_path, checkerboard):
        from skimage import io

        src = tmp_path / "src.tif"
        dst = tmp_path / "dst.tif"
        io.imsave(src, checkerboard, check_contrast=False)
        io.imsave(dst, checkerboard.T.copy(), check_contrast=False)

        assert app.presenter.load_source_image(src, modality_name="BSE") is True
        assert app.presenter.load_destination_image(dst, modality_name="SE") is True

        # on_data_loaded repopulates the modality selectors and canvases.
        app.on_data_loaded()
        app.update_idletasks()

        assert app.presenter.source_image is not None

    def test_placing_points_updates_the_display(self, app, tmp_path, checkerboard):
        from skimage import io

        src = tmp_path / "src.tif"
        dst = tmp_path / "dst.tif"
        io.imsave(src, checkerboard, check_contrast=False)
        io.imsave(dst, checkerboard, check_contrast=False)

        app.presenter.load_source_image(src, modality_name="BSE")
        app.presenter.load_destination_image(dst, modality_name="SE")
        app.on_data_loaded()

        app.presenter.add_point("source", 20, 20)
        app.presenter.add_point("destination", 22, 21)
        app.update_idletasks()

        src_points, dst_points = app.presenter.get_points()
        np.testing.assert_array_equal(src_points, [[20, 20]])
        np.testing.assert_array_equal(dst_points, [[22, 21]])
