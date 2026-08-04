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

    def test_a_theme_is_active(self, app):
        """The window always ends up with a usable ttk theme.

        On Tk 8.6 this is the packaged Azure theme. If a future Tk cannot load
        it, the fallback must still leave a working theme rather than raising
        out of __init__.
        """
        from tkinter import ttk

        active = ttk.Style(app).theme_use()
        assert active
        assert active in ttk.Style(app).theme_names()
        assert active == app.theme_name

    @pytest.mark.skipif(
        tk.TkVersion >= 9.0,
        reason="packaged Azure theme targets Tk 8.6; see test_a_theme_is_active",
    )
    def test_packaged_azure_theme_loads_on_tk8(self, app):
        """Proves the theme resolved from inside the package, not the repo.

        A failure to source the .tcl leaves ttk on a built-in theme, so this is
        the end-to-end check that moving resources into the package worked.
        """
        from tkinter import ttk

        assert ttk.Style(app).theme_use().startswith("azure")

    def test_starts_with_no_data(self, app):
        assert app.presenter.source_image is None
        assert app.presenter.destination_image is None


class TestThemeFallback:
    """The window still opens when the packaged theme cannot be loaded.

    This is the Tk 9 scenario: Tcl refuses a bounded `package require Tk 8.6`
    against a 9.x interpreter, which used to raise straight out of __init__ and
    stop the application from opening at all.
    """

    def test_window_opens_when_the_theme_fails_to_load(self, monkeypatch):
        from tkinter import ttk

        import tpsreg.GUI as gui_module

        def broken_theme(_style):
            raise gui_module.tk.TclError(
                'version conflict for package "Tk": have 9.0.3, need 8.6'
            )

        monkeypatch.setattr(gui_module, "theme_path", broken_theme)

        try:
            window = gui_module.ModernDistortionCorrectionView()
        except tk.TclError as exc:
            pytest.skip(f"no usable display: {exc}")

        try:
            window.update_idletasks()
            active = ttk.Style(window).theme_use()

            # It fell back rather than raising, and the fallback is real.
            assert active in ttk.Style(window).theme_names()
            assert not active.startswith("azure")
            assert window.theme_name == active

            # The window is genuinely usable, not a half-built shell.
            assert window.title()
            assert window.presenter is not None
            window.set_status("still working")
        finally:
            with contextlib.suppress(tk.TclError):
                window.destroy()

    def test_unknown_style_is_rejected(self, app):
        """Validation lives in tpsreg.theme.get_palette, which names the options."""
        with pytest.raises(ValueError, match="Unknown theme"):
            app._style_call("solarized")


class TestPopupStyling:
    """Popup windows must look like the main window.

    ttk styling reaches ttk widgets application-wide, but never the Toplevel
    itself or plain Tk widgets like a canvas. Those were being left at the
    platform default, or hardcoded to "gray", which is what made the popups
    look inconsistent.
    """

    @staticmethod
    def _images():
        rng = np.random.default_rng(0)
        image = (rng.random((48, 48)) * 255).astype(np.uint8)
        points = np.array([[8, 8], [30, 30], [16, 40]], dtype=float)
        return image, points

    @pytest.fixture
    def popups(self, app):
        """Every popup viewer, opened and cleaned up afterwards."""
        import tpsreg.GUI as gui_module

        image, points = self._images()
        stack = np.repeat(image[None, ...], 3, axis=0)

        opened = []
        try:
            opened.append(
                (
                    "matched points",
                    gui_module.MatchedPointsViewer(app, image, image, points, points),
                )
            )
            opened.append(
                ("2D preview", gui_module.Interactive2DViewer(app, image, image))
            )
            opened.append(
                ("3D preview", gui_module.Interactive3DViewer(app, stack, stack))
            )
            for _, viewer in opened:
                viewer.root.update_idletasks()
            yield opened
        finally:
            for _, viewer in opened:
                with contextlib.suppress(tk.TclError):
                    viewer.root.destroy()

    def test_popup_windows_match_the_main_window(self, app, popups):
        for name, viewer in popups:
            assert viewer.root["background"] == app.palette.background, (
                f"the {name} window does not match the main window background"
            )

    def test_popup_canvases_match_the_palette(self, app, popups):
        """These were hardcoded to "gray" regardless of theme."""
        for name, viewer in popups:
            assert viewer.canvas["background"] == app.palette.canvas, (
                f"the {name} canvas does not use the palette"
            )

    def test_popups_inherit_the_palette_object(self, app, popups):
        for _, viewer in popups:
            assert viewer.palette is app.palette

    def test_main_window_has_a_palette(self, app):
        from tpsreg.theme import Palette

        assert isinstance(app.palette, Palette)

    def test_main_window_background_comes_from_the_palette(self, app):
        assert app["background"] == app.palette.background

    def test_main_canvases_use_the_palette(self, app):
        for canvas in (app.left_canvas, app.right_canvas):
            assert canvas["background"] == app.palette.canvas

    def test_palette_matches_the_active_ttk_theme(self, app):
        """A light palette with the dark ttk theme would look broken."""
        if app.theme_name.startswith("azure"):
            assert app.theme_name == app.palette.ttk_theme


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


class FakeEvent:
    """The attributes the canvas handlers read off a Tk event."""

    def __init__(self, x, y, widget=None):
        self.x = x
        self.y = y
        self.widget = widget


def at_image(app, canvas_type, image_x, image_y):
    """A press/motion event aimed at a given image pixel.

    The widget's highlight border offsets the drawing origin, so widget
    coordinates are not image coordinates even at 100%. This is the honest
    inverse of what the handlers do, which keeps the tests about behaviour
    rather than about Tk's border conventions.
    """
    canvas, scale = app._canvas_for(canvas_type)
    return FakeEvent(
        int(image_x * scale - canvas.canvasx(0)),
        int(image_y * scale - canvas.canvasy(0)),
    )


@pytest.fixture
def with_images(app, tmp_path, checkerboard):
    """The window with a loaded image pair, ready for canvas interaction."""
    from skimage import io

    src = tmp_path / "src.tif"
    dst = tmp_path / "dst.tif"
    io.imsave(src, checkerboard, check_contrast=False)
    io.imsave(dst, checkerboard.T.copy(), check_contrast=False)

    app.presenter.load_source_image(src, modality_name="BSE")
    app.presenter.load_destination_image(dst, modality_name="SE")
    app.on_data_loaded()
    app.update_idletasks()
    return app


class TestPlacingPointsByClicking:
    """A press and release that does not move is still an ordinary click.

    Placement moved from press to release so that a press landing on a marker
    can become a drag instead. These check the original behaviour survived.
    """

    def test_a_click_places_a_point(self, with_images):
        with_images._on_canvas_press(at_image(with_images, "source", 20, 20), "source")
        with_images._on_canvas_release(
            at_image(with_images, "source", 20, 20), "source"
        )

        src, _ = with_images.presenter.get_points()
        np.testing.assert_array_equal(src, [[20, 20]])

    def test_a_click_on_each_side_completes_a_pair(self, with_images):
        for canvas_type, x in (("source", 20), ("destination", 24)):
            event = at_image(with_images, canvas_type, x, x)
            with_images._on_canvas_press(event, canvas_type)
            with_images._on_canvas_release(event, canvas_type)

        src, dst = with_images.presenter.get_points()
        np.testing.assert_array_equal(src, [[20, 20]])
        np.testing.assert_array_equal(dst, [[24, 24]])

    def test_a_click_outside_the_image_is_ignored(self, with_images):
        with_images._on_canvas_press(
            at_image(with_images, "source", 9999, 9999), "source"
        )
        with_images._on_canvas_release(
            at_image(with_images, "source", 9999, 9999), "source"
        )

        src, _ = with_images.presenter.get_points()
        assert src.size == 0

    def test_clicking_before_an_image_is_loaded_places_nothing(self, app):
        app._on_canvas_press(FakeEvent(20, 20), "source")
        app._on_canvas_release(FakeEvent(20, 20), "source")

        src, _ = app.presenter.get_points()
        assert src.size == 0

    def test_a_press_with_no_image_arms_nothing(self, app):
        """A release must not act on a press that was refused."""
        app._on_canvas_press(FakeEvent(20, 20), "source")
        assert app._drag is None

    @pytest.mark.parametrize("zoom", [200, 400])
    def test_every_screen_pixel_of_a_cell_maps_to_that_cell(self, with_images, zoom):
        """Zoomed in, one image pixel covers a block of screen pixels.

        All of them have to report the same image coordinate. The conversion
        used to divide the event position and the canvas origin separately and
        truncate each, which made the last screen pixel of every block report
        the next image pixel along -- so at 400% a click could be recorded a
        whole pixel away from the feature it was aimed at, in exactly the
        situation where the extra precision of zooming in is wanted.
        """
        with_images.current_src_zoom = zoom
        with_images.update_display()
        with_images.update_idletasks()

        canvas, scale = with_images._canvas_for("source")
        target = 12
        origin = int(target * scale - canvas.canvasx(0))

        reported = {
            with_images._event_to_image(FakeEvent(origin + offset, origin), "source")[0]
            for offset in range(int(scale))
        }
        assert reported == {target}

    def test_the_cursor_readout_agrees_with_where_a_click_lands(self, with_images):
        """A readout that disagrees with the click is worse than none."""
        event = at_image(with_images, "source", 31, 17)
        with_images._on_canvas_motion(event, "source")
        assert with_images.cursor_label.cget("text") == "Cursor: 31, 17"

        with_images._on_canvas_press(event, "source")
        with_images._on_canvas_release(event, "source")
        src, _ = with_images.presenter.get_points()
        np.testing.assert_array_equal(src, [[31, 17]])

    def test_a_tiny_wobble_still_counts_as_a_click(self, with_images):
        """Nobody presses and releases on the exact same pixel."""
        start = at_image(with_images, "source", 20, 20)
        wobbled = FakeEvent(start.x + 1, start.y)
        with_images._on_canvas_press(start, "source")
        with_images._on_canvas_drag(wobbled, "source")
        with_images._on_canvas_release(wobbled, "source")

        src, _ = with_images.presenter.get_points()
        assert len(src) == 1


class TestDraggingAPoint:
    """Nudging a misplaced marker instead of deleting and re-placing the pair."""

    @pytest.fixture
    def with_pair(self, with_images):
        with_images.presenter.add_point("source", 20, 20)
        with_images.presenter.add_point("destination", 24, 24)
        with_images.update_idletasks()
        return with_images

    def test_dragging_a_marker_moves_it(self, with_pair):
        with_pair._on_canvas_press(at_image(with_pair, "source", 20, 20), "source")
        with_pair._on_canvas_drag(at_image(with_pair, "source", 30, 32), "source")
        with_pair._on_canvas_release(at_image(with_pair, "source", 30, 32), "source")

        src, _ = with_pair.presenter.get_points()
        np.testing.assert_array_equal(src, [[30, 32]])

    def test_dragging_does_not_place_an_extra_point(self, with_pair):
        with_pair._on_canvas_press(at_image(with_pair, "source", 20, 20), "source")
        with_pair._on_canvas_drag(at_image(with_pair, "source", 30, 32), "source")
        with_pair._on_canvas_release(at_image(with_pair, "source", 30, 32), "source")

        src, _ = with_pair.presenter.get_points()
        assert len(src) == 1

    def test_dragging_a_source_marker_leaves_its_partner(self, with_pair):
        with_pair._on_canvas_press(at_image(with_pair, "source", 20, 20), "source")
        with_pair._on_canvas_drag(at_image(with_pair, "source", 30, 32), "source")
        with_pair._on_canvas_release(at_image(with_pair, "source", 30, 32), "source")

        _, dst = with_pair.presenter.get_points()
        np.testing.assert_array_equal(dst, [[24, 24]])

    def test_the_marker_follows_the_cursor_during_the_drag(self, with_pair):
        """Feedback while dragging, not just on release."""
        with_pair._on_canvas_press(at_image(with_pair, "source", 20, 20), "source")
        with_pair._on_canvas_drag(at_image(with_pair, "source", 28, 28), "source")

        src, _ = with_pair.presenter.get_points()
        np.testing.assert_array_equal(src, [[28, 28]])

    def test_a_whole_drag_undoes_in_one_step(self, with_pair):
        with_pair._on_canvas_press(at_image(with_pair, "source", 20, 20), "source")
        for step in range(21, 40):
            with_pair._on_canvas_drag(
                at_image(with_pair, "source", step, step), "source"
            )
        with_pair._on_canvas_release(at_image(with_pair, "source", 39, 39), "source")

        with_pair.presenter.undo()
        src, _ = with_pair.presenter.get_points()
        np.testing.assert_array_equal(src, [[20, 20]])

    def test_grabbing_a_marker_and_releasing_changes_nothing(self, with_pair):
        """A click near an existing point must not stack a second one on it."""
        near = at_image(with_pair, "source", 22, 21)
        with_pair._on_canvas_press(near, "source")
        with_pair._on_canvas_release(near, "source")

        src, _ = with_pair.presenter.get_points()
        np.testing.assert_array_equal(src, [[20, 20]])

    def test_dragging_from_empty_space_places_nothing(self, with_pair):
        """A drag that grabbed nothing is not a click either."""
        before, _ = with_pair.presenter.get_points()

        with_pair._on_canvas_press(at_image(with_pair, "source", 5, 5), "source")
        with_pair._on_canvas_drag(at_image(with_pair, "source", 40, 40), "source")
        with_pair._on_canvas_release(at_image(with_pair, "source", 40, 40), "source")

        after, _ = with_pair.presenter.get_points()
        np.testing.assert_array_equal(after, before)

    def test_a_drag_off_the_image_leaves_the_point_where_it_was(self, with_pair):
        with_pair._on_canvas_press(at_image(with_pair, "source", 20, 20), "source")
        with_pair._on_canvas_drag(at_image(with_pair, "source", 9999, 9999), "source")
        with_pair._on_canvas_release(
            at_image(with_pair, "source", 9999, 9999), "source"
        )

        src, _ = with_pair.presenter.get_points()
        np.testing.assert_array_equal(src, [[20, 20]])

    def test_a_drag_event_for_the_other_canvas_is_ignored(self, with_pair):
        """Motion is delivered to the widget the press started on."""
        with_pair._on_canvas_press(at_image(with_pair, "source", 20, 20), "source")
        with_pair._on_canvas_drag(
            at_image(with_pair, "destination", 40, 40), "destination"
        )

        src, _ = with_pair.presenter.get_points()
        np.testing.assert_array_equal(src, [[20, 20]])

    def test_a_release_with_no_press_is_survivable(self, with_pair):
        with_pair._drag = None
        with_pair._on_canvas_release(at_image(with_pair, "source", 30, 30), "source")

    def test_the_grab_radius_holds_its_screen_size_when_zoomed(self, with_pair):
        """Zoomed in, a marker covers more pixels, so the radius must shrink.

        The radius is quoted in screen pixels and divided by the zoom scale. At
        400% a press 8 image-pixels away is 32 screen pixels away, which is
        well outside the grab and must place a new point instead.
        """
        with_pair.current_src_zoom = 400
        with_pair._on_canvas_press(at_image(with_pair, "source", 28, 28), "source")
        assert with_pair._drag["index"] is None

    def test_hidden_points_cannot_be_grabbed(self, with_pair):
        """Nothing is drawn, so there is nothing to aim at."""
        with_pair.show_points = False
        with_pair._on_canvas_press(at_image(with_pair, "source", 20, 20), "source")
        assert with_pair._drag["index"] is None
        with_pair.show_points = True

    def test_a_destination_marker_drags_too(self, with_pair):
        press = at_image(with_pair, "destination", 24, 24)
        target = at_image(with_pair, "destination", 35, 36)
        with_pair._on_canvas_press(press, "destination")
        with_pair._on_canvas_drag(target, "destination")
        with_pair._on_canvas_release(target, "destination")

        src, dst = with_pair.presenter.get_points()
        np.testing.assert_array_equal(dst, [[35, 36]])
        np.testing.assert_array_equal(src, [[20, 20]])


class TestLinkedViewers:
    """Keeping the two panels showing the same thing."""

    def test_starts_unlinked(self, app):
        assert app.link_views_var.get() is False

    def test_linking_matches_the_destination_zoom_to_the_source(self, app):
        app.zoom_src_var.set("200%")
        app.current_src_zoom = 200
        app.zoom_dst_var.set("50%")

        app.link_views_var.set(True)
        app._on_link_views_changed()

        assert app.zoom_dst_var.get() == "200%"
        assert app.current_dst_zoom == 200

    def test_changing_the_source_zoom_carries_the_destination(self, app):
        app.link_views_var.set(True)
        app.zoom_src_var.set("300%")
        app._on_zoom_changed(FakeEvent(0, 0, widget=app.zoom_src_combo))

        assert app.zoom_dst_var.get() == "300%"
        assert app.current_dst_zoom == 300

    def test_changing_the_destination_zoom_carries_the_source(self, app):
        app.link_views_var.set(True)
        app.zoom_dst_var.set("25%")
        app._on_zoom_changed(FakeEvent(0, 0, widget=app.zoom_dst_combo))

        assert app.zoom_src_var.get() == "25%"
        assert app.current_src_zoom == 25

    def test_unlinked_zooms_stay_independent(self, app):
        app.link_views_var.set(False)
        app.zoom_src_var.set("300%")
        app.zoom_dst_var.set("50%")
        app._on_zoom_changed(FakeEvent(0, 0, widget=app.zoom_src_combo))

        assert app.zoom_dst_var.get() == "50%"
        assert app.current_dst_zoom == 50

    def test_scrolling_one_panel_scrolls_the_other_when_linked(self, with_images):
        with_images.left_canvas.config(scrollregion=(0, 0, 2000, 2000))
        with_images.right_canvas.config(scrollregion=(0, 0, 2000, 2000))
        with_images.update_idletasks()

        with_images.link_views_var.set(True)
        with_images._on_scroll("source", "y", "moveto", "0.5")
        with_images.update_idletasks()

        assert with_images.right_canvas.yview()[0] == pytest.approx(
            with_images.left_canvas.yview()[0], abs=0.02
        )

    def test_scrolling_leaves_the_other_panel_alone_when_unlinked(self, with_images):
        with_images.left_canvas.config(scrollregion=(0, 0, 2000, 2000))
        with_images.right_canvas.config(scrollregion=(0, 0, 2000, 2000))
        with_images.update_idletasks()

        with_images.link_views_var.set(False)
        before = with_images.right_canvas.yview()[0]
        with_images._on_scroll("source", "y", "moveto", "0.5")
        with_images.update_idletasks()

        assert with_images.right_canvas.yview()[0] == pytest.approx(before)

    def test_scrolling_the_destination_carries_the_source(self, with_images):
        with_images.left_canvas.config(scrollregion=(0, 0, 2000, 2000))
        with_images.right_canvas.config(scrollregion=(0, 0, 2000, 2000))
        with_images.update_idletasks()

        with_images.link_views_var.set(True)
        with_images._on_scroll("destination", "x", "moveto", "0.4")
        with_images.update_idletasks()

        assert with_images.left_canvas.xview()[0] == pytest.approx(
            with_images.right_canvas.xview()[0], abs=0.02
        )


class TestEditMenuState:
    """Undo and Redo are greyed out when they would do nothing."""

    @staticmethod
    def _state(app, label):
        return str(app.edit_menu.entrycget(label, "state"))

    def test_both_start_disabled(self, app):
        assert self._state(app, "Undo") == "disabled"
        assert self._state(app, "Redo") == "disabled"

    def test_undo_is_enabled_after_an_edit(self, with_images):
        with_images.presenter.add_point("source", 20, 20)
        assert self._state(with_images, "Undo") == "normal"

    def test_redo_is_enabled_after_undoing(self, with_images):
        with_images.presenter.add_point("source", 20, 20)
        with_images.presenter.undo()
        with_images.on_points_changed()
        assert self._state(with_images, "Redo") == "normal"

    def test_the_menu_state_agrees_with_the_presenter(self, with_images):
        with_images.presenter.add_point("source", 20, 20)
        with_images.presenter.add_point("destination", 24, 24)

        for _ in range(4):
            expected = with_images.presenter.can_undo()
            assert (self._state(with_images, "Undo") == "normal") is expected
            with_images.presenter.undo()
            with_images.on_points_changed()

    def test_a_new_project_disables_both_again(self, with_images):
        with_images.presenter.add_point("source", 20, 20)
        assert self._state(with_images, "Undo") == "normal"

        with_images.presenter.new_project()
        assert self._state(with_images, "Undo") == "disabled"
        assert self._state(with_images, "Redo") == "disabled"


class TestRememberingTheLastDirectory:
    """Every file dialog should open where the last one left off."""

    def test_nothing_is_remembered_at_first(self, app):
        assert app._last_directory is None

    def test_choosing_a_file_remembers_its_folder(self, app, tmp_path):
        chosen = tmp_path / "data" / "image.tif"
        chosen.parent.mkdir()
        chosen.touch()

        app._remember_directory(str(chosen))
        assert app._last_directory == chosen.parent.resolve()

    def test_cancelling_a_dialog_changes_nothing(self, app, tmp_path):
        app._remember_directory(str(tmp_path / "x.tif"))
        remembered = app._last_directory

        app._remember_directory("")  # Tk returns "" when cancelled
        assert app._last_directory == remembered

    def test_the_remembered_folder_is_offered_to_the_next_dialog(
        self, app, tmp_path, monkeypatch
    ):
        from tkinter import filedialog

        first = tmp_path / "session" / "source.tif"
        first.parent.mkdir()
        first.touch()

        seen = {}

        def fake_dialog(**kwargs):
            seen.update(kwargs)
            return ""

        monkeypatch.setattr(filedialog, "askopenfilename", fake_dialog)

        app._remember_directory(str(first))
        app._ask_open(title="Second dialog")

        assert seen["initialdir"] == str(first.parent.resolve())

    def test_an_explicit_directory_wins(self, app, tmp_path, monkeypatch):
        """The checkpoint dialog starts next to the current checkpoint."""
        from tkinter import filedialog

        other = tmp_path / "checkpoints"
        other.mkdir()
        app._last_directory = tmp_path

        seen = {}
        monkeypatch.setattr(
            filedialog, "askopenfilename", lambda **kw: seen.update(kw) or ""
        )

        app._ask_open(initialdir=other, title="Checkpoint")
        assert seen["initialdir"] == str(other)

    def test_a_folder_that_no_longer_exists_is_not_offered(self, app, tmp_path):
        app._last_directory = tmp_path / "deleted"
        assert app._initial_directory() is None

    def test_a_save_dialog_remembers_too(self, app, tmp_path, monkeypatch):
        from tkinter import filedialog

        target = tmp_path / "exports"
        target.mkdir()
        monkeypatch.setattr(
            filedialog, "asksaveasfilename", lambda **kw: str(target / "out.tif")
        )

        app._ask_save(title="Export")
        assert app._last_directory == target.resolve()

    def test_a_multi_file_dialog_remembers_the_first(self, app, tmp_path, monkeypatch):
        from tkinter import filedialog

        folder = tmp_path / "stack"
        folder.mkdir()
        monkeypatch.setattr(
            filedialog,
            "askopenfilenames",
            lambda **kw: (str(folder / "a.tif"), str(folder / "b.tif")),
        )

        app._ask_open_many(title="Open stack")
        assert app._last_directory == folder.resolve()


class TestPreviewComparisonModes:
    """The 2D preview offers more than a wipe."""

    @pytest.fixture
    def viewer(self, app):
        import tpsreg.GUI as gui_module

        rng = np.random.default_rng(0)
        image = (rng.random((48, 48)) * 255).astype(np.uint8)

        window = gui_module.Interactive2DViewer(app, image, image.T.copy())
        window.root.update_idletasks()
        try:
            yield window
        finally:
            with contextlib.suppress(tk.TclError):
                window.root.destroy()

    def test_it_opens_in_the_familiar_wipe_mode(self, viewer):
        assert viewer.blend_mode_var.get() == "wipe"

    def test_every_mode_renders(self, viewer):
        from tpsreg import overlays

        for mode in overlays.BLEND_MODES:
            viewer.blend_mode_var.set(mode)
            viewer._on_blend_mode_changed()
            viewer.root.update_idletasks()
            assert viewer.blend_images().size == (48, 48)

    def test_the_modes_actually_differ(self, viewer):
        rendered = {}
        for mode in ("wipe", "checkerboard", "difference"):
            viewer.blend_mode_var.set(mode)
            rendered[mode] = np.asarray(viewer.blend_images())

        assert not np.array_equal(rendered["wipe"], rendered["checkerboard"])
        assert not np.array_equal(rendered["checkerboard"], rendered["difference"])

    def test_the_wipe_sliders_are_disabled_outside_wipe_mode(self, viewer):
        viewer.blend_mode_var.set("difference")
        viewer._on_blend_mode_changed()
        assert "disabled" in viewer.x_slider.state()
        assert "disabled" in viewer.y_slider.state()

    def test_the_wipe_sliders_come_back(self, viewer):
        viewer.blend_mode_var.set("difference")
        viewer._on_blend_mode_changed()
        viewer.blend_mode_var.set("wipe")
        viewer._on_blend_mode_changed()

        assert "disabled" not in viewer.x_slider.state()
        assert "disabled" not in viewer.y_slider.state()

    def test_the_tile_size_only_applies_to_the_checkerboard(self, viewer):
        viewer.blend_mode_var.set("checkerboard")
        viewer._on_blend_mode_changed()
        assert str(viewer.tile_spinbox["state"]) == "normal"

        viewer.blend_mode_var.set("wipe")
        viewer._on_blend_mode_changed()
        assert str(viewer.tile_spinbox["state"]) == "disabled"

    def test_changing_the_tile_size_changes_the_render(self, viewer):
        viewer.blend_mode_var.set("checkerboard")
        viewer.tile_size_var.set(4)
        small = np.asarray(viewer.blend_images())
        viewer.tile_size_var.set(24)
        large = np.asarray(viewer.blend_images())

        assert not np.array_equal(small, large)

    def test_the_viewer_still_matches_the_palette(self, app, viewer):
        assert viewer.root["background"] == app.palette.background
        assert viewer.canvas["background"] == app.palette.canvas


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
