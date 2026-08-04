"""Tests for the presenter layer, driven through a fake view."""

from __future__ import annotations

import numpy as np
import pytest
from skimage import io

from tpsreg.models import TransformType
from tpsreg.presenter import ApplicationPresenter, CropMode


@pytest.fixture
def image_pair(tmp_path, checkerboard):
    """A source and destination TIFF written to disk."""
    src_path = tmp_path / "source.tif"
    dst_path = tmp_path / "destination.tif"
    io.imsave(src_path, checkerboard, check_contrast=False)
    io.imsave(dst_path, checkerboard.T.copy(), check_contrast=False)
    return src_path, dst_path


@pytest.fixture
def loaded(presenter, image_pair):
    """A presenter with both images loaded."""
    src_path, dst_path = image_pair
    presenter.load_source_image(src_path, resolution=1.0, modality_name="BSE")
    presenter.load_destination_image(dst_path, resolution=1.0, modality_name="SE")
    return presenter


class TestInitialState:
    """A freshly constructed presenter."""

    def test_starts_empty(self, presenter):
        assert presenter.source_image is None
        assert presenter.destination_image is None
        assert presenter.current_slice == 0

    def test_no_unsaved_changes_when_empty(self, presenter):
        assert presenter.has_unsaved_changes() is False

    def test_new_project_notifies_the_view(self, presenter, fake_view):
        presenter.new_project()
        assert "project_reset" in fake_view.calls


class TestLoading:
    """Loading images."""

    def test_source_load_sets_data_and_notifies(self, presenter, fake_view, image_pair):
        src_path, _ = image_pair
        assert presenter.load_source_image(src_path, modality_name="BSE") is True

        assert presenter.source_image is not None
        assert "BSE" in presenter.source_image.modalities
        assert "data_loaded" in fake_view.calls

    def test_destination_load_sets_data(self, presenter, image_pair):
        _, dst_path = image_pair
        assert presenter.load_destination_image(dst_path, modality_name="SE") is True
        assert "SE" in presenter.destination_image.modalities

    def test_missing_file_reports_error_without_raising(
        self, presenter, fake_view, tmp_path
    ):
        assert presenter.load_source_image(tmp_path / "nope.tif") is False
        assert fake_view.errors, "the view should have been told about the failure"

    def test_loading_marks_the_project_unsaved(self, loaded):
        assert loaded.has_unsaved_changes() is True

    def test_second_source_modality_is_added(self, loaded, tmp_path, checkerboard):
        """Loading another source image adds a modality on the same grid.

        This path used to call add_modality with three positional arguments
        against a one-argument method, so the documented workflow of layering
        modalities failed on the source side while working on the destination.
        """
        extra = tmp_path / "source_se.tif"
        io.imsave(extra, checkerboard, check_contrast=False)

        assert loaded.load_source_image(extra, modality_name="SE2") is True

        assert sorted(loaded.get_source_modalities()) == ["BSE", "SE2"]
        # The view should switch to whatever was just loaded.
        assert loaded.current_source_mode == "SE2"

    def test_second_source_modality_reports_shape_mismatch(self, loaded, tmp_path):
        """A differently sized second modality is refused, not silently added."""
        mismatched = np.zeros((16, 16), dtype=np.uint8)
        mismatched[4:12, 4:12] = 255
        path = tmp_path / "wrong_size.tif"
        io.imsave(path, mismatched, check_contrast=False)

        assert loaded.load_source_image(path, modality_name="SE2") is False
        assert loaded.get_source_modalities() == ["BSE"]

    def test_second_destination_modality_is_added(self, loaded, tmp_path, checkerboard):
        extra = tmp_path / "dest_bse.tif"
        io.imsave(extra, checkerboard.T.copy(), check_contrast=False)

        assert loaded.load_destination_image(extra, modality_name="BSE2") is True
        assert sorted(loaded.get_destination_modalities()) == ["BSE2", "SE"]

    def test_slice_range_for_a_single_image(self, loaded):
        assert loaded.get_slice_range() == (0, 0)

    def test_modalities_are_reported(self, loaded):
        assert loaded.get_source_modalities() == ["BSE"]
        assert loaded.get_destination_modalities() == ["SE"]


class TestPointEditing:
    """Adding and removing control points."""

    def test_add_source_point_requests_its_partner(self, loaded, fake_view):
        loaded.add_point("source", 10, 12)
        assert "destination" in fake_view.requested_points

    def test_completing_a_pair_stores_it(self, loaded):
        loaded.add_point("source", 10, 12)
        loaded.add_point("destination", 20, 22)

        src, dst = loaded.get_points()
        np.testing.assert_array_equal(src, [[10, 12]])
        np.testing.assert_array_equal(dst, [[20, 22]])

    def test_out_of_bounds_point_is_refused(self, loaded, fake_view):
        loaded.add_point("source", 9999, 9999)
        src, _ = loaded.get_points()
        assert src.size == 0

    def test_bounds_check_reflects_image_size(self, loaded):
        assert loaded.is_point_in_bounds("source", 5, 5) is True
        assert loaded.is_point_in_bounds("source", 9999, 5) is False

    def test_remove_point_drops_the_pair(self, loaded):
        loaded.add_point("source", 10, 12)
        loaded.add_point("destination", 20, 22)
        loaded.remove_point(0)

        src, dst = loaded.get_points()
        assert src.size == 0
        assert dst.size == 0

    def test_removing_a_missing_point_is_survivable(self, loaded):
        """Right-clicking empty canvas must not raise."""
        loaded.remove_point(5)
        src, _ = loaded.get_points()
        assert src.size == 0

    def test_clear_points(self, loaded):
        loaded.add_point("source", 10, 12)
        loaded.add_point("destination", 20, 22)
        loaded.clear_points(slice_only=False)

        src, _ = loaded.get_points()
        assert src.size == 0

    def test_undo_steps_back_one_click_at_a_time(self, loaded):
        """Each click is one undoable action, including a dangling source point.

        Manual placement used to bypass the undo history entirely, so Edit >
        Undo did nothing for the primary workflow.
        """
        loaded.add_point("source", 10, 12)
        loaded.add_point("destination", 20, 22)
        loaded.add_point("source", 30, 32)
        loaded.add_point("destination", 40, 42)

        # One undo removes the last click: the second destination point.
        loaded.undo()
        src, dst = loaded.get_points()
        assert len(src) == 2
        assert len(dst) == 1

        # A second undo removes the dangling source point, restoring a
        # consistent single pair.
        loaded.undo()
        src, dst = loaded.get_points()
        assert len(src) == 1
        assert len(dst) == 1

    def test_redo_reapplies_undone_clicks(self, loaded):
        loaded.add_point("source", 10, 12)
        loaded.add_point("destination", 20, 22)
        loaded.add_point("source", 30, 32)
        loaded.add_point("destination", 40, 42)

        loaded.undo()
        loaded.undo()
        loaded.redo()
        loaded.redo()

        src, dst = loaded.get_points()
        assert len(src) == 2
        assert len(dst) == 2
        np.testing.assert_array_equal(src, [[10, 12], [30, 32]])
        np.testing.assert_array_equal(dst, [[20, 22], [40, 42]])


class TestMovingPoints:
    """Dragging a marker, as the canvas drives it."""

    @pytest.fixture
    def with_pair(self, loaded):
        loaded.add_point("source", 10, 12)
        loaded.add_point("destination", 20, 22)
        return loaded

    def test_moving_a_source_point(self, with_pair):
        assert with_pair.move_point("source", 0, 15, 18) is True

        src, dst = with_pair.get_points()
        np.testing.assert_array_equal(src, [[15, 18]])
        np.testing.assert_array_equal(dst, [[20, 22]])

    def test_moving_a_destination_point(self, with_pair):
        assert with_pair.move_point("destination", 0, 25, 27) is True

        src, dst = with_pair.get_points()
        np.testing.assert_array_equal(src, [[10, 12]])
        np.testing.assert_array_equal(dst, [[25, 27]])

    def test_a_move_outside_the_image_is_refused(self, with_pair):
        assert with_pair.move_point("source", 0, 9999, 9999) is False
        np.testing.assert_array_equal(with_pair.get_points()[0], [[10, 12]])

    def test_moving_a_point_that_does_not_exist_is_refused(self, with_pair):
        assert with_pair.move_point("source", 7, 15, 18) is False

    def test_an_unknown_side_is_refused_not_raised(self, with_pair, fake_view):
        """The canvas type comes from a binding; a typo must not crash the app."""
        assert with_pair.move_point("middle", 0, 15, 18) is False
        assert fake_view.errors

    def test_moving_before_an_image_is_loaded_is_refused(self, presenter):
        assert presenter.move_point("source", 0, 5, 5) is False

    def test_a_move_marks_the_project_modified(self, with_pair):
        with_pair.project_manager.is_modified = False
        with_pair.move_point("source", 0, 15, 18)
        assert with_pair.project_manager.is_modified is True

    def test_a_move_tells_the_view_to_redraw(self, with_pair, fake_view):
        before = fake_view.calls.count("points_changed")
        with_pair.move_point("source", 0, 15, 18)
        assert fake_view.calls.count("points_changed") > before

    def test_a_transient_move_still_redraws(self, with_pair, fake_view):
        """The marker has to follow the cursor while the drag is in progress."""
        before = fake_view.calls.count("points_changed")
        with_pair.move_point("source", 0, 15, 18, transient=True)
        assert fake_view.calls.count("points_changed") > before
        np.testing.assert_array_equal(with_pair.get_points()[0], [[15, 18]])

    def test_a_transient_move_leaves_the_project_clean(self, with_pair):
        """Intermediate drag frames must not rewrite the points file."""
        with_pair.project_manager.is_modified = False
        with_pair.move_point("source", 0, 15, 18, transient=True)
        assert with_pair.project_manager.is_modified is False

    def test_committing_marks_the_project_modified(self, with_pair):
        with_pair.project_manager.is_modified = False
        with_pair.move_point("source", 0, 15, 18, transient=True)
        with_pair.commit_point_move()
        assert with_pair.project_manager.is_modified is True

    def test_a_whole_drag_undoes_in_one_step(self, with_pair):
        """The gesture is one edit, not one per mouse motion."""
        with_pair.move_point("source", 0, 11, 13)
        for step in range(14, 30):
            with_pair.move_point("source", 0, step, step, transient=True)
        with_pair.commit_point_move()

        with_pair.undo()
        np.testing.assert_array_equal(with_pair.get_points()[0], [[10, 12]])

    def test_a_destination_move_respects_matched_resolutions(self, loaded):
        """Displayed destination coordinates are at the source scale.

        add_point converts them back before storing, and a move has to make
        exactly the same conversion or dragging a point would teleport it.
        """
        loaded.set_image_resolutions(1.0, 2.0)
        loaded.match_resolutions = True

        loaded.add_point("source", 10, 12)
        loaded.add_point("destination", 20, 24)
        stored_after_add = loaded.get_points()[1].copy()

        loaded.move_point("destination", 0, 20, 24)
        np.testing.assert_array_equal(loaded.get_points()[1], stored_after_add)


class TestFindingPointsNearACursor:
    """Hit testing, which decides whether a press grabs a marker."""

    @pytest.fixture
    def with_points(self, loaded):
        for x, y in ((10, 10), (30, 30), (50, 20)):
            loaded.add_point("source", x, y)
            loaded.add_point("destination", x + 1, y + 1)
        return loaded

    def test_a_press_on_a_marker_finds_it(self, with_points):
        assert with_points.find_point_near("source", 30, 30, radius=8) == 1

    def test_a_press_near_a_marker_finds_it(self, with_points):
        assert with_points.find_point_near("source", 33, 32, radius=8) == 1

    def test_a_press_in_empty_space_finds_nothing(self, with_points):
        assert with_points.find_point_near("source", 200, 200, radius=8) is None

    def test_just_outside_the_radius_finds_nothing(self, with_points):
        assert with_points.find_point_near("source", 40, 30, radius=8) is None

    def test_the_nearest_marker_wins(self, with_points):
        """Two markers within reach: the closer one is the one being grabbed."""
        assert with_points.find_point_near("source", 48, 21, radius=40) == 2

    def test_destination_points_are_searched_separately(self, with_points):
        assert with_points.find_point_near("destination", 31, 31, radius=4) == 1

    def test_no_points_at_all_finds_nothing(self, loaded):
        assert loaded.find_point_near("source", 10, 10, radius=8) is None

    def test_hit_testing_respects_matched_resolutions(self, loaded):
        """Markers are drawn at the source scale, so hits are tested there too."""
        loaded.set_image_resolutions(1.0, 2.0)
        loaded.add_point("source", 10, 10)
        loaded.add_point("destination", 10, 10)

        loaded.match_resolutions = True
        # Stored at 10,10 but drawn at 20,20; a press at 20,20 must hit.
        assert loaded.find_point_near("destination", 20, 20, radius=3) == 0
        assert loaded.find_point_near("destination", 10, 10, radius=3) is None


class TestCheckingPointsBeforeEstimating:
    """The presenter's wrapper around the validation checks."""

    def test_no_points_is_reported_as_an_error(self, loaded):
        issues = loaded.check_points(TransformType.TPS)
        assert issues
        assert any(issue.is_error for issue in issues)

    def test_good_points_pass(self, loaded):
        for x, y in ((5, 5), (45, 6), (6, 45), (44, 44), (25, 12), (12, 26)):
            loaded.add_point("source", x, y)
            loaded.add_point("destination", x + 1, y + 1)

        assert loaded.check_points(TransformType.TPS) == []

    def test_duplicate_points_are_reported(self, loaded):
        for x, y in ((5, 5), (45, 6), (6, 45), (44, 44), (25, 12), (5, 5)):
            loaded.add_point("source", x, y)
            loaded.add_point("destination", x + 1, y + 1)

        codes = {issue.code for issue in loaded.check_points(TransformType.TPS)}
        assert "duplicate_source_points" in codes

    def test_coverage_uses_the_loaded_image_size(self, loaded):
        """Clustered points only look clustered relative to the image."""
        for x, y in ((2, 2), (5, 2), (2, 5), (5, 5), (3, 4), (4, 3)):
            loaded.add_point("source", x, y)
            loaded.add_point("destination", x, y)

        codes = {issue.code for issue in loaded.check_points(TransformType.TPS)}
        assert "poor_coverage" in codes

    def test_defaults_to_tps_when_no_type_is_given(self, loaded):
        assert loaded.check_points() == loaded.check_points(TransformType.TPS)

    def test_checking_without_an_image_still_works(self, presenter):
        """The dialog may be reached before anything is loaded."""
        issues = presenter.check_points(TransformType.TPS)
        assert {issue.code for issue in issues} == {"no_points"}


class TestUndoAvailability:
    """What the Edit menu should be showing."""

    def test_nothing_to_undo_on_a_fresh_project(self, presenter):
        assert presenter.can_undo() is False
        assert presenter.can_redo() is False

    def test_placing_a_point_makes_undo_available(self, loaded):
        loaded.add_point("source", 10, 12)
        assert loaded.can_undo() is True

    def test_undoing_makes_redo_available(self, loaded):
        loaded.add_point("source", 10, 12)
        loaded.undo()
        assert loaded.can_redo() is True


class TestAssessingTheTransform:
    """The quality report the presenter assembles."""

    @staticmethod
    def _grid(extent=40.0, n=4):
        axis = np.linspace(4.0, extent, n)
        return np.stack(np.meshgrid(axis, axis), -1).reshape(-1, 2)

    def _place(self, presenter, src, dst):
        for (sx, sy), (dx, dy) in zip(src, dst, strict=True):
            presenter.add_point("source", int(sx), int(sy))
            presenter.add_point("destination", int(dx), int(dy))

    def test_no_points_gives_no_report(self, loaded):
        assert loaded.assess_transform(TransformType.TPS) is None

    def test_a_clean_fit_reports_no_outliers(self, loaded):
        grid = self._grid()
        self._place(loaded, grid, grid)

        quality = loaded.assess_transform(TransformType.TPS)
        assert quality is not None
        assert not quality.outliers.any()
        assert not quality.has_folds

    def test_a_misplaced_point_is_flagged(self, loaded):
        grid = self._grid()
        src = grid.copy()
        src[6] = src[6] + np.array([18.0, -14.0])
        self._place(loaded, src, grid)

        quality = loaded.assess_transform(TransformType.TPS)
        assert quality.outliers[6]
        assert quality.worst_point == 6

    def test_coverage_comes_from_the_loaded_image(self, loaded):
        grid = self._grid()
        self._place(loaded, grid, grid)

        quality = loaded.assess_transform(TransformType.TPS)
        assert quality.coverage is not None
        assert 0 < quality.coverage <= 1

    def test_the_expensive_part_can_be_skipped(self, loaded):
        grid = self._grid()
        self._place(loaded, grid, grid)

        quality = loaded.assess_transform(
            TransformType.TPS, include_leave_one_out=False
        )
        assert quality.leave_one_out.size == 0

    def test_a_degenerate_fit_reports_the_error_rather_than_raising(
        self, loaded, fake_view
    ):
        """Collinear points cannot be fitted; the view should hear about it."""
        for i in range(5):
            loaded.add_point("source", 5 + i * 5, 5 + i * 5)
            loaded.add_point("destination", 5 + i * 5, 5 + i * 5)

        assert loaded.assess_transform(TransformType.TPS) is None
        assert fake_view.errors


class TestDisplayState:
    """View mode, slice and resolution settings."""

    def test_set_current_slice(self, loaded):
        loaded.set_current_slice(0)
        assert loaded.current_slice == 0

    def test_set_modes(self, loaded):
        loaded.set_source_mode("BSE")
        loaded.set_destination_mode("SE")
        assert loaded.current_source_mode == "BSE"
        assert loaded.current_dest_mode == "SE"

    def test_set_and_read_resolutions(self, loaded):
        loaded.set_image_resolutions(0.5, 1.5)
        assert loaded.get_resolutions() == (0.5, 1.5)

    def test_toggle_clahe(self, loaded):
        assert loaded.clahe_active_source is False
        loaded.toggle_clahe("source")
        assert loaded.clahe_active_source is True
        loaded.toggle_clahe("source")
        assert loaded.clahe_active_source is False

    def test_toggle_match_resolutions(self, loaded):
        assert loaded.match_resolutions is False
        loaded.toggle_match_resolutions()
        assert loaded.match_resolutions is True

    def test_get_current_images_returns_both(self, loaded):
        src, dst = loaded.get_current_images()
        assert src.ndim >= 2
        assert dst.ndim >= 2


class TestTransforms:
    """Estimating and previewing a correction."""

    @staticmethod
    def _add_grid(presenter, jitter=0.0):
        """Place a spread of matched points on both images."""
        coords = [(10, 10), (10, 50), (50, 10), (50, 50), (30, 30), (20, 45)]
        for x, y in coords:
            presenter.add_point("source", x, y)
            presenter.add_point("destination", int(x + jitter), int(y + jitter))

    def test_transform_without_points_reports_an_error(self, loaded, fake_view):
        result = loaded.apply_transform(TransformType.TPS, return_data=True)
        assert result is None
        assert any("control point" in e for e in fake_view.errors)

    def test_transform_with_points_produces_a_warp(self, loaded):
        self._add_grid(loaded)
        warped, _src_img, dst_img = loaded.apply_transform(
            TransformType.TPS, return_data=True
        )
        assert warped.shape[:2] == dst_img.shape[:2]
        assert np.all(np.isfinite(warped))

    def test_affine_transform_also_works(self, loaded):
        self._add_grid(loaded, jitter=2.0)
        warped, _, _ = loaded.apply_transform(
            TransformType.TPS_AFFINE, return_data=True
        )
        assert np.all(np.isfinite(warped))

    def test_preview_reaches_the_view(self, loaded, fake_view):
        self._add_grid(loaded)
        loaded.apply_transform(TransformType.TPS, preview=True)
        assert "preview_2d" in fake_view.calls

    def test_source_crop_mode_returns_source_sized_output(self, loaded):
        self._add_grid(loaded)
        warped, src_img, _ = loaded.apply_transform(
            TransformType.TPS, crop_mode=CropMode.SOURCE, return_data=True
        )
        assert warped.shape[0] <= src_img.shape[0]
        assert warped.shape[1] <= src_img.shape[1]

    def test_export_transform_writes_a_file(self, loaded, tmp_path):
        self._add_grid(loaded)
        path = tmp_path / "transform.npy"
        assert loaded.export_transform(path, TransformType.TPS) is True
        assert path.exists()

    def test_matched_points_view_is_notified(self, loaded, fake_view):
        self._add_grid(loaded)
        loaded.show_matched_points()
        assert "matched_points" in fake_view.calls


class TestProjectPersistence:
    """Saving and reloading a project."""

    def test_save_and_reload_restores_points(self, loaded, tmp_path, fake_view):
        loaded.add_point("source", 10, 12)
        loaded.add_point("destination", 20, 22)

        path = tmp_path / "project.json"
        assert loaded.save_project(path) is True
        assert path.exists()

        fresh = ApplicationPresenter()
        fresh.set_view(fake_view)
        assert fresh.load_project(path) is True

        src, dst = fresh.get_points()
        np.testing.assert_array_equal(src, [[10, 12]])
        np.testing.assert_array_equal(dst, [[20, 22]])

    def test_save_clears_the_unsaved_flag(self, loaded, tmp_path):
        loaded.save_project(tmp_path / "project.json")
        assert loaded.has_unsaved_changes() is False

    def test_loading_a_broken_project_reports_an_error(
        self, presenter, fake_view, tmp_path
    ):
        path = tmp_path / "broken.json"
        path.write_text("{not json")

        assert presenter.load_project(path) is False
        assert fake_view.errors


class TestAutoDetection:
    """Automatic point detection dispatch."""

    def test_sift_on_featureless_images_reports_an_error(
        self, presenter, fake_view, tmp_path
    ):
        blank = np.zeros((64, 64), dtype=np.uint8)
        src_path = tmp_path / "a.tif"
        dst_path = tmp_path / "b.tif"
        io.imsave(src_path, blank, check_contrast=False)
        io.imsave(dst_path, blank, check_contrast=False)

        presenter.load_source_image(src_path, modality_name="BSE")
        presenter.load_destination_image(dst_path, modality_name="SE")

        presenter.auto_detect_points("sift")
        assert fake_view.errors, "no-matches should surface as a view error"

    def test_checkpoint_path_round_trips(self, presenter, tmp_path):
        path = tmp_path / "model.ckpt"
        presenter.set_checkpoint_path(path)
        assert presenter.get_checkpoint_path() == str(path)
