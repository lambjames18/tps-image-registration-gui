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
