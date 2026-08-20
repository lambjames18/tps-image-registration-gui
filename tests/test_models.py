"""Tests for the data model and business logic layer."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tpsreg.models import (
    DataFormat,
    ImageData,
    ImageLoader,
    ImageProcessor,
    Point,
    PointAutoIdentifier,
    PointManager,
    PointSet,
    ProjectManager,
    TransformManager,
    TransformType,
    _read_numeric_table,
    _rescale_unit_interval,
)


class TestNumericTableReader:
    """The fast table reader used to parse .ang scans."""

    def test_reads_a_plain_table(self, tmp_path):
        path = tmp_path / "table.txt"
        path.write_text("1 2 3\n4 5 6\n")
        np.testing.assert_array_equal(
            _read_numeric_table(path), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        )

    def test_skips_header_lines(self, tmp_path):
        path = tmp_path / "table.txt"
        path.write_text("# header\n# more header\n7 8\n9 10\n")
        np.testing.assert_array_equal(
            _read_numeric_table(path, skip_header=2), [[7.0, 8.0], [9.0, 10.0]]
        )

    def test_single_row_is_still_two_dimensional(self, tmp_path):
        """load_ang indexes rows, so a one-row scan must not collapse to 1D."""
        path = tmp_path / "table.txt"
        path.write_text("1 2 3\n")
        assert _read_numeric_table(path).shape == (1, 3)

    def test_falls_back_when_values_are_missing(self, tmp_path):
        """loadtxt raises on ragged rows; genfromtxt fills them with NaN.

        Real scans occasionally carry blank entries, so the fallback is what
        keeps those files loadable.
        """
        path = tmp_path / "ragged.txt"
        path.write_text("1,2,3\n4,,6\n")

        # Comma-delimited with a hole: loadtxt cannot parse this at all.
        with pytest.raises(ValueError):
            np.loadtxt(path, dtype=float, ndmin=2)

        data = _read_numeric_table(path)
        assert data.shape[0] >= 1
        assert np.isnan(data).any(), "the fallback should surface holes as NaN"

    def test_nan_literals_are_preserved_for_the_caller_to_zero(self, tmp_path):
        path = tmp_path / "table.txt"
        path.write_text("1 nan 3\n")
        assert np.isnan(_read_numeric_table(path)).any()


class TestPoint:
    """The Point value object."""

    def test_defaults_to_two_element_array(self):
        np.testing.assert_array_equal(Point(3.0, 4.0).to_array(), [3.0, 4.0])

    def test_include_slice_prepends_index(self):
        point = Point(3.0, 4.0, slice_idx=7)
        np.testing.assert_array_equal(point.to_array(include_slice=True), [7, 3.0, 4.0])

    def test_slice_zero_still_yields_three_elements_when_requested(self):
        """Shape must follow the flag, not the slice value.

        Deriving it from slice_idx returned a 2-element array for points on
        slice 0 of a 3D stack, silently corrupting stack registration.
        """
        point = Point(1.0, 2.0, slice_idx=0)
        assert point.to_array(include_slice=True).shape == (3,)
        np.testing.assert_array_equal(point.to_array(include_slice=True), [0, 1.0, 2.0])

    def test_nonzero_slice_without_flag_yields_two_elements(self):
        assert Point(1.0, 2.0, slice_idx=5).to_array().shape == (2,)


class TestPointSet:
    """Per-slice point storage."""

    def test_add_and_retrieve_for_one_slice(self):
        point_set = PointSet()
        point_set.add_point(Point(1.0, 2.0, 0))
        point_set.add_point(Point(3.0, 4.0, 0))

        np.testing.assert_array_equal(
            point_set.get_points_array(0), [[1.0, 2.0], [3.0, 4.0]]
        )

    def test_missing_slice_returns_empty(self):
        assert PointSet().get_points_array(3).size == 0

    def test_all_slices_include_index_column(self):
        point_set = PointSet()
        point_set.add_point(Point(1.0, 2.0, 0))
        point_set.add_point(Point(5.0, 6.0, 2))

        all_points = point_set.get_points_array()
        assert all_points.shape == (2, 3)
        np.testing.assert_array_equal(all_points[:, 0], [0, 2])

    def test_remove_existing_point_reports_true(self):
        point_set = PointSet()
        point_set.add_point(Point(1.0, 2.0, 0))
        assert point_set.remove_point(0, 0) is True
        assert point_set.get_points_array(0).size == 0

    def test_remove_out_of_range_reports_false(self):
        """A no-op removal must not claim success."""
        point_set = PointSet()
        point_set.add_point(Point(1.0, 2.0, 0))
        assert point_set.remove_point(0, 99) is False
        assert len(point_set.points[0]) == 1

    def test_remove_from_missing_slice_reports_false(self):
        assert PointSet().remove_point(5, 0) is False

    def test_clear_single_slice_leaves_others(self):
        point_set = PointSet()
        point_set.add_point(Point(1.0, 2.0, 0))
        point_set.add_point(Point(3.0, 4.0, 1))

        point_set.clear(0)
        assert 0 not in point_set.points
        assert len(point_set.points[1]) == 1

    def test_clear_all(self):
        point_set = PointSet()
        point_set.add_point(Point(1.0, 2.0, 0))
        point_set.add_point(Point(3.0, 4.0, 1))
        point_set.clear()
        assert point_set.points == {}

    def test_dict_round_trip(self):
        original = PointSet()
        original.add_point(Point(1.5, 2.5, 0))
        original.add_point(Point(3.5, 4.5, 2))

        restored = PointSet.from_dict(original.to_dict())

        np.testing.assert_array_equal(
            restored.get_points_array(), original.get_points_array()
        )

    def test_move_existing_point_reports_true(self):
        point_set = PointSet()
        point_set.add_point(Point(1.0, 2.0, 0))
        point_set.add_point(Point(3.0, 4.0, 0))

        assert point_set.move_point(0, 1, 30.0, 40.0) is True
        np.testing.assert_array_equal(
            point_set.get_points_array(0), [[1.0, 2.0], [30.0, 40.0]]
        )

    def test_move_keeps_the_point_in_place_in_the_list(self):
        """Indices are how the two sides stay paired; moving must not reorder."""
        point_set = PointSet()
        for x in (1.0, 2.0, 3.0):
            point_set.add_point(Point(x, 0.0, 0))

        point_set.move_point(0, 0, 99.0, 99.0)
        np.testing.assert_array_equal(
            point_set.get_points_array(0), [[99.0, 99.0], [2.0, 0.0], [3.0, 0.0]]
        )

    def test_move_out_of_range_reports_false(self):
        point_set = PointSet()
        point_set.add_point(Point(1.0, 2.0, 0))
        assert point_set.move_point(0, 99, 5.0, 5.0) is False
        np.testing.assert_array_equal(point_set.get_points_array(0), [[1.0, 2.0]])

    def test_move_on_a_missing_slice_reports_false(self):
        assert PointSet().move_point(7, 0, 1.0, 1.0) is False

    def test_a_moved_point_keeps_its_slice(self):
        point_set = PointSet()
        point_set.add_point(Point(1.0, 2.0, 3))
        point_set.move_point(3, 0, 8.0, 9.0)
        assert point_set.points[3][0].slice_idx == 3


class TestPointManager:
    """Paired point management, undo and redo."""

    def test_add_pair_keeps_both_sides(self):
        manager = PointManager()
        manager.add_point_pair(Point(1.0, 2.0, 0), Point(3.0, 4.0, 0))

        src, dst = manager.get_point_pairs(0)
        np.testing.assert_array_equal(src, [[1.0, 2.0]])
        np.testing.assert_array_equal(dst, [[3.0, 4.0]])

    def test_remove_pair_removes_both_sides(self):
        manager = PointManager()
        manager.add_point_pair(Point(1.0, 2.0, 0), Point(3.0, 4.0, 0))

        assert manager.remove_point_pair(0, 0) is True
        src, dst = manager.get_point_pairs(0)
        assert src.size == 0
        assert dst.size == 0

    def test_remove_missing_pair_reports_false(self):
        manager = PointManager()
        manager.add_point_pair(Point(1.0, 2.0, 0), Point(3.0, 4.0, 0))
        assert manager.remove_point_pair(0, 5) is False

    def test_failed_removal_does_not_touch_undo_history(self):
        """A rejected removal must not consume an undo step."""
        manager = PointManager()
        manager.add_point_pair(Point(1.0, 2.0, 0), Point(3.0, 4.0, 0))
        manager.add_point_pair(Point(5.0, 6.0, 0), Point(7.0, 8.0, 0))

        manager.remove_point_pair(0, 99)
        manager.undo()

        # Undo should roll back the second add, not a phantom removal.
        src, _ = manager.get_point_pairs(0)
        assert len(src) == 1

    def test_undo_restores_previous_state(self):
        manager = PointManager()
        manager.add_point_pair(Point(1.0, 2.0, 0), Point(3.0, 4.0, 0))
        manager.add_point_pair(Point(5.0, 6.0, 0), Point(7.0, 8.0, 0))

        assert manager.undo() is True
        src, _ = manager.get_point_pairs(0)
        assert len(src) == 1

    def test_redo_reapplies_undone_change(self):
        manager = PointManager()
        manager.add_point_pair(Point(1.0, 2.0, 0), Point(3.0, 4.0, 0))
        manager.add_point_pair(Point(5.0, 6.0, 0), Point(7.0, 8.0, 0))

        manager.undo()
        assert manager.redo() is True
        src, _ = manager.get_point_pairs(0)
        assert len(src) == 2

    def test_undo_with_no_history_reports_false(self):
        assert PointManager().undo() is False


class TestPointManagerMoves:
    """Dragging a marker moves one side of a pair."""

    @staticmethod
    def _manager():
        manager = PointManager()
        manager.add_point_pair(Point(1.0, 2.0, 0), Point(3.0, 4.0, 0))
        manager.add_point_pair(Point(5.0, 6.0, 0), Point(7.0, 8.0, 0))
        return manager

    def test_moving_a_source_point(self):
        manager = self._manager()
        assert manager.move_point("source", 0, 1, 50.0, 60.0) is True

        src, _ = manager.get_point_pairs(0)
        np.testing.assert_array_equal(src, [[1.0, 2.0], [50.0, 60.0]])

    def test_moving_a_source_point_leaves_its_partner_alone(self):
        """The two sides are separate features; only the dragged one moves."""
        manager = self._manager()
        manager.move_point("source", 0, 1, 50.0, 60.0)

        _, dst = manager.get_point_pairs(0)
        np.testing.assert_array_equal(dst, [[3.0, 4.0], [7.0, 8.0]])

    def test_moving_a_destination_point(self):
        manager = self._manager()
        manager.move_point("destination", 0, 0, 30.0, 40.0)

        src, dst = manager.get_point_pairs(0)
        np.testing.assert_array_equal(dst, [[30.0, 40.0], [7.0, 8.0]])
        np.testing.assert_array_equal(src, [[1.0, 2.0], [5.0, 6.0]])

    def test_an_unknown_side_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown point set"):
            self._manager().move_point("middle", 0, 0, 1.0, 1.0)

    def test_moving_a_missing_point_reports_false(self):
        assert self._manager().move_point("source", 0, 99, 1.0, 1.0) is False

    def test_a_failed_move_does_not_consume_an_undo_step(self):
        manager = self._manager()
        manager.move_point("source", 0, 99, 1.0, 1.0)
        manager.undo()

        # The undo should roll back the second add, not a phantom move.
        src, _ = manager.get_point_pairs(0)
        assert len(src) == 1

    def test_a_move_can_be_undone(self):
        manager = self._manager()
        manager.move_point("source", 0, 1, 50.0, 60.0)

        assert manager.undo() is True
        src, _ = manager.get_point_pairs(0)
        np.testing.assert_array_equal(src, [[1.0, 2.0], [5.0, 6.0]])

    def test_an_undone_move_can_be_redone(self):
        manager = self._manager()
        manager.move_point("source", 0, 1, 50.0, 60.0)
        manager.undo()

        assert manager.redo() is True
        src, _ = manager.get_point_pairs(0)
        np.testing.assert_array_equal(src, [[1.0, 2.0], [50.0, 60.0]])

    def test_a_drag_is_a_single_undo_step(self):
        """A drag fires a move per mouse motion; undo must not replay each one.

        Only the first step of a gesture records history, so one undo returns
        the point to where it started rather than to the previous frame of the
        drag.
        """
        manager = self._manager()

        manager.move_point("source", 0, 1, 10.0, 10.0)  # first step: recorded
        for step in range(2, 40):
            manager.move_point(
                "source", 0, 1, float(step * 10), float(step * 10), record_history=False
            )

        assert manager.undo() is True
        src, _ = manager.get_point_pairs(0)
        np.testing.assert_array_equal(src[1], [5.0, 6.0])

    def test_a_long_drag_does_not_flood_the_history(self):
        """50 recorded steps would push every earlier edit off the stack."""
        manager = self._manager()
        depth_before = len(manager._history)

        manager.move_point("source", 0, 1, 10.0, 10.0)
        for step in range(200):
            manager.move_point(
                "source", 0, 1, float(step), float(step), record_history=False
            )

        assert len(manager._history) == depth_before + 1


class TestUndoRedoAvailability:
    """Whether the menu entries should be enabled."""

    def test_a_fresh_manager_can_do_neither(self):
        manager = PointManager()
        assert manager.can_undo() is False
        assert manager.can_redo() is False

    def test_after_an_edit_undo_is_available(self):
        manager = PointManager()
        manager.add_point_pair(Point(1.0, 2.0, 0), Point(3.0, 4.0, 0))
        assert manager.can_undo() is True
        assert manager.can_redo() is False

    def test_after_undoing_redo_is_available(self):
        manager = PointManager()
        manager.add_point_pair(Point(1.0, 2.0, 0), Point(3.0, 4.0, 0))
        manager.undo()
        assert manager.can_redo() is True

    def test_after_redoing_everything_redo_is_unavailable(self):
        manager = PointManager()
        manager.add_point_pair(Point(1.0, 2.0, 0), Point(3.0, 4.0, 0))
        manager.undo()
        manager.redo()
        assert manager.can_redo() is False

    def test_a_new_edit_invalidates_the_redo_branch(self):
        manager = PointManager()
        manager.add_point_pair(Point(1.0, 2.0, 0), Point(3.0, 4.0, 0))
        manager.add_point_pair(Point(5.0, 6.0, 0), Point(7.0, 8.0, 0))
        manager.undo()
        assert manager.can_redo() is True

        manager.add_point_pair(Point(9.0, 9.0, 0), Point(9.0, 9.0, 0))
        assert manager.can_redo() is False

    @pytest.mark.parametrize("edits", range(4))
    def test_availability_always_agrees_with_what_undo_does(self, edits):
        """The greyed-out state must never lie about what will happen."""
        manager = PointManager()
        for i in range(edits):
            manager.add_point_pair(Point(float(i), 0.0, 0), Point(float(i), 0.0, 0))

        for _ in range(edits + 2):
            expected = manager.can_undo()
            assert manager.undo() is expected

    @pytest.mark.parametrize("edits", range(1, 4))
    def test_redo_availability_agrees_with_what_redo_does(self, edits):
        manager = PointManager()
        for i in range(edits):
            manager.add_point_pair(Point(float(i), 0.0, 0), Point(float(i), 0.0, 0))
        while manager.undo():
            pass

        for _ in range(edits + 2):
            expected = manager.can_redo()
            assert manager.redo() is expected

    def test_clear_points_for_one_slice(self):
        manager = PointManager()
        manager.add_point_pair(Point(1.0, 2.0, 0), Point(3.0, 4.0, 0))
        manager.add_point_pair(Point(5.0, 6.0, 1), Point(7.0, 8.0, 1))

        manager.clear_points(0)
        assert manager.get_point_pairs(0).__getitem__(0).size == 0
        assert len(manager.get_point_pairs(1)[0]) == 1

    def test_save_and_load_round_trip(self, tmp_path):
        manager = PointManager()
        manager.add_point_pair(Point(1.0, 2.0, 0), Point(3.0, 4.0, 0))
        manager.add_point_pair(Point(5.0, 6.0, 0), Point(7.0, 8.0, 0))

        src_path = tmp_path / "src.txt"
        dst_path = tmp_path / "dst.txt"
        manager.save_to_file(src_path, dst_path)

        assert src_path.exists()
        assert dst_path.exists()

        restored = PointManager()
        restored.load_source_from_file(src_path)
        restored.load_destination_from_file(dst_path)

        np.testing.assert_allclose(
            restored.get_point_pairs(0)[0], manager.get_point_pairs(0)[0]
        )


class TestImageData:
    """The multimodal image container."""

    @pytest.fixture
    def image_data(self):
        return ImageData(
            data={"BSE": np.zeros((3, 20, 30, 1), dtype=np.uint8)},
            resolution=0.5,
            paths={"BSE": ["bse.tif"]},
            metadata={"dataformat": DataFormat.IMAGE.value},
        )

    def test_shape_comes_from_first_modality(self, image_data):
        assert image_data.shape == (3, 20, 30, 1)

    def test_modalities_listed(self, image_data):
        assert image_data.modalities == ["BSE"]

    def test_get_slice_returns_one_plane(self, image_data):
        assert image_data.get_slice("BSE", 1).shape == (20, 30, 1)

    def test_unknown_modality_raises(self, image_data):
        with pytest.raises(KeyError, match="not found"):
            image_data.get_slice("EBSD", 0)

    def test_slice_out_of_range_raises(self, image_data):
        with pytest.raises(IndexError, match="out of range"):
            image_data.get_slice("BSE", 99)

    def test_add_matching_modality(self, image_data):
        extra = ImageData(
            data={"SE": np.ones((3, 20, 30, 1), dtype=np.uint8)},
            resolution=0.5,
            paths={"SE": ["se.tif"]},
            metadata={"dataformat": DataFormat.IMAGE.value},
        )
        image_data.add_modality(extra)
        assert sorted(image_data.modalities) == ["BSE", "SE"]

    def test_add_modality_rejects_shape_mismatch(self, image_data):
        extra = ImageData(
            data={"SE": np.ones((3, 40, 30, 1), dtype=np.uint8)},
            resolution=0.5,
            paths={"SE": ["se.tif"]},
            metadata={"dataformat": DataFormat.IMAGE.value},
        )
        with pytest.raises(ValueError, match="does not match existing shape"):
            image_data.add_modality(extra)

    def test_add_modality_rejects_format_mismatch(self, image_data):
        extra = ImageData(
            data={"IQ": np.ones((3, 20, 30, 1), dtype=np.uint8)},
            resolution=0.5,
            paths={"IQ": ["scan.ang"]},
            metadata={"dataformat": DataFormat.ANG.value},
        )
        with pytest.raises(ValueError, match="format does not match"):
            image_data.add_modality(extra)

    def test_add_modality_rejects_multiple_at_once(self, image_data):
        extra = ImageData(
            data={
                "SE": np.ones((3, 20, 30, 1), dtype=np.uint8),
                "CI": np.ones((3, 20, 30, 1), dtype=np.uint8),
            },
            resolution=0.5,
            paths={"SE": ["se.tif"], "CI": ["ci.tif"]},
            metadata={"dataformat": DataFormat.IMAGE.value},
        )
        with pytest.raises(ValueError, match="exactly one modality"):
            image_data.add_modality(extra)


class TestImageLoader:
    """Loading image files."""

    @pytest.fixture
    def tiff_path(self, tmp_path, checkerboard):
        from skimage import io

        path = tmp_path / "image.tif"
        io.imsave(path, checkerboard, check_contrast=False)
        return path

    def test_loads_tiff_into_stack_layout(self, tiff_path):
        data = ImageLoader.load(tiff_path, modality_name="BSE")
        assert data.modalities == ["BSE"]
        # Always (slices, height, width, channels), even for a single 2D image.
        assert data.data["BSE"].ndim == 4
        assert data.data["BSE"].shape[0] == 1

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ImageLoader.load(tmp_path / "absent.tif")

    def test_unsupported_extension_raises(self, tmp_path):
        path = tmp_path / "notes.xyz"
        path.write_text("nope")
        with pytest.raises(ValueError, match="Unsupported file format"):
            ImageLoader.load(path)

    def test_empty_path_list_raises(self):
        with pytest.raises(ValueError, match="empty list"):
            ImageLoader.load([])

    def test_mixed_extensions_rejected(self, tmp_path, checkerboard):
        from skimage import io

        tif = tmp_path / "a.tif"
        png = tmp_path / "b.png"
        io.imsave(tif, checkerboard, check_contrast=False)
        io.imsave(png, checkerboard, check_contrast=False)

        with pytest.raises(ValueError, match="same extension"):
            ImageLoader.load([tif, png])

    def test_loads_a_stack_from_multiple_files(self, tmp_path, checkerboard):
        from skimage import io

        paths = []
        for i in range(3):
            path = tmp_path / f"slice_{i}.tif"
            io.imsave(path, checkerboard, check_contrast=False)
            paths.append(path)

        data = ImageLoader.load(paths, modality_name="BSE")
        assert data.data["BSE"].shape[0] == 3

    def test_single_element_list_behaves_like_a_bare_path(self, tiff_path):
        data = ImageLoader.load([tiff_path], modality_name="BSE")
        assert data.data["BSE"].shape[0] == 1

    def test_constant_image_does_not_produce_nan(self, tmp_path):
        """A blank slice has zero dynamic range; normalization must survive it."""
        from skimage import io

        path = tmp_path / "flat.tif"
        io.imsave(path, np.full((16, 16), 42, dtype=np.uint8), check_contrast=False)

        data = ImageLoader.load(path, modality_name="BSE")
        assert np.all(np.isfinite(data.data["BSE"].astype(float)))


class TestImageProcessor:
    """Image processing helpers."""

    def test_normalize_spans_full_uint8_range(self, rng):
        image = rng.random((32, 32)) * 1000
        result = ImageProcessor.normalize_to_uint8(image)

        assert result.dtype == np.uint8
        assert result.min() == 0
        assert result.max() == 255

    def test_normalize_passes_through_uint8(self, checkerboard):
        assert ImageProcessor.normalize_to_uint8(checkerboard) is checkerboard

    def test_normalize_constant_image_yields_zeros(self):
        """Constant input divides by zero; the result must be defined."""
        result = ImageProcessor.normalize_to_uint8(np.full((8, 8), 7.0))
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, np.zeros((8, 8), dtype=np.uint8))

    def test_normalize_handles_constant_channel(self, rng):
        """One flat channel must not poison the others with NaN."""
        image = np.empty((8, 8, 2))
        image[..., 0] = rng.random((8, 8)) * 100
        image[..., 1] = 5.0

        result = ImageProcessor.normalize_to_uint8(image)
        np.testing.assert_array_equal(result[..., 1], np.zeros((8, 8), dtype=np.uint8))
        assert result[..., 0].max() == 255

    def test_resize_by_one_is_a_noop(self, checkerboard):
        assert ImageProcessor.resize_image(checkerboard, 1.0) is checkerboard

    def test_resize_halves_spatial_dimensions(self, checkerboard):
        result = ImageProcessor.resize_image(checkerboard, 0.5)
        assert result.shape == (32, 32)

    def test_resize_preserves_channel_axis(self, rng):
        image = (rng.random((20, 30, 3)) * 255).astype(np.uint8)
        result = ImageProcessor.resize_image(image, 2.0)
        assert result.shape == (40, 60, 3)

    def test_resize_preserves_stack_axis(self, rng):
        stack = (rng.random((4, 20, 30, 1)) * 255).astype(np.uint8)
        result = ImageProcessor.resize_image(stack, 0.5)
        assert result.shape == (4, 10, 15, 1)

    def test_negative_scale_rejected(self, checkerboard):
        with pytest.raises(ValueError, match="must be positive"):
            ImageProcessor.resize_image(checkerboard, -1.0)

    def test_clahe_preserves_shape_and_dtype(self, checkerboard):
        result = ImageProcessor.apply_clahe(checkerboard)
        assert result.shape == checkerboard.shape
        assert result.dtype == np.uint8

    def test_clahe_on_constant_image_yields_zeros(self):
        result = ImageProcessor.apply_clahe(np.full((32, 32), 100, dtype=np.uint8))
        assert result.shape == (32, 32)
        np.testing.assert_array_equal(result, np.zeros((32, 32), dtype=np.uint8))

    def test_clahe_increases_contrast(self):
        """A low-contrast image should come back using more of the range."""
        image = np.zeros((64, 64), dtype=np.uint8)
        image[16:48, 16:48] = 10
        image[24:40, 24:40] = 20

        result = ImageProcessor.apply_clahe(image)
        assert np.ptp(result) > np.ptp(image)

    def test_clahe_handles_stack(self, rng):
        stack = (rng.random((2, 32, 32, 1)) * 255).astype(np.uint8)
        result = ImageProcessor.apply_clahe(stack)
        assert result.shape == stack.shape


class TestRescaleHelper:
    """The shared unit-interval rescaling helper."""

    def test_maps_to_zero_one(self, rng):
        result = _rescale_unit_interval(rng.random((10, 10)) * 500 - 100)
        assert result.min() == pytest.approx(0.0)
        assert result.max() == pytest.approx(1.0)

    def test_constant_input_maps_to_zeros(self):
        np.testing.assert_array_equal(
            _rescale_unit_interval(np.full((4, 4), 3.0)), np.zeros((4, 4))
        )


class TestTransformManager:
    """Transform estimation and application."""

    @pytest.fixture
    def points(self):
        xs, ys = np.meshgrid(np.linspace(5, 55, 4), np.linspace(5, 55, 4))
        return np.column_stack([xs.ravel(), ys.ravel()]).astype(float)

    def test_empty_points_rejected(self):
        manager = TransformManager()
        with pytest.raises(ValueError, match="cannot be empty"):
            manager.estimate_transform(
                np.array([]), np.array([]), TransformType.TPS, (10, 10)
            )

    def test_point_count_mismatch_rejected(self, points):
        manager = TransformManager()
        with pytest.raises(ValueError, match="mismatch"):
            manager.estimate_transform(points, points[:-1], TransformType.TPS, (64, 64))

    def test_estimate_returns_usable_transform(self, points):
        manager = TransformManager()
        tform = manager.estimate_transform(points, points, TransformType.TPS, (64, 64))
        assert tform._estimated
        # The transform is its coefficients, not a field over the grid.
        assert tform.params.shape == (len(points) + 3, 2)
        assert tform.size == (64, 64)

    def test_only_the_spline_is_offered(self, points):
        """The affine-only variant is gone; TPS is the whole enum."""
        assert list(TransformType) == [TransformType.TPS]

    def test_apply_transform_matches_output_shape(self, points, checkerboard):
        manager = TransformManager()
        tform = manager.estimate_transform(points, points, TransformType.TPS, (64, 64))
        warped = manager.apply_transform(checkerboard, tform, (64, 64))
        assert warped.shape == (64, 64)

    def test_stack_estimate_without_n_slices(self, points):
        """n_slices defaults to the point extent rather than crashing on None."""
        manager = TransformManager()
        src = np.column_stack([np.zeros(len(points)), points])
        src = np.vstack([src, np.column_stack([np.full(len(points), 2), points])])
        dst = src.copy()

        transforms = manager.estimate_transform_stack(
            src, dst, TransformType.TPS, (32, 32)
        )
        assert set(transforms) == {0, 1, 2}

    def test_stack_estimate_interpolates_missing_slices(self, points):
        manager = TransformManager()
        src = np.vstack(
            [
                np.column_stack([np.zeros(len(points)), points]),
                np.column_stack([np.full(len(points), 4), points]),
            ]
        )
        dst = src.copy()

        transforms = manager.estimate_transform_stack(
            src, dst, TransformType.TPS, (32, 32), n_slices=5
        )
        assert set(transforms) == {0, 1, 2, 3, 4}
        # Keyed slices keep their coefficients; interpolated ones carry a
        # blended field, because coefficients fitted to different control
        # points cannot be averaged.
        for index, tform in transforms.items():
            assert tform(np.array([[10.0, 10.0]])).shape == (1, 2)
            if index in (0, 4):
                assert tform.params is not None
            else:
                assert tform.field.shape == (2, 32, 32)

    def test_export_transform_npy(self, points, tmp_path):
        manager = TransformManager()
        tform = manager.estimate_transform(points, points, TransformType.TPS, (16, 16))

        path = tmp_path / "tform.npy"
        manager.export_transform(tform, path, format="npy")
        assert np.load(path).shape == (len(points) + 3, 2)

    def test_export_unsupported_format_raises(self, points, tmp_path):
        manager = TransformManager()
        tform = manager.estimate_transform(points, points, TransformType.TPS, (16, 16))
        with pytest.raises(ValueError, match="Unsupported export format"):
            manager.export_transform(tform, tmp_path / "t.xyz", format="xyz")


class TestProjectManager:
    """Project serialization."""

    def test_save_writes_readable_json(self, tmp_path):
        manager = ProjectManager()
        points = PointManager()
        points.add_point_pair(Point(1.0, 2.0, 0), Point(3.0, 4.0, 0))

        path = tmp_path / "project.json"
        manager.save_project(path, points, {"resolution": 1.5})

        payload = json.loads(path.read_text())
        assert payload["settings"]["resolution"] == 1.5
        assert payload["source_points"]["0"] == [[1.0, 2.0]]

    def test_save_clears_modified_flag(self, tmp_path):
        manager = ProjectManager()
        manager.mark_modified()
        assert manager.is_modified

        manager.save_project(tmp_path / "p.json", PointManager(), {})
        assert not manager.is_modified

    def test_round_trip(self, tmp_path):
        manager = ProjectManager()
        points = PointManager()
        points.add_point_pair(Point(1.0, 2.0, 0), Point(3.0, 4.0, 0))

        path = tmp_path / "project.json"
        manager.save_project(path, points, {"mode": "tps"})

        loaded = ProjectManager().load_project(path)
        assert loaded["settings"] == {"mode": "tps"}

        restored = PointSet.from_dict(loaded["source_points"])
        np.testing.assert_array_equal(restored.get_points_array(0), [[1.0, 2.0]])

    def test_reset_clears_state(self, tmp_path):
        manager = ProjectManager()
        manager.save_project(tmp_path / "p.json", PointManager(), {})
        manager.mark_modified()

        manager.reset()
        assert manager.project_path is None
        assert not manager.is_modified

    def test_loading_missing_file_raises(self, tmp_path):
        with pytest.raises(OSError):
            ProjectManager().load_project(tmp_path / "absent.json")


class TestPointAutoIdentifier:
    """Automatic point detection dispatch."""

    def test_unknown_method_lists_the_valid_ones(self):
        with pytest.raises(ValueError, match="Available methods"):
            PointAutoIdentifier.detect_points(
                np.zeros((8, 8)), np.zeros((8, 8)), method="telepathy"
            )

    def test_checkpoint_path_in_kwargs_does_not_collide(self, monkeypatch):
        """Passing checkpoint_path used to raise 'got multiple values'."""
        captured = {}

        def fake_detect(source, destination, checkpoint_path=None, **kwargs):
            captured["checkpoint_path"] = checkpoint_path
            captured["kwargs"] = kwargs
            return np.array([]), np.array([])

        monkeypatch.setattr(
            PointAutoIdentifier, "detect_points_matchanything", fake_detect
        )

        PointAutoIdentifier.detect_points(
            np.zeros((8, 8)),
            np.zeros((8, 8)),
            method="matchanything",
            checkpoint_path="/models/roma.ckpt",
            num_samples=25,
        )

        assert captured["checkpoint_path"] == "/models/roma.ckpt"
        assert captured["kwargs"] == {"num_samples": 25}

    def test_class_checkpoint_used_when_none_supplied(self, monkeypatch):
        captured = {}

        def fake_detect(source, destination, checkpoint_path=None, **kwargs):
            captured["checkpoint_path"] = checkpoint_path
            return np.array([]), np.array([])

        monkeypatch.setattr(
            PointAutoIdentifier, "detect_points_matchanything", fake_detect
        )
        monkeypatch.setattr(PointAutoIdentifier, "checkpoint_path", "/default.ckpt")

        PointAutoIdentifier.detect_points(
            np.zeros((8, 8)), np.zeros((8, 8)), method="matchanything"
        )
        assert captured["checkpoint_path"] == "/default.ckpt"

    def test_sift_on_featureless_images_returns_empty(self):
        """No keypoints is a normal outcome, not an exception."""
        blank = np.zeros((64, 64), dtype=np.uint8)
        src, dst = PointAutoIdentifier.detect_points(blank, blank, method="sift")
        assert src.size == 0
        assert dst.size == 0

    def test_setting_checkpoint_clears_cached_matcher(self):
        PointAutoIdentifier._matchanything_matcher = object()
        try:
            PointAutoIdentifier.set_checkpoint_path("/new/path.ckpt")
            assert PointAutoIdentifier._matchanything_matcher is None
            assert PointAutoIdentifier.checkpoint_path == "/new/path.ckpt"
        finally:
            PointAutoIdentifier.checkpoint_path = None
            PointAutoIdentifier._matchanything_matcher = None
