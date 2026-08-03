"""Tests for the warping convenience layer."""

from __future__ import annotations

import numpy as np
import pytest

from tpsreg.tps import ThinPlateSplineTransform
from tpsreg.warping import (
    get_transform,
    get_transform_params,
    set_transform_params,
    transform_coords,
    transform_image,
    transform_image_stack,
)


@pytest.fixture
def grid_points():
    xs, ys = np.meshgrid(np.linspace(5, 55, 4), np.linspace(5, 55, 4))
    return np.column_stack([xs.ravel(), ys.ravel()]).astype(float)


class TestGetTransform:
    """Selecting a transform by name."""

    def test_tps_mode(self, grid_points):
        tform = get_transform(grid_points, grid_points, "tps", (64, 64))
        assert isinstance(tform, ThinPlateSplineTransform)
        assert not tform.affine_only

    def test_tps_affine_mode(self, grid_points):
        tform = get_transform(grid_points, grid_points, "tps affine", (64, 64))
        assert isinstance(tform, ThinPlateSplineTransform)
        assert tform.affine_only

    def test_mode_is_case_insensitive(self, grid_points):
        tform = get_transform(grid_points, grid_points, "TPS", (64, 64))
        assert isinstance(tform, ThinPlateSplineTransform)

    def test_skimage_modes_delegate(self, grid_points):
        from skimage.transform import AffineTransform

        tform = get_transform(grid_points, grid_points + 3, "affine")
        assert isinstance(tform, AffineTransform)

    def test_params_round_trip(self, grid_points):
        tform = get_transform(grid_points, grid_points, "tps", (32, 32))
        params = get_transform_params(tform)

        restored = ThinPlateSplineTransform()
        set_transform_params(restored, params)

        assert restored._estimated
        np.testing.assert_array_equal(restored.params, params)


class TestTransformCoords:
    """Mapping coordinates."""

    def test_returns_one_point_per_input(self, grid_points):
        result = transform_coords(grid_points, grid_points, mode="tps", size=(64, 64))
        assert result.shape == grid_points.shape

    def test_can_also_return_params(self, grid_points):
        result, params = transform_coords(
            grid_points, grid_points, mode="tps", return_params=True, size=(64, 64)
        )
        assert result.shape == grid_points.shape
        assert params.shape == (2, 64, 64)


class TestTransformImage:
    """Warping a single image."""

    def test_output_defaults_to_input_shape(self, grid_points, checkerboard):
        warped = transform_image(checkerboard, grid_points, grid_points)
        assert warped.shape == checkerboard.shape

    def test_explicit_output_shape_is_honoured(self, grid_points, checkerboard):
        warped = transform_image(
            checkerboard, grid_points, grid_points, output_shape=(40, 50)
        )
        assert warped.shape == (40, 50)

    def test_identity_points_roughly_preserve_the_image(self, grid_points):
        """A no-op warp should not move content around."""
        image = np.zeros((64, 64), dtype=float)
        image[20:44, 20:44] = 1.0

        warped = transform_image(image, grid_points, grid_points, order=1)

        # The grid is 1-based, so the result is shifted by one pixel; compare
        # the interiors rather than demanding an exact match.
        assert warped[21:43, 21:43].mean() > 0.9
        assert warped[:15, :15].mean() < 0.1

    def test_returns_params_when_asked(self, grid_points, checkerboard):
        warped, params = transform_image(
            checkerboard, grid_points, grid_points, return_params=True
        )
        assert warped.shape == checkerboard.shape
        assert params.shape == (2, *checkerboard.shape)


class TestTransformImageStack:
    """Warping a stack with interpolation between keyed slices."""

    @pytest.fixture
    def stack(self):
        images = np.zeros((5, 32, 32), dtype=float)
        images[:, 8:24, 8:24] = 1.0
        return images

    @pytest.fixture
    def stack_points(self, grid_points):
        """Points on the first and last slice only."""
        scaled = grid_points / 2  # fit inside a 32x32 slice
        first = np.column_stack([np.zeros(len(scaled)), scaled])
        last = np.column_stack([np.full(len(scaled), 4), scaled])
        return np.vstack([first, last])

    def test_output_matches_input_stack_length(self, stack, stack_points):
        warped = transform_image_stack(stack, stack_points, stack_points)
        assert warped.shape[0] == stack.shape[0]

    def test_output_shape_is_honoured(self, stack, stack_points):
        warped = transform_image_stack(
            stack, stack_points, stack_points, output_shape=(20, 24)
        )
        assert warped.shape == (5, 20, 24)

    def test_params_cover_every_slice(self, stack, stack_points):
        _, params = transform_image_stack(
            stack, stack_points, stack_points, return_params=True
        )
        assert params.shape[0] == stack.shape[0]
        assert np.all(np.isfinite(params))

    def test_interpolated_slices_lie_between_their_neighbours(
        self, stack, stack_points, grid_points
    ):
        """Middle slices must be interpolated, not copied from a neighbour."""
        scaled = grid_points / 2
        first = np.column_stack([np.zeros(len(scaled)), scaled])
        # Give the last slice a different warp so interpolation is observable.
        last = np.column_stack([np.full(len(scaled), 4), scaled + 3.0])
        srcs = np.vstack([first, last])
        dsts = np.vstack(
            [
                np.column_stack([np.zeros(len(scaled)), scaled]),
                np.column_stack([np.full(len(scaled), 4), scaled]),
            ]
        )

        _, params = transform_image_stack(stack, srcs, dsts, return_params=True)

        # Slice 2 sits halfway between the two keyed slices.
        midpoint = (params[0] + params[4]) / 2
        np.testing.assert_allclose(params[2], midpoint, rtol=1e-4, atol=1e-4)

    def test_unkeyed_first_slice_is_extended(self, stack, grid_points):
        """Points only on later slices must not break the interpolation."""
        scaled = grid_points / 2
        srcs = np.vstack(
            [
                np.column_stack([np.full(len(scaled), 1), scaled]),
                np.column_stack([np.full(len(scaled), 3), scaled]),
            ]
        )
        warped = transform_image_stack(stack, srcs, srcs.copy())
        assert warped.shape[0] == 5
        assert np.all(np.isfinite(warped))

    def test_precomputed_params_skip_estimation(self, stack, stack_points):
        _, params = transform_image_stack(
            stack, stack_points, stack_points, return_params=True
        )
        reused = transform_image_stack(stack, None, None, params=params)
        assert reused.shape[0] == stack.shape[0]

    def test_no_points_at_all_raises(self, stack):
        empty = np.empty((0, 3))
        with pytest.raises(ValueError, match="No control points"):
            transform_image_stack(stack, empty, empty)
