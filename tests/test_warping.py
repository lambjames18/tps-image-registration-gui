"""Tests for the warping convenience layer."""

from __future__ import annotations

import numpy as np
import pytest

from tpsreg.tps import ThinPlateSplineTransform
from tpsreg.warping import (
    DEFAULT_TILE,
    MIN_TILE,
    _ShiftedTransform,
    _tile_for,
    _warp_tiled,
    get_transform,
    get_transform_params,
    set_transform_params,
    transform_coords,
    transform_image,
    transform_image_stack,
    warp,
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
        assert params.shape == (len(grid_points) + 3, 2)


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

        # Exactly preserved, not shifted. This used to be checked loosely on
        # the interior because the evaluation grid was 1-based and every warp
        # came out a pixel off in both axes.
        np.testing.assert_allclose(warped, image, atol=1e-6)

    def test_returns_params_when_asked(self, grid_points, checkerboard):
        warped, params = transform_image(
            checkerboard, grid_points, grid_points, return_params=True
        )
        assert warped.shape == checkerboard.shape
        assert params.shape == (len(grid_points) + 3, 2)


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


class TestTiledWarping:
    """Warping large outputs without a coordinate array for the whole thing."""

    @pytest.fixture
    def transform(self, rng):
        from tpsreg.tps import ThinPlateSplineTransform

        dst = rng.uniform(10, 110, size=(8, 2))
        src = dst + rng.normal(0, 4, size=(8, 2))
        tform = ThinPlateSplineTransform()
        tform.estimate(src, dst, (120, 150))
        return tform

    @pytest.mark.parametrize("order", [0, 1, 3])
    @pytest.mark.parametrize(
        "image_kind", ["gray_uint8", "gray_uint16", "gray_float", "rgb_uint8"]
    )
    def test_tiling_matches_skimage_exactly(self, transform, rng, order, image_kind):
        """The tiled path must be indistinguishable from warping in one pass.

        skimage's behaviour is not uniform -- at order 0 it preserves the input
        dtype and range, at higher orders it converts to float in [0, 1] and
        clips -- so this is checked across both, for every dtype the
        application loads.
        """
        from skimage import transform as tf

        images = {
            "gray_uint8": (rng.random((120, 150)) * 255).astype(np.uint8),
            "gray_uint16": (rng.random((120, 150)) * 65535).astype(np.uint16),
            "gray_float": rng.random((120, 150)),
            "rgb_uint8": (rng.random((120, 150, 3)) * 255).astype(np.uint8),
        }
        image = images[image_kind]

        reference = tf.warp(
            image,
            transform,
            output_shape=(120, 150),
            mode="constant",
            cval=0,
            order=order,
        )
        tiled = _warp_tiled(image, transform, (120, 150), order=order, cval=0, tile=32)

        assert tiled.dtype == reference.dtype
        np.testing.assert_array_equal(tiled, reference)

    @pytest.mark.parametrize("tile", [7, 32, 64, 4096])
    def test_the_tile_size_does_not_change_the_result(self, transform, rng, tile):
        """Tiling is a memory strategy; it must not be visible in the output.

        A tile that does not divide the output evenly is included on purpose:
        the edge tiles are the ones an off-by-one would show up in.
        """
        image = (rng.random((120, 150)) * 255).astype(np.uint8)

        reference = _warp_tiled(image, transform, (120, 150), tile=4096)
        np.testing.assert_array_equal(
            _warp_tiled(image, transform, (120, 150), tile=tile), reference
        )

    def test_warp_delegates_small_outputs_to_skimage(self, transform, rng):
        from skimage import transform as tf

        image = (rng.random((120, 150)) * 255).astype(np.uint8)
        np.testing.assert_array_equal(
            warp(image, transform, (120, 150)),
            tf.warp(
                image,
                transform,
                output_shape=(120, 150),
                mode="constant",
                cval=0,
                order=0,
            ),
        )

    def test_warp_defaults_to_the_input_shape(self, transform, rng):
        image = (rng.random((120, 150)) * 255).astype(np.uint8)
        assert warp(image, transform).shape == (120, 150)

    def test_an_offset_tile_maps_to_the_same_place(self, transform):
        """The tile's origin has to be added back before mapping.

        Forgetting it warps every tile as though it were the top-left one,
        which tiles the source image instead of warping it.
        """
        shifted = _ShiftedTransform(transform, 40, 25)
        local = np.array([[3.0, 4.0]])
        np.testing.assert_allclose(shifted(local), transform(np.array([[43.0, 29.0]])))


class TestTileSizing:
    """Choosing a tile when the caller does not."""

    def test_an_explicit_tile_is_respected(self):
        assert _tile_for(object(), 64) == 64

    def test_a_transform_without_control_points_gets_the_default(self):
        assert _tile_for(object(), None) == DEFAULT_TILE

    def test_more_control_points_means_smaller_tiles(self):
        """Work per tile scales with the control point count."""
        from tpsreg.tps import ThinPlateSplineTransform

        def tile_for_n(n):
            tform = ThinPlateSplineTransform()
            tform.control_points = np.zeros((n, 2))
            return _tile_for(tform, None)

        assert tile_for_n(1000) < tile_for_n(10)

    def test_the_tile_stays_within_its_bounds(self):
        from tpsreg.tps import ThinPlateSplineTransform

        for n in (1, 50, 5000, 100_000):
            tform = ThinPlateSplineTransform()
            tform.control_points = np.zeros((n, 2))
            assert MIN_TILE <= _tile_for(tform, None) <= DEFAULT_TILE
