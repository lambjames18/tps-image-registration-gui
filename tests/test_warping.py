"""Tests for the warping convenience layer."""

from __future__ import annotations

import numpy as np
import pytest
from skimage import transform as tf

from tpsreg.tps import ThinPlateSplineTransform
from tpsreg.warping import (
    DEFAULT_TILE,
    MIN_TILE,
    _ShiftedTransform,
    _tile_for,
    _warp_tiled,
    get_transform,
    get_transform_params,
    homography_matrix,
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
        assert tform._estimated

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


class TestMatrixFastPath:
    """Transforms that are really a matrix should be handed over as one.

    skimage then warps in Cython, computing source coordinates as it goes
    rather than through a Python call per tile. Measured at 8192x8192 this was
    101 s against 3.8 s, so the thing worth guarding is that the shortcut
    produces the same picture and is only taken when it is valid.
    """

    @pytest.fixture
    def image(self, rng):
        return (rng.random((120, 150)) * 255).astype(np.uint8)

    @pytest.mark.parametrize(
        "tform",
        [
            tf.EuclideanTransform(rotation=0.05, translation=(3.0, -2.0)),
            tf.SimilarityTransform(scale=1.1, translation=(4.0, 1.0)),
            tf.AffineTransform(scale=(1.05, 0.95), shear=0.02),
            tf.ProjectiveTransform(
                matrix=np.array([[1.01, 0.02, 3.0], [0.0, 0.99, -2.0], [0.0, 0.0, 1.0]])
            ),
        ],
    )
    def test_it_is_recognised(self, tform):
        matrix = homography_matrix(tform)
        assert matrix is not None
        np.testing.assert_allclose(matrix, tform.params)

    def test_a_bare_matrix_is_recognised(self):
        matrix = np.eye(3)
        np.testing.assert_array_equal(homography_matrix(matrix), matrix)

    def test_a_spline_is_not(self, grid_points):
        """The important negative: a spline must never be taken for a matrix."""
        tform = ThinPlateSplineTransform()
        tform.estimate(grid_points + 1.0, grid_points, (64, 64))
        assert homography_matrix(tform) is None

    def test_something_unrelated_is_not(self):
        assert homography_matrix(object()) is None
        assert homography_matrix(np.zeros((2, 2))) is None

    @pytest.mark.parametrize("order", [1, 3])
    def test_it_matches_skimage(self, image, order):
        """The contract: warping through a transform equals skimage's answer."""
        tform = tf.EuclideanTransform(rotation=0.05, translation=(3.0, -2.0))

        np.testing.assert_array_equal(
            warp(image, tform, (120, 150), order=order),
            tf.warp(
                image,
                tform,
                output_shape=(120, 150),
                mode="constant",
                cval=0,
                order=order,
            ),
        )

    def test_bilinear_agrees_with_the_general_path(self, image):
        """At order 1 the shortcut is the same arithmetic, so it must agree.

        Not asserted at order 3: skimage's Cython path interpolates with a
        cubic convolution while its scipy path uses a prefiltered B-spline, and
        the two genuinely differ by ~0.1 of a normalised level. That is
        skimage's inconsistency, not ours -- see the size test below.
        """

        class Opaque:
            """Hides the matrix, forcing the general path."""

            def __init__(self, tform):
                self._tform = tform

            def __call__(self, coords):
                return self._tform(coords)

        tform = tf.EuclideanTransform(rotation=0.05, translation=(3.0, -2.0))

        np.testing.assert_allclose(
            warp(image, tform, (120, 150), order=1),
            warp(image, Opaque(tform), (120, 150), order=1),
            atol=1e-12,
        )

    @pytest.mark.parametrize("order", [1, 3])
    def test_the_image_size_does_not_change_the_interpolation(
        self, image, order, monkeypatch
    ):
        """A matrix transform must warp the same way at every output size.

        It used not to. Outputs under the tiling threshold went to skimage
        whole, which takes its Cython path for a matrix; larger ones were tiled
        through a callable, which does not. At order 3 those two disagree, so
        the same transform on the same data gave different pixels either side
        of 4 Mpx. Handing the matrix over directly is what removes that.
        """
        tform = tf.EuclideanTransform(rotation=0.05, translation=(3.0, -2.0))
        small = warp(image, tform, (120, 150), order=order)

        monkeypatch.setattr("tpsreg.warping.TILING_THRESHOLD", 0)
        as_if_huge = warp(image, tform, (120, 150), order=order)

        np.testing.assert_array_equal(small, as_if_huge)

    def test_order_zero_still_works(self, image):
        """Order 0 has no Cython path, so it must fall through, not break."""
        tform = tf.EuclideanTransform(translation=(4.0, -3.0))
        result = warp(image, tform, (120, 150), order=0)

        assert result.dtype == image.dtype
        np.testing.assert_array_equal(
            result,
            tf.warp(
                image,
                tform,
                output_shape=(120, 150),
                mode="constant",
                cval=0,
                order=0,
            ),
        )


class TestClipping:
    """Clipping is hoisted out of the tile loop, so it must still happen.

    skimage clips every warp call to the input's range, and its clip scans the
    whole input image -- per tile that is a full pass over the source for every
    tile. Doing it once is worth 15x at 8192x8192, but only if the result is
    unchanged, which is what these check.
    """

    @pytest.fixture
    def transform(self, rng):
        dst = rng.uniform(10, 110, size=(8, 2))
        tform = ThinPlateSplineTransform()
        tform.estimate(dst + rng.normal(0, 4, size=(8, 2)), dst, (120, 150))
        return tform

    def test_cubic_overshoot_is_clipped_away(self, transform, rng):
        """Order 3 rings past the input range at any sharp edge.

        Without clipping a bright edge comes back brighter than anything that
        was ever in the image, which is exactly what the clip exists to stop.
        """
        image = np.zeros((120, 150))
        image[40:80, 50:100] = 1.0

        tiled = _warp_tiled(image, transform, (120, 150), order=3, cval=0, tile=32)

        assert tiled.max() <= image.max() + 1e-12
        assert tiled.min() >= image.min() - 1e-12

    def test_a_cval_outside_the_input_range_survives(self, transform):
        """skimage widens the range rather than clipping the fill value away.

        With a fill above everything in the image, a plain clip would drag the
        background down to the image maximum and lose the distinction between
        "outside" and "bright".
        """
        image = np.full((120, 150), 0.5)

        tiled = _warp_tiled(image, transform, (200, 200), order=1, cval=9.0, tile=32)
        reference = tf.warp(
            image,
            transform,
            output_shape=(200, 200),
            mode="constant",
            cval=9.0,
            order=1,
        )

        np.testing.assert_array_equal(tiled, reference)
        assert tiled.max() == pytest.approx(9.0)


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
