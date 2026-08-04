"""Tests for the preview compositing modes.

Array in, array out. Keeping these out of the view is what lets the
checkerboard and difference views be checked without opening a window.
"""

from __future__ import annotations

import numpy as np
import pytest

from tpsreg.overlays import (
    BLEND_MODES,
    DEFAULT_TILE_SIZE,
    checkerboard,
    composite,
    difference,
    to_rgb,
    wipe,
)


@pytest.fixture
def black():
    return np.zeros((64, 64), dtype=np.uint8)


@pytest.fixture
def white():
    return np.full((64, 64), 255, dtype=np.uint8)


class TestToRgb:
    """Getting anything the application produces into (H, W, 3) uint8."""

    def test_grayscale_becomes_three_identical_channels(self):
        gray = np.arange(16, dtype=np.uint8).reshape(4, 4)
        rgb = to_rgb(gray)
        assert rgb.shape == (4, 4, 3)
        for channel in range(3):
            np.testing.assert_array_equal(rgb[:, :, channel], gray)

    def test_single_channel_is_expanded(self):
        image = np.zeros((4, 4, 1), dtype=np.uint8)
        assert to_rgb(image).shape == (4, 4, 3)

    def test_rgb_passes_through(self):
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        np.testing.assert_array_equal(to_rgb(image), image)

    def test_rgba_loses_its_alpha(self):
        """These composites do their own blending; a stray alpha would fight it."""
        image = np.zeros((4, 4, 4), dtype=np.uint8)
        image[..., 3] = 128
        assert to_rgb(image).shape == (4, 4, 3)

    def test_floats_are_treated_as_zero_to_one(self):
        """The package's own images are normalised floats."""
        image = np.array([[0.0, 0.5, 1.0]], dtype=np.float64)
        rgb = to_rgb(image)
        assert rgb.dtype == np.uint8
        np.testing.assert_array_equal(rgb[0, :, 0], [0, 127, 255])

    def test_float32_is_handled_too(self):
        image = np.ones((2, 2), dtype=np.float32)
        assert to_rgb(image).max() == 255

    def test_out_of_range_floats_are_clipped_not_wrapped(self):
        """Wrapping would turn an overbright pixel black."""
        image = np.array([[-0.5, 1.5]], dtype=np.float64)
        np.testing.assert_array_equal(to_rgb(image)[0, :, 0], [0, 255])

    def test_wide_integers_are_clipped(self):
        image = np.array([[-20, 300]], dtype=np.int16)
        np.testing.assert_array_equal(to_rgb(image)[0, :, 0], [0, 255])

    def test_a_1d_array_is_rejected(self):
        with pytest.raises(ValueError, match="2D or 3D"):
            to_rgb(np.zeros(4, dtype=np.uint8))

    def test_an_odd_channel_count_is_rejected(self):
        with pytest.raises(ValueError, match="channel count"):
            to_rgb(np.zeros((4, 4, 5), dtype=np.uint8))


class TestWipe:
    """The original slider behaviour, as an array operation."""

    def test_alpha_of_one_shows_only_the_overlay(self, black, white):
        result = wipe(white, black, np.ones((64, 64)))
        assert result.min() == 255

    def test_alpha_of_zero_shows_only_the_background(self, black, white):
        result = wipe(white, black, np.zeros((64, 64)))
        assert result.max() == 0

    def test_a_half_alpha_lands_halfway(self, black, white):
        result = wipe(white, black, np.full((64, 64), 0.5))
        assert result.mean() == pytest.approx(127.5, abs=1)

    def test_the_mask_can_vary_across_the_image(self, black, white):
        alphas = np.zeros((64, 64))
        alphas[:32] = 1.0
        result = wipe(white, black, alphas)
        assert result[:32].min() == 255
        assert result[32:].max() == 0

    def test_out_of_range_alphas_are_clipped(self, black, white):
        result = wipe(white, black, np.full((64, 64), 5.0))
        assert result.max() == 255

    def test_output_is_uint8(self, black, white):
        assert wipe(white, black, np.full((64, 64), 0.3)).dtype == np.uint8


class TestCheckerboard:
    """Interleaved squares, for spotting a whole-image offset."""

    def test_alternating_tiles_come_from_alternating_images(self, black, white):
        result = checkerboard(white, black, tile_size=16)
        assert result[0, 0, 0] == 255  # tile (0, 0) -> overlay
        assert result[0, 16, 0] == 0  # tile (0, 1) -> background
        assert result[16, 0, 0] == 0  # tile (1, 0) -> background
        assert result[16, 16, 0] == 255  # tile (1, 1) -> overlay

    def test_both_images_are_equally_represented(self, black, white):
        """An even split, so neither image is favoured."""
        result = checkerboard(white, black, tile_size=16)
        assert result.mean() == pytest.approx(127.5, abs=1)

    def test_a_larger_tile_makes_larger_squares(self, black, white):
        result = checkerboard(white, black, tile_size=32)
        assert result[0, 31, 0] == 255
        assert result[0, 32, 0] == 0

    def test_a_tile_of_one_alternates_every_pixel(self, black, white):
        result = checkerboard(white, black, tile_size=1)
        assert result[0, 0, 0] == 255
        assert result[0, 1, 0] == 0

    def test_a_zero_tile_does_not_divide_by_zero(self, black, white):
        """The spinbox is bounded, but a project file need not be."""
        assert checkerboard(white, black, tile_size=0).shape == (64, 64, 3)

    def test_a_negative_tile_is_survivable(self, black, white):
        assert checkerboard(white, black, tile_size=-4).shape == (64, 64, 3)

    def test_identical_images_produce_a_seamless_result(self):
        rng = np.random.default_rng(0)
        image = (rng.random((64, 64)) * 255).astype(np.uint8)
        result = checkerboard(image, image, tile_size=8)
        np.testing.assert_array_equal(result, to_rgb(image))


class TestDifference:
    """Absolute difference: black means aligned."""

    def test_identical_images_are_black(self):
        rng = np.random.default_rng(0)
        image = (rng.random((32, 32)) * 255).astype(np.uint8)
        assert difference(image, image).max() == 0

    def test_opposites_are_white(self, black, white):
        assert difference(white, black).min() == 255

    def test_the_result_does_not_depend_on_the_order(self, black, white):
        np.testing.assert_array_equal(
            difference(white, black), difference(black, white)
        )

    def test_a_small_difference_stays_small(self):
        """int8 arithmetic would wrap 250 - 10 into something nonsensical."""
        a = np.full((8, 8), 250, dtype=np.uint8)
        b = np.full((8, 8), 10, dtype=np.uint8)
        assert difference(a, b).max() == 240

    def test_output_is_uint8(self, black, white):
        assert difference(white, black).dtype == np.uint8


class TestMismatchedSizes:
    """A warped image and its target need not agree to the pixel."""

    @pytest.mark.parametrize("mode", BLEND_MODES)
    def test_every_mode_survives_a_size_difference(self, mode):
        overlay = np.zeros((30, 40), dtype=np.uint8)
        background = np.zeros((32, 38), dtype=np.uint8)
        result = composite(mode, overlay, background)
        assert result.shape == (30, 38, 3)

    def test_the_shared_region_is_the_smaller_of_each_dimension(self, black):
        big = np.zeros((100, 20), dtype=np.uint8)
        assert difference(black, big).shape == (64, 20, 3)


class TestComposite:
    """The dispatcher the view actually calls."""

    @pytest.mark.parametrize("mode", BLEND_MODES)
    def test_every_advertised_mode_works(self, mode, black, white):
        result = composite(mode, white, black, alphas=np.ones((64, 64)))
        assert result.shape == (64, 64, 3)
        assert result.dtype == np.uint8

    def test_an_unknown_mode_lists_the_options(self, black, white):
        with pytest.raises(ValueError, match="Unknown blend mode"):
            composite("dissolve", white, black)

    def test_wipe_without_a_mask_shows_the_overlay(self, black, white):
        """Opening the preview before a slider moves must not blow up."""
        assert composite("wipe", white, black).min() == 255

    def test_the_default_mode_is_the_familiar_one(self):
        """Existing users should find the preview behaving as it always did."""
        assert BLEND_MODES[0] == "wipe"

    def test_the_tile_size_reaches_the_checkerboard(self, black, white):
        small = composite("checkerboard", white, black, tile_size=4)
        large = composite("checkerboard", white, black, tile_size=32)
        assert not np.array_equal(small, large)

    def test_the_default_tile_is_visible_at_typical_sizes(self):
        """A tile bigger than the image would show one flat square."""
        assert 4 <= DEFAULT_TILE_SIZE <= 64
