"""Tests for the thin-plate spline transform."""

from __future__ import annotations

import numpy as np
import pytest

from tpsreg.tps import MIN_CONTROL_POINTS, ThinPlateSplineTransform


class TestValidation:
    """Control point validation."""

    def test_mismatched_shapes_rejected(self):
        src = np.array([[0, 0], [1, 1], [2, 2]], dtype=float)
        dst = np.array([[0, 0], [1, 1]], dtype=float)
        with pytest.raises(ValueError, match="same shape"):
            ThinPlateSplineTransform._check_valid_points(src, dst)

    def test_one_dimensional_input_rejected(self):
        pts = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match=r"expected \(N, 2\)"):
            ThinPlateSplineTransform._check_valid_points(pts, pts)

    def test_three_dimensional_coordinates_rejected(self):
        pts = np.zeros((5, 3))
        pts[:, 0] = np.arange(5)
        with pytest.raises(ValueError, match="2D coordinates"):
            ThinPlateSplineTransform._check_valid_points(pts, pts)

    def test_too_few_points_rejected(self):
        pts = np.array([[0.0, 0.0], [1.0, 1.0]])
        with pytest.raises(ValueError, match=f"At least {MIN_CONTROL_POINTS}"):
            ThinPlateSplineTransform._check_valid_points(pts, pts)

    def test_duplicate_source_points_rejected(self):
        src = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
        dst = np.array([[0.0, 0.0], [2.0, 3.0], [1.0, 1.0]])
        with pytest.raises(ValueError, match="Source control points"):
            ThinPlateSplineTransform._check_valid_points(src, dst)

    def test_duplicate_destination_points_rejected(self):
        src = np.array([[0.0, 0.0], [2.0, 3.0], [1.0, 1.0]])
        dst = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
        with pytest.raises(ValueError, match="Destination control points"):
            ThinPlateSplineTransform._check_valid_points(src, dst)

    def test_valid_points_accepted(self, square_grid_points):
        assert ThinPlateSplineTransform._check_valid_points(
            square_grid_points, square_grid_points + 1
        )


class TestEstimation:
    """Fitting the spline."""

    def test_call_before_estimate_raises(self):
        tform = ThinPlateSplineTransform()
        with pytest.raises(ValueError, match="not estimated"):
            tform(np.array([[0, 0]]))

    def test_estimate_sets_state(self, square_grid_points):
        tform = ThinPlateSplineTransform()
        assert tform.estimate(square_grid_points, square_grid_points, (100, 100))
        assert tform._estimated
        assert tform.size == (100, 100)
        assert tform.params.shape == (2, 100, 100)

    def test_identity_maps_points_to_themselves(self, square_grid_points):
        """A spline fitted to identical point sets is the identity."""
        tform = ThinPlateSplineTransform()
        tform.estimate(square_grid_points, square_grid_points, (100, 100))

        query = np.array([[20, 30], [50, 50], [70, 80]])
        result = tform(query)

        # params is built on a 1-based grid, so a query at index i returns the
        # coordinate i+1; the mapping is the identity up to that offset.
        np.testing.assert_allclose(result, query + 1, atol=1e-3)

    def test_pure_translation_recovered(self, square_grid_points):
        """Fitting a known translation reproduces it across the grid."""
        shift = np.array([5.0, -3.0])
        src = square_grid_points + shift
        dst = square_grid_points

        tform = ThinPlateSplineTransform()
        tform.estimate(src, dst, (100, 100))

        query = np.array([[25, 25], [60, 40]])
        result = tform(query)
        np.testing.assert_allclose(result, query + 1 + shift, atol=1e-2)

    def test_affine_only_ignores_bending(self, square_grid_points, rng):
        """affine_only drops the non-linear term, so the two differ on a warp."""
        dst = square_grid_points
        src = square_grid_points + rng.normal(0, 3, square_grid_points.shape)

        full = ThinPlateSplineTransform(affine_only=False)
        full.estimate(src, dst, (100, 100))

        affine = ThinPlateSplineTransform(affine_only=True)
        affine.estimate(src, dst, (100, 100))

        assert full.params.shape == affine.params.shape
        assert not np.allclose(full.params, affine.params)

    def test_collinear_points_raise_clear_error(self):
        """A degenerate system should explain itself, not leak LinAlgError."""
        src = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        dst = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        tform = ThinPlateSplineTransform()
        with pytest.raises(ValueError, match="collinear|solve"):
            tform.estimate(src, dst, (10, 10))

    def test_non_square_output_shape(self, square_grid_points):
        """The grid is indexed (height, width); a non-square size proves it."""
        tform = ThinPlateSplineTransform()
        tform.estimate(square_grid_points, square_grid_points, (40, 90))
        assert tform.params.shape == (2, 40, 90)

    def test_out_of_range_coordinates_are_clamped(self, square_grid_points):
        """warp() queries the full output grid; queries must not IndexError."""
        tform = ThinPlateSplineTransform()
        tform.estimate(square_grid_points, square_grid_points, (50, 50))

        result = tform(np.array([[999, 999], [-5, -5]]))
        assert result.shape == (2, 2)
        assert np.all(np.isfinite(result))

    def test_progress_callback_reports_every_chunk(self, square_grid_points):
        seen = []
        tform = ThinPlateSplineTransform(chunk_size=1000)
        tform.estimate(
            square_grid_points,
            square_grid_points,
            (100, 100),
            progress_callback=lambda done, total: seen.append((done, total)),
        )

        assert seen, "progress callback was never invoked"
        assert seen[-1][0] == seen[-1][1], "final call should report completion"
        assert [done for done, _ in seen] == list(range(1, len(seen) + 1))

    def test_affine_only_skips_progress_callback(self, square_grid_points):
        """There is no bending pass to report on when affine_only is set."""
        seen = []
        tform = ThinPlateSplineTransform(affine_only=True)
        tform.estimate(
            square_grid_points,
            square_grid_points,
            (100, 100),
            progress_callback=lambda done, total: seen.append((done, total)),
        )
        assert seen == []

    def test_chunking_matches_single_pass(self, square_grid_points, rng):
        """Chunk size is a memory knob; it must not change the result."""
        dst = square_grid_points
        src = square_grid_points + rng.normal(0, 2, square_grid_points.shape)

        one_chunk = ThinPlateSplineTransform(chunk_size=10_000)
        one_chunk.estimate(src, dst, (50, 50))

        many_chunks = ThinPlateSplineTransform(chunk_size=1000)
        many_chunks.estimate(src, dst, (50, 50))

        np.testing.assert_allclose(one_chunk.params, many_chunks.params, rtol=1e-5)


class TestChunkSizeEstimation:
    """The memory-budget heuristic."""

    def test_larger_budget_gives_larger_chunks(self):
        tform = ThinPlateSplineTransform()
        small = tform._estimate_chunk_size(10**8, 100, available_memory_gb=0.5)
        large = tform._estimate_chunk_size(10**8, 100, available_memory_gb=8.0)
        assert large > small

    def test_never_exceeds_total_pixels(self):
        tform = ThinPlateSplineTransform()
        assert tform._estimate_chunk_size(500, 10, available_memory_gb=64.0) <= 1000

    def test_has_a_floor(self):
        """A tiny budget must not produce a 1-pixel-per-chunk pathology."""
        tform = ThinPlateSplineTransform()
        assert tform._estimate_chunk_size(10**8, 10**6, available_memory_gb=0.001) >= 1000


class TestSystemMatrix:
    """The TPS L matrix."""

    def test_shape_and_symmetry(self, square_grid_points):
        L = ThinPlateSplineTransform._TPS_makeL(square_grid_points)
        k = len(square_grid_points)
        assert L.shape == (k + 3, k + 3)
        np.testing.assert_allclose(L[:k, :k], L[:k, :k].T)

    def test_diagonal_of_kernel_block_is_zero(self, square_grid_points):
        """U(0) = 0, so the kernel block must have a zero diagonal."""
        L = ThinPlateSplineTransform._TPS_makeL(square_grid_points)
        k = len(square_grid_points)
        np.testing.assert_array_equal(np.diag(L[:k, :k]), np.zeros(k))

    def test_constraint_block_layout(self, square_grid_points):
        L = ThinPlateSplineTransform._TPS_makeL(square_grid_points)
        k = len(square_grid_points)
        np.testing.assert_array_equal(L[:k, k], np.ones(k))
        np.testing.assert_array_equal(L[:k, k + 1 : k + 3], square_grid_points)
        np.testing.assert_array_equal(L[k, :k], np.ones(k))
        np.testing.assert_array_equal(L[k + 1 :, :k], square_grid_points.T)
