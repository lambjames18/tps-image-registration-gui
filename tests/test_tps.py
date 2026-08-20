"""Tests for the thin-plate spline transform."""

from __future__ import annotations

import numpy as np
import pytest

from tpsreg.tps import (
    MIN_CONTROL_POINTS,
    ThinPlateSplineTransform,
    loocv_residuals,
    select_regularization,
)


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

        # The transform is its coefficients: K weights plus three affine terms.
        assert tform.params.shape == (len(square_grid_points) + 3, 2)
        assert tform.control_points.shape == square_grid_points.shape

    def test_fitting_does_not_build_a_field(self, square_grid_points):
        """The dense field is a cache, and nothing asked for it yet.

        Building it up front is what made the cost of a fit scale with the
        image instead of with the control points.
        """
        tform = ThinPlateSplineTransform()
        tform.estimate(square_grid_points, square_grid_points, (100, 100))
        assert tform.field is None

    def test_coefficients_are_small_whatever_the_grid(self, square_grid_points):
        """A 400 Mpx destination used to mean a 3.2 GB transform."""
        small = ThinPlateSplineTransform()
        small.estimate(square_grid_points, square_grid_points, (100, 100))

        huge = ThinPlateSplineTransform()
        huge.estimate(square_grid_points, square_grid_points, (20_000, 20_000))

        assert huge.params.shape == small.params.shape
        assert huge.params.nbytes < 1024
        np.testing.assert_allclose(huge.params, small.params)

    def test_identity_maps_points_to_themselves(self, square_grid_points):
        """A spline fitted to identical point sets is the identity.

        It used to be the identity plus one pixel in each axis: the field was
        sampled on a 1-based grid while control points and queries are
        0-based, so every warp carried a systematic one-pixel bias. The
        previous version of this test asserted the offset rather than
        questioning it.
        """
        tform = ThinPlateSplineTransform()
        tform.estimate(square_grid_points, square_grid_points, (100, 100))

        query = np.array([[20, 30], [50, 50], [70, 80]], dtype=float)
        np.testing.assert_allclose(tform(query), query, atol=1e-6)

    def test_identity_is_exact_at_fractional_coordinates(self, square_grid_points):
        """Queries are no longer truncated to whole pixels before lookup."""
        tform = ThinPlateSplineTransform()
        tform.estimate(square_grid_points, square_grid_points, (100, 100))

        query = np.array([[20.25, 30.75], [49.5, 50.5]])
        np.testing.assert_allclose(tform(query), query, atol=1e-6)

    def test_pure_translation_recovered(self, square_grid_points):
        """Fitting a known translation reproduces it across the grid."""
        shift = np.array([5.0, -3.0])
        src = square_grid_points + shift
        dst = square_grid_points

        tform = ThinPlateSplineTransform()
        tform.estimate(src, dst, (100, 100))

        query = np.array([[25, 25], [60, 40]], dtype=float)
        np.testing.assert_allclose(tform(query), query + shift, atol=1e-6)

    def test_collinear_points_raise_clear_error(self):
        """A degenerate system should explain itself, not leak LinAlgError."""
        src = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        dst = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        tform = ThinPlateSplineTransform()
        with pytest.raises(ValueError, match=r"collinear|solve"):
            tform.estimate(src, dst, (10, 10))

    def test_non_square_output_shape(self, square_grid_points):
        """The field is indexed (height, width); a non-square size proves it."""
        tform = ThinPlateSplineTransform()
        tform.estimate(square_grid_points, square_grid_points, (40, 90))
        assert tform.size == (40, 90)
        assert tform.build_field().shape == (2, 40, 90)

    def test_out_of_range_coordinates_are_clamped(self, square_grid_points):
        """warp() queries the full output grid; queries must not IndexError."""
        tform = ThinPlateSplineTransform()
        tform.estimate(square_grid_points, square_grid_points, (50, 50))

        result = tform(np.array([[999, 999], [-5, -5]]))
        assert result.shape == (2, 2)
        assert np.all(np.isfinite(result))

    def test_progress_callback_reports_every_chunk(self, square_grid_points):
        """Progress is reported while building a field, the only slow part.

        A fit on its own is milliseconds and has nothing to report.
        """
        seen = []
        tform = ThinPlateSplineTransform(chunk_size=1000)
        tform.estimate(
            square_grid_points,
            square_grid_points,
            (100, 100),
            build_field=True,
            progress_callback=lambda done, total: seen.append((done, total)),
        )

        assert seen, "progress callback was never invoked"
        assert seen[-1][0] == seen[-1][1], "final call should report completion"
        assert [done for done, _ in seen] == list(range(1, len(seen) + 1))

    def test_fitting_alone_reports_no_progress(self, square_grid_points):
        """Nothing to report when no field is being built."""
        seen = []
        tform = ThinPlateSplineTransform()
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
        one_chunk.estimate(src, dst, (50, 50), build_field=True)

        many_chunks = ThinPlateSplineTransform(chunk_size=1000)
        many_chunks.estimate(src, dst, (50, 50), build_field=True)

        np.testing.assert_allclose(one_chunk.params, many_chunks.params, rtol=1e-5)
        np.testing.assert_allclose(one_chunk.field, many_chunks.field, rtol=1e-5)


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
        assert (
            tform._estimate_chunk_size(10**8, 10**6, available_memory_gb=0.001) >= 1000
        )


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


class TestDirectMapping:
    """Evaluating the spline without building a grid."""

    @pytest.fixture
    def deformable(self, square_grid_points, rng):
        """A fit with genuine local distortion, not just an affine."""
        dst = square_grid_points
        src = square_grid_points + rng.normal(0, 5, square_grid_points.shape)
        tform = ThinPlateSplineTransform()
        tform.estimate(src, dst, (100, 100))
        return tform

    def test_map_returns_one_point_per_input(self, deformable):
        query = np.array([[10.0, 20.0], [30.0, 40.0], [55.0, 15.0]])
        assert deformable.map(query).shape == query.shape

    def test_map_reproduces_the_control_points(self, square_grid_points, rng):
        """The spline interpolates: it must hit its own control points."""
        dst = square_grid_points
        src = square_grid_points + rng.normal(0, 5, square_grid_points.shape)

        tform = ThinPlateSplineTransform()
        tform.estimate(src, dst, (100, 100))

        np.testing.assert_allclose(tform.map(dst), src, atol=1e-6)

    def test_mapping_needs_no_grid_size(self, square_grid_points):
        """size is advisory now; mapping works without one."""
        tform = ThinPlateSplineTransform()
        tform.estimate(square_grid_points, square_grid_points)

        assert tform.size is None
        query = np.array([[12.0, 34.0]])
        np.testing.assert_allclose(tform.map(query), query, atol=1e-6)

    def test_mapping_is_independent_of_the_recorded_size(self, square_grid_points, rng):
        """A coordinate maps the same however big the grid is said to be."""
        dst = square_grid_points
        src = square_grid_points + rng.normal(0, 4, square_grid_points.shape)
        query = np.array([[25.0, 35.0], [70.0, 60.0]])

        small = ThinPlateSplineTransform()
        small.estimate(src, dst, (100, 100))
        huge = ThinPlateSplineTransform()
        huge.estimate(src, dst, (50_000, 50_000))

        np.testing.assert_allclose(small.map(query), huge.map(query))

    def test_chunking_does_not_change_the_mapping(self, deformable, rng):
        """Chunk size is a memory knob for mapping too."""
        query = rng.uniform(0, 99, size=(5000, 2))

        one_chunk = deformable.map(query, available_memory_gb=64.0)
        deformable.chunk_size = 97
        many_chunks = deformable.map(query)

        np.testing.assert_allclose(one_chunk, many_chunks)

    def test_map_rejects_the_wrong_shape(self, deformable):
        with pytest.raises(ValueError, match=r"\(N, 2\)"):
            deformable.map(np.array([1.0, 2.0, 3.0]))

    def test_map_before_estimate_raises(self):
        with pytest.raises(ValueError, match="not estimated"):
            ThinPlateSplineTransform().map(np.array([[0.0, 0.0]]))

    def test_an_empty_query_is_survivable(self, deformable):
        assert deformable.map(np.empty((0, 2))).shape == (0, 2)


class TestFieldCache:
    """The dense field, which is now optional and resolution-configurable."""

    @pytest.fixture
    def deformable(self, square_grid_points, rng):
        dst = square_grid_points
        src = square_grid_points + rng.normal(0, 5, square_grid_points.shape)
        tform = ThinPlateSplineTransform()
        tform.estimate(src, dst, (100, 100))
        return tform

    def test_a_full_field_is_exact_on_its_own_samples(self, deformable):
        """Where the field was evaluated, it holds the exact answer."""
        rows, cols = np.meshgrid(np.arange(0, 100, 7.0), np.arange(0, 100, 9.0))
        query = np.column_stack([cols.ravel(), rows.ravel()])

        deformable.build_field((100, 100))
        np.testing.assert_allclose(deformable(query), deformable.map(query), atol=1e-4)

    def test_a_full_field_interpolates_between_its_samples(self, deformable, rng):
        """A cache is not free even at full resolution.

        Between samples the field is interpolated, so a fractional query is
        close but not exact. Callers that need exactness should use map().
        """
        query = rng.uniform(0, 99, size=(500, 2))
        exact = deformable.map(query)

        deformable.build_field((100, 100))
        error = np.linalg.norm(deformable(query) - exact, axis=1)
        assert 0 < error.mean() < 0.05

    def test_a_coarse_field_stays_well_under_a_pixel(self, deformable, rng):
        """Sub-pixel accuracy is not needed, so a coarse field is a fair trade.

        These control points sit about 27 px apart on a 100 px grid, which is
        dense distortion; a real stitched image has its points much further
        apart relative to the grid and does correspondingly better.
        """
        query = rng.uniform(0, 99, size=(500, 2))
        exact = deformable.map(query)

        deformable.build_field((100, 100), downsample=4)
        error = np.linalg.norm(deformable(query) - exact, axis=1)
        assert error.mean() < 0.1
        assert error.max() < 0.6

    def test_field_error_falls_away_quadratically(self, deformable, rng):
        """Bilinear interpolation: halving the step should quarter the error.

        This is the relationship that makes the resolution knob predictable,
        rather than any single number, which depends on how far apart the
        control points are.
        """
        query = rng.uniform(0, 99, size=(1000, 2))
        exact = deformable.map(query)

        errors = {}
        for step in (2, 4, 8):
            deformable.build_field((100, 100), downsample=step)
            errors[step] = np.linalg.norm(deformable(query) - exact, axis=1).mean()

        for coarse, fine in ((4, 2), (8, 4)):
            ratio = errors[coarse] / errors[fine]
            assert 2.5 < ratio < 6.0, f"1/{coarse} vs 1/{fine} scaled by {ratio:.2f}"

    def test_downsampling_saves_memory(self, deformable):
        full = deformable.build_field((100, 100)).nbytes
        coarse = deformable.build_field((100, 100), downsample=4).nbytes
        assert coarse < full / 8

    def test_the_field_spans_the_whole_grid(self, deformable):
        """The last sample is pinned to the final pixel, never short of it."""
        deformable.build_field((100, 90), downsample=7)
        assert deformable._field_xs[-1] == 89
        assert deformable._field_ys[-1] == 99

    def test_clearing_the_field_returns_to_direct_evaluation(self, deformable, rng):
        query = rng.uniform(0, 99, size=(200, 2))
        deformable.build_field((100, 100), downsample=8)
        coarse = deformable(query)

        deformable.clear_field()
        assert deformable.field is None
        np.testing.assert_allclose(deformable(query), deformable.map(query))
        assert not np.allclose(coarse, deformable(query), atol=1e-9)

    def test_build_field_needs_a_size_from_somewhere(self, square_grid_points):
        tform = ThinPlateSplineTransform()
        tform.estimate(square_grid_points, square_grid_points)
        with pytest.raises(ValueError, match="No grid size"):
            tform.build_field()

    def test_build_field_reuses_the_estimated_size(self, deformable):
        assert deformable.build_field().shape == (2, 100, 100)

    def test_estimate_can_build_the_field_up_front(self, square_grid_points):
        """The old behaviour, now opt-in."""
        tform = ThinPlateSplineTransform()
        tform.estimate(
            square_grid_points, square_grid_points, (60, 60), build_field=True
        )
        assert tform.field.shape == (2, 60, 60)

    def test_queries_beyond_the_grid_are_clamped(self, deformable):
        """warp() asks about the whole output, which can exceed the field."""
        deformable.build_field((50, 50))
        result = deformable(np.array([[999.0, 999.0], [-5.0, -5.0]]))
        assert result.shape == (2, 2)
        assert np.all(np.isfinite(result))


class TestInstalledFields:
    """A transform carrying a field but no coefficients."""

    @staticmethod
    def _field(shape=(2, 20, 30)):
        rows = np.arange(shape[1], dtype=float)[:, None]
        cols = np.arange(shape[2], dtype=float)[None, :]
        return np.stack(
            [np.broadcast_to(cols, shape[1:]), np.broadcast_to(rows, shape[1:])]
        )

    def test_a_field_alone_makes_a_usable_transform(self):
        """Interpolated stack slices have no coefficients of their own."""
        tform = ThinPlateSplineTransform()
        tform.set_field(self._field(), size=(20, 30))

        query = np.array([[5.0, 7.0], [12.0, 3.0]])
        np.testing.assert_allclose(tform(query), query, atol=1e-6)

    def test_a_field_alone_still_has_no_coefficients(self):
        tform = ThinPlateSplineTransform()
        tform.set_field(self._field(), size=(20, 30))
        assert tform.params is None
        with pytest.raises(ValueError, match="not estimated"):
            tform.map(np.array([[1.0, 1.0]]))

    def test_a_malformed_field_is_rejected(self):
        tform = ThinPlateSplineTransform()
        with pytest.raises(ValueError, match=r"\(2, h, w\)"):
            tform.set_field(np.zeros((3, 4, 5)))

    def test_the_size_is_inferred_when_not_given(self):
        tform = ThinPlateSplineTransform()
        tform.set_field(self._field((2, 20, 30)))
        assert tform.size == (20, 30)


def _grid(n=5, extent=90.0):
    axis = np.linspace(10.0, extent, n)
    return np.stack(np.meshgrid(axis, axis), -1).reshape(-1, 2)


def _bulge(points, amplitude=0.35, sigma=35.0):
    """A real local deformation, distinct from noise."""
    offset = points - np.array([50.0, 50.0])
    radius = np.linalg.norm(offset, axis=1, keepdims=True)
    return points + amplitude * offset * np.exp(-((radius / sigma) ** 2))


class TestRegularization:
    """Letting the spline miss its control points."""

    def test_off_by_default(self, square_grid_points):
        tform = ThinPlateSplineTransform()
        tform.estimate(square_grid_points, square_grid_points, (100, 100))
        assert tform.regularization == 0.0
        assert tform.effective_regularization == 0.0

    def test_zero_still_interpolates_exactly(self, square_grid_points, rng):
        """The default must not change what this has always done."""
        dst = square_grid_points
        src = square_grid_points + rng.normal(0, 3, dst.shape)

        tform = ThinPlateSplineTransform(regularization=0.0)
        tform.estimate(src, dst, (100, 100))

        np.testing.assert_allclose(tform.map(dst), src, atol=1e-6)

    def test_a_positive_strength_stops_interpolating(self, square_grid_points, rng):
        dst = square_grid_points
        src = square_grid_points + rng.normal(0, 3, dst.shape)

        tform = ThinPlateSplineTransform(regularization=0.1)
        tform.estimate(src, dst, (100, 100))

        assert np.linalg.norm(tform.map(dst) - src, axis=1).max() > 1e-3

    def test_more_smoothing_means_less_bending(self, square_grid_points, rng):
        """The point of it: trade fidelity at the points for smoothness."""
        from tpsreg import metrics

        dst = square_grid_points
        src = square_grid_points + rng.normal(0, 4, dst.shape)

        energies = []
        for strength in (0.0, 0.001, 0.01, 0.1):
            tform = ThinPlateSplineTransform(regularization=strength)
            tform.estimate(src, dst, (100, 100))
            energies.append(metrics.bending_energy(tform))

        assert energies == sorted(energies, reverse=True), energies

    def test_an_affine_is_unaffected_by_smoothing(self, square_grid_points):
        """There is no bending to penalise, so nothing should change."""
        dst = square_grid_points
        src = square_grid_points * 1.5 + np.array([4.0, -3.0])
        query = np.array([[25.0, 35.0], [70.0, 60.0]])

        exact = ThinPlateSplineTransform()
        exact.estimate(src, dst, (100, 100))
        smoothed = ThinPlateSplineTransform(regularization=0.5)
        smoothed.estimate(src, dst, (100, 100))

        np.testing.assert_allclose(smoothed.map(query), exact.map(query), atol=1e-6)

    def test_a_negative_strength_is_refused(self, square_grid_points):
        """Subtracting from the diagonal is a different, unstable system."""
        tform = ThinPlateSplineTransform(regularization=-0.1)
        with pytest.raises(ValueError, match="must not be negative"):
            tform.estimate(square_grid_points, square_grid_points, (100, 100))

    def test_an_unknown_keyword_is_refused(self, square_grid_points):
        tform = ThinPlateSplineTransform(regularization="strong")
        with pytest.raises(ValueError, match="Unknown regularization"):
            tform.estimate(square_grid_points, square_grid_points, (100, 100))

    def test_the_strength_is_roughly_scale_invariant(self, rng):
        """The same number should mean roughly the same at any image size.

        Exactly invariant is not achievable: r**2 log(r**2) is not
        scale-homogeneous. Within a factor of a few is enough for the number
        to be usable and for the automatic search to start in the right place.
        """
        axis = np.linspace(0.1, 0.9, 4)
        unit = np.stack(np.meshgrid(axis, axis), -1).reshape(-1, 2)
        noise = rng.normal(0, 0.02, unit.shape)

        relative = []
        for extent in (100.0, 20000.0):
            dst = unit * extent
            tform = ThinPlateSplineTransform(regularization=0.01)
            tform.estimate(dst + noise * extent, dst, (int(extent), int(extent)))
            relative.append(
                np.linalg.norm(tform.map(dst) - (dst + noise * extent), axis=1).mean()
                / extent
            )

        ratio = max(relative) / min(relative)
        assert ratio < 5, f"a 200x scale change moved the effect {ratio:.1f}x"


class TestClosedFormLeaveOneOut:
    """The identity that makes the automatic search affordable."""

    @staticmethod
    def _brute_force(src, dst, strength):
        """Refit K times, the obvious way, for comparison.

        The strength is rescaled for each reduced set so every fold uses the
        same *absolute* penalty. The strength is normalised by a kernel scale
        computed from the points, and dropping one changes that scale, so
        passing the same normalised number would quietly be penalising each
        fold slightly differently -- and the identity being checked here holds
        for a fixed penalty.
        """
        from tpsreg.tps import _kernel_scale

        full_scale = _kernel_scale(dst)
        residuals = np.empty(len(dst))
        for index in range(len(dst)):
            keep = np.ones(len(dst), dtype=bool)
            keep[index] = False
            rescaled = strength * full_scale / _kernel_scale(dst[keep])
            reduced = ThinPlateSplineTransform(regularization=rescaled)
            reduced.estimate(src[keep], dst[keep])
            residuals[index] = np.linalg.norm(
                reduced.map(dst[index : index + 1])[0] - src[index]
            )
        return residuals

    @pytest.mark.parametrize("strength", [0.001, 0.01, 0.1])
    def test_it_matches_refitting(self, strength, rng):
        """If these ever disagree, the fast path is silently wrong."""
        dst = _grid()
        src = dst + rng.normal(0, 2, dst.shape)

        np.testing.assert_allclose(
            loocv_residuals(src, dst, strength),
            self._brute_force(src, dst, strength),
            rtol=1e-6,
            atol=1e-9,
        )

    def test_one_residual_per_point(self, rng):
        dst = _grid()
        assert loocv_residuals(dst + rng.normal(0, 1, dst.shape), dst, 0.01).shape == (
            len(dst),
        )

    def test_an_exact_fit_has_small_residuals_for_clean_points(self):
        dst = _grid()
        residuals = loocv_residuals(dst + np.array([3.0, -2.0]), dst, 0.0)
        assert residuals.max() < 1e-6


class TestSelectingTheStrength:
    """Cross-validated choice."""

    def test_clean_points_get_no_smoothing(self):
        """Nothing to smooth, so it should not smooth."""
        dst = _grid()
        best, _, _ = select_regularization(_bulge(dst), dst)
        assert best == 0.0

    def test_a_pure_translation_gets_no_smoothing(self):
        """Every strength fits this perfectly; the tie must go to zero.

        Without an absolute floor on the tie tolerance this picked an
        arbitrary strength, decided by floating-point noise among scores that
        were all around 1e-15.
        """
        dst = _grid()
        best, _, _ = select_regularization(dst + np.array([3.0, -2.0]), dst)
        assert best == 0.0

    def test_noisy_points_get_some_smoothing(self, rng):
        dst = _grid()
        best, _, _ = select_regularization(
            _bulge(dst) + rng.normal(0, 2, dst.shape), dst
        )
        assert best > 0

    def test_it_returns_the_whole_curve(self):
        dst = _grid()
        best, candidates, scores = select_regularization(_bulge(dst), dst)
        assert len(candidates) == len(scores)
        assert best in candidates

    def test_a_custom_candidate_list_is_honoured(self, rng):
        dst = _grid()
        src = _bulge(dst) + rng.normal(0, 2, dst.shape)
        candidates = np.array([0.0, 0.05, 0.5])

        best, returned, scores = select_regularization(src, dst, candidates)
        np.testing.assert_array_equal(returned, candidates)
        assert best in candidates
        assert len(scores) == 3

    @pytest.mark.parametrize("noise", [1.0, 2.0, 4.0])
    def test_the_automatic_choice_beats_exact_interpolation(self, noise, rng):
        """The claim this feature rests on, over the whole field.

        Exact interpolation reproduces click errors exactly, so with noisy
        points it is fitting the noise. Measured against the deformation that
        actually generated the data.
        """
        dst = _grid()
        truth = _bulge(dst)
        src = truth + rng.normal(0, noise, dst.shape)

        auto = ThinPlateSplineTransform(regularization="auto")
        auto.estimate(src, dst, (100, 100))
        exact = ThinPlateSplineTransform()
        exact.estimate(src, dst, (100, 100))

        query = rng.uniform(10, 90, size=(2000, 2))
        expected = _bulge(query)
        auto_error = np.linalg.norm(auto.map(query) - expected, axis=1).mean()
        exact_error = np.linalg.norm(exact.map(query) - expected, axis=1).mean()

        assert auto_error <= exact_error + 1e-9

    def test_auto_records_what_it_chose(self, rng):
        dst = _grid()
        tform = ThinPlateSplineTransform(regularization="auto")
        tform.estimate(_bulge(dst) + rng.normal(0, 2, dst.shape), dst, (100, 100))

        assert tform.regularization == "auto"
        assert isinstance(tform.effective_regularization, float)
        assert tform.effective_regularization > 0
