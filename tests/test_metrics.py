"""Tests for the transform quality metrics.

Pure numpy and no display. Several of these check a metric against a
transform whose correct answer is known analytically -- the identity, a pure
scale -- because a quality measure that is subtly wrong is worse than none.
"""

from __future__ import annotations

import numpy as np
import pytest

from tpsreg import metrics
from tpsreg.tps import ThinPlateSplineTransform

#: A well-spread, non-degenerate set of destination control points.
CORNERS = np.array(
    [
        [20.0, 20.0],
        [80.0, 20.0],
        [20.0, 80.0],
        [80.0, 80.0],
        [50.0, 35.0],
        [35.0, 60.0],
        [65.0, 60.0],
    ]
)


#: A denser, regular set. Leave-one-out asks the remaining points to predict
#: the held-out one, so it needs enough of them for that question to have an
#: answer; with only a handful the geometry dominates the signal. See
#: TestLeaveOneOutNeedsEnoughPoints.
GRID = np.stack(
    np.meshgrid(np.linspace(10.0, 90.0, 4), np.linspace(10.0, 90.0, 4)), -1
).reshape(-1, 2)


def fit(src, dst=CORNERS, size=(100, 100)):
    tform = ThinPlateSplineTransform()
    tform.estimate(src, dst, size)
    return tform


class TestPlainResidualsAreUseless:
    """The premise the rest of this module rests on."""

    def test_a_spline_passes_exactly_through_its_control_points(self, rng):
        """Which is why the obvious quality check tells you nothing.

        If this ever stops being true, leave-one-out stops being necessary and
        this module should be reconsidered.
        """
        src = CORNERS + rng.normal(0, 3, CORNERS.shape)
        tform = fit(src)

        residuals = np.linalg.norm(tform.map(CORNERS) - src, axis=1)
        assert residuals.max() < 1e-6

    def test_even_a_badly_placed_point_has_no_residual(self, rng):
        src = CORNERS + rng.normal(0, 3, CORNERS.shape)
        src[4] += np.array([30.0, -25.0])
        tform = fit(src)

        residuals = np.linalg.norm(tform.map(CORNERS) - src, axis=1)
        assert residuals[4] < 1e-6, "a wrong point is invisible to plain residuals"


class TestLeaveOneOut:
    """The per-point measure that does work."""

    def test_one_residual_per_point(self, rng):
        src = CORNERS + rng.normal(0, 2, CORNERS.shape)
        assert metrics.leave_one_out_residuals(src, CORNERS).shape == (len(CORNERS),)

    def test_a_consistent_field_gives_small_residuals(self):
        """A smooth deformation is predictable from its neighbours."""
        src = CORNERS + np.array([4.0, -2.0])
        residuals = metrics.leave_one_out_residuals(src, CORNERS)
        assert np.nanmax(residuals) < 1e-6

    def test_an_identity_fit_gives_zero_residuals(self):
        residuals = metrics.leave_one_out_residuals(CORNERS, CORNERS)
        assert np.nanmax(residuals) < 1e-6

    def test_a_scale_is_predictable_from_its_neighbours(self):
        residuals = metrics.leave_one_out_residuals(CORNERS * 2.5, CORNERS)
        assert np.nanmax(residuals) < 1e-6

    def test_a_misplaced_point_stands_out(self, rng):
        """The whole point: find the correspondence that was clicked wrong."""
        src = GRID + rng.normal(0, 1, GRID.shape)
        src[6] += np.array([25.0, -20.0])

        residuals = metrics.leave_one_out_residuals(src, GRID)
        assert np.nanargmax(residuals) == 6
        assert residuals[6] > 3 * np.nanmedian(residuals)

    def test_too_few_points_yields_nan_not_an_error(self):
        """Three points minus one is two, which cannot be fitted."""
        three = np.array([[0.0, 0.0], [50.0, 5.0], [5.0, 50.0]])
        residuals = metrics.leave_one_out_residuals(three, three)
        assert residuals.shape == (3,)
        assert np.all(np.isnan(residuals))

    def test_a_degenerate_reduction_yields_nan_for_that_point_only(self):
        """Dropping one point can leave the rest collinear."""
        points = np.array(
            [[0.0, 0.0], [10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [15.0, 5.0]]
        )
        residuals = metrics.leave_one_out_residuals(points, points)

        # Removing the off-line point leaves four collinear ones.
        assert np.isnan(residuals[4])
        assert residuals.shape == (5,)

    def test_mismatched_inputs_are_rejected(self):
        with pytest.raises(ValueError, match=r"\(K, 2\)"):
            metrics.leave_one_out_residuals(CORNERS, CORNERS[:-1])

    def test_it_finds_the_bad_point_wherever_it_is(self, rng):
        """Not just in the middle: an edge point is the harder case."""
        for bad in (0, 3, 7, 12, 15):
            src = GRID + rng.normal(0, 1, GRID.shape)
            src[bad] += np.array([30.0, 25.0])
            residuals = metrics.leave_one_out_residuals(src, GRID)
            assert np.nanargmax(residuals) == bad, f"missed a bad point at {bad}"


class TestLeaveOneOutNeedsEnoughPoints:
    """Its weakness, stated rather than hidden.

    The measure asks the remaining points to predict the held-out one. With
    very few points that question has no good answer for any of them, so every
    residual is large and a genuinely bad point does not stand out. It becomes
    reliable once the points are dense enough to constrain each other.
    """

    @staticmethod
    def _corrupt(points, index, rng):
        src = points + rng.normal(0, 1, points.shape)
        src[index] += np.array([25.0, -20.0])
        return src

    def test_a_sparse_set_can_hide_a_bad_point(self, rng):
        """Documented so the metric is not trusted beyond what it can do."""
        src = self._corrupt(CORNERS, 4, rng)
        residuals = metrics.leave_one_out_residuals(src, CORNERS)

        # Every residual is large, so the bad one is not distinctive.
        assert np.nanmedian(residuals) > 5

    def test_a_dense_set_finds_it_reliably(self, rng):
        src = self._corrupt(GRID, 6, rng)
        residuals = metrics.leave_one_out_residuals(src, GRID)

        assert np.nanargmax(residuals) == 6
        assert residuals[6] > 5 * np.nanmedian(residuals)


class TestOutlierFlagging:
    """Deciding which residuals are worth showing."""

    def test_uniform_residuals_flag_nothing(self):
        assert not metrics.flag_outliers(np.full(10, 2.0)).any()

    def test_one_large_residual_is_flagged(self):
        residuals = np.array([1.0, 1.1, 0.9, 1.05, 0.95, 1.0, 12.0])
        flags = metrics.flag_outliers(residuals)
        assert flags[-1]
        assert not flags[:-1].any()

    def test_the_median_is_used_rather_than_the_mean(self):
        """A bad point inflates the standard deviation enough to hide itself.

        With mean and standard deviation the outlier below scores under 3, so
        a conventional z-score would miss it entirely.
        """
        residuals = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 20.0])

        classic = (residuals[-1] - residuals.mean()) / residuals.std()
        assert classic < 3, "the setup must actually defeat a plain z-score"

        assert metrics.flag_outliers(residuals)[-1]

    def test_nan_residuals_are_never_flagged(self):
        residuals = np.array([1.0, 1.0, 1.0, 1.0, np.nan, 30.0])
        flags = metrics.flag_outliers(residuals)
        assert not flags[4]
        assert flags[5]

    def test_too_few_points_flags_nothing(self):
        assert not metrics.flag_outliers(np.array([1.0, 50.0])).any()

    def test_a_higher_threshold_flags_less(self):
        residuals = np.array([1.0, 1.1, 0.9, 1.05, 0.95, 1.0, 6.0])
        assert metrics.flag_outliers(residuals, threshold=3.5).any()
        assert not metrics.flag_outliers(residuals, threshold=500.0).any()

    def test_all_zero_residuals_flag_nothing(self):
        """A perfect fit must not report every point as an outlier."""
        assert not metrics.flag_outliers(np.zeros(8)).any()


class TestJacobian:
    """Detecting a mapping that folds over itself."""

    def test_the_identity_has_unit_determinant(self):
        determinants = metrics.jacobian_determinant(fit(CORNERS), downsample=4)
        np.testing.assert_allclose(determinants, 1.0, atol=1e-6)

    def test_a_uniform_scale_has_a_known_determinant(self):
        """Doubling in both axes quadruples area."""
        determinants = metrics.jacobian_determinant(fit(CORNERS * 2), downsample=4)
        np.testing.assert_allclose(determinants, 4.0, atol=1e-6)

    def test_a_translation_leaves_the_determinant_at_one(self):
        tform = fit(CORNERS + np.array([7.0, -4.0]))
        np.testing.assert_allclose(
            metrics.jacobian_determinant(tform, downsample=4), 1.0, atol=1e-6
        )

    def test_a_clean_fit_has_no_folds(self, rng):
        determinants = metrics.jacobian_determinant(
            fit(CORNERS + rng.normal(0, 2, CORNERS.shape)), downsample=4
        )
        assert determinants.min() > 0
        assert metrics.folded_fraction(determinants) == 0

    def test_crossed_correspondences_fold(self, rng):
        """Two swapped points make the mapping turn inside out between them.

        This is the failure per-point measures cannot see: the control points
        on either side of the fold are still matched exactly.
        """
        src = CORNERS + rng.normal(0, 2, CORNERS.shape)
        src[[4, 5]] = src[[5, 4]]

        determinants = metrics.jacobian_determinant(fit(src), downsample=4)
        assert determinants.min() < 0
        assert metrics.folded_fraction(determinants) > 0

    def test_folded_fraction_of_nothing_is_zero(self):
        assert metrics.folded_fraction(np.empty(0)) == 0.0

    def test_folded_fraction_counts_non_positive(self):
        assert metrics.folded_fraction(np.array([1.0, -1.0, 0.0, 2.0])) == 0.5

    def test_a_size_is_required_from_somewhere(self, rng):
        tform = ThinPlateSplineTransform()
        tform.estimate(CORNERS + rng.normal(0, 1, CORNERS.shape), CORNERS)
        with pytest.raises(ValueError, match="No grid size"):
            metrics.jacobian_determinant(tform)

    def test_an_explicit_size_overrides(self, rng):
        tform = fit(CORNERS + rng.normal(0, 1, CORNERS.shape))
        assert metrics.jacobian_determinant(
            tform, size=(40, 60), downsample=10
        ).shape == (
            4,
            6,
        )


class TestBendingEnergy:
    """How far the warp is from an affine."""

    @pytest.mark.parametrize(
        "name,src",
        [
            ("identity", CORNERS),
            ("translation", CORNERS + np.array([5.0, -3.0])),
            ("scale", CORNERS * 2),
        ],
    )
    def test_affine_warps_have_no_bending(self, name, src):
        assert metrics.bending_energy(fit(src)) == pytest.approx(0, abs=1e-9)

    def test_a_deformation_has_positive_bending(self, rng):
        assert (
            metrics.bending_energy(fit(CORNERS + rng.normal(0, 5, CORNERS.shape))) > 0
        )

    def test_more_deformation_means_more_bending(self, rng):
        gentle = fit(CORNERS + rng.normal(0, 1, CORNERS.shape))
        violent = fit(CORNERS + rng.normal(0, 10, CORNERS.shape))
        assert metrics.bending_energy(violent) > metrics.bending_energy(gentle)

    def test_it_is_never_negative(self, rng):
        """Non-negative by construction; a negative value would mean a bug."""
        for scale in (0.5, 2.0, 8.0):
            energy = metrics.bending_energy(
                fit(CORNERS + rng.normal(0, scale, CORNERS.shape))
            )
            assert energy >= 0

    def test_a_transform_without_coefficients_reports_zero(self):
        assert metrics.bending_energy(ThinPlateSplineTransform()) == 0.0


class TestAssess:
    """The combined report."""

    def test_a_good_fit_reports_clean(self, rng):
        src = CORNERS + rng.normal(0, 1, CORNERS.shape)
        quality = metrics.assess(fit(src), src, CORNERS, image_shape=(100, 100))

        assert not quality.has_folds
        assert quality.min_jacobian > 0
        assert not quality.outliers.any()
        assert quality.coverage is not None

    def test_a_misplaced_point_is_named(self, rng):
        src = GRID + rng.normal(0, 1, GRID.shape)
        src[6] += np.array([25.0, -20.0])

        quality = metrics.assess(fit(src, dst=GRID), src, GRID, image_shape=(100, 100))
        assert quality.worst_point == 6
        assert quality.outliers[6]

    def test_folds_are_reported(self, rng):
        src = CORNERS + rng.normal(0, 2, CORNERS.shape)
        src[[4, 5]] = src[[5, 4]]

        quality = metrics.assess(fit(src), src, CORNERS, image_shape=(100, 100))
        assert quality.has_folds
        assert quality.min_jacobian < 0

    def test_coverage_is_skipped_without_an_image_shape(self, rng):
        src = CORNERS + rng.normal(0, 1, CORNERS.shape)
        assert metrics.assess(fit(src), src, CORNERS).coverage is None

    def test_leave_one_out_can_be_skipped(self, rng):
        """It is the expensive part; large point sets may not want it."""
        src = CORNERS + rng.normal(0, 1, CORNERS.shape)
        quality = metrics.assess(fit(src), src, CORNERS, include_leave_one_out=False)
        assert quality.leave_one_out.size == 0
        assert quality.worst_point is None
        # The cheap measures still ran.
        assert quality.min_jacobian > 0

    def test_the_summary_mentions_a_problem(self, rng):
        src = GRID + rng.normal(0, 1, GRID.shape)
        src[6] += np.array([25.0, -20.0])

        summary = metrics.assess(
            fit(src, dst=GRID), src, GRID, image_shape=(100, 100)
        ).summary()
        assert "leave-one-out" in summary
        assert "check" in summary

    def test_the_summary_of_nothing_is_survivable(self):
        assert metrics.TransformQuality().summary()

    def test_median_residual_ignores_failed_points(self):
        quality = metrics.TransformQuality(leave_one_out=np.array([1.0, np.nan, 3.0]))
        assert quality.median_residual == 2.0
