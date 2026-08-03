"""Tests for RANSAC outlier rejection."""

from __future__ import annotations

import numpy as np
import pytest

from tpsreg.ransac import (
    _distance_to_affine_subspace,
    _fit_affine_subspace,
    _normalize_correspondences,
    affine_ransac_filter,
    deformable_ransac_filter,
    projective_ransac_filter,
    ransac_filter,
)


@pytest.fixture
def correspondences_with_outliers(rng):
    """A known affine warp with 64 inliers and 30 gross outliers.

    Returns ``(src, dst, true_inlier_mask)`` with the rows shuffled so index
    order carries no information.
    """
    grid = np.mgrid[0:100:8j, 0:100:8j].T.reshape(-1, 2)
    n_inliers = len(grid)

    warped = np.empty_like(grid)
    warped[:, 0] = 0.98 * grid[:, 0] + 0.05 * grid[:, 1] + 2.0
    warped[:, 1] = -0.02 * grid[:, 0] + 1.02 * grid[:, 1] + 1.5
    warped += rng.normal(0, 0.3, warped.shape)

    n_outliers = 30
    src_outliers = rng.random((n_outliers, 2)) * 100
    dst_outliers = rng.random((n_outliers, 2)) * 100 + rng.normal(
        0, 20, (n_outliers, 2)
    )

    src = np.vstack([grid, src_outliers])
    dst = np.vstack([warped, dst_outliers])

    truth = np.zeros(len(src), dtype=bool)
    truth[:n_inliers] = True

    order = rng.permutation(len(src))
    return src[order], dst[order], truth[order]


class TestNormalization:
    """Correspondence normalization makes thresholds unit-independent."""

    def test_centroid_moves_to_origin(self, rng):
        src = rng.random((20, 2)) * 500 + 1000
        dst = rng.random((20, 2)) * 3

        src_norm, dst_norm, _, _ = _normalize_correspondences(src, dst)

        np.testing.assert_allclose(src_norm.mean(axis=0), [0, 0], atol=1e-10)
        np.testing.assert_allclose(dst_norm.mean(axis=0), [0, 0], atol=1e-10)

    def test_mean_distance_becomes_sqrt_two(self, rng):
        src = rng.random((20, 2)) * 500
        dst = rng.random((20, 2)) * 500

        src_norm, dst_norm, _, _ = _normalize_correspondences(src, dst)

        for points in (src_norm, dst_norm):
            mean_dist = np.sqrt((points**2).sum(axis=1).mean())
            assert mean_dist == pytest.approx(np.sqrt(2))

    def test_identical_points_do_not_divide_by_zero(self):
        """A degenerate set has zero spread; scale must fall back to 1."""
        points = np.zeros((5, 2))
        src_norm, dst_norm, (_, src_scale), _ = _normalize_correspondences(
            points, points
        )
        assert src_scale == 1.0
        assert np.all(np.isfinite(src_norm))
        assert np.all(np.isfinite(dst_norm))


class TestAffineSubspace:
    """Fitting and measuring against the 4D correspondence subspace."""

    def test_exact_affine_data_has_zero_residual(self, rng):
        src = rng.random((30, 2))
        dst = np.column_stack([2 * src[:, 0] + 1, 3 * src[:, 1] - 2])

        centroid, basis = _fit_affine_subspace(src, dst)
        distances = _distance_to_affine_subspace(src, dst, centroid, basis)

        np.testing.assert_allclose(distances, 0, atol=1e-10)

    def test_outlier_has_large_residual(self, rng):
        src = rng.random((30, 2))
        dst = src * 2

        centroid, basis = _fit_affine_subspace(src, dst)

        outlier_src = np.array([[0.5, 0.5]])
        outlier_dst = np.array([[50.0, -30.0]])
        distance = _distance_to_affine_subspace(
            outlier_src, outlier_dst, centroid, basis
        )
        assert distance[0] > 1.0

    def test_basis_is_orthonormal(self, rng):
        src = rng.random((30, 2))
        dst = rng.random((30, 2))
        _, basis = _fit_affine_subspace(src, dst)
        np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=1e-10)


class TestDeformableFilter:
    """The Tran et al. deformable method."""

    def test_separates_inliers_from_outliers(self, correspondences_with_outliers):
        src, dst, truth = correspondences_with_outliers

        inliers = deformable_ransac_filter(
            src, dst, threshold=0.2, max_trials=200, random_seed=42
        )

        recall = (inliers & truth).sum() / truth.sum()
        precision = (inliers & truth).sum() / max(inliers.sum(), 1)
        assert recall > 0.9, f"missed too many true inliers (recall={recall:.2f})"
        assert precision > 0.9, f"kept too many outliers (precision={precision:.2f})"

    def test_returns_boolean_mask_of_right_length(self, rng):
        src = rng.random((25, 2)) * 100
        dst = src + rng.normal(0, 0.5, src.shape)
        inliers = deformable_ransac_filter(src, dst, random_seed=1)
        assert inliers.dtype == bool
        assert inliers.shape == (25,)

    def test_seeded_runs_are_reproducible(self, correspondences_with_outliers):
        src, dst, _ = correspondences_with_outliers
        first = deformable_ransac_filter(src, dst, max_trials=50, random_seed=7)
        second = deformable_ransac_filter(src, dst, max_trials=50, random_seed=7)
        np.testing.assert_array_equal(first, second)

    def test_does_not_disturb_global_numpy_random_state(self, rng):
        """A library must not reseed the caller's global RNG."""
        src = rng.random((20, 2)) * 100
        dst = src * 1.1

        np.random.seed(1234)
        expected = np.random.random(3)

        np.random.seed(1234)
        deformable_ransac_filter(src, dst, max_trials=20, random_seed=99)
        actual = np.random.random(3)

        np.testing.assert_array_equal(expected, actual)

    def test_tighter_threshold_keeps_fewer_points(self, correspondences_with_outliers):
        src, dst, _ = correspondences_with_outliers
        loose = deformable_ransac_filter(
            src, dst, threshold=1.0, max_trials=100, random_seed=3
        )
        tight = deformable_ransac_filter(
            src, dst, threshold=0.05, max_trials=100, random_seed=3
        )
        assert tight.sum() <= loose.sum()

    @pytest.mark.parametrize(
        ("src_shape", "dst_shape", "message"),
        [
            ((2, 2), (2, 2), "at least 3"),
            ((5, 2), (4, 2), "same shape"),
            ((5, 3), (5, 3), "Nx2"),
        ],
    )
    def test_invalid_input_rejected(self, src_shape, dst_shape, message):
        src = np.zeros(src_shape)
        dst = np.zeros(dst_shape)
        with pytest.raises(ValueError, match=message):
            deformable_ransac_filter(src, dst)


class TestParametricFilters:
    """The scikit-image-backed affine and projective methods."""

    def test_affine_filter_finds_inliers(self, correspondences_with_outliers):
        src, dst, truth = correspondences_with_outliers
        inliers = affine_ransac_filter(
            src, dst, threshold=3.0, max_trials=500, random_seed=42
        )
        recall = (inliers & truth).sum() / truth.sum()
        assert recall > 0.8

    def test_projective_filter_finds_inliers(self, correspondences_with_outliers):
        src, dst, truth = correspondences_with_outliers
        inliers = projective_ransac_filter(
            src, dst, threshold=3.0, max_trials=500, random_seed=42
        )
        recall = (inliers & truth).sum() / truth.sum()
        assert recall > 0.8

    def test_affine_needs_three_points(self):
        with pytest.raises(ValueError, match="at least 3"):
            affine_ransac_filter(np.zeros((2, 2)), np.zeros((2, 2)))

    def test_projective_needs_four_points(self):
        with pytest.raises(ValueError, match="at least 4"):
            projective_ransac_filter(np.zeros((3, 2)), np.zeros((3, 2)))


class TestDispatch:
    """The ransac_filter front door."""

    @pytest.mark.parametrize(
        "method", ["deformable", "affine", "projective", "homography"]
    )
    def test_known_methods_return_a_mask(self, method, rng):
        src = rng.random((30, 2)) * 100
        dst = src * 1.05 + rng.normal(0, 0.2, (30, 2))
        threshold = 0.2 if method == "deformable" else 5.0

        inliers = ransac_filter(
            src, dst, threshold=threshold, method=method, random_seed=5
        )
        assert inliers.dtype == bool
        assert inliers.shape == (30,)

    def test_unknown_method_rejected(self, rng):
        src = rng.random((10, 2))
        with pytest.raises(ValueError, match="Unknown RANSAC method"):
            ransac_filter(src, src, method="magic")
