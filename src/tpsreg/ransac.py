"""
RANSAC filtering for thin-plate spline deformable registration.

Based on "In Defence of RANSAC for Outlier Rejection in Deformable Registration"
by Tran et al.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _normalize_correspondences(
    src_points: np.ndarray,
    dst_points: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize correspondence data for RANSAC.

    The paper normalizes data so that the centroid lies at origin and
    mean distance to origin is sqrt(2).

    Args:
        src_points: Nx2 array of source points
        dst_points: Nx2 array of destination points

    Returns:
        Tuple of (normalized_src, normalized_dst, src_transform, dst_transform)
        where transforms are (centroid, scale)
    """
    # Normalize source points
    src_centroid = src_points.mean(axis=0)
    src_centered = src_points - src_centroid
    src_mean_dist = np.sqrt((src_centered**2).sum(axis=1).mean())
    src_scale = np.sqrt(2) / src_mean_dist if src_mean_dist > 0 else 1.0
    src_normalized = src_centered * src_scale

    # Normalize destination points
    dst_centroid = dst_points.mean(axis=0)
    dst_centered = dst_points - dst_centroid
    dst_mean_dist = np.sqrt((dst_centered**2).sum(axis=1).mean())
    dst_scale = np.sqrt(2) / dst_mean_dist if dst_mean_dist > 0 else 1.0
    dst_normalized = dst_centered * dst_scale

    return (
        src_normalized,
        dst_normalized,
        (src_centroid, src_scale),
        (dst_centroid, dst_scale),
    )


def _fit_affine_subspace(
    src_points: np.ndarray,
    dst_points: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a 2D affine subspace to correspondences in 4D space.

    Each correspondence is represented as xi = [xi, yi, x'i, y'i]^T in R^4.
    We fit a 2D affine subspace using SVD on the centered data.

    Args:
        src_points: Nx2 array of source points
        dst_points: Nx2 array of destination points

    Returns:
        Tuple of (centroid, basis_vectors) where:
        - centroid: 4D mean point
        - basis_vectors: 4x2 matrix of the two principal directions in feature space
    """
    # Create 4D correspondence matrix
    correspondences = np.hstack([src_points, dst_points])  # Nx4

    # Center the data
    centroid = correspondences.mean(axis=0)
    centered = correspondences - centroid  # Nx4

    # SVD: centered = U @ diag(S) @ Vt. The right singular vectors (rows of Vt)
    # give the principal directions in the 4D feature space.
    _, _, Vt = np.linalg.svd(centered, full_matrices=True)

    # V = Vt.T, so the first 2 columns of V (first 2 rows of Vt) span the 2D
    # affine subspace within the 4D correspondence space.
    basis = Vt[:2, :].T  # 4x2

    return centroid, basis


def _distance_to_affine_subspace(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    centroid: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    """
    Compute orthogonal distance from correspondences to fitted affine subspace.

    Args:
        src_points: Nx2 array of source points
        dst_points: Nx2 array of destination points
        centroid: 4D centroid of the subspace
        basis: 4xK matrix of basis vectors (K <= 2 for 2D subspace, but may be less with few samples)

    Returns:
        Array of N distances
    """
    # Create 4D correspondence matrix
    correspondences = np.hstack([src_points, dst_points])  # Nx4

    # Center relative to subspace centroid
    centered = correspondences - centroid  # Nx4

    # Project onto the subspace spanned by basis vectors
    # projection = basis @ (basis.T @ centered.T)
    # We need to be careful with dimensions since basis may have fewer than 2 columns
    projection = (centered @ basis) @ basis.T  # Nx4

    # Compute residual (orthogonal distance)
    residual = centered - projection  # Nx4
    distances = np.sqrt((residual**2).sum(axis=1))  # N

    return distances


def deformable_ransac_filter(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    threshold: float = 0.2,
    max_trials: int = 100,
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """
    Apply RANSAC filtering to remove outlier matches for deformable registration.

    This implements the approach from Tran et al., fitting a 2D affine subspace
    in the 4D correspondence space (x, y, x', y') to identify inlier matches.

    Args:
        src_points: Nx2 array of source points (template)
        dst_points: Nx2 array of destination points (target)

        threshold: RANSAC inlier threshold (default: 0.2).
                   IMPORTANT: This is in the NORMALIZED coordinate space!

                   Data is normalized so mean distance = √2, making threshold
                   unit-independent. Typical ranges:

                   - 0.05-0.15: Very conservative, rejects many outliers
                   - 0.15-0.30: Standard, good balance
                   - 0.30-0.50: Permissive, allows more outliers
                   - 1.0+:      Very permissive, likely includes outliers

                   Inliers typically cluster at distances 0.01-0.05.
                   Outliers typically appear at distances > 0.2.

                   The old default of 5.5 was designed for unnormalized
                   pixels - use 0.2 instead for normalized data!

        max_trials: Maximum number of RANSAC iterations (default: 100).
                    Increase for higher outlier rates (20% outliers → 100,
                    50% outliers → 500).

        random_seed: Random seed for reproducibility (optional)

    Returns:
        Boolean array of shape (N,) indicating inlier matches (True = inlier)

    Raises:
        ValueError: If input points have invalid shape or insufficient points

    Example:
        >>> src = np.array([[10, 20], [30, 40], ...])  # Nx2
        >>> dst = np.array([[12, 22], [31, 41], ...])  # Nx2
        >>> inliers = ransac_filter(src, dst, threshold=0.2)
        >>> inlier_src = src[inliers]
        >>> inlier_dst = dst[inliers]
    """
    # Validate inputs
    src_points = np.asarray(src_points, dtype=np.float64)
    dst_points = np.asarray(dst_points, dtype=np.float64)

    if src_points.ndim != 2 or src_points.shape[1] != 2:
        raise ValueError("Points must be Nx2 arrays")
    if src_points.shape != dst_points.shape:
        raise ValueError("Source and destination points must have same shape")
    if src_points.shape[0] < 3:
        raise ValueError("Need at least 3 point correspondences for RANSAC")

    # A local generator keeps the caller's global numpy random state intact.
    rng = np.random.default_rng(random_seed)

    N = src_points.shape[0]
    min_samples = 3  # Minimal set to define 2D affine subspace in 4D

    # Normalize the data
    src_norm, dst_norm, _, _ = _normalize_correspondences(src_points, dst_points)
    norm_threshold = threshold  # Threshold is already normalized

    best_inliers = np.zeros(N, dtype=bool)
    best_num_inliers = 0

    # RANSAC iterations
    for _ in range(max_trials):
        # Randomly sample minimal set
        sample_indices = rng.choice(N, size=min_samples, replace=False)
        sample_src = src_norm[sample_indices]
        sample_dst = dst_norm[sample_indices]

        try:
            # Fit 2D affine subspace to minimal sample
            centroid, basis = _fit_affine_subspace(sample_src, sample_dst)

            # Compute distances for all correspondences
            distances = _distance_to_affine_subspace(
                src_norm, dst_norm, centroid, basis
            )

            # Identify inliers
            inliers = distances <= norm_threshold
            num_inliers = inliers.sum()

            # Update best model if this is better
            if num_inliers > best_num_inliers:
                best_num_inliers = num_inliers
                best_inliers = inliers.copy()

        except np.linalg.LinAlgError:
            # Skip degenerate sample
            continue

    # Refit with all inliers for final model (optional but recommended)
    if best_num_inliers >= min_samples:
        inlier_src = src_norm[best_inliers]
        inlier_dst = dst_norm[best_inliers]

        try:
            centroid, basis = _fit_affine_subspace(inlier_src, inlier_dst)
            distances = _distance_to_affine_subspace(
                src_norm, dst_norm, centroid, basis
            )
            best_inliers = distances <= norm_threshold
        except np.linalg.LinAlgError:
            pass  # Keep previous result

    return best_inliers


def _skimage_ransac_filter(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    model_class: type,
    min_samples: int,
    threshold: float,
    max_trials: int,
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """Run scikit-image's RANSAC with a parametric transform model.

    Returns
    -------
    np.ndarray
        Boolean inlier mask.

    Raises
    ------
    ValueError
        If there are fewer than ``min_samples`` correspondences.
    RuntimeError
        If RANSAC could not fit a model.
    """
    from skimage.measure import ransac

    src_points = np.asarray(src_points, dtype=np.float64)
    dst_points = np.asarray(dst_points, dtype=np.float64)

    if src_points.shape != dst_points.shape:
        raise ValueError("Source and destination points must have same shape")
    if src_points.shape[0] < min_samples:
        raise ValueError(f"Need at least {min_samples} point correspondences")

    try:
        model, inliers = ransac(
            (src_points, dst_points),
            model_class,
            min_samples=min_samples,
            residual_threshold=threshold,
            max_trials=max_trials,
            rng=random_seed,
        )
    except TypeError:
        # scikit-image renamed random_state -> rng in 0.25.
        model, inliers = ransac(
            (src_points, dst_points),
            model_class,
            min_samples=min_samples,
            residual_threshold=threshold,
            max_trials=max_trials,
            random_state=random_seed,
        )
    except Exception as exc:
        raise RuntimeError(f"RANSAC fitting failed: {exc}") from exc

    # skimage returns (None, None) when no model reaches the inlier threshold.
    if model is None or inliers is None:
        raise RuntimeError(
            f"RANSAC could not fit a {model_class.__name__} to the given "
            f"correspondences; try increasing the threshold or max_trials."
        )

    return inliers


def affine_ransac_filter(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    threshold: float = 5.5,
    max_trials: int = 1000,
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """Filter correspondences with a global affine model.

    Simpler than the deformable method: it assumes a single affine transform
    explains every inlier, rather than fitting the 4D correspondence manifold.

    Args:
        src_points: Nx2 array of source points
        dst_points: Nx2 array of destination points
        threshold: RANSAC inlier threshold in pixels
        max_trials: Maximum number of RANSAC iterations
        random_seed: Seed for reproducibility

    Returns:
        Boolean mask indicating inlier matches
    """
    from skimage.transform import AffineTransform

    return _skimage_ransac_filter(
        src_points,
        dst_points,
        AffineTransform,
        min_samples=3,
        threshold=threshold,
        max_trials=max_trials,
        random_seed=random_seed,
    )


def projective_ransac_filter(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    threshold: float = 5.5,
    max_trials: int = 1000,
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """Filter correspondences with a global projective (homography) model.

    Args:
        src_points: Nx2 array of source points
        dst_points: Nx2 array of destination points
        threshold: RANSAC inlier threshold in pixels
        max_trials: Maximum number of RANSAC iterations
        random_seed: Seed for reproducibility

    Returns:
        Boolean mask indicating inlier matches
    """
    from skimage.transform import ProjectiveTransform

    return _skimage_ransac_filter(
        src_points,
        dst_points,
        ProjectiveTransform,
        min_samples=4,
        threshold=threshold,
        max_trials=max_trials,
        random_seed=random_seed,
    )


def ransac_filter(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    threshold: float = 5.5,
    max_trials: int = 100,
    method: str = "deformable",
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """
    General RANSAC filtering interface.

    Args:
        src_points: Nx2 array of source points
        dst_points: Nx2 array of destination points
        threshold: RANSAC inlier threshold. Note the scale differs by method:
                   the deformable method works in normalized coordinates
                   (~0.05-0.5) while the affine and projective methods work in
                   pixels (~5.0).
        max_trials: Maximum number of RANSAC iterations
        method: "deformable" for the Tran et al. method, "affine" for
                scikit-image affine RANSAC, "projective"/"homography" for
                scikit-image projective RANSAC
        random_seed: Random seed for reproducibility

    Returns:
        Boolean mask indicating inlier matches
    """
    if method == "deformable":
        return deformable_ransac_filter(
            src_points, dst_points, threshold, max_trials, random_seed
        )
    elif method == "affine":
        return affine_ransac_filter(
            src_points, dst_points, threshold, max_trials, random_seed
        )
    elif method in ("projective", "homography"):
        return projective_ransac_filter(
            src_points, dst_points, threshold, max_trials, random_seed
        )
    else:
        raise ValueError(f"Unknown RANSAC method: {method}")

