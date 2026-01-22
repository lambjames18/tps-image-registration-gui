"""
RANSAC filtering for thin-plate spline deformable registration.

Based on "In Defence of RANSAC for Outlier Rejection in Deformable Registration"
by Tran et al.
"""

import numpy as np
from typing import Tuple


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

    # SVD: centered = U @ diag(S) @ Vt
    # U is NxN, S is min(N,4), Vt is 4x4
    # The right singular vectors (rows of Vt, or columns of V) give
    # the principal directions in the 4D feature space
    U, S, Vt = np.linalg.svd(centered, full_matrices=True)

    # V = Vt.T, so we take the first 2 columns of V (first 2 rows of Vt)
    # These correspond to the 2D affine subspace in the 4D space
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
    random_seed: int = None,
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

    if src_points.shape[0] < 3:
        raise ValueError("Need at least 3 point correspondences for RANSAC")
    if src_points.shape != dst_points.shape:
        raise ValueError("Source and destination points must have same shape")
    if src_points.ndim != 2 or src_points.shape[1] != 2:
        raise ValueError("Points must be Nx2 arrays")

    # Set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)

    N = src_points.shape[0]
    min_samples = 3  # Minimal set to define 2D affine subspace in 4D

    # Normalize the data
    src_norm, dst_norm, src_transform, dst_transform = _normalize_correspondences(
        src_points, dst_points
    )
    norm_threshold = threshold  # Threshold is already normalized

    best_inliers = np.zeros(N, dtype=bool)
    best_num_inliers = 0

    # RANSAC iterations
    for trial in range(max_trials):
        # Randomly sample minimal set
        sample_indices = np.random.choice(N, size=min_samples, replace=False)
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


def affine_ransac_filter(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    threshold: float = 5.5,
    max_trials: int = 1000,
) -> np.ndarray:
    """
    Alternative RANSAC implementation using scikit-image (simpler affine approach).

    This is a simpler alternative using scikit-image's built-in RANSAC,
    but it uses a full affine transform rather than fitting the 4D manifold.

    Args:
        src_points: Nx2 array of source points
        dst_points: Nx2 array of destination points
        threshold: RANSAC inlier threshold in pixels
        max_trials: Maximum number of RANSAC iterations

    Returns:
        Boolean mask indicating inlier matches
    """
    try:
        from skimage.measure import ransac
        from skimage.transform import AffineTransform
    except ImportError:
        raise ImportError("scikit-image is required for this function")

    src_points = np.asarray(src_points, dtype=np.float64)
    dst_points = np.asarray(dst_points, dtype=np.float64)

    if src_points.shape[0] < 3:
        raise ValueError("Need at least 3 point correspondences")

    # Use RANSAC with affine transform
    try:
        model, inliers = ransac(
            (src_points, dst_points),
            AffineTransform,
            min_samples=3,
            residual_threshold=threshold,
            max_trials=max_trials,
        )
        return inliers
    except Exception as e:
        raise RuntimeError(f"RANSAC fitting failed: {e}")


def projective_ransac_filter(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    threshold: float = 5.5,
    max_trials: int = 1000,
) -> np.ndarray:
    """
    Alternative RANSAC implementation using scikit-image (projective transform).

    This uses scikit-image's built-in RANSAC with a projective transform.

    Args:
        src_points: Nx2 array of source points
        dst_points: Nx2 array of destination points
        threshold: RANSAC inlier threshold in pixels
        max_trials: Maximum number of RANSAC iterations

    Returns:
        Boolean mask indicating inlier matches
    """
    try:
        from skimage.measure import ransac
        from skimage.transform import ProjectiveTransform
    except ImportError:
        raise ImportError("scikit-image is required for this function")

    src_points = np.asarray(src_points, dtype=np.float64)
    dst_points = np.asarray(dst_points, dtype=np.float64)

    if src_points.shape[0] < 4:
        raise ValueError("Need at least 4 point correspondences")

    # Use RANSAC with projective transform
    try:
        model, inliers = ransac(
            (src_points, dst_points),
            ProjectiveTransform,
            min_samples=4,
            residual_threshold=threshold,
            max_trials=max_trials,
        )
        return inliers
    except Exception as e:
        raise RuntimeError(f"RANSAC fitting failed: {e}")


def ransac_filter(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    threshold: float = 5.5,
    max_trials: int = 100,
    method: str = "deformable",
    random_seed: int = None,
) -> np.ndarray:
    """
    General RANSAC filtering interface.

    Args:
        src_points: Nx2 array of source points
        dst_points: Nx2 array of destination points
        threshold: RANSAC inlier threshold
        max_trials: Maximum number of RANSAC iterations
        method: "deformable" for Tran et al. method, "affine" for scikit-image affine RANSAC,
                "projective" or "homography" for scikit-image projective RANSAC
        random_seed: Random seed for reproducibility (only for deformable method)

    Returns:
        Boolean mask indicating inlier matches
    """
    if method == "deformable":
        return deformable_ransac_filter(
            src_points, dst_points, threshold, max_trials, random_seed
        )
    elif method == "affine":
        return affine_ransac_filter(src_points, dst_points, threshold, max_trials)
    elif method == "projective" or method == "homography":
        return projective_ransac_filter(src_points, dst_points, threshold, max_trials)
    else:
        raise ValueError(f"Unknown RANSAC method: {method}")


# Example usage and testing
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Create synthetic test data with a known warp
    np.random.seed(42)

    # Generate inliers from structured locations
    grid_size = 8
    src = np.mgrid[0 : 100 : grid_size * 1j, 0 : 100 : grid_size * 1j].T.reshape(-1, 2)
    n_inliers = len(src)

    # Apply a known warp (affine + small nonlinear)
    dst = src.copy()
    a11, a12, a13 = 0.98, 0.05, 2.0
    a21, a22, a23 = -0.02, 1.02, 1.5
    dst[:, 0] = a11 * src[:, 0] + a12 * src[:, 1] + a13
    dst[:, 1] = a21 * src[:, 0] + a22 * src[:, 1] + a23

    # Add small matching error to inliers
    dst += np.random.randn(*dst.shape) * 0.3

    # Add LARGE random outliers (completely wrong matches)
    n_outliers = 30
    src_outliers = np.random.rand(n_outliers, 2) * 100
    # Generate destination for outliers completely independently
    dst_outliers = (
        np.random.rand(n_outliers, 2) * 100 + np.random.randn(n_outliers, 2) * 20
    )

    # Combine
    src_all = np.vstack([src, src_outliers])
    dst_all = np.vstack([dst, dst_outliers])

    # Shuffle to mix inliers and outliers
    shuffle_idx = np.random.permutation(len(src_all))
    src_all = src_all[shuffle_idx]
    dst_all = dst_all[shuffle_idx]
    inlier_mask_true = np.zeros(len(src_all), dtype=bool)
    inlier_mask_true[shuffle_idx < n_inliers] = True

    # Try different thresholds
    print("=" * 70)
    print("RANSAC Filtering - Synthetic Data with Random Outliers")
    print("=" * 70)

    for threshold in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]:
        inliers = ransac_filter(
            src_all,
            dst_all,
            threshold=threshold,
            max_trials=200,
            random_seed=42,
            method="deformable",
        )

        # Measure performance
        tp = (inliers & inlier_mask_true).sum()  # True positives
        fp = (inliers & ~inlier_mask_true).sum()  # False positives
        fn = (~inliers & inlier_mask_true).sum()  # False negatives
        tn = (~inliers & ~inlier_mask_true).sum()  # True negatives

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        print(f"\nThreshold = {threshold}:")
        print(f"  Detected inliers: {inliers.sum()}")
        print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        print(f"  Recall:     {recall*100:6.1f}% (proportion of true inliers found)")
        print(
            f"  Precision:  {precision*100:6.1f}% (proportion of detections that are correct)"
        )
        print(
            f"  Specificity:{specificity*100:6.1f}% (proportion of true outliers found)"
        )

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(
            src_all[~inliers, 0],
            src_all[~inliers, 1],
            c="red",
            label="Outliers",
            alpha=0.5,
        )
        ax.scatter(
            src_all[inliers, 0],
            src_all[inliers, 1],
            c="green",
            label="Inliers",
            alpha=0.5,
        )
        ax.scatter(
            src_outliers[:, 0],
            src_outliers[:, 1],
            c="black",
            label="True Outliers",
            marker="x",
        )
        ax.legend()
        ax.set_title(f"RANSAC Inliers and Outliers (Threshold={threshold})")
        plt.show()
