"""Measures of how good a fitted transform actually is.

The obvious check -- how far the fit lands from each control point -- says
nothing about a thin-plate spline. It interpolates, so it passes exactly
through every control point it was given: residuals come out around 1e-12
whether the correspondences are good or catastrophically wrong. A point
clicked 40 pixels off its true partner is indistinguishable from a perfect one
by that measure.

What does work:

Leave-one-out
    Refit without each point in turn and see how far the fit misses it. A
    point the rest of the field disagrees with stands out immediately. This is
    the per-point number worth showing.

Jacobian determinant
    Where the mapping folds over itself, the determinant goes negative. That
    is the failure that produces a warp with mirrored patches in it, and it is
    invisible in any per-point measure because the control points themselves
    are still matched exactly.

Bending energy
    How far the warp is from a plain affine. Large values mean the spline is
    working hard, which is either real distortion or bad points.

Everything here is pure numpy, so it can be tested without a display.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from tpsreg.validation import convex_hull_area

logger = logging.getLogger(__name__)

#: Fewest control points at which leave-one-out means anything. Below this the
#: measure asks the remaining points to predict the held-out one and they
#: cannot, so every residual is large and a genuinely bad point does not stand
#: out. Measured on a regular grid with one point displaced 25 px, the bad
#: point ranks first from nine points upwards and not reliably below.
MIN_POINTS_FOR_RESIDUALS = 9

#: Modified z-score above which a leave-one-out residual is called an outlier.
#: Uses the median and MAD rather than mean and standard deviation, because a
#: bad point inflates the standard deviation enough to hide itself.
OUTLIER_THRESHOLD = 3.5

#: 0.6745 is the MAD of a standard normal, so dividing by it puts the modified
#: z-score on the same scale as an ordinary one.
_MAD_SCALE = 0.6745


@dataclass
class TransformQuality:
    """What a fitted transform looks like from the outside.

    Attributes
    ----------
    leave_one_out:
        ``(K,)`` distance, in pixels, between where each control point
        actually is and where a fit without it predicts. NaN for points whose
        reduced fit was degenerate.
    outliers:
        ``(K,)`` bool. True where the leave-one-out residual is far enough
        above the median to be worth a second look.
    worst_point:
        Index of the largest leave-one-out residual, or None if there are none.
    min_jacobian:
        Smallest Jacobian determinant over the sampled grid. Negative means
        the mapping folds over itself somewhere.
    folded_fraction:
        Fraction of the sampled grid where the determinant is not positive.
    bending_energy:
        How far the warp departs from an affine. Zero for a pure affine.
    coverage:
        Fraction of the image enclosed by the control points, or None when no
        image shape was supplied.
    """

    leave_one_out: np.ndarray = field(default_factory=lambda: np.empty(0))
    outliers: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    worst_point: int | None = None
    min_jacobian: float = float("nan")
    folded_fraction: float = 0.0
    bending_energy: float = 0.0
    coverage: float | None = None

    @property
    def has_folds(self) -> bool:
        """True if the mapping turns itself inside out anywhere."""
        return self.folded_fraction > 0

    @property
    def median_residual(self) -> float:
        """Typical leave-one-out residual, ignoring points that failed."""
        if self.leave_one_out.size == 0:
            return float("nan")
        return float(np.nanmedian(self.leave_one_out))

    def summary(self) -> str:
        """One-line description, for a status bar."""
        if self.leave_one_out.size == 0:
            return "No quality metrics available."

        parts = [f"median leave-one-out {self.median_residual:.2f} px"]
        n_outliers = int(np.count_nonzero(self.outliers))
        if n_outliers:
            parts.append(
                f"{n_outliers} point{'s' if n_outliers != 1 else ''} to check"
                + (
                    f" (worst: {self.worst_point})"
                    if self.worst_point is not None
                    else ""
                )
            )
        if self.has_folds:
            parts.append(f"folded over {self.folded_fraction:.1%} of the image")
        if self.coverage is not None:
            parts.append(f"covering {self.coverage:.0%}")
        return "; ".join(parts)


def leave_one_out_residuals(
    src: np.ndarray,
    dst: np.ndarray,
) -> np.ndarray:
    """Distance each control point falls from a fit that excludes it.

    A thin-plate spline interpolates, so its residual at its own control
    points is zero by construction and tells you nothing. Dropping a point and
    asking the rest of the field where it should have been does tell you
    something: a mistyped correspondence disagrees with its neighbours, and
    this is what that disagreement looks like in pixels.

    Parameters
    ----------
    src, dst:
        ``(K, 2)`` corresponding control points.

    Returns
    -------
    np.ndarray
        ``(K,)`` distances in pixels. NaN where the reduced point set was
        itself degenerate -- three points minus one is two, which cannot be
        fitted -- so a small point set yields all-NaN rather than an error.

    Notes
    -----
    This refits K times, each an ``O(K**3)`` solve, so it is ``O(K**4)``
    overall: about 15 ms at 50 control points, 0.3 s at 200, 2.5 s at 400.
    Fine on demand, too slow to run on every edit.

    It needs enough points to be meaningful. The measure asks the remaining
    points to predict the held-out one, and below roughly nine well-spread
    points that question has no good answer for *any* of them: every residual
    comes out large and a genuinely bad point stops standing out. Measured on
    a regular grid with one point displaced 25 px, the bad point ranks first
    from nine points upwards, and by a widening margin as the set gets denser.
    Treat a handful of points as too few to judge this way.
    """
    from tpsreg.tps import ThinPlateSplineTransform

    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)

    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
        raise ValueError(
            f"Expected matching (K, 2) point arrays, got {src.shape} and {dst.shape}."
        )

    n_points = len(src)
    residuals = np.full(n_points, np.nan)

    for index in range(n_points):
        keep = np.ones(n_points, dtype=bool)
        keep[index] = False

        try:
            reduced = ThinPlateSplineTransform()
            reduced.estimate(src[keep], dst[keep])
            predicted = reduced.map(dst[index : index + 1])[0]
        except ValueError as exc:
            # Removing a point can leave the rest collinear or too few to fit.
            # That is information about the point set, not an error worth
            # aborting the whole assessment for.
            logger.debug("Leave-one-out fit failed at point %d: %s", index, exc)
            continue

        residuals[index] = float(np.linalg.norm(predicted - src[index]))

    return residuals


def flag_outliers(
    residuals: np.ndarray, threshold: float = OUTLIER_THRESHOLD
) -> np.ndarray:
    """Mark residuals far enough above the median to be suspicious.

    Uses the median and the median absolute deviation rather than the mean and
    standard deviation. One badly placed point inflates the standard deviation
    enough to bring itself back inside the threshold, which is exactly the
    case this needs to catch.

    Returns
    -------
    np.ndarray
        ``(K,)`` bool. All False when everything looks alike, or when there is
        too little to compare against.
    """
    residuals = np.asarray(residuals, dtype=float)
    flags = np.zeros(residuals.shape, dtype=bool)

    finite = np.isfinite(residuals)
    if np.count_nonzero(finite) < 3:
        return flags

    values = residuals[finite]
    median = np.median(values)
    mad = np.median(np.abs(values - median))

    if mad == 0:
        # Every residual is identical, so nothing stands out -- unless one is
        # simply much bigger than a uniform zero baseline.
        flags[finite] = values > median
        return flags

    scores = _MAD_SCALE * (values - median) / mad
    flags[finite] = scores > threshold
    return flags


def jacobian_determinant(
    tform,
    size: tuple[int, int] | None = None,
    downsample: int = 8,
) -> np.ndarray:
    """Determinant of the mapping's Jacobian across the destination grid.

    Where this is positive the mapping preserves orientation and the warp is
    well behaved. Where it is negative the mapping has folded over itself, and
    the warped image will contain mirrored patches. Folds are invisible to any
    per-point measure, because the control points are still matched exactly on
    either side of one.

    Parameters
    ----------
    tform:
        A fitted transform exposing ``map``.
    size:
        ``(height, width)`` to cover. Defaults to the transform's own size.
    downsample:
        Sample every Nth pixel. Folds are regional, so a coarse grid finds
        them; the default keeps this at a few milliseconds even for a large
        image.

    Returns
    -------
    np.ndarray
        ``(h, w)`` determinants over the sampled grid.
    """
    size = size or getattr(tform, "size", None)
    if size is None:
        raise ValueError(
            "No grid size available. Pass size=(height, width), or use a "
            "transform that was estimated against one."
        )

    height, width = int(size[0]), int(size[1])
    step = max(1, int(downsample))

    xs = np.arange(0, width, step, dtype=float)
    ys = np.arange(0, height, step, dtype=float)
    grid_x, grid_y = np.meshgrid(xs, ys)

    mapped = tform.map(np.column_stack([grid_x.ravel(), grid_y.ravel()]))
    field_x = mapped[:, 0].reshape(grid_x.shape)
    field_y = mapped[:, 1].reshape(grid_x.shape)

    if field_x.shape[0] < 2 or field_x.shape[1] < 2:
        # np.gradient needs at least two samples along each axis.
        return np.ones_like(field_x)

    # np.gradient returns (d/drow, d/dcol); rows step in y, columns in x.
    dxdy, dxdx = np.gradient(field_x, step, step)
    dydy, dydx = np.gradient(field_y, step, step)

    return dxdx * dydy - dxdy * dydx


def folded_fraction(determinants: np.ndarray) -> float:
    """Fraction of the sampled grid where the mapping is not orientation preserving."""
    determinants = np.asarray(determinants, dtype=float)
    if determinants.size == 0:
        return 0.0
    return float(np.count_nonzero(determinants <= 0) / determinants.size)


def bending_energy(tform) -> float:
    """How far the warp departs from a plain affine.

    The spline's weights against its own kernel, ``w' K w``. Zero for a pure
    affine -- no bending at all -- and growing as the deformation gets more
    local. Useful as a relative measure between fits of the same scene rather
    than as an absolute number, since it carries units of the coordinates.

    The quantity is non-negative by construction, and is returned as computed
    rather than through ``abs``: a negative value would mean the weights and
    the kernel disagree about the transform, which is worth surfacing rather
    than hiding.
    """
    from scipy.spatial.distance import cdist

    from tpsreg.tps import _kernel

    control_points = getattr(tform, "control_points", None)
    weights = getattr(tform, "coefficients", None)
    if control_points is None or weights is None:
        return 0.0

    weights = weights[:-3]
    kernel = _kernel(cdist(control_points, control_points, "euclidean"))
    np.fill_diagonal(kernel, 0)

    return float(np.trace(weights.T @ kernel @ weights))


def assess(
    tform,
    src: np.ndarray,
    dst: np.ndarray,
    image_shape: tuple[int, ...] | None = None,
    downsample: int = 8,
    include_leave_one_out: bool = True,
) -> TransformQuality:
    """Collect every measure into one report.

    Parameters
    ----------
    tform:
        The fitted transform being assessed.
    src, dst:
        The ``(K, 2)`` correspondences it was fitted to.
    image_shape:
        ``(height, width, ...)`` of the source image, for the coverage figure.
    downsample:
        Grid spacing for the Jacobian.
    include_leave_one_out:
        The expensive part. Turn it off for a quick check on a large point
        set; see :func:`leave_one_out_residuals` for the cost.
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)

    quality = TransformQuality()

    if include_leave_one_out and len(src):
        quality.leave_one_out = leave_one_out_residuals(src, dst)
        quality.outliers = flag_outliers(quality.leave_one_out)
        if np.any(np.isfinite(quality.leave_one_out)):
            quality.worst_point = int(np.nanargmax(quality.leave_one_out))

    try:
        determinants = jacobian_determinant(tform, downsample=downsample)
        quality.min_jacobian = float(determinants.min())
        quality.folded_fraction = folded_fraction(determinants)
    except ValueError as exc:
        logger.debug("Could not evaluate the Jacobian: %s", exc)

    quality.bending_energy = bending_energy(tform)

    if image_shape is not None and len(image_shape) >= 2:
        area = float(image_shape[0]) * float(image_shape[1])
        if area > 0:
            quality.coverage = convex_hull_area(src) / area

    return quality
