"""Checks that run over control points before a transform is estimated.

Estimating a thin-plate spline over bad control points does not fail politely.
Coincident source points make the system matrix singular, collinear points make
the affine block singular, and points huddled in one corner produce a fit that
extrapolates wildly everywhere else. All three used to surface as a LinAlgError
or a nonsense image several seconds later.

Everything here is pure numpy so the view can ask "is this going to work?"
before it starts, and so the checks can be tested without a display.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)

#: Control points needed before a transform can be estimated at all.
MINIMUM_POINTS: dict[str, int] = {
    "tps": 3,
    "tps_affine": 3,
}

#: Fallback when a transform type is not listed above.
DEFAULT_MINIMUM_POINTS = 3

#: Below this, a TPS has so little to work with that it is effectively affine.
COMFORTABLE_POINTS = 6

#: Two points closer than this (in pixels) count as the same point.
DUPLICATE_TOLERANCE = 1.0

#: Points spanning less of the image than this get a coverage warning.
COVERAGE_WARNING_FRACTION = 0.1

#: Ratio of the smaller to the larger spread of the point cloud, below which
#: the points are treated as lying on a line.
COLLINEAR_TOLERANCE = 1e-3


class Severity(Enum):
    """How much a problem matters."""

    #: Estimation cannot succeed; refuse to start.
    ERROR = "error"
    #: Estimation will run, but the result is likely to disappoint.
    WARNING = "warning"


@dataclass(frozen=True)
class Issue:
    """One problem found in a set of control points.

    Attributes
    ----------
    severity:
        Whether this blocks estimation or merely warns about it.
    code:
        Stable identifier, for tests and for callers that want to react to a
        specific problem rather than parse prose.
    message:
        Text shown to the user. Says what is wrong and what to do about it.
    """

    severity: Severity
    code: str
    message: str

    @property
    def is_error(self) -> bool:
        return self.severity is Severity.ERROR


def minimum_points(transform_type: object) -> int:
    """Control points needed by a transform type.

    Accepts a ``TransformType`` member, its value, or any string; anything
    unrecognised falls back to the three points a TPS needs, which is the
    smallest useful number for the transforms this application offers.
    """
    name = getattr(transform_type, "value", transform_type)
    if not isinstance(name, str):
        return DEFAULT_MINIMUM_POINTS
    return MINIMUM_POINTS.get(name.lower(), DEFAULT_MINIMUM_POINTS)


def _as_2d(points: object) -> np.ndarray:
    """Coerce a point collection to an ``(N, 2)`` float array.

    An empty or malformed input becomes ``(0, 2)`` rather than raising, so the
    "you have no points" message wins over a shape error.
    """
    array = np.asarray(points, dtype=float)
    if array.size == 0:
        return np.empty((0, 2), dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] < 2:
        return np.empty((0, 2), dtype=float)
    return array[:, :2]


def _duplicate_pairs(points: np.ndarray, tolerance: float) -> list[tuple[int, int]]:
    """Indices of point pairs that sit on top of each other.

    Uses the full pairwise distance matrix. Control point counts are in the
    tens or low hundreds, so the N**2 memory is a few hundred kilobytes at
    worst and the clarity is worth more than the saving.
    """
    if len(points) < 2:
        return []

    deltas = points[:, None, :] - points[None, :, :]
    distances = np.hypot(deltas[..., 0], deltas[..., 1])
    rows, cols = np.nonzero(np.triu(distances <= tolerance, k=1))
    return [(int(r), int(c)) for r, c in zip(rows, cols, strict=True)]


def _is_collinear(points: np.ndarray, tolerance: float = COLLINEAR_TOLERANCE) -> bool:
    """True when every point lies on (or very near) a single straight line.

    Compares the two singular values of the centred coordinates: a line has
    all its spread along one direction, so the second value collapses to zero.
    """
    if len(points) < 3:
        # Two points are always collinear, but that is caught by the
        # "too few points" check, which gives a more useful message.
        return False

    centred = points - points.mean(axis=0)
    singular_values = np.linalg.svd(centred, compute_uv=False)
    if singular_values[0] == 0:
        # Every point is identical; the duplicate check reports that better.
        return False
    return bool(singular_values[1] / singular_values[0] < tolerance)


def convex_hull_area(points: np.ndarray) -> float:
    """Area of the convex hull of a point cloud.

    Implemented directly (monotone chain, then the shoelace formula) rather
    than pulled from scipy, which is not otherwise a dependency. Degenerate
    inputs -- fewer than three points, or collinear ones -- have zero area.
    """
    points = _as_2d(points)
    if len(points) < 3:
        return 0.0

    ordered = np.unique(points, axis=0)
    ordered = ordered[np.lexsort((ordered[:, 1], ordered[:, 0]))]
    if len(ordered) < 3:
        return 0.0

    def cross(o, a, b) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def half(sequence) -> list:
        chain: list = []
        for point in sequence:
            while len(chain) >= 2 and cross(chain[-2], chain[-1], point) <= 0:
                chain.pop()
            chain.append(point)
        return chain

    hull = half(ordered)[:-1] + half(ordered[::-1])[:-1]
    if len(hull) < 3:
        return 0.0

    hull_array = np.array(hull)
    x, y = hull_array[:, 0], hull_array[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def check_control_points(
    src_points: object,
    dst_points: object,
    transform_type: object = "tps",
    image_shape: tuple[int, ...] | None = None,
    duplicate_tolerance: float = DUPLICATE_TOLERANCE,
    coverage_fraction: float = COVERAGE_WARNING_FRACTION,
) -> list[Issue]:
    """Inspect a set of correspondences and report what will go wrong.

    Parameters
    ----------
    src_points, dst_points:
        Corresponding control points, ``(N, 2)`` in ``(x, y)`` order.
    transform_type:
        A ``TransformType`` member or its value; decides the minimum count.
    image_shape:
        ``(height, width, ...)`` of the source image. Only needed for the
        coverage warning, which is skipped when it is not supplied.
    duplicate_tolerance:
        Distance in pixels below which two points count as coincident.
    coverage_fraction:
        Warn when the points enclose less than this fraction of the image.

    Returns
    -------
    list[Issue]
        Errors first, then warnings. Empty means nothing looks wrong.

    Notes
    -----
    Structural problems short-circuit: if the counts do not match there is no
    meaningful geometry to inspect, so the geometric checks are skipped rather
    than piling confusing extra messages onto the real one.
    """
    src = _as_2d(src_points)
    dst = _as_2d(dst_points)
    issues: list[Issue] = []

    if len(src) == 0 or len(dst) == 0:
        return [
            Issue(
                Severity.ERROR,
                "no_points",
                "No control points have been placed. Click matching features "
                "in both images to create at least "
                f"{minimum_points(transform_type)} pairs.",
            )
        ]

    if len(src) != len(dst):
        return [
            Issue(
                Severity.ERROR,
                "count_mismatch",
                f"There are {len(src)} source points but {len(dst)} "
                "destination points. Every point needs a partner; finish or "
                "remove the incomplete pair.",
            )
        ]

    required = minimum_points(transform_type)
    if len(src) < required:
        issues.append(
            Issue(
                Severity.ERROR,
                "too_few_points",
                f"Only {len(src)} point pair{'s' if len(src) != 1 else ''} "
                f"placed; at least {required} are needed to estimate this "
                "transform.",
            )
        )

    # Coincident source points give the spline two different answers at the
    # same location, which makes its system matrix singular.
    src_duplicates = _duplicate_pairs(src, duplicate_tolerance)
    if src_duplicates:
        first, second = src_duplicates[0]
        issues.append(
            Issue(
                Severity.ERROR,
                "duplicate_source_points",
                f"Source points {first} and {second} are in the same place "
                f"({len(src_duplicates)} such pair"
                f"{'s' if len(src_duplicates) != 1 else ''} in total). "
                "Move or delete one of each; the fit cannot resolve two "
                "answers at one location.",
            )
        )

    # Coincident destination points are solvable, just contradictory: two
    # separate source features are being sent to the same spot.
    dst_duplicates = _duplicate_pairs(dst, duplicate_tolerance)
    if dst_duplicates:
        first, second = dst_duplicates[0]
        issues.append(
            Issue(
                Severity.WARNING,
                "duplicate_destination_points",
                f"Destination points {first} and {second} are in the same "
                "place, so two different source features map to one spot. "
                "This usually means a point was placed twice by mistake.",
            )
        )

    if _is_collinear(src):
        issues.append(
            Issue(
                Severity.ERROR,
                "collinear_source_points",
                "All source points lie on a straight line. Spread them out "
                "across the image; a line gives no information about the "
                "direction perpendicular to it.",
            )
        )
    elif _is_collinear(dst):
        issues.append(
            Issue(
                Severity.ERROR,
                "collinear_destination_points",
                "All destination points lie on a straight line. Spread them "
                "out across the image.",
            )
        )

    if required <= len(src) < COMFORTABLE_POINTS:
        issues.append(
            Issue(
                Severity.WARNING,
                "sparse_points",
                f"Only {len(src)} point pairs. The result will be close to a "
                "plain affine fit; add more pairs to capture local "
                "distortion.",
            )
        )

    coverage = _coverage(src, image_shape)
    if coverage is not None and coverage < coverage_fraction:
        issues.append(
            Issue(
                Severity.WARNING,
                "poor_coverage",
                f"The points enclose about {coverage:.0%} of the source "
                "image. Everything outside that region is extrapolated and "
                "may be distorted; add points nearer the edges.",
            )
        )

    issues.sort(key=lambda issue: 0 if issue.is_error else 1)
    return issues


def _coverage(points: np.ndarray, image_shape: tuple[int, ...] | None) -> float | None:
    """Fraction of the image enclosed by the points, or None if unknowable."""
    if image_shape is None or len(image_shape) < 2:
        return None

    height, width = image_shape[0], image_shape[1]
    image_area = float(height) * float(width)
    if image_area <= 0:
        return None

    return convex_hull_area(points) / image_area


def has_errors(issues: list[Issue]) -> bool:
    """True if any issue blocks estimation."""
    return any(issue.is_error for issue in issues)


def format_issues(issues: list[Issue]) -> str:
    """Render issues as a bulleted message for a dialog."""
    return "\n\n".join(f"• {issue.message}" for issue in issues)
