"""Thin-plate spline transformation.

The transform *is* the fitted coefficients: a few hundred bytes that map any
destination coordinate back to its source. Everything else -- the dense
displacement field covering a whole image -- is a cache, built only when
something asks for it, at whatever resolution that caller wants.

This used to be the other way round. :meth:`ThinPlateSplineTransform.estimate`
evaluated the spline over the entire destination grid and kept the result as
the transform, so the cost of a fit scaled with the image rather than with the
number of control points:

===================  ==============  ============
Destination grid     Dense field     Coefficients
===================  ==============  ============
1 Mpx (1000x1000)    8 MB            0.7 KB
400 Mpx (stitched)   3.2 GB          0.7 KB
1600 Mpx             12.8 GB         0.7 KB
===================  ==============  ============

Mapping a handful of points paid the same price: 40 control points fit in
2.5 ms, but asking where 20 of them landed built the whole field first.

Notes
-----
The kernel is ``U(r) = r**2 * log(r**2)``, which is twice the textbook
``r**2 * log(r)``. The factor is absorbed into the fitted weights, so the
transform is the same; it just has to be spelled the same way here and in
:meth:`_TPS_makeL` or the two halves disagree.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
from scipy.spatial.distance import cdist

from tpsreg.validation import points_are_collinear

logger = logging.getLogger(__name__)

# Minimum number of non-collinear control points needed to solve the TPS system.
MIN_CONTROL_POINTS = 3


#: Candidate regularization strengths searched when ``regularization="auto"``.
#: Spread over six decades because the useful value is not knowable in
#: advance: it depends on how noisy the clicks are relative to the real
#: deformation. Zero is included so "no smoothing at all" can win.
AUTO_REGULARIZATION_GRID = np.concatenate([[0.0], np.logspace(-6, 0, 25)])

#: Relative slack when picking a winner from the cross-validation scores. Any
#: strength scoring within this of the best counts as tied, and the smallest of
#: those wins -- smooth no more than the data asks for.
SELECTION_TOLERANCE = 0.01

#: Absolute floor for that slack, in pixels. A purely relative tolerance is
#: useless when the best score is around 1e-15, which is what happens whenever
#: the deformation has no bending in it: every strength fits a translation
#: perfectly, and the winner would be decided by floating-point noise. Nothing
#: below this is a real difference in registration accuracy.
SELECTION_FLOOR_PX = 1e-6


def _kernel(distances: np.ndarray) -> np.ndarray:
    """Radial basis ``U(r) = r**2 log(r**2)``, with ``U(0) = 0``.

    Substituting 1 for a zero distance makes ``log(1) = 0`` rather than
    ``log(0) = -inf``, which is the same answer without the warning.
    """
    squared = distances * distances
    squared[distances == 0] = 1
    return squared * np.log(squared)


def _kernel_scale(control_points: np.ndarray) -> float:
    """Typical magnitude of the kernel block, used to normalise strength.

    Regularization is added to the kernel block, so a raw strength only means
    something relative to what is already there -- and the kernel grows with
    the square of the coordinates. A value tuned on a 1000 px image would do
    nothing at all on a 20000 px stitched one.

    Dividing by this makes the strength roughly comparable across image sizes.
    Only roughly: ``r**2 log(r**2)`` is not scale-homogeneous, so a 200-fold
    change in coordinates still shifts the effect by about twofold. That is
    close enough for a slider to be usable and for an automatic search to
    start in the right decade, which is all it is for.
    """
    kernel = _kernel(cdist(control_points, control_points, "euclidean"))
    np.fill_diagonal(kernel, 0)
    scale = float(np.abs(kernel).mean())
    return scale if scale > 0 else 1.0


def loocv_residuals(
    src: np.ndarray, dst: np.ndarray, regularization: float
) -> np.ndarray:
    """Leave-one-out residuals at one regularization strength, in closed form.

    Refitting K times is the obvious way, and
    :func:`tpsreg.metrics.leave_one_out_residuals` does exactly that. It is not
    needed here. Fitting is a linear smoother -- the predictions at the control
    points are a fixed matrix times the targets -- and for any linear smoother
    the leave-one-out residual follows from a single fit:

        residual_i = w_i / M_ii

    where ``w`` are the fitted bending weights and ``M`` maps targets to those
    weights. Verified against brute-force refitting to within 1e-13, at about
    a fourteenth of the cost for 16 control points and better from there up.

    The identity holds for a fixed penalty. Note that the strength is
    normalised by a kernel scale computed from the control points, so the same
    normalised number is a slightly different absolute penalty for a set with
    one point removed -- around 1.5% on sixteen points. That does not affect
    the search in :func:`select_regularization`, which holds the point set
    fixed and varies only the strength, but it does mean a brute-force
    comparison has to rescale per fold to be comparing the same thing.

    Parameters
    ----------
    src, dst:
        ``(K, 2)`` corresponding control points.
    regularization:
        Normalised strength; see :meth:`ThinPlateSplineTransform.estimate`.

    Returns
    -------
    np.ndarray
        ``(K,)`` distances in pixels.
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    n_points = len(dst)

    system = ThinPlateSplineTransform._TPS_makeL(dst)
    if regularization:
        system[:n_points, :n_points] += (
            regularization * _kernel_scale(dst) * np.eye(n_points)
        )

    targets = np.vstack(
        [
            np.concatenate([src[:, 0], np.zeros(3)]),
            np.concatenate([src[:, 1], np.zeros(3)]),
        ]
    ).T

    weights = np.linalg.solve(system, targets)[:n_points]

    # M maps targets to weights; its diagonal is the per-point leverage.
    smoother = np.linalg.solve(
        system, np.vstack([np.eye(n_points), np.zeros((3, n_points))])
    )[:n_points]
    leverage = np.diag(smoother)

    # A zero on the diagonal means that point contributes no bending, so
    # leaving it out changes nothing there; report no error rather than inf.
    safe = np.where(np.abs(leverage) > 0, leverage, np.inf)
    return np.linalg.norm(weights / safe[:, None], axis=1)


def select_regularization(
    src: np.ndarray,
    dst: np.ndarray,
    candidates: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Choose a regularization strength by leave-one-out cross-validation.

    Tries each candidate and keeps the one whose held-out predictions are
    best. This is what makes the setting usable: the strength is in units
    nobody has an intuition for, and the right value depends on how noisy the
    clicks are relative to the deformation being fitted, which differs per
    dataset.

    Returns
    -------
    tuple
        ``(best, candidates, scores)``. The scores are root-mean-square
        leave-one-out residuals in pixels, so they can be plotted or shown.

    Notes
    -----
    Scored on RMS rather than median deliberately. The case this is for is a
    few noisy clicks, and RMS notices them; a median would report the typical
    point as fine and pick no smoothing at all.
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)

    if candidates is None:
        candidates = AUTO_REGULARIZATION_GRID
    candidates = np.asarray(candidates, dtype=float)

    scores = np.full(len(candidates), np.inf)
    for index, candidate in enumerate(candidates):
        try:
            residuals = loocv_residuals(src, dst, candidate)
        except np.linalg.LinAlgError:
            # A degenerate system at this strength; the others may still work.
            continue
        if np.all(np.isfinite(residuals)):
            scores[index] = float(np.sqrt(np.mean(residuals**2)))

    if not np.any(np.isfinite(scores)):
        logger.warning(
            "Cross-validation found no usable regularization; not smoothing."
        )
        return 0.0, candidates, scores

    # Among strengths that score essentially the same, take the smallest.
    # Plain argmin picks an arbitrary one when several tie, which they do
    # whenever the deformation has no bending in it: a pure translation is
    # fitted perfectly at every strength, and reporting "smoothing 0.1" for
    # data that needs none is both confusing and needlessly lossy.
    lowest = float(np.nanmin(scores))
    tolerance = max(SELECTION_TOLERANCE * lowest, SELECTION_FLOOR_PX)
    tied = np.flatnonzero(scores <= lowest + tolerance)
    best = float(candidates[tied].min())

    logger.info(
        "Cross-validation chose regularization %.3g (RMS leave-one-out %.3f px, "
        "%d strength(s) within tolerance)",
        best,
        lowest,
        len(tied),
    )
    return best, candidates, scores


class ThinPlateSplineTransform:
    """Thin-plate spline mapping destination coordinates back to source.

    The fitted coefficients are the transform. :meth:`map` evaluates them
    directly at any coordinates, which is exact and costs
    ``O(n_coords * n_control_points)``.

    A dense field over a whole grid is available through :meth:`build_field`
    for callers that will query most of a grid repeatedly, and can be built at
    reduced resolution. It is a cache: clearing it changes speed, never
    results, beyond the interpolation error a downsampled field introduces.

    Parameters
    ----------
    chunk_size:
        Number of coordinates to evaluate per chunk. Defaults to a value
        derived from ``available_memory_gb``.
    dtype:
        Floating point type for a cached field. Coordinate mapping always
        works in double precision; the field is a cache, where float32 halves
        the memory for an error far below the pixel it is stored in.

    Attributes
    ----------
    control_points:
        ``(K, 2)`` destination control points the spline was fitted to.
    coefficients:
        ``(K + 3, 2)`` solution: K bending weights followed by the three
        affine terms.
    size:
        ``(height, width)`` the transform was estimated against, when one was
        supplied. Advisory: mapping does not need it.
    """

    def __init__(
        self,
        regularization: float | str = 0.0,
        chunk_size: int | None = None,
        dtype: type = np.float32,
    ):
        self._estimated = False
        self.control_points: np.ndarray | None = None
        self.coefficients: np.ndarray | None = None
        self.size: tuple[int, int] | None = None
        self.regularization = regularization
        #: Strength actually used, after resolving "auto". None until fitted.
        self.effective_regularization: float | None = None
        self.chunk_size = chunk_size
        self.dtype = dtype

        self._field: np.ndarray | None = None
        self._field_step: int = 1
        # Sample positions of the cached field, along each axis. Populated by
        # build_field/set_field; only read when _field is not None.
        self._field_xs: np.ndarray | None = None
        self._field_ys: np.ndarray | None = None

    # ------------------------------------------------------------------
    # The transform itself
    # ------------------------------------------------------------------

    @property
    def params(self) -> np.ndarray | None:
        """The fitted coefficients.

        This is what gets exported and what defines the transform. It used to
        be the dense field, which is why exporting a TPS to CSV or TXT always
        failed: ``np.savetxt`` refuses a 3D array.
        """
        return self.coefficients

    @params.setter
    def params(self, value: np.ndarray) -> None:
        self.coefficients = np.asarray(value, dtype=float)
        self._estimated = True

    @property
    def weights(self) -> np.ndarray:
        """The ``(K, 2)`` bending weights."""
        self._require_estimate()
        return self.coefficients[:-3]

    @property
    def affine(self) -> np.ndarray:
        """The ``(3, 2)`` affine terms, ordered ``[constant, dx, dy]``."""
        self._require_estimate()
        return self.coefficients[-3:]

    def _require_estimate(self) -> None:
        if not self._estimated or self.coefficients is None:
            raise ValueError(
                "Transformation not estimated. Call estimate() before applying it."
            )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def map(
        self,
        coords: np.ndarray,
        available_memory_gb: float = 2.0,
    ) -> np.ndarray:
        """Map destination coordinates to source coordinates.

        Evaluates the spline directly, at whatever coordinates are asked for,
        in double precision. No grid is built and nothing is rounded, so this
        is exact at fractional coordinates.

        Parameters
        ----------
        coords:
            ``(N, 2)`` array of ``(x, y)`` destination coordinates.
        available_memory_gb:
            Budget used to pick a chunk size when ``chunk_size`` is None. The
            distance matrix is the only large intermediate, so this bounds
            peak memory regardless of how many coordinates are passed.

        Returns
        -------
        np.ndarray
            ``(N, 2)`` array of ``(x, y)`` source coordinates.
        """
        self._require_estimate()

        coords = np.asarray(coords, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(
                f"Expected an (N, 2) array of coordinates, got {coords.shape}."
            )

        affine = self.affine
        # a1 + ax*x + ay*y, for both output components at once.
        mapped = affine[0] + coords[:, 0:1] * affine[1] + coords[:, 1:2] * affine[2]

        if len(coords) == 0:
            return mapped

        weights = self.weights
        n_coords = len(coords)
        chunk_size = self.chunk_size or self._estimate_chunk_size(
            n_coords, len(self.control_points), available_memory_gb
        )

        for start in range(0, n_coords, chunk_size):
            block = coords[start : start + chunk_size]
            distances = cdist(block, self.control_points, "euclidean")
            mapped[start : start + chunk_size] += _kernel(distances) @ weights

        return mapped

    def __call__(self, coords: np.ndarray) -> np.ndarray:
        """Map coordinates, as :func:`skimage.transform.warp` expects.

        Uses a cached field when one has been built, otherwise evaluates
        directly. Either way the result is float: the previous implementation
        truncated the query to int before looking it up, which quantised every
        warp to whole pixels and made the interpolation order meaningless.

        A transform carrying only a field -- one blended between slices, say
        -- is usable here even though it has no coefficients of its own.
        """
        if not self._estimated:
            raise ValueError(
                "Transformation not estimated. Call estimate() before applying it."
            )

        coords = np.asarray(coords, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(
                f"Expected an (N, 2) array of coordinates, got {coords.shape}."
            )

        if self._field is None:
            return self.map(coords)
        return self._sample_field(coords)

    # ------------------------------------------------------------------
    # The optional dense-field cache
    # ------------------------------------------------------------------

    @property
    def field(self) -> np.ndarray | None:
        """The cached ``(2, h, w)`` displacement field, if one was built."""
        return self._field

    @property
    def field_step(self) -> int:
        """Spacing, in destination pixels, between cached field samples."""
        return self._field_step

    def build_field(
        self,
        size: tuple[int, int] | None = None,
        downsample: int = 1,
        available_memory_gb: float = 2.0,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        """Evaluate the spline over a grid and cache the result.

        Worth it when most of a grid will be queried repeatedly. For anything
        less, :meth:`map` is both faster and smaller.

        Parameters
        ----------
        size:
            ``(height, width)`` to cover. Defaults to the size the transform
            was estimated against.
        downsample:
            Sample every Nth pixel. The spline is smooth between control
            points, so a coarse field costs very little accuracy and saves the
            square of this factor in memory: at 1/4 resolution a 400 Mpx grid
            needs 200 MB rather than 3.2 GB.
        available_memory_gb:
            Budget for the evaluation, which is chunked regardless of the
            field size.
        progress_callback:
            ``callback(completed_chunks, total_chunks)``, for a progress bar.

        Returns
        -------
        np.ndarray
            The ``(2, h, w)`` field, also stored on the transform.
        """
        self._require_estimate()

        size = size or self.size
        if size is None:
            raise ValueError(
                "No grid size available. Pass size=(height, width), or "
                "estimate the transform against one."
            )

        height, width = int(size[0]), int(size[1])
        step = max(1, int(downsample))

        # Sample positions start at 0 and step outwards; the last sample is
        # pinned to the final pixel so the field always spans the whole grid
        # and sampling never has to extrapolate past its right or bottom edge.
        xs = np.arange(0, width, step, dtype=float)
        ys = np.arange(0, height, step, dtype=float)
        if xs[-1] != width - 1:
            xs = np.append(xs, width - 1)
        if ys[-1] != height - 1:
            ys = np.append(ys, height - 1)

        grid_x, grid_y = np.meshgrid(xs, ys)
        coords = np.column_stack([grid_x.ravel(), grid_y.ravel()])

        n_coords = len(coords)
        n_control_points = (
            0 if self.control_points is None else len(self.control_points)
        )
        chunk_size = self.chunk_size or self._estimate_chunk_size(
            n_coords, n_control_points, available_memory_gb
        )
        n_chunks = int(np.ceil(n_coords / chunk_size))
        logger.info(
            "Building a %dx%d displacement field (1/%d resolution) over %d "
            "samples in %d chunk(s)",
            len(ys),
            len(xs),
            step,
            n_coords,
            n_chunks,
        )

        mapped = np.empty((n_coords, 2), dtype=self.dtype)
        for chunk_index in range(n_chunks):
            start = chunk_index * chunk_size
            stop = min(start + chunk_size, n_coords)
            mapped[start:stop] = self.map(coords[start:stop])
            if progress_callback is not None:
                progress_callback(chunk_index + 1, n_chunks)

        self._field = mapped.T.reshape(2, len(ys), len(xs))
        self._field_step = step
        self._field_xs = xs
        self._field_ys = ys
        self.size = (height, width)
        return self._field

    def set_field(
        self, field: np.ndarray, size: tuple[int, int] | None = None, step: int = 1
    ) -> None:
        """Install a pre-computed field, e.g. one interpolated between slices.

        The transform is then usable for warping without coefficients, which
        is what the 3D stack path needs: consecutive slices are fitted to
        different control points, so their coefficients cannot be blended, but
        their fields share a grid and can.
        """
        field = np.asarray(field)
        if field.ndim != 3 or field.shape[0] != 2:
            raise ValueError(f"Expected a (2, h, w) field, got {field.shape}.")

        self._field = field
        self._field_step = max(1, int(step))
        self._estimated = True

        if size is None:
            size = (
                (field.shape[1] - 1) * self._field_step + 1,
                (field.shape[2] - 1) * self._field_step + 1,
            )
        self.size = (int(size[0]), int(size[1]))
        self._field_ys = np.linspace(0, self.size[0] - 1, field.shape[1])
        self._field_xs = np.linspace(0, self.size[1] - 1, field.shape[2])

    def clear_field(self) -> None:
        """Drop the cached field, freeing its memory."""
        self._field = None
        self._field_step = 1

    def _sample_field(self, coords: np.ndarray) -> np.ndarray:
        """Bilinearly sample the cached field at arbitrary coordinates."""
        field = np.moveaxis(self._field, 0, -1)
        height, width = field.shape[:2]

        # Position within the sample grid, clamped so queries beyond the
        # estimated size return the edge rather than extrapolating.
        col = np.interp(coords[:, 0], self._field_xs, np.arange(width, dtype=float))
        row = np.interp(coords[:, 1], self._field_ys, np.arange(height, dtype=float))

        col0 = np.clip(np.floor(col).astype(int), 0, width - 1)
        row0 = np.clip(np.floor(row).astype(int), 0, height - 1)
        col1 = np.clip(col0 + 1, 0, width - 1)
        row1 = np.clip(row0 + 1, 0, height - 1)

        fx = (col - col0)[:, None]
        fy = (row - row0)[:, None]

        top = field[row0, col0] * (1 - fx) + field[row0, col1] * fx
        bottom = field[row1, col0] * (1 - fx) + field[row1, col1] * fx
        return top * (1 - fy) + bottom * fy

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def _estimate_chunk_size(
        self,
        n_coords: int,
        n_control_points: int,
        available_memory_gb: float = 2.0,
    ) -> int:
        """Pick a chunk size that keeps peak memory near the given budget.

        Parameters
        ----------
        n_coords:
            Total number of coordinates to process.
        n_control_points:
            Number of control points.
        available_memory_gb:
            Memory budget in GB for the computation.

        Returns
        -------
        int
            Number of coordinates to process per chunk.
        """
        if n_control_points <= 0:
            return max(1000, n_coords)

        # Peak usage per coordinate is dominated by the distance matrix and
        # the kernel matrix, each chunk x n_control_points doubles; 4x leaves
        # headroom for the intermediates SciPy allocates.
        memory_per_coord = n_control_points * 8 * 4

        available_bytes = available_memory_gb * 1024**3
        chunk_size = int(available_bytes / memory_per_coord)

        # At least 1000 coordinates per chunk, never more than were asked for.
        return max(1000, min(chunk_size, max(n_coords, 1)))

    def _resolve_regularization(self, src: np.ndarray, dst: np.ndarray) -> float:
        """Turn the requested setting into a number.

        ``"auto"`` runs the cross-validated search; anything else is taken as
        the strength itself. Negative values are refused rather than quietly
        clamped: subtracting from the kernel diagonal is not a weaker fit, it
        is a different and possibly unsolvable system.
        """
        requested = self.regularization

        if isinstance(requested, str):
            if requested.lower() != "auto":
                raise ValueError(
                    f"Unknown regularization: {requested!r}. Expected a "
                    'number or "auto".'
                )
            best, _, _ = select_regularization(src, dst)
            return best

        strength = float(requested)
        if strength < 0:
            raise ValueError(f"Regularization must not be negative; got {strength}.")
        return strength

    @staticmethod
    def _check_valid_points(src: np.ndarray, dst: np.ndarray) -> bool:
        """Validate a set of control point correspondences.

        Raises
        ------
        ValueError
            If the arrays disagree in shape, are not 2D coordinates, hold fewer
            than :data:`MIN_CONTROL_POINTS` points, or are degenerate.
        """
        src = np.asarray(src)
        dst = np.asarray(dst)

        if src.shape != dst.shape:
            raise ValueError(
                f"Source and destination points must have the same shape; "
                f"got {src.shape} and {dst.shape}."
            )
        if src.ndim != 2:
            raise ValueError(
                f"Incorrect shape for control points; expected (N, 2), "
                f"received {src.shape}."
            )
        if src.shape[1] != 2:
            raise ValueError("Control points must be 2D coordinates.")
        if src.shape[0] < MIN_CONTROL_POINTS:
            raise ValueError(
                f"At least {MIN_CONTROL_POINTS} control points are required; "
                f"got {src.shape[0]}."
            )

        if np.unique(src, axis=0).shape[0] != src.shape[0]:
            raise ValueError("Source control points contain duplicates.")
        if np.unique(dst, axis=0).shape[0] != dst.shape[0]:
            raise ValueError("Destination control points contain duplicates.")

        # Collinear points are rejected here rather than left to LAPACK. The
        # system matrix is built from the destination points, and a line makes
        # its [1, x, y] block rank-deficient -- but whether np.linalg.solve
        # notices depends on the LAPACK build. macOS Accelerate raised;
        # OpenBLAS on the CI runners returned a garbage transform for the same
        # points. Testing the geometry directly costs microseconds and gives
        # the same answer everywhere.
        if points_are_collinear(dst):
            raise ValueError(
                "Destination control points are collinear, which leaves the "
                "thin-plate spline system singular. Spread the points out "
                "across the image."
            )
        if points_are_collinear(src):
            raise ValueError(
                "Source control points are collinear, so the transform would "
                "collapse the image onto a line. Spread the points out."
            )

        return True

    def estimate(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        size: tuple[int, int] | None = None,
        available_memory_gb: float = 2.0,
        progress_callback: Callable[[int, int], None] | None = None,
        build_field: bool = False,
        downsample: int = 1,
    ) -> bool:
        """Fit the spline mapping destination points back to source points.

        Solves for the coefficients and stops there. The cost is set by the
        number of control points, not the size of the image: 40 points fit in
        a couple of milliseconds whatever the destination grid is.

        Parameters
        ----------
        src:
            ``(N, 2)`` control points in source coordinates.
        dst:
            ``(N, 2)`` control points in destination coordinates.
        size:
            ``(height, width)`` of the destination grid. Optional, and only
            advisory unless a field is being built; kept so callers that
            already pass it keep working and so exports can record it.
        available_memory_gb:
            Memory budget used to pick a chunk size when building a field.
        progress_callback:
            ``callback(completed_chunks, total_chunks)``, used while building
            a field. A fit on its own has nothing to report progress about.
        build_field:
            Also build the dense field cache, reproducing the old behaviour of
            evaluating the whole grid up front. Off by default.
        downsample:
            Resolution of that field; see :meth:`build_field`.

        Notes
        -----
        Regularization (set on the constructor) decides whether the spline has
        to pass exactly through every control point. At 0 -- the default, and
        what this has always done -- it does, which means it also reproduces
        every click error exactly. Real control points carry error, so exact
        interpolation of them is not the same as an accurate transform: the
        fit contorts itself to honour mistakes.

        A positive strength relaxes that, buying smoothness with the slack.
        ``"auto"`` picks the strength by leave-one-out cross-validation, which
        matters because the number is in units nobody has an intuition for and
        the right value depends on how noisy the clicks are relative to the
        deformation. On synthetic data with a known answer and 1.5 px of click
        noise, the automatic choice roughly halved the error against the true
        deformation; with clean points it selects 0 and changes nothing.

        Returns
        -------
        bool
            True when the estimation succeeded.
        """
        self._check_valid_points(src, dst)

        src = np.asarray(src, dtype=float)
        dst = np.asarray(dst, dtype=float)

        # Control points live on the destination grid; the spline maps them
        # back to the source, which is the direction skimage.warp needs.
        n = dst.shape[0]
        L = self._TPS_makeL(dst)

        strength = self._resolve_regularization(src, dst)
        self.effective_regularization = strength
        if strength:
            # The whole of regularization: relax the requirement that the
            # spline pass exactly through its control points, by adding to the
            # diagonal of the kernel block. The fit is then free to miss them,
            # and buys smoothness with the slack.
            L[:n, :n] += strength * _kernel_scale(dst) * np.eye(n)

        # Right-hand side, padded with the three affine constraints.
        Y = np.vstack(
            [
                np.concatenate([src[:, 0], np.zeros(3)]),
                np.concatenate([src[:, 1], np.zeros(3)]),
            ]
        ).T

        try:
            coefficients = np.linalg.solve(L, Y)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "Could not solve the thin-plate spline system. This usually "
                "means the control points are collinear or nearly coincident."
            ) from exc

        # A silently bad solve is worse than a loud one, and whether LAPACK
        # raises on a near-singular system varies by build. Substituting the
        # answer back is O(K**2) against the O(K**3) solve, so the check is
        # free, and it catches any ill-conditioning rather than only the
        # degenerate geometries checked above.
        if not np.all(np.isfinite(coefficients)):
            raise ValueError(
                "The thin-plate spline solution is not finite. The control "
                "points are too close to degenerate; spread them out or "
                "remove near-coincident pairs."
            )

        residual = np.abs(L @ coefficients - Y).max()
        scale = max(np.abs(Y).max(), 1.0)
        if residual > 1e-6 * scale:
            raise ValueError(
                "The thin-plate spline system could not be solved accurately "
                f"(residual {residual:.3g}). The control points are close to "
                "collinear or coincident; spread them out."
            )

        self.control_points = dst
        self.coefficients = coefficients
        self.size = (int(size[0]), int(size[1])) if size is not None else None
        self._field = None
        self._field_step = 1
        self._estimated = True

        logger.debug(
            "Fitted a thin-plate spline to %d control points (%d bytes of "
            "coefficients)",
            n,
            coefficients.nbytes,
        )

        if build_field:
            self.build_field(
                size,
                downsample=downsample,
                available_memory_gb=available_memory_gb,
                progress_callback=progress_callback,
            )

        return True

    @staticmethod
    def _TPS_makeL(cp: np.ndarray) -> np.ndarray:
        """Build the ``(K+3, K+3)`` TPS system matrix for K control points."""
        K = cp.shape[0]
        L = np.zeros((K + 3, K + 3))

        # P block
        L[:K, K] = 1
        L[:K, K + 1 : K + 3] = cp
        # P.T block
        L[K, :K] = 1
        L[K + 1 :, :K] = cp.T

        U = _kernel(cdist(cp, cp, "euclidean"))
        np.fill_diagonal(U, 0)
        L[:K, :K] = U

        return L
