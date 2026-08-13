"""Slice-by-slice registration of an image stack.

Aligns a folder of images into a common frame by matching consecutive pairs
with MatchAnything and fitting a transform to the matches. Written for
serial-sectioning data, where each slice resembles its neighbour closely but
the first and last slice may have little in common.

The matching model is injected rather than imported here, so the whole
pipeline can be exercised with a stand-in. That matters: the model needs torch
and a checkpoint, neither of which belongs in a test.

Two things are worth understanding before using it.

**Reference frame.** ``previous`` registers each slice to the one before it and
composes the results, which is what makes matching easy -- consecutive slices
look alike. The cost is drift: every pair contributes its own small error and
they accumulate along the stack, so slice 500 can be far from slice 0 even
though every individual pair looked good. ``first`` and ``middle`` register
everything against one slice, which cannot drift but asks the matcher to
relate images that may be very different. Neither is right for every dataset,
so :func:`register_stack` reports the drift it accumulated and leaves the
choice to the caller.

**Composition.** Slices are never warped more than once. Transforms are
composed and applied in a single pass, so a hundred-slice stack does not
accumulate a hundred rounds of interpolation blur. Where every step is a
matrix -- translation, rigid and affine -- the composition is a matrix
product, so the cost of warping slice 300 is the same as slice 1. A chain
containing a spline cannot reduce that way and is evaluated link by link,
which does grow with depth; ``first`` or ``middle`` avoids it entirely by
never building a chain.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from skimage import transform as sktransform

from tpsreg import metrics, overlays
from tpsreg.tps import ThinPlateSplineTransform
from tpsreg.warping import warp

logger = logging.getLogger(__name__)

#: Transform models, from most constrained to least. A more constrained model
#: needs fewer matches and cannot invent deformation that is not there, so it
#: is the safer default when the physical situation allows it.
TRANSFORM_TYPES: tuple[str, ...] = ("translation", "rigid", "affine", "tps")

#: How each slice chooses what to register against.
REFERENCE_MODES: tuple[str, ...] = ("previous", "first", "middle")

#: Matched pairs needed by each model. A transform fitted to its bare minimum
#: has no redundancy at all: it will reproduce the matches exactly whether or
#: not they were right.
MINIMUM_MATCHES: dict[str, int] = {
    "translation": 1,
    "rigid": 2,
    "affine": 3,
    "tps": 3,
}

#: Below this many matches a pair is fitted but flagged. Chosen so there is
#: enough redundancy for the residuals to mean something.
COMFORTABLE_MATCHES = 8

#: Image extensions picked up from an input folder.
IMAGE_SUFFIXES = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")

#: Chain length past which sequential spline registration is worth warning
#: about. A matrix chain collapses to one matrix and costs nothing to lengthen;
#: a spline chain does not, so every slice must be evaluated through every
#: spline before it and the warp cost grows with position in the stack.
TPS_CHAIN_WARNING_DEPTH = 25


class TranslationTransform:
    """A pure shift, fitted as the median offset between matched points.

    scikit-image has no translation-only estimator -- its "euclidean" model
    includes rotation. Fitting the shift directly is both trivial and more
    robust than fitting a richer model and discarding the parts that are not
    wanted, because a richer model spends its extra freedom absorbing noise.

    The median rather than the mean: a handful of bad matches survive RANSAC
    often enough to matter, and the median ignores them.
    """

    def __init__(self, offset: np.ndarray):
        self.offset = np.asarray(offset, dtype=float).reshape(2)

    @classmethod
    def estimate(cls, src: np.ndarray, dst: np.ndarray) -> TranslationTransform:
        """Fit the shift mapping ``src`` onto ``dst``."""
        src = np.asarray(src, dtype=float)
        dst = np.asarray(dst, dtype=float)
        return cls(np.median(dst - src, axis=0))

    def __call__(self, coords: np.ndarray) -> np.ndarray:
        return np.asarray(coords, dtype=float) + self.offset

    @property
    def params(self) -> np.ndarray:
        return self.offset.copy()

    def as_matrix(self) -> np.ndarray:
        """The shift as a homogeneous matrix; see `warping.homography_matrix`."""
        matrix = np.eye(3)
        matrix[:2, 2] = self.offset
        return matrix

    def describe(self) -> dict[str, Any]:
        return {"dx": float(self.offset[0]), "dy": float(self.offset[1])}


class ChainedTransform:
    """Several transforms applied in order, as one callable.

    Sequential registration produces a transform per adjacent pair. Warping
    through them one at a time would resample the image once per step, and a
    hundred-slice stack would arrive at slice 100 having been interpolated a
    hundred times. Composing the coordinate mappings instead means the image
    is resampled exactly once, however long the chain.

    Composing them as *functions*, though, means every one of them is called
    for every output pixel, so slice 300 costs three hundred times slice 1 --
    on a long stack that dominates the run. When every link is a matrix, which
    is the case for translation, rigid and affine, the chain collapses into a
    single matrix at construction and the depth stops mattering at all. Only a
    spline in the chain forces the general path.
    """

    def __init__(self, transforms: Sequence[Any]):
        self.transforms = list(transforms)
        self._collapsed = _collapse_to_matrix(self.transforms)

    def __call__(self, coords: np.ndarray) -> np.ndarray:
        if self._collapsed is not None:
            return np.asarray(self._collapsed(coords), dtype=float)

        mapped = np.asarray(coords, dtype=float)
        for transform in self.transforms:
            mapped = np.asarray(transform(mapped), dtype=float)
        return mapped

    def __len__(self) -> int:
        return len(self.transforms)

    def as_matrix(self) -> np.ndarray | None:
        """The whole chain as one matrix, or ``None`` if it does not reduce."""
        if self._collapsed is None:
            return None
        return np.asarray(self._collapsed.params, dtype=float)

    @property
    def params(self) -> np.ndarray | None:
        return self.as_matrix()

    def describe(self) -> dict[str, Any]:
        matrix = self.as_matrix()
        described: dict[str, Any] = {"chained": len(self.transforms)}
        if matrix is not None:
            described["matrix"] = matrix.tolist()
        return described


def _collapse_to_matrix(transforms: Sequence[Any]) -> Any | None:
    """Multiply a chain of matrix transforms into one, or give up.

    Returns a skimage transform so the mapping is skimage's own tested
    implementation rather than a second copy of it here -- and so that
    :func:`tpsreg.warping.warp` recognises it and takes the Cython path.
    """
    from tpsreg.warping import homography_matrix

    combined = np.eye(3)
    for transform in transforms:
        matrix = homography_matrix(transform)
        if matrix is None:
            return None
        # Applied in list order, so each one multiplies on the left.
        combined = matrix @ combined

    return sktransform.ProjectiveTransform(matrix=combined)


class IdentityTransform:
    """Used where a pair could not be registered, so the slice passes through."""

    def __call__(self, coords: np.ndarray) -> np.ndarray:
        return np.asarray(coords, dtype=float)

    def as_matrix(self) -> np.ndarray:
        return np.eye(3)

    def describe(self) -> dict[str, Any]:
        return {"identity": True}


@dataclass
class PairResult:
    """What happened when one slice was registered to its reference."""

    index: int
    reference_index: int
    name: str = ""
    transform_type: str = "tps"
    n_raw_matches: int = 0
    n_matches: int = 0
    residual_median: float = float("nan")
    residual_max: float = float("nan")
    #: Distance the slice moved relative to its reference, at the image
    #: centre. A quick sanity number: a jump between neighbouring slices is
    #: usually a bad registration rather than a real feature.
    displacement: float = float("nan")
    min_jacobian: float = float("nan")
    folded_fraction: float = 0.0
    seconds: float = 0.0
    ok: bool = True
    warnings: list[str] = field(default_factory=list)

    @property
    def is_reference(self) -> bool:
        return self.index == self.reference_index


@dataclass
class StackResult:
    """The whole run."""

    transform_type: str
    reference_mode: str
    pairs: list[PairResult] = field(default_factory=list)
    #: Cumulative shift of each slice from the first, at the image centre.
    #: In "previous" mode this is where drift shows up.
    cumulative_displacement: list[float] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def failed(self) -> list[PairResult]:
        return [pair for pair in self.pairs if not pair.ok]

    @property
    def flagged(self) -> list[PairResult]:
        return [pair for pair in self.pairs if pair.warnings]

    def summary(self) -> str:
        """A few lines a human can read after a long run."""
        total = len(self.pairs)
        failures = len(self.failed)
        flagged = len(self.flagged)

        matches = [p.n_matches for p in self.pairs if not p.is_reference]
        residuals = [
            p.residual_median for p in self.pairs if np.isfinite(p.residual_median)
        ]

        lines = [
            f"{total} slice(s), {self.transform_type} against the "
            f"{self.reference_mode} slice, in {self.seconds:.1f}s",
            f"  matches:   median {int(np.median(matches))} per pair"
            if matches
            else "  matches:   none",
            f"  residuals: median {np.median(residuals):.2f} px"
            if residuals
            else "  residuals: not available",
        ]
        if self.cumulative_displacement:
            lines.append(
                f"  drift:     {self.cumulative_displacement[-1]:.1f} px "
                "from first to last slice"
            )
        if failures:
            lines.append(f"  FAILED:    {failures} slice(s) left unregistered")
        if flagged:
            lines.append(f"  flagged:   {flagged} slice(s) worth checking")
        return "\n".join(lines)


def estimate_pair_transform(
    moving_points: np.ndarray,
    reference_points: np.ndarray,
    transform_type: str,
    shape: tuple[int, int] | None = None,
) -> Any:
    """Fit a transform taking reference coordinates to moving coordinates.

    That direction is the one warping needs: to fill an output pixel in the
    reference frame, you must know where to read from in the moving image.

    Parameters
    ----------
    moving_points, reference_points:
        ``(N, 2)`` matched coordinates, in ``(x, y)``.
    transform_type:
        One of :data:`TRANSFORM_TYPES`.
    shape:
        ``(height, width)`` of the reference frame. Recorded on a spline so
        exports are self-describing; not otherwise needed.

    Raises
    ------
    ValueError
        If the transform type is unknown, or there are too few matches for it.
    """
    if transform_type not in TRANSFORM_TYPES:
        raise ValueError(
            f"Unknown transform type: {transform_type!r}. "
            f"Expected one of {TRANSFORM_TYPES}."
        )

    moving_points = np.asarray(moving_points, dtype=float)
    reference_points = np.asarray(reference_points, dtype=float)

    needed = MINIMUM_MATCHES[transform_type]
    if len(moving_points) < needed:
        raise ValueError(
            f"{transform_type} needs at least {needed} matches; "
            f"got {len(moving_points)}."
        )

    if transform_type == "translation":
        return TranslationTransform.estimate(reference_points, moving_points)

    if transform_type == "tps":
        transform = ThinPlateSplineTransform()
        transform.estimate(moving_points, reference_points, shape)
        return transform

    model = {"rigid": "euclidean", "affine": "affine"}[transform_type]
    estimated = sktransform.estimate_transform(model, reference_points, moving_points)
    if not np.all(np.isfinite(np.asarray(estimated.params, dtype=float))):
        raise ValueError(
            f"The {transform_type} fit did not converge; the matches are "
            "probably degenerate."
        )
    return estimated


def pair_residuals(
    transform: Any,
    moving_points: np.ndarray,
    reference_points: np.ndarray,
    transform_type: str,
) -> np.ndarray:
    """How far each match sits from what the fitted transform predicts.

    For the constrained models this is the ordinary residual, which is
    meaningful because the model cannot pass through every point.

    A spline can, and does: it interpolates its control points exactly, so its
    ordinary residual is zero regardless of whether the matches were any good.
    Leave-one-out is used instead, which asks the remaining matches to predict
    each held-out one. See :mod:`tpsreg.metrics`.
    """
    moving_points = np.asarray(moving_points, dtype=float)
    reference_points = np.asarray(reference_points, dtype=float)

    if transform_type == "tps":
        if len(moving_points) < metrics.MIN_POINTS_FOR_RESIDUALS:
            # Below this, every leave-one-out residual is large whatever the
            # matches look like, so the number would mislead rather than
            # inform. See metrics.MIN_POINTS_FOR_RESIDUALS.
            return np.full(len(moving_points), np.nan)
        return metrics.leave_one_out_residuals(moving_points, reference_points)

    predicted = np.asarray(transform(reference_points), dtype=float)
    return np.linalg.norm(predicted - moving_points, axis=1)


def _describe(transform: Any) -> dict[str, Any]:
    """A JSON-serialisable description of a fitted transform."""
    if hasattr(transform, "describe"):
        return transform.describe()

    params = getattr(transform, "params", None)
    if params is None:
        return {"type": type(transform).__name__}

    params = np.asarray(params, dtype=float)
    if params.ndim == 2 and params.shape == (3, 3):
        return {"matrix": params.tolist()}
    # A spline: the coefficients are the transform, but there can be a lot of
    # them, so record their shape and leave the values to the .npy export.
    return {"coefficients_shape": list(params.shape)}


def _centre_displacement(transform: Any, shape: tuple[int, int]) -> float:
    """How far the image centre moves under a transform, in pixels.

    One number per slice that a human can scan down: neighbouring slices
    should move by similar amounts, and a spike is worth looking at.
    """
    height, width = shape[:2]
    centre = np.array([[width / 2.0, height / 2.0]])
    try:
        moved = np.asarray(transform(centre), dtype=float)
    except Exception:  # pragma: no cover - a transform that cannot be called
        return float("nan")
    return float(np.linalg.norm(moved - centre))


def register_stack(
    images: Sequence[np.ndarray],
    match_fn: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, Any]],
    transform_type: str = "rigid",
    reference_mode: str = "previous",
    names: Sequence[str] | None = None,
    on_progress: Callable[[int, int, PairResult], None] | None = None,
) -> tuple[list[Any], StackResult]:
    """Register every slice of a stack into a common frame.

    Parameters
    ----------
    images:
        The stack, in order. Slices may differ in size; each is registered
        into the reference slice's frame.
    match_fn:
        ``match_fn(moving, reference) -> (moving_points, reference_points,
        confidences)``. Injected so the pipeline can be tested without the
        model, and so a different matcher can be substituted.
    transform_type:
        One of :data:`TRANSFORM_TYPES`.
    reference_mode:
        One of :data:`REFERENCE_MODES`; see the module docstring for the
        trade-off between them.
    names:
        Labels for each slice, used in the report.
    on_progress:
        ``on_progress(done, total, result)`` after each slice.

    Returns
    -------
    tuple
        ``(transforms, result)``. Each transform maps reference-frame
        coordinates to that slice's own coordinates, ready to hand to
        :func:`tpsreg.warping.warp`.

    Notes
    -----
    A slice that cannot be registered does not abort the run. It is recorded
    as failed, given the identity, and the stack continues -- one bad pair in
    a five-hundred slice stack should not cost the other four hundred and
    ninety-nine.

    In ``previous`` mode a failed pair breaks the chain: every later slice is
    composed through the identity where that link should have been, so they
    are all shifted by however much that pair actually moved. The failure is
    reported for exactly this reason.
    """
    if transform_type not in TRANSFORM_TYPES:
        raise ValueError(
            f"Unknown transform type: {transform_type!r}. "
            f"Expected one of {TRANSFORM_TYPES}."
        )
    if reference_mode not in REFERENCE_MODES:
        raise ValueError(
            f"Unknown reference mode: {reference_mode!r}. "
            f"Expected one of {REFERENCE_MODES}."
        )
    if not images:
        raise ValueError("No images to register.")

    if (
        transform_type == "tps"
        and reference_mode == "previous"
        and len(images) > TPS_CHAIN_WARNING_DEPTH
    ):
        logger.warning(
            "Sequential spline registration over %d slices: warping slice N "
            "evaluates N splines, so the cost grows along the stack and the "
            "last slices dominate the run. The constrained models do not have "
            "this problem -- they compose into a single matrix -- and neither "
            "does reference_mode='first' or 'middle', where no chain is built. "
            "Consider one of those if the warp step is too slow.",
            len(images),
        )

    names = (
        list(names) if names is not None else [f"slice_{i}" for i in range(len(images))]
    )
    started = time.perf_counter()

    anchor = {"previous": 0, "first": 0, "middle": len(images) // 2}[reference_mode]
    result = StackResult(transform_type=transform_type, reference_mode=reference_mode)

    # Transform from the anchor frame into each slice's own frame.
    to_slice: list[Any] = [IdentityTransform() for _ in images]
    # The step from each slice's predecessor, kept for the chain.
    steps: dict[int, Any] = {}

    for index, image in enumerate(images):
        pair_started = time.perf_counter()

        if index == anchor:
            pair = PairResult(
                index=index,
                reference_index=index,
                name=names[index],
                transform_type=transform_type,
                displacement=0.0,
                seconds=time.perf_counter() - pair_started,
            )
            result.pairs.append(pair)
            if on_progress is not None:
                on_progress(index + 1, len(images), pair)
            continue

        if reference_mode == "previous":
            reference_index = index - 1
        else:
            reference_index = anchor

        pair, step = _register_one(
            image,
            images[reference_index],
            index=index,
            reference_index=reference_index,
            name=names[index],
            transform_type=transform_type,
            match_fn=match_fn,
        )
        pair.seconds = time.perf_counter() - pair_started
        steps[index] = step

        if reference_mode == "previous":
            # Compose from the anchor forwards, so the image is resampled once.
            chain = [steps[i] for i in range(1, index + 1) if i in steps]
            to_slice[index] = ChainedTransform(chain)
        else:
            to_slice[index] = step

        result.pairs.append(pair)
        if on_progress is not None:
            on_progress(index + 1, len(images), pair)

    reference_shape = images[anchor].shape[:2]
    result.cumulative_displacement = [
        _centre_displacement(transform, reference_shape) for transform in to_slice
    ]
    result.seconds = time.perf_counter() - started

    return to_slice, result


def _register_one(
    moving: np.ndarray,
    reference: np.ndarray,
    index: int,
    reference_index: int,
    name: str,
    transform_type: str,
    match_fn: Callable[..., tuple[np.ndarray, np.ndarray, Any]],
) -> tuple[PairResult, Any]:
    """Match and fit one pair, recording everything and never raising.

    Returns the result and the fitted transform together. A failure yields the
    identity, so the caller always has something to compose with.
    """
    pair = PairResult(
        index=index,
        reference_index=reference_index,
        name=name,
        transform_type=transform_type,
    )

    try:
        moving_points, reference_points, _confidence = match_fn(moving, reference)
    except Exception as exc:
        pair.ok = False
        pair.warnings.append(f"matching failed: {exc}")
        logger.error("Slice %d: matching failed: %s", index, exc)
        return pair, IdentityTransform()

    moving_points = np.asarray(moving_points, dtype=float).reshape(-1, 2)
    reference_points = np.asarray(reference_points, dtype=float).reshape(-1, 2)
    pair.n_raw_matches = len(moving_points)
    pair.n_matches = len(moving_points)

    try:
        transform = estimate_pair_transform(
            moving_points,
            reference_points,
            transform_type,
            shape=reference.shape[:2],
        )
    except Exception as exc:
        pair.ok = False
        pair.warnings.append(f"fit failed: {exc}")
        logger.error("Slice %d: fit failed: %s", index, exc)
        return pair, IdentityTransform()

    residuals = pair_residuals(
        transform, moving_points, reference_points, transform_type
    )
    if len(residuals) and np.any(np.isfinite(residuals)):
        pair.residual_median = float(np.nanmedian(residuals))
        pair.residual_max = float(np.nanmax(residuals))

    pair.displacement = _centre_displacement(transform, reference.shape[:2])

    if transform_type == "tps":
        try:
            determinants = metrics.jacobian_determinant(
                transform, size=reference.shape[:2]
            )
            pair.min_jacobian = float(determinants.min())
            pair.folded_fraction = metrics.folded_fraction(determinants)
        except Exception as exc:  # pragma: no cover - depends on the fit
            logger.debug("Slice %d: could not evaluate the Jacobian: %s", index, exc)

    if pair.n_matches < COMFORTABLE_MATCHES:
        pair.warnings.append(
            f"only {pair.n_matches} matches; the fit has little redundancy"
        )
    if pair.folded_fraction > 0:
        pair.warnings.append(
            f"the mapping folds over {pair.folded_fraction:.1%} of the frame"
        )

    return pair, transform


def flag_outlying_slices(result: StackResult, tolerance: float = 3.5) -> list[int]:
    """Slices whose displacement stands out from their neighbours.

    Consecutive slices of a serial section move by similar amounts. One that
    jumps is usually a mis-registration, and it is the thing worth looking at
    first in a stack too large to inspect by eye.

    Uses the median and MAD rather than the mean and standard deviation: a
    single bad slice inflates the standard deviation enough to hide itself.
    """
    displacements = np.array(
        [
            pair.displacement
            for pair in result.pairs
            if not pair.is_reference and np.isfinite(pair.displacement)
        ]
    )
    indices = [
        pair.index
        for pair in result.pairs
        if not pair.is_reference and np.isfinite(pair.displacement)
    ]

    if len(displacements) < 3:
        return []

    median = np.median(displacements)
    mad = np.median(np.abs(displacements - median))
    if mad == 0:
        # Every slice moved by the same amount except, possibly, one. The
        # score is undefined, but "nothing stands out" would be exactly wrong
        # here: anything away from the median is the thing to look at.
        return [
            index
            for index, value in zip(indices, displacements, strict=True)
            if value != median
        ]

    scores = 0.6745 * np.abs(displacements - median) / mad
    return [
        index for index, score in zip(indices, scores, strict=True) if score > tolerance
    ]


def write_report(
    result: StackResult, directory: Path, extra: dict | None = None
) -> None:
    """Write the run's numbers as JSON and CSV.

    JSON keeps everything; the CSV is one row per slice, which is what you
    actually want when scanning a long stack or plotting drift.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    payload = {
        "transform_type": result.transform_type,
        "reference_mode": result.reference_mode,
        "seconds": result.seconds,
        "cumulative_displacement": result.cumulative_displacement,
        "outlying_slices": flag_outlying_slices(result),
        "pairs": [asdict(pair) for pair in result.pairs],
    }
    if extra:
        payload.update(extra)

    (directory / "report.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    with (directory / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "index",
                "name",
                "reference_index",
                "n_matches",
                "residual_median_px",
                "residual_max_px",
                "displacement_px",
                "cumulative_displacement_px",
                "min_jacobian",
                "folded_fraction",
                "seconds",
                "ok",
                "warnings",
            ]
        )
        for pair in result.pairs:
            cumulative = (
                result.cumulative_displacement[pair.index]
                if pair.index < len(result.cumulative_displacement)
                else float("nan")
            )
            writer.writerow(
                [
                    pair.index,
                    pair.name,
                    pair.reference_index,
                    pair.n_matches,
                    f"{pair.residual_median:.4f}",
                    f"{pair.residual_max:.4f}",
                    f"{pair.displacement:.4f}",
                    f"{cumulative:.4f}",
                    f"{pair.min_jacobian:.4f}",
                    f"{pair.folded_fraction:.4f}",
                    f"{pair.seconds:.2f}",
                    "yes" if pair.ok else "NO",
                    "; ".join(pair.warnings),
                ]
            )


def match_figure(
    moving: np.ndarray,
    reference: np.ndarray,
    moving_points: np.ndarray,
    reference_points: np.ndarray,
    max_lines: int = 60,
) -> np.ndarray:
    """The two images side by side with matched points joined.

    Rendered with numpy and skimage rather than matplotlib, so it costs no
    extra dependency and no figure lifecycle. Bad matches are obvious here in
    a way no summary number conveys: they are the lines that cross the others.
    """
    from skimage.draw import line as draw_line

    left = overlays.to_rgb(reference)
    right = overlays.to_rgb(moving)

    height = max(left.shape[0], right.shape[0])
    canvas = np.zeros((height, left.shape[1] + right.shape[1], 3), dtype=np.uint8)
    canvas[: left.shape[0], : left.shape[1]] = left
    canvas[: right.shape[0], left.shape[1] :] = right

    offset = left.shape[1]
    moving_points = np.asarray(moving_points, dtype=float).reshape(-1, 2)
    reference_points = np.asarray(reference_points, dtype=float).reshape(-1, 2)

    count = min(len(moving_points), max_lines)
    if count == 0:
        return canvas

    # Evenly spaced rather than the first N, so the picture represents the
    # whole field instead of one corner of it.
    chosen = np.linspace(0, len(moving_points) - 1, count).astype(int)
    colours = _line_colours(count)

    for colour, i in zip(colours, chosen, strict=True):
        r0, c0 = int(reference_points[i][1]), int(reference_points[i][0])
        r1, c1 = int(moving_points[i][1]), int(moving_points[i][0] + offset)
        if not (0 <= r0 < height and 0 <= r1 < height):
            continue
        if not (0 <= c0 < canvas.shape[1] and 0 <= c1 < canvas.shape[1]):
            continue
        rows, cols = draw_line(r0, c0, r1, c1)
        canvas[rows, cols] = colour

    return canvas


def _line_colours(count: int) -> np.ndarray:
    """A spread of hues, so neighbouring lines stay tellable apart."""
    hues = np.linspace(0, 1, max(count, 1), endpoint=False)
    from colorsys import hsv_to_rgb

    return np.array(
        [[int(c * 255) for c in hsv_to_rgb(h, 0.9, 1.0)] for h in hues], dtype=np.uint8
    )


def apply_transforms(
    images: Sequence[np.ndarray],
    transforms: Sequence[Any],
    output_shape: tuple[int, int] | None = None,
    order: int = 1,
) -> list[np.ndarray]:
    """Warp every slice into the reference frame.

    Uses :func:`tpsreg.warping.warp`, which tiles large outputs rather than
    materialising a coordinate array for the whole frame.
    """
    if output_shape is None:
        output_shape = images[0].shape[:2]

    return [
        warp(image, transform, output_shape=output_shape, order=order, cval=0)
        for image, transform in zip(images, transforms, strict=True)
    ]


def find_images(
    directory: Path, suffixes: Sequence[str] = IMAGE_SUFFIXES
) -> list[Path]:
    """Image files in a folder, in natural order.

    Sorted so that ``slice_2`` comes before ``slice_10``. Plain lexical
    sorting puts them the other way round, which silently reorders a stack --
    the kind of mistake that produces a plausible-looking but wrong result.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    wanted = {suffix.lower() for suffix in suffixes}
    paths = [p for p in directory.iterdir() if p.suffix.lower() in wanted]
    return sorted(paths, key=_natural_key)


def _natural_key(path: Path) -> tuple:
    """Sort key that reads runs of digits as numbers."""
    import re

    parts = re.split(r"(\d+)", path.name)
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)
