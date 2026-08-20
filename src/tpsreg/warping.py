"""Convenience helpers for estimating and applying image transformations.

These wrap :class:`tpsreg.tps.ThinPlateSplineTransform` and scikit-image's
transform estimators behind one interface, so callers can switch between a
deformable spline and a rigid model by changing a string.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from skimage import img_as_float
from skimage import transform as tf

from tpsreg.tps import ThinPlateSplineTransform

logger = logging.getLogger(__name__)

#: Transform names handled by this module in addition to scikit-image's own
#: ("euclidean", "similarity", "affine", "piecewise-affine", "projective",
#: "polynomial").
TPS_MODES = ("tps",)

#: Largest output tile edge, in pixels, used when warping.
#: ``skimage.transform.warp`` builds one coordinate array for the whole output
#: -- 16 bytes per pixel, so 6.4 GB for a 400 Mpx stitched image before any
#: pixels are read. Warping a tile at a time bounds that by the tile instead.
#:
#: Measured, big tiles are worse on both counts: warping 9 Mpx over 30 control
#: points took 9.1 s and 832 MB at 1024, against 2.5 s and 61 MB at 256. The
#: per-tile kernel matrix stops fitting in cache long before it stops fitting
#: in memory.
DEFAULT_TILE = 256

#: Smallest tile the adaptive sizing will choose. Below this the per-tile
#: overhead starts to dominate.
MIN_TILE = 128

#: Memory the per-tile kernel matrix is allowed to occupy. Its size is
#: ``tile**2 * n_control_points * 8`` bytes, twice over (distances, then the
#: kernel), which is what makes the tile depend on the control point count.
TILE_MEMORY_BUDGET_GB = 0.125

#: Outputs at or below this many pixels go through skimage in one pass; tiling
#: only pays for itself once the coordinate array is the problem.
TILING_THRESHOLD = 4_000_000


def _tile_for(tform: Any, tile: int | None) -> int:
    """Choose a tile edge, from the control point count when not told.

    A fixed tile cannot be right for every fit: the work per tile scales with
    the number of control points, so what is comfortable for ten is heavy for
    four hundred.
    """
    if tile is not None:
        return max(1, int(tile))

    control_points = getattr(tform, "control_points", None)
    if control_points is None or len(control_points) == 0:
        return DEFAULT_TILE

    budget = TILE_MEMORY_BUDGET_GB * 1024**3
    max_coords = budget / (len(control_points) * 8 * 2)
    return int(np.clip(np.sqrt(max_coords), MIN_TILE, DEFAULT_TILE))


def warp(
    image: np.ndarray,
    tform: Any,
    output_shape: tuple[int, int] | None = None,
    order: int = 0,
    cval: float = 0,
    tile: int | None = None,
) -> np.ndarray:
    """Warp an image through a transform, a tile at a time when it is large.

    Equivalent to :func:`skimage.transform.warp` with ``mode="constant"``, and
    delegates to it for small outputs. The difference is peak memory: skimage
    materialises coordinates for the entire output at once, which is what put
    a ceiling on the image sizes this tool could handle.

    Parameters
    ----------
    image:
        ``(H, W)`` or ``(H, W, C)`` input.
    tform:
        Anything callable with an ``(N, 2)`` array of ``(x, y)`` destination
        coordinates that returns source coordinates in the same form.
    output_shape:
        ``(height, width)`` of the result. Defaults to the input shape.
    order:
        Spline interpolation order.
    cval:
        Value for output pixels that map outside the input.
    tile:
        Edge length of each output tile. Chosen from the control point count
        when not given; see :func:`_tile_for`.

    Returns
    -------
    np.ndarray
        The warped image, with the dtype and range skimage would produce.
    """
    if output_shape is None:
        output_shape = image.shape[:2]
    output_shape = (int(output_shape[0]), int(output_shape[1]))

    if output_shape[0] * output_shape[1] <= TILING_THRESHOLD:
        return tf.warp(
            image,
            tform,
            output_shape=output_shape,
            mode="constant",
            cval=cval,
            order=order,
        )

    return _warp_tiled(image, tform, output_shape, order=order, cval=cval, tile=tile)


class _ShiftedTransform:
    """A transform with its query coordinates offset into a tile.

    Each tile asks about its own local coordinates; the transform is defined
    over the whole output, so the tile's origin is added back before mapping.
    """

    def __init__(self, tform: Any, left: int, top: int):
        self._tform = tform
        self._offset = np.array([float(left), float(top)])

    def __call__(self, coords: np.ndarray) -> np.ndarray:
        return self._tform(np.asarray(coords, dtype=float) + self._offset)


def _warp_tiled(
    image: np.ndarray,
    tform: Any,
    output_shape: tuple[int, int],
    order: int = 0,
    cval: float = 0,
    tile: int | None = None,
) -> np.ndarray:
    """Warp one output tile at a time, letting skimage sample each tile.

    Sampling is delegated rather than reimplemented, because skimage's
    behaviour is not uniform and quietly changing it would alter every export:
    at ``order=0`` it preserves the input dtype and range, while at higher
    orders it converts integer images to float in ``[0, 1]`` and clips the
    result to the input's range. Handing each tile to skimage gets all of that
    for free, and identically, for every dtype and order.

    Only the coordinates are ours, and they are the thing worth tiling: one
    array for a 400 Mpx output is 6.4 GB, while a 1024-pixel tile needs 16 MB.
    """
    height, width = output_shape
    tile = _tile_for(tform, tile)

    # At order >= 1 skimage converts the image to float on every call. Doing
    # it once up front and declaring the range preserved turns that into a
    # no-op, instead of re-converting the whole image for every tile.
    source = image if order == 0 else img_as_float(image)
    preserve_range = order != 0

    n_tiles = ((height + tile - 1) // tile) * ((width + tile - 1) // tile)
    logger.info(
        "Warping a %dx%d output as %d tile(s) of up to %dx%d",
        height,
        width,
        n_tiles,
        tile,
        tile,
    )

    warped: np.ndarray | None = None

    for top in range(0, height, tile):
        bottom = min(top + tile, height)
        for left in range(0, width, tile):
            right = min(left + tile, width)

            piece = tf.warp(
                source,
                _ShiftedTransform(tform, left, top),
                output_shape=(bottom - top, right - left),
                mode="constant",
                cval=cval,
                order=order,
                preserve_range=preserve_range,
            )

            if warped is None:
                warped = np.empty((height, width, *piece.shape[2:]), dtype=piece.dtype)
            warped[top:bottom, left:right] = piece

    if warped is None:  # pragma: no cover - zero-sized output
        return np.empty((height, width), dtype=image.dtype)
    return warped


def get_transform(src: np.ndarray, dst: np.ndarray, mode: str, *args, **kwargs) -> Any:
    """Estimate a transform from point correspondences.

    Parameters
    ----------
    src, dst:
        ``(N, 2)`` arrays of corresponding coordinates.
    mode:
        ``"tps"``, or any mode accepted by
        :func:`skimage.transform.estimate_transform`.
    *args, **kwargs:
        Forwarded to the underlying estimator. For the TPS modes the first
        positional argument is the ``(height, width)`` reference size.

    Returns
    -------
    The estimated transform object.
    """
    mode_lower = mode.lower()

    if mode_lower == "tps":
        tform = ThinPlateSplineTransform()
        tform.estimate(src, dst, *args, **kwargs)
    else:
        tform = tf.estimate_transform(mode_lower, src, dst, *args, **kwargs)

    return tform


def get_transform_params(tform: Any) -> np.ndarray:
    """Return the parameter array of an estimated transform.

    For a TPS this is the fitted coefficients -- a few hundred bytes -- rather
    than the dense field it used to be.
    """
    return tform.params


def set_transform_params(tform: Any, params: np.ndarray) -> None:
    """Install pre-computed parameters on a transform, marking it estimated.

    A TPS also needs its control points to be usable, so installing
    coefficients alone is only meaningful alongside
    ``tform.control_points``. To install a dense field instead -- which is
    self-contained -- use :meth:`ThinPlateSplineTransform.set_field`.
    """
    tform.params = params
    tform._estimated = True


def interpolate_fields(
    lower: np.ndarray, upper: np.ndarray, weight: float
) -> np.ndarray:
    """Blend two displacement fields.

    Slices are fitted to their own control points, so their coefficients have
    different lengths and different meanings and cannot be averaged. Their
    displacement fields share a grid, which makes them the representation that
    can be interpolated -- the one job the dense form is genuinely better at.
    """
    return (1.0 - weight) * np.asarray(lower) + weight * np.asarray(upper)


def transform_coords(
    src: np.ndarray,
    dst: np.ndarray,
    mode: str = "tps",
    return_params: bool = False,
    *args,
    **kwargs,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Transform coordinates from source to destination.

    Parameters
    ----------
    src, dst:
        ``(N, 2)`` arrays of corresponding coordinates.
    mode:
        Transformation mode; see :func:`get_transform`.
    return_params:
        Also return the fitted parameter array.
    *args, **kwargs:
        Forwarded to the estimator.

    Returns
    -------
    np.ndarray
        ``(N, 2)`` transformed coordinates, or ``(coords, params)`` when
        ``return_params`` is True.

    Notes
    -----
    This evaluates the spline at the requested points and nothing else. It
    used to build the dense field over the whole destination grid first and
    then index it, which for a handful of points was around five orders of
    magnitude slower and needed hundreds of megabytes to answer a question
    whose answer is a few hundred bytes.
    """
    tform = get_transform(src, dst, mode, *args, **kwargs)
    warped = tform(np.asarray(src, dtype=float))

    if return_params:
        return warped, get_transform_params(tform)
    return warped


def transform_image(
    image: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    output_shape: tuple[int, int] | None = None,
    mode: str = "tps",
    order: int = 0,
    return_params: bool = False,
    *args,
    **kwargs,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Warp a single image using a transform fitted to point correspondences.

    Parameters
    ----------
    image:
        ``(H, W)`` or ``(H, W, C)`` input image.
    src, dst:
        ``(N, 2)`` arrays of corresponding coordinates.
    output_shape:
        ``(height, width)`` of the result. Defaults to the input shape.
    mode:
        Transformation mode; see :func:`get_transform`.
    order:
        Spline interpolation order passed to :func:`skimage.transform.warp`.
    return_params:
        Also return the fitted parameter array.

    Returns
    -------
    np.ndarray
        The warped image, or ``(image, params)`` when ``return_params`` is True.
    """
    if output_shape is None:
        output_shape = image.shape[:2]

    # The TPS modes record the destination size; they no longer need it to
    # evaluate, but passing it keeps exports self-describing.
    if mode.lower() in TPS_MODES and not args and "size" not in kwargs:
        args = (tuple(output_shape),)

    tform = get_transform(src, dst, mode, *args, **kwargs)
    warped = warp(image, tform, output_shape=output_shape, order=order, cval=0)

    if return_params:
        return warped, get_transform_params(tform)
    return warped


def transform_image_stack(
    images: np.ndarray,
    srcs: np.ndarray,
    dsts: np.ndarray,
    output_shape: tuple[int, int] | None = None,
    mode: str = "tps",
    order: int = 0,
    params: np.ndarray | None = None,
    return_params: bool = False,
    downsample: int = 1,
    *args,
    **kwargs,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Warp a stack of images, interpolating between keyed slices.

    Control points are only needed on some slices. Those slices are fitted and
    the transform is linearly interpolated in between, which is what makes a
    serial-sectioning dataset practical to align.

    Parameters
    ----------
    images:
        ``(N, H, W)`` or ``(N, H, W, C)`` stack.
    srcs, dsts:
        ``(M, 3)`` arrays of ``[slice, x, y]`` correspondences.
    output_shape:
        ``(height, width)`` of each output slice. Defaults to the input size.
    mode:
        Transformation mode; see :func:`get_transform`.
    order:
        Spline interpolation order.
    params:
        Pre-computed per-slice displacement fields, skipping estimation. This
        is the one place the dense form is still the working representation;
        see the note below.
    return_params:
        Also return the per-slice field array.
    downsample:
        Resolution of the per-slice fields. The whole stack is held in memory
        at once, so this is the knob that decides whether a large stack fits:
        1/4 resolution costs about 0.004 px and a sixteenth of the memory.

    Returns
    -------
    np.ndarray
        The warped stack, or ``(stack, fields)`` when ``return_params`` is
        True.

    Notes
    -----
    Unlike everywhere else, the per-slice representation here is the dense
    field rather than the coefficients. Interpolating between slices needs a
    shared representation, and consecutive slices are fitted to different
    control points -- different in number and in position -- so their
    coefficients cannot be averaged. Their fields can.
    """
    if output_shape is None:
        output_shape = images.shape[1:3]
    output_shape = tuple(output_shape)

    estimate_args = args
    if mode.lower() in TPS_MODES and not args and "size" not in kwargs:
        estimate_args = (output_shape,)

    if params is None:
        srcs = np.asarray(srcs, dtype=float)
        dsts = np.asarray(dsts, dtype=float)

        slice_numbers = np.arange(images.shape[0])
        slice_numbers_with_points = np.unique(srcs[:, 0]).astype(int)

        if slice_numbers_with_points.size == 0:
            raise ValueError("No control points supplied for any slice")

        # Interpolation needs the first and last slice to be keyed. Users should
        # place points there; duplicate the nearest keyed slice when they don't.
        if slice_numbers[0] not in slice_numbers_with_points:
            logger.info(
                "First slice has no points; extending from slice %d",
                slice_numbers_with_points[0],
            )
            src_temp = srcs[srcs[:, 0] == slice_numbers_with_points[0], 1:]
            dst_temp = dsts[dsts[:, 0] == slice_numbers_with_points[0], 1:]
            pad = np.zeros((src_temp.shape[0], 1))
            srcs = np.concatenate([np.concatenate([pad, src_temp], axis=1), srcs])
            dsts = np.concatenate([np.concatenate([pad, dst_temp], axis=1), dsts])
            slice_numbers_with_points = np.concatenate([[0], slice_numbers_with_points])

        if slice_numbers[-1] not in slice_numbers_with_points:
            logger.info(
                "Last slice has no points; extending from slice %d",
                slice_numbers_with_points[-1],
            )
            src_temp = srcs[srcs[:, 0] == slice_numbers_with_points[-1], 1:]
            dst_temp = dsts[dsts[:, 0] == slice_numbers_with_points[-1], 1:]
            pad = np.full((src_temp.shape[0], 1), slice_numbers[-1])
            srcs = np.concatenate([srcs, np.concatenate([pad, src_temp], axis=1)])
            dsts = np.concatenate([dsts, np.concatenate([pad, dst_temp], axis=1)])
            slice_numbers_with_points = np.concatenate(
                [slice_numbers_with_points, [slice_numbers[-1]]]
            )

        logger.debug("Slices with control points: %s", slice_numbers_with_points)

        # Fit each keyed slice and evaluate it onto the shared grid, building
        # the "knots" along the z axis. Fields rather than coefficients: each
        # slice is fitted to its own control points, so their coefficient
        # vectors differ in length and meaning and cannot be blended, whereas
        # their fields all live on this one grid.
        params = None
        for slice_number in slice_numbers_with_points:
            src = srcs[srcs[:, 0] == slice_number, 1:]
            dst = dsts[dsts[:, 0] == slice_number, 1:]
            tform_temp = get_transform(src, dst, mode, *estimate_args, **kwargs)
            slice_field = tform_temp.build_field(output_shape, downsample=downsample)
            if params is None:
                params = np.zeros(
                    (images.shape[0], *slice_field.shape), dtype=slice_field.dtype
                )
            params[slice_number] = slice_field

        # Linearly interpolate between consecutive knots.
        for i in range(1, len(slice_numbers_with_points)):
            lower = slice_numbers_with_points[i - 1]
            upper = slice_numbers_with_points[i]
            if upper - lower < 2:
                continue
            params[lower : upper + 1] = np.linspace(
                params[lower], params[upper], upper - lower + 1
            )

    tform = ThinPlateSplineTransform()

    output = []
    for i in range(images.shape[0]):
        tform.set_field(params[i], size=output_shape, step=downsample)
        output.append(
            warp(images[i], tform, output_shape=output_shape, order=order, cval=0)
        )

    stack = np.array(output)
    if return_params:
        return stack, params
    return stack
