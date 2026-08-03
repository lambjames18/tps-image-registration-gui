"""Convenience helpers for estimating and applying image transformations.

These wrap :class:`tpsreg.tps.ThinPlateSplineTransform` and scikit-image's
transform estimators behind one interface, so callers can switch between a
deformable spline and a rigid model by changing a string.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import numpy as np
from skimage import transform as tf

from tpsreg.tps import ThinPlateSplineTransform

logger = logging.getLogger(__name__)

#: Transform names handled by this module in addition to scikit-image's own
#: ("euclidean", "similarity", "affine", "piecewise-affine", "projective",
#: "polynomial").
TPS_MODES = ("tps", "tps affine")


def get_transform(src: np.ndarray, dst: np.ndarray, mode: str, *args, **kwargs) -> Any:
    """Estimate a transform from point correspondences.

    Parameters
    ----------
    src, dst:
        ``(N, 2)`` arrays of corresponding coordinates.
    mode:
        ``"tps"``, ``"tps affine"``, or any mode accepted by
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
    elif mode_lower == "tps affine":
        tform = ThinPlateSplineTransform(affine_only=True)
        tform.estimate(src, dst, *args, **kwargs)
    else:
        tform = tf.estimate_transform(mode_lower, src, dst, *args, **kwargs)

    return tform


def get_transform_params(tform: Any) -> np.ndarray:
    """Return the parameter array of an estimated transform."""
    return tform.params


def set_transform_params(tform: Any, params: np.ndarray) -> None:
    """Install pre-computed parameters on a transform, marking it estimated."""
    tform.params = params
    tform._estimated = True


def transform_coords(
    src: np.ndarray,
    dst: np.ndarray,
    mode: str = "tps",
    return_params: bool = False,
    *args,
    **kwargs,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
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
    """
    tform = get_transform(src, dst, mode, *args, **kwargs)
    warped = tform(src)

    if return_params:
        return warped, get_transform_params(tform)
    return warped


def transform_image(
    image: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    output_shape: Optional[Tuple[int, int]] = None,
    mode: str = "tps",
    order: int = 0,
    return_params: bool = False,
    *args,
    **kwargs,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
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

    # The TPS modes evaluate over the destination grid, so they need its size.
    if mode.lower() in TPS_MODES and not args and "size" not in kwargs:
        args = (tuple(output_shape),)

    tform = get_transform(src, dst, mode, *args, **kwargs)
    warped = tf.warp(
        image, tform, mode="constant", cval=0, order=order, output_shape=output_shape
    )

    if return_params:
        return warped, get_transform_params(tform)
    return warped


def transform_image_stack(
    images: np.ndarray,
    srcs: np.ndarray,
    dsts: np.ndarray,
    output_shape: Optional[Tuple[int, int]] = None,
    mode: str = "tps",
    order: int = 0,
    params: Optional[np.ndarray] = None,
    return_params: bool = False,
    *args,
    **kwargs,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """Warp a stack of images, interpolating parameters between keyed slices.

    Control points are only needed on some slices. Parameters are fitted on
    those slices and linearly interpolated in between, which is what makes a
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
        Pre-computed per-slice parameters, skipping estimation entirely.
    return_params:
        Also return the per-slice parameter array.

    Returns
    -------
    np.ndarray
        The warped stack, or ``(stack, params)`` when ``return_params`` is True.
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
            logger.info("First slice has no points; extending from slice %d",
                        slice_numbers_with_points[0])
            src_temp = srcs[srcs[:, 0] == slice_numbers_with_points[0], 1:]
            dst_temp = dsts[dsts[:, 0] == slice_numbers_with_points[0], 1:]
            pad = np.zeros((src_temp.shape[0], 1))
            srcs = np.concatenate([np.concatenate([pad, src_temp], axis=1), srcs])
            dsts = np.concatenate([np.concatenate([pad, dst_temp], axis=1), dsts])
            slice_numbers_with_points = np.concatenate(
                [[0], slice_numbers_with_points]
            )

        if slice_numbers[-1] not in slice_numbers_with_points:
            logger.info("Last slice has no points; extending from slice %d",
                        slice_numbers_with_points[-1])
            src_temp = srcs[srcs[:, 0] == slice_numbers_with_points[-1], 1:]
            dst_temp = dsts[dsts[:, 0] == slice_numbers_with_points[-1], 1:]
            pad = np.full((src_temp.shape[0], 1), slice_numbers[-1])
            srcs = np.concatenate([srcs, np.concatenate([pad, src_temp], axis=1)])
            dsts = np.concatenate([dsts, np.concatenate([pad, dst_temp], axis=1)])
            slice_numbers_with_points = np.concatenate(
                [slice_numbers_with_points, [slice_numbers[-1]]]
            )

        logger.debug("Slices with control points: %s", slice_numbers_with_points)

        # Fit each keyed slice, building the "knots" along the z axis.
        params = None
        for slice_number in slice_numbers_with_points:
            src = srcs[srcs[:, 0] == slice_number, 1:]
            dst = dsts[dsts[:, 0] == slice_number, 1:]
            tform_temp = get_transform(src, dst, mode, *estimate_args, **kwargs)
            slice_params = get_transform_params(tform_temp)
            if params is None:
                params = np.zeros((images.shape[0], *slice_params.shape))
            params[slice_number] = slice_params

        # Linearly interpolate parameters between consecutive knots.
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
        set_transform_params(tform, params[i])
        output.append(
            tf.warp(
                images[i],
                tform,
                output_shape=output_shape,
                mode="constant",
                cval=0,
                order=order,
            )
        )

    stack = np.array(output)
    if return_params:
        return stack, params
    return stack
