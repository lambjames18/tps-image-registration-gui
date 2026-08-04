"""Ways of showing a warped image against its target.

The preview window only ever offered a wipe: drag a slider and one image
covers the other. That answers "is the edge in the right place along this
line?" but not "is the whole thing aligned?". Checkerboard and difference
answer the second question, and both are one array operation.

These are plain array functions with no Tk in sight, so the compositing can be
checked without opening a window.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

#: Selectable modes, in the order the view should offer them.
BLEND_MODES: tuple[str, ...] = ("wipe", "checkerboard", "difference")

#: Default checkerboard square, in pixels.
DEFAULT_TILE_SIZE = 32


def to_rgb(image: np.ndarray) -> np.ndarray:
    """Return ``image`` as ``(H, W, 3)`` uint8.

    Accepts grayscale, single-channel, RGB, or RGBA input, and both float
    images in ``[0, 1]`` and integer images in ``[0, 255]``.
    """
    array = np.asarray(image)

    if array.dtype.kind == "f":
        # Float images are the [0, 1] convention used throughout the package.
        array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    if array.ndim == 2:
        return np.stack([array] * 3, axis=-1)

    if array.ndim != 3:
        raise ValueError(f"Expected a 2D or 3D image, got shape {array.shape}")

    if array.shape[2] == 1:
        return np.repeat(array, 3, axis=2)
    if array.shape[2] == 3:
        return array
    if array.shape[2] == 4:
        # Drop alpha: these composites manage their own blending.
        return array[:, :, :3]

    raise ValueError(f"Unsupported channel count: {array.shape[2]}")


def _matched(overlay: np.ndarray, background: np.ndarray) -> tuple[np.ndarray, ...]:
    """Both images as RGB, cropped to the region they share.

    A warped image and its target can differ by a row or column depending on
    the crop mode. Cropping is what keeps that from being a broadcast error in
    the middle of a preview.
    """
    top = to_rgb(overlay)
    bottom = to_rgb(background)

    if top.shape != bottom.shape:
        height = min(top.shape[0], bottom.shape[0])
        width = min(top.shape[1], bottom.shape[1])
        logger.debug(
            "Preview images differ in size (%s vs %s); using the shared %dx%d region",
            top.shape,
            bottom.shape,
            height,
            width,
        )
        top = top[:height, :width]
        bottom = bottom[:height, :width]

    return top, bottom


def wipe(overlay: np.ndarray, background: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """Blend ``overlay`` over ``background`` using a per-pixel alpha mask.

    ``alphas`` is the 0-to-1 mask the wipe sliders build. This reproduces what
    the preview window has always done, as an array operation rather than a
    round trip through PIL.
    """
    top, bottom = _matched(overlay, background)

    mask = np.asarray(alphas, dtype=float)
    if mask.shape != top.shape[:2]:
        mask = mask[: top.shape[0], : top.shape[1]]
    mask = np.clip(mask, 0.0, 1.0)[:, :, None]

    blended = top.astype(float) * mask + bottom.astype(float) * (1.0 - mask)
    return np.clip(blended, 0, 255).astype(np.uint8)


def checkerboard(
    overlay: np.ndarray, background: np.ndarray, tile_size: int = DEFAULT_TILE_SIZE
) -> np.ndarray:
    """Interleave the two images in alternating squares.

    Misalignment shows up as features stepping sideways at every tile
    boundary, which is far easier to see than a brightness difference.
    """
    top, bottom = _matched(overlay, background)
    height, width = top.shape[:2]

    tile_size = max(1, int(tile_size))
    rows = np.arange(height)[:, None] // tile_size
    cols = np.arange(width)[None, :] // tile_size
    mask = ((rows + cols) % 2 == 0)[:, :, None]

    return np.where(mask, top, bottom)


def difference(overlay: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Absolute difference between the two images.

    A perfect alignment is black. Edges that glow are edges that did not line
    up, and the brightness is proportional to how far off they are.
    """
    top, bottom = _matched(overlay, background)
    return np.abs(top.astype(np.int16) - bottom.astype(np.int16)).astype(np.uint8)


def composite(
    mode: str,
    overlay: np.ndarray,
    background: np.ndarray,
    alphas: np.ndarray | None = None,
    tile_size: int = DEFAULT_TILE_SIZE,
) -> np.ndarray:
    """Combine two images using the named mode.

    Raises
    ------
    ValueError
        If ``mode`` is not one of :data:`BLEND_MODES`.
    """
    if mode == "wipe":
        if alphas is None:
            alphas = np.ones(to_rgb(overlay).shape[:2], dtype=float)
        return wipe(overlay, background, alphas)
    if mode == "checkerboard":
        return checkerboard(overlay, background, tile_size)
    if mode == "difference":
        return difference(overlay, background)

    raise ValueError(f"Unknown blend mode: {mode!r}. Expected one of {BLEND_MODES}.")
