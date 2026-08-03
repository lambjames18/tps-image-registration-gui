"""Thin-plate spline transformation.

The transform is stored as a dense displacement field covering the destination
grid, which makes it directly usable as the ``inverse_map`` callable expected by
:func:`skimage.transform.warp`.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

# Minimum number of non-collinear control points needed to solve the TPS system.
MIN_CONTROL_POINTS = 3


class ThinPlateSplineTransform:
    """Thin-plate spline mapping from destination coordinates back to source.

    The transform is evaluated over the whole destination grid at estimation
    time and cached in :attr:`params` as a ``(2, height, width)`` array of
    source coordinates. Calling the instance then reduces to an array lookup,
    which is what makes warping large stacks tractable.

    Parameters
    ----------
    affine_only:
        Fit only the affine part of the spline, discarding the bending energy
        term. Useful when the expected distortion is a pure affine.
    chunk_size:
        Number of destination pixels to evaluate per chunk. Defaults to a value
        derived from ``available_memory_gb``.
    dtype:
        Floating point type used for the (large) intermediate distance matrices.
    """

    def __init__(
        self,
        affine_only: bool = False,
        chunk_size: Optional[int] = None,
        dtype: type = np.float32,
    ):
        self._estimated = False
        self.params: Optional[np.ndarray] = None
        self.size: Optional[Tuple[int, int]] = None
        self.affine_only = affine_only
        self.chunk_size = chunk_size
        self.dtype = dtype

    def __call__(self, coords: np.ndarray) -> np.ndarray:
        """Map destination coordinates to source coordinates.

        Parameters
        ----------
        coords:
            ``(N, 2)`` array of ``(x, y)`` destination coordinates.

        Returns
        -------
        np.ndarray
            ``(N, 2)`` array of ``(x, y)`` source coordinates.

        Raises
        ------
        ValueError
            If the transform has not been estimated yet.
        """
        if not self._estimated:
            raise ValueError(
                "Transformation not estimated. Call estimate() before applying it."
            )

        params = np.moveaxis(self.params, 0, -1)
        coords = np.asarray(coords).astype(int)

        # Clamp to the sampled grid: skimage.transform.warp queries coordinates
        # over the full output shape, which can exceed the estimated grid when
        # the caller asks for a larger output than the reference size.
        height, width = params.shape[:2]
        rows = np.clip(coords[:, 1], 0, height - 1)
        cols = np.clip(coords[:, 0], 0, width - 1)

        return params[rows, cols]

    def _estimate_chunk_size(
        self,
        n_pixels: int,
        n_control_points: int,
        available_memory_gb: float = 2.0,
    ) -> int:
        """Estimate a chunk size that keeps peak memory near the given budget.

        Parameters
        ----------
        n_pixels:
            Total number of pixels to process.
        n_control_points:
            Number of control points.
        available_memory_gb:
            Memory budget in GB for the computation.

        Returns
        -------
        int
            Number of pixels to process per chunk.
        """
        bytes_per_element = np.dtype(self.dtype).itemsize

        # Peak usage per pixel is dominated by the distance matrix and the U
        # matrix, each chunk_size x n_control_points; 4x leaves headroom for
        # the intermediates SciPy allocates.
        memory_per_pixel = n_control_points * bytes_per_element * 4

        available_bytes = available_memory_gb * 1024**3
        chunk_size = int(available_bytes / memory_per_pixel)

        # At least 1000 pixels per chunk, never more than the whole image.
        return max(1000, min(chunk_size, n_pixels))

    @staticmethod
    def _check_valid_points(src: np.ndarray, dst: np.ndarray) -> bool:
        """Validate a set of control point correspondences.

        Raises
        ------
        ValueError
            If the arrays disagree in shape, are not 2D coordinates, hold fewer
            than :data:`MIN_CONTROL_POINTS` points, or contain duplicates.
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

        return True

    def estimate(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        size: Tuple[int, int],
        available_memory_gb: float = 2.0,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> bool:
        """Estimate the spline mapping between source and destination points.

        Parameters
        ----------
        src:
            ``(N, 2)`` control points in source coordinates.
        dst:
            ``(N, 2)`` control points in destination coordinates.
        size:
            ``(height, width)`` of the destination grid.
        available_memory_gb:
            Memory budget used to pick a chunk size when ``chunk_size`` is None.
        progress_callback:
            Optional ``callback(completed_chunks, total_chunks)`` invoked after
            each chunk, so a GUI can drive a progress bar.

        Returns
        -------
        bool
            True when the estimation succeeded.

        Notes
        -----
        The number N of source and destination points must match.
        """
        self._check_valid_points(src, dst)

        src = np.asarray(src, dtype=float)
        dst = np.asarray(dst, dtype=float)

        # Control points live on the destination grid; the spline maps them
        # back to the source, which is the direction skimage.warp needs.
        cps = np.vstack([dst[:, 0], dst[:, 1]]).T
        xt = src[:, 0]
        yt = src[:, 1]
        n = cps.shape[0]

        L = self._TPS_makeL(cps)

        # Right-hand side, padded with the three affine constraints.
        xt_aug = np.concatenate([xt, np.zeros(3)])
        yt_aug = np.concatenate([yt, np.zeros(3)])
        Y = np.vstack([xt_aug, yt_aug]).T

        try:
            params = np.linalg.solve(L, Y)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "Could not solve the thin-plate spline system. This usually "
                "means the control points are collinear or nearly coincident."
            ) from exc

        wi = params[:n, :]
        a1 = params[n, :]
        ax = params[n + 1, :]
        ay = params[n + 2, :]

        # At (x, y) in the destination, the corresponding source point is
        # a1 + ax*x + ay*y + sum(wi * U(r)).
        height, width = int(size[0]), int(size[1])
        n_pixels = width * height

        x = np.linspace(1, width, width)
        y = np.linspace(1, height, height)
        xgd, ygd = np.meshgrid(x, y)

        # Affine component, evaluated over the full grid at once.
        affine = np.einsum("i,jk->ijk", ax, xgd) + np.einsum("i,jk->ijk", ay, ygd)
        affine[0, :, :] += a1[0]
        affine[1, :, :] += a1[1]

        if self.affine_only:
            self.params = affine.astype(self.dtype)
        else:
            pixels = np.vstack([xgd.flatten(), ygd.flatten()]).T
            del xgd, ygd, x, y

            chunk_size = self.chunk_size or self._estimate_chunk_size(
                n_pixels, n, available_memory_gb
            )
            n_chunks = int(np.ceil(n_pixels / chunk_size))
            logger.info(
                "Computing bending transformation over %d pixels in %d chunk(s) "
                "of ~%d pixels",
                n_pixels,
                n_chunks,
                chunk_size,
            )

            bend = np.zeros((2, height, width), dtype=self.dtype)

            for chunk_idx in range(n_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min((chunk_idx + 1) * chunk_size, n_pixels)
                chunk_pixels = pixels[start_idx:end_idx]

                R = cdist(chunk_pixels, cps, "euclidean").astype(self.dtype)
                Rsq = R * R
                Rsq[R == 0] = 1  # U(0) = 0 via log(1); avoids log(0)
                U = Rsq * np.log(Rsq)

                bend_chunk = U @ wi

                chunk_len = end_idx - start_idx
                flat_indices = start_idx + np.arange(chunk_len)
                y_indices = flat_indices // width
                x_indices = flat_indices % width
                bend[:, y_indices, x_indices] = bend_chunk.T.reshape(2, chunk_len)

                del R, Rsq, U, bend_chunk

                if progress_callback is not None:
                    progress_callback(chunk_idx + 1, n_chunks)

            self.params = (affine + bend).astype(self.dtype)

        self.size = (height, width)
        self._estimated = True
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

        R = cdist(cp, cp, "euclidean")
        Rsq = R * R
        # U(0) is 0; substituting 1 makes log(1) = 0 rather than log(0) = -inf.
        Rsq[R == 0] = 1
        U = Rsq * np.log(Rsq)
        np.fill_diagonal(U, 0)
        L[:K, :K] = U

        return L
