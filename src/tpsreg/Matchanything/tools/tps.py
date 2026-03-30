import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from skimage import transform as tf
from tqdm import tqdm


class ThinPlateSplineTransform:
    def __init__(self, affine_only=False, chunk_size=None, dtype=np.float32):
        self._estimated = False
        self.params = None
        self.affine_only = affine_only
        self.chunk_size = chunk_size
        self.dtype = dtype

    def __call__(self, coords):
        """Transform coordinates from source to destination using thin plate spline."""
        if not self._estimated:
            raise ValueError("Transformation not estimated.")
        params = np.moveaxis(self.params, 0, -1)
        coords = np.asarray(coords).astype(int)
        out = params[(coords[:, 1], coords[:, 0])]
        return out

    def _estimate_chunk_size(self, n_pixels, n_control_points, available_memory_gb=2.0):
        """
        Estimate optimal chunk size based on memory constraints.

        Parameters
        ----------
        n_pixels : int
            Total number of pixels to process
        n_control_points : int
            Number of control points
        available_memory_gb : float
            Available memory in GB for the computation

        Returns
        -------
        chunk_size : int
            Optimal number of pixels to process per chunk
        """
        # Memory per chunk: chunk_size * n_control_points * bytes_per_float
        bytes_per_element = np.dtype(self.dtype).itemsize

        # Main memory consumers:
        # 1. Distance matrix: chunk_size × n_control_points
        # 2. U matrix: chunk_size × n_control_points
        # 3. Intermediate arrays: ~2x the above for safety
        memory_per_pixel = (
            n_control_points * bytes_per_element * 4
        )  # 4x for safety margin

        available_bytes = available_memory_gb * 1024**3
        chunk_size = int(available_bytes / memory_per_pixel)

        # Clamp to reasonable bounds
        chunk_size = max(
            1000, min(chunk_size, n_pixels)
        )  # At least 1000, at most all pixels

        return chunk_size

    def estimate(self, src, dst, size, available_memory_gb=2.0):
        """Estimate optimal spline mappings between source and destination points.

        Parameters
        ----------
        src : (N, 2) array_like
            Control points at source coordinates.
        dst : (N, 2) array_like
            Control points at destination coordinates.
        size : tuple
            Size of the reference image (height, width).

        Returns
        -------
        success: bool
            True indicates that the estimation was successful.

        Notes
        -----
        The number N of source and destination points must match.
        """
        # convert input pixels in arrays. cps are control points
        xs = np.asarray(dst[:, 0])
        ys = np.array(dst[:, 1])
        cps = np.vstack([xs, ys]).T
        xt = np.asarray(src[:, 0])
        yt = np.array(src[:, 1])
        n = len(xs)
        # print("Number of control points:", n)

        # construct L
        L = self._TPS_makeL(cps)

        # construct Y
        xtAug = np.concatenate([xt, np.zeros(3)])
        ytAug = np.concatenate([yt, np.zeros(3)])
        Y = np.vstack([xtAug, ytAug]).T

        # calculate unknown params in (W | a).T
        params = np.linalg.solve(L, Y)
        wi = params[:n, :]
        a1 = params[n, :]
        ax = params[n + 1, :]
        ay = params[n + 2, :]

        # Thin plate spline calculation
        # at some point (x,y) in reference, the corresponding point in the distorted data is at
        # [X,Y] = a1 + ax*xRef + ay*yRef + sum(wi*Ui)
        # dimensions of reference image in pixels
        lx = size[1]
        ly = size[0]

        # for fineness of grid, if you want to fix all points, leave nx=lx, ny=ly
        nx = lx  # num points along reference x-direction, full correction will have nx = lx
        ny = ly  # num points along reference y-direction, full correction will have ny = ly
        n_pixels = nx * ny

        # (x,y) coordinates from reference image
        x = np.linspace(1, lx, nx)
        y = np.linspace(1, ly, ny)
        xgd, ygd = np.meshgrid(x, y)
        pixels = np.vstack([xgd.flatten(), ygd.flatten()]).T

        # affine transformation portion
        axs = np.einsum("i,jk->ijk", ax, xgd)
        ays = np.einsum("i,jk->ijk", ay, ygd)
        affine = axs + ays
        affine[0, :, :] += a1[0]
        affine[1, :, :] += a1[1]
        del xgd, ygd, x, y

        if self.affine_only:
            self.params = affine
        else:
            # Determine chunk size
            if self.chunk_size is None:
                chunk_size = self._estimate_chunk_size(n_pixels, n, available_memory_gb)
            else:
                chunk_size = self.chunk_size

            n_chunks = int(np.ceil(n_pixels / chunk_size))
            print(
                f"Processing {n_pixels:,} pixels in {n_chunks} chunk(s) of ~{chunk_size:,} pixels each"
            )

            # Compute bending portion in chunks
            print("Computing bending transformation...")
            bend = np.zeros((2, ny, nx), dtype=self.dtype)

            for chunk_idx in tqdm(range(n_chunks)):
                start_idx = chunk_idx * chunk_size
                end_idx = min((chunk_idx + 1) * chunk_size, n_pixels)
                chunk_pixels = pixels[start_idx:end_idx]

                # Vectorized distance calculation for this chunk
                R = cdist(chunk_pixels, cps, "euclidean").astype(self.dtype)
                Rsq = R * R
                Rsq[R == 0] = 1  # Avoid log(0)
                U = Rsq * np.log(Rsq)

                # Matrix multiplication: (chunk_size, n) @ (n, 2) = (chunk_size, 2)
                bend_chunk = U @ wi

                # Reshape and place into output array
                chunk_len = end_idx - start_idx
                bend_chunk_reshaped = bend_chunk.T.reshape(2, chunk_len)

                # Map flat indices back to 2D coordinates
                y_indices = (start_idx + np.arange(chunk_len)) // nx
                x_indices = (start_idx + np.arange(chunk_len)) % nx

                bend[:, y_indices, x_indices] = bend_chunk_reshaped

                # Clean up chunk memory
                del R, Rsq, U, bend_chunk, bend_chunk_reshaped

            self.params = affine + bend

        self.size = size
        self._estimated = True
        return True

    def _TPS_makeL(self, cp):
        """Function to make the L matrix for thin plate spline calculation."""
        # cp: [K x 2] control points
        # L: [(K+3) x (K+3)]
        K = cp.shape[0]
        L = np.zeros((K + 3, K + 3))
        # make P in L
        L[:K, K] = 1
        L[:K, K + 1 : K + 3] = cp
        # make P.T in L
        L[K, :K] = 1
        L[K + 1 :, :K] = cp.T
        R = cdist(cp, cp, "euclidean")
        Rsq = R * R
        Rsq[R == 0] = (
            1  # avoid log(0) undefined, will correct itself as log(1) = 0, so U(0) = 0
        )
        U = Rsq * np.log(Rsq)
        np.fill_diagonal(U, 0)  # should be redundant
        L[:K, :K] = U
        return L


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time

    tform = ThinPlateSplineTransform()

    Ns = [100, 1000, 5000]
    results = []
    for N in Ns:
        np.random.seed(0)
        src = np.random.rand(5000, 2) * N
        dst = src + (np.random.rand(5000, 2) - 0.5) * 10  # small random displacement
        size = (N, N)

        start_time = time.time()
        tform.estimate(src, dst, size, available_memory_gb=20.0)
        end_time = time.time()

        results.append((N, end_time - start_time))

    results = np.array(results)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    ax.plot(results[:, 0], results[:, 1], "o-", color="blue", label="Time (s)")
    ax.set_xscale("log")
    ax.set_xlabel("Number of Control Points")
    ax.set_ylabel("Time (s)")
    ax.legend(loc="upper left")
    plt.title("Thin Plate Spline Transform Performance (5k X 5k image)")
    plt.show()
