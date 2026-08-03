"""tpsreg - multimodal image registration via thin-plate splines.

The public surface is intentionally small. Importing this package pulls in only
NumPy/SciPy-level dependencies; the Tkinter GUI and the optional deep-learning
matcher are imported lazily by the modules that need them.
"""

from importlib.metadata import PackageNotFoundError, version

from tpsreg.ransac import ransac_filter
from tpsreg.tps import ThinPlateSplineTransform
from tpsreg.warping import (
    transform_coords,
    transform_image,
    transform_image_stack,
)

try:
    __version__ = version("tpsreg")
except PackageNotFoundError:  # pragma: no cover - only hit in a source tree
    __version__ = "0.0.0.dev0"

__all__ = [
    "ThinPlateSplineTransform",
    "__version__",
    "ransac_filter",
    "transform_coords",
    "transform_image",
    "transform_image_stack",
]
