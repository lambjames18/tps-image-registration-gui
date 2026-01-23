"""
models.py - Data Models and Business Logic for Distortion Correction

This module contains the core business logic separated from the UI.
"""

import json
import logging
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Self, Tuple, Optional, Any, Union

import h5py
import numpy as np
from scipy import ndimage as ndi
from skimage import io, transform

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize as RESIZE
from kornia.enhance import equalize_clahe

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logger = logging.getLogger(__name__)


class DataFormat(Enum):
    """Enumeration of supported data formats."""

    ANG = "ang"
    H5 = "h5"
    DREAM3D = "dream3d"
    IMAGE = "image"
    RAW_IMAGE = "raw_image"


class TransformType(Enum):
    """Enumeration of available transformation types."""

    TPS = "tps"
    TPS_AFFINE = "tps_affine"


@dataclass
class Point:
    """Represents a control point."""

    x: float
    y: float
    slice_idx: int = 0

    def to_array(self) -> np.ndarray:
        """Convert to numpy array [x, y] or [slice, x, y]."""
        if self.slice_idx == 0:
            return np.array([self.x, self.y])
        return np.array([self.slice_idx, self.x, self.y])


@dataclass
class PointSet:
    """Collection of control points for an image."""

    points: Dict[int, List[Point]] = field(default_factory=dict)

    def add_point(self, point: Point) -> None:
        """Add a point to the set."""
        if point.slice_idx not in self.points:
            self.points[point.slice_idx] = []
        self.points[point.slice_idx].append(point)

    def remove_point(self, slice_idx: int, point_idx: int) -> bool:
        """Remove a point by index."""
        try:
            if slice_idx in self.points and 0 <= point_idx < len(
                self.points[slice_idx]
            ):
                self.points[slice_idx].pop(point_idx)
                logger.debug("Point index in range for removal")
            else:
                logger.debug("Point index out of range for removal")
            return True
        except Exception as e:
            logger.error("Failed to remove point from PointSet.")
            return False

    def get_points_array(self, slice_idx: Optional[int] = None) -> np.ndarray:
        """Get points as numpy array for a specific slice or all slices."""
        if slice_idx is not None:
            if slice_idx not in self.points:
                return np.array([])
            return np.array([[p.x, p.y] for p in self.points[slice_idx]])

        # Get all points with slice information
        all_points = []
        for slice_idx, slice_points in self.points.items():
            for point in slice_points:
                all_points.append([slice_idx, point.x, point.y])
        return np.array(all_points) if all_points else np.array([])

    def clear(self, slice_idx: Optional[int] = None) -> None:
        """Clear points for a specific slice or all slices."""
        if slice_idx is not None:
            self.points.pop(slice_idx, None)
        else:
            self.points.clear()

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            str(slice_idx): [[p.x, p.y] for p in points]
            for slice_idx, points in self.points.items()
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "PointSet":
        """Create PointSet from dictionary."""
        point_set = cls()
        for slice_idx, points in data.items():
            for x, y in points:
                point_set.add_point(Point(x, y, int(slice_idx)))
        return point_set


@dataclass
class ImageData:
    """Container for image data and metadata."""

    data: Dict[str, np.ndarray]
    resolution: float
    paths: Dict[str, Path]  # Maps modality name to file path
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get the shape of the first data array."""
        if self.data:
            first_key = next(iter(self.data.keys()))
            return self.data[first_key].shape
        return (0, 0, 0)

    @property
    def modalities(self) -> List[str]:
        """Get list of available modalities."""
        return list(self.data.keys())

    @property
    def path(self) -> Path:
        """Get the path of the first modality (for backward compatibility)."""
        if self.paths:
            return next(iter(self.paths.values()))
        return Path(".")

    def get_slice(self, modality: str, slice_idx: int = 0) -> np.ndarray:
        """Get a specific slice of data."""
        if modality not in self.data:
            raise KeyError(f"Modality '{modality}' not found in image data")

        data = self.data[modality]
        if slice_idx >= data.shape[0]:
            raise IndexError(
                f"Slice index {slice_idx} out of range for shape {data.shape}"
            )

        return data[slice_idx]

    def add_modality(self, new_image_data: Self) -> None:
        """Add a new modality to the image data."""
        modality_names = list(new_image_data.data.keys())

        # Make sure only one modality is being added and formats match
        if len(modality_names) != 1:
            raise ValueError("New image data must contain exactly one modality")
        # Check data format compatibility
        elif new_image_data.metadata.get("dataformat") != self.metadata.get(
            "dataformat"
        ):
            raise ValueError(
                "New image data format does not match existing image data format"
            )
        # Check shape compatibility
        elif new_image_data.shape != self.shape:
            raise ValueError(
                f"New image data shape {new_image_data.shape} does not match existing shape {self.shape}"
            )

        # Get the relevant data
        modality_name = modality_names[0]
        data = new_image_data.data[modality_name]
        path = new_image_data.paths[
            modality_name
        ]  # Might be multiple if it is a stack of images for 3D

        self.data[modality_name] = data
        self.paths[modality_name] = path
        logger.info(f"Added modality '{modality_name}' from {path} to image data")


class PointManager:
    """Manages control points for image registration."""

    def __init__(self):
        self.source_points = PointSet()
        self.destination_points = PointSet()
        self._history: List[Tuple[str, Any]] = []
        self._history_index = -1
        self.max_history = 50
        logger.info("PointManager initialized")

    def add_point_pair(self, src_point: Point, dst_point: Point) -> None:
        """Add a pair of corresponding points."""
        self._save_state()
        self.source_points.add_point(src_point)
        self.destination_points.add_point(dst_point)
        logger.debug(f"Added point pair: src={src_point}, dst={dst_point}")

    def remove_point_pair(self, slice_idx: int, point_idx: int) -> bool:
        """Remove a pair of corresponding points."""
        self._save_state()
        src_removed = self.source_points.remove_point(slice_idx, point_idx)
        dst_removed = self.destination_points.remove_point(slice_idx, point_idx)
        logger.debug(
            f"Attempted to remove point pair at slice {slice_idx}, index {point_idx} with success {src_removed} and {dst_removed}"
        )
        success = src_removed and dst_removed

        if success:
            logger.debug(f"Removed point pair at slice {slice_idx}, index {point_idx}")
        else:
            logger.warning(
                f"Failed to remove point pair at slice {slice_idx}, index {point_idx}"
            )

        return success

    def get_point_pairs(
        self, slice_idx: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get corresponding point pairs as numpy arrays."""
        src_points = self.source_points.get_points_array(slice_idx)
        dst_points = self.destination_points.get_points_array(slice_idx)
        return src_points, dst_points

    def clear_points(self, slice_idx: Optional[int] = None) -> None:
        """Clear all points or points for a specific slice."""
        self._save_state()
        self.source_points.clear(slice_idx)
        self.destination_points.clear(slice_idx)
        logger.info(
            f"Cleared points for slice {slice_idx if slice_idx is not None else 'all'}"
        )

    def save_to_file(self, src_path: Path, dst_path: Path) -> None:
        """Save points to text files."""
        try:
            src_array = self.source_points.get_points_array()
            dst_array = self.destination_points.get_points_array()

            if src_array.size > 0:
                np.savetxt(src_path, src_array, fmt="%d", delimiter=" ")
                logger.info(f"Saved {len(src_array)} source points to {src_path}")

            if dst_array.size > 0:
                np.savetxt(dst_path, dst_array, fmt="%d", delimiter=" ")
                logger.info(f"Saved {len(dst_array)} destination points to {dst_path}")

        except Exception as e:
            logger.error(f"Failed to save points: {e}")
            raise

    def load_source_from_file(
        self, src_path: Path, current_slice: int = 0, is_2d: bool = True
    ) -> None:
        """Load source points from text file."""
        try:
            self.source_points = self._load_points_from_file(
                src_path, current_slice, is_2d
            )
            logger.info(f"Loaded source points from {src_path}")
        except Exception as e:
            logger.error(f"Failed to load source points: {e}")
            raise

    def load_destination_from_file(
        self, dst_path: Path, current_slice: int = 0, is_2d: bool = True
    ) -> None:
        """Load destination points from text file."""
        try:
            self.destination_points = self._load_points_from_file(
                dst_path, current_slice, is_2d
            )
            logger.info(f"Loaded destination points from {dst_path}")
        except Exception as e:
            logger.error(f"Failed to load destination points: {e}")
            raise

    def load_from_json(self, json_data: dict) -> None:
        """Load points from JSON data."""
        try:
            self.source_points = self._load_points_from_json(
                json_data.get("source_points", {})
            )
            self.destination_points = self._load_points_from_json(
                json_data.get("destination_points", {})
            )
            logger.info("Loaded points from JSON data")
        except Exception as e:
            logger.error(f"Failed to load points from JSON: {e}")
            raise

    def _load_points_from_file(
        self, path: Path, current_slice: int = 0, is_2d: bool = True
    ) -> PointSet:
        """Load points from a single file.

        Args:
            path: Path to the point file
            current_slice: The current active slice index
            is_2d: Whether the data is 2D (single slice) or 3D (multiple slices)

        Returns:
            PointSet with loaded points

        Point file formats:
            - 2 columns (x, y): Points are added to current_slice
            - 3 columns (slice_idx, x, y):
                - For 3D data: Points are added to their respective slices
                - For 2D data: slice_idx is ignored, all points go to slice 0
        """
        point_set = PointSet()

        if not path.exists():
            logger.warning(f"Point file does not exist: {path}")
            return point_set

        try:
            data = np.loadtxt(path, dtype=int)
            if data.size == 0:
                return point_set

            if data.ndim == 1:
                data = data.reshape(1, -1)

            # Handle 2D or 3D points
            if data.shape[1] == 2:
                # 2D points (x, y) - apply to current slice
                for x, y in data:
                    point_set.add_point(Point(int(x), int(y), current_slice))
                logger.info(f"Loaded {len(data)} 2D points to slice {current_slice}")

            elif data.shape[1] == 3:
                # 3D points (slice_idx, x, y)
                if is_2d:
                    # Ignore slice index for 2D data, add all to slice 0
                    for slice_idx, x, y in data:
                        point_set.add_point(Point(int(x), int(y), 0))
                    logger.info(
                        f"Loaded {len(data)} points to slice 0 (2D data, slice indices ignored)"
                    )
                else:
                    # Use slice indices for 3D data
                    for slice_idx, x, y in data:
                        point_set.add_point(Point(int(x), int(y), int(slice_idx)))
                    logger.info(
                        f"Loaded {len(data)} 3D points to their respective slices"
                    )
            else:
                raise ValueError(
                    f"Invalid point format in {path}: expected 2 or 3 columns, got {data.shape[1]}"
                )

        except Exception as e:
            logger.error(f"Error reading point file {path}: {e}")
            raise

        return point_set

    def _load_points_from_json(self, points: dict) -> PointSet:
        """Load points from a numpy array."""
        point_set = PointSet()

        try:
            for slice_idx, pts in points.items():
                for x, y in pts:
                    point_set.add_point(Point(x, y, int(slice_idx)))

        except Exception as e:
            logger.error(f"Error loading points from json: {e}")
            raise

        return point_set

    def _save_state(self) -> None:
        """Save current state for undo functionality."""
        state = {
            "source": self.source_points.to_dict(),
            "destination": self.destination_points.to_dict(),
        }

        # Remove any states after current index
        self._history = self._history[: self._history_index + 1]

        # Add new state
        self._history.append(("state", state))

        # Limit history size
        if len(self._history) > self.max_history:
            self._history.pop(0)
        else:
            self._history_index += 1

    def undo(self) -> bool:
        """Undo last operation."""
        if self._history_index > 0:
            self._history_index -= 1
            state = self._history[self._history_index][1]
            self.source_points = PointSet.from_dict(state["source"])
            self.destination_points = PointSet.from_dict(state["destination"])
            logger.debug("Undid last operation")
            return True
        return False

    def redo(self) -> bool:
        """Redo last undone operation."""
        if self._history_index < len(self._history) - 1:
            self._history_index += 1
            state = self._history[self._history_index][1]
            self.source_points = PointSet.from_dict(state["source"])
            self.destination_points = PointSet.from_dict(state["destination"])
            logger.debug("Redid operation")
            return True
        return False


class PointAutoIdentifier:
    """Automatically identifies corresponding control points between images."""

    ENGINES = {
        "sift": "detect_points_sift",
        "matchanything": "detect_points_matchanything",
    }
    checkpoint_path = None
    _matchanything_matcher = None  # Cache ROMA matcher for reuse

    def __init__(self):
        logger.info("PointAutoIdentifier initialized")

    @classmethod
    def detect_points(
        cls,
        source_image: np.ndarray,
        destination_image: np.ndarray,
        method: str = "sift",
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Identify corresponding points between source and destination images.

        Args:
            source_image: Source image as numpy array
            destination_image: Destination image as numpy array
            method: Method to use for point identification ('sift', 'matchanything', etc.)
            **kwargs: Additional parameters for the identification method

        Returns:
            Tuple of numpy arrays: (source_points, destination_points)
        """
        loader_method = getattr(cls, cls.ENGINES[method], None)
        if loader_method is None:
            raise ValueError(f"Unsupported point identification method: {method}")
        logger.info(f"Identifying points using method '{method}'")
        if method == "matchanything":
            return loader_method(
                source_image,
                destination_image,
                checkpoint_path=kwargs.get("checkpoint_path", cls.checkpoint_path),
                **kwargs,
            )
        return loader_method(source_image, destination_image, **kwargs)

    @staticmethod
    def detect_points_sift(
        source_image: np.ndarray, destination_image: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detect matching points between images using SIFT keypoint detector.

        Args:
            source_image: Source image as numpy array
            destination_image: Destination image as numpy array
            **kwargs: Additional parameters:
                - max_ratio: Maximum ratio for Lowe's ratio test (default: 0.75)
                - min_matches: Minimum number of matches required (default: 4)

        Returns:
            Tuple of numpy arrays: (source_points, destination_points)
        """
        from skimage.feature import SIFT, match_descriptors
        from skimage.transform import AffineTransform
        from skimage.filters import gaussian
        from ransac import ransac_filter as ransac

        # Get parameters
        max_ratio = kwargs.get("max_ratio", 0.75)
        min_matches = kwargs.get("min_matches", 4)
        sigma = kwargs.get("sigma", 0.5)
        ransac_threshold = kwargs.get("ransac_threshold", 5.5)
        ransac_max_trials = kwargs.get("ransac_max_trials", 1000)
        ransac_method = kwargs.get("ransac_method", "deformable")

        # Ensure images are grayscale and uint8
        if source_image.ndim > 2:
            source_gray = source_image[:, :, 0]
        else:
            source_gray = source_image

        if destination_image.ndim > 2:
            destination_gray = destination_image[:, :, 0]
        else:
            destination_gray = destination_image

        # Convert to float32
        source_gray = source_gray.astype(np.float32)
        destination_gray = destination_gray.astype(np.float32)

        # Perform a slight blur to reduce noise
        source_gray = gaussian(source_gray, sigma=sigma)
        destination_gray = gaussian(destination_gray, sigma=sigma)

        # Normalize to 0-1 range for SIFT
        source_gray = (source_gray - source_gray.min()) / (
            source_gray.max() - source_gray.min()
        )
        destination_gray = (destination_gray - destination_gray.min()) / (
            destination_gray.max() - destination_gray.min()
        )

        try:
            # Detect SIFT keypoints and descriptors
            sift_detector = SIFT()

            sift_detector.detect_and_extract(source_gray)
            keypoints_src = sift_detector.keypoints
            descriptors_src = sift_detector.descriptors

            sift_detector.detect_and_extract(destination_gray)
            keypoints_dst = sift_detector.keypoints
            descriptors_dst = sift_detector.descriptors

            logger.info(f"Detected {len(keypoints_src)} keypoints in source image")
            logger.info(f"Detected {len(keypoints_dst)} keypoints in destination image")

            if len(keypoints_src) < min_matches or len(keypoints_dst) < min_matches:
                logger.warning(
                    f"Insufficient keypoints detected (need at least {min_matches})"
                )
                return np.array([]), np.array([])

            # Match descriptors using Lowe's ratio test
            matches = match_descriptors(
                descriptors_src, descriptors_dst, max_ratio=max_ratio, cross_check=True
            )

            logger.info(f"Found {len(matches)} initial matches")

            if len(matches) < min_matches:
                logger.warning(
                    f"Insufficient matches found (need at least {min_matches})"
                )
                return np.array([]), np.array([])

            # Extract matched keypoint coordinates
            src_points = keypoints_src[matches[:, 0]][:, ::-1]  # (y, x) -> (x, y)
            dst_points = keypoints_dst[matches[:, 1]][:, ::-1]  # (y, x) -> (x, y)

            # Use RANSAC to filter outliers
            try:
                inliers = ransac(
                    src_points,
                    dst_points,
                    threshold=ransac_threshold,
                    max_trials=ransac_max_trials,
                    method=ransac_method,
                )

                src_points = src_points[inliers]
                dst_points = dst_points[inliers]

                logger.info(f"After RANSAC filtering: {len(src_points)} inlier matches")

            except Exception as e:
                logger.warning(f"RANSAC filtering failed: {e}, using all matches")

            if len(src_points) < min_matches:
                logger.warning(
                    f"Insufficient inlier matches after RANSAC (need at least {min_matches})"
                )
                return np.array([]), np.array([])

            # Only return num_samples best matches based on fractional distance
            num_samples = kwargs.get("num_samples", 10)
            if len(src_points) > num_samples:
                # Compute distances between matched points
                src_pts_norm = src_points / np.array(
                    [source_image.shape[1], source_image.shape[0]]
                )
                dst_pts_norm = dst_points / np.array(
                    [destination_image.shape[1], destination_image.shape[0]]
                )
                distances = np.linalg.norm(src_pts_norm - dst_pts_norm, axis=1)
                sorted_indices = np.argsort(distances)
                top_indices = sorted_indices[:num_samples]
                src_points = src_points[top_indices]
                dst_points = dst_points[top_indices]
                logger.info(f"Selected top {num_samples} matches based on distance")

            return src_points.astype(int), dst_points.astype(int)

        except Exception as e:
            logger.error(f"SIFT point detection failed: {e}")
            return np.array([]), np.array([])

    @staticmethod
    def detect_points_matchanything(
        source_image: np.ndarray,
        destination_image: np.ndarray,
        checkpoint_path: str = None,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Detect matching points between images using ROMA (MatchAnything).

        Args:
            source_image: Source image as numpy array (H, W) or (H, W, C)
            destination_image: Destination image as numpy array (H, W) or (H, W, C)
            **kwargs: Additional parameters:
                - confidence_threshold: Minimum confidence (default: 0.1)
                - ransac_filter: Whether to apply RANSAC (default: True)
                - ransac_threshold: RANSAC threshold in pixels (default: 5.5)
                - num_samples: Maximum number of matches to sample (default: 5000)
                - checkpoint_path: Path to model checkpoint
                - device: Device to use ('cuda' or 'cpu')
                - resize_by_stretch: Whether to resize by stretching (default: True)
                - coarse_resolution: Resolution for coarse matching (default: (560, 560))
                - upsample_resolution: Resolution for upsampling (default: (864, 864))

        Returns:
            Tuple of (source_points, destination_points) as numpy arrays (Nx2)
        """
        try:
            from roma_matcher import create_matcher, apply_matcher

            # If matcher is not cached, create it
            if PointAutoIdentifier._matchanything_matcher is None:
                PointAutoIdentifier._matchanything_matcher = create_matcher(
                    checkpoint_path=checkpoint_path
                )
            matcher = PointAutoIdentifier._matchanything_matcher

            src_points, dst_points, confidences = apply_matcher(
                matcher,
                source_image,
                destination_image,
                ransac_filter=kwargs.get("ransac_filter", True),
                ransac_threshold=kwargs.get("ransac_threshold", 0.05),
                ransac_method=kwargs.get("ransac_method", "deformable"),
                ransac_max_trials=kwargs.get("ransac_max_trials", 100),
            )

            logger.info(f"ROMA detected {len(src_points)} matches")

            if len(src_points) < 4:
                logger.warning(
                    "Insufficient matches found (< 4). Cannot estimate transformation."
                )
                return np.array([]), np.array([])

            selection_method = "grid"  # grid, random, naive

            # Take the top N matches based on confidence
            num_samples = kwargs.get("num_samples", 10)
            if len(confidences) > num_samples:
                # First define a minimum confidence threshold
                sorted_indices = np.argsort(-confidences)
                min_confidence = confidences[sorted_indices[num_samples - 1]]
                # Filter matches by confidence
                mask = confidences >= min_confidence
                src_points = src_points[mask]
                dst_points = dst_points[mask]
                confidences = confidences[mask]

                if len(src_points) > num_samples:

                    # Select a subset equal to the num_samples that tries to maximize the spread
                    if selection_method == "grid":
                        grid_size = int(np.sqrt(num_samples))
                        row_bins = np.linspace(0, source_image.shape[0], grid_size + 1)
                        col_bins = np.linspace(0, source_image.shape[1], grid_size + 1)
                        selected_indices = []
                        deficit = 0
                        for i in range(grid_size):
                            for j in range(grid_size):
                                cell_mask = (
                                    (src_points[:, 1] >= row_bins[i])
                                    & (src_points[:, 1] < row_bins[i + 1])
                                    & (src_points[:, 0] >= col_bins[j])
                                    & (src_points[:, 0] < col_bins[j + 1])
                                )
                                cell_indices = np.where(cell_mask)[0]
                                if len(cell_indices) > 0:
                                    chosen_idx = np.random.choice(
                                        cell_indices, size=1 + deficit
                                    )
                                    selected_indices.extend(chosen_idx)
                                    deficit = 0
                                else:
                                    deficit += 1
                                if len(selected_indices) >= num_samples:
                                    break
                            if len(selected_indices) >= num_samples:
                                break
                        selected_indices = np.array(selected_indices)[:num_samples]
                        src_points = src_points[selected_indices]
                        dst_points = dst_points[selected_indices]
                        confidences = confidences[selected_indices]

                    # Select random matches
                    elif selection_method == "random":
                        selected_indices = np.random.choice(
                            len(src_points), size=num_samples, replace=False
                        )
                        src_points = src_points[selected_indices]
                        dst_points = dst_points[selected_indices]
                        confidences = confidences[selected_indices]

                    # Just take the top N matches
                    elif selection_method == "naive":
                        src_points = src_points[:num_samples]
                        dst_points = dst_points[:num_samples]
                        confidences = confidences[:num_samples]

                logger.info(
                    f"Selected top {num_samples} matches (min confidence: {confidences[-1]:.4f})"
                )

            return src_points.astype(int), dst_points.astype(int)

        except ImportError as e:
            logger.error(
                f"Failed to import ROMA matcher. Make sure roma_matcher.py is available: {e}"
            )
            return np.array([]), np.array([])
        except FileNotFoundError as e:
            logger.error(f"ROMA checkpoint not found: {e}")
            logger.error(
                "Please download weights from: https://drive.google.com/file/d/12L3g9-w8rR9K2L4rYaGaDJ7NqX1D713d/view"
            )
            return np.array([]), np.array([])
        except Exception as e:
            logger.error(f"Matchanything point detection failed: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return np.array([]), np.array([])

    @classmethod
    def set_checkpoint_path(cls, path: str) -> None:
        """Set the checkpoint path for MatchAnything."""
        cls.checkpoint_path = path
        # Clear cached matcher to reload with new checkpoint
        cls._matchanything_matcher = None
        logger.info(f"Set MatchAnything checkpoint path to: {path}")


class ImageLoader:
    """Handles loading and preprocessing of various image formats."""

    SUPPORTED_FORMATS = {
        ".ang": "load_ang",
        ".h5": "load_h5",
        ".dream3d": "load_dream3d",
        ".tif": "load_image",
        ".tiff": "load_image",
        ".png": "load_image",
        ".jpg": "load_image",
        ".jpeg": "load_image",
    }

    @classmethod
    def load(
        cls,
        path: Union[str, Path, List, Tuple],
        resolution: float = 1.0,
        modality_name: str = None,
    ) -> ImageData:
        """Load image data from file."""
        is_list = type(path) in [list, tuple]
        if is_list and len(path) == 0:
            raise ValueError("Provided empty list of image paths")
        elif is_list and len(path) == 1:
            is_list = False
            path = path[0]

        if not is_list:
            path = Path(path).resolve()

            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")

            suffix = path.suffix.lower()
            if suffix not in cls.SUPPORTED_FORMATS:
                raise ValueError(f"Unsupported file format: {suffix}")

            loader_method = getattr(cls, cls.SUPPORTED_FORMATS[suffix])

        else:
            loader_method = cls.load_images
            first_path = Path(path[0]).resolve()
            first_suffix = first_path.suffix.lower()
            for i in range(len(path)):
                _p = Path(path[i]).resolve()

                if not _p.exists():
                    raise FileNotFoundError(f"File not found: {path}")

                suffix = _p.suffix.lower()
                if suffix != first_suffix:
                    raise ValueError(
                        f"When prividing a list of images, all images must have the same extension"
                    )

                path[i] = _p

        try:
            logger.info(f"Loading {suffix} file(s)")

            # Call the appropriate loader method
            data, res, metadata = loader_method(path, modality_name)

            # Use provided resolution if loader didn't return one
            if res is None:
                res = resolution

            if metadata is None:
                metadata = {}

            # Create paths dictionary
            # Paths dictionary follow the data structure:
            # Mode
            #   -> Slice 0 Path
            #   -> Slice 1 Path
            #   -> ...

            paths = {}

            # Default modality name
            if modality_name is None:
                modality_name = "data"

            # Convert path to list for iteration
            if not is_list:
                path = [path]

            paths[modality_name] = path

            return ImageData(data=data, resolution=res, paths=paths, metadata=metadata)

        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            raise

    @staticmethod
    def load_ang(
        path: Path, modality_name: str = None
    ) -> Tuple[Dict[str, np.ndarray], float, Dict[str, Any]]:
        """Load ANG file format."""
        col_names = None
        header = ""
        header_lines = 0
        ncols = nrows = 0
        res = 1.0

        with open(path, "r") as f:
            for line in f:
                header += line
                if "NCOLS_ODD" in line:
                    ncols = int(line.split(": ")[1].strip())
                elif "NROWS" in line:
                    nrows = int(line.split(": ")[1].strip())
                elif "COLUMN_HEADERS" in line:
                    col_names = (
                        line.replace("\n", "").split(": ")[1].strip().split(", ")
                    )
                elif "XSTEP" in line:
                    res = float(line.split(": ")[1].strip())
                elif "HEADER: End" in line:
                    break
                header_lines += 1

        if col_names is None:
            col_names = ["phi1", "PHI", "phi2", "x", "y", "IQ", "CI", "Phase index"]

        raw_data = np.genfromtxt(path, skip_header=header_lines, dtype=float)
        raw_data[np.isnan(raw_data)] = 0.0

        if raw_data.shape[0] != ncols * nrows:
            raise ValueError(
                f"Data size mismatch: expected {ncols*nrows}, got {raw_data.shape[0]}"
            )

        n_entries = raw_data.shape[-1]
        data = raw_data.reshape((nrows, ncols, n_entries))

        out = {}
        for i, name in enumerate(col_names[:n_entries]):
            arr = data[:, :, i]
            arr = np.fliplr(np.rot90(arr, k=3)).T
            out[name] = arr.reshape((1,) + arr.shape + (1,))

        # Create Euler angles array
        if all(k in out for k in ["phi1", "PHI", "phi2"]):
            out["EulerAngles"] = np.stack(
                [out["phi1"], out["PHI"], out["phi2"]], axis=-1
            )

        metadata = {"header": header, "dataformat": DataFormat.ANG.value}
        return out, res, metadata

    @staticmethod
    def load_h5(
        path: Path, modality_name: str = None
    ) -> Tuple[Dict[str, np.ndarray], float, Dict[str, Any]]:
        """Load H5 file format."""
        with h5py.File(path, "r") as h5:
            # Find the EBSD data entry
            if "1" in h5.keys():
                entry = "1"
            elif "Scan 1" in h5.keys():
                entry = "Scan 1"
            else:
                raise ValueError("Could not find EBSD data in the H5 file")

            ebsd_data = h5[f"{entry}/EBSD/Data"]
            nrows = h5[f"{entry}/EBSD/Header/nRows"][0]
            ncols = h5[f"{entry}/EBSD/Header/nColumns"][0]
            res = h5[f"{entry}/EBSD/Header/Step X"][0]

            keys = list(ebsd_data.keys())
            data = {
                key.upper()
                .replace(" ", "-"): ebsd_data[key][...]
                .reshape(1, nrows, ncols, -1)
                for key in keys
            }

            # Create Euler angles array if components exist
            if all(k in data for k in ["PHI1", "PHI", "PHI2"]):
                data["EULERANGLES"] = np.stack(
                    [data["PHI1"], data["PHI"], data["PHI2"]], axis=-1
                ).astype(float)

        metadata = {"dataformat": DataFormat.H5.value}
        return data, res, metadata

    @staticmethod
    def load_dream3d(
        path: Path, modality_name: str = None
    ) -> Tuple[Dict[str, np.ndarray], float, Dict[str, Any]]:
        """Load DREAM3D file format."""
        with h5py.File(path, "r") as h5:
            if "DataStructure" in h5.keys():
                ebsd_data = h5["DataStructure/DataContainer/CellData"]
                res = np.asarray(
                    h5["DataStructure/DataContainer"].attrs.get("_SPACING")
                )[1]
            elif "DataContainers" in h5.keys():
                ebsd_data = h5["DataContainers/ImageDataContainer/CellData"]
                res = h5["DataContainers/ImageDataContainer/_SIMPL_GEOMETRY/SPACING"][
                    ...
                ][1]
            else:
                raise ValueError("Could not find EBSD data in the DREAM3D file")

            ebsd_keys = list(ebsd_data.keys())
            data = {key: ebsd_data[key][...] for key in ebsd_keys}

        metadata = {"dataformat": DataFormat.DREAM3D.value}
        return data, res, metadata

    @staticmethod
    def load_image(
        path: Path, modality_name: str = "Intensity"
    ) -> Tuple[Dict[str, np.ndarray], None]:
        """Load standard image formats with optional modality name."""
        im = io.imread(path, as_gray=True).astype(np.float32)

        # Normalize to 0-255 range
        im = np.around((im - np.min(im)) / (np.max(im) - np.min(im)) * 255, 0)
        im = im.astype(np.uint8)

        if im.ndim == 2:
            im = im.reshape((im.shape[0], im.shape[1], 1))

        im = im.reshape((1,) + im.shape)

        metadata = {"dataformat": DataFormat.IMAGE.value}
        return {modality_name: im}, None, metadata

    @staticmethod
    def load_images(
        paths: list, modality_name: str = "Intensity"
    ) -> Tuple[Dict[str, np.ndarray], None, Dict[str, Any]]:
        """Load a list of standard image formats."""
        images = np.array(
            [
                io.imread(paths[i], as_gray=True).astype(np.float32)
                for i in range(len(paths))
            ]
        )

        # Put in a channel axis if needed
        if images.ndim == 3:
            images = images.reshape(images.shape + (1,))

        # Normalize to 0-255 range
        # mn = np.min(images, axis=(1, 2, 3), keepdims=True)
        # mx = np.max(images, axis=(1, 2, 3), keepdims=True)
        # rnge = mx - mn
        # images = np.around(255 * (images - mn) / rnge, 0)
        # images = images.astype(np.uint8)

        metadata = {"dataformat": DataFormat.IMAGE.value}
        return {modality_name: images}, None, metadata


class ImageWriter:
    """Handles saving images to disk."""

    SUPPORTED_FORMATS = {
        ".ang": "save_ang",
        ".h5": "save_h5",
        ".dream3d": "save_dream3d",
        ".tif": "save_image",
        ".tiff": "save_image",
        ".png": "save_image",
        ".jpg": "save_image",
        ".jpeg": "save_image",
    }

    @classmethod
    def save(
        cls,
        image_data: Union[np.ndarray, Dict[str, np.ndarray]],
        path: Path,
        *args,
        **kwargs,
    ) -> None:
        """Save image data to file."""
        try:
            # Determine file format from path
            ext = path.suffix.lower()
            if ext in cls.SUPPORTED_FORMATS:
                save_method = cls.SUPPORTED_FORMATS[ext]
                getattr(cls, save_method)(image_data, path, *args, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {ext}")

        except Exception as e:
            logger.error(f"Failed to save image to {path}: {e}")
            raise

    @staticmethod
    def save_image(image_data: np.ndarray, path: Path) -> None:
        """Save image to file."""
        try:
            io.imsave(path, image_data)
            logger.info(f"Saved image to {path}")
        except Exception as e:
            logger.error(f"Failed to save image to {path}: {e}")
            raise

    @staticmethod
    def save_dream3d(
        image_data: Dict[str, np.ndarray],
        path: Union[str, Path],
        original_path: Union[str, Path],
    ) -> None:
        """Save image data to DREAM3D format."""
        # Ensure paths are Path objects
        if not isinstance(path, Path):
            path = Path(path)
        if not isinstance(original_path, Path):
            original_path = Path(original_path)

        # Get path to original xdmf file associated with the DREAM3D file
        original_xdmf_path = original_path.with_suffix(".xdmf")
        xdmf_path = path.with_suffix(".xdmf")

        # First verify the original file exists
        if not original_path.exists():
            raise FileNotFoundError(f"Original DREAM3D file not found: {original_path}")
        if not original_xdmf_path.exists():
            raise FileNotFoundError(
                f"Original XDMF file not found: {original_xdmf_path}"
            )

        # Copy the original to the new location
        import shutil

        shutil.copyfile(original_path, path)
        shutil.copyfile(original_xdmf_path, xdmf_path)

        # Replace the Dream3D file name in the XDMF file
        def replace_dream3d_filename_in_xdmf(xdmf_path, new_dream3d_filename):
            """Replaces the DREAM3D filename in the XDMF file with a new filename."""
            with open(xdmf_path, "r") as file:
                xdmf_content = file.readlines()

            with open(xdmf_path, "w") as file:
                for line in xdmf_content:
                    if ".dream3d:/" in line:
                        idx = line.index(":")
                        line = " " * 8 + new_dream3d_filename + line[idx:]
                    file.write(line)

        replace_dream3d_filename_in_xdmf(xdmf_path, path.name)

        # Function for determining the cell data path
        first_modality = next(iter(image_data.keys()))

        def recursive_search(name, obj):
            if isinstance(obj, h5py.Dataset):
                if name.endswith(first_modality):
                    return name
                else:
                    return None
            for key, item in obj.items():
                result = recursive_search(f"{name}/{key}", item)
                if result:
                    return result
            return None

        # Define Dream3d data types
        dream3d_dtypes = {
            np.uint8: "DataArray<uint8_t> ",
            np.int8: "DataArray<int8_t> ",
            np.uint16: "DataArray<uint16_t> ",
            np.int16: "DataArray<int16_t> ",
            np.uint32: "DataArray<uint32_t> ",
            np.int32: "DataArray<int32_t> ",
            np.uint64: "DataArray<uint64_t> ",
            np.int64: "DataArray<int64_t> ",
            np.float32: "DataArray<float> ",
            np.float64: "DataArray<double> ",
            bool: "DataArray<bool> ",
        }
        xdmf_dtype_formats = {  # (NumberType, Precision)
            np.uint8: ("UChar", "1"),
            np.int8: ("Char", "1"),
            np.uint16: ("UInt", "2"),
            np.int16: ("Int", "2"),
            np.uint32: ("UInt", "4"),
            np.int32: ("Int", "4"),
            np.uint64: ("UInt", "8"),
            np.int64: ("Int", "8"),
            np.float32: ("Float", "4"),
            np.float64: ("Float", "8"),
            bool: ("uchar", "1"),
        }

        # Function to create a new dataset
        def add_dataset_to_h5(h5group, name, data):
            dtype = data.dtype.type
            if dtype not in dream3d_dtypes:
                raise ValueError(f"Unsupported data type for DREAM3D: {dtype}")
            dset = h5group.create_dataset(name, data=data, dtype=dtype)
            dset.attrs["ComponentDimensions"] = np.uint64([data.shape[-1]])
            dset.attrs["Tuple Axis Dimensions"] = np.bytes_(
                f"x={str(data.shape[2])},y={str(data.shape[1])},z={str(data.shape[0])} "
            )
            dset.attrs["DataArrayVersion"] = np.int32([2])
            dset.attrs["ObjectType"] = np.bytes_(dream3d_dtypes[dtype])
            dset.attrs["TupleDimensions"] = np.uint64(np.squeeze(data.shape[:-1][::-1]))

            return dset

        def add_dataset_to_xdmf(xdmf_path, dataset_name, data_array):
            """Adds a new dataset to an existing XDMF file."""
            # Read the existing XDMF file
            with open(xdmf_path, "r") as file:
                xdmf_content = file.readlines()

            # Break the xdmf content into lines for easier manipulation
            xdmf_content = [line.replace("\n", "") for line in xdmf_content]

            # Make sure the shape of the data_array is compatible
            if data_array.ndim == 3:
                data_array = data_array.reshape(data_array.shape + (1,))
            elif data_array.ndim < 3:
                raise ValueError("data_array must be at least 3-dimensional")
            elif data_array.ndim > 4:
                raise ValueError("data_array must be at most 4-dimensional")
            dimensions = (
                xdmf_content[["<Topology" in line for line in xdmf_content].index(True)]
                .split("Dimensions=")[1]
                .split('"')[1]
                .strip()
            )
            data_array_dims = " ".join(map(str, np.array(data_array.shape[0:3]) + 1))
            if dimensions != data_array_dims:
                raise ValueError(
                    "data_array dimensions are not compatible with XDMF Topology dimensions"
                )

            # Make sure an entry with the same name does not already exist
            for line in xdmf_content:
                if f'Attribute Name="{dataset_name}"' in line:
                    raise ValueError(
                        f"An Attribute with the name '{dataset_name}' already exists"
                    )

            # Determine the insertion point (put the new entry at the end of the Grid section)
            insertion_index = ["</Grid>" in line for line in xdmf_content].index(True)

            # Gather relevant data for the new dataset
            data_type, precision = xdmf_dtype_formats[data_array.dtype.type]
            dimensions = " ".join(map(str, data_array.shape))
            attribute_type = (
                "Scalar"
                if (data_array.ndim == 3)
                or ((data_array.ndim == 4) and (data_array.shape[-1] == 1))
                else "Vector"
            )
            file_path = (
                xdmf_content[
                    [".dream3d:/" in line for line in xdmf_content].index(True)
                ]
                .strip()
                .split("/")
            )
            file_path[-1] = dataset_name
            file_path = "/".join(file_path)

            # Create the new DataItem entry
            xdmf_content.insert(
                insertion_index,
                f'    <Attribute Name="{dataset_name}" AttributeType="{attribute_type}" Center="Cell">',
            )
            xdmf_content.insert(
                insertion_index + 1,
                f'      <DataItem Format="HDF" Dimensions="{dimensions}" NumberType="{data_type}" Precision="{precision}" >',
            )
            xdmf_content.insert(
                insertion_index + 2,
                f"        {file_path}",
            )
            xdmf_content.insert(insertion_index + 3, "      </DataItem>")
            xdmf_content.insert(insertion_index + 4, "    </Attribute>")

            # Write the modified content back to the XDMF file
            with open(xdmf_path, "w") as file:
                for line in xdmf_content:
                    file.write(line + "\n")

        # Now open the new file and write the new data
        with h5py.File(path, "r+") as h5:
            cell_data_path = recursive_search("", h5)
            if cell_data_path is None:
                raise ValueError(
                    f"Could not find cell data for modality '{first_modality}' in DREAM3D file"
                )
            cell_data_path = cell_data_path.replace(f"/{first_modality}", "")
            cell_data = h5[cell_data_path]

            for modality, data in image_data.items():
                if modality in cell_data:
                    dataset = cell_data[modality]
                    if dataset.shape != data.shape:
                        raise ValueError(
                            f"Data shape mismatch for modality '{modality}': expected {dataset.shape}, got {data.shape}"
                        )
                    dataset[...] = data
                else:
                    # Create new dataset
                    logger.info(f"Creating new dataset for modality '{modality}'")
                    # Add dataset to H5 and XDMF
                    add_dataset_to_h5(cell_data, modality, data)
                    add_dataset_to_xdmf(xdmf_path, modality, data)

    @staticmethod
    def save_ang(
        image_data: Dict[str, np.ndarray],
        path: Path,
        ang_header: str,
        resolution: float,
    ) -> None:
        """Save image data to ANG format."""
        # Create new x/y grid
        k = next(iter(image_data.keys()))
        x, y = np.meshgrid(
            np.arange(image_data[k].shape[1]), np.arange(image_data[k].shape[0])
        )
        x = x.astype(np.float32) * resolution
        y = y.astype(np.float32) * resolution
        image_data["x" if "x" in image_data else "X"] = x
        image_data["y" if "y" in image_data else "Y"] = y

        col_names = list(image_data.keys())
        data_out = []
        for i, key in enumerate(col_names):
            data_out.append(image_data[key].reshape(-1, 1).astype(np.float32))
        data_out = np.hstack(data_out)
        np.savetxt(
            path,
            data_out,
            header=ang_header[:-1],
            fmt="%.5f",
            delimiter=" ",
            comments="",
        )

    @staticmethod
    def save_h5(
        image_data: Dict[str, np.ndarray], path: Path, resolution: float
    ) -> None:
        """Save image data to HDF5 format."""
        raise NotImplementedError("Saving to HDF5 format is not yet implemented.")


class TransformManager:
    """Manages image transformations and warping operations."""

    def __init__(self):
        self.supported_transforms = list(TransformType)
        self._last_transform = None
        self._last_params = None
        logger.info("TransformManager initialized")

    def estimate_transform(
        self,
        src_points: np.ndarray,
        dst_points: np.ndarray,
        transform_type: TransformType,
        output_shape: Tuple[int, int],
    ) -> Any:
        """Estimate transformation parameters from point correspondences."""
        if src_points.size == 0 or dst_points.size == 0:
            raise ValueError("Cannot estimate transform with empty point sets")

        if src_points.shape[0] != dst_points.shape[0]:
            raise ValueError(
                f"Point count mismatch: {src_points.shape[0]} vs {dst_points.shape[0]}"
            )

        try:
            # Import TPS here to avoid circular dependency
            from tps import ThinPlateSplineTransform

            affine_only = transform_type == TransformType.TPS_AFFINE
            tform = ThinPlateSplineTransform(affine_only=affine_only)
            tform.estimate(
                src_points,
                dst_points,
                output_shape,
            )

            self._last_transform = tform
            logger.debug(f"Estimated {transform_type.value} transform")
            return tform

        except Exception as e:
            logger.error(f"Failed to estimate transform: {e}")
            raise

    def apply_transform(
        self,
        image: np.ndarray,
        tform: Any,
        output_shape: Optional[Tuple[int, int]] = None,
        order: int = 0,
    ) -> np.ndarray:
        """Apply transformation to an image."""
        try:
            if output_shape is None:
                output_shape = image.shape[:2]

            warped = transform.warp(
                image,
                tform,
                output_shape=output_shape,
                mode="constant",
                cval=0,
                order=order,
            )

            logger.debug(f"Applied transform to image of shape {image.shape}")
            return warped

        except Exception as e:
            logger.error(f"Failed to apply transform: {e}")
            raise

    def apply_transform_stack(
        self,
        image_stack: np.ndarray,
        src_points: np.ndarray,
        dst_points: np.ndarray,
        transform_type: TransformType,
        output_shape: Optional[Tuple[int, int]] = None,
        order: int = 0,
    ) -> np.ndarray:
        """Apply transformation to a stack of images with interpolation between slices."""
        if output_shape is None:
            output_shape = image_stack.shape[1:3]

        # Get unique slices with points
        slice_indices = np.unique(src_points[:, 0]).astype(int)

        # Estimate transforms for each slice with points
        transforms = {}
        for slice_idx in slice_indices:
            mask = src_points[:, 0] == slice_idx
            src_pts = src_points[mask, 1:]
            dst_pts = dst_points[mask, 1:]

            tform = self.estimate_transform(
                src_pts, dst_pts, transform_type, output_shape
            )
            transforms[slice_idx] = tform

        # Apply transforms with interpolation
        output_stack = np.zeros(
            (image_stack.shape[0], *output_shape, image_stack.shape[-1])
        )

        for i in range(image_stack.shape[0]):
            # Find nearest transform or interpolate
            if i in transforms:
                tform = transforms[i]
            else:
                # Use nearest neighbor for now (could implement interpolation)
                nearest_idx = min(transforms.keys(), key=lambda x: abs(x - i))
                tform = transforms[nearest_idx]

            output_stack[i] = self.apply_transform(
                image_stack[i], tform, output_shape, order
            )

        logger.info(f"Applied transform to stack of {image_stack.shape[0]} images")
        return output_stack

    def export_transform(self, tform: Any, path: Path, format: str = "npy") -> None:
        """Export transformation parameters to file."""
        try:
            if hasattr(tform, "params"):
                params = tform.params
            else:
                params = np.array(tform)

            if format == "npy":
                np.save(path, params)
            elif format in ["csv", "txt"]:
                delimiter = "," if format == "csv" else " "
                np.savetxt(path, params, delimiter=delimiter)
            else:
                raise ValueError(f"Unsupported export format: {format}")

            logger.info(f"Exported transform to {path}")

        except Exception as e:
            logger.error(f"Failed to export transform: {e}")
            raise


class ImageProcessor:
    """Handles image processing operations."""

    @staticmethod
    def apply_clahe(
        image: np.ndarray,
        clip_limit: float = 20.0,
        kernel_size: Tuple[int, int] = (8, 8),
    ) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        try:
            tensor = torch.tensor(image).float()
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0).unsqueeze(0)
            elif tensor.ndim == 3:
                tensor = tensor.permute(2, 0, 1).unsqueeze(0)
            elif tensor.ndim == 4:
                tensor = tensor.permute(0, 3, 1, 2).contiguous()

            tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
            tensor = equalize_clahe(tensor, clip_limit, kernel_size)
            tensor = torch.round(
                255 * (tensor - tensor.min()) / (tensor.max() - tensor.min())
            )
            result = tensor.detach().numpy().astype(np.uint8)
            if result.ndim == 3:
                result = np.transpose(result, (1, 2, 0))
            elif result.ndim == 4:
                result = np.transpose(result, (0, 2, 3, 1))
            result = result.reshape(image.shape)

            logger.debug(f"Applied CLAHE with clip_limit={clip_limit}")
            return result

        except Exception as e:
            logger.error(f"Failed to apply CLAHE: {e}")
            raise

    @staticmethod
    def resize_image(image: np.ndarray, scale: float) -> np.ndarray:
        """Resize image by scale factor."""
        if scale == 1.0:
            return image

        try:
            if image.ndim == 4:
                tensor = torch.tensor(np.transpose(image, (0, 3, 1, 2))).float()
            elif image.ndim == 3:
                tensor = (
                    torch.tensor(np.transpose(image, (2, 0, 1))).unsqueeze(0).float()
                )
            else:
                tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0).float()

            new_size = (int(tensor.shape[2] * scale), int(tensor.shape[3] * scale))

            resized = RESIZE(tensor, new_size, InterpolationMode.NEAREST)
            resized = resized.detach().numpy()

            if image.ndim == 4:  # (B, C, H, W) -> (B, H, W, C)
                resized = np.transpose(resized, (0, 2, 3, 1))
            elif image.ndim == 3:  # (1, C, H, W) -> (H, W, C)
                resized = np.transpose(resized[0], (1, 2, 0))
            else:  # (1, 1, H, W) -> (H, W)
                resized = resized[0, 0]

            logger.debug(f"Resized image by factor {scale}")
            return resized

        except Exception as e:
            logger.error(f"Failed to resize image: {e}")
            raise

    @staticmethod
    def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
        """Normalize image to uint8 range."""
        if image.dtype == np.uint8:
            return image

        try:
            with np.errstate(divide="ignore", invalid="ignore"):
                image = image.astype(np.float32)
                if image.ndim == 3:
                    normalized = np.around(
                        255
                        * (image - image.min(axis=(0, 1)))
                        / (image.max(axis=(0, 1)) - image.min(axis=(0, 1))),
                        0,
                    ).astype(np.uint8)
                else:
                    normalized = np.around(
                        255 * (image - image.min()) / (image.max() - image.min()), 0
                    ).astype(np.uint8)

            return normalized

        except Exception as e:
            logger.error(f"Failed to normalize image: {e}")
            raise


class ProjectManager:
    """Manages project state and serialization."""

    def __init__(self):
        self.project_path: Optional[Path] = None
        self.is_modified = False
        self.metadata = {"version": "2.0", "created": None, "modified": None}
        logger.info("ProjectManager initialized")

    def mark_modified(self) -> None:
        """Mark the project as modified."""
        self.is_modified = True

    def reset(self) -> None:
        """Reset project manager to initial state."""
        self.project_path = None
        self.is_modified = False
        self.metadata = {"version": "2.0", "created": None, "modified": None}

    def save_project(
        self,
        path: Path,
        point_manager: PointManager,
        settings: Dict[str, Any],
    ) -> None:
        """Save complete project state."""
        try:
            project_data = {
                "metadata": self.metadata,
                "source_points": point_manager.source_points.to_dict(),
                "destination_points": point_manager.destination_points.to_dict(),
                "settings": settings,
            }

            with open(path, "w") as f:
                json.dump(project_data, f, indent=2)

            self.project_path = path
            self.is_modified = False
            logger.info(f"Saved project to {path}")

        except Exception as e:
            logger.error(f"Failed to save project: {e}")
            raise

    def load_project(self, path: Path) -> Dict[str, Any]:
        """Load project state from file."""
        try:
            with open(path, "r") as f:
                project_data = json.load(f)

            self.project_path = path
            self.is_modified = False
            logger.info(f"Loaded project from {path}")

            return project_data

        except Exception as e:
            logger.error(f"Failed to load project: {e}")
            raise
