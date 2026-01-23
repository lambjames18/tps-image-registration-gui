"""
presenter.py - MVP Presenter for Distortion Correction Application

This module acts as the intermediary between the model and view layers.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Union
from enum import Enum
import numpy as np

from models import (
    Point,
    PointManager,
    PointAutoIdentifier,
    ImageLoader,
    ImageWriter,
    TransformManager,
    ImageProcessor,
    ProjectManager,
    ImageData,
    TransformType,
    DataFormat,
)

# Configure logging
logger = logging.getLogger(__name__)


class ViewMode(Enum):
    """Enumeration of view modes."""

    SPLIT_HORIZONTAL = "horizontal"
    SPLIT_VERTICAL = "vertical"
    OVERLAY = "overlay"
    SIDE_BY_SIDE = "side_by_side"


class CropMode(Enum):
    """Enumeration of crop modes."""

    SOURCE = "source"
    DESTINATION = "destination"


class ApplicationPresenter:
    """Main presenter that coordinates between model and view components."""

    def __init__(self):
        # Model components
        self.point_manager = PointManager()
        self.transform_manager = TransformManager()
        self.project_manager = ProjectManager()
        self.image_processor = ImageProcessor()
        self.image_writer = ImageWriter()
        self.point_auto_identifier = PointAutoIdentifier()

        # Data storage
        self.source_image: Optional[ImageData] = None
        self.destination_image: Optional[ImageData] = None
        self.current_slice = 0
        self.current_source_mode = "Intensity"
        self.current_dest_mode = "Intensity"

        # State flags
        self.clahe_active_source = False
        self.clahe_active_dest = False
        self.match_resolutions = False

        # View reference (will be set by view)
        self.view = None

        # Paths
        self.source_points_path: Optional[Path] = None
        self.dest_points_path: Optional[Path] = None

        logger.info("ApplicationPresenter initialized")

    def set_view(self, view) -> None:
        """Set the view reference."""
        self.view = view

    def new_project(self) -> None:
        """Create a new empty project."""
        try:
            # Reset all model components
            self.point_manager = PointManager()
            self.transform_manager = TransformManager()
            self.project_manager = ProjectManager()
            self.image_processor = ImageProcessor()
            self.image_writer = ImageWriter()

            # Clear data storage
            self.source_image = None
            self.destination_image = None
            self.current_slice = 0
            self.current_source_mode = "Intensity"
            self.current_dest_mode = "Intensity"

            # Reset state flags
            self.clahe_active_source = False
            self.clahe_active_dest = False
            self.match_resolutions = False

            # Clear paths
            self.source_points_path = None
            self.dest_points_path = None

            logger.info("New project created")
            self._notify_view_project_reset()

        except Exception as e:
            logger.error(f"Failed to create new project: {e}")
            self._notify_view_error(
                f"Failed to create new project: {str(e)}, ({parse_error()})"
            )

    def has_unsaved_changes(self) -> bool:
        """Check if there are unsaved changes."""
        # Check if there's any data loaded
        has_data = (
            self.source_image is not None
            or self.destination_image is not None
            or len(self.point_manager.source_points.points) > 0
        )

        # If there's data and no project path, consider it unsaved
        if has_data and self.project_manager.project_path is None:
            return True

        # Check if project has been modified
        return has_data and self.project_manager.is_modified

    # ========== Data Loading ==========

    def load_source_image(
        self,
        path: Union[str, Path, List[Path], List[str]],
        resolution: float = 1.0,
        modality_name: str = None,
    ) -> bool:
        """Load source (distorted) image."""
        try:
            # Convert path to Path object if needed
            is_single_image = True
            if isinstance(path, (list, tuple)):
                path = [Path(p) if isinstance(p, str) else p for p in path]
                is_single_image = False
            elif isinstance(path, str):
                path = Path(path)

            # Check if single image file (already determined for list case)
            if is_single_image:
                is_single_image = path.suffix.lower() in [
                    ".tif",
                    ".tiff",
                    ".png",
                    ".jpg",
                    ".jpeg",
                ]

            # Read in the new data
            new_image_data = ImageLoader.load(path, resolution, modality_name)

            # Make sure the new data has the same number of slices as the source image
            if self.destination_image:
                if (
                    self.destination_image.shape[0]
                    != list(new_image_data.data.values())[0].shape[0]
                ):
                    raise ValueError(
                        "Source and destination images must have the same number of slices."
                    )

            # If this is the first time reading a source image, set it
            if self.source_image is None:
                logger.debug(
                    f"New source image. Setting with modalities: {new_image_data.modalities}"
                )
                self.source_image = new_image_data
                # Set default points path if not set
                if self.source_points_path is None:
                    if isinstance(path, (list, tuple)):
                        path = Path(path[0])
                    self.source_points_path = path.parent / "source_pts.txt"
                # Set the current mode to the first available modality
                if self.source_image and self.source_image.modalities:
                    self.current_source_mode = self.source_image.modalities[0]
            else:
                logger.debug(
                    f"Adding modality to existing source image: {new_image_data.modalities}"
                )
                # Add the new modality to existing image data
                modality_key = list(new_image_data.data.keys())[0]
                modality_path = list(new_image_data.paths.values())[0]
                self.source_image.add_modality(
                    modality_key, new_image_data.data[modality_key], modality_path
                )
                # Switch to the newly added modality
                self.current_source_mode = modality_key

            self.project_manager.mark_modified()
            self._notify_view_data_loaded()
            return True

        except Exception as e:
            logger.error(f"Failed to load source image: {e} ({parse_error()})")
            self._notify_view_error(
                f"Failed to load source image: {str(e)}, ({parse_error()})"
            )
            return False

    def load_destination_image(
        self,
        path: Union[Path, List[Path]],
        resolution: float = 1.0,
        modality_name: str = None,
    ) -> bool:
        """Load destination (control) image."""
        try:
            # Convert path to Path object if needed
            if isinstance(path, (list, tuple)):
                path = [Path(p) if isinstance(p, str) else p for p in path]
            elif isinstance(path, str):
                path = Path(path)

            # Read in the new data
            new_image_data = ImageLoader.load(path, resolution, modality_name)

            # Make sure the new data has the same number of slices as the source image
            if self.source_image:
                if (
                    self.source_image.shape[0]
                    != list(new_image_data.data.values())[0].shape[0]
                ):
                    raise ValueError(
                        "Source and destination images must have the same number of slices."
                    )

            # If this is the first time reading a destination image, set it
            if self.destination_image is None:
                self.destination_image = new_image_data
                # Set default points path if not set
                if self.dest_points_path is None:
                    if isinstance(path, (list, tuple)):
                        path = Path(path[0])
                    self.dest_points_path = path.parent / "destination_pts.txt"
                # Set the current mode to the first available modality
                if self.destination_image and self.destination_image.modalities:
                    self.current_dest_mode = self.destination_image.modalities[0]
            else:
                # Add the new modality to existing image data
                self.destination_image.add_modality(new_image_data)
                # Switch to the newly added modality
                self.current_dest_mode = new_image_data.modalities[0]
                logger.info(
                    f"Added modality '{new_image_data.modalities[0]}' to destination image and switched to it"
                )

            self.project_manager.mark_modified()
            self._notify_view_data_loaded()
            return True

        except Exception as e:
            logger.error(f"Failed to load destination image: {e} ({parse_error()})")
            self._notify_view_error(
                f"Failed to load destination image: {str(e)}, ({parse_error()})"
            )
            return False

    def load_source_points(self, source_path: Path) -> bool:
        """Load source control points from file."""
        try:
            # Determine if data is 2D (single slice)
            is_2d = (self.source_image is None) or (self.source_image.shape[0] == 1)

            self.point_manager.load_source_from_file(
                source_path, current_slice=self.current_slice, is_2d=is_2d
            )
            self.source_points_path = source_path
            logger.info(f"Loaded source control points from {source_path}")
            self._notify_view_points_changed()
            return True

        except Exception as e:
            logger.error(f"Failed to load source points: {e}")
            self._notify_view_error(
                f"Failed to load source points: {str(e)}, ({parse_error()})"
            )
            return False

    def load_destination_points(self, dest_path: Path) -> bool:
        """Load destination control points from file."""
        try:
            # Determine if data is 2D (single slice)
            is_2d = (self.destination_image is None) or (
                self.destination_image.shape[0] == 1
            )

            self.point_manager.load_destination_from_file(
                dest_path, current_slice=self.current_slice, is_2d=is_2d
            )
            self.dest_points_path = dest_path
            logger.info(f"Loaded destination control points from {dest_path}")
            self._notify_view_points_changed()
            return True

        except Exception as e:
            logger.error(f"Failed to load destination points: {e}")
            self._notify_view_error(
                f"Failed to load destination points: {str(e)}, ({parse_error()})"
            )
            return False

    # ========== Point Management ==========

    def set_checkpoint_path(self, path: Path) -> None:
        """Set the checkpoint path for MatchAnything point detection."""
        try:
            PointAutoIdentifier.set_checkpoint_path(str(path))
            logger.info(f"Checkpoint path set to: {path}")
        except Exception as e:
            logger.error(f"Failed to set checkpoint path: {e}")
            self._notify_view_error(
                f"Failed to set checkpoint path: {str(e)}, ({parse_error()})"
            )

    def get_checkpoint_path(self) -> Optional[str]:
        """Get the current checkpoint path for MatchAnything."""
        return PointAutoIdentifier.checkpoint_path

    def auto_detect_points(self, method: str, **kwargs) -> None:
        """Automatically detect control points using specified method.

        Args:
            method: Detection method ('sift' or 'matchanything')
            **kwargs: Method-specific parameters:
                For SIFT:
                    - max_ratio: Lowe's ratio test threshold (default: 0.75)
                    - min_matches: Minimum matches required (default: 4)
                    - sigma: Gaussian blur sigma (default: 0.5)
                    - num_samples: Number of RANSAC samples (default: 10)
                    - ransac_threshold: RANSAC inlier threshold (default: 5.5)
                    - ransac_max_trials: Max RANSAC iterations (default: 1000)
                    - ransac_method: RANSAC method (default: 'deformable')
                For MatchAnything:
                    - num_samples: Number of point samples (default: 10)
                    - ransac_filter: Enable RANSAC filtering (default: True)
                    - ransac_threshold: RANSAC threshold (default: 0.05)
                    - ransac_max_trials: Max RANSAC iterations (default: 100)
                    - ransac_method: RANSAC method (default: 'deformable')
        """
        try:
            if self.source_image is None or self.destination_image is None:
                raise ValueError("Both source and destination images must be loaded.")

            # Get current images
            src_img, dst_img = self.get_current_images(normalize=True)

            # Set default kwargs if not provided
            if method == "sift":
                defaults = {
                    "max_ratio": 0.75,
                    "min_matches": 4,
                    "sigma": 0.5,
                    "num_samples": 10,
                    "ransac_threshold": 5.5,
                    "ransac_max_trials": 1000,
                    "ransac_method": "deformable",
                }
            else:
                defaults = {
                    "num_samples": 10,
                    "ransac_filter": True,
                    "ransac_threshold": 0.05,
                    "ransac_max_trials": 100,
                    "ransac_method": "deformable",
                }

            # Merge defaults with provided kwargs (provided values override defaults)
            detection_kwargs = {**defaults, **kwargs}

            # Detect points
            src_points, dst_points = self.point_auto_identifier.detect_points(
                src_img, dst_img, method=method, **detection_kwargs
            )
            if src_points.size == 0 or dst_points.size == 0:
                logger.warning("No points detected by auto identifier")
                return False
            logger.info(f"Detected {len(src_points)} points")

            # Make sure there are no duplicate points
            src_points, idx = np.unique(src_points, axis=0, return_index=True)
            dst_points = dst_points[idx]
            dst_points, idx = np.unique(dst_points, axis=0, return_index=True)
            src_points = src_points[idx]
            logger.info(
                f"{detection_kwargs['num_samples'] - len(src_points)} duplicate points removed ({len(src_points)} unique points remain)"
            )

            # Remove points that are already present
            existing_src_points, existing_dst_points = (
                self.point_manager.get_point_pairs(self.current_slice)
            )
            if existing_src_points.size > 0 and existing_dst_points.size > 0:
                existing_src_delta = set(map(tuple, existing_src_points))
                existing_dst_delta = set(map(tuple, existing_dst_points))
                points = np.array(
                    [
                        (sp, dp)
                        for sp, dp in zip(src_points, dst_points)
                        if tuple(sp) not in existing_src_delta
                        and tuple(dp) not in existing_dst_delta
                    ]
                )
                if points.size == 0:
                    logger.warning(
                        "No new points to add after removing existing points"
                    )
                    return False
                logger.info(f"{len(src_points) - len(points)} existing points removed")
                src_points = np.array([p[0] for p in points])
                dst_points = np.array([p[1] for p in points])

            # Scale points if resolutions are matched
            if self.match_resolutions:
                src_res, dst_res = self.get_resolutions()
                res_scale = src_res / dst_res
                dst_points = np.array(
                    [(p[0] * res_scale, p[1] * res_scale) for p in dst_points]
                )

            logger.info(
                f"Auto-detected {len(src_points)} new control points using {method}"
            )
            # Add detected points to point manager
            for sp, dp in zip(src_points, dst_points):
                self.point_manager.source_points.add_point(
                    Point(int(sp[0]), int(sp[1]), self.current_slice)
                )
                self.point_manager.destination_points.add_point(
                    Point(int(dp[0]), int(dp[1]), self.current_slice)
                )

            self._save_points()
            self.project_manager.mark_modified()
            self._notify_view_points_changed()

            return True

        except Exception as e:
            logger.error(f"Auto point detection failed: {e}")
            self._notify_view_error(
                f"Auto point detection failed: {str(e)}, ({parse_error()})"
            )
            return False

    def is_point_in_bounds(self, source: str, x: int, y: int) -> bool:
        """Check if a point is within the bounds of the specified image.

        Args:
            source: "source" or "destination"
            x: x coordinate
            y: y coordinate

        Returns:
            True if point is within bounds, False otherwise
        """
        try:
            if source == "source":
                if not self.source_image:
                    return False
                # Get current image dimensions (accounting for match_resolutions)
                img = self.source_image.get_slice(
                    self.current_source_mode, self.current_slice
                )
                height, width = img.shape[:2]
            elif source == "destination":
                if not self.destination_image:
                    return False
                # Get current image dimensions (accounting for match_resolutions)
                img = self.destination_image.get_slice(
                    self.current_dest_mode, self.current_slice
                )
                height, width = img.shape[:2]

                # Adjust bounds if resolutions are matched
                if self.match_resolutions and self.source_image:
                    src_res, dst_res = self.get_resolutions()
                    res_scale = dst_res / src_res
                    width = int(width * res_scale)
                    height = int(height * res_scale)
            else:
                return False

            # Check if point is within bounds
            return 0 <= x < width and 0 <= y < height

        except Exception as e:
            logger.error(f"Error checking point bounds: {e}")
            return False

    def add_point(self, source: str, x: int, y: int) -> None:
        """Add a control point."""
        try:
            point = Point(x, y, self.current_slice)

            if source == "source" and self.source_image is not None:
                # Need to add corresponding point in destination
                # For now, add at same location (user will adjust)
                self.point_manager.source_points.add_point(point)
                self._notify_view_request_corresponding_point("destination")
            elif source == "destination" and self.destination_image is not None:
                # Correct for matched resolutions
                if self.match_resolutions:
                    src_res, dst_res = self.get_resolutions()
                    res_scale = src_res / dst_res
                    point = Point(
                        int(x * res_scale), int(y * res_scale), self.current_slice
                    )
                # Add destination point
                self.point_manager.destination_points.add_point(point)

            self._save_points()
            self.project_manager.mark_modified()
            self._notify_view_points_changed()

        except Exception as e:
            logger.error(f"Failed to add point: {e}")
            self._notify_view_error(f"Failed to add point: {str(e)}, ({parse_error()})")

    def remove_point(self, point_index: int) -> None:
        """Remove a control point."""
        try:
            success = self.point_manager.remove_point_pair(
                self.current_slice, point_index
            )

            if success:
                self._save_points()
                self.project_manager.mark_modified()
                self._notify_view_points_changed()

        except Exception as e:
            logger.error(f"Failed to remove point: {e}")
            self._notify_view_error(
                f"Failed to remove point: {str(e)}, ({parse_error()})"
            )

    def clear_points(self, slice_only: bool = True) -> None:
        """Clear control points."""
        try:
            if slice_only:
                self.point_manager.clear_points(self.current_slice)
            else:
                self.point_manager.clear_points()

            self._save_points()
            self.project_manager.mark_modified()
            self._notify_view_points_changed()

        except Exception as e:
            logger.error(f"Failed to clear points: {e}")
            self._notify_view_error(
                f"Failed to clear points: {str(e)}, ({parse_error()})"
            )

    def get_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current points for display."""
        return self.point_manager.get_point_pairs(self.current_slice)

    def undo(self) -> None:
        """Undo last point operation."""
        if self.point_manager.undo():
            self._notify_view_points_changed()

    def redo(self) -> None:
        """Redo last undone point operation."""
        if self.point_manager.redo():
            self._notify_view_points_changed()

    # ========== Image Processing ==========

    def toggle_clahe(self, source: str) -> None:
        """Toggle CLAHE for source or destination image."""
        if source == "source":
            self.clahe_active_source = not self.clahe_active_source
        else:
            self.clahe_active_dest = not self.clahe_active_dest

        self._notify_view_update_display()

    def toggle_match_resolutions(self):
        self.match_resolutions = not self.match_resolutions
        self._notify_view_update_display()

    def get_current_images(
        self,
        scale: float = None,
        src_scale: float = None,
        dst_scale: float = None,
        normalize=True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get current images for display."""
        if scale is not None:
            src_scale = scale
            dst_scale = scale
        elif src_scale is None and dst_scale is not None:
            src_scale = dst_scale
            logger.warning("Source scale not provided; using destination scale")
        elif dst_scale is None and src_scale is not None:
            dst_scale = src_scale
            logger.warning("Destination scale not provided; using source scale")
        elif src_scale is None and dst_scale is None:
            src_scale = 1.0
            dst_scale = 1.0
            logger.info("Scaling factors not provided; using 1.0 for both")

        # Get the current source image
        if not self.source_image:
            src_img = None
        else:
            try:
                src_img = self.source_image.get_slice(
                    self.current_source_mode, self.current_slice
                )
                channels = src_img.shape[2]
                src_img = np.squeeze(src_img)

                # Apply CLAHE if active
                if self.clahe_active_source:
                    src_img = self.image_processor.apply_clahe(src_img)

                # Resize if needed
                if src_scale != 1.0:
                    src_img = self.image_processor.resize_image(src_img, src_scale)

                if src_img.ndim == 2:
                    src_img = src_img.reshape(src_img.shape + (channels,))

                # Normalize to uint8
                if normalize:
                    src_img = self.image_processor.normalize_to_uint8(src_img)

            except Exception as e:
                logger.error(f"Failed to get current source image: {e}")
                src_img = None

        # Get the current destination image
        if not self.destination_image:
            dst_img = None
        else:
            try:
                dst_img = self.destination_image.get_slice(
                    self.current_dest_mode, self.current_slice
                )
                channels = dst_img.shape[2]
                dst_img = np.squeeze(dst_img)

                # Reconcile image resolutions (not done for EBSD)
                if self.match_resolutions:
                    downscale = (
                        self.destination_image.resolution / self.source_image.resolution
                    )
                    dst_img = self.image_processor.resize_image(dst_img, downscale)

                # Apply CLAHE if active
                if self.clahe_active_dest:
                    dst_img = self.image_processor.apply_clahe(dst_img)

                # Resize if needed
                if dst_scale != 1.0:
                    dst_img = self.image_processor.resize_image(dst_img, dst_scale)

                if dst_img.ndim == 2:
                    dst_img = dst_img.reshape(dst_img.shape + (channels,))

                # Normalize to uint8
                if normalize:
                    dst_img = self.image_processor.normalize_to_uint8(dst_img)

            except Exception as e:
                logger.error(f"Failed to get current destination image: {e}")
                dst_img = None

        return src_img, dst_img

    def get_current_image_stacks(self, normalize=True) -> Tuple[np.ndarray, np.ndarray]:
        """Get current image stacks for 3D processing."""
        # Get the current source image stack
        if not self.source_image:
            src_stack = None
        else:
            src_stack = self.source_image.data[self.current_source_mode]
            # Apply CLAHE if active
            if self.clahe_active_source:
                src_stack = self.image_processor.apply_clahe(src_stack)

            # Normalize to uint8
            if normalize:
                src_stack = self.image_processor.normalize_to_uint8(src_stack)

        # Get the current destination image stack
        if not self.destination_image:
            dst_stack = None
        else:
            dst_stack = self.destination_image.data[self.current_dest_mode]

            # Reconcile image resolutions (not done for EBSD)
            if self.match_resolutions:
                downscale = (
                    self.destination_image.resolution / self.source_image.resolution
                )
                dst_stack = self.image_processor.resize_image(dst_stack, downscale)

            # Apply CLAHE if active
            if self.clahe_active_dest:
                dst_stack = self.image_processor.apply_clahe(dst_stack)

            # Normalize to uint8
            if normalize:
                dst_stack = self.image_processor.normalize_to_uint8(dst_stack)

        return src_stack, dst_stack

    # ========== Transformation ==========

    def apply_transform(
        self,
        transform_type: TransformType,
        crop_mode: CropMode = CropMode.DESTINATION,
        normalize: bool = True,
        preview: bool = False,
        return_images: bool = False,
    ) -> Optional[np.ndarray]:
        """Apply transformation to current image/slice."""
        try:
            src_points, dst_points = self.point_manager.get_point_pairs(
                self.current_slice
            )

            if src_points.size == 0 or dst_points.size == 0:
                self._notify_view_error("No control points defined for transformation")
                return None

            # Correct points for matched resolutions
            if self.match_resolutions:
                src_res, dst_res = self.get_resolutions()
                res_scale = dst_res / src_res
                dst_points = np.array(
                    [(p[0] * res_scale, p[1] * res_scale) for p in dst_points]
                )

            # Get current source image
            src_img, dst_img = self.get_current_images(normalize=normalize)

            # Estimate transform
            output_shape = dst_img.shape[:2]
            tform = self.transform_manager.estimate_transform(
                src_points, dst_points, transform_type, output_shape
            )

            # Apply transform
            warped = self.transform_manager.apply_transform(
                src_img, tform, output_shape
            )
            # Apply cropping if requested
            if crop_mode == CropMode.SOURCE:
                dummy = np.ones_like(src_img)
                dummy = self.transform_manager.apply_transform(
                    dummy, tform, output_shape
                )
                slc = self._get_cropping_slice(dummy, dst_img, src_img.shape, crop_mode)
                warped = warped[slc[1:]]
                dst_img = dst_img[slc[1:]]

            if preview:
                self._notify_view_show_preview(warped, dst_img)

            if return_images:
                return warped, src_img, dst_img

        except Exception as e:
            logger.error(f"Failed to apply transform: {e}")
            self._notify_view_error(
                f"Failed to apply transform: {str(e)}, ({parse_error()})"
            )
            return None

    def apply_transform_3d(
        self,
        transform_type: TransformType,
        crop_mode: CropMode = CropMode.SOURCE,
        normalize: bool = True,
        preview: bool = False,
        return_images: bool = False,
    ) -> Optional[np.ndarray]:
        """Apply transformation to entire 3D stack."""
        try:
            # Get point pairs
            src_points, dst_points = self.point_manager.get_point_pairs()

            if src_points.size == 0 or dst_points.size == 0:
                self._notify_view_error("No control points defined for transformation")
                return None

            # Correct points for matched resolutions
            if self.match_resolutions:
                src_res, dst_res = self.get_resolutions()
                res_scale = dst_res / src_res
                dst_points = np.array(
                    [(p[0], p[1] * res_scale, p[2] * res_scale) for p in dst_points]
                )

            # Get image stacks
            src_stack, dst_stack = self.get_current_image_stacks(normalize=normalize)

            # Apply transformation
            output_shape = dst_stack.shape[1:3]
            warped_stack = self.transform_manager.apply_transform_stack(
                src_stack, src_points, dst_points, transform_type, output_shape
            )

            # Apply cropping if needed
            if crop_mode == CropMode.SOURCE:
                dummy_stack = np.ones_like(src_stack)
                dummy_stack = self.transform_manager.apply_transform_stack(
                    dummy_stack, src_points, dst_points, transform_type, output_shape
                )
                slc = self._get_cropping_slice(
                    warped_stack, dummy_stack, src_stack.shape, crop_mode
                )
                warped_stack = warped_stack[slc]
                dst_stack = dst_stack[slc]

            if preview:
                self._notify_view_show_preview(warped_stack, dst_stack)

            if return_images:
                return warped_stack, src_stack, dst_stack

        except Exception as e:
            logger.error(f"Failed to apply 3D transform: {e}")
            self._notify_view_error(
                f"Failed to apply 3D transform: {str(e)}, ({parse_error()})"
            )
            return None

    def export_data(
        self,
        path: Path,
        data_format: DataFormat,
        crop_mode: CropMode,
        transform_type: TransformType,
    ) -> bool:
        """Export corrected image data."""
        try:
            # For image export, we just warp the current mode and save the result
            if data_format in {DataFormat.IMAGE, DataFormat.RAW_IMAGE}:
                warped_img, src_img, dst_img = self.apply_transform(
                    transform_type,
                    crop_mode,
                    normalize=False,
                    preview=False,
                    return_images=True,
                )
                if warped_img is None:
                    return False
                if data_format == DataFormat.IMAGE:
                    warped_img = self.image_processor.normalize_to_uint8(warped_img)
                    src_img = self.image_processor.normalize_to_uint8(src_img)
                    dst_img = self.image_processor.normalize_to_uint8(dst_img)

                self.image_writer.save_image(warped_img, path)
                self.image_writer.save_image(
                    src_img, path.with_name(path.stem + "_src" + path.suffix)
                )
                self.image_writer.save_image(
                    dst_img, path.with_name(path.stem + "_dst" + path.suffix)
                )
                return True

            elif data_format == DataFormat.ANG:
                if self.source_image.metadata.get("dataformat") != DataFormat.ANG.value:
                    self._notify_view_error(
                        "Source image is not in .ang format; cannot export as .ang"
                    )
                    return False
                # loop over all modalities and export
                warped_imgs = {}
                src_imgs = {}

                for mode in self.source_image.modalities:
                    if mode.lower() == "eulerangles":
                        continue
                    self.set_source_mode(mode)
                    self._notify_view_update_display()
                    warped, src, dst_img = self.apply_transform(
                        transform_type,
                        crop_mode,
                        normalize=False,
                        preview=False,
                        return_images=True,
                    )
                    if warped is None:
                        return False
                    warped_imgs[mode] = warped
                    src_imgs[mode] = src

                # Save dst image
                self.image_writer.save(dst_img, path.with_name(path.stem + "_dst.tif"))

                # Save warped images as .ang
                self.image_writer.save(
                    warped_imgs,
                    path.with_name(path.stem + ".ang"),
                    self.source_image.metadata["header"],
                    self.source_image.resolution,
                )
                self.image_writer.save(
                    src_imgs,
                    path.with_name(path.stem + "_original.ang"),
                    self.source_image.metadata["header"],
                    self.source_image.resolution,
                )

            elif data_format == DataFormat.H5:
                raise NotImplementedError("H5 export not yet implemented")

            elif data_format == DataFormat.DREAM3D:
                if crop_mode is not CropMode.SOURCE:
                    self._notify_view_error(
                        "Only source cropping is supported for DREAM.3D export"
                    )
                    return False
                # loop over all modalities and export
                warped_stacks = {}

                dst_modes = self.destination_image.modalities
                for mode in self.source_image.modalities:
                    self.set_source_mode(mode)
                    self._notify_view_update_display()
                    warped_stack, src_stack, dst_stack = self.apply_transform_3d(
                        transform_type,
                        crop_mode,
                        normalize=False,
                        preview=False,
                        return_images=True,
                    )
                    if warped_stack is None:
                        return False
                    warped_stacks[mode] = warped_stack
                    if self.current_dest_mode in dst_modes:
                        warped_stacks[self.current_dest_mode] = dst_stack
                        dst_modes.pop(dst_modes.index(self.current_dest_mode))
                        if len(dst_modes) > 0:
                            self.set_destination_mode(dst_modes[0])
                            self._notify_view_update_display()

                self.image_writer.save(warped_stacks, path, self.source_image.path[0])

        except Exception as e:
            logger.error(f"Failed to export data: {e}")
            self._notify_view_error(
                f"Failed to export data: {str(e)}, ({parse_error()})"
            )
            return False

    def export_transform(self, path: Path, transform_type: TransformType) -> bool:
        """Export transformation parameters."""
        try:
            src_points, dst_points = self.point_manager.get_point_pairs(
                self.current_slice
            )

            if src_points.size == 0 or dst_points.size == 0:
                self._notify_view_error("No control points defined for export")
                return False

            # Get destination shape for TPS
            dst_img = self.destination_image.get_slice(
                self.current_dest_mode, self.current_slice
            )
            output_shape = dst_img.shape[:2]

            # Estimate transform
            tform = self.transform_manager.estimate_transform(
                src_points, dst_points, transform_type, size=output_shape
            )

            # Export
            format = path.suffix[1:] if path.suffix else "npy"
            self.transform_manager.export_transform(tform, path, format)

            return True

        except Exception as e:
            logger.error(f"Failed to export transform: {e}")
            self._notify_view_error(
                f"Failed to export transform: {str(e)}, ({parse_error()})"
            )
            return False

    # ========== Project Management ==========

    def save_project(self, path: Path) -> bool:
        """Save current project."""
        try:
            source_paths = {
                k: [str(p) for p in v] for k, v in self.source_image.paths.items()
            }
            destination_paths = {
                k: [str(p) for p in v] for k, v in self.destination_image.paths.items()
            }

            settings = {
                "current_slice": self.current_slice,
                "source_mode": self.current_source_mode,
                "dest_mode": self.current_dest_mode,
                "clahe_source": self.clahe_active_source,
                "clahe_dest": self.clahe_active_dest,
                "match_resolutions": self.match_resolutions,
                "source_resolution": str(self.source_image.resolution),
                "destination_resolution": str(self.destination_image.resolution),
                "source_paths": source_paths,
                "destination_paths": destination_paths,
                "checkpoint_path": self.get_checkpoint_path(),
            }

            # Use the first path for backward compatibility
            self.project_manager.save_project(
                path,
                self.point_manager,
                settings,
            )

            return True

        except Exception as e:
            logger.error(f"Failed to save project: {e}")
            self._notify_view_error(
                f"Failed to save project: {str(e)}, ({parse_error()})"
            )
            return False

    def load_project(self, path: Path) -> bool:
        """Load project from file."""
        try:
            project_data = self.project_manager.load_project(path)
            settings = project_data.get("settings", {})

            # Check if this is a new-style project with multiple paths per modality
            source_paths_dict = settings.get("source_paths", {})
            dest_paths_dict = settings.get("destination_paths", {})

            if source_paths_dict:
                # New-style project: load each modality separately
                for modality_name, modality_path in source_paths_dict.items():
                    path = [Path(p) for p in modality_path]
                    self.load_source_image(
                        path,
                        float(settings.get("source_resolution", 1.0)),
                        modality_name=modality_name,
                    )
                    # For .ang, .h5, .dream3d files, only load the first one as it contains all modalities
                    if path[0].suffix.lower() in [".ang", ".h5", ".dream3d"]:
                        break
            else:
                # Old-style project: load single image
                self.load_source_image(
                    Path(project_data["source_image"]),
                    float(settings.get("source_resolution", 1.0)),
                )

            if dest_paths_dict:
                # New-style project: load each modality separately
                for modality_name, modality_path in dest_paths_dict.items():
                    path = [Path(p) for p in modality_path]
                    self.load_destination_image(
                        path,
                        float(settings.get("destination_resolution", 1.0)),
                        modality_name=modality_name,
                    )
                    # For .ang, .h5, .dream3d files, only load the first one as it contains all modalities
                    if path[0].suffix.lower() in [".ang", ".h5", ".dream3d"]:
                        break
            else:
                # Old-style project: load single image
                self.load_destination_image(
                    Path(project_data["destination_image"]),
                    float(settings.get("destination_resolution", 1.0)),
                )

            # Load points
            self.point_manager.load_from_json(project_data)

            # Load settings
            self.current_slice = settings.get("current_slice", 0)
            self.current_source_mode = settings.get("source_mode", "Intensity")
            self.current_dest_mode = settings.get("dest_mode", "Intensity")
            self.clahe_active_source = settings.get("clahe_source", False)
            self.clahe_active_dest = settings.get("clahe_dest", False)
            self.match_resolutions = settings.get("match_resolutions", False)

            # Load checkpoint path if present
            checkpoint_path = settings.get("checkpoint_path")
            if checkpoint_path:
                self.set_checkpoint_path(Path(checkpoint_path))

            self._notify_view_project_loaded()
            return True

        except Exception as e:
            logger.error(f"Failed to load project: {e}")
            self._notify_view_error(
                f"Failed to load project: {str(e)}, ({parse_error()})"
            )
            return False

    # ========== Navigation ==========

    def set_current_slice(self, slice_idx: int) -> None:
        """Set current slice for 3D data."""
        if self.source_image and 0 <= slice_idx < self.source_image.shape[0]:
            self.current_slice = slice_idx
            self._notify_view_update_display()

    def set_source_mode(self, mode: str) -> None:
        """Set display mode for source image."""
        if self.source_image and mode in self.source_image.modalities:
            self.current_source_mode = mode
            self._notify_view_update_display()

    def set_destination_mode(self, mode: str) -> None:
        """Set display mode for destination image."""
        if self.destination_image and mode in self.destination_image.modalities:
            self.current_dest_mode = mode
            self._notify_view_update_display()

    def set_image_resolutions(self, src_res: float, dst_res: float) -> None:
        """Set image resolutions. This reloads the data with the correct resolutions."""
        if self.source_image:
            self.source_image.resolution = src_res
        if self.destination_image:
            self.destination_image.resolution = dst_res

    def get_slice_range(self) -> Tuple[int, int]:
        """Get valid slice range."""
        if self.source_image:
            return 0, self.source_image.shape[0] - 1
        return 0, 0

    def get_source_modalities(self) -> List[str]:
        """Get available source image modalities."""
        if self.source_image:
            return self.source_image.modalities
        return []

    def get_destination_modalities(self) -> List[str]:
        """Get available destination image modalities."""
        if self.destination_image:
            return self.destination_image.modalities
        return []

    def get_resolutions(self) -> Tuple[float, float]:
        """Get image resolutions."""
        src_res = self.source_image.resolution if self.source_image else 1.0
        dst_res = self.destination_image.resolution if self.destination_image else 1.0
        return src_res, dst_res

    def show_matched_points(self) -> None:
        """Show visualization of matched control points between source and destination images."""
        try:
            # Get current images
            src_img, dst_img = self.get_current_images(normalize=True)

            if src_img is None or dst_img is None:
                self._notify_view_error(
                    "Both source and destination images must be loaded"
                )
                return

            # Get point pairs for current slice
            src_points, dst_points = self.point_manager.get_point_pairs(
                self.current_slice
            )

            if src_points.size == 0 or dst_points.size == 0:
                self._notify_view_error("No control points defined for current slice")
                return

            # Scale points if resolutions are matched
            if self.match_resolutions:
                src_res, dst_res = self.get_resolutions()
                res_scale = dst_res / src_res
                dst_points = np.array(
                    [(p[0] * res_scale, p[1] * res_scale) for p in dst_points]
                )

            # Notify view to show matched points
            self._notify_view_show_matched_points(
                src_img, dst_img, src_points, dst_points
            )

        except Exception as e:
            logger.error(f"Failed to show matched points: {e}")
            self._notify_view_error(
                f"Failed to show matched points: {str(e)}, ({parse_error()})"
            )

    # ========== Private Helper Methods ==========

    def _save_points(self) -> None:
        """Save points to file."""
        if self.source_points_path and self.dest_points_path:
            try:
                self.point_manager.save_to_file(
                    self.source_points_path, self.dest_points_path
                )
            except Exception as e:
                logger.error(f"Failed to save points: {e}")

    def _get_cropping_slice(
        self,
        warped: np.ndarray,
        reference: np.ndarray,
        original_shape: Tuple,
        crop_mode: CropMode,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get cropping slice based on crop mode. Returns slice objects for each dimension (Z, Y, X, C)."""
        if crop_mode == CropMode.SOURCE:
            # Crop to match source grid size
            if warped.ndim == 4:
                ridx = 1
                cidx = 2
            else:
                ridx = 0
                cidx = 1

            # Find centroid of warped non-zero region
            nonzero = np.where(warped > 0)
            if len(nonzero[ridx]) > 0:
                centroid_r = int(np.mean(nonzero[ridx]))
                centroid_c = int(np.mean(nonzero[cidx]))

                # Calculate crop region
                r_start = max(0, centroid_r - original_shape[ridx] // 2)
                r_end = min(warped.shape[ridx], r_start + original_shape[ridx])
                c_start = max(0, centroid_c - original_shape[cidx] // 2)
                c_end = min(warped.shape[cidx], c_start + original_shape[cidx])
            else:
                r_start = 0
                r_end = original_shape[ridx]
                c_start = 0
                c_end = original_shape[cidx]
            return (
                slice(None),
                slice(r_start, r_end),
                slice(c_start, c_end),
                slice(None),
            )

        else:
            return (slice(None), slice(None), slice(None), slice(None))

    # ========== View Notification Methods ==========

    def _notify_view_data_loaded(self) -> None:
        """Notify view that data has been loaded."""
        if self.view:
            self.view.on_data_loaded()

    def _notify_view_points_changed(self) -> None:
        """Notify view that points have changed."""
        if self.view:
            self.view.on_points_changed()

    def _notify_view_update_display(self) -> None:
        """Notify view to update display."""
        if self.view:
            self.view.on_display_update_needed()

    def _notify_view_error(self, message: str) -> None:
        """Notify view of an error."""
        if self.view:
            self.view.on_error(message)

    def _notify_view_show_preview(
        self, warped: np.ndarray, reference: np.ndarray
    ) -> None:
        """Notify view to show transformation preview."""
        if self.view:
            if warped.ndim == 4:
                self.view.on_show_preview_3d(warped, reference)
            else:
                self.view.on_show_preview_2d(warped, reference)

    def _notify_view_project_loaded(self) -> None:
        """Notify view that a project has been loaded."""
        if self.view:
            self.view.on_project_loaded()

    def _notify_view_request_corresponding_point(self, target: str) -> None:
        """Notify view to request corresponding point from user."""
        if self.view:
            self.view.on_request_corresponding_point(target)

    def _notify_view_project_reset(self) -> None:
        """Notify view that a new project has been created."""
        if self.view:
            self.view.on_project_reset()

    def _notify_view_show_matched_points(
        self,
        src_img: np.ndarray,
        dst_img: np.ndarray,
        src_points: np.ndarray,
        dst_points: np.ndarray,
    ) -> None:
        """Notify view to show matched points visualization."""
        if self.view:
            self.view.on_show_matched_points(src_img, dst_img, src_points, dst_points)


def parse_error():
    import sys
    import os

    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    return (exc_type, fname, exc_tb.tb_lineno)
