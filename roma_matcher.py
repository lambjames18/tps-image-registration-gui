"""
roma_matcher.py - Standalone ROMA Matcher for Control Point Detection

This module provides a clean interface to ROMA (Regression Matcher with Augmentation)
for automatic control point detection between image pairs.

Usage:
    from roma_matcher import RomaMatcher

    # Initialize matcher
    matcher = RomaMatcher(
        checkpoint_path="matchanything_roma.ckpt",
        confidence_threshold=0.1
    )

    # Detect points between two images
    src_points, dst_points, confidences = matcher.detect_points(
        source_image,  # numpy array (H, W) or (H, W, C)
        dest_image     # numpy array (H, W) or (H, W, C)
    )
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image

# Import standalone roma_model
from roma_model import get_roma_model
from roma_config import get_config

logger = logging.getLogger(__name__)


class RomaMatcher:
    """
    Wrapper class for ROMA (Regression Matcher with Augmentation) model.

    This class provides a simple interface for detecting corresponding control
    points between two images using the ROMA model.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
    ):
        """
        Initialize the ROMA matcher.

        Args:
            checkpoint_path: Path to the pretrained .ckpt file
            confidence_threshold: Minimum confidence for accepting matches (0.0-1.0)
            num_samples: Maximum number of matches to sample
            device: Device to run inference on ("cuda" or "cpu")
            resize_by_stretch: If True, resize images by stretching; if False, pad to square
            normalize_images: Whether to normalize images to [0, 1] range
            coarse_resolution: Resolution for coarse matching (must be divisible by 14)
            upsample_resolution: Resolution for upsampling (must be divisible by 8)
            use_symmetric_matching: Whether to use symmetric matching
            use_certainty_attenuation: Whether to attenuate certainty scores
        """
        self.checkpoint_path = Path(checkpoint_path)

        self.config = get_config()

        self.confidence_threshold = self.config["match_thresh"]
        self.num_samples = self.config["sample"]["n_sample"]
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.resize_by_stretch = self.config["resize_by_stretch"]
        self.normalize_images = self.config["normalize_img"]
        self.coarse_resolution = self.config["test_time"]["coarse_res"]
        self.upsample_resolution = self.config["test_time"]["upsample_res"]
        self.use_symmetric_matching = self.config["test_time"]["symmetric"]
        self.use_certainty_attenuation = self.config["test_time"]["attenuate_cert"]

        # Verify checkpoint exists
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}\n"
                f"Please download the weights from: "
                f"https://drive.google.com/file/d/12L3g9-w8rR9K2L4rYaGaDJ7NqX1D713d/view"
            )

        # Initialize model
        logger.info("Initializing ROMA model...")
        self.model = self._load_model()
        logger.info("ROMA model loaded successfully")

    def _load_model(self):
        """Load the ROMA model with pretrained weights."""
        logger.info(f"Loading checkpoint from {self.checkpoint_path}")

        # Use the factory function from roma_model
        model = get_roma_model(
            checkpoint_path=str(self.checkpoint_path),
            device=str(self.device),
            amp=self.config["model"]["amp"],
            coarse_backbone_type=self.config["model"]["coarse_backbone"],
            coarse_feat_dim=self.config["model"]["coarse_feat_dim"],
            medium_feat_dim=self.config["model"]["medium_feat_dim"],
            coarse_patch_size=self.config["model"]["coarse_patch_size"],
            coarse_resolution=self.config["test_time"]["coarse_res"],
            upsample_resolution=self.config["test_time"]["upsample_res"],
            symmetric=self.config["test_time"]["symmetric"],
            attenuate_cert=self.config["test_time"]["attenuate_cert"],
        )

        # Set sampling parameters
        model.sample_mode = self.config["sample"]["method"]
        model.sample_thresh = self.config["sample"]["thresh"]

        return model

    def _prepare_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Prepare image for ROMA inference.

        Args:
            image: Input image as numpy array (H, W) or (H, W, C)

        Returns:
            Image tensor in CHW format with values in [0, 1]
        """
        # Convert to float32, normalize to [0, 1]
        image = image.astype(np.float32)
        image /= image.max()  # Normalize to [0, 1]

        # Handle grayscale images
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        elif image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)

        # Convert to CHW format
        image = np.transpose(image, (2, 0, 1))

        # Convert to torch tensor
        image_tensor = torch.from_numpy(image).float()

        return image_tensor

    def detect_points(
        self,
        source_image: np.ndarray,
        destination_image: np.ndarray,
        ransac_filter: bool = False,
        ransac_threshold: float = 5.5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect corresponding control points between two images.

        Args:
            source_image: Source image as numpy array (H, W) or (H, W, C)
            destination_image: Destination image as numpy array (H, W) or (H, W, C)
            ransac_filter: Whether to apply RANSAC filtering to remove outliers
            ransac_threshold: RANSAC inlier threshold in pixels (only used if ransac_filter=True)

        Returns:
            Tuple of (source_points, destination_points, confidences):
                - source_points: Nx2 array of (x, y) coordinates in source image
                - destination_points: Nx2 array of (x, y) coordinates in destination image
                - confidences: N array of confidence scores for each match
        """
        # Prepare images
        img0_tensor = self._prepare_image(source_image)
        img1_tensor = self._prepare_image(destination_image)

        # Run ROMA inference using self_inference_time_match
        print("im0_tensor shape:", img0_tensor.shape)
        print("im1_tensor shape:", img1_tensor.shape)
        with torch.no_grad():
            warp, certainty = self.model.self_inference_time_match(
                img0_tensor.to(self.device),
                img1_tensor.to(self.device),
                resize_by_stretch=self.resize_by_stretch,
                norm_img=self.normalize_images,
            )

        # Get image dimensions
        if self.resize_by_stretch:
            H_A, W_A = source_image.shape[:2]
            H_B, W_B = destination_image.shape[:2]
        else:
            H_A = W_A = max(source_image.shape[:2])
            H_B = W_B = max(destination_image.shape[:2])

        # Sample matches
        matches, certainty_sampled = self.model.sample(
            warp, certainty, num=self.num_samples
        )

        # Convert to pixel coordinates
        kpts0, kpts1 = self.model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)

        # Plot detected matches before filtering
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(source_image.astype(np.float32) / 255.0)
        ax[1].imshow(destination_image.astype(np.float32) / 255.0)
        for i in range(0, kpts0.shape[0], 100):
            ax[0].scatter(
                kpts0[i, 0].cpu().numpy(),
                kpts0[i, 1].cpu().numpy(),
                s=300,
                marker=r"${}$".format(i),
            )
            ax[1].scatter(
                kpts1[i, 0].cpu().numpy(),
                kpts1[i, 1].cpu().numpy(),
                s=300,
                marker=r"${}$".format(i),
            )
        plt.show()

        # Apply confidence threshold
        mask = certainty_sampled > self.confidence_threshold

        # Mask borders (ensure points are within image bounds)
        mask *= (kpts0[:, 0] >= 0) * (kpts0[:, 0] < W_A)
        mask *= (kpts0[:, 1] >= 0) * (kpts0[:, 1] < H_A)
        mask *= (kpts1[:, 0] >= 0) * (kpts1[:, 0] < W_B)
        mask *= (kpts1[:, 1] >= 0) * (kpts1[:, 1] < H_B)

        # Extract filtered matches
        mkpts0 = kpts0[mask].cpu().numpy()
        mkpts1 = kpts1[mask].cpu().numpy()
        mconf = certainty_sampled[mask].cpu().numpy()

        logger.info(
            f"Detected {len(mkpts0)} matches with confidence > {self.confidence_threshold}"
        )

        # Optional RANSAC filtering
        if ransac_filter and len(mkpts0) >= 4:
            inliers = self._ransac_filter(mkpts0, mkpts1, ransac_threshold)
            mkpts0 = mkpts0[inliers]
            mkpts1 = mkpts1[inliers]
            mconf = mconf[inliers]
            logger.info(f"After RANSAC filtering: {len(mkpts0)} inliers")

        return mkpts0, mkpts1, mconf

    def _ransac_filter(
        self,
        src_points: np.ndarray,
        dst_points: np.ndarray,
        threshold: float = 5.5,
    ) -> np.ndarray:
        """
        Apply RANSAC filtering to remove outlier matches.

        Args:
            src_points: Nx2 array of source points
            dst_points: Nx2 array of destination points
            threshold: RANSAC inlier threshold in pixels

        Returns:
            Boolean mask indicating inlier matches
        """
        try:
            from skimage.measure import ransac
            from skimage.transform import AffineTransform

            # Use RANSAC to filter outliers with affine model
            model, inliers = ransac(
                (src_points, dst_points),
                AffineTransform,
                min_samples=3,
                residual_threshold=threshold,
                max_trials=1000,
            )

            return inliers

        except Exception as e:
            logger.warning(f"RANSAC filtering failed: {e}. Returning all matches.")
            return np.ones(len(src_points), dtype=bool)

    def detect_points_with_metadata(
        self,
        source_image: np.ndarray,
        destination_image: np.ndarray,
        ransac_filter: bool = False,
        ransac_threshold: float = 5.5,
    ) -> dict:
        """
        Detect points and return detailed metadata.

        Args:
            source_image: Source image as numpy array
            destination_image: Destination image as numpy array
            ransac_filter: Whether to apply RANSAC filtering
            ransac_threshold: RANSAC inlier threshold in pixels

        Returns:
            Dictionary containing:
                - 'source_points': Nx2 array of source coordinates
                - 'destination_points': Nx2 array of destination coordinates
                - 'confidences': N array of confidence scores
                - 'num_matches': Number of detected matches
                - 'mean_confidence': Mean confidence of matches
                - 'ransac_applied': Whether RANSAC filtering was applied
        """
        src_pts, dst_pts, conf = self.detect_points(
            source_image,
            destination_image,
            ransac_filter=ransac_filter,
            ransac_threshold=ransac_threshold,
        )

        return {
            "source_points": src_pts,
            "destination_points": dst_pts,
            "confidences": conf,
            "num_matches": len(src_pts),
            "mean_confidence": float(np.mean(conf)) if len(conf) > 0 else 0.0,
            "ransac_applied": ransac_filter,
        }

    def __call__(
        self, source_image: np.ndarray, destination_image: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convenience method to detect points (returns only point arrays).

        Args:
            source_image: Source image as numpy array
            destination_image: Destination image as numpy array
            **kwargs: Additional arguments passed to detect_points

        Returns:
            Tuple of (source_points, destination_points)
        """
        src_pts, dst_pts, _ = self.detect_points(
            source_image, destination_image, **kwargs
        )
        return src_pts, dst_pts


# Convenience function for one-time use
def detect_points_roma(
    source_image: np.ndarray,
    destination_image: np.ndarray,
    checkpoint_path: str,
    confidence_threshold: float = 0.1,
    ransac_filter: bool = False,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function to detect points without manually creating a matcher.

    Note: This creates a new matcher instance each time it's called.
    For multiple detections, create a RomaMatcher instance and reuse it.

    Args:
        source_image: Source image as numpy array
        destination_image: Destination image as numpy array
        checkpoint_path: Path to pretrained checkpoint
        confidence_threshold: Minimum confidence for matches
        ransac_filter: Whether to apply RANSAC filtering
        **kwargs: Additional arguments passed to RomaMatcher

    Returns:
        Tuple of (source_points, destination_points, confidences)
    """
    matcher = RomaMatcher(
        checkpoint_path=checkpoint_path,
    )

    return matcher.detect_points(
        source_image, destination_image, ransac_filter=ransac_filter
    )
