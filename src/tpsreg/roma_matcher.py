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

import pytorch_lightning as pl
from Matchanything.src.lightning.lightning_loftr import PL_LoFTR, MatchAnything_Model
from Matchanything.src.config.default import get_cfg_defaults
from ransac import ransac_filter as ransac

logger = logging.getLogger(__name__)


def _prepare_image(im: np.ndarray) -> torch.Tensor:
    """
    Prepares an image for input to the matcher.
    """
    # Convert to float32 numpy array
    im = im.astype(np.float32)

    # Normalize function
    im = im / im.max() if im.max() > 0 else im
    im = np.clip(im, 0.0, 1.0)

    # Normalize to [0, 1]
    if im.ndim == 2:
        im = np.stack([im] * 3, axis=-1)
    elif im.shape[2] == 1:
        im = np.concatenate([im] * 3, axis=-1)
    tensor = torch.from_numpy(im.transpose(2, 0, 1))
    # rgb_uint8 = (im * 255.0).round().clip(0, 255).astype(np.uint8)
    return tensor  # , rgb_uint8


def get_config(checkpoint_path: str = None) -> Tuple[object, dict]:
    cfg = {
        "eval_dataset": True,
        "main_cfg_path": "Matchanything/configs/models/roma_model.py",  # "configs/models/eloftr_model.py", "configs/models/roma_model.py",  # Required, replace with actual path
        "ckpt_path": (
            checkpoint_path
            if checkpoint_path is not None
            else "Matchanything/weights/matchanything_roma.ckpt"
        ),  # "weights/matchanything_eloftr.ckpt", # "weights/matchanything_roma.ckpt", #"weights/minima_roma.pth", , "weights/minima_roma.pth"
        "thr": 0.1,
        "method": "matchanything_roma",  # "matchanything_eloftr", #"minima_roma", # "SIFT"
        "transformation_type": "tps",  # "affine",  # "homo",
        "imgresize": 832,
        "divisible_by": 32,  # this factor is utilized in the image loading pipeline to ensure the image sizes are divisible by this number, for ELoFTR 32 is be necessary, does not affect the RoMa loading pipeline
        "resize_by_stretch": True,  # whether to resize by stretch or by padding one dimension (keeping aspect ratio)
        "npe": True,
        "npe2": False,
        "ckpt32": False,
        "fp32": False,
        "dataset_name": "AF9628-Martensitic_SEM-SE-Stitch_EBSD_SameSlice",  # "All", #"5842WCu-Spalled_SEM-SE_SEM-BSE_Multiscale", #"5842WCu_Spalled_Cropping_Study", #"All", #"5842WCu-Spalled_SEM-SE_SEM-BSE_Multiscale", #"In718SS-CHESS", #"C103_fracture_surfaces",  #"AMSpalledTa", #"DDRX_Dislocations_900C_largeFOV", #"DDRX_Dislocations_900C", #"DDRX_Dislocations_1100C", #"MAX_Phase_Dislocations", #"CoNi90-AM-DIC", #"DIC_EBSD_Multimodal_In718", #"TRIP1_steel_LOM_EBSD", #"In718_same_slice_BSE_EBSD", #"SE2_EBSD_X2CrNi12",# "Martensitic_steel_AF_9628_SEM_EBSD",#"5842WCu_Spalled", #"In718SS-CHESS", ##"CoNi90_mid_OM-2-BSE", #"CoNi67_mid_OM-2-high_OM", #"CoNi67_high_OM-2-SE", #"CP700BC_cracks_block5_pedestal1_0", #"AMSpalledTa", #"CoNi67", "Liver_CT-MR"
        "data_root": "data/test_data",
        "output_root": "results",
        "plot_matches": True,
        "plot_matches_alpha": 0.2,
        "plot_matches_color": "error",  # options: ['green', 'error', 'conf']
        "plot_align": True,
        "plot_refinement": False,
        "plot_checkerboard": True,
        "rigid_ransac_thr": 0.05,  # 10.0,
        "ransac_filter": True,
        "ransac_method": "deformable",  # "affine", "homography", "deformable"
        "elastix_ransac_thr": 40.0,
        "normalize_img": True,
        "RANSAC_correspondence_plane": False,
        "comment": "",
    }

    ##########################
    config = get_cfg_defaults()
    # method, estimator = (cfg["method"]).split("@-@")[0], (cfg["method"]).split("@-@")[1]
    if cfg["method"] != "None" and cfg["method"] != "SIFT":
        config.merge_from_file(cfg["main_cfg_path"])

        pl.seed_everything(config.TRAINER.SEED)
        config.METHOD = cfg["method"]
        # Config overwrite:
        if config.LOFTR.COARSE.ROPE:
            assert config.DATASET.NPE_NAME is not None
        if config.DATASET.NPE_NAME is not None:
            config.LOFTR.COARSE.NPE = [832, 832, cfg["imgresize"], cfg["imgresize"]]

        if cfg["thr"] is not None:
            config.LOFTR.MATCH_COARSE.THR = cfg["thr"]

        config.ROMA.RESIZE_BY_STRETCH = cfg["resize_by_stretch"]
        config.DATASET.RESIZE_BY_STRETCH = cfg["resize_by_stretch"]

    return config, cfg


# Convenience function for one-time use
def detect_points_matchanything(
    source_image: np.ndarray,
    destination_image: np.ndarray,
    checkpoint_path: str = None,
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
    # Initialize matcher
    matcher = create_matcher(checkpoint_path=checkpoint_path)

    # Apply model to detect points
    mkpts1, mkpts0, mconf = apply_matcher(
        matcher,
        source_image,
        destination_image,
        ransac_filter=kwargs.get("ransac_filter", True),
        ransac_threshold=kwargs.get("ransac_threshold", 0.05),
        ransac_method=kwargs.get("ransac_method", "deformable"),
    )

    return mkpts1, mkpts0, mconf


def create_matcher(checkpoint_path: str = None) -> object:
    """
    Create and return a ROMA matcher instance.

    Args:
        checkpoint_path: Path to pretrained checkpoint
    Returns:
        Initialized ROMA matcher instance
    """
    config, cfg = get_config(checkpoint_path=checkpoint_path)
    matcher = PL_LoFTR(config, pretrained_ckpt=cfg["ckpt_path"], test_mode=True).matcher
    matcher.eval().cuda()
    return matcher


def apply_matcher(
    matcher: MatchAnything_Model,
    source_image: np.ndarray,
    destination_image: np.ndarray,
    ransac_filter: bool = True,
    ransac_threshold: float = 0.05,
    ransac_method: str = "deformable",
    ransac_max_trials: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply the given matcher model to detect points between source and destination images.

    Args:
        matcher: Pre-initialized matcher model
        source_image: Source image as numpy array
        destination_image: Destination image as numpy array
    Returns:
        Tuple of (source_points, destination_points, confidences)
    """

    # Prepare images
    source_tensor = _prepare_image(source_image).unsqueeze(0).cuda()
    dest_tensor = _prepare_image(destination_image).unsqueeze(0).cuda()
    data = {
        "image0_rgb_origin": dest_tensor,
        "image1_rgb_origin": source_tensor,
    }

    # Perform matching
    with torch.no_grad():
        with torch.autocast(enabled=True, device_type="cuda"):
            matcher(data)

        mkpts0 = data["mkpts0_f"].cpu().numpy()
        mkpts1 = data["mkpts1_f"].cpu().numpy()
        mconf = data["mconf"].cpu().numpy()
    print(f"Total matches found: {len(mkpts0)}")

    # matcher.sample()

    # Apply RANSAC filtering if enabled
    if ransac_filter and len(mkpts0) >= 4:
        inliers = ransac(
            mkpts0,
            mkpts1,
            threshold=ransac_threshold,
            method=ransac_method,
            max_trials=ransac_max_trials,
        )
        mkpts0 = mkpts0[inliers]
        mkpts1 = mkpts1[inliers]
        mconf = mconf[inliers]
        print(f"Matches after RANSAC filtering: {len(mkpts0)}")

    return mkpts1, mkpts0, mconf
