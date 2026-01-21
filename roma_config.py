"""
roma_config.py - Configuration for ROMA Matcher

This module provides configuration settings for the ROMA model,
following the MatchAnything configuration structure.
"""

config = {
    # Match confidence threshold
    "match_thresh": 0.0,
    # Image preprocessing
    "resize_by_stretch": True,
    "normalize_img": False,
    # Model configuration
    "model": {
        "coarse_backbone": "DINOv2_large",  # Backbone architecture
        "coarse_feat_dim": 1024,  # Feature dimension from DINOv2
        "medium_feat_dim": 512,  # Medium-level feature dimension
        "coarse_patch_size": 14,  # Patch size for DINOv2
        "amp": True,  # Use automatic mixed precision (FP16)
    },
    # Sampling configuration
    "sample": {
        "method": "threshold_balanced",  # Sampling method
        "n_sample": 5000,  # Number of matches to sample
        "thresh": 0.05,  # Threshold for balanced sampling
    },
    # Test-time configuration
    "test_time": {
        "coarse_res": (560, 560),  # Resolution for coarse matching
        "upsample": True,  # Enable upsampling
        "upsample_res": (864, 864),  # Resolution for upsampling
        "symmetric": True,  # Use symmetric matching
        "attenuate_cert": True,  # Attenuate certainty scores
    },
}


def get_config():
    """Retrieve the ROMA configuration dictionary."""
    return config
