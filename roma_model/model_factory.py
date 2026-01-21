"""
model_factory.py - Factory functions for creating ROMA model instances

Extracted from experiments/roma_outdoor.py with only inference-relevant code.
"""

import torch
from torch import nn
import warnings

from roma_model.models.matcher import (
    RegressionMatcher,
    Decoder,
    ConvRefiner,
    GP,
    CosKernel,
)
from roma_model.models.transformer import Block, TransformerDecoder, MemEffAttention
from roma_model.models.encoders import CNNandDinov2


def get_model(
    pretrained_backbone=True,
    amp=True,
    coarse_resolution=(560, 560),
    coarse_backbone_type="DINOv2_large",
    coarse_feat_dim=1024,
    medium_feat_dim=512,
    coarse_patch_size=14,
    upsample_preds=False,
    symmetric=False,
    attenuate_cert=False,
    **kwargs
):
    """
    Create ROMA model for inference.

    Args:
        pretrained_backbone: Whether to use pretrained DINOv2 weights (default: True)
        amp: Use automatic mixed precision (default: True)
        coarse_resolution: Resolution for coarse matching (default: (560, 560))
        coarse_backbone_type: Backbone type - 'DINOv2' (default: 'DINOv2')
        coarse_feat_dim: Feature dimension from coarse backbone (default: 1024)
        medium_feat_dim: Medium-level feature dimension (default: 512)
        coarse_patch_size: Patch size for DINOv2 (default: 14)
        upsample_preds: Whether to upsample predictions (default: False)
        symmetric: Use symmetric matching (default: False)
        attenuate_cert: Attenuate certainty scores (default: False)
        **kwargs: Additional arguments passed to RegressionMatcher

    Returns:
        RegressionMatcher: Initialized ROMA model
    """
    # Suppress TypedStorage deprecation warnings
    warnings.filterwarnings(
        "ignore", category=UserWarning, message="TypedStorage is deprecated"
    )

    # Feature dimensions
    gp_dim = medium_feat_dim
    feat_dim = medium_feat_dim
    decoder_dim = gp_dim + feat_dim
    cls_to_coord_res = 64

    # Coordinate decoder (transformer-based)
    coordinate_decoder = TransformerDecoder(
        nn.Sequential(
            *[Block(decoder_dim, 8, attn_class=MemEffAttention) for _ in range(5)]
        ),
        decoder_dim,
        cls_to_coord_res**2 + 1,
        is_classifier=True,
        amp=amp,
        pos_enc=False,
    )

    # ConvRefiner parameters
    dw = True
    hidden_blocks = 8
    kernel_size = 5
    displacement_emb = "linear"
    disable_local_corr_grad = True

    # Multi-scale convolutional refiners
    conv_refiner = nn.ModuleDict(
        {
            "16": ConvRefiner(
                2 * medium_feat_dim + 128 + (2 * 7 + 1) ** 2,
                2 * medium_feat_dim + 128 + (2 * 7 + 1) ** 2,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=128,
                local_corr_radius=7,
                corr_in_other=True,
                amp=amp,
                disable_local_corr_grad=disable_local_corr_grad,
                bn_momentum=0.01,
            ),
            "8": ConvRefiner(
                2 * medium_feat_dim + 64 + (2 * 3 + 1) ** 2,
                2 * medium_feat_dim + 64 + (2 * 3 + 1) ** 2,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=64,
                local_corr_radius=3,
                corr_in_other=True,
                amp=amp,
                disable_local_corr_grad=disable_local_corr_grad,
                bn_momentum=0.01,
            ),
            "4": ConvRefiner(
                2 * int(medium_feat_dim / 2) + 32 + (2 * 2 + 1) ** 2,
                2 * int(medium_feat_dim / 2) + 32 + (2 * 2 + 1) ** 2,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=32,
                local_corr_radius=2,
                corr_in_other=True,
                amp=amp,
                disable_local_corr_grad=disable_local_corr_grad,
                bn_momentum=0.01,
            ),
            "2": ConvRefiner(
                2 * 64 + 16,
                128 + 16,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=16,
                amp=amp,
                disable_local_corr_grad=disable_local_corr_grad,
                bn_momentum=0.01,
            ),
            "1": ConvRefiner(
                2 * 9 + 6,
                24,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=6,
                amp=amp,
                disable_local_corr_grad=disable_local_corr_grad,
                bn_momentum=0.01,
            ),
        }
    )

    # Gaussian Process for coarse matching
    kernel_temperature = 0.2
    learn_temperature = False
    no_cov = True
    kernel = CosKernel
    only_attention = False
    basis = "fourier"

    gp16 = GP(
        kernel,
        T=kernel_temperature,
        learn_temperature=learn_temperature,
        only_attention=only_attention,
        gp_dim=gp_dim,
        basis=basis,
        no_cov=no_cov,
    )
    gps = nn.ModuleDict({"16": gp16})

    # Feature projection layers
    proj16 = nn.Sequential(
        nn.Conv2d(coarse_feat_dim, medium_feat_dim, 1, 1),
        nn.BatchNorm2d(medium_feat_dim),
    )
    proj8 = nn.Sequential(
        nn.Conv2d(512, medium_feat_dim, 1, 1), nn.BatchNorm2d(medium_feat_dim)
    )
    proj4 = nn.Sequential(
        nn.Conv2d(256, int(medium_feat_dim / 2), 1, 1),
        nn.BatchNorm2d(int(medium_feat_dim / 2)),
    )
    proj2 = nn.Sequential(nn.Conv2d(128, 64, 1, 1), nn.BatchNorm2d(64))
    proj1 = nn.Sequential(nn.Conv2d(64, 9, 1, 1), nn.BatchNorm2d(9))

    proj = nn.ModuleDict(
        {
            "16": proj16,
            "8": proj8,
            "4": proj4,
            "2": proj2,
            "1": proj1,
        }
    )

    # Decoder with multi-scale refinement
    displacement_dropout_p = 0.0
    gm_warp_dropout_p = 0.0

    decoder = Decoder(
        coordinate_decoder,
        gps,
        proj,
        conv_refiner,
        amp=amp,
        detach=True,
        scales=["16", "8", "4", "2", "1"],
        displacement_dropout_p=displacement_dropout_p,
        gm_warp_dropout_p=gm_warp_dropout_p,
    )

    # Encoder (CNN + DINOv2)
    h, w = coarse_resolution
    encoder = CNNandDinov2(
        cnn_kwargs=dict(pretrained=pretrained_backbone, amp=amp),
        amp=amp,
        use_vgg=True,
        coarse_backbone=coarse_backbone_type,
        coarse_patch_size=coarse_patch_size,
        coarse_feat_dim=coarse_feat_dim,
    )

    # Complete matcher
    matcher = RegressionMatcher(
        encoder,
        decoder,
        h=h,
        w=w,
        upsample_preds=upsample_preds,
        symmetric=symmetric,
        attenuate_cert=attenuate_cert,
        **kwargs
    )

    return matcher


# Convenience function with preset configurations
def get_roma_model(
    checkpoint_path=None,
    device="cuda",
    amp=True,
    coarse_backbone_type="DINOv2_large",
    coarse_feat_dim=1024,
    medium_feat_dim=512,
    coarse_patch_size=14,
    coarse_resolution=(560, 560),
    upsample_resolution=(864, 864),
    symmetric=True,
    attenuate_cert=True,
):
    """
    Create and load ROMA model with standard inference settings.

    Args:
        checkpoint_path: Path to checkpoint file (optional)
        device: Device to load model on ('cuda' or 'cpu')
        coarse_resolution: Resolution for coarse matching
        upsample_resolution: Resolution for upsampling
        symmetric: Use symmetric matching
        attenuate_cert: Attenuate certainty scores

    Returns:
        RegressionMatcher: Loaded ROMA model ready for inference
    """
    # Create model
    model = get_model(
        pretrained_backbone=True,
        amp=amp,
        coarse_resolution=coarse_resolution,
        coarse_backbone_type=coarse_backbone_type,
        coarse_feat_dim=coarse_feat_dim,
        medium_feat_dim=medium_feat_dim,
        coarse_patch_size=coarse_patch_size,
        upsample_preds=True,
        symmetric=symmetric,
        attenuate_cert=attenuate_cert,
    )

    # Set upsample resolution
    if upsample_resolution is not None:
        model.upsample_res = upsample_resolution

    # Load checkpoint if provided
    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, map_location="cpu")

        # Handle different checkpoint formats
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # Remove 'matcher.' prefix if present
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("matcher."):
                cleaned_state_dict[k.replace("matcher.", "", 1)] = v
            elif k.startswith("model."):
                cleaned_state_dict[k.replace("model.", "", 1)] = v
            else:
                cleaned_state_dict[k] = v

        model.load_state_dict(cleaned_state_dict, strict=False)

    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()

    return model
