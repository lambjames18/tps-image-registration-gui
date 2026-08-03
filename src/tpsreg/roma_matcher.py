"""ROMA / MatchAnything matcher for automatic control point detection.

This module wraps the vendored MatchAnything model (``tpsreg.Matchanything``) so
the rest of the application can request point correspondences without knowing
anything about PyTorch Lightning or the model configuration.

Everything here depends on the optional ``matchanything`` extra::

    pip install "tpsreg[matchanything]"

Torch and the vendored model are imported lazily inside the functions that need
them, so importing this module is cheap and never fails on a machine without
those dependencies installed.

Usage::

    from tpsreg.roma_matcher import apply_matcher, create_matcher

    matcher = create_matcher(checkpoint_path="matchanything_roma.ckpt")
    src_points, dst_points, confidences = apply_matcher(
        matcher,
        source_image,  # numpy array (H, W) or (H, W, C)
        dest_image,
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from tpsreg.ransac import ransac_filter as ransac

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

logger = logging.getLogger(__name__)

#: Directory holding the vendored MatchAnything sources and configs.
MATCHANYTHING_ROOT = Path(__file__).parent / "Matchanything"

#: Default model config, resolved relative to the installed package rather than
#: the current working directory.
DEFAULT_CONFIG_PATH = MATCHANYTHING_ROOT / "configs" / "models" / "roma_model.py"

#: Default checkpoint location, used when the caller does not supply one.
DEFAULT_CHECKPOINT_PATH = MATCHANYTHING_ROOT / "weights" / "matchanything_roma.ckpt"

_MISSING_DEPS_MESSAGE = (
    "Automatic point detection with MatchAnything requires the optional "
    "'matchanything' dependencies. Install them with:\n"
    '    pip install "tpsreg[matchanything]"'
)


def _import_torch():
    """Import torch, raising a helpful error if the extra is not installed."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on install state
        raise ImportError(_MISSING_DEPS_MESSAGE) from exc
    return torch


def select_device(preferred: str | None = None) -> str:
    """Choose the best available torch device.

    Parameters
    ----------
    preferred:
        Explicit device string (``"cuda"``, ``"mps"``, ``"cpu"``). When given,
        it is honoured if available and otherwise falls back to autodetection
        with a warning.

    Returns
    -------
    str
        One of ``"cuda"``, ``"mps"`` or ``"cpu"``.
    """
    torch = _import_torch()

    def _available(name: str) -> bool:
        if name == "cuda":
            return torch.cuda.is_available()
        if name == "mps":
            return getattr(torch.backends, "mps", None) is not None and (
                torch.backends.mps.is_available()
            )
        return name == "cpu"

    if preferred is not None:
        if _available(preferred):
            return preferred
        logger.warning(
            "Requested device '%s' is unavailable; autodetecting.", preferred
        )

    for candidate in ("cuda", "mps"):
        if _available(candidate):
            return candidate
    return "cpu"


def _prepare_image(im: np.ndarray) -> torch.Tensor:
    """Convert an image array into a normalized 3-channel float tensor."""
    torch = _import_torch()

    im = np.asarray(im).astype(np.float32)

    # Scale into [0, 1]; a constant image would otherwise divide by zero.
    peak = im.max()
    if peak > 0:
        im = im / peak
    im = np.clip(im, 0.0, 1.0)

    # The model expects three channels.
    if im.ndim == 2:
        im = np.stack([im] * 3, axis=-1)
    elif im.shape[2] == 1:
        im = np.concatenate([im] * 3, axis=-1)
    elif im.shape[2] > 3:
        im = im[:, :, :3]

    return torch.from_numpy(np.ascontiguousarray(im.transpose(2, 0, 1)))


def get_config(checkpoint_path: str | Path | None = None) -> tuple[Any, dict]:
    """Build the model configuration for the vendored MatchAnything model.

    Parameters
    ----------
    checkpoint_path:
        Path to the pretrained ``.ckpt`` file. Defaults to
        :data:`DEFAULT_CHECKPOINT_PATH`.

    Returns
    -------
    tuple
        ``(yacs_config, raw_settings_dict)``.
    """
    try:
        import pytorch_lightning as pl

        from tpsreg.Matchanything.src.config.default import get_cfg_defaults
    except ImportError as exc:  # pragma: no cover - depends on install state
        raise ImportError(_MISSING_DEPS_MESSAGE) from exc

    settings = {
        "main_cfg_path": str(DEFAULT_CONFIG_PATH),
        "ckpt_path": str(checkpoint_path or DEFAULT_CHECKPOINT_PATH),
        "thr": 0.1,
        "method": "matchanything_roma",
        "transformation_type": "tps",
        "imgresize": 832,
        # Ensures image sizes are divisible by this factor during loading.
        # Required by ELoFTR; ignored by the RoMa pipeline.
        "divisible_by": 32,
        # Resize by stretching rather than padding to preserve aspect ratio.
        "resize_by_stretch": True,
        "npe": True,
        "fp32": False,
        "rigid_ransac_thr": 0.05,
        "ransac_filter": True,
        "ransac_method": "deformable",
        "normalize_img": True,
    }

    config = get_cfg_defaults()
    config.merge_from_file(settings["main_cfg_path"])

    pl.seed_everything(config.TRAINER.SEED)
    config.METHOD = settings["method"]

    if config.LOFTR.COARSE.ROPE and config.DATASET.NPE_NAME is None:
        raise ValueError(
            "Model config enables rotary position embeddings but DATASET.NPE_NAME "
            "is unset; the checkpoint and config are mismatched."
        )
    if config.DATASET.NPE_NAME is not None:
        config.LOFTR.COARSE.NPE = [
            832,
            832,
            settings["imgresize"],
            settings["imgresize"],
        ]

    config.LOFTR.MATCH_COARSE.THR = settings["thr"]
    config.ROMA.RESIZE_BY_STRETCH = settings["resize_by_stretch"]
    config.DATASET.RESIZE_BY_STRETCH = settings["resize_by_stretch"]

    return config, settings


def create_matcher(
    checkpoint_path: str | Path | None = None,
    device: str | None = None,
) -> Any:
    """Instantiate the MatchAnything model and move it to the best device.

    Parameters
    ----------
    checkpoint_path:
        Path to the pretrained checkpoint.
    device:
        Explicit torch device. Autodetected when omitted.

    Returns
    -------
    The evaluation-mode matcher module.

    Raises
    ------
    FileNotFoundError
        If the checkpoint does not exist.
    ImportError
        If the optional ``matchanything`` dependencies are missing.
    """
    try:
        from tpsreg.Matchanything.src.lightning.lightning_loftr import PL_LoFTR
    except ImportError as exc:  # pragma: no cover - depends on install state
        raise ImportError(_MISSING_DEPS_MESSAGE) from exc

    config, settings = get_config(checkpoint_path=checkpoint_path)

    ckpt = Path(settings["ckpt_path"])
    if not ckpt.exists():
        raise FileNotFoundError(
            f"MatchAnything checkpoint not found: {ckpt}\n"
            "Download the weights and point the GUI at them via "
            "'Auto > Set MatchAnything checkpoint...'."
        )

    resolved_device = select_device(device)
    logger.info("Loading MatchAnything checkpoint %s onto %s", ckpt, resolved_device)

    matcher = PL_LoFTR(config, pretrained_ckpt=str(ckpt), test_mode=True).matcher
    matcher = matcher.eval().to(resolved_device)
    # Remember the device so apply_matcher does not have to redetect it.
    matcher._tpsreg_device = resolved_device
    return matcher


def apply_matcher(
    matcher: Any,
    source_image: np.ndarray,
    destination_image: np.ndarray,
    ransac_filter: bool = True,
    ransac_threshold: float = 0.05,
    ransac_method: str = "deformable",
    ransac_max_trials: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect matching points between two images with a prepared matcher.

    Parameters
    ----------
    matcher:
        A model returned by :func:`create_matcher`.
    source_image, destination_image:
        Images as numpy arrays, ``(H, W)`` or ``(H, W, C)``.
    ransac_filter:
        Whether to reject outlier correspondences with RANSAC.
    ransac_threshold, ransac_method, ransac_max_trials:
        Forwarded to :func:`tpsreg.ransac.ransac_filter`.

    Returns
    -------
    tuple
        ``(source_points, destination_points, confidences)``.
    """
    torch = _import_torch()

    device = getattr(matcher, "_tpsreg_device", None) or select_device()

    source_tensor = _prepare_image(source_image).unsqueeze(0).to(device)
    dest_tensor = _prepare_image(destination_image).unsqueeze(0).to(device)
    data = {
        "image0_rgb_origin": dest_tensor,
        "image1_rgb_origin": source_tensor,
    }

    with torch.no_grad():
        # Autocast is only beneficial (and only correct) on CUDA here.
        if device == "cuda":
            with torch.autocast(enabled=True, device_type="cuda"):
                matcher(data)
        else:
            matcher(data)

        mkpts0 = data["mkpts0_f"].cpu().numpy()
        mkpts1 = data["mkpts1_f"].cpu().numpy()
        mconf = data["mconf"].cpu().numpy()

    logger.info("MatchAnything found %d raw matches", len(mkpts0))

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
        logger.info("MatchAnything kept %d matches after RANSAC", len(mkpts0))

    return mkpts1, mkpts0, mconf


def detect_points_matchanything(
    source_image: np.ndarray,
    destination_image: np.ndarray,
    checkpoint_path: str | Path | None = None,
    device: str | None = None,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect points in one call, creating a throwaway matcher.

    Prefer :func:`create_matcher` plus :func:`apply_matcher` when detecting
    points more than once: loading the model is by far the slowest step.

    Parameters
    ----------
    source_image, destination_image:
        Images as numpy arrays.
    checkpoint_path:
        Path to the pretrained checkpoint.
    device:
        Explicit torch device. Autodetected when omitted.
    **kwargs:
        RANSAC settings forwarded to :func:`apply_matcher`.

    Returns
    -------
    tuple
        ``(source_points, destination_points, confidences)``.
    """
    matcher = create_matcher(checkpoint_path=checkpoint_path, device=device)
    return apply_matcher(
        matcher,
        source_image,
        destination_image,
        ransac_filter=kwargs.get("ransac_filter", True),
        ransac_threshold=kwargs.get("ransac_threshold", 0.05),
        ransac_method=kwargs.get("ransac_method", "deformable"),
        ransac_max_trials=kwargs.get("ransac_max_trials", 100),
    )
