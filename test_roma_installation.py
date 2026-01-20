"""
test_roma_installation.py - Verify ROMA installation and setup

This script tests the ROMA integration to ensure all dependencies
are correctly installed and the model can be loaded.
"""

import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def test_imports():
    """Test if all required modules can be imported."""
    print("\n" + "=" * 60)
    print("Testing Imports")
    print("=" * 60)

    required_modules = [
        ("numpy", "NumPy"),
        ("torch", "PyTorch"),
        ("PIL", "Pillow"),
        ("skimage", "scikit-image"),
        ("kornia", "Kornia"),
        ("einops", "Einops"),
    ]

    all_passed = True
    for module_name, display_name in required_modules:
        try:
            __import__(module_name)
            print(f"✓ {display_name:20s} - OK")
        except ImportError as e:
            print(f"✗ {display_name:20s} - FAILED: {e}")
            all_passed = False

    return all_passed


def test_cuda():
    """Test if CUDA is available."""
    print("\n" + "=" * 60)
    print("Testing CUDA")
    print("=" * 60)

    try:
        import torch

        if torch.cuda.is_available():
            print(f"✓ CUDA is available")
            print(f"  Device count: {torch.cuda.device_count()}")
            print(f"  Device name: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠ CUDA not available - will use CPU (slower)")
            return False

    except Exception as e:
        print(f"✗ CUDA check failed: {e}")
        return False


def test_checkpoint():
    """Test if the checkpoint file exists."""
    print("\n" + "=" * 60)
    print("Testing Checkpoint")
    print("=" * 60)

    checkpoint_path = Path("./matchanything_roma.ckpt")

    if checkpoint_path.exists():
        size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        print(f"✓ Checkpoint found: {checkpoint_path}")
        print(f"  Size: {size_mb:.1f} MB")
        return True
    else:
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        print("\nPlease download the checkpoint from:")
        print("https://drive.google.com/file/d/12L3g9-w8rR9K2L4rYaGaDJ7NqX1D713d/view")
        print(f"\nPlace it in: {checkpoint_path.absolute()}")
        return False


def test_roma_matcher():
    """Test if RomaMatcher can be imported."""
    print("\n" + "=" * 60)
    print("Testing RomaMatcher Import")
    print("=" * 60)

    try:
        from roma_matcher import RomaMatcher

        print("OK RomaMatcher imported successfully")
        return True
    except ImportError as e:
        if "torch" in str(e).lower():
            print("WARN RomaMatcher requires torch (install with: pip install torch)")
            return None  # Not a failure, just missing dependency
        else:
            print(f"ERROR Failed to import RomaMatcher: {e}")
            return False
    except Exception as e:
        print(f"ERROR Failed to import RomaMatcher: {e}")
        return False


def test_roma_config():
    """Test if roma_config can be imported."""
    print("\n" + "=" * 60)
    print("Testing Configuration")
    print("=" * 60)

    try:
        from roma_config import (
            get_roma_config,
            get_preset_config,
            AVAILABLE_PRESETS,
        )

        print("✓ Configuration module imported successfully")
        print(f"  Available presets: {', '.join(AVAILABLE_PRESETS.keys())}")

        # Test getting a preset
        ebsd_config = get_preset_config("ebsd")
        print(f"✓ EBSD preset loaded successfully")
        print(f"  Confidence threshold: {ebsd_config['match_thresh']}")
        print(f"  Num samples: {ebsd_config['sample']['n_sample']}")

        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def test_models_integration():
    """Test if models.py has been updated with ROMA integration."""
    print("\n" + "=" * 60)
    print("Testing models.py Integration")
    print("=" * 60)

    try:
        from models import PointAutoIdentifier

        engines = PointAutoIdentifier.ENGINES

        if "matchanything" in engines or "roma" in engines:
            print("✓ ROMA engine registered in PointAutoIdentifier")
            print(f"  Available engines: {', '.join(engines.keys())}")
            return True
        else:
            print("⚠ ROMA engine not found in PointAutoIdentifier")
            print(f"  Available engines: {', '.join(engines.keys())}")
            return False

    except Exception as e:
        print(f"✗ Failed to test models.py integration: {e}")
        return False


def test_model_loading():
    """Test if the model can actually be loaded (full integration test)."""
    print("\n" + "=" * 60)
    print("Testing Model Loading (Full Integration)")
    print("=" * 60)

    try:
        from roma_matcher import RomaMatcher
        import torch
        import numpy as np

        # Check if checkpoint exists (try both old and new locations)
        checkpoint_path = Path("./matchanything_roma.ckpt")

        if not checkpoint_path.exists():
            print("WARN Skipping model loading test - checkpoint not found")
            print(
                "  Download from: https://drive.google.com/file/d/12L3g9-w8rR9K2L4rYaGaDJ7NqX1D713d/view"
            )
            return None

        print("Loading ROMA model (this may take a moment)...")

        # Try to create matcher
        device = "cuda" if torch.cuda.is_available() else "cpu"
        matcher = RomaMatcher(
            checkpoint_path=str(checkpoint_path),
            confidence_threshold=0.1,
            device=device,
        )

        print(f"OK Model loaded successfully on {device}")

        # Test with dummy data
        print("Testing inference with dummy data...")
        dummy_src = np.random.rand(256, 256).astype(np.float32)
        dummy_dst = np.random.rand(256, 256).astype(np.float32)

        src_pts, dst_pts, conf = matcher.detect_points(dummy_src, dummy_dst)

        print(f"OK Inference test passed")
        print(f"  Detected {len(src_pts)} matches (dummy data, may be 0)")

        return True

    except ImportError as e:
        if "torch" in str(e).lower():
            print("WARN Model loading requires torch (install with: pip install torch)")
            return None
        else:
            print(f"ERROR Model loading failed - import error: {e}")
            return False
    except FileNotFoundError as e:
        print(f"WARN Model loading skipped - checkpoint not found")
        return None
    except Exception as e:
        print(f"ERROR Model loading failed: {e}")
        import traceback

        print("\nFull error traceback:")
        print(traceback.format_exc())
        return False


def run_all_tests():
    """Run all tests and provide summary."""
    print("\n" + "=" * 60)
    print("ROMA Installation Test Suite")
    print("=" * 60)

    results = {
        "Imports": test_imports(),
        "CUDA": test_cuda(),
        "Checkpoint File": test_checkpoint(),
        "RomaMatcher Import": test_roma_matcher(),
        "Configuration": test_roma_config(),
        "models.py Integration": test_models_integration(),
        "Model Loading": test_model_loading(),
    }

    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = 0
    failed = 0
    skipped = 0

    for test_name, result in results.items():
        if result is True:
            status = "OK PASSED"
            passed += 1
        elif result is False:
            status = "ERROR FAILED"
            failed += 1
        else:
            status = "WARN SKIPPED"
            skipped += 1

        print(f"{test_name:30s} {status}")

    print("=" * 60)
    print(f"Total: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 60)

    # Provide recommendations
    print("\nRecommendations:")

    if failed == 0 and passed > 0:
        print("OK All critical tests passed! ROMA is ready to use.")
        if skipped > 0:
            print(
                f"  ({skipped} optional tests skipped - usually due to missing torch)"
            )
    elif failed > 0:
        print("ERROR Some tests failed. Please address the following:")
    else:
        print("WARN Most tests were skipped. Please address the following:")

        if not results["Imports"]:
            print("\n1. Install missing Python packages:")
            print(
                "   pip install torch torchvision kornia einops loguru pillow scikit-image scipy"
            )

        if not results["CUDA"]:
            print("\n2. CUDA not available:")
            print("   - For GPU acceleration, install PyTorch with CUDA support")
            print("   - Or use CPU mode (slower but functional)")

        if not results["Checkpoint File"]:
            print("\n3. Download the model checkpoint:")
            print(
                "   https://drive.google.com/file/d/12L3g9-w8rR9K2L4rYaGaDJ7NqX1D713d/view"
            )

        if not results["MatchAnything Directory"]:
            print("\n4. MatchAnything directory structure missing:")
            print("   - Ensure MatchAnything folder is in the correct location")
            print("   - Check that all subdirectories are present")

        if results["Model Loading"] is False:
            print("\n5. Model loading failed:")
            print("   - Check the error traceback above")
            print("   - Ensure all dependencies are installed")
            print("   - Verify checkpoint file integrity")

    print("\nFor more information, see: ROMA_INTEGRATION.md")


if __name__ == "__main__":
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user.")
    except Exception as e:
        print(f"\n\nUnexpected error during testing: {e}")
        import traceback

        print(traceback.format_exc())
