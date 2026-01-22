"""
roma_example.py - Example usage of the ROMA matcher

This script demonstrates how to use the ROMA matcher for automatic
control point detection between image pairs.
"""

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
from skimage import io, transform
import matplotlib.pyplot as plt
from pathlib import Path

from roma_matcher import detect_points_matchanything


def example_visualization():
    """
    Example: Visualize the detected matches.
    """
    print("\n" + "=" * 60)
    print("Example 4: Visualizing Matches")
    print("=" * 60)

    # Create synthetic test images with known transformation
    dest_image = io.imread(
        "D:/Research/Datasets-Projects/MatchAnything_Datasets/CoNi-AM67_SEM-EBSD_SameSliceSerialSectioning/EBSD_000_IQ.tiff"
    )
    source_image = io.imread(
        "D:/Research/Datasets-Projects/MatchAnything_Datasets/CoNi-AM67_SEM-EBSD_SameSliceSerialSectioning/BSE_000.tif"
    )
    dest_image = np.dstack([dest_image] * 3)
    source_image = np.dstack([source_image] * 3)

    print(f"Source image shape: {source_image.shape}")
    print(f"Destination image shape: {dest_image.shape}")

    src_points, dst_points, confidences = detect_points_matchanything(
        source_image,
        dest_image,
        checkpoint_path=None,
    )

    print(f"Detected {len(src_points)} matches")

    if len(src_points) > 100:
        src_points = src_points[::100]
        dst_points = dst_points[::100]
        confidences = confidences[::100]
    # Visualize matches
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Source image with points
    axes[0].imshow(source_image, cmap="gray")
    for i in range(len(src_points)):
        axes[0].scatter(
            src_points[i, 0],
            src_points[i, 1],
            c="red",
            s=500 * confidences[i],
            alpha=0.5,
            marker=r"${}$".format(i),
        )
    # axes[0].scatter(src_points[:, 0], src_points[:, 1], c="red", s=10, alpha=0.5)
    axes[0].set_title(f"Source Image ({len(src_points)} points)")
    axes[0].axis("off")

    # Destination image with points
    axes[1].imshow(dest_image, cmap="gray")
    for i in range(len(dst_points)):
        axes[1].scatter(
            dst_points[i, 0],
            dst_points[i, 1],
            c="blue",
            s=200 * confidences[i],
            alpha=0.5,
            marker=r"${}$".format(i),
        )
    # axes[1].scatter(dst_points[:, 0], dst_points[:, 1], c="blue", s=10, alpha=0.5)
    axes[1].set_title(f"Destination Image ({len(dst_points)} points)")
    axes[1].axis("off")

    plt.tight_layout()
    output_path = Path("roma_matches_visualization.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Visualization saved to: {output_path}")


if __name__ == "__main__":
    print("\nROMA Matcher Examples")
    print("=" * 60)
    print("Note: These examples use dummy data for demonstration.")
    print("Replace with actual image loading for real use cases.")
    print("=" * 60)

    # Run examples
    # try:
    #     example_basic_usage()
    # except Exception as e:
    #     print(f"Example 1 failed: {e}")

    # try:
    #     example_preset_config()
    # except Exception as e:
    #     print(f"Example 2 failed: {e}")

    # try:
    #     example_with_metadata()
    # except Exception as e:
    #     print(f"Example 3 failed: {e}")

    example_visualization()

    # example_integration_with_gui()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
