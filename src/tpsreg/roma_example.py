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

from roma_matcher import create_matcher, apply_matcher


def select_points_subset(
    num_samples, shape, src_points, dst_points, confidences, selection_method
):
    # Take the top N matches based on confidence
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
                row_bins = np.linspace(0, shape[0], grid_size + 1)
                col_bins = np.linspace(0, shape[1], grid_size + 1)
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

    return src_points, dst_points, confidences


def example_visualization():
    """
    Visualize the detected matches.
    """
    print("\n" + "=" * 60)
    print("Visualizing Matches")
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

    matcher = create_matcher(checkpoint_path=None)
    src_points, dst_points, confidences = apply_matcher(
        matcher,
        source_image,
        dest_image,
        ransac_filter=True,
        ransac_threshold=0.05,
        ransac_method="deformable",
        ransac_max_trials=100,
    )

    src_points, dst_points, confidences = select_points_subset(
        num_samples=100,
        shape=source_image.shape,
        src_points=src_points,
        dst_points=dst_points,
        confidences=confidences,
        selection_method="grid",
    )

    print(f"Detected {len(src_points)} matches")

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

    example_visualization()

    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)
