import os
import matplotlib
from matplotlib import pyplot as plt

matplotlib.use("agg")
import matplotlib.cm as cm
import numpy as np
from PIL import Image
import cv2
from kornia.geometry.epipolar import numeric
import torch

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent.resolve()))

from src.utils.plotting import error_colormap, dynamic_alpha
from src.utils.metrics import symmetric_epipolar_distance
from notebooks.notebooks_utils import make_matching_figure


def plot_matches(
    img0_origin,
    img1_origin,
    mkpts0,
    mkpts1,
    mconf,
    vertical,
    draw_match_type,
    alpha,
    save_path,
    inverse=False,
    match_error=None,
    error_thr=5e-3,
    color_type="error",
    text=[""],
):
    if inverse:
        img0_origin, img1_origin, mkpts0, mkpts1 = (
            img1_origin,
            img0_origin,
            mkpts1,
            mkpts0,
        )
    img0_origin = np.copy(img0_origin) / 255.0
    img1_origin = np.copy(img1_origin) / 255.0
    # Draw
    alpha = dynamic_alpha(
        len(mkpts0),
        milestones=[0, 200, 500, 1000, 2000, 4000],
        alphas=[1.0, 0.5, 0.3, 0.2, 0.15, 0.09],
    )
    if color_type == "conf":
        color = error_colormap(mconf, thr=None, alpha=alpha)
    elif color_type == "green":
        mconf = np.ones_like(mconf) * 0.15
        color = error_colormap(mconf, thr=None, alpha=alpha)
    else:
        color = error_colormap(
            np.zeros((len(mconf),)) if match_error is None else match_error,
            error_thr,
            alpha=alpha,
        )

    text = [""]

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig = make_matching_figure(
        img0_origin,
        img1_origin,
        mkpts0,
        mkpts1,
        color,
        text=text,
        path=save_path,
        vertical=vertical,
        plot_size_factor=2.2,
        draw_match_type=draw_match_type,
        r_normalize_factor=0.30,
    )


def blend_img(img0, img1, alpha=0.4, save_path=None, blend_method="weighted_sum"):
    img0, img1 = Image.fromarray(np.array(img0)), Image.fromarray(np.array(img1))
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Blend:
    if blend_method == "weighted_sum":
        blended_img = Image.blend(img0, img1, alpha=alpha)
    else:
        raise NotImplementedError

    blended_img.save(save_path)


def warp_img(img0, img1, H, fill_white=False):
    img0 = np.copy(img0).astype(np.uint8)
    img1 = np.copy(img1).astype(np.uint8)
    if fill_white:
        img0_warped = cv2.warpAffine(
            np.array(img0),
            H[:2, :],
            [img1.shape[1], img1.shape[0]],
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=[255, 255, 255],
        )
    else:
        img0_warped = cv2.warpAffine(
            np.array(img0),
            H[:2, :],
            [img1.shape[1], img1.shape[0]],
            flags=cv2.INTER_LINEAR,
        )
    return img0_warped


def warp_img_and_blend(
    img0_origin, img1_origin, H, save_path, alpha=0.4, inverse=False
):
    if inverse:
        img0_origin, img1_origin = img1_origin, img0_origin
        H = np.linalg.inv(H)
    img0_origin = np.copy(img0_origin).astype(np.uint8)
    img1_origin = np.copy(img1_origin).astype(np.uint8)

    # Warp
    img0_warpped = Image.fromarray(
        warp_img(img0_origin, img1_origin, H, fill_white=False)
    )

    # Blend and save:
    blend_img(
        img0_warpped, Image.fromarray(img1_origin), alpha=alpha, save_path=save_path
    )


def epipolar_error(mkpts0, mkpts1, T_0to1, K0, K1):
    Tx = numeric.cross_product_matrix(torch.from_numpy(T_0to1)[:3, 3])
    E_mat = Tx @ T_0to1[:3, :3]
    return symmetric_epipolar_distance(
        torch.from_numpy(mkpts0),
        torch.from_numpy(mkpts1),
        E_mat,
        torch.from_numpy(K0),
        torch.from_numpy(K1),
    ).numpy()

def checkerboard_overlap(img0, img1, save_path, block_size=64):
    # Ensure same spatial dimensions
    if img0.shape[:2] != img1.shape[:2]:
        raise ValueError("Images must have same width and height")

    # Convert to uint8
    img0 = np.asarray(img0, dtype=np.uint8)
    img1 = np.asarray(img1, dtype=np.uint8)

    rows, cols = img0.shape[:2]

    # Compute row/col block indices
    row_blocks = np.arange(rows) // block_size
    col_blocks = np.arange(cols) // block_size

    # Vectorized checkerboard mask (outer addition)
    mask = (row_blocks[:, None] + col_blocks[None, :]) % 2 == 0

    # Expand mask to match channel count
    if img0.ndim == 3:
        mask = mask[:, :, None]

    # Combine
    overlapped = np.where(mask, img0, img1)

    # Ensure directory exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlapped).save(save_path)

import numpy as np
from PIL import Image

def checkerboard_transition(img0, img1, block_size=48, steps=20, save_path="transition.gif"):
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    img0 = np.asarray(img0, dtype=np.uint8)
    img1 = np.asarray(img1, dtype=np.uint8)

    rows, cols = img0.shape[:2]

    # Block grid
    row_blocks = np.arange(rows) // block_size
    col_blocks = np.arange(cols) // block_size
    checker = (row_blocks[:, None] + col_blocks[None, :]) % 2

    # Unique block IDs for animation ordering
    block_ids = checker.copy()
    # Example: animate white squares first, then black
    order = np.unique(block_ids)

    frames = []

    for t in np.linspace(0, 1, steps):
        # Compute which blocks have transitioned at step t
        threshold = t * len(order)
        active_blocks = order[:int(threshold)]  # blocks that should show img1

        # Build mask
        mask = np.isin(block_ids, active_blocks)
        if img0.ndim == 3:
            mask = mask[:, :, None]

        frame = np.where(mask, img1, img0).astype(np.uint8)
        frames.append(Image.fromarray(frame))

    frames[0].save(save_path, save_all=True, append_images=frames[1:], optimize=True, duration=50, loop=0)

def correspondence_query_plot(img_t, img_s, gt_matches, pred_matches=None, save_path=None, figsize=(20,12)):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    h = max(img_t.shape[0], img_s.shape[0])
    w = img_t.shape[1] + img_s.shape[1]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:img_t.shape[0], :img_t.shape[1]] = img_t
    canvas[:img_s.shape[0], img_t.shape[1]:img_t.shape[1] + img_s.shape[1]] = img_s
    x_offset = img_t.shape[1]
    
    #matches = sample["matches"].cpu().numpy()
    xt = gt_matches[:,0]
    yt = gt_matches[:,1]
    xs = gt_matches[:,2]
    ys = gt_matches[:,3]
    plt.figure(figsize=figsize)
    plt.imshow(canvas)
    plt.axis("off")
    for i in range(len(xt)):
        xs_shift = xs[i] + x_offset
        plt.plot([xs_shift, xt[i]], [ys[i], yt[i]], color="cyan", linewidth=0.8)
        plt.scatter(xt[i], yt[i], color="red", s=25)
        plt.scatter(xs_shift, ys[i], color="yellow", s=25)
    if pred_matches is not None:
        px = pred_matches[:,0]
        py = pred_matches[:,1]
        plt.scatter(px + x_offset, py, color="cyan", s=25)
        for i in range(len(px)):
            plt.plot([xs[i] + x_offset, px[i] + x_offset], [ys[i], py[i]], color="magenta", linewidth=0.8)
    plt.title("Matched Feature Visualization")
    plt.show()
    save_path = "correspondence_plot.png" if save_path is None else save_path
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()








