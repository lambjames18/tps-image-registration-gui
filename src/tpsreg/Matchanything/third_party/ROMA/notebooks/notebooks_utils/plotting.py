import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import hsv_to_rgb
import pylab as pl
import matplotlib.cm as cm
import cv2


def visualize_features(feat, img_h, img_w, save_path=None):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3, svd_solver="arpack")
    img = pca.fit_transform(feat).reshape(img_h * 2, img_w, 3)
    img_norm = cv2.normalize(
        img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3
    )
    img_resized = cv2.resize(
        img_norm, (img_w * 8, img_h * 2 * 8), interpolation=cv2.INTER_LINEAR
    )
    img_colormap = img_resized
    img1, img2 = img_colormap[: img_h * 8, :, :], img_colormap[img_h * 8 :, :, :]
    img_gapped = np.hstack(
        (img1, np.ones((img_h * 8, 10, 3), dtype=np.uint8) * 255, img2)
    )
    if save_path is not None:
        cv2.imwrite(save_path, img_gapped)

    fig, axes = plt.subplots(1, 1, dpi=200)
    axes.imshow(img_gapped)
    axes.get_yaxis().set_ticks([])
    axes.get_xaxis().set_ticks([])
    plt.tight_layout(pad=0.5)
    return fig


def make_matching_figure(
    img0,
    img1,
    mkpts0,
    mkpts1,
    color,
    kpts0=None,
    kpts1=None,
    text=[],
    path=None,
    draw_detection=False,
    draw_match_type='corres', # ['color', 'corres', None]
    r_normalize_factor=0.4,
    white_center=True,
    vertical=False,
    use_position_color=False,
    draw_local_window=False,
    window_size=(9, 9),
    plot_size_factor=1, # Point size and line width
    anchor_pts0=None,
    anchor_pts1=None,
):
    # draw image pair
    fig, axes = (
        plt.subplots(2, 1, figsize=(10, 6), dpi=600)
        if vertical
        else plt.subplots(1, 2, figsize=(10, 6), dpi=600)
    )
    axes[0].imshow(img0)
    axes[1].imshow(img1)
    for i in range(2):  # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)

    if use_position_color:
        mean_coord = np.mean(mkpts0, axis=0)
        x_center, y_center = mean_coord
        # NOTE: set r_normalize_factor to a smaller number will make plotted figure more contrastive.
        position_color = matching_coord2color(
            mkpts0,
            x_center,
            y_center,
            r_normalize_factor=r_normalize_factor,
            white_center=white_center,
        )
        color[:, :3] = position_color

    if draw_detection and kpts0 is not None and kpts1 is not None:
        # color = 'g'
        color = 'r'
        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=1 * plot_size_factor)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=1 * plot_size_factor)

    if draw_match_type is 'corres':
        # draw matches
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        fig.lines = [
            matplotlib.lines.Line2D(
                (fkpts0[i, 0], fkpts1[i, 0]),
                (fkpts0[i, 1], fkpts1[i, 1]),
                transform=fig.transFigure,
                c=color[i],
                linewidth=1* plot_size_factor,
            )
            for i in range(len(mkpts0))
        ]

        axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color[:, :3], s=2* plot_size_factor)
        axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color[:, :3], s=2* plot_size_factor)
    elif draw_match_type is 'color':
        # x_center = img0.shape[-1] / 2
        # y_center = img1.shape[-2] / 2
        
        mean_coord = np.mean(mkpts0, axis=0)
        x_center, y_center = mean_coord
        # NOTE: set r_normalize_factor to a smaller number will make plotted figure more contrastive.
        kpts_color = matching_coord2color(
            mkpts0,
            x_center,
            y_center,
            r_normalize_factor=r_normalize_factor,
            white_center=white_center,
        )
        axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=kpts_color, s=1 * plot_size_factor)
        axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=kpts_color, s=1 * plot_size_factor)

    if draw_local_window:
        anchor_pts0 = mkpts0 if anchor_pts0 is None else anchor_pts0
        anchor_pts1 = mkpts1 if anchor_pts1 is None else anchor_pts1
        plot_local_windows(
            anchor_pts0, color=(1, 0, 0, 0.4), lw=0.2, ax_=0, window_size=window_size
        )
        plot_local_windows(
            anchor_pts1, color=(1, 0, 0, 0.4), lw=0.2, ax_=1, window_size=window_size
        )  # lw =0.2

    # put txts
    txt_color = "k" if img0[:100, :200].mean() > 200 else "w"
    fig.text(
        0.01,
        0.99,
        "\n".join(text),
        transform=fig.axes[0].transAxes,
        fontsize=15,
        va="top",
        ha="left",
        color=txt_color,
    )
    plt.tight_layout(pad=1)

    # save or return figure
    if path:
        plt.savefig(str(path), bbox_inches="tight", pad_inches=0)
        plt.close()
    else:
        return fig

def make_triple_matching_figure(
    img0,
    img1,
    img2,
    mkpts01,
    mkpts12,
    color01,
    color12,
    text=[],
    path=None,
    draw_match=True,
    r_normalize_factor=0.4,
    white_center=True,
    vertical=False,
    draw_local_window=False,
    window_size=(9, 9),
    anchor_pts0=None,
    anchor_pts1=None,
):
    # draw image pair
    fig, axes = (
        plt.subplots(3, 1, figsize=(10, 6), dpi=600)
        if vertical
        else plt.subplots(1, 3, figsize=(10, 6), dpi=600)
    )
    axes[0].imshow(img0)
    axes[1].imshow(img1)
    axes[2].imshow(img2)
    for i in range(3):  # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)

    if draw_match:
        # draw matches for [0,1]
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts01[0]))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts01[1]))
        fig.lines = [
            matplotlib.lines.Line2D(
                (fkpts0[i, 0], fkpts1[i, 0]),
                (fkpts0[i, 1], fkpts1[i, 1]),
                transform=fig.transFigure,
                c=color01[i],
                linewidth=1,
            )
            for i in range(len(mkpts01[0]))
        ]

        axes[0].scatter(mkpts01[0][:, 0], mkpts01[0][:, 1], c=color01[:, :3], s=1)
        axes[1].scatter(mkpts01[1][:, 0], mkpts01[1][:, 1], c=color01[:, :3], s=1)

        fig.canvas.draw()
        # draw matches for [1,2]
        fkpts1_1 = transFigure.transform(axes[1].transData.transform(mkpts12[0]))
        fkpts2 = transFigure.transform(axes[2].transData.transform(mkpts12[1]))
        fig.lines += [
            matplotlib.lines.Line2D(
                (fkpts1_1[i, 0], fkpts2[i, 0]),
                (fkpts1_1[i, 1], fkpts2[i, 1]),
                transform=fig.transFigure,
                c=color12[i],
                linewidth=1,
            )
            for i in range(len(mkpts12[0]))
        ]

        axes[1].scatter(mkpts12[0][:, 0], mkpts12[0][:, 1], c=color12[:, :3], s=1)
        axes[2].scatter(mkpts12[1][:, 0], mkpts12[1][:, 1], c=color12[:, :3], s=1)

    # # put txts
    # txt_color = "k" if img0[:100, :200].mean() > 200 else "w"
    # fig.text(
    #     0.01,
    #     0.99,
    #     "\n".join(text),
    #     transform=fig.axes[0].transAxes,
    #     fontsize=15,
    #     va="top",
    #     ha="left",
    #     color=txt_color,
    # )
    plt.tight_layout(pad=0.1)

    # save or return figure
    if path:
        plt.savefig(str(path), bbox_inches="tight", pad_inches=0)
        plt.close()
    else:
        return fig


def matching_coord2color(kpts, x_center, y_center, r_normalize_factor=0.4, white_center=True):
    """
        r_normalize_factor is used to visualize clearer according to points space distribution
        r_normalize_factor maxium=1, larger->points darker/brighter
    """
    if not white_center:
        # dark center points
        V, H = np.mgrid[0:1:10j, 0:1:360j]
        S = np.ones_like(V)
    else:
        # white center points
        S, H = np.mgrid[0:1:10j, 0:1:360j]
        V = np.ones_like(S)

    HSV = np.dstack((H, S, V))
    RGB = hsv_to_rgb(HSV)
    """
    # used to visualize hsv 
    pl.imshow(RGB, origin="lower", extent=[0, 360, 0, 1], aspect=150)
    pl.xlabel("H")
    pl.ylabel("S")
    pl.title("$V_{HSV}=1$")
    pl.show()
    """
    kpts = np.copy(kpts)
    distance = kpts - np.array([x_center, y_center])[None]
    r_max = np.percentile(np.linalg.norm(distance, axis=1), 85)
    # r_max = np.sqrt((x_center) ** 2 + (y_center) ** 2)
    kpts[:, 0] = kpts[:, 0] - x_center  # x
    kpts[:, 1] = kpts[:, 1] - y_center  # y

    r = np.sqrt(kpts[:, 0] ** 2 + kpts[:, 1] ** 2) + 1e-6
    r_normalized = r / (r_max * r_normalize_factor)
    r_normalized[r_normalized > 1] = 1
    r_normalized = (r_normalized) * 9

    cos_theta = kpts[:, 0] / r  # x / r
    theta = np.arccos(cos_theta)  # from 0 to pi
    change_angle_mask = kpts[:, 1] < 0
    theta[change_angle_mask] = 2 * np.pi - theta[change_angle_mask]
    theta_degree = np.degrees(theta)
    theta_degree[theta_degree == 360] = 0  # to avoid overflow
    theta_degree = theta_degree / 360 * 360
    kpts_color = RGB[r_normalized.astype(int), theta_degree.astype(int)]
    return kpts_color


def show_image_pair(img0, img1, path=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=200)
    axes[0].imshow(img0, cmap="gray")
    axes[1].imshow(img1, cmap="gray")
    for i in range(2):  # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)
    if path:
        plt.savefig(str(path), bbox_inches="tight", pad_inches=0)
    return fig

def plot_local_windows(kpts, color="r", lw=1, ax_=0, window_size=(9, 9)):
    ax = plt.gcf().axes
    for kpt in kpts:
        ax[ax_].add_patch(
            matplotlib.patches.Rectangle(
                (
                    kpt[0] - (window_size[0] // 2) - 1,
                    kpt[1] - (window_size[1] // 2) - 1,
                ),
                window_size[0] + 1,
                window_size[1] + 1,
                lw=lw,
                color=color,
                fill=False,
            )
        )

