import pytorch_lightning as pl
from tqdm import tqdm
import os.path as osp
import numpy as np
import subprocess
import pandas as pd
from loguru import logger
from PIL import Image
from dataset_registry import dataset_list

Image.MAX_IMAGE_PIXELS = None
import torch

from torch.utils.data import DataLoader, ConcatDataset

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.resolve()))

from src.lightning.lightning_loftr import PL_LoFTR
from src.config.default import get_cfg_defaults
from src.utils.dataset import dict_to_cuda
from src.utils.metrics import estimate_homo, ransac_correspondence_plane, estimate_pose, relative_pose_error
from src.utils.homography_utils import warp_points

from src.datasets.common_data_pair import CommonDataset
from src.utils.metrics import error_auc
from tools_utils.plot import checkerboard_overlap, checkerboard_transition, correspondence_query_plot, plot_matches, warp_img_and_blend, blend_img, epipolar_error
from tools_utils.data_io import save_h5

# The warping function which supports Thin plate splines transform
from warping import transform_image, get_transform
from skimage import transform as tf


import os
from pathlib import Path

import numpy as np
from PIL import Image

import os
from pathlib import Path

import numpy as np
from PIL import Image


def _save_image_auto_dtype(arr: np.ndarray, path: str):
    """
    Save a HxWxC image array (float or uint) as a standard 8-bit image.

    - Floats in [0, 1] → scaled to [0, 255]
    - Floats in [0, 255] → clipped and cast
    - Other dtypes → clipped/cast to uint8
    """
    img = arr

    if np.issubdtype(img.dtype, np.floating):
        img = np.nan_to_num(img)
        vmin = float(img.min())
        vmax = float(img.max())

        if 0.0 <= vmin and vmax <= 1.0 + 1e-3:
            # Float image likely in [0, 1]
            img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            # Your case: float in [0, 255]
            img = np.clip(img, 0.0, 255.0).astype(np.uint8)
    else:
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

    Image.fromarray(img).save(path)


def crop_expand_until_full_image(
    img1: np.ndarray,
    corner_points_target_warped: np.ndarray,
    gt_points: np.ndarray,
    base_filename: str,
    out_dir: str,
    step: int = 10,
):
    """
    Expand the rectangular hull of corner_points_target_warped by 'step' pixels per iteration,
    clamping to image borders per side. Continue until the crop becomes the full image.
    Save each crop and corresponding keypoints (with appropriate offsets).

    - First crop uses pad = 0 (just the bounding box).
    - Next crops: pad = 10, 20, 30, ... (or whatever 'step' is).
    - Expansion is per-side, but each side is clamped to [0, width] / [0, height].
    - Last saved crop is the full original image.

    Filenames:
      <root>+0px.<ext>, <root>+10px.<ext>, ...

    Keypoints:
      Saved as:
        <root>+0px_keypoints.npy
        <root>+0px_keypoints.txt
        etc.
    """

    img_h, img_w = img1.shape[:2]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Initial axis-aligned bounding box of the 4 corner points ---
    pts = np.asarray(corner_points_target_warped)
    xs = pts[:, 0]
    ys = pts[:, 1]

    xmin = int(np.floor(xs.min()))
    xmax = int(np.ceil(xs.max()))
    ymin = int(np.floor(ys.min()))
    ymax = int(np.ceil(ys.max()))

    # Clamp to image bounds
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(img_w, xmax)
    ymax = min(img_h, ymax)

    # --- 2. Split base filename into root + extension ---
    root, ext = os.path.splitext(base_filename)
    if ext == "":
        ext = ".png"

    pad = 0
    while True:
        # --- 3. Expand and clamp per side ---
        xmin_pad = max(0, xmin - pad)
        ymin_pad = max(0, ymin - pad)
        xmax_pad = min(img_w, xmax + pad)
        ymax_pad = min(img_h, ymax + pad)

        # Sanity check: if box is empty, break
        if xmin_pad >= xmax_pad or ymin_pad >= ymax_pad:
            break

        # --- 4. Crop the image ---
        crop = img1[ymin_pad:ymax_pad, xmin_pad:xmax_pad].copy()

        # --- 5. Adjust keypoints: subtract current crop's top-left ---
        gt_crop = gt_points.astype(np.float32).copy()
        gt_crop[:, 0] -= xmin_pad  # x offset
        gt_crop[:, 1] -= ymin_pad  # y offset

        # --- 6. Save image and keypoints ---
        img_name = f"{root}+{pad}px{ext}"
        img_path = out_dir / img_name

        _save_image_auto_dtype(crop, img_path)

        # Numpy binary
        kp_npy_path = out_dir / f"{root}+{pad}px_keypoints.npy"
        np.save(kp_npy_path, gt_crop)

        # CSV-style text
        kp_txt_path = out_dir / f"{root}+{pad}px_keypoints.txt"
        np.savetxt(kp_txt_path, gt_crop, fmt="%.3f", delimiter=",")

        print(f"Saved {img_path.name}, keypoints: {kp_npy_path.name}, {kp_txt_path.name}")

        # --- 7. Stop if the crop is now the full image ---
        if (
            xmin_pad == 0
            and ymin_pad == 0
            and xmax_pad == img_w
            and ymax_pad == img_h
        ):
            # This crop is the original image; we're done.
            break

        # Otherwise, increase global padding and continue.
        pad += step


# def save_crop(crop: np.ndarray, path: str):
#     # Handle float images
#     if np.issubdtype(crop.dtype, np.floating):
#         # Replace NaNs/Infs just in case
#         crop = np.nan_to_num(crop)

#         # Check value range to decide how to scale
#         vmin = float(crop.min())
#         vmax = float(crop.max())

#         if vmax <= 1.0 + 1e-3 and vmin >= 0.0:
#             # Likely 0–1, scale to 0–255
#             crop_to_save = (np.clip(crop, 0.0, 1.0) * 255.0).astype(np.uint8)
#         else:
#             # Already 0–255 (your case), just clip and cast
#             crop_to_save = np.clip(crop, 0.0, 255.0).astype(np.uint8)
#     else:
#         # Non-float images: ensure valid range for uint8
#         crop_to_save = crop
#         if crop_to_save.dtype != np.uint8:
#             crop_to_save = np.clip(crop_to_save, 0, 255).astype(np.uint8)

#     Image.fromarray(crop_to_save).save(path)


def crop_expanding_region(
    img1: np.ndarray,
    corner_points_target_warped: np.ndarray,
    gt_points: np.ndarray,
    base_filename: str,
    out_dir: str,
    step: int = 10,
):
    """
    img1: HxWx3 numpy array (original image)
    corner_points_target_warped: 4x2 numpy array of corner coordinates (x, y)
    gt_points: Nx2 numpy array of keypoints (x, y) in original image coords,
               all inside the region of interest
    base_filename: e.g. "my_region.png" (used to build output names)
    out_dir: folder to save crops and keypoints
    step: how much to expand per iteration (in pixels per side)
    """

    img_h, img_w = img1.shape[:2]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Get initial axis-aligned bounding box of the 4 corner points ---
    pts = np.asarray(corner_points_target_warped)
    xs = pts[:, 0]
    ys = pts[:, 1]

    # Use floor/ceil to ensure we fully contain the region
    xmin = int(np.floor(xs.min()))
    xmax = int(np.ceil(xs.max()))
    ymin = int(np.floor(ys.min()))
    ymax = int(np.ceil(ys.max()))

    # Clamp initial box to image bounds just in case
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(img_w, xmax)   # note: xmax, ymax will be used as exclusive indices
    ymax = min(img_h, ymax)

    # --- 2. Parse base name + extension so we can insert “+10px” before suffix ---
    root, ext = os.path.splitext(base_filename)
    if ext == "":
        # default extension if user didn't pass one
        ext = ".png"

    pad = 0
    while True:
        # Expanded bounding box (still using [ymin:ymax, xmin:xmax] slicing)
        xmin_pad = xmin - pad
        ymin_pad = ymin - pad
        xmax_pad = xmax + pad
        ymax_pad = ymax + pad

        # Check if this expanded box still lies fully within the image
        # xmax_pad and ymax_pad are exclusive indices so we allow equality with width/height
        if xmin_pad < 0 or ymin_pad < 0 or xmax_pad > img_w or ymax_pad > img_h:
            break

        # --- 3. Crop the image ---
        crop = img1[ymin_pad:ymax_pad, xmin_pad:xmax_pad].copy()

        # --- 4. Adjust keypoints: subtract the top-left corner of the crop ---
        gt_crop = gt_points.copy().astype(np.float32)
        gt_crop[:, 0] -= xmin_pad  # x offset
        gt_crop[:, 1] -= ymin_pad  # y offset

        # --- 5. Save image and keypoints ---
        # e.g. "my_region+0px.png", "my_region+10px.png", ...
        img_name = f"{root}+{pad}px{ext}"
        img_path = out_dir / img_name

        # Save image
        _save_image_auto_dtype(crop, img_path)

        # Save keypoints in a couple of simple formats
        # 1) Numpy binary
        kp_npy_path = out_dir / f"{root}+{pad}px_keypoints.npy"
        np.save(kp_npy_path, gt_crop)

        # 2) CSV-style text (x,y per line)
        kp_txt_path = out_dir / f"{root}+{pad}px_keypoints.txt"
        np.savetxt(kp_txt_path, gt_crop, fmt="%.3f", delimiter=",")

        print(f"Saved {img_path.name}, keypoints: {kp_npy_path.name}, {kp_txt_path.name}")

        # --- 6. Increase padding and repeat ---
        pad += step


CONFIG = {
    "eval_dataset": True,
    "main_cfg_path": "configs/models/roma_model.py", #"configs/models/eloftr_model.py", "configs/models/roma_model.py",  # Required, replace with actual path
    "ckpt_path": "weights/matchanything_roma.ckpt", #"weights/matchanything_eloftr.ckpt", # "weights/matchanything_roma.ckpt", #"weights/minima_roma.pth", , "weights/minima_roma.pth"
    "thr": 0.1,
    "method": "matchanything_roma", #"matchanything_eloftr", #"minima_roma", # "SIFT"
    "transformation_type": "tps", #"affine",  # "homo",
    "imgresize": 832,
    "divisible_by": 32, # this factor is utilized in the image loading pipeline to ensure the image sizes are divisible by this number, for ELoFTR 32 is be necessary, does not affect the RoMa loading pipeline
    "npe": True,
    "npe2": False,
    "ckpt32": False,
    "fp32": False,
    "dataset_name": "5842WCu-Spalled_SEM-SE_SEM-BSE_Multiscale", #"In718SS-CHESS", #"C103_fracture_surfaces",  #"AMSpalledTa", #"DDRX_Dislocations_900C_largeFOV", #"DDRX_Dislocations_900C", #"DDRX_Dislocations_1100C", #"MAX_Phase_Dislocations", #"CoNi90-AM-DIC", #"DIC_EBSD_Multimodal_In718", #"TRIP1_steel_LOM_EBSD", #"In718_same_slice_BSE_EBSD", #"SE2_EBSD_X2CrNi12",# "Martensitic_steel_AF_9628_SEM_EBSD",#"5842WCu_Spalled", #"In718SS-CHESS", ##"CoNi90_mid_OM-2-BSE", #"CoNi67_mid_OM-2-high_OM", #"CoNi67_high_OM-2-SE", #"CP700BC_cracks_block5_pedestal1_0", #"AMSpalledTa", #"CoNi67", "Liver_CT-MR"
    "data_root": "data/test_data",
    "output_root": "results",
    "plot_matches": False,
    "plot_matches_alpha": 0.2,
    "plot_matches_color": "error",  # options: ['green', 'error', 'conf']
    "plot_align": False,
    "plot_refinement": False,
    "plot_checkerboard": False,
    "rigid_ransac_thr": 5.5, #10.0,
    "elastix_ransac_thr": 40.0,
    "normalize_img": True,
    "RANSAC_correspondence_plane": False,
    "comment": "",
}


def array_rgb2gray(img):
    return (img * np.array([0.2989, 0.5870, 0.1140])[None, None]).sum(axis=-1)


def run_pipeline(config):
    cfg = config.copy()
    # Load data:
    datasets = []
    cfg["npz_root"] = cfg["data_root"] + "/" + cfg["dataset_name"] + "/" + "eval_indexs"
    cfg["npz_list_path"] = cfg["npz_root"] + "/" + "val_list.txt"
    cfg["output_path"] = cfg["output_root"] + "/" + cfg["dataset_name"] + "_" + cfg["method"]

    # fetch git commit hash and add to cfg
    cmd = ["git", "rev-parse", "HEAD"]
    cfg["git_hash"] = subprocess.check_output(cmd).decode("utf-8").strip()


    # load the text file which references all scenes to evaluate/inference
    with open(cfg["npz_list_path"], "r") as f:
        npz_names = [name.split()[0] for name in f.readlines()]
    npz_names = [f"{n}.npz" for n in npz_names]
    data_root = cfg["data_root"]

    # print(f"npz_names is {npz_names}.")
    # print(f"data_root is {data_root}.")
    # print(f"npz_root is {cfg['npz_root']}.")
    # print(f"npz List Path: {cfg['npz_list_path']}")
    print(f"[INFO] The Ransac Threshold is: {cfg['rigid_ransac_thr']}")

    vis_output_path = cfg["output_path"]
    Path(vis_output_path).mkdir(parents=True, exist_ok=True)
    (Path(vis_output_path) / "npz").mkdir(parents=True, exist_ok=True)

    # save the config used
    with open(cfg['output_path'] + "/" + f"config_dict_{cfg['dataset_name']}_{cfg['method']}_{cfg['comment']}.txt", "w") as f:
        f.write(str(cfg))
    
    ##########################
    config = get_cfg_defaults()
    # method, estimator = (cfg["method"]).split("@-@")[0], (cfg["method"]).split("@-@")[1]
    if cfg["method"] != "None" and cfg["method"] != "SIFT":
        config.merge_from_file(cfg["main_cfg_path"])

        pl.seed_everything(config.TRAINER.SEED)
        config.METHOD = cfg["method"]
        print(
            f"Method: {cfg['method']} with transformation: {cfg['transformation_type']}"
        )
        # Config overwrite:
        if config.LOFTR.COARSE.ROPE:
            assert config.DATASET.NPE_NAME is not None
        if config.DATASET.NPE_NAME is not None:
            config.LOFTR.COARSE.NPE = [832, 832, cfg["imgresize"], cfg["imgresize"]]

        if "visible_sar" in cfg["npz_list_path"]:
            config.DATASET.RESIZE_BY_STRETCH = True

        if cfg["thr"] is not None:
            config.LOFTR.MATCH_COARSE.THR = cfg["thr"]

        matcher = PL_LoFTR(
            config, pretrained_ckpt=cfg["ckpt_path"], test_mode=True
        ).matcher
        matcher.eval().cuda()
    elif cfg["method"] == "SIFT":
        matcher = "SIFT"
    else:
        matcher = None

    for npz_name in tqdm(npz_names):
        npz_path = osp.join(cfg["npz_root"], npz_name)
        try:
            np.load(npz_path, allow_pickle=True)
        except:
            logger.info(f"{npz_path} cannot be opened!")
            continue

        # print(f"npz_names is {npz_names}.")
        # print(f"data_root is {data_root}.")
        # print(f"npz_root is {args.npz_root}.")
        # print(f"npz List Path: {args.npz_list_path}")

        datasets.append(
            CommonDataset(
                data_root,
                npz_path,
                mode="test",
                min_overlap_score=-1,
                img_resize=cfg["imgresize"],
                df=cfg["divisible_by"] if "divisible_by" in cfg else None,
                img_padding=False,
                depth_padding=True,
                testNpairs=None,
                fp16=False,
                load_origin_rgb=True,
                read_gray=True,
                normalize_img=cfg["normalize_img"] if "normalize_img" in cfg else False,
                resize_by_stretch=config.DATASET.RESIZE_BY_STRETCH,
                gt_matches_padding_n=100,
                dataset_name=cfg["dataset_name"],
                #auto_contrast_stretch=cfg["auto_contrast_stretch"] if "auto_contrast_stretch" in cfg else False,
            )
        )

    concat_dataset = ConcatDataset(datasets)

    dataloader = DataLoader(
        concat_dataset, num_workers=0, pin_memory=True, batch_size=1, drop_last=False
    )
    errors = []  # distance
    result_dict = {}
    results_df = pd.DataFrame(columns=["SceneID", "Registration Target", "Registration Source", "Mean Euclidean Error", "Max Euclidean Error", "Mean Euclidean Error [m]", "Max Euclidean Error [m]", "Num GT Matches Used", "Num Matches Raw", "Num Matches RANSAC Inliers"])
    pose_error = []

    eval_mode = "gt_homo"
    for id, data in enumerate(tqdm(dataloader)):
        img0, img1 = (data["image0_rgb_origin"] * 255.0)[0].permute(
            1, 2, 0
        ).numpy().round().squeeze(), (data["image1_rgb_origin"] * 255.0)[0].permute(
            1, 2, 0
        ).numpy().round().squeeze()
        img_1_h, img_1_w = img1.shape[:2]
        pair_name = "@-@".join(
            [
                data["pair_names"][0][0].split("/", 1)[1],
                data["pair_names"][1][0].split("/", 1)[1],
            ]
        ).replace("/", "_")
        if cfg["eval_dataset"]:
            homography_gt = data["homography"][0].numpy()
            if "gt_2D_matches" in data and data["gt_2D_matches"].shape[-1] == 4:
                gt_2D_matches = data["gt_2D_matches"][0].numpy()  # N * 4
                eval_coord = gt_2D_matches[:, :2]
                gt_points = gt_2D_matches[:, 2:]
                # ransac_mode = "homo" if "FIRE" in cfg["npz_list_path"] else "affine"
        
        h, w = img0.shape[0], img0.shape[1]
        corner_points_target = np.array([[0, 0], [0, h-1], [w-1, 0], [w-1, h-1]])

        reference_img_shape = tuple(
                int(v) for v in data["origin_img_size0"].squeeze(0)
            )

        H_est = get_transform(
            gt_points,
            eval_coord,
            mode="tps",
            size=reference_img_shape,
        )
        
        # if evaluation is done, transform points
        if cfg["eval_dataset"] and "gt_2D_matches" in data and data["gt_2D_matches"].shape[-1] == 4:
            corner_points_target_warpped = H_est(corner_points_target)  

        print(corner_points_target_warpped)      
        # img1: HxWx3 uint8 array
        # corner_points_target_warped: (4, 2) array of floats
        # gt_points: (N, 2) array
        crop_expand_until_full_image(
            img1,
            corner_points_target_warpped,
            gt_points,
            base_filename="roi.tif",
            out_dir=cfg['output_path'] + "/" + "crops",
            step=10,
        )


if __name__ == "__main__":
    if CONFIG["dataset_name"] == "All":
        # run on all datasets within the registry
        for dataset_name in tqdm(dataset_list):
            print(f"Running evaluation/inference on dataset: {dataset_name}")
            CONFIG["dataset_name"] = dataset_name
            run_pipeline(CONFIG)
    elif isinstance(CONFIG["dataset_name"], list):
        # run on a specified list of datasets
        for dataset_name in CONFIG["dataset_name"]:
            if dataset_name in dataset_list:
                print(f"Running evaluation/inference on dataset: {dataset_name}")
                CONFIG["dataset_name"] = dataset_name
                run_pipeline(CONFIG)
            else:
                print(f"Dataset {dataset_name} not found in dataset registry.")
    else:
        # run on a single specified dataset
        if CONFIG["dataset_name"] in dataset_list:
            print(f"Running evaluation/inference on dataset: {CONFIG['dataset_name']}")
            run_pipeline(CONFIG)
        else:
            print(f"Dataset {CONFIG['dataset_name']} not found in dataset registry.")