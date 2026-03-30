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
    "dataset_name": "AF9628-Martensitic_SEM-SE-Stitch_EBSD_SameSlice", #"All", #"In718SS-CHESS", #"C103_fracture_surfaces",  #"AMSpalledTa", #"DDRX_Dislocations_900C_largeFOV", #"DDRX_Dislocations_900C", #"DDRX_Dislocations_1100C", #"MAX_Phase_Dislocations", #"CoNi90-AM-DIC", #"DIC_EBSD_Multimodal_In718", #"TRIP1_steel_LOM_EBSD", #"In718_same_slice_BSE_EBSD", #"SE2_EBSD_X2CrNi12",# "Martensitic_steel_AF_9628_SEM_EBSD",#"5842WCu_Spalled", #"In718SS-CHESS", ##"CoNi90_mid_OM-2-BSE", #"CoNi67_mid_OM-2-high_OM", #"CoNi67_high_OM-2-SE", #"CP700BC_cracks_block5_pedestal1_0", #"AMSpalledTa", #"CoNi67", "Liver_CT-MR"
    "data_root": "data/test_data",
    "output_root": "results",
    "plot_matches": False,
    "plot_matches_alpha": 0.2,
    "plot_matches_color": "error",  # options: ['green', 'error', 'conf']
    "plot_align": False,
    "plot_refinement": False,
    "plot_checkerboard": False,
    "rigid_ransac_thr":40.0,
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
                img0_gt_pts = gt_2D_matches[:, :2]
                img1_gt_pts = gt_2D_matches[:, 2:]
        # Perform RANSAC filtering for tps, with larger hard error margin to remove outliers
                H_est = None
                inliers = None
                try:
                    if not cfg["RANSAC_correspondence_plane"]:
                        _, inliers = estimate_homo(
                            img0_gt_pts,
                            img1_gt_pts,
                            thresh=cfg["rigid_ransac_thr"],  # maybe adjust?
                            #conf=cfg["ransac_confidence"],
                            mode="homo",
                            #ransac_method=cfg["ransac_method"],
                            #ransac_maxIters=cfg["ransac_max_iters"],
                        )
                    if inliers is not None:
                        print(
                            f"[INFO] RANSAC inliers for {pair_name}: {int(inliers.sum())}/{len(img0_gt_pts)}"
                        )
        
                except Exception as exc:
                    print(f"[WARN] RANSAC failed for {pair_name}: {exc}")

                if inliers is not None:
                    mask = inliers.astype(bool).ravel()
                    img0_gt_pts_filtered = img0_gt_pts[mask]
                    img1_gt_pts_filtered = img1_gt_pts[mask]
                
        
        
        correspondence_query_plot(
                            np.copy(img0).astype(np.uint8), 
                            np.copy(img1).astype(np.uint8), 
                            gt_2D_matches, 
                            pred_matches=None,
                            save_path=Path(vis_output_path) / "gt_correspondence_plot" / f"{pair_name}_raw.png", 
                            figsize=(20,12)
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