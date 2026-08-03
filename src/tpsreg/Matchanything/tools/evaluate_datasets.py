import pytorch_lightning as pl
from tqdm import tqdm
import numpy as np
import cv2
from pathlib import Path
from loguru import logger
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
import torch
from torch.utils.data import DataLoader, ConcatDataset

import sys

sys.path.append(str(Path(__file__).parent.parent.resolve()))

from src.lightning.lightning_loftr import PL_LoFTR
from src.config.default import get_cfg_defaults
from src.utils.dataset import dict_to_cuda
from src.utils.metrics import (
    estimate_homo,
    estimate_pose,
    relative_pose_error,
    error_auc,
)
from src.utils.homography_utils import warp_points
from src.datasets.common_data_pair import CommonDataset
from tools_utils.plot import plot_matches, warp_img_and_blend, epipolar_error
from tools_utils.data_io import save_h5

# Update the values below instead of using argparse.
CONFIG = {
    "run_evaluation": True,
    "dataset_name": "Liver_CT-MR",  # "DIC_EBSD_Multimodal_In718",
    "method": "matchanything_roma@-@ransac_affine",
    "main_cfg_path": Path("./configs/models/roma_model.py"),
    "ckpt_path": Path("./weights/matchanything_roma.ckpt"),
    "data_root": Path("./data/test_data"),
    # "npz_root": None, #Path("./data/test_data/{CONFIG['dataset_name']}/npz"),
    # "npz_list_path": None, #Path(".data/test_data/Liver_CT-MR/eval_indexs/val_list.txt"),
    "pairs_txt_path": None,
    "output_root": Path("./results"),
    "imgresize": 832, # adopted from eval scripts
    "match_threshold": 0.1,
    "max_features": 5000,
    "ransac_mode": "homo",
    "ransac_method": "USAC_MAGSAC",
    "ransac_threshold": 3.0,
    "ransac_confidence": 0.999,
    "ransac_max_iters": 10000,
    "auto_contrast_stretch": False,
    "load_origin_rgb": True,
    "read_gray": True,
    "normalize_img": False,
    "gt_matches_padding_n": 100,
    "comment": "",
    "plot_matches": False,
    "plot_matches_alpha": 0.2,
    "plot_matches_color": "error",
    "plot_matches_filtered": False,
    "plot_align": False,
    "batch_size": 1,
    "num_workers": 4,
}


PATH_KEYS = {
    "main_cfg_path",
    "ckpt_path",
    "data_root",
    "npz_root",
    "npz_list_path",
    "pairs_txt_path",
    "output_root",
}


def _normalize_pair_component(path_str: str) -> str:
    posix = path_str.replace("\\", "/")
    if "/" in posix:
        posix = posix.split("/", 1)[1]
    return posix.replace("/", "_")


def _format_pair_key(name0: str, name1: str) -> str:
    return "@-@".join(
        [_normalize_pair_component(name0), _normalize_pair_component(name1)]
    )

def _resolve_ransac_method(name: str):
    if not hasattr(cv2, name):
        raise ValueError(f"Unsupported RANSAC method '{name}'")
    return getattr(cv2, name)


def build_matcher(cfg):
    method_token = cfg["method"]
    method_name, estimator_name = method_token.split("@-@")
    loftr_cfg = get_cfg_defaults()
    if method_name != "None":
        loftr_cfg.merge_from_file(str(cfg["main_cfg_path"]))
        pl.seed_everything(loftr_cfg.TRAINER.SEED)
        loftr_cfg.METHOD = method_name
        if loftr_cfg.LOFTR.COARSE.ROPE:
            assert loftr_cfg.DATASET.NPE_NAME is not None
        if loftr_cfg.DATASET.NPE_NAME is not None:
            loftr_cfg.LOFTR.COARSE.NPE = [832, 832, cfg["imgresize"], cfg["imgresize"]]
        if cfg["match_threshold"] is not None:
            loftr_cfg.LOFTR.MATCH_COARSE.THR = cfg["match_threshold"]
            loftr_cfg.ROMA.MATCH_THRESH = cfg["match_threshold"]
        if cfg["max_features"] is not None:
            loftr_cfg.ROMA.SAMPLE.N_SAMPLE = cfg["max_features"]
        matcher = PL_LoFTR(
            loftr_cfg, pretrained_ckpt=str(cfg["ckpt_path"]), test_mode=True
        ).matcher
        matcher.eval().cuda()
    else:
        matcher = None
    return matcher, loftr_cfg, estimator_name


def build_dataloader(cfg, loftr_cfg):
    datasets = []
    has_ground_truth = False
    if cfg["npz_list_path"].exists():
        with open(cfg["npz_list_path"], "r") as f:
            raw_names = [line.strip().split()[0] for line in f if line.strip()]
        for name in raw_names:
            filename = name if name.endswith(".npz") else f"{name}.npz"
            full_path = (
                (cfg["npz_root"] / filename) if cfg["npz_root"] else Path(filename)
            )
            if not full_path.exists():
                logger.warning(f"Skipping missing npz file: {full_path}")
                continue
            try:
                np.load(full_path, allow_pickle=True)
            except Exception as exc:
                logger.warning(
                    f"Skipping npz '{full_path}' because it could not be loaded ({exc})"
                )
                continue
            datasets.append(
                CommonDataset(
                    root_dir=str(cfg["data_root"]),
                    npz_path=str(full_path),
                    mode="test",
                    min_overlap_score=-1,
                    img_resize=cfg["imgresize"],
                    df=None,
                    img_padding=False,
                    depth_padding=True,
                    testNpairs=None,
                    fp16=False,
                    dataset_name=cfg["dataset_name"],
                    load_origin_rgb=cfg["load_origin_rgb"],
                    read_gray=cfg["read_gray"],
                    normalize_img=cfg["normalize_img"],
                    resize_by_stretch=False,
                    gt_matches_padding_n=cfg["gt_matches_padding_n"],
                    auto_contrast_stretch=cfg["auto_contrast_stretch"],
                )
            )
        has_ground_truth = True
    elif cfg["pairs_txt_path"]:
        with open(cfg["pairs_txt_path"], "r") as f:
            pairs = [tuple(line.strip().split()[:2]) for line in f if line.strip()]
        datasets.append(
            CommonDataset(
                root_dir=str(cfg["data_root"]),
                npz_path=None,
                mode="test",
                img_resize=cfg["imgresize"],
                df=None,
                img_padding=False,
                depth_padding=False,
                testNpairs=None,
                fp16=False,
                dataset_name=cfg["dataset_name"],
                pairs_list=pairs,
                load_origin_rgb=cfg["load_origin_rgb"],
                read_gray=cfg["read_gray"],
                normalize_img=cfg["normalize_img"],
                resize_by_stretch=False,
                gt_matches_padding_n=cfg["gt_matches_padding_n"],
                auto_contrast_stretch=cfg["auto_contrast_stretch"],
            )
        )
        has_ground_truth = False
    else:
        raise ValueError(
            "Please specify either 'npz_list_path' for evaluation or 'pairs_txt_path' for inference in CONFIG."
        )
    if not datasets:
        raise RuntimeError("No datasets available after filtering.")
    dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    dataloader = DataLoader(
        dataset,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        batch_size=cfg["batch_size"],
        drop_last=False,
    )
    return dataloader, has_ground_truth


def run_pipeline(config):
    cfg = config.copy()
    for key in PATH_KEYS:
        if cfg.get(key) is not None:
            cfg[key] = Path(cfg[key])
    output_dir = cfg["output_root"] / cfg["dataset_name"]
    (output_dir / "npz").mkdir(parents=True, exist_ok=True)
    (output_dir / "demo_matches").mkdir(parents=True, exist_ok=True)
    (output_dir / "aligned").mkdir(parents=True, exist_ok=True)

    cfg["npz_root"] = cfg["data_root"] / cfg["dataset_name"] / "eval_indexs"
    cfg["npz_list_path"] = cfg["npz_root"] / "val_list.txt"

    cfg["ransac_method"] = _resolve_ransac_method(cfg["ransac_method"])

    matcher, loftr_cfg, estimator_name = build_matcher(cfg)
    dataloader, has_ground_truth = build_dataloader(cfg, loftr_cfg)
    run_evaluation = bool(cfg["run_evaluation"] and has_ground_truth)
    logger.info(
        "Processing {} pairs",
        len(dataloader.dataset) if hasattr(dataloader, "dataset") else "unknown",
    )
    logger.info("Mode: {}", "evaluation" if run_evaluation else "inference")

    errors = []
    result_dict = {}
    npz_identifier = str(cfg["npz_list_path"]) if cfg.get("npz_list_path") else ""

    for data in tqdm(dataloader, desc="Processing pairs", leave=False):
        img0 = (
            (data["image0_rgb_origin"] * 255.0)[0]
            .permute(1, 2, 0)
            .cpu()
            .numpy()
            .round()
            .astype(np.uint8)
        )
        img1 = (
            (data["image1_rgb_origin"] * 255.0)[0]
            .permute(1, 2, 0)
            .cpu()
            .numpy()
            .round()
            .astype(np.uint8)
        )

        name0 = data["pair_names"][0][0]
        name1 = data["pair_names"][1][0]
        reference_img_shape = tuple(int(v) for v in data["origin_img_size0"])
        source_img_shape = tuple(int(v) for v in data["origin_img_size1"])
        
        print(f"Reference image (img0) shape: {reference_img_shape}")
        print(f"Source image (img1) shape: {source_img_shape}")

        print(
            f"Processing pair: {name0} - {name1}, min-max are: {img0.min()} - {img0.max()} and {img1.min()} - {img1.max()} respectively"
        )
        print(
            f"Processing pair: {name0} - {name1}, min-max (tensors) are: {data['image0_rgb_origin'][0].min()} - {data['image0_rgb_origin'][0].max()} and {data['image1_rgb_origin'][0].min()} - {data['image1_rgb_origin'][0].max()} respectively"
        )

        pair_tag = _format_pair_key(name0, name1)
        homography_gt = data.get("homography", torch.zeros((1, 3, 3)))[0].numpy()

        active_estimator = estimator_name
        eval_mode = None
        eval_coord = None
        gt_points = None
        pose = None

        if run_evaluation:
            if "gt_2D_matches" in data and data["gt_2D_matches"].shape[-1] == 4:
                gt_2D_matches = data["gt_2D_matches"][0].numpy()
                eval_coord = gt_2D_matches[:, :2]
                gt_points = gt_2D_matches[:, 2:]
                eval_mode = "gt_match"
                if "FIRE" in npz_identifier:
                    cfg["ransac_mode"] = "homo"
            elif homography_gt.sum() != 0:
                h, w = img0.shape[:2]
                eval_coord = np.array([[0, 0], [0, h], [w, 0], [w, h]])
                gt_points = warp_points(eval_coord, homography_gt, inverse=False)
                eval_mode = "gt_homo"
            else:
                K0 = data["K0"].cpu().numpy()[0]
                K1 = data["K1"].cpu().numpy()[0]
                T_0to1 = data["T_0to1"].cpu().numpy()[0]
                eval_mode = "pose_error"
                active_estimator = "pose"
        else:
            eval_mode = active_estimator

        if matcher is None:
            raise NotImplementedError(
                "Currently only matcher-based evaluation is supported."
            )
        if eval_mode == "gt_match" and eval_coord is not None:
            data.update({"query_points": torch.from_numpy(eval_coord)[None]})

        batch_cuda = dict_to_cuda(data)
        with torch.no_grad():
            with torch.autocast(enabled=loftr_cfg.LOFTR.FP16, device_type="cuda"):
                matcher(batch_cuda)

            mkpts0 = batch_cuda["mkpts0_f"].cpu().numpy()
            mkpts1 = batch_cuda["mkpts1_f"].cpu().numpy()
            mconf = batch_cuda["mconf"].cpu().numpy()
        
        if cfg["ransac_method"] == cv2.USAC_PROSAC:
            order = np.argsort(-mconf)
            mkpts0 = mkpts0[order]
            mkpts1 = mkpts1[order]
            mconf = mconf[order]

        print(f"{pair_tag}: {len(mkpts0)} matches")

        eval_points_warped = None
        match_error = None
        inliers = None
        if active_estimator == "ransac_affine":
            H_est, inliers = estimate_homo(
                mkpts0,
                mkpts1,
                thresh=cfg["ransac_threshold"],
                conf=cfg["ransac_confidence"],
                mode=cfg["ransac_mode"],
                ransac_method=cfg["ransac_method"],
                ransac_maxIters=cfg["ransac_max_iters"],
            )
            if inliers is not None:
                print(
                    f"[INFO] RANSAC inliers for {pair_tag}: {int(inliers.sum())}/{len(mkpts0)}"
                )
            if run_evaluation and eval_coord is not None and H_est is not None:
                eval_points_warped = warp_points(eval_coord, H_est, inverse=False)
            if cfg["plot_align"] and H_est is not None:
                warp_img_and_blend(
                    img0,
                    img1,
                    H_est,
                    save_path=output_dir
                    / "aligned"
                    / f"{pair_tag}_{cfg['method']}.png",
                    alpha=0.5,
                    inverse=True,
                )
        elif active_estimator == "pose":
            pose = estimate_pose(
                mkpts0,
                mkpts1,
                data["K0"].cpu().numpy()[0],
                data["K1"].cpu().numpy()[0],
                cfg["ransac_threshold"],
                conf=0.99999,
            )
        else:
            raise NotImplementedError(f"Unsupported estimator '{active_estimator}'.")

        if run_evaluation:
            if eval_mode == "pose_error":
                if pose is None:
                    t_err = R_err = np.inf
                else:
                    R, t, inliers = pose
                    t_err, R_err = relative_pose_error(
                        T_0to1, R, t, ignore_gt_t_thr=0.0
                    )
                error_val = max(t_err, R_err)
                match_error = epipolar_error(mkpts0, mkpts1, T_0to1, K0, K1)
            else:
                if eval_mode == "gt_homo":
                    match_error = np.linalg.norm(
                        warp_points(mkpts0, homography_gt, inverse=False) - mkpts1,
                        axis=-1,
                    )
                error_val = float(
                    np.mean(np.linalg.norm(eval_points_warped - gt_points, axis=1))
                )
            errors.append(error_val)
            result_dict[_format_pair_key(name0, name1)] = error_val

        if cfg["plot_matches"]:
            draw_match_type = "corres"
            color_type = cfg["plot_matches_color"]
            plot_matches(
                img0,
                img1,
                mkpts0,
                mkpts1,
                mconf,
                vertical=False,
                draw_match_type=draw_match_type,
                alpha=cfg["plot_matches_alpha"],
                save_path=output_dir
                / "demo_matches"
                / f"{pair_tag}_{draw_match_type}.pdf",
                inverse=False,
                match_error=(
                    match_error
                    if (color_type == "error" and match_error is not None)
                    else None
                ),
                error_thr=5,
                color_type=color_type,
            )

    if run_evaluation and errors:
        errors_np = np.array(errors)
        success_rate = error_auc(
            errors_np, thresholds=[5, 10, 20], method="success_rate"
        )
        print(success_rate)
        auc_metric = error_auc(
            errors_np,
            thresholds=[5, 10, 20],
            method="fire_paper" if "FIRE" in npz_identifier else "exact_auc",
        )
        print(auc_metric)
        save_h5(
            result_dict,
            output_dir
            / f"eval_{cfg['dataset_name']}_{cfg['method']}_{cfg['comment']}_error.h5",
        )
    elif run_evaluation:
        logger.warning("No evaluation metrics were produced.")

    logger.info("Finished processing dataset.")


if __name__ == "__main__":
    run_pipeline(CONFIG)
