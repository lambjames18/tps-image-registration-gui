import cv2
from pathlib import Path
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys

sys.path.append(str(Path(__file__).parent.parent.resolve()))

from warping import transform_image
from tools_utils.plot import plot_matches, warp_img_and_blend, blend_img
from src.datasets.common_data_pair import CommonDataset
from src.lightning.lightning_loftr import PL_LoFTR
from src.config.default import get_cfg_defaults
from src.utils.dataset import dict_to_cuda
from src.utils.metrics import estimate_homo

CONFIG = {
    "dataset_name": "CrackGrowth_LOM_BM_Bolzen_AWH3-Mz-S16o",  # "DIC_EBSD_Multimodal_In718", #"SE2_EBSD_X2CrNi12", # "Tribeam_tilt_BSE_CI"
    "method": "matchanything_roma",
    "main_cfg_path": Path("./configs/models/roma_model.py"),
    "ckpt_path": Path("./weights/matchanything_roma.ckpt"),
    "data_root": Path("./data/test_data"),
    "pairs_txt_path": None,
    "output_root": Path("./results"),
    "match_threshold": 0.1,
    "max_features": 200,  # 5000,
    "imgresize": None,
    "ransac_mode": "homo",
    "ransac_method": "USAC_MAGSAC",
    "ransac_threshold": 10.0,  # 2.0,
    "ransac_confidence": 0.999,
    "ransac_max_iters": 10000,
    "normalize_img": True,
    "plot_matches": True,
    "plot_align": True,
    "plot_matches_filtered": True,
    "apply_tps_warp": True,
    "auto_contrast_stretch": False,
    "comment": "",
}

PATH_KEYS = {
    "main_cfg_path",
    "ckpt_path",
    "data_root",
    "pairs_txt_path",
    "output_root",
}


def _sanitize_pair_name(path0: str, path1: str) -> str:
    return f"{Path(path0).name}@-@{Path(path1).name}"


def _resolve_ransac_method(name: str):
    if not hasattr(cv2, name):
        raise ValueError(f"Unsupported RANSAC method '{name}'")
    return getattr(cv2, name)


def build_matcher(cfg):
    config = get_cfg_defaults()
    if cfg["method"] != "None":
        config.merge_from_file(str(cfg["main_cfg_path"]))
        pl.seed_everything(config.TRAINER.SEED)
        config.METHOD = cfg["method"]
        if config.LOFTR.COARSE.ROPE:
            assert config.DATASET.NPE_NAME is not None
        if config.DATASET.NPE_NAME is not None:
            config.LOFTR.COARSE.NPE = [832, 832, cfg["imgresize"], cfg["imgresize"]]
        if cfg["match_threshold"] is not None:
            config.LOFTR.MATCH_COARSE.THR = cfg["match_threshold"]
            config.ROMA.MATCH_THRESH = cfg["match_threshold"]
        if cfg["max_features"] is not None:
            config.ROMA.SAMPLE.N_SAMPLE = cfg["max_features"]
        matcher = PL_LoFTR(
            config, pretrained_ckpt=str(cfg["ckpt_path"]), test_mode=True
        ).matcher
        matcher.eval().cuda()
    else:
        matcher = None
    return matcher, config


def build_dataloader(cfg):
    dataset_dir = cfg["data_root"] / cfg["dataset_name"]
    pairs_txt = cfg["pairs_txt_path"] or (dataset_dir / "pairs.txt")
    if not pairs_txt.exists():
        raise FileNotFoundError(f"Pair list not found: {pairs_txt}")

    pairs = []
    with open(pairs_txt, "r") as f:
        for line in f:
            if line.strip():
                p0, p1 = line.strip().split()[:2]
                pairs.append((p0, p1))

    dataset = CommonDataset(
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
        load_origin_rgb=True,
        read_gray=True,
        normalize_img=cfg["normalize_img"],
        resize_by_stretch=False,
        gt_matches_padding_n=100,
        auto_contrast_stretch=cfg["auto_contrast_stretch"],
    )
    return DataLoader(dataset, batch_size=1, shuffle=False)


def run_pairwise_inference(config):
    cfg = config.copy()
    for key in PATH_KEYS:
        if cfg.get(key) is not None:
            cfg[key] = Path(cfg[key])

    matcher, loftr_cfg = build_matcher(cfg)
    dataloader = build_dataloader(cfg)

    image_dir = cfg["data_root"] / cfg["dataset_name"]
    output_dir = cfg["output_root"] / cfg["dataset_name"]
    (output_dir / "npz").mkdir(parents=True, exist_ok=True)
    (output_dir / "demo_matches").mkdir(parents=True, exist_ok=True)
    (output_dir / "aligned").mkdir(parents=True, exist_ok=True)

    cfg["ransac_method"] = _resolve_ransac_method(cfg["ransac_method"])

    for data in tqdm(dataloader, desc="Pairs", leave=False):
        batch = dict_to_cuda(data)
        rgb0 = data["image0_rgb_numpy"][0]
        rgb1 = data["image1_rgb_numpy"][0]
        data["image0_rgb_origin"][0]

        print(
            f"Processing pair: {data['pair_names'][0][0]} - {data['pair_names'][1][0]}, min-max are: {rgb0.min()} - {rgb0.max()} and {rgb1.min()} - {rgb1.max()} respectively"
        )
        print(
            f"Processing pair: {data['pair_names'][0][0]} - {data['pair_names'][1][0]}, min-max (tensors) are: {data['image0_rgb_origin'][0].min()} - {data['image0_rgb_origin'][0].max()} and {data['image1_rgb_origin'][0].min()} - {data['image1_rgb_origin'][0].max()} respectively"
        )

        reference_img_shape = tuple(int(v) for v in data["origin_img_size0"])
        name0 = data["pair_names"][0][0]
        name1 = data["pair_names"][1][0]
        pair_tag = _sanitize_pair_name(name0, name1)

        with torch.no_grad():
            with torch.autocast(device_type="cuda", enabled=True):
                matcher(batch)
                
            mkpts0 = batch["mkpts0_f"].cpu().numpy()
            mkpts1 = batch["mkpts1_f"].cpu().numpy()
            mconf = batch["mconf"].cpu().numpy()

        if cfg["ransac_method"] == cv2.USAC_PROSAC:
            order = np.argsort(-mconf)
            mkpts0 = mkpts0[order]
            mkpts1 = mkpts1[order]
            mconf = mconf[order]

        print(f"{pair_tag}: {len(mkpts0)} matches")

        npz_suffix = f"_{cfg['comment']}" if cfg["comment"] else ""
        np.savez(
            output_dir / "npz" / f"{pair_tag}{npz_suffix}.npz",
            mkpts0=mkpts0,
            mkpts1=mkpts1,
            mconf=mconf,
            img0=name0,
            img1=name1,
        )
        text_imprint_raw = [f"matches: {len(mkpts0)}"]
        if cfg["plot_matches"]:
            plot_matches(
                rgb0,
                rgb1,
                mkpts0,
                mkpts1,
                mconf,
                vertical=False,
                draw_match_type="corres",
                alpha=0.10,
                save_path=output_dir / "demo_matches" / f"{pair_tag}.pdf",
                inverse=False,
                match_error=None,
                error_thr=None,
                color_type="green",
                text=text_imprint_raw,
            )

        H_est = None
        inliers = None
        try:
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
                text_imprint_ransac = [f"matches: {len(mkpts0)}"]
        except Exception as exc:
            print(f"[WARN] RANSAC failed for {pair_tag}: {exc}")

        if inliers is not None:
            mask = inliers.astype(bool).ravel()
            mkpts0_filtered = mkpts0[mask]
            mkpts1_filtered = mkpts1[mask]
            mconf_filtered = mconf[mask]
            np.savez(
                output_dir / "npz" / f"{pair_tag}{npz_suffix}_filtered.npz",
                mkpts0=mkpts0_filtered,
                mkpts1=mkpts1_filtered,
                mconf=mconf_filtered,
                img0=name0,
                img1=name1,
            )
            if cfg["plot_matches_filtered"]:
                plot_matches(
                    rgb0,
                    rgb1,
                    mkpts0_filtered,
                    mkpts1_filtered,
                    mconf_filtered,
                    vertical=False,
                    draw_match_type="corres",
                    alpha=0.10,
                    save_path=output_dir / "demo_matches" / f"{pair_tag}_filtered.pdf",
                    inverse=False,
                    match_error=None,
                    error_thr=None,
                    color_type="green",
                    text=text_imprint_ransac,
                )
        else:
            mkpts0_filtered = mkpts0
            mkpts1_filtered = mkpts1

        rgb1_tps_warped = None
        if cfg["apply_tps_warp"] and inliers is not None and len(mkpts0_filtered) >= 4:
            rgb1_tps_warped, _ = transform_image(
                rgb1.cpu().numpy(),
                mkpts1_filtered,
                mkpts0_filtered,
                size=reference_img_shape,
                output_shape=reference_img_shape,
                mode="tps",
                return_params=True,
            )

        if cfg["plot_align"] and len(mkpts0) >= 4:
            if H_est is not None:
                warp_img_and_blend(
                    rgb0,
                    rgb1,
                    H_est,
                    save_path=output_dir / "aligned" / f"{pair_tag}.png",
                    alpha=0.5,
                    inverse=True,
                )
            if rgb1_tps_warped is not None:
                blend_img(
                    np.copy(rgb0).astype(np.uint8),
                    rgb1_tps_warped,
                    alpha=0.5,
                    save_path=output_dir / "aligned" / f"{pair_tag}_tps.png",
                    blend_method="weighted_sum",
                )


if __name__ == "__main__":
    run_pairwise_inference(CONFIG)
