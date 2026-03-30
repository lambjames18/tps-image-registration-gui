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


CONFIG = {
    "eval_dataset": True,
    "main_cfg_path": "configs/models/roma_model.py", #"configs/models/eloftr_model.py", "configs/models/roma_model.py",  # Required, replace with actual path
    "ckpt_path": "weights/matchanything_roma.ckpt", #"weights/matchanything_eloftr.ckpt", # "weights/matchanything_roma.ckpt", #"weights/minima_roma.pth", , "weights/minima_roma.pth"
    "thr": 0.1,
    "method": "matchanything_roma", #"matchanything_eloftr", #"minima_roma", # "SIFT"
    "transformation_type": "tps", #"affine",  # "homo",
    "imgresize": 832,
    "divisible_by": 32, # this factor is utilized in the image loading pipeline to ensure the image sizes are divisible by this number, for ELoFTR 32 is be necessary, does not affect the RoMa loading pipeline
    "resize_by_stretch": True, # whether to resize by stretch or by padding one dimension (keeping aspect ratio)
    "npe": True,
    "npe2": False,
    "ckpt32": False,
    "fp32": False,
    "dataset_name": "AF9628-Martensitic_SEM-SE-Stitch_EBSD_SameSlice",#"All", #"5842WCu-Spalled_SEM-SE_SEM-BSE_Multiscale", #"5842WCu_Spalled_Cropping_Study", #"All", #"5842WCu-Spalled_SEM-SE_SEM-BSE_Multiscale", #"In718SS-CHESS", #"C103_fracture_surfaces",  #"AMSpalledTa", #"DDRX_Dislocations_900C_largeFOV", #"DDRX_Dislocations_900C", #"DDRX_Dislocations_1100C", #"MAX_Phase_Dislocations", #"CoNi90-AM-DIC", #"DIC_EBSD_Multimodal_In718", #"TRIP1_steel_LOM_EBSD", #"In718_same_slice_BSE_EBSD", #"SE2_EBSD_X2CrNi12",# "Martensitic_steel_AF_9628_SEM_EBSD",#"5842WCu_Spalled", #"In718SS-CHESS", ##"CoNi90_mid_OM-2-BSE", #"CoNi67_mid_OM-2-high_OM", #"CoNi67_high_OM-2-SE", #"CP700BC_cracks_block5_pedestal1_0", #"AMSpalledTa", #"CoNi67", "Liver_CT-MR"
    "data_root": "data/test_data",
    "output_root": "results",
    "plot_matches": True,
    "plot_matches_alpha": 0.2,
    "plot_matches_color": "error",  # options: ['green', 'error', 'conf']
    "plot_align": True,
    "plot_refinement": False,
    "plot_checkerboard": True,
    "rigid_ransac_thr": 5.5, #10.0,
    "elastix_ransac_thr": 40.0,
    "normalize_img": True,
    "RANSAC_correspondence_plane": False,
    "comment": "",
}


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


        if cfg["thr"] is not None:
            config.LOFTR.MATCH_COARSE.THR = cfg["thr"]

        config.ROMA.RESIZE_BY_STRETCH = cfg["resize_by_stretch"]
        config.DATASET.RESIZE_BY_STRETCH = cfg["resize_by_stretch"]

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
    results_df = pd.DataFrame(columns=["SceneID", "Registration Target", "Registration Source", "Downsampling Factor Target", "Downsampling Factor Source", "FOV ratio", "Mean Euclidean Error", "Max Euclidean Error", "Mean Euclidean Error [m]", "Max Euclidean Error [m]", "Num GT Matches Used", "Num Matches Raw", "Num Matches RANSAC Inliers"])
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
                eval_mode = "gt_match"
                # ransac_mode = "homo" if "FIRE" in cfg["npz_list_path"] else "affine"
            elif homography_gt.sum() != 0:
                h, w = img0.shape[0], img0.shape[1]
                eval_coord = np.array([[0, 0], [0, h], [w, 0], [w, h]])
                # ransac_mode = "affine"
                assert (
                    homography_gt.sum() != 0
                ), f"Evaluation should either using gt match, or using gt homography warp."
            else:
                eval_mode = "pose_error"
                K0 = data["K0"].cpu().numpy()[0]
                K1 = data["K1"].cpu().numpy()[0]
                T_0to1 = data["T_0to1"].cpu().numpy()[0]
                cfg["transformation_type"] = "pose"

        # Perform matching
        if matcher is not None:
            if matcher != "SIFT":
                if cfg["eval_dataset"] and eval_mode in ["gt_match"]:
                    data.update({"query_points": torch.from_numpy(eval_coord)[None]})
                batch = dict_to_cuda(data)

                with torch.no_grad():
                    with torch.autocast(enabled=config.LOFTR.FP16, device_type="cuda"):
                        matcher(batch)

                    mkpts0 = batch["mkpts0_f"].cpu().numpy()
                    mkpts1 = batch["mkpts1_f"].cpu().numpy()
                    mconf = batch["mconf"].cpu().numpy()
            else:
                # # SIFT matching
                # sift = cv2.SIFT_create()
                # kp1, des1 = sift.detectAndCompute(
                #     array_rgb2gray(img0).astype(np.uint8), None
                # )
                # kp2, des2 = sift.detectAndCompute(
                #     array_rgb2gray(img1).astype(np.uint8), None
                # )
                # bf = cv2.BFMatcher()
                # matches = bf.knnMatch(des1, des2, k=2)

                # # Apply ratio test
                # good = []
                # for m, n in matches:
                #     if m.distance < 0.75 * n.distance:
                #         good.append(m)

                # mkpts0 = np.array([kp1[m.queryIdx].pt for m in good])
                # mkpts1 = np.array([kp2[m.trainIdx].pt for m in good])
                # mconf = np.array([1.0] * len(mkpts0))
                raise NotImplementedError("SIFT matching not implemented yet.")

            name0 = data["pair_names"][0][0]
            name1 = data["pair_names"][1][0]

            reference_img_shape = tuple(
                int(v) for v in data["origin_img_size0"].squeeze(0)
            )
            source_img_shape = tuple(
                int(v) for v in data["origin_img_size1"].squeeze(0)
            )
            
            if cfg["method"] == "matchanything_roma":
                inference_resolution_target = (560, 560)
                inference_resolution_source = (560, 560)
            elif cfg["method"] == "matchanything_eloftr":
                inference_resolution_target = tuple(
                    int(v) for v in data["img_size0"].squeeze(0)
                )
                inference_resolution_source = tuple(
                    int(v) for v in data["img_size1"].squeeze(0)
                )
            if cfg["resize_by_stretch"]:
                # if resize by stretch, compute mean downsampling factor because both dimensions are scaled differently
                dsf_target = np.mean(np.array(inference_resolution_target)/np.array(reference_img_shape))
                dsf_source = np.mean(np.array(inference_resolution_source)/np.array(source_img_shape))
            else:
                # if resize by padding, compute downsampling factor based on longer dimension only since the other dimension is padded
                dsf_target = np.max(np.array(inference_resolution_target))/np.max(np.array(reference_img_shape))
                dsf_source = np.max(np.array(inference_resolution_source))/np.max(np.array(source_img_shape))
                            

            print(f"Reference image (img0) shape: {reference_img_shape}")
            print(f"Source image (img1) shape: {source_img_shape}")

            print(
                f"Processing pair: {name0} - {name1}, min-max are: {img0.min()} - {img0.max()} and {img1.min()} - {img1.max()} respectively"
            )
            print(
                f"Processing pair: {name0} - {name1}, min-max (tensors) are: {data['image0_rgb_origin'][0].min()} - {data['image0_rgb_origin'][0].max()} and {data['image1_rgb_origin'][0].min()} - {data['image1_rgb_origin'][0].max()} respectively"
            )

            print(f"Number of keypoints in mkpts0: {len(mkpts0)}")
            print(f"Number of keypoints in mkpts1: {len(mkpts1)}")

            # Get warped points by homography:
            if cfg["transformation_type"] in ["affine", "homo"]:
                H_est, _ = estimate_homo(
                    mkpts0,
                    mkpts1,
                    thresh=cfg["rigid_ransac_thr"],
                    mode=cfg["transformation_type"],
                )
                if cfg["eval_dataset"]:
                    # Warp points for eval:
                    eval_points_warpped = warp_points(eval_coord, H_est, inverse=False)

                # Warp images and blend:
                if cfg["plot_align"]:
                    warp_img_and_blend(
                        img0,
                        img1,
                        H_est,
                        save_path=Path(vis_output_path)
                        / "aligned"
                        / f"{pair_name}_{cfg['method']}.png",
                        alpha=0.5,
                        inverse=True,
                    )
            elif cfg["transformation_type"] == "tps":
                # Perform RANSAC filtering for tps, with larger hard error margin to remove outliers
                H_est = None
                inliers = None
                try:
                    if not cfg["RANSAC_correspondence_plane"]:
                        _, inliers = estimate_homo(
                            mkpts0,
                            mkpts1,
                            thresh=cfg["rigid_ransac_thr"],  # maybe adjust?
                            #conf=cfg["ransac_confidence"],
                            mode="homo",
                            #ransac_method=cfg["ransac_method"],
                            #ransac_maxIters=cfg["ransac_max_iters"],
                        )
                    else:
                        _, inliers = ransac_correspondence_plane(
                            mkpts0,
                            mkpts1,
                            max_iters=100,
                            dist_thresh=0.03,
                            random_state=42,
                        )
                    if inliers is not None:
                        print(
                            f"[INFO] RANSAC inliers for {pair_name}: {int(inliers.sum())}/{len(mkpts0)}"
                        )
                        text_imprint_ransac = [f"matches: {len(mkpts0)}"]
                except Exception as exc:
                    print(f"[WARN] RANSAC failed for {pair_name}: {exc}")

                if inliers is not None:
                    mask = inliers.astype(bool).ravel()
                    mkpts0_filtered = mkpts0[mask]
                    mkpts1_filtered = mkpts1[mask]
                    mconf_filtered = mconf[mask]

                    npz_suffix = f"_{cfg['comment']}" if cfg["comment"] else ""
                    np.savez(
                        Path(vis_output_path)
                        / "npz"
                        / f"{pair_name}{npz_suffix}_filtered.npz",
                        mkpts0=mkpts0_filtered,
                        mkpts1=mkpts1_filtered,
                        mconf=mconf_filtered,
                        img0=name0,
                        img1=name1,
                    )
                else:
                    print("[WARN] Inliers not found. Continuing without filtering.")
                    mkpts0_filtered = mkpts0
                    mkpts1_filtered = mkpts1
                    mconf_filtered = mconf
                
                       
                if mkpts0_filtered.size > 0 and mkpts1_filtered.size > 0:
                    #Filter for duplicates in mkpts1_filtered and mkpts0_filtered
                    if len(mkpts1_filtered) != len(np.unique(mkpts1_filtered, axis=0)):
                        mkpts1_filtered, idx = np.unique(mkpts1_filtered, axis=0, return_index=True)
                        mkpts0_filtered = mkpts0_filtered[idx]
                    if len(mkpts0_filtered) != len(np.unique(mkpts0_filtered, axis=0)):
                        mkpts0_filtered, idx = np.unique(mkpts0_filtered, axis=0, return_index=True)
                        mkpts1_filtered = mkpts1_filtered[idx]
                    if mkpts0_filtered.size > 0 and mkpts1_filtered.size > 0:
                        print(
                            f"[INFO] After 'unique' filtering inliers for {pair_name}: {len(mkpts0_filtered)}/{len(mkpts0)}."
                        )
                    H_est = get_transform(
                        mkpts1_filtered,
                        mkpts0_filtered,
                        mode="tps",
                        size=reference_img_shape,
                    )
                    # warp the image
                    img1_tps_warped = tf.warp(
                        img1,
                        H_est,
                        mode="constant",
                        cval=0,
                        order=1,
                        output_shape=reference_img_shape,
                    )
                    # if evaluation is done, transform points
                    if cfg["eval_dataset"] and "gt_2D_matches" in data and data["gt_2D_matches"].shape[-1] == 4:
                        eval_points_warpped = H_est(eval_coord)

                    if cfg["plot_align"]:
                        blend_img(
                            np.copy(img0).astype(np.uint8),
                            np.copy(img1_tps_warped).astype(np.uint8),
                            alpha=0.5,
                            save_path=Path(vis_output_path) / "aligned" / f"{pair_name}_tps.png",
                            blend_method="weighted_sum",
                        )
                    if cfg["plot_checkerboard"]:
                        checkerboard_overlap(
                            np.copy(img0).astype(np.uint8),
                            np.copy(img1_tps_warped).astype(np.uint8),
                            save_path=Path(vis_output_path) / "checkerboard" / f"{pair_name}_tps.png",
                            block_size=64,
                        )
                    correspondence_query_plot(
                            np.copy(img0).astype(np.uint8), 
                            np.copy(img1).astype(np.uint8), 
                            gt_2D_matches, 
                            pred_matches=eval_points_warpped, 
                            save_path=Path(vis_output_path) / "query_correspondence_plot" / f"{pair_name}_tps.png", 
                            figsize=(20,12)
                        )
                    
                    if cfg["method"] == "matchanything_roma":
                        correspondence_query_plot(
                            np.copy(img0).astype(np.uint8), 
                            np.copy(img1).astype(np.uint8), 
                            gt_2D_matches, 
                            pred_matches=batch['query_points_warpped'].detach().cpu().numpy() if 'query_points_warpped' in batch else None, 
                            save_path=Path(vis_output_path) / "query_correspondence_plot" / f"{pair_name}_raw.png", 
                            figsize=(20,12)
                        )
                else:
                    print(f"[INFO] Skipping pair {pair_name} as no keypoints were detected.")
                    results_df.loc[len(results_df)] = [
                        data["scene_id"],
                        data["pair_names"][0][0].split("/")[-1],
                        data["pair_names"][1][0].split("/")[-1],
                        f"{dsf_target:.5f}" if dsf_target is not None else None, #"Mean Downsampling Factor Target"
                        f"{dsf_source:.5f}" if dsf_source is not None else None, #"Mean Downsampling Factor Source"
                        f"{fov_ratio:.5f}" if fov_ratio is not None else None, #"FOV_ratio"
                        None,
                        None,
                        None,
                        None,
                        f"{len(gt_points)}",
                        f"{len(mkpts0)}" if mkpts0 is not None else "0",
                        f"{int(inliers.sum()) if 'inliers' in locals() and inliers is not None else '0'}",
                    ]
                    continue

            elif cfg["transformation_type"] == "pose" and cfg["eval_dataset"]:
                pose = estimate_pose(
                    mkpts0, mkpts1, K0, K1, cfg["rigid_ransac_thr"], conf=0.99999
                )
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        if cfg["eval_dataset"]:
            if eval_mode == "pose_error":
                if pose is None:
                    t_err, R_err = np.inf, np.inf
                else:
                    R, t, inliers = pose
                    t_err, R_err = relative_pose_error(
                        T_0to1, R, t, ignore_gt_t_thr=0.0
                    )
                error = max(t_err, R_err)
                errors.append(error)
                match_error = epipolar_error(mkpts0, mkpts1, T_0to1, K0, K1)
                plot_text = f"R_err_{R_err:.2}_t_err_{t_err:.2}"
                thr = 3e-3
                print(f"max_error:{error}")
            else:
                if eval_mode == "gt_homo":
                    gt_points = warp_points(eval_coord, homography_gt, inverse=False)
                    match_error = np.linalg.norm(
                        warp_points(mkpts0, homography_gt, inverse=False) - mkpts1,
                        axis=-1,
                    )
                else:
                    match_error = None

                thr = 5  # Pix
                error = np.linalg.norm(eval_points_warpped - gt_points, axis=1)
                mean_error = np.mean(error)
                max_error = np.max(error)
                if "Physical Pixel Size [m]" in data["image_metadata1"] and "Physical Pixel Size [m]" in data["image_metadata0"]:
                    physical_pixel_size_source = data["image_metadata1"]["Physical Pixel Size [m]"].cpu().numpy()[0] # the pixel size of image1 is used since error is computed in image1 (source) coordinate
                    physical_pixel_size_target = data["image_metadata0"]["Physical Pixel Size [m]"].cpu().numpy()[0]
                    mean_error_m = mean_error * physical_pixel_size_source
                    max_error_m = max_error * physical_pixel_size_source

                    physical_resolution_target = physical_pixel_size_target * np.array(reference_img_shape)
                    physical_resolution_source = physical_pixel_size_source * np.array(source_img_shape)
                    fov_target = np.prod(physical_resolution_target) # in m^2
                    fov_source = np.prod(physical_resolution_source) # in m^2
                    if fov_target < fov_source:
                        fov_ratio = fov_target / fov_source
                    else:
                        fov_ratio = fov_source / fov_target
                else:
                    mean_error_m = None
                    max_error_m = None
                    fov_ratio = None

                print(f"error: {mean_error}")
                errors.append(mean_error)

            result_dict[
                "@-@".join(
                    [
                        data["pair_names"][0][0].split("/", 1)[1],
                        data["pair_names"][1][0].split("/", 1)[1],
                    ]
                )
            ] = mean_error

            results_df.loc[len(results_df)] = [
                data["scene_id"],
                data["pair_names"][0][0].split("/")[-1],
                data["pair_names"][1][0].split("/")[-1],
                f"{dsf_target:.5f}" if dsf_target is not None else None, #"Downsampling Factor Target"
                f"{dsf_source:.5f}" if dsf_source is not None else None, #"Downsampling Factor Source"
                f"{fov_ratio:.5f}" if fov_ratio is not None else None, #"FOV_ratio"
                f"{mean_error:.5f}" if mean_error is not None else None,
                f"{max_error:.5f}" if max_error is not None else None,
                f"{mean_error_m:.12f}" if mean_error_m is not None else None,
                f"{max_error_m:.12f}" if max_error_m is not None else None,
                f"{len(gt_points)}",
                f"{len(mkpts0)}" if mkpts0 is not None else "0",
                f"{int(inliers.sum()) if 'inliers' in locals() and inliers is not None else '0'}",
            ]

        if cfg["plot_matches"] and matcher is not None:
            if cfg["eval_dataset"]:
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
                    save_path=Path(vis_output_path)
                    / "demo_matches"
                    / f"{pair_name}_{draw_match_type}.pdf",
                    inverse=False,
                    match_error=match_error if color_type == "error" else None,
                    error_thr=thr,
                    color_type=color_type,
                )
            else:
                draw_match_type = "corres"
                plot_matches(
                    img0,
                    img1,
                    mkpts0,
                    mkpts1,
                    mconf,
                    vertical=False,
                    draw_match_type=draw_match_type,
                    alpha=cfg["plot_matches_alpha"],
                    save_path=Path(vis_output_path)
                    / "demo_matches"
                    / f"{pair_name}_{draw_match_type}.pdf",
                    inverse=False,
                    match_error=None,
                    error_thr=None,
                    color_type="conf",
                )

    if cfg["eval_dataset"]:
        # Success Rate Metric:
        SR_metric = error_auc(
            np.array(errors), thresholds=[5, 10, 20], method="success_rate"
        )
        print(SR_metric)

        # AUC Metric:
        AUC_metric = error_auc(
            np.array(errors),
            thresholds=[5, 10, 20],
            method="fire_paper" if "FIRE" in cfg["npz_list_path"] else "exact_auc",
        )
        print(AUC_metric)

        # save the config used
        with open(cfg['output_path'] + "/" + f"dataset_results_{cfg['dataset_name']}_{cfg['method']}_{cfg['comment']}.txt", "w") as f:
            f.write(str(SR_metric) + "\n" + str(AUC_metric))

        save_h5(
            result_dict,
            (
                Path(cfg["output_path"])
                / f'eval_{cfg["dataset_name"]}_{cfg["method"]}_{cfg["comment"]}_error.h5'
            ),
        )

        results_df.to_excel(Path(cfg["output_path"])
                / f'eval_{cfg["dataset_name"]}_{cfg["method"]}_{cfg["comment"]}_error.xlsx', index=False)


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