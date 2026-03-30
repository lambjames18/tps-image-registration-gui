import argparse
import pytorch_lightning as pl
from tqdm import tqdm
import os.path as osp
import numpy as np
from loguru import logger
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import torch

from torch.utils.data import (
    DataLoader,
    ConcatDataset)

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from src.lightning.lightning_loftr import PL_LoFTR
from src.config.default import get_cfg_defaults
from src.utils.dataset import dict_to_cuda
from src.utils.metrics import estimate_homo, estimate_pose, relative_pose_error
from src.utils.homography_utils import warp_points

from src.datasets.common_data_pair import CommonDataset
from src.utils.metrics import error_auc
from tools_utils.plot import plot_matches, warp_img_and_blend, epipolar_error
from tools_utils.data_io import save_h5

CONFIG = {
    'main_cfg_path': 'configs/models/roma_model.py',  # Required, replace with actual path
    'ckpt_path': 'weights/matchanything_roma.ckpt',
    'thr': 0.1,
    'method': 'matchanything_roma@-@ransac_affine',
    'imgresize': 832,
    'npe': True,
    'npe2': False,
    'ckpt32': False,
    'fp32': False,
    'data_root': 'data/test_data',
    'npz_root': 'data/test_data/Liver_CT-MR/eval_indexs',
    'npz_list_path': 'data/test_data/Liver_CT-MR/eval_indexs/val_list.txt',
    'plot_matches': True,
    'plot_matches_alpha': 0.2,
    'plot_matches_color': 'error',  # options: ['green', 'error', 'conf']
    'plot_align': True,
    'plot_refinement': False,
    'output_path': 'results/Liver_CT-MR',
    'rigid_ransac_thr': 3.0,
    'elastix_ransac_thr': 40.0,
    'comment': '',
    'dataset_name': 'Liver_CT-MR'
}

def array_rgb2gray(img):
    return (img * np.array([0.2989, 0.5870, 0.1140])[None, None]).sum(axis=-1)


def run_pipeline(config):
    cfg = config.copy()
    # Load data:
    datasets = []
    with open(cfg['npz_list_path'], 'r') as f:
        npz_names = [name.split()[0] for name in f.readlines()]
    npz_names = [f'{n}.npz' for n in npz_names]
    data_root = cfg['data_root']

    #print(f"npz_names is {npz_names}.")
    #print(f"data_root is {data_root}.")
    #print(f"npz_root is {cfg['npz_root']}.")
    #print(f"npz List Path: {cfg['npz_list_path']}")
    print(f"The Ransac Threshold is: {cfg['rigid_ransac_thr']}")

    vis_output_path = cfg['output_path']
    Path(vis_output_path).mkdir(parents=True, exist_ok=True)

    ##########################
    config = get_cfg_defaults()
    method, estimator = (cfg['method']).split('@-@')[0], (cfg['method']).split('@-@')[1]
    if method != 'None':
        config.merge_from_file(cfg['main_cfg_path'])

        pl.seed_everything(config.TRAINER.SEED)
        config.METHOD = method
        print(f"Method: {method}")
        # Config overwrite:
        if config.LOFTR.COARSE.ROPE:
            assert config.DATASET.NPE_NAME is not None
        if config.DATASET.NPE_NAME is not None:
            config.LOFTR.COARSE.NPE = [832, 832, cfg['imgresize'], cfg['imgresize']]

        if "visible_sar" in cfg['npz_list_path']:
            config.DATASET.RESIZE_BY_STRETCH = True

        if cfg['thr'] is not None:
            config.LOFTR.MATCH_COARSE.THR = cfg['thr']

        matcher = PL_LoFTR(config, pretrained_ckpt=cfg['ckpt_path'], test_mode=True).matcher
        matcher.eval().cuda()
    else:
        matcher = None

    for npz_name in tqdm(npz_names):
        npz_path = osp.join(cfg['npz_root'], npz_name)
        try:
            np.load(npz_path, allow_pickle=True)
        except:
            logger.info(f"{npz_path} cannot be opened!")
            continue
        
        #print(f"npz_names is {npz_names}.")
        #print(f"data_root is {data_root}.")
        #print(f"npz_root is {args.npz_root}.")
        #print(f"npz List Path: {args.npz_list_path}")

        datasets.append(
            CommonDataset(data_root, npz_path, mode='test', min_overlap_score=-1, img_resize=cfg['imgresize'], df=None, img_padding=False, depth_padding=True, testNpairs=None, fp16=False, load_origin_rgb=True, read_gray=True, normalize_img=False, resize_by_stretch=config.DATASET.RESIZE_BY_STRETCH, gt_matches_padding_n=100, dataset_name=cfg['dataset_name']))

    concat_dataset = ConcatDataset(datasets)

    dataloader = DataLoader(concat_dataset, num_workers=4, pin_memory=True, batch_size=1, drop_last=False)
    errors = [] # distance
    result_dict = {}
    pose_error = []

    eval_mode = 'gt_homo'
    for id, data in enumerate(tqdm(dataloader)):
        img0, img1 = (data['image0_rgb_origin'] * 255.)[0].permute(1,2,0).numpy().round().squeeze(), (data['image1_rgb_origin'] * 255.)[0].permute(1,2,0).numpy().round().squeeze()
        img_1_h, img_1_w = img1.shape[:2]
        pair_name = '@-@'.join([data['pair_names'][0][0].split('/', 1)[1], data['pair_names'][1][0].split('/', 1)[1]]).replace('/', '_')
        homography_gt = data['homography'][0].numpy()
        if 'gt_2D_matches' in data and data["gt_2D_matches"].shape[-1] == 4:
            gt_2D_matches = data["gt_2D_matches"][0].numpy() # N * 4
            eval_coord = gt_2D_matches[:, :2]
            gt_points = gt_2D_matches[:, 2:]
            eval_mode = 'gt_match'
            ransac_mode = 'homo' if 'FIRE' in cfg['npz_list_path'] else 'affine'
        elif homography_gt.sum() != 0:
            h, w = img0.shape[0], img0.shape[1]
            eval_coord = np.array([[0, 0], [0, h], [w, 0], [w, h]])
            ransac_mode = 'affine'
            assert homography_gt.sum() != 0, f"Evaluation should either using gt match, or using gt homography warp."
        else: 
            eval_mode = 'pose_error'
            K0 = data['K0'].cpu().numpy()[0]
            K1 = data['K1'].cpu().numpy()[0]
            T_0to1 = data['T_0to1'].cpu().numpy()[0]
            estimator = 'pose'
        
        # Perform matching
        if matcher is not None:
            if eval_mode in ['gt_match']:
                data.update({'query_points': torch.from_numpy(eval_coord)[None]})
            batch = dict_to_cuda(data)

            with torch.no_grad():
                with torch.autocast(enabled=config.LOFTR.FP16, device_type='cuda'):
                    matcher(batch)

                mkpts0 = batch['mkpts0_f'].cpu().numpy()
                mkpts1 = batch['mkpts1_f'].cpu().numpy()
                mconf = batch['mconf'].cpu().numpy()

            name0 = data["pair_names"][0][0]
            name1 = data["pair_names"][1][0]
            
            reference_img_shape = tuple(int(v) for v in data["origin_img_size0"].squeeze(0))
            source_img_shape = tuple(int(v) for v in data["origin_img_size1"].squeeze(0))
            
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


            # Get warpped points by homography:
            if estimator == "ransac_affine":
                H_est, _ = estimate_homo(mkpts0, mkpts1, thresh=cfg['rigid_ransac_thr'], mode=ransac_mode)
                # Warp points for eval:
                eval_points_warpped = warp_points(eval_coord, H_est, inverse=False)

                # Warp images and blend:
                if cfg['plot_align']:
                    warp_img_and_blend(img0, img1, H_est, save_path=Path(vis_output_path)/'aligned'/f"{pair_name}_{cfg['method']}.png", alpha=0.5, inverse=True)
            elif estimator == 'pose':
                pose = estimate_pose(mkpts0, mkpts1, K0, K1, cfg['rigid_ransac_thr'], conf=0.99999)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        if eval_mode == 'pose_error':
            if pose is None:
                t_err, R_err = np.inf, np.inf
            else:
                R, t, inliers = pose
                t_err, R_err = relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0)
            error = max(t_err, R_err)
            errors.append(error)
            match_error = epipolar_error(mkpts0, mkpts1, T_0to1, K0, K1)
            plot_text = f"R_err_{R_err:.2}_t_err_{t_err:.2}"
            thr = 3e-3
            print(f"max_error:{error}")
        else:
            if eval_mode == 'gt_homo':
                gt_points = warp_points(eval_coord, homography_gt, inverse=False)
                match_error = np.linalg.norm(warp_points(mkpts0, homography_gt, inverse=False) - mkpts1, axis=-1)
            else:
                match_error = None
            
            thr = 5 # Pix
            error = np.mean(np.linalg.norm(eval_points_warpped - gt_points, axis=1))
            print(f"error: {error}")
            errors.append(error)

        result_dict['@-@'.join([data['pair_names'][0][0].split('/', 1)[1], data['pair_names'][1][0].split('/', 1)[1]])] = error

        if cfg['plot_matches'] and matcher is not None:
            draw_match_type='corres'
            color_type=cfg['plot_matches_color']
            plot_matches(img0, img1, mkpts0, mkpts1, mconf, vertical=False, draw_match_type=draw_match_type, alpha=cfg['plot_matches_alpha'], save_path=Path(vis_output_path)/'demo_matches'/f"{pair_name}_{draw_match_type}.pdf", inverse=False, match_error=match_error if color_type == 'error' else None, error_thr=thr, color_type=color_type)

    # Success Rate Metric:
    metric = error_auc(np.array(errors), thresholds=[5,10,20], method="success_rate")
    print(metric)

    # AUC Metric:
    metric = error_auc(np.array(errors), thresholds=[5,10,20], method='fire_paper' if 'FIRE' in cfg['npz_list_path'] else 'exact_auc')
    print(metric)

    save_h5(result_dict, (Path(cfg["output_path"]) / f'eval_{cfg["dataset_name"]}_{cfg["method"]}_{cfg["comment"]}_error.h5'))

if __name__ == "__main__":
    run_pipeline(CONFIG)