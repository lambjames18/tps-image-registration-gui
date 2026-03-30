import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from ..experiments.roma_outdoor import get_model

class MatchAnything_Model(nn.Module):
    def __init__(self, config, test_mode=False) -> None:
        super().__init__()
        self.config = config
        self.test_mode = test_mode
        self.resize_by_stretch = config['resize_by_stretch']
        self.norm_image = config['normalize_img']
        model_config = self.config['model']
        if not test_mode :
            self.model = get_model(pretrained_backbone=True, amp=model_config['amp'], coarse_backbone_type=model_config['coarse_backbone'], coarse_feat_dim=model_config['coarse_feat_dim'], medium_feat_dim=model_config['medium_feat_dim'], coarse_patch_size=model_config['coarse_patch_size']) # Train mode
        else:
            self.model = get_model(pretrained_backbone=True, amp=model_config['amp'], coarse_backbone_type=model_config['coarse_backbone'], coarse_feat_dim=model_config['coarse_feat_dim'], medium_feat_dim=model_config['medium_feat_dim'], coarse_patch_size=model_config['coarse_patch_size'], coarse_resolution=self.config['test_time']['coarse_res'], symmetric=self.config['test_time']['symmetric'], upsample_preds=self.config['test_time']['upsample'], attenuate_cert=self.config['test_time']['attenutate_cert']) # Test mode
            self.model.upsample_res = self.config['test_time']['upsample_res'] 
        self.model.sample_mode = self.config['sample']['method']
        self.model.sample_thresh = self.config['sample']['thresh']

    def forward(self, data):
        if not self.test_mode:
            return self.forward_train_framework(data)
        else:
            return self.forward_inference(data)
    
    def forward_train_framework(self, data):
        # Get already resize & padded images by dataloader
        img0, img1 = data['image0'], data['image1'] # B * C * H * W
        corresps = self.model.forward({"im_A": img0, "im_B": img1}, batched=True)

        data.update({"corresps":corresps}) # for supervision

        warp, certainity = self.model.self_train_time_match(data, corresps) # batched and padded

        m_bids = []
        mkpts0_f = []
        mkpts1_f = []
        m_conf = []
        for b_id in range(warp.shape[0]):
            if self.resize_by_stretch:
                H_A, W_A = data["origin_img_size0"][b_id][0], data["origin_img_size0"][b_id][1]
                H_B, W_B = data["origin_img_size1"][b_id][0], data["origin_img_size1"][b_id][1]
            else:
                # By padding:
                H_A, W_A = data["origin_img_size0"][b_id].max(), data["origin_img_size0"][b_id].max()
                H_B, W_B = data["origin_img_size1"][b_id].max(), data["origin_img_size1"][b_id].max()
            # # Sample matches for estimation
            matches, certainity_ = self.model.sample(warp[b_id], certainity[b_id], num=self.config['sample']['n_sample'])
            kpts0, kpts1 = self.model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
            m_bids.append(torch.ones((kpts0.shape[0],), device=matches.device, dtype=torch.long) * b_id)
            mkpts0_f.append(kpts0)
            mkpts1_f.append(kpts1)
            m_conf.append(certainity_)
        data.update({'m_bids': torch.cat(m_bids), "mkpts0_f": torch.cat(mkpts0_f), "mkpts1_f": torch.cat(mkpts1_f), "mconf": torch.cat(m_conf)})
    
    def forward_inference(self, data):
        # Assume Loaded image in original image shape
        if 'image0_rgb_origin' in data:
            img0, img1 = data['image0_rgb_origin'][0], data['image1_rgb_origin'][0]
        elif 'image0_rgb' in data:
            img0, img1 = data['image0_rgb'][0], data['image1_rgb'][0]
        else:
            raise NotImplementedError
        warp, dense_certainity = self.model.self_inference_time_match(img0, img1, resize_by_stretch=self.resize_by_stretch, norm_img=self.norm_image)

        if self.resize_by_stretch:
            H_A, W_A = img0.shape[-2], img0.shape[-1]
            H_B, W_B = img1.shape[-2], img1.shape[-1]
        else:
            A_max_edge = max(img0.shape[-2:])
            H_A, W_A = A_max_edge, A_max_edge
            B_max_edge = max(img1.shape[-2:])
            H_B, W_B = B_max_edge, B_max_edge

        # Sample matches for estimation
        matches, certainity = self.model.sample(warp, dense_certainity, num=self.config['sample']['n_sample'])
        kpts0, kpts1 = self.model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)

        mask = certainity > self.config['match_thresh']
        # Mask borders:
        mask *= (kpts0[:, 0] <= img0.shape[-1]-1) * (kpts0[:, 1] <= img0.shape[-2]-1) * (kpts1[:, 0] <= img1.shape[-1]-1) * (kpts1[:, 1] <= img1.shape[-2]-1)
        data.update({'m_bids': torch.zeros_like(kpts0[:, 0])[mask], "mkpts0_f": kpts0[mask], "mkpts1_f": kpts1[mask], "mconf": certainity[mask]})

        # Warp query points:
        if 'query_points' in data:
            detector_kpts0 = data['query_points'].to(torch.float32) # B * N * 2
            within_mask = (detector_kpts0[..., 0] >= 0) & (detector_kpts0[..., 0] <= (W_A - 1)) & (detector_kpts0[..., 1] >= 0) & (detector_kpts0[..., 1] <= (H_A - 1))
            internal_detector_kpts0 = detector_kpts0[within_mask] 
            warped_detector_kpts0, cert_A_to_B = self.model.warp_keypoints(internal_detector_kpts0, warp, dense_certainity, H_A, W_A, H_B, W_B)
            data.update({"query_points_warpped": warped_detector_kpts0})
        return data

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)