from math import log
from loguru import logger as loguru_logger

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from kornia.utils import create_meshgrid

from .geometry import warp_kpts, homo_warp_kpts, homo_warp_kpts_glue, homo_warp_kpts_with_mask, homo_warp_kpts_with_mask_f, homo_warp_kpts_glue_with_mask, homo_warp_kpts_glue_with_mask_f, warp_kpts_by_sparse_gt_matches_fast, warp_kpts_by_sparse_gt_matches_fine_chunks

from kornia.geometry.subpix import dsnt
from kornia.utils.grid import create_meshgrid

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

##############  ↓  Coarse-Level supervision  ↓  ##############

@torch.no_grad()
def mask_pts_at_padded_regions(grid_pt, mask):
    """For megadepth dataset, zero-padding exists in images"""
    mask = repeat(mask, 'n h w -> n (h w) c', c=2)
    grid_pt[~mask.bool()] = 0
    return grid_pt


@torch.no_grad()
def spvs_coarse(data, config):
    """
    Update:
        data (dict): {
            "conf_matrix_gt": [N, hw0, hw1],
            'spv_b_ids': [M]
            'spv_i_ids': [M]
            'spv_j_ids': [M]
            'spv_w_pt0_i': [N, hw0, 2], in original image resolution
            'spv_pt1_i': [N, hw1, 2], in original image resolution
        }
        
    NOTE:
        - for scannet dataset, there're 3 kinds of resolution {i, c, f}
        - for megadepth dataset, there're 4 kinds of resolution {i, i_resize, c, f}
    """
    # 1. misc
    device = data['image0'].device
    N, _, H0, W0 = data['image0'].shape
    _, _, H1, W1 = data['image1'].shape
    
    if 'loftr' in config.METHOD:
        scale = config['LOFTR']['RESOLUTION'][0]

    scale0 = scale * data['scale0'][:, None] if 'scale0' in data else scale
    scale1 = scale * data['scale1'][:, None] if 'scale0' in data else scale
    h0, w0, h1, w1 = map(lambda x: x // scale, [H0, W0, H1, W1])

    if config['LOFTR']['MATCH_COARSE']['MTD_SPVS'] and not config['LOFTR']['FORCE_LOOP_BACK']:
        # 2. warp grids
        # create kpts in meshgrid and resize them to image resolution
        grid_pt0_c = create_meshgrid(h0, w0, False, device).reshape(1, h0*w0, 2).repeat(N, 1, 1)    # [N, hw, 2]
        grid_pt0_i = scale0 * grid_pt0_c
        grid_pt1_c = create_meshgrid(h1, w1, False, device).reshape(1, h1*w1, 2).repeat(N, 1, 1)
        grid_pt1_i = scale1 * grid_pt1_c

        correct_0to1 = torch.zeros((grid_pt0_i.shape[0], grid_pt0_i.shape[1]), dtype=torch.bool, device=grid_pt0_i.device)
        w_pt0_i = torch.zeros_like(grid_pt0_i)

        valid_dpt_b_mask = data['T_0to1'].sum(dim=-1).sum(dim=-1) != 0
        valid_homo_warp_mask = (data['homography'].sum(dim=-1).sum(dim=-1) != 0) | (data['homo_sample_normed'].sum(dim=-1).sum(dim=-1) != 0)
        valid_gt_match_warp_mask = (data['gt_matches_mask'][:, 0] != 0) # N

        if valid_homo_warp_mask.sum() != 0:
            if data['homography'].sum()==0:
                if 'homo_mask0' in data and (data['homo_mask0'].sum()!=0):  # the key 'depth_mask' only exits when using the dataste "CommonDataSetHomoWarp"
                    correct_0to1_homo, w_pt0_i_homo = homo_warp_kpts_with_mask(grid_pt0_i[valid_homo_warp_mask], scale, data['homo_mask0'][valid_homo_warp_mask], data['norm_pixel_mat'][valid_homo_warp_mask], data['homo_sample_normed'][valid_homo_warp_mask], original_size1=data['origin_img_size1'][valid_homo_warp_mask])
                else:
                    correct_0to1_homo, w_pt0_i_homo = homo_warp_kpts(grid_pt0_i[valid_homo_warp_mask], data['norm_pixel_mat'][valid_homo_warp_mask], \
                                                        data['homo_sample_normed'][valid_homo_warp_mask], original_size1=data['origin_img_size1'][valid_homo_warp_mask])
            else:
                if 'homo_mask0' in data and (data['homo_mask0']==0).sum()!=0:
                    correct_0to1_homo, w_pt0_i_homo = homo_warp_kpts_glue_with_mask(grid_pt0_i[valid_homo_warp_mask], scale, data['homo_mask0'][valid_homo_warp_mask], data['homography'][valid_homo_warp_mask], original_size1=data['origin_img_size1'][valid_homo_warp_mask])
                else:
                    correct_0to1_homo, w_pt0_i_homo = homo_warp_kpts_glue(grid_pt0_i[valid_homo_warp_mask], data['homography'][valid_homo_warp_mask], \
                                                        original_size1=data['origin_img_size1'][valid_homo_warp_mask])
            correct_0to1[valid_homo_warp_mask] = correct_0to1_homo
            w_pt0_i[valid_homo_warp_mask] = w_pt0_i_homo
        
        if valid_gt_match_warp_mask.sum() != 0:
            correct_0to1_dpt, w_pt0_i_dpt = warp_kpts_by_sparse_gt_matches_fast(grid_pt0_i[valid_gt_match_warp_mask], data['gt_matches'][valid_gt_match_warp_mask], scale0=scale0[valid_gt_match_warp_mask], current_h=h0, current_w=w0)
            correct_0to1[valid_gt_match_warp_mask] = correct_0to1_dpt
            w_pt0_i[valid_gt_match_warp_mask] = w_pt0_i_dpt

        if valid_dpt_b_mask.sum() != 0:
            correct_0to1_dpt, w_pt0_i_dpt = warp_kpts(grid_pt0_i[valid_dpt_b_mask], data['depth0'][valid_dpt_b_mask], data['depth1'][valid_dpt_b_mask], data['T_0to1'][valid_dpt_b_mask], data['K0'][valid_dpt_b_mask], data['K1'][valid_dpt_b_mask], consistency_thr=0.05)
            correct_0to1[valid_dpt_b_mask] = correct_0to1_dpt
            w_pt0_i[valid_dpt_b_mask] = w_pt0_i_dpt

        w_pt0_c = w_pt0_i / scale1

        # 3. check if mutual nearest neighbor
        w_pt0_c_round = w_pt0_c[:, :, :].round() # [N, hw, 2]
        if config.LOFTR.LOSS.COARSE_OVERLAP_WEIGHT:
            w_pt0_c_error = (1.0 - 2*torch.abs(w_pt0_c - w_pt0_c_round)).prod(-1)
        w_pt0_c_round = w_pt0_c_round.long() # [N, hw, 2]
        nearest_index1 = w_pt0_c_round[..., 0] + w_pt0_c_round[..., 1] * w1 # [N, hw]

        # corner case: out of boundary
        def out_bound_mask(pt, w, h):
            return (pt[..., 0] < 0) + (pt[..., 0] >= w) + (pt[..., 1] < 0) + (pt[..., 1] >= h)
        nearest_index1[out_bound_mask(w_pt0_c_round, w1, h1)] = -1

        correct_0to1[:, 0] = False  # ignore the top-left corner
        
        # 4. construct a gt conf_matrix
        mask1 = torch.stack([data['mask1'].reshape(-1, h1*w1)[_b, _i] for _b, _i in enumerate(nearest_index1)], dim=0)
        correct_0to1 = correct_0to1 * data['mask0'].reshape(-1, h0*w0) * mask1
        
        conf_matrix_gt = torch.zeros(N, h0*w0, h1*w1, device=device, dtype=torch.bool)
        b_ids, i_ids = torch.where(correct_0to1 != 0)
        j_ids = nearest_index1[b_ids, i_ids]
        valid_j_ids = j_ids != -1
        b_ids, i_ids, j_ids = map(lambda x: x[valid_j_ids], [b_ids, i_ids, j_ids])

        conf_matrix_gt[b_ids, i_ids, j_ids] = 1
        
        # overlap weight
        if config.LOFTR.LOSS.COARSE_OVERLAP_WEIGHT:
            conf_matrix_error_gt = w_pt0_c_error[b_ids, i_ids]
            assert torch.all(conf_matrix_error_gt >= -0.001)
            assert torch.all(conf_matrix_error_gt <= 1.001)
            data.update({'conf_matrix_error_gt': conf_matrix_error_gt})
        data.update({'conf_matrix_gt': conf_matrix_gt})

        # 5. save coarse matches(gt) for training fine level
        if len(b_ids) == 0:
            loguru_logger.warning(f"No groundtruth coarse match found for: {data['pair_names']}")
            # this won't affect fine-level loss calculation
            b_ids = torch.tensor([0], device=device)
            i_ids = torch.tensor([0], device=device)
            j_ids = torch.tensor([0], device=device)

        data.update({
            'spv_b_ids': b_ids,
            'spv_i_ids': i_ids,
            'spv_j_ids': j_ids
        })

        data.update({'mkpts0_c_gt_b_ids': b_ids})
        data.update({'mkpts0_c_gt': torch.stack([i_ids % w0, i_ids // w0], dim=-1) * scale0[b_ids, 0]})
        data.update({'mkpts1_c_gt': torch.stack([j_ids % w1, j_ids // w1], dim=-1) * scale1[b_ids, 0]})

        # 6. save intermediate results (for fast fine-level computation)
        data.update({
            'spv_w_pt0_i': w_pt0_i,
            'spv_pt1_i': grid_pt1_i,
            # 'correct_0to1_c': correct_0to1 
        })
    else:
        raise NotImplementedError

def compute_supervision_coarse(data, config):
    spvs_coarse(data, config)

@torch.no_grad()
def get_gt_flow(data, h, w):
    device = data['image0'].device
    B, _, H0, W0 = data['image0'].shape
    scale = H0 / h

    scale0 = scale * data['scale0'][:, None] if 'scale0' in data else scale
    scale1 = scale * data['scale1'][:, None] if 'scale0' in data else scale

    x1_n = torch.meshgrid(
        *[
            torch.linspace(
                -1 + 1 / n, 1 - 1 / n, n, device=device
            )
            for n in (B, h, w)
        ]
    )
    grid_coord = torch.stack((x1_n[2], x1_n[1]), dim=-1).reshape(B, h*w, 2) # normalized
    grid_coord = torch.stack(
        (w * (grid_coord[..., 0] + 1) / 2, h * (grid_coord[..., 1] + 1) / 2), dim=-1
    )  # [-1+1/h, 1-1/h] -> [0.5, h-0.5]
    grid_coord_in_origin = grid_coord * scale0

    correct_0to1 = torch.zeros((grid_coord_in_origin.shape[0], grid_coord_in_origin.shape[1]), dtype=torch.bool, device=device)
    w_pt0_i = torch.zeros_like(grid_coord_in_origin)

    valid_dpt_b_mask = data['T_0to1'].sum(dim=-1).sum(dim=-1) != 0
    valid_homo_warp_mask = (data['homography'].sum(dim=-1).sum(dim=-1) != 0) | (data['homo_sample_normed'].sum(dim=-1).sum(dim=-1) != 0)
    valid_gt_match_warp_mask = (data['gt_matches_mask'] != 0)[:, 0]

    if valid_homo_warp_mask.sum() != 0:
        if data['homography'].sum()==0:
            if 'homo_mask0' in data and (data['homo_mask0'].sum()!=0):
                # data['load_mask'] = True or False,  data['depth_mask'] = depth_mask or None
                correct_0to1_homo, w_pt0_i_homo = homo_warp_kpts_with_mask(grid_coord_in_origin[valid_homo_warp_mask], int(scale), data['homo_mask0'][valid_homo_warp_mask], data['norm_pixel_mat'][valid_homo_warp_mask], \
                                                        data['homo_sample_normed'][valid_homo_warp_mask], original_size1=data['origin_img_size1'][valid_homo_warp_mask])
            else:
                correct_0to1_homo, w_pt0_i_homo = homo_warp_kpts(grid_coord_in_origin[valid_homo_warp_mask], data['norm_pixel_mat'][valid_homo_warp_mask], data['homo_sample_normed'][valid_homo_warp_mask], \
                                                            original_size1=data['origin_img_size1'][valid_homo_warp_mask])
        else:
            if 'homo_mask0' in data and (data['homo_mask0']==0).sum()!=0:
                correct_0to1_homo, w_pt0_i_homo = homo_warp_kpts_glue_with_mask(grid_coord_in_origin[valid_homo_warp_mask], int(scale), data['homo_mask0'][valid_homo_warp_mask], data['homography'][valid_homo_warp_mask], \
                                                        original_size1=data['origin_img_size1'][valid_homo_warp_mask])
            else:
                correct_0to1_homo, w_pt0_i_homo = homo_warp_kpts_glue(grid_coord_in_origin[valid_homo_warp_mask], data['homography'][valid_homo_warp_mask], \
                                                        original_size1=data['origin_img_size1'][valid_homo_warp_mask])
        correct_0to1[valid_homo_warp_mask] = correct_0to1_homo
        w_pt0_i[valid_homo_warp_mask] = w_pt0_i_homo

    if valid_gt_match_warp_mask.sum() != 0:
        correct_0to1_dpt, w_pt0_i_dpt = warp_kpts_by_sparse_gt_matches_fast(grid_coord_in_origin[valid_gt_match_warp_mask], data['gt_matches'][valid_gt_match_warp_mask], scale0=scale0[valid_gt_match_warp_mask], current_h=h, current_w=w)
        correct_0to1[valid_gt_match_warp_mask] = correct_0to1_dpt
        w_pt0_i[valid_gt_match_warp_mask] = w_pt0_i_dpt
    if valid_dpt_b_mask.sum() != 0:
        correct_0to1_dpt, w_pt0_i_dpt = warp_kpts(grid_coord_in_origin[valid_dpt_b_mask], data['depth0'][valid_dpt_b_mask], data['depth1'][valid_dpt_b_mask], data['T_0to1'][valid_dpt_b_mask], data['K0'][valid_dpt_b_mask], data['K1'][valid_dpt_b_mask], consistency_thr=0.05)
        correct_0to1[valid_dpt_b_mask] = correct_0to1_dpt
        w_pt0_i[valid_dpt_b_mask] = w_pt0_i_dpt
    
    w_pt0_c = w_pt0_i / scale1

    def out_bound_mask(pt, w, h):
        return (pt[..., 0] < 0) + (pt[..., 0] >= w) + (pt[..., 1] < 0) + (pt[..., 1] >= h)
    correct_0to1[out_bound_mask(w_pt0_c, w, h)] = 0

    w_pt0_n = torch.stack(
        (2 * w_pt0_c[..., 0] / w - 1, 2 * w_pt0_c[..., 1] / h - 1), dim=-1
    )  # from [0.5,h-0.5] -> [-1+1/h, 1-1/h]
    # w_pt1_c = w_pt1_i / scale0

    if scale > 8:
        data.update({'mkpts0_c_gt': grid_coord_in_origin[correct_0to1]})
        data.update({'mkpts1_c_gt': w_pt0_i[correct_0to1]})

    return w_pt0_n.reshape(B, h, w, 2), correct_0to1.float().reshape(B, h, w)

@torch.no_grad()
def compute_roma_supervision(data, config):
    gt_flow = {}
    for scale in list(data["corresps"]):
        scale_corresps = data["corresps"][scale]
        flow_pre_delta = rearrange(scale_corresps['flow'] if 'flow'in scale_corresps else scale_corresps['dense_flow'], "b d h w -> b h w d")
        b, h, w, d = flow_pre_delta.shape
        gt_warp, gt_prob = get_gt_flow(data, h, w)
        gt_flow[scale] = {'gt_warp': gt_warp, "gt_prob": gt_prob}
    
    data.update({"gt": gt_flow})

##############  ↓  Fine-Level supervision  ↓  ##############

@static_vars(counter = 0)
@torch.no_grad()
def spvs_fine(data, config, logger = None):
    """
    Update:
        data (dict):{
            "expec_f_gt": [M, 2]}
    """
    # 1. misc
    # w_pt0_i, pt1_i = data.pop('spv_w_pt0_i'), data.pop('spv_pt1_i')
    if config.LOFTR.FINE.MTD_SPVS:
        pt1_i = data['spv_pt1_i']
    else:
        spv_w_pt0_i, pt1_i = data['spv_w_pt0_i'], data['spv_pt1_i']
    if 'loftr' in config.METHOD:
        scale = config['LOFTR']['RESOLUTION'][1]
        scale_c = config['LOFTR']['RESOLUTION'][0]
        radius = config['LOFTR']['FINE_WINDOW_SIZE'] // 2

    # 2. get coarse prediction
    b_ids, i_ids, j_ids = data['b_ids'], data['i_ids'], data['j_ids']

    # 3. compute gt
    scalei0 = scale * data['scale0'][b_ids] if 'scale0' in data else scale
    scale0 = scale * data['scale0'] if 'scale0' in data else scale
    scalei1 = scale * data['scale1'][b_ids] if 'scale0' in data else scale
    
    if config.LOFTR.FINE.MTD_SPVS:
        W = config['LOFTR']['FINE_WINDOW_SIZE']
        WW = W*W
        device = data['image0'].device

        N, _, H0, W0 = data['image0'].shape
        _, _, H1, W1 = data['image1'].shape

        if config.LOFTR.ALIGN_CORNER is False:
            hf0, wf0, hf1, wf1 = data['hw0_f'][0], data['hw0_f'][1], data['hw1_f'][0], data['hw1_f'][1]
            hc0, wc0, hc1, wc1 = data['hw0_c'][0], data['hw0_c'][1], data['hw1_c'][0], data['hw1_c'][1]
            # loguru_logger.info('hf0, wf0, hf1, wf1', hf0, wf0, hf1, wf1)
        else:
            hf0, wf0, hf1, wf1 = map(lambda x: x // scale, [H0, W0, H1, W1])
            hc0, wc0, hc1, wc1 = map(lambda x: x // scale_c, [H0, W0, H1, W1])
        
        m = b_ids.shape[0]
        if m == 0:
            conf_matrix_f_gt = torch.zeros(m, WW, WW, device=device)
            
            data.update({'conf_matrix_f_gt': conf_matrix_f_gt})
            if config.LOFTR.LOSS.FINE_OVERLAP_WEIGHT:
                conf_matrix_f_error_gt = torch.zeros(1, device=device)
                data.update({'conf_matrix_f_error_gt': conf_matrix_f_error_gt})
            if config.LOFTR.MATCH_FINE.MULTI_REGRESS:
                data.update({'expec_f': torch.zeros(1, 3, device=device)})
                data.update({'expec_f_gt': torch.zeros(1, 2, device=device)})
                
            if config.LOFTR.MATCH_FINE.LOCAL_REGRESS:
                data.update({'expec_f': torch.zeros(1, 2, device=device)})
                data.update({'expec_f_gt': torch.zeros(1, 2, device=device)})
        else:
            grid_pt0_f = create_meshgrid(hf0, wf0, False, device) - W // 2 + 0.5 # [1, hf0, wf0, 2] # use fine coordinates
            # grid_pt0_f = create_meshgrid(hf0, wf0, False, device) + 0.5 # [1, hf0, wf0, 2] # use fine coordinates
            grid_pt0_f = rearrange(grid_pt0_f, 'n h w c -> n c h w')
            # 1. unfold(crop) all local windows
            if config.LOFTR.ALIGN_CORNER is False: # even windows
                if config.LOFTR.MATCH_FINE.MULTI_REGRESS or (config.LOFTR.MATCH_FINE.LOCAL_REGRESS and W == 10):
                    grid_pt0_f_unfold = F.unfold(grid_pt0_f, kernel_size=(W, W), stride=W-2, padding=1) # overlap windows W-2 padding=1
                else:
                    grid_pt0_f_unfold = F.unfold(grid_pt0_f, kernel_size=(W, W), stride=W, padding=0)
            else:
                grid_pt0_f_unfold = F.unfold(grid_pt0_f[..., :-1, :-1], kernel_size=(W, W), stride=W, padding=W//2)
            grid_pt0_f_unfold = rearrange(grid_pt0_f_unfold, 'n (c ww) l -> n l ww c', ww=W**2) # [1, hc0*wc0, W*W, 2]
            grid_pt0_f_unfold = repeat(grid_pt0_f_unfold[0], 'l ww c -> N l ww c', N=N)

            # 2. select only the predicted matches
            grid_pt0_f_unfold = grid_pt0_f_unfold[data['b_ids'], data['i_ids']]  # [m, ww, 2]
            grid_pt0_f_unfold = scalei0[:,None,:] * grid_pt0_f_unfold  # [m, ww, 2]
            
            # use depth mask
            if 'homo_mask0' in data and (data['homo_mask0'].sum()!=0):
                # depth_mask --> (n, 1, hf, wf)
                homo_mask0 = data['homo_mask0']
                homo_mask0 = F.unfold(homo_mask0[..., :-1, :-1], kernel_size=(W, W), stride=W, padding=W//2)
                homo_mask0 = rearrange(homo_mask0, 'n (c ww) l -> n l ww c', ww=W**2)  # [1, hc0*wc0, W*W, 1]
                homo_mask0 = repeat(homo_mask0[0], 'l ww c -> N l ww c', N=N)
                # select only the predicted matches
                homo_mask0 = homo_mask0[data['b_ids'], data['i_ids']]
                
            correct_0to1_f_list, w_pt0_i_list = [], []
            
            correct_0to1_f = torch.zeros(m, WW, device=device, dtype=torch.bool)
            w_pt0_i = torch.zeros(m, WW, 2, device=device, dtype=torch.float32)
            for b in range(N):
                mask = b_ids == b

                match = int(mask.sum())
                skip_reshape = False
                if match == 0:
                    print(f"no pred fine matches, skip!")
                    continue
                if (data['homography'][b].sum() != 0) | (data['homo_sample_normed'][b].sum() != 0):
                    if data['homography'][b].sum()==0:
                        if 'homo_mask0' in data and (data['homo_mask0'].sum()!=0):
                            correct_0to1_f_mask, w_pt0_i_mask = homo_warp_kpts_with_mask_f(grid_pt0_f_unfold[mask].reshape(1,-1,2), homo_mask0[mask].reshape(1,-1), data['norm_pixel_mat'][[b]], \
                                                                    data['homo_sample_normed'][[b]], data['origin_img_size0'][[b]], data['origin_img_size1'][[b]])
                        else:
                            correct_0to1_f_mask, w_pt0_i_mask = homo_warp_kpts(grid_pt0_f_unfold[mask].reshape(1,-1,2), data['norm_pixel_mat'][[b]], \
                                                                    data['homo_sample_normed'][[b]], data['origin_img_size0'][[b]], data['origin_img_size1'][[b]])
                    else:
                        if 'homo_mask0' in data and (data['homo_mask0'].sum()!=0):
                            correct_0to1_f_mask, w_pt0_i_mask = homo_warp_kpts_glue_with_mask_f(grid_pt0_f_unfold[mask].reshape(1,-1,2), homo_mask0[mask].reshape(1,-1), data['homography'][[b]], \
                                                                    data['origin_img_size0'][[b]], data['origin_img_size1'][[b]])
                        else:
                            correct_0to1_f_mask, w_pt0_i_mask = homo_warp_kpts_glue(grid_pt0_f_unfold[mask].reshape(1,-1,2), data['homography'][[b]], \
                                                                    data['origin_img_size0'][[b]], data['origin_img_size1'][[b]])
                elif data['T_0to1'][b].sum() != 0:
                    correct_0to1_f_mask, w_pt0_i_mask = warp_kpts(grid_pt0_f_unfold[mask].reshape(1,-1,2), data['depth0'][[b],...],
                            data['depth1'][[b],...], data['T_0to1'][[b],...], 
                            data['K0'][[b],...], data['K1'][[b],...]) # [k, WW], [k, WW, 2]
                elif data['gt_matches_mask'][b].sum() != 0:
                    correct_0to1_f_mask, w_pt0_i_mask = warp_kpts_by_sparse_gt_matches_fine_chunks(grid_pt0_f_unfold[mask], gt_matches=data['gt_matches'][[b]], dist_thr=scale0[[b]].max(dim=-1)[0])
                    skip_reshape = True
                correct_0to1_f[mask] = correct_0to1_f_mask.reshape(match, WW) if not skip_reshape else correct_0to1_f_mask
                w_pt0_i[mask] = w_pt0_i_mask.reshape(match, WW, 2) if not skip_reshape else w_pt0_i_mask

            delta_w_pt0_i = w_pt0_i - pt1_i[b_ids, j_ids][:,None,:] # [m, WW, 2]
            delta_w_pt0_f = delta_w_pt0_i / scalei1[:,None,:] + W // 2 - 0.5
            delta_w_pt0_f_round = delta_w_pt0_f[:, :, :].round()
            if config.LOFTR.LOSS.FINE_OVERLAP_WEIGHT and config.LOFTR.LOSS.FINE_OVERLAP_WEIGHT2:
                w_pt0_f_error = (1.0 - torch.abs(delta_w_pt0_f - delta_w_pt0_f_round)).prod(-1) # [0.25, 1]
            elif config.LOFTR.LOSS.FINE_OVERLAP_WEIGHT:
                w_pt0_f_error = (1.0 - 2*torch.abs(delta_w_pt0_f - delta_w_pt0_f_round)).prod(-1) # [0, 1]     
            delta_w_pt0_f_round = delta_w_pt0_f_round.long()

        
            nearest_index1 = delta_w_pt0_f_round[..., 0] + delta_w_pt0_f_round[..., 1] * W # [m, WW]
            
            def out_bound_mask(pt, w, h):
                return (pt[..., 0] < 0) + (pt[..., 0] >= w) + (pt[..., 1] < 0) + (pt[..., 1] >= h)
            ob_mask = out_bound_mask(delta_w_pt0_f_round, W, W)
            nearest_index1[ob_mask] = 0
            
            correct_0to1_f[ob_mask] = 0
            m_ids_d, i_ids_d = torch.where(correct_0to1_f != 0)
            
            j_ids_d = nearest_index1[m_ids_d, i_ids_d]

            # For plotting:
            mkpts0_f_gt = grid_pt0_f_unfold[m_ids_d, i_ids_d] # [m, 2]
            mkpts1_f_gt = w_pt0_i[m_ids_d, i_ids_d] # [m, 2]
            data.update({'mkpts0_f_gt_b_ids': m_ids_d})
            data.update({'mkpts0_f_gt': mkpts0_f_gt})
            data.update({'mkpts1_f_gt': mkpts1_f_gt})
            
            if config.LOFTR.MATCH_FINE.MULTI_REGRESS:
                assert not config.LOFTR.MATCH_FINE.LOCAL_REGRESS
                expec_f_gt = delta_w_pt0_f - W // 2 + 0.5 # use delta(e.g. [-3.5,3.5]) in regression rather than [0,W] (e.g. [0,7])
                expec_f_gt = expec_f_gt[m_ids_d, i_ids_d] / (W // 2 - 1) # specific radius for overlaped even windows & align_corner=False
                data.update({'expec_f_gt': expec_f_gt})
                data.update({'m_ids_d': m_ids_d, 'i_ids_d': i_ids_d})
            else: # spv fine dual softmax
                if config.LOFTR.MATCH_FINE.LOCAL_REGRESS:
                    expec_f_gt = delta_w_pt0_f - delta_w_pt0_f_round
                    
                    # mask fine windows boarder                
                    j_ids_d_il, j_ids_d_jl = j_ids_d // W, j_ids_d % W
                    if config.LOFTR.MATCH_FINE.LOCAL_REGRESS_NOMASK:
                        mask = None
                        m_ids_dl, i_ids_dl, j_ids_d_il, j_ids_d_jl = m_ids_d.to(torch.long), i_ids_d.to(torch.long), j_ids_d_il.to(torch.long), j_ids_d_jl.to(torch.long)
                    else:
                        mask = (j_ids_d_il >= 1) & (j_ids_d_il < W-1) & (j_ids_d_jl >= 1) & (j_ids_d_jl < W-1)
                        if W == 10:
                            i_ids_d_il, i_ids_d_jl = i_ids_d // W, i_ids_d % W
                            mask = mask & (i_ids_d_il >= 1) & (i_ids_d_il <= W-2) & (i_ids_d_jl >= 1) & (i_ids_d_jl <= W-2)                        

                        m_ids_dl, i_ids_dl, j_ids_d_il, j_ids_d_jl = m_ids_d[mask].to(torch.long), i_ids_d[mask].to(torch.long), j_ids_d_il[mask].to(torch.long), j_ids_d_jl[mask].to(torch.long)
                    if mask is not None:
                        loguru_logger.info(f'percent of gt mask.sum / mask.numel: {mask.sum().float()/mask.numel():.2f}')
                    if m_ids_dl.numel() == 0:
                        loguru_logger.warning(f"No groundtruth fine match found for local regress: {data['pair_names']}")
                        data.update({'expec_f_gt': torch.zeros(1, 2, device=device)})
                        data.update({'expec_f': torch.zeros(1, 2, device=device)})
                    else:
                        expec_f_gt = expec_f_gt[m_ids_dl, i_ids_dl]
                        data.update({"expec_f_gt": expec_f_gt})
                        
                        data.update({"m_ids_dl": m_ids_dl,
                                        "i_ids_dl": i_ids_dl,
                                        "j_ids_d_il": j_ids_d_il,
                                        "j_ids_d_jl": j_ids_d_jl
                                        })
                else: # no fine regress
                    pass 
                
                # spv fine dual softmax
                conf_matrix_f_gt = torch.zeros(m, WW, WW, device=device, dtype=torch.bool)
                conf_matrix_f_gt[m_ids_d, i_ids_d, j_ids_d] = 1
                data.update({'conf_matrix_f_gt': conf_matrix_f_gt})
                if config.LOFTR.LOSS.FINE_OVERLAP_WEIGHT:
                    w_pt0_f_error = w_pt0_f_error[m_ids_d, i_ids_d]
                    assert torch.all(w_pt0_f_error >= -0.001)
                    assert torch.all(w_pt0_f_error <= 1.001)
                    data.update({'conf_matrix_f_error_gt': w_pt0_f_error})
                
                conf_matrix_f_gt_sum = conf_matrix_f_gt.sum()
                if  conf_matrix_f_gt_sum != 0:
                    pass
                else:
                    loguru_logger.info(f'[no gt plot]no fine matches to supervise')
            
    else:
        expec_f_gt = (spv_w_pt0_i[b_ids, i_ids] - pt1_i[b_ids, j_ids]) / scalei1 / 4  # [M, 2]
        data.update({"expec_f_gt": expec_f_gt})


def compute_supervision_fine(data, config, logger=None):
    data_source = data['dataset_name'][0]
    if data_source.lower() in ['scannet', 'megadepth']:
        spvs_fine(data, config, logger)
    else:
        raise NotImplementedError