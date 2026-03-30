import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.geometry.subpix import dsnt
from kornia.utils.grid import create_meshgrid

from loguru import logger

class FineMatching(nn.Module):
    """FineMatching with s2d paradigm"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.topk = config['match_fine']['topk']
        self.mtd_spvs = config['fine']['mtd_spvs']
        self.align_corner = config['align_corner']
        self.fix_bias = config['fix_bias']
        self.normfinem = config['match_fine']['normfinem']
        self.fix_fine_matching = config['match_fine']['fix_fine_matching']
        self.mutual_nearest = config['match_fine']['force_nearest']
        self.skip_fine_softmax = config['match_fine']['skip_fine_softmax']
        self.normfeat = config['match_fine']['normfeat']
        self.use_sigmoid = config['match_fine']['use_sigmoid']
        self.local_regress = config['match_fine']['local_regress']
        self.local_regress_rmborder = config['match_fine']['local_regress_rmborder']
        self.local_regress_nomask = config['match_fine']['local_regress_nomask']
        self.local_regress_temperature = config['match_fine']['local_regress_temperature']
        self.local_regress_padone = config['match_fine']['local_regress_padone']
        self.local_regress_slice = config['match_fine']['local_regress_slice']
        self.local_regress_slicedim = config['match_fine']['local_regress_slicedim']
        self.local_regress_inner = config['match_fine']['local_regress_inner']
        self.multi_regress = config['match_fine']['multi_regress']
    def forward(self, feat_0, feat_1, data):
        """
        Args:
            feat0 (torch.Tensor): [M, WW, C]
            feat1 (torch.Tensor): [M, WW, C]
            data (dict)
        Update:
            data (dict):{
                'expec_f' (torch.Tensor): [M, 3],
                'mkpts0_f' (torch.Tensor): [M, 2],
                'mkpts1_f' (torch.Tensor): [M, 2]}
        """
        M, WW, C = feat_0.shape
        W = int(math.sqrt(WW))
        if self.fix_bias:
            scale = 2
        else:
            scale = data['hw0_i'][0] / data['hw0_f'][0]
        self.M, self.W, self.WW, self.C, self.scale = M, W, WW, C, scale

        # corner case: if no coarse matches found
        if M == 0:
            assert self.training == False, "M is always >0, when training, see coarse_matching.py"
            # logger.warning('No matches found in coarse-level.')
            if self.mtd_spvs:
                data.update({
                    'conf_matrix_f': torch.empty(0, WW, WW, device=feat_0.device),
                    'mkpts0_f': data['mkpts0_c'],
                    'mkpts1_f': data['mkpts1_c'],
                })
                # if self.local_regress:
                #     data.update({
                #         'sim_matrix_f': torch.empty(0, WW, WW, device=feat_0.device),
                #     })
                return
            else:
                data.update({
                    'expec_f': torch.empty(0, 3, device=feat_0.device),
                    'mkpts0_f': data['mkpts0_c'],
                    'mkpts1_f': data['mkpts1_c'],
                })
                return

        if self.mtd_spvs:
            with torch.autocast(enabled=False, device_type='cuda'):
                # feat_0 = feat_0 / feat_0.size(-2)
                if self.local_regress_slice:
                    feat_ff0, feat_ff1 = feat_0[...,-self.local_regress_slicedim:], feat_1[...,-self.local_regress_slicedim:]
                    feat_f0, feat_f1 = feat_0[...,:-self.local_regress_slicedim], feat_1[...,:-self.local_regress_slicedim]
                    conf_matrix_ff = torch.einsum('mlc,mrc->mlr', feat_ff0, feat_ff1 / (self.local_regress_slicedim)**.5)
                else:
                    feat_f0, feat_f1 = feat_0, feat_1
                if self.normfinem:
                    feat_f0 = feat_f0 / C**.5
                    feat_f1 = feat_f1 / C**.5
                    conf_matrix_f = torch.einsum('mlc,mrc->mlr', feat_f0, feat_f1)
                else:
                    if self.local_regress_slice:
                        conf_matrix_f = torch.einsum('mlc,mrc->mlr', feat_f0, feat_f1 / (C - self.local_regress_slicedim)**.5)
                    else:
                        conf_matrix_f = torch.einsum('mlc,mrc->mlr', feat_f0, feat_f1 / C**.5)
                
                if self.normfeat:
                    feat_f0, feat_f1 = torch.nn.functional.normalize(feat_f0.float(), p=2, dim=-1), torch.nn.functional.normalize(feat_f1.float(), p=2, dim=-1)

            if self.config['fp16log']:
                logger.info(f'sim_matrix: {conf_matrix_f.abs().max()}')
            # sim_matrix *= 1. / C**.5 # normalize

            if self.multi_regress:
                assert not self.local_regress
                assert not self.normfinem and not self.normfeat
                heatmap = F.softmax(conf_matrix_f, 2).view(M, WW, W, W) # [M, WW, W, W]
                
                assert (W - 2) == (self.config['resolution'][0] // self.config['resolution'][1]) # c8
                windows_scale = (W - 1) / (self.config['resolution'][0] // self.config['resolution'][1])
                
                coords_normalized = dsnt.spatial_expectation2d(heatmap, True) * windows_scale # [M, WW, 2]
                grid_normalized = create_meshgrid(W, W, True, heatmap.device).reshape(1, -1, 2)[:,None,:,:] * windows_scale # [1, 1, WW, 2]
                
                # compute std over <x, y>
                var = torch.sum(grid_normalized**2 * heatmap.view(M, WW, WW, 1), dim=-2) - coords_normalized**2 # ([1,1,WW,2] * [M,WW,WW,1])->[M,WW,2]
                std = torch.sum(torch.sqrt(torch.clamp(var, min=1e-10)), -1) # [M,WW]  clamp needed for numerical stability
                
                # for fine-level supervision
                data.update({'expec_f': torch.cat([coords_normalized, std.unsqueeze(-1)], -1)}) # [M, WW, 2]
                
                # get the least uncertain matches                
                val, idx = torch.topk(std, self.topk, dim=-1, largest=False) # [M,topk]
                coords_normalized = coords_normalized[torch.arange(M, device=conf_matrix_f.device, dtype=torch.long)[:,None], idx] # [M,topk]
                
                grid = create_meshgrid(W, W, False, idx.device) - W // 2 + 0.5 # [1, W, W, 2]
                grid = grid.reshape(1, -1, 2).expand(M, -1, -1) # [M, WW, 2]
                delta_l = torch.gather(grid, 1, idx.unsqueeze(-1).expand(-1, -1, 2)) # [M, topk, 2] in (x, y)

                # compute absolute kpt coords
                self.get_multi_fine_match_align(delta_l, coords_normalized, data)

                
            else:

                if self.skip_fine_softmax:
                    pass
                elif self.use_sigmoid:
                    conf_matrix_f = torch.sigmoid(conf_matrix_f)
                else:
                    if self.local_regress:
                        del feat_f0, feat_f1
                        softmax_matrix_f = F.softmax(conf_matrix_f, 1) * F.softmax(conf_matrix_f, 2)
                        # softmax_matrix_f = conf_matrix_f
                        if self.local_regress_inner:
                            softmax_matrix_f = softmax_matrix_f.reshape(M, self.WW, self.W+2, self.W+2)
                            softmax_matrix_f = softmax_matrix_f[...,1:-1,1:-1].reshape(M, self.WW, self.WW)
                        # if self.training:
                        # for fine-level supervision
                        data.update({'conf_matrix_f': softmax_matrix_f})
                        if self.local_regress_slice:
                            data.update({'sim_matrix_ff': conf_matrix_ff})
                        else:
                            data.update({'sim_matrix_f': conf_matrix_f})

                    else:
                        conf_matrix_f = F.softmax(conf_matrix_f, 1) * F.softmax(conf_matrix_f, 2)

                        # for fine-level supervision
                        data.update({'conf_matrix_f': conf_matrix_f})

                # compute absolute kpt coords
                if self.local_regress:
                    self.get_fine_ds_match(softmax_matrix_f, data)
                    del softmax_matrix_f
                    idx_l, idx_r = data['idx_l'], data['idx_r']
                    del data['idx_l'], data['idx_r']
                    m_ids = torch.arange(M, device=idx_l.device, dtype=torch.long).unsqueeze(-1).expand(-1, self.topk)
                    # if self.training:
                    m_ids = m_ids[:len(data['mconf']) // self.topk]
                    idx_r_iids, idx_r_jids = idx_r // W, idx_r % W
                    
                    # remove boarder
                    if self.local_regress_nomask:
                        # log for inner precent
                        # mask = (idx_r_iids >= 1) & (idx_r_iids <= W-2) & (idx_r_jids >= 1) & (idx_r_jids <= W-2)
                        # mask_sum = mask.sum()
                        # logger.info(f'total fine match: {mask.numel()}; regressed fine match: {mask_sum}, per: {mask_sum / mask.numel()}')
                        mask = None                        
                        m_ids, idx_l, idx_r_iids, idx_r_jids = m_ids.reshape(-1), idx_l.reshape(-1), idx_r_iids.reshape(-1), idx_r_jids.reshape(-1)
                        if self.local_regress_inner: # been sliced before
                            delta = create_meshgrid(3, 3, True, conf_matrix_f.device).to(torch.long) # [1, 3, 3, 2]
                        else:
                            # no mask + 1 for padding
                            delta = create_meshgrid(3, 3, True, conf_matrix_f.device).to(torch.long) + torch.tensor([1], dtype=torch.long, device=conf_matrix_f.device) # [1, 3, 3, 2]
                        
                        m_ids = m_ids[...,None,None].expand(-1, 3, 3)
                        idx_l = idx_l[...,None,None].expand(-1, 3, 3) # [m, k, 3, 3]

                        idx_r_iids = idx_r_iids[...,None,None].expand(-1, 3, 3) + delta[None, ..., 1]
                        idx_r_jids = idx_r_jids[...,None,None].expand(-1, 3, 3) + delta[None, ..., 0]
                        
                        if idx_l.numel() == 0:
                            data.update({
                                'mkpts0_f': data['mkpts0_c'],
                                'mkpts1_f': data['mkpts1_c'],
                            })
                            return
                        
                        if self.local_regress_slice:
                            conf_matrix_f = conf_matrix_ff
                        if self.local_regress_inner:
                            conf_matrix_f = conf_matrix_f.reshape(M, self.WW, self.W+2, self.W+2)
                        else:
                            conf_matrix_f = conf_matrix_f.reshape(M, self.WW, self.W, self.W)
                            conf_matrix_f = F.pad(conf_matrix_f, (1,1,1,1))
                    else:
                        mask = (idx_r_iids >= 1) & (idx_r_iids <= W-2) & (idx_r_jids >= 1) & (idx_r_jids <= W-2)
                        if W == 10:
                            idx_l_iids, idx_l_jids = idx_l // W, idx_l % W
                            mask = mask & (idx_l_iids >= 1) & (idx_l_iids <= W-2) & (idx_l_jids >= 1) & (idx_l_jids <= W-2)                        
                        
                        m_ids = m_ids[mask].to(torch.long)
                        idx_l, idx_r_iids, idx_r_jids = idx_l[mask].to(torch.long), idx_r_iids[mask].to(torch.long), idx_r_jids[mask].to(torch.long)

                        m_ids, idx_l, idx_r_iids, idx_r_jids = m_ids.reshape(-1), idx_l.reshape(-1), idx_r_iids.reshape(-1), idx_r_jids.reshape(-1)
                        mask = mask.reshape(-1)

                        delta = create_meshgrid(3, 3, True, conf_matrix_f.device).to(torch.long) # [1, 3, 3, 2]

                        m_ids = m_ids[:,None,None].expand(-1, 3, 3)
                        idx_l = idx_l[:,None,None].expand(-1, 3, 3) # [m, 3, 3]
                        # bug !!!!!!!!! 1,0 rather 0,1
                        # idx_r_iids = idx_r_iids[...,None,None].expand(-1, 3, 3) + delta[None, ..., 0]
                        # idx_r_jids = idx_r_jids[...,None,None].expand(-1, 3, 3) + delta[None, ..., 1]
                        idx_r_iids = idx_r_iids[:,None,None].expand(-1, 3, 3) + delta[..., 1]
                        idx_r_jids = idx_r_jids[:,None,None].expand(-1, 3, 3) + delta[..., 0]
                        
                        if idx_l.numel() == 0:
                            data.update({
                                'mkpts0_f': data['mkpts0_c'],
                                'mkpts1_f': data['mkpts1_c'],
                            })
                            return
                        if not self.local_regress_slice:
                            conf_matrix_f = conf_matrix_f.reshape(M, self.WW, self.W, self.W)
                        else:
                            conf_matrix_f = conf_matrix_ff.reshape(M, self.WW, self.W, self.W)
                    
                    conf_matrix_f = conf_matrix_f[m_ids, idx_l, idx_r_iids, idx_r_jids]
                    conf_matrix_f = conf_matrix_f.reshape(-1, 9)
                    if self.local_regress_padone: # follow the training detach the gradient of center
                        conf_matrix_f[:,4] = -1e4
                        heatmap = F.softmax(conf_matrix_f / self.local_regress_temperature, -1)
                        logger.info(f'maxmax&maxmean of heatmap: {heatmap.view(-1).max()}, {heatmap.view(-1).min(), heatmap.max(-1)[0].mean()}')
                        heatmap[:,4] = 1.0 # no need gradient calculation in inference
                        logger.info(f'min of heatmap: {heatmap.view(-1).min()}')
                        heatmap = heatmap.reshape(-1, 3, 3)
                        # heatmap = torch.ones_like(softmax) # ones_like for detach the gradient of center
                        # heatmap[:,:4], heatmap[:,5:] = softmax[:,:4], softmax[:,5:]
                        # heatmap = heatmap.reshape(-1, 3, 3)
                    else:
                        conf_matrix_f = F.softmax(conf_matrix_f / self.local_regress_temperature, -1)
                        # logger.info(f'max&min&mean of heatmap: {conf_matrix_f.view(-1).max()}, {conf_matrix_f.view(-1).min(), conf_matrix_f.max(-1)[0].mean()}')
                        heatmap = conf_matrix_f.reshape(-1, 3, 3)
                    
                    # compute coordinates from heatmap
                    coords_normalized = dsnt.spatial_expectation2d(heatmap[None], True)[0]
                    
                    # coords_normalized_l2 = coords_normalized.norm(p=2, dim=-1)
                    # logger.info(f'mean&max&min abs of local: {coords_normalized_l2.mean(), coords_normalized_l2.max(), coords_normalized_l2.min()}')
                    
                    # compute absolute kpt coords
                    
                    if data['bs'] == 1:
                        scale1 = scale * data['scale1'] if 'scale0' in data else scale
                    else:
                        if mask is not None:
                            scale1 = scale * data['scale1'][data['b_ids']][:len(data['mconf']) // self.topk,...][:,None,:].expand(-1, self.topk, 2).reshape(-1, 2)[mask] if 'scale0' in data else scale
                        else:
                            scale1 = scale * data['scale1'][data['b_ids']][:len(data['mconf']) // self.topk,...][:,None,:].expand(-1, self.topk, 2).reshape(-1, 2) if 'scale0' in data else scale

                    self.get_fine_match_local(coords_normalized, data, scale1, mask, True)
                    
                else:
                    self.get_fine_ds_match(conf_matrix_f, data)
                    
            
        else:
            if self.align_corner is True:
                feat_f0, feat_f1 = feat_0, feat_1
                feat_f0_picked = feat_f0_picked = feat_f0[:, WW//2, :]
                sim_matrix = torch.einsum('mc,mrc->mr', feat_f0_picked, feat_f1)
                softmax_temp = 1. / C**.5
                heatmap = torch.softmax(softmax_temp * sim_matrix, dim=1).view(-1, W, W)

                # compute coordinates from heatmap
                coords_normalized = dsnt.spatial_expectation2d(heatmap[None], True)[0]  # [M, 2]
                grid_normalized = create_meshgrid(W, W, True, heatmap.device).reshape(1, -1, 2)  # [1, WW, 2]

                # compute std over <x, y>
                var = torch.sum(grid_normalized**2 * heatmap.view(-1, WW, 1), dim=1) - coords_normalized**2  # [M, 2]
                std = torch.sum(torch.sqrt(torch.clamp(var, min=1e-10)), -1)  # [M]  clamp needed for numerical stability
                
                # for fine-level supervision
                data.update({'expec_f': torch.cat([coords_normalized, std.unsqueeze(1)], -1)})

                # compute absolute kpt coords
                self.get_fine_match(coords_normalized, data)
            else:
                feat_f0, feat_f1 = feat_0, feat_1
                # even matching windows while coarse grid not aligned to fine grid!!!
                # assert W == 5, "others size not checked"
                if self.fix_bias:
                    assert W % 2 == 1, "W must be odd when select"
                    feat_f0_picked = feat_f0[:, WW//2]
                
                else:
                    # assert W == 6, "others size not checked"
                    assert W % 2 == 0, "W must be even when coarse grid not aligned to fine grid(average)"
                    feat_f0_picked = (feat_f0[:, WW//2 - W//2 - 1] + feat_f0[:, WW//2 - W//2] + feat_f0[:, WW//2 + W//2] + feat_f0[:, WW//2 + W//2 - 1]) / 4
                sim_matrix = torch.einsum('mc,mrc->mr', feat_f0_picked, feat_f1)
                softmax_temp = 1. / C**.5
                heatmap = torch.softmax(softmax_temp * sim_matrix, dim=1).view(-1, W, W)
                
                # compute coordinates from heatmap
                windows_scale = (W - 1) / (self.config['resolution'][0] // self.config['resolution'][1])
                
                coords_normalized = dsnt.spatial_expectation2d(heatmap[None], True)[0] * windows_scale # [M, 2]
                grid_normalized = create_meshgrid(W, W, True, heatmap.device).reshape(1, -1, 2) * windows_scale # [1, WW, 2]
                
                # compute std over <x, y>
                var = torch.sum(grid_normalized**2 * heatmap.view(-1, WW, 1), dim=1) - coords_normalized**2 # [M, 2]
                std = torch.sum(torch.sqrt(torch.clamp(var, min=1e-10)), -1) # [M]  clamp needed for numerical stability
                
                # for fine-level supervision
                data.update({'expec_f': torch.cat([coords_normalized, std.unsqueeze(1)], -1)})
                
                # compute absolute kpt coords
                self.get_fine_match_align(coords_normalized, data)
                

    @torch.no_grad()
    def get_fine_match(self, coords_normed, data):
        W, WW, C, scale = self.W, self.WW, self.C, self.scale

        # mkpts0_f and mkpts1_f
        mkpts0_f = data['mkpts0_c']
        scale1 = scale * data['scale1'][data['b_ids']] if 'scale0' in data else scale
        mkpts1_f = data['mkpts1_c'] + (coords_normed * (W // 2) * scale1)[:len(data['mconf'])]

        data.update({
            "mkpts0_f": mkpts0_f,
            "mkpts1_f": mkpts1_f
        })
    
    def get_fine_match_local(self, coords_normed, data, scale1, mask, reserve_border=True):
        W, WW, C, scale = self.W, self.WW, self.C, self.scale

        if mask is None:
            mkpts0_c, mkpts1_c = data['mkpts0_c'], data['mkpts1_c']
        else:
            data['mkpts0_c'], data['mkpts1_c'] = data['mkpts0_c'].reshape(-1, 2), data['mkpts1_c'].reshape(-1, 2)
            mkpts0_c, mkpts1_c = data['mkpts0_c'][mask], data['mkpts1_c'][mask]
            mask_sum = mask.sum()
            logger.info(f'total fine match: {mask.numel()}; regressed fine match: {mask_sum}, per: {mask_sum / mask.numel()}')
        # print(mkpts0_c.shape, mkpts1_c.shape, coords_normed.shape, scale1.shape)
        # print(data['mkpts0_c'].shape, data['mkpts1_c'].shape)
        # mkpts0_f and mkpts1_f
        mkpts0_f = mkpts0_c
        mkpts1_f = mkpts1_c + (coords_normed * (3 // 2) * scale1)
        
        if reserve_border and mask is not None:
            mkpts0_f, mkpts1_f = torch.cat([mkpts0_f, data['mkpts0_c'][~mask].reshape(-1, 2)]), torch.cat([mkpts1_f, data['mkpts1_c'][~mask].reshape(-1, 2)])
        else:
            pass

        del data['mkpts0_c'], data['mkpts1_c']
        data.update({
            "mkpts0_f": mkpts0_f,
            "mkpts1_f": mkpts1_f
        })
    
    # can be used for both aligned and not aligned
    @torch.no_grad()
    def get_fine_match_align(self, coord_normed, data):
        W, WW, C, scale = self.W, self.WW, self.C, self.scale
        c2f = self.config['resolution'][0] // self.config['resolution'][1]
        # mkpts0_f and mkpts1_f
        mkpts0_f = data['mkpts0_c']
        scale1 = scale * data['scale1'][data['b_ids']] if 'scale0' in data else scale
        mkpts1_f = data['mkpts1_c'] + (coord_normed * (c2f // 2) * scale1)[:len(data['mconf'])]
        
        data.update({
            "mkpts0_f": mkpts0_f,
            "mkpts1_f": mkpts1_f
        })

    @torch.no_grad()
    def get_multi_fine_match_align(self, delta_l, coord_normed, data):
        W, WW, C, scale = self.W, self.WW, self.C, self.scale
        c2f = self.config['resolution'][0] // self.config['resolution'][1]
        # mkpts0_f and mkpts1_f
        scale0 = scale * data['scale0'][data['b_ids']] if 'scale0' in data else torch.tensor([[scale, scale]], device=delta_l.device)
        scale1 = scale * data['scale1'][data['b_ids']] if 'scale0' in data else torch.tensor([[scale, scale]], device=delta_l.device)
        mkpts0_f = (data['mkpts0_c'][:,None,:] + (delta_l * scale0[:,None,:])[:len(data['mconf']),...]).reshape(-1, 2)
        mkpts1_f = (data['mkpts1_c'][:,None,:] + (coord_normed * (c2f // 2) * scale1[:,None,:])[:len(data['mconf'])]).reshape(-1, 2)
        
        data.update({
            "mkpts0_f": mkpts0_f,
            "mkpts1_f": mkpts1_f,
            "mconf": data['mconf'][:,None].expand(-1, self.topk).reshape(-1)
        })

    @torch.no_grad()
    def get_fine_ds_match(self, conf_matrix, data):
        W, WW, C, scale = self.W, self.WW, self.C, self.scale
        
        # select topk matches
        m, _, _ = conf_matrix.shape
        
        
        if self.mutual_nearest:
            pass
            
        
        elif not self.fix_fine_matching: # only allow one2mul but mul2one
        
            val, idx_r = conf_matrix.max(-1) # (m, WW), (m, WW)
            val, idx_l = torch.topk(val, self.topk, dim = -1) # (m, topk), (m, topk)
            idx_r = torch.gather(idx_r, 1, idx_l) # (m, topk)
            
            # mkpts0_c use xy coordinate, so we don't need to convert it to hw coordinate
            # grid = create_meshgrid(W, W, False, conf_matrix.device).transpose(-3,-2) - W // 2 + 0.5 # (1, W, W, 2)
            grid = create_meshgrid(W, W, False, conf_matrix.device) - W // 2 + 0.5 # (1, W, W, 2)
            grid = grid.reshape(1, -1, 2).expand(m, -1, -1) # (m, WW, 2)
            delta_l = torch.gather(grid, 1, idx_l.unsqueeze(-1).expand(-1, -1, 2)) # (m, topk, 2)
            delta_r = torch.gather(grid, 1, idx_r.unsqueeze(-1).expand(-1, -1, 2)) # (m, topk, 2)
            
            # mkpts0_f and mkpts1_f
            scale0 = scale * data['scale0'][data['b_ids']] if 'scale0' in data else scale
            scale1 = scale * data['scale1'][data['b_ids']] if 'scale0' in data else scale
            
            if torch.is_tensor(scale0) and scale0.numel() > 1: # num of scale0 > 1
                mkpts0_f = (data['mkpts0_c'][:,None,:] + (delta_l * scale0[:,None,:])[:len(data['mconf']),...]).reshape(-1, 2)
                mkpts1_f = (data['mkpts1_c'][:,None,:] + (delta_r * scale1[:,None,:])[:len(data['mconf']),...]).reshape(-1, 2)
            else: # scale0 is a float
                mkpts0_f = (data['mkpts0_c'][:,None,:] + (delta_l * scale0)[:len(data['mconf']),...]).reshape(-1, 2)
                mkpts1_f = (data['mkpts1_c'][:,None,:] + (delta_r * scale1)[:len(data['mconf']),...]).reshape(-1, 2)

        else: # allow one2mul mul2one and mul2mul
            conf_matrix = conf_matrix.reshape(m, -1)
            if self.local_regress: # for the compatibility of former config
                conf_matrix = conf_matrix[:len(data['mconf']),...]
            val, idx = torch.topk(conf_matrix, self.topk, dim = -1)
            idx_l = idx // WW
            idx_r = idx % WW

            if self.local_regress:
                data.update({'idx_l': idx_l, 'idx_r': idx_r})

            # mkpts0_c use xy coordinate, so we don't need to convert it to hw coordinate
            # grid = create_meshgrid(W, W, False, conf_matrix.device).transpose(-3,-2) - W // 2 + 0.5 # (1, W, W, 2)
            grid = create_meshgrid(W, W, False, conf_matrix.device) - W // 2 + 0.5
            grid = grid.reshape(1, -1, 2).expand(m, -1, -1)
            delta_l = torch.gather(grid, 1, idx_l.unsqueeze(-1).expand(-1, -1, 2))
            delta_r = torch.gather(grid, 1, idx_r.unsqueeze(-1).expand(-1, -1, 2))
        
            # mkpts0_f and mkpts1_f
            scale0 = scale * data['scale0'][data['b_ids']] if 'scale0' in data else scale
            scale1 = scale * data['scale1'][data['b_ids']] if 'scale0' in data else scale
            
            if self.local_regress:
                if torch.is_tensor(scale0) and scale0.numel() > 1: # num of scale0 > 1
                    mkpts0_f = (data['mkpts0_c'][:,None,:] + (delta_l * scale0[:len(data['mconf']),...][:,None,:])).reshape(-1, 2)
                    mkpts1_f = (data['mkpts1_c'][:,None,:] + (delta_r * scale1[:len(data['mconf']),...][:,None,:])).reshape(-1, 2)
                else: # scale0 is a float
                    mkpts0_f = (data['mkpts0_c'][:,None,:] + (delta_l * scale0)).reshape(-1, 2)
                    mkpts1_f = (data['mkpts1_c'][:,None,:] + (delta_r * scale1)).reshape(-1, 2)
                
            else:
                if torch.is_tensor(scale0) and scale0.numel() > 1: # num of scale0 > 1
                    mkpts0_f = (data['mkpts0_c'][:,None,:] + (delta_l * scale0[:,None,:])[:len(data['mconf']),...]).reshape(-1, 2)
                    mkpts1_f = (data['mkpts1_c'][:,None,:] + (delta_r * scale1[:,None,:])[:len(data['mconf']),...]).reshape(-1, 2)
                else: # scale0 is a float
                    mkpts0_f = (data['mkpts0_c'][:,None,:] + (delta_l * scale0)[:len(data['mconf']),...]).reshape(-1, 2)
                    mkpts1_f = (data['mkpts1_c'][:,None,:] + (delta_r * scale1)[:len(data['mconf']),...]).reshape(-1, 2)
        del data['mkpts0_c'], data['mkpts1_c']
        data['mconf'] = data['mconf'].reshape(-1, 1).expand(-1, self.topk).reshape(-1)
        # data['mconf'] = val.reshape(-1)[:len(data['mconf'])]*0.1 + data['mconf']
        
        if self.local_regress:
            data.update({
                "mkpts0_c": mkpts0_f,
                "mkpts1_c": mkpts1_f
            })
        else:
            data.update({
                "mkpts0_f": mkpts0_f,
                "mkpts1_f": mkpts1_f
            })
            
