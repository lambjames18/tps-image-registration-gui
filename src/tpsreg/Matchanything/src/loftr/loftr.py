import torch
import torch.nn as nn
from einops.einops import rearrange

from .backbone import build_backbone
# from third_party.matchformer.model.backbone import build_backbone as build_backbone_matchformer
from .utils.position_encoding import PositionEncodingSine
from .loftr_module import LocalFeatureTransformer, FinePreprocess
from .utils.coarse_matching import CoarseMatching
from .utils.fine_matching import FineMatching

from loguru import logger

class LoFTR(nn.Module):
    def __init__(self, config, profiler=None):
        super().__init__()
        # Misc
        self.config = config
        self.profiler = profiler

        # Modules
        self.backbone = build_backbone(config)
        if not (self.config['coarse']['skip'] or self.config['coarse']['rope'] or self.config['coarse']['pan'] or self.config['coarse']['token_mixer'] is not None):
            self.pos_encoding = PositionEncodingSine(
                config['coarse']['d_model'],
                temp_bug_fix=config['coarse']['temp_bug_fix'],
                npe=config['coarse']['npe'],
                )
        if self.config['coarse']['abspe']:
            self.pos_encoding = PositionEncodingSine(
                config['coarse']['d_model'],
                temp_bug_fix=config['coarse']['temp_bug_fix'],
                npe=config['coarse']['npe'],
                )
            
        if self.config['coarse']['skip'] is False:
            self.loftr_coarse = LocalFeatureTransformer(config)
        self.coarse_matching = CoarseMatching(config['match_coarse'])
        # self.fine_preprocess = FinePreprocess(config).float()
        self.fine_preprocess = FinePreprocess(config)
        if self.config['fine']['skip'] is False:
            self.loftr_fine = LocalFeatureTransformer(config["fine"])
        self.fine_matching = FineMatching(config)

    def forward(self, data):
        """ 
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """
        # 1. Local Feature CNN
        data.update({
            'bs': data['image0'].size(0),
            'hw0_i': data['image0'].shape[2:], 'hw1_i': data['image1'].shape[2:]
        })

        if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
            # feats_c, feats_f = self.backbone(torch.cat([data['image0'], data['image1']], dim=0))
            ret_dict = self.backbone(torch.cat([data['image0'], data['image1']], dim=0))
            feats_c, feats_f = ret_dict['feats_c'], ret_dict['feats_f']
            if self.config['inter_feat']:
                data.update({
                    'feats_x2': ret_dict['feats_x2'],
                    'feats_x1': ret_dict['feats_x1'],
                })
            if self.config['coarse_feat_only']:
                (feat_c0, feat_c1) = feats_c.split(data['bs'])
                feat_f0, feat_f1 = None, None
            else:
                (feat_c0, feat_c1), (feat_f0, feat_f1) = feats_c.split(data['bs']), feats_f.split(data['bs'])
        else:  # handle different input shapes
            # (feat_c0, feat_f0), (feat_c1, feat_f1) = self.backbone(data['image0']), self.backbone(data['image1'])
            ret_dict0, ret_dict1 = self.backbone(data['image0']), self.backbone(data['image1'])
            feat_c0, feat_f0 = ret_dict0['feats_c'], ret_dict0['feats_f']
            feat_c1, feat_f1 = ret_dict1['feats_c'], ret_dict1['feats_f']
            if self.config['inter_feat']:
                data.update({
                    'feats_x2_0': ret_dict0['feats_x2'],
                    'feats_x1_0': ret_dict0['feats_x1'],
                    'feats_x2_1': ret_dict1['feats_x2'],
                    'feats_x1_1': ret_dict1['feats_x1'],
                })
            if self.config['coarse_feat_only']:
                feat_f0, feat_f1 = None, None


        mul = self.config['resolution'][0] // self.config['resolution'][1]
        # mul = 4
        if self.config['fix_bias']:
            data.update({
                'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:],
                'hw0_f': feat_f0.shape[2:] if feat_f0 is not None else [(feat_c0.shape[2]-1) * mul+1, (feat_c0.shape[3]-1) * mul+1] ,
                'hw1_f': feat_f1.shape[2:] if feat_f1 is not None else [(feat_c1.shape[2]-1) * mul+1, (feat_c1.shape[3]-1) * mul+1]
            })
        else:
            data.update({
                'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:],
                'hw0_f': feat_f0.shape[2:] if feat_f0 is not None else [feat_c0.shape[2] * mul, feat_c0.shape[3] * mul] ,
                'hw1_f': feat_f1.shape[2:] if feat_f1 is not None else [feat_c1.shape[2] * mul, feat_c1.shape[3] * mul]
            })

        # 2. coarse-level loftr module
        # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
        if self.config['coarse']['skip']:
            mask_c0 = mask_c1 = None  # mask is useful in training
            if 'mask0' in data:
                mask_c0, mask_c1 = data['mask0'], data['mask1']
            feat_c0 = rearrange(feat_c0, 'n c h w -> n (h w) c')
            feat_c1 = rearrange(feat_c1, 'n c h w -> n (h w) c')

        elif self.config['coarse']['pan']:
            # assert feat_c0.shape[0] == 1, 'batch size must be 1 when using mask Xformer now'
            if self.config['coarse']['abspe']:
                feat_c0 = self.pos_encoding(feat_c0)
                feat_c1 = self.pos_encoding(feat_c1)

            mask_c0 = mask_c1 = None  # mask is useful in training
            if 'mask0' in data:
                mask_c0, mask_c1 = data['mask0'], data['mask1']
            if self.config['matchability']: # else match in loftr_coarse
                feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1, data=data)
            else:
                feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)

            feat_c0 = rearrange(feat_c0, 'n c h w -> n (h w) c')
            feat_c1 = rearrange(feat_c1, 'n c h w -> n (h w) c')
        else:
            if not (self.config['coarse']['rope'] or self.config['coarse']['token_mixer'] is not None):
                feat_c0 = rearrange(self.pos_encoding(feat_c0), 'n c h w -> n (h w) c')
                feat_c1 = rearrange(self.pos_encoding(feat_c1), 'n c h w -> n (h w) c')

            mask_c0 = mask_c1 = None  # mask is useful in training
            if self.config['coarse']['rope']:
                if 'mask0' in data:
                    mask_c0, mask_c1 = data['mask0'], data['mask1']
            else:
                if 'mask0' in data:
                    mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)
            feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)
            if self.config['coarse']['rope']:
                feat_c0 = rearrange(feat_c0, 'n c h w -> n (h w) c')
                feat_c1 = rearrange(feat_c1, 'n c h w -> n (h w) c')
        
        # detect nan
        if self.config['replace_nan'] and (torch.any(torch.isnan(feat_c0)) or torch.any(torch.isnan(feat_c1))):
            logger.info(f'replace nan in coarse attention')
            logger.info(f"feat_c0_nan_num: {torch.isnan(feat_c0).int().sum()}, feat_c1_nan_num: {torch.isnan(feat_c1).int().sum()}")
            logger.info(f"feat_c0: {feat_c0}, feat_c1: {feat_c1}")
            logger.info(f"feat_c0_max: {feat_c0.abs().max()}, feat_c1_max: {feat_c1.abs().max()}")
            feat_c0[torch.isnan(feat_c0)] = 0
            feat_c1[torch.isnan(feat_c1)] = 0
            logger.info(f"feat_c0_nanmax: {feat_c0.abs().max()}, feat_c1_nanmax: {feat_c1.abs().max()}")
            
                # 3. match coarse-level
        if not self.config['matchability']: # else match in loftr_coarse
            self.coarse_matching(feat_c0, feat_c1, data, 
                                    mask_c0=mask_c0.view(mask_c0.size(0), -1) if mask_c0 is not None else mask_c0, 
                                    mask_c1=mask_c1.view(mask_c1.size(0), -1) if mask_c1 is not None else mask_c1
                                    )
                                    
        #return data['conf_matrix'],feat_c0,feat_c1,data['feats_x2'],data['feats_x1']

        # norm FPNfeat
        if self.config['norm_fpnfeat']:
            feat_c0, feat_c1 = map(lambda feat: feat / feat.shape[-1]**.5,
                            [feat_c0, feat_c1])
        if self.config['norm_fpnfeat2']:
            assert self.config['inter_feat']
            logger.info(f'before norm_fpnfeat2 max of feat_c0, feat_c1:{feat_c0.abs().max()}, {feat_c1.abs().max()}')
            if data['hw0_i'] == data['hw1_i']:
                logger.info(f'before norm_fpnfeat2 max of data[feats_x2], data[feats_x1]:{data["feats_x2"].abs().max()}, {data["feats_x1"].abs().max()}')
                feat_c0, feat_c1, data['feats_x2'], data['feats_x1'] = map(lambda feat: feat / feat.shape[-1]**.5,
                                [feat_c0, feat_c1, data['feats_x2'], data['feats_x1']])
            else:
                feat_c0, feat_c1, data['feats_x2_0'], data['feats_x2_1'], data['feats_x1_0'], data['feats_x1_1'] = map(lambda feat: feat / feat.shape[-1]**.5,
                                [feat_c0, feat_c1, data['feats_x2_0'], data['feats_x2_1'], data['feats_x1_0'], data['feats_x1_1']])
                
        
        # 4. fine-level refinement
        with torch.autocast(enabled=False, device_type="cuda"):
            feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(feat_f0, feat_f1, feat_c0, feat_c1, data)
        
        # detect nan
        if self.config['replace_nan'] and (torch.any(torch.isnan(feat_f0_unfold)) or torch.any(torch.isnan(feat_f1_unfold))):
            logger.info(f'replace nan in fine_preprocess')
            logger.info(f"feat_f0_unfold_nan_num: {torch.isnan(feat_f0_unfold).int().sum()}, feat_f1_unfold_nan_num: {torch.isnan(feat_f1_unfold).int().sum()}")
            logger.info(f"feat_f0_unfold: {feat_f0_unfold}, feat_f1_unfold: {feat_f1_unfold}")
            logger.info(f"feat_f0_unfold_max: {feat_f0_unfold}, feat_f1_unfold_max: {feat_f1_unfold}")
            feat_f0_unfold[torch.isnan(feat_f0_unfold)] = 0
            feat_f1_unfold[torch.isnan(feat_f1_unfold)] = 0
            logger.info(f"feat_f0_unfold_nanmax: {feat_f0_unfold}, feat_f1_unfold_nanmax: {feat_f1_unfold}")
        
        if self.config['fp16log'] and feat_c0 is not None:
            logger.info(f"c0: {feat_c0.abs().max()}, c1: {feat_c1.abs().max()}")
        del feat_c0, feat_c1, mask_c0, mask_c1
        if feat_f0_unfold.size(0) != 0:  # at least one coarse level predicted
            if self.config['fine']['pan']:
                m, ww, c = feat_f0_unfold.size() # [m, ww, c]
                w = self.config['fine_window_size']
                feat_f0_unfold, feat_f1_unfold = self.loftr_fine(feat_f0_unfold.reshape(m, c, w, w), feat_f1_unfold.reshape(m, c, w, w))
                feat_f0_unfold = rearrange(feat_f0_unfold, 'm c w h -> m (w h) c')
                feat_f1_unfold = rearrange(feat_f1_unfold, 'm c w h -> m (w h) c')
            elif self.config['fine']['skip']:
                pass
            else:
                feat_f0_unfold, feat_f1_unfold = self.loftr_fine(feat_f0_unfold, feat_f1_unfold)
        # 5. match fine-level
        # log forward nan
        if self.config['fp16log']:
            if feat_f0_unfold.size(0) != 0 and feat_f0 is not None:
                logger.info(f"f0: {feat_f0.abs().max()}, f1: {feat_f1.abs().max()}, uf0: {feat_f0_unfold.abs().max()}, uf1: {feat_f1_unfold.abs().max()}")
            elif feat_f0_unfold.size(0) != 0:
                logger.info(f"uf0: {feat_f0_unfold.abs().max()}, uf1: {feat_f1_unfold.abs().max()}")
            # elif feat_c0 is not None:
            #     logger.info(f"c0: {feat_c0.abs().max()}, c1: {feat_c1.abs().max()}")
            
        with torch.autocast(enabled=False, device_type="cuda"):
            self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)

        return data

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)

    def refine(self, data):
        """ 
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """
        # 1. Local Feature CNN
        data.update({
            'bs': data['image0'].size(0),
            'hw0_i': data['image0'].shape[2:], 'hw1_i': data['image1'].shape[2:]
        })
        feat_f0, feat_f1 = None, None
        feat_c0, feat_c1 = data['feat_c0'], data['feat_c1']
        # 4. fine-level refinement
        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(feat_f0, feat_f1, feat_c0, feat_c1, data)
        if feat_f0_unfold.size(0) != 0:  # at least one coarse level predicted
            if self.config['fine']['pan']:
                m, ww, c = feat_f0_unfold.size() # [m, ww, c]
                w = self.config['fine_window_size']
                feat_f0_unfold, feat_f1_unfold = self.loftr_fine(feat_f0_unfold.reshape(m, c, w, w), feat_f1_unfold.reshape(m, c, w, w))
                feat_f0_unfold = rearrange(feat_f0_unfold, 'm c w h -> m (w h) c')
                feat_f1_unfold = rearrange(feat_f1_unfold, 'm c w h -> m (w h) c')
            elif self.config['fine']['skip']:
                pass
            else:
                feat_f0_unfold, feat_f1_unfold = self.loftr_fine(feat_f0_unfold, feat_f1_unfold)
        # 5. match fine-level
        # log forward nan
        if self.config['fp16log']:
            if feat_f0_unfold.size(0) != 0 and feat_f0 is not None and feat_c0 is not None:
                logger.info(f"c0: {feat_c0.abs().max()}, c1: {feat_c1.abs().max()}, f0: {feat_f0.abs().max()}, f1: {feat_f1.abs().max()}, uf0: {feat_f0_unfold.abs().max()}, uf1: {feat_f1_unfold.abs().max()}")
            elif feat_f0 is not None and feat_c0 is not None:
                logger.info(f"c0: {feat_c0.abs().max()}, c1: {feat_c1.abs().max()}, f0: {feat_f0.abs().max()}, f1: {feat_f1.abs().max()}")
            elif feat_c0 is not None:
                logger.info(f"c0: {feat_c0.abs().max()}, c1: {feat_c1.abs().max()}")
            
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)
        return data