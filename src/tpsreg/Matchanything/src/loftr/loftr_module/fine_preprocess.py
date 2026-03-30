import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange, repeat
from ..backbone.repvgg import RepVGGBlock

from loguru import logger

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class FinePreprocess(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.cat_c_feat = config['fine_concat_coarse_feat']
        self.sample_c_feat = config['fine_sample_coarse_feat']
        self.fpn_inter_feat = config['inter_feat']
        self.rep_fpn = config['rep_fpn']
        self.deploy = config['rep_deploy']
        self.multi_regress = config['match_fine']['multi_regress']
        self.local_regress = config['match_fine']['local_regress']
        self.local_regress_inner = config['match_fine']['local_regress_inner']
        block_dims = config['resnetfpn']['block_dims']
            
        self.mtd_spvs = self.config['fine']['mtd_spvs']
        self.align_corner = self.config['align_corner']
        self.fix_bias = self.config['fix_bias']

        if self.mtd_spvs:
            self.W = self.config['fine_window_size']
        else:
            # assert False, 'fine_window_matching_size to be revised' # good notification!
            # self.W = self.config['fine_window_matching_size']
            self.W = self.config['fine_window_size']
            
        self.backbone_type = self.config['backbone_type']
            
        d_model_c = self.config['coarse']['d_model']
        d_model_f = self.config['fine']['d_model']
        self.d_model_f = d_model_f
        if self.fpn_inter_feat:
            if self.rep_fpn:
                self.layer3_outconv = conv1x1(block_dims[2], block_dims[2])
                self.layer2_outconv = conv1x1(block_dims[1], block_dims[2])
                self.layer2_outconv2 = []
                self.layer2_outconv2.append(RepVGGBlock(in_channels=block_dims[2], out_channels=block_dims[2], kernel_size=3,
                                      stride=1, padding=1, groups=1, deploy=self.deploy, use_se=False, leaky=0.01))
                self.layer2_outconv2.append(RepVGGBlock(in_channels=block_dims[2], out_channels=block_dims[1], kernel_size=3,
                                      stride=1, padding=1, groups=1, deploy=self.deploy, use_se=False, leaky=-2))
                self.layer2_outconv2 = nn.ModuleList(self.layer2_outconv2)
                self.layer1_outconv = conv1x1(block_dims[0], block_dims[1])
                self.layer1_outconv2 = []
                self.layer1_outconv2.append(RepVGGBlock(in_channels=block_dims[1], out_channels=block_dims[1], kernel_size=3,
                                      stride=1, padding=1, groups=1, deploy=self.deploy, use_se=False, leaky=0.01))
                self.layer1_outconv2.append(RepVGGBlock(in_channels=block_dims[1], out_channels=block_dims[0], kernel_size=3,
                                      stride=1, padding=1, groups=1, deploy=self.deploy, use_se=False, leaky=-2))
                self.layer1_outconv2 = nn.ModuleList(self.layer1_outconv2)
                
            else:
                self.layer3_outconv = conv1x1(block_dims[2], block_dims[2])
                self.layer2_outconv = conv1x1(block_dims[1], block_dims[2])
                self.layer2_outconv2 = nn.Sequential(
                    conv3x3(block_dims[2], block_dims[2]),
                    nn.BatchNorm2d(block_dims[2]),
                    nn.LeakyReLU(),
                    conv3x3(block_dims[2], block_dims[1]),
                )
                self.layer1_outconv = conv1x1(block_dims[0], block_dims[1])
                self.layer1_outconv2 = nn.Sequential(
                    conv3x3(block_dims[1], block_dims[1]),
                    nn.BatchNorm2d(block_dims[1]),
                    nn.LeakyReLU(),
                    conv3x3(block_dims[1], block_dims[0]),
                )
        elif self.cat_c_feat:
            self.down_proj = nn.Linear(d_model_c, d_model_f, bias=True)
            self.merge_feat = nn.Linear(2*d_model_f, d_model_f, bias=True)
        if self.sample_c_feat:
            self.down_proj = nn.Linear(d_model_c, d_model_f, bias=True)
        
        
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode="fan_out", nonlinearity="relu")

    def inter_fpn(self, feat_c, x2, x1, stride):
        feat_c = self.layer3_outconv(feat_c)
        feat_c = F.interpolate(feat_c, scale_factor=2., mode='bilinear', align_corners=False)
        x2 = self.layer2_outconv(x2)
        if self.rep_fpn:
            x2 = x2 + feat_c
            for layer in self.layer2_outconv2:
                x2 = layer(x2)
        else:
            x2 = self.layer2_outconv2(x2+feat_c)

        x2 = F.interpolate(x2, scale_factor=2., mode='bilinear', align_corners=False)
        x1 = self.layer1_outconv(x1)
        if self.rep_fpn:
            x1 = x1 + x2
            for layer in self.layer1_outconv2:
                x1 = layer(x1)
        else:
            x1 = self.layer1_outconv2(x1+x2)

        if stride == 4:
            logger.info('stride == 4')
            
        elif stride == 8:
            logger.info('stride == 8')
            x1 = F.interpolate(x1, scale_factor=2., mode='bilinear', align_corners=False)
        else:
            logger.info('stride not in {4,8}')
            assert False
        return x1
    
    def forward(self, feat_f0, feat_f1, feat_c0, feat_c1, data):
        W = self.W
        if self.fix_bias:
            stride = 4
        else:
            stride = data['hw0_f'][0] // data['hw0_c'][0]

        data.update({'W': W})
        if data['b_ids'].shape[0] == 0:
            feat0 = torch.empty(0, self.W**2, self.d_model_f, device=feat_c0.device)
            feat1 = torch.empty(0, self.W**2, self.d_model_f, device=feat_c0.device)
            # return feat0, feat1
            return feat0.float(), feat1.float()

        if self.fpn_inter_feat:
            if data['hw0_i'] != data['hw1_i']:
                if self.align_corner is False:
                    assert self.backbone_type != 's2dnet'

                    feat_c0 = rearrange(feat_c0, 'b (h w) c -> b c h w', h=data['hw0_c'][0])
                    feat_c1 = rearrange(feat_c1, 'b (h w) c -> b c h w', h=data['hw1_c'][0])
                    x2_0, x1_0 = data['feats_x2_0'], data['feats_x1_0']
                    x2_1, x1_1 = data['feats_x2_1'], data['feats_x1_1']
                    del data['feats_x2_0'], data['feats_x1_0'], data['feats_x2_1'], data['feats_x1_1']
                    feat_f0, feat_f1 = self.inter_fpn(feat_c0, x2_0, x1_0, stride), self.inter_fpn(feat_c1, x2_1, x1_1, stride)

                    if self.local_regress_inner:
                        assert W == 8
                        feat_f0 = F.unfold(feat_f0, kernel_size=(W, W), stride=stride, padding=0)
                        feat_f0 = rearrange(feat_f0, 'n (c ww) l -> n l ww c', ww=W**2)
                        feat_f1 = F.unfold(feat_f1, kernel_size=(W+2, W+2), stride=stride, padding=1)
                        feat_f1 = rearrange(feat_f1, 'n (c ww) l -> n l ww c', ww=(W+2)**2)
                    elif W == 10 and self.multi_regress:
                        feat_f0 = F.unfold(feat_f0, kernel_size=(W, W), stride=stride, padding=1)
                        feat_f0 = rearrange(feat_f0, 'n (c ww) l -> n l ww c', ww=W**2)
                        feat_f1 = F.unfold(feat_f1, kernel_size=(W, W), stride=stride, padding=1)
                        feat_f1 = rearrange(feat_f1, 'n (c ww) l -> n l ww c', ww=W**2)
                    elif W == 10:
                        feat_f0 = F.unfold(feat_f0, kernel_size=(W, W), stride=stride, padding=1)
                        feat_f0 = rearrange(feat_f0, 'n (c ww) l -> n l ww c', ww=W**2)
                        feat_f1 = F.unfold(feat_f1, kernel_size=(W, W), stride=stride, padding=1)
                        feat_f1 = rearrange(feat_f1, 'n (c ww) l -> n l ww c', ww=W**2)
                    else:
                        assert not self.multi_regress
                        feat_f0 = F.unfold(feat_f0, kernel_size=(W, W), stride=stride, padding=0)
                        feat_f0 = rearrange(feat_f0, 'n (c ww) l -> n l ww c', ww=W**2)
                        feat_f1 = F.unfold(feat_f1, kernel_size=(W, W), stride=stride, padding=0)
                        feat_f1 = rearrange(feat_f1, 'n (c ww) l -> n l ww c', ww=W**2)

                    # 2. select only the predicted matches
                    feat_f0 = feat_f0[data['b_ids'], data['i_ids']]  # [n, ww, cf]
                    feat_f1 = feat_f1[data['b_ids'], data['j_ids']]

                    return feat_f0, feat_f1

            else:
                if self.align_corner is False:
                    feat_c = torch.cat([feat_c0, feat_c1], 0)
                    feat_c = rearrange(feat_c, 'b (h w) c -> b c h w', h=data['hw0_c'][0]) # 1/8 256
                    x2 = data['feats_x2'].float() # 1/4 128
                    x1 = data['feats_x1'].float() # 1/2 64
                    del data['feats_x2'], data['feats_x1']
                    assert self.backbone_type != 's2dnet'
                    feat_c = self.layer3_outconv(feat_c)
                    feat_c = F.interpolate(feat_c, scale_factor=2., mode='bilinear', align_corners=False)
                    x2 = self.layer2_outconv(x2)
                    if self.rep_fpn:
                        x2 = x2 + feat_c
                        for layer in self.layer2_outconv2:
                            x2 = layer(x2)
                    else:
                        x2 = self.layer2_outconv2(x2+feat_c)

                    x2 = F.interpolate(x2, scale_factor=2., mode='bilinear', align_corners=False)
                    x1 = self.layer1_outconv(x1)
                    if self.rep_fpn:
                        x1 = x1 + x2
                        for layer in self.layer1_outconv2:
                            x1 = layer(x1)
                    else:
                        x1 = self.layer1_outconv2(x1+x2)
                    
                    if stride == 4:
                        # logger.info('stride == 4')
                        pass
                    elif stride == 8:
                        # logger.info('stride == 8')
                        x1 = F.interpolate(x1, scale_factor=2., mode='bilinear', align_corners=False)
                    else:
                        # logger.info('stride not in {4,8}')
                        assert False
                            
                    feat_f0, feat_f1 = torch.chunk(x1, 2, dim=0)

                    # 1. unfold(crop) all local windows
                    if self.local_regress_inner:
                        assert W == 8
                        feat_f0 = F.unfold(feat_f0, kernel_size=(W, W), stride=stride, padding=0)
                        feat_f0 = rearrange(feat_f0, 'n (c ww) l -> n l ww c', ww=W**2)
                        feat_f1 = F.unfold(feat_f1, kernel_size=(W+2, W+2), stride=stride, padding=1)
                        feat_f1 = rearrange(feat_f1, 'n (c ww) l -> n l ww c', ww=(W+2)**2)
                    elif self.multi_regress or (self.local_regress and W == 10):
                        feat_f0 = F.unfold(feat_f0, kernel_size=(W, W), stride=stride, padding=1)
                        feat_f0 = rearrange(feat_f0, 'n (c ww) l -> n l ww c', ww=W**2)
                        feat_f1 = F.unfold(feat_f1, kernel_size=(W, W), stride=stride, padding=1)
                        feat_f1 = rearrange(feat_f1, 'n (c ww) l -> n l ww c', ww=W**2)
                    elif W == 10:
                        feat_f0 = F.unfold(feat_f0, kernel_size=(W, W), stride=stride, padding=1)
                        feat_f0 = rearrange(feat_f0, 'n (c ww) l -> n l ww c', ww=W**2)
                        feat_f1 = F.unfold(feat_f1, kernel_size=(W, W), stride=stride, padding=1)
                        feat_f1 = rearrange(feat_f1, 'n (c ww) l -> n l ww c', ww=W**2)

                    else:
                        feat_f0 = F.unfold(feat_f0, kernel_size=(W, W), stride=stride, padding=0)
                        feat_f0 = rearrange(feat_f0, 'n (c ww) l -> n l ww c', ww=W**2)
                        feat_f1 = F.unfold(feat_f1, kernel_size=(W, W), stride=stride, padding=0)
                        feat_f1 = rearrange(feat_f1, 'n (c ww) l -> n l ww c', ww=W**2)

                    # 2. select only the predicted matches
                    feat_f0 = feat_f0[data['b_ids'], data['i_ids']]  # [n, ww, cf]
                    feat_f1 = feat_f1[data['b_ids'], data['j_ids']]

                    return feat_f0, feat_f1
                elif self.fix_bias:
                    feat_c = torch.cat([feat_c0, feat_c1], 0)
                    feat_c = rearrange(feat_c, 'b (h w) c -> b c h w', h=data['hw0_c'][0])
                    x2 = data['feats_x2'].float()
                    x1 = data['feats_x1'].float()
                    assert self.backbone_type != 's2dnet'
                    x3_out = self.layer3_outconv(feat_c)
                    x3_out_2x = F.interpolate(x3_out, size=((x3_out.size(-2)-1)*2+1, (x3_out.size(-1)-1)*2+1), mode='bilinear', align_corners=False)
                    x2 = self.layer2_outconv(x2)
                    x2 = self.layer2_outconv2(x2+x3_out_2x)

                    x2 = F.interpolate(x2, size=((x2.size(-2)-1)*2+1, (x2.size(-1)-1)*2+1), mode='bilinear', align_corners=False)
                    x1_out = self.layer1_outconv(x1)
                    x1_out = self.layer1_outconv2(x1_out+x2)
                    x0_out = x1_out

                    feat_f0, feat_f1 = torch.chunk(x0_out, 2, dim=0)

                    # 1. unfold(crop) all local windows
                    feat_f0_unfold = F.unfold(feat_f0, kernel_size=(W, W), stride=stride, padding=W//2)
                    feat_f0_unfold = rearrange(feat_f0_unfold, 'n (c ww) l -> n l ww c', ww=W**2)
                    feat_f1_unfold = F.unfold(feat_f1, kernel_size=(W, W), stride=stride, padding=W//2)
                    feat_f1_unfold = rearrange(feat_f1_unfold, 'n (c ww) l -> n l ww c', ww=W**2)

                    # 2. select only the predicted matches
                    feat_f0_unfold = feat_f0_unfold[data['b_ids'], data['i_ids']]  # [n, ww, cf]
                    feat_f1_unfold = feat_f1_unfold[data['b_ids'], data['j_ids']]

                    return feat_f0_unfold, feat_f1_unfold

                

        elif self.sample_c_feat:
            if self.align_corner is False:
                # easy implemented but memory consuming
                feat_c = self.down_proj(torch.cat([feat_c0,
                                                        feat_c1], 0)) # [n, (h w), c] -> [2n, (h w), cf]
                feat_c = rearrange(feat_c, 'n (h w) c -> n c h w', h=data['hw0_c'][0], w=data['hw0_c'][1])
                feat_f = F.interpolate(feat_c, scale_factor=8., mode='bilinear', align_corners=False) # [2n, cf, hf, wf]
                feat_f_unfold = F.unfold(feat_f, kernel_size=(W, W), stride=stride, padding=0)
                feat_f_unfold = rearrange(feat_f_unfold, 'n (c ww) l -> n l ww c', ww=W**2)
                feat_f0_unfold, feat_f1_unfold = torch.chunk(feat_f_unfold, 2, dim=0)
                feat_f0_unfold = feat_f0_unfold[data['b_ids'], data['i_ids']] # [m, ww, cf]
                feat_f1_unfold = feat_f1_unfold[data['b_ids'], data['j_ids']] # [m, ww, cf]
                # return feat_f0_unfold, feat_f1_unfold
                return feat_f0_unfold.float(), feat_f1_unfold.float()
        else:
            if self.align_corner is False:
                # 1. unfold(crop) all local windows
                assert False, 'maybe exist bugs'
                feat_f0_unfold = F.unfold(feat_f0, kernel_size=(W, W), stride=stride, padding=0)
                feat_f0_unfold = rearrange(feat_f0_unfold, 'n (c ww) l -> n l ww c', ww=W**2)
                feat_f1_unfold = F.unfold(feat_f1, kernel_size=(W, W), stride=stride, padding=0)
                feat_f1_unfold = rearrange(feat_f1_unfold, 'n (c ww) l -> n l ww c', ww=W**2)

                # 2. select only the predicted matches
                feat_f0_unfold = feat_f0_unfold[data['b_ids'], data['i_ids']]  # [n, ww, cf]
                feat_f1_unfold = feat_f1_unfold[data['b_ids'], data['j_ids']]

                # option: use coarse-level loftr feature as context: concat and linear
                if self.cat_c_feat:
                    feat_c_win = self.down_proj(torch.cat([feat_c0[data['b_ids'], data['i_ids']],
                                                        feat_c1[data['b_ids'], data['j_ids']]], 0))  # [2n, c]
                    feat_cf_win = self.merge_feat(torch.cat([
                        torch.cat([feat_f0_unfold, feat_f1_unfold], 0),  # [2n, ww, cf]
                        repeat(feat_c_win, 'n c -> n ww c', ww=W**2),  # [2n, ww, cf]
                    ], -1))
                    feat_f0_unfold, feat_f1_unfold = torch.chunk(feat_cf_win, 2, dim=0)
                
                return feat_f0_unfold, feat_f1_unfold
                
            else:
                # 1. unfold(crop) all local windows
                if self.fix_bias:
                    feat_f0_unfold = F.unfold(feat_f0, kernel_size=(W, W), stride=stride, padding=W//2)
                    feat_f0_unfold = rearrange(feat_f0_unfold, 'n (c ww) l -> n l ww c', ww=W**2)
                    feat_f1_unfold = F.unfold(feat_f1, kernel_size=(W, W), stride=stride, padding=W//2)
                    feat_f1_unfold = rearrange(feat_f1_unfold, 'n (c ww) l -> n l ww c', ww=W**2)
                else:
                    feat_f0_unfold = F.unfold(feat_f0, kernel_size=(W, W), stride=stride, padding=W//2)
                    feat_f0_unfold = rearrange(feat_f0_unfold, 'n (c ww) l -> n l ww c', ww=W**2)
                    feat_f1_unfold = F.unfold(feat_f1, kernel_size=(W, W), stride=stride, padding=W//2)
                    feat_f1_unfold = rearrange(feat_f1_unfold, 'n (c ww) l -> n l ww c', ww=W**2)

                # 2. select only the predicted matches
                feat_f0_unfold = feat_f0_unfold[data['b_ids'], data['i_ids']]  # [n, ww, cf]
                feat_f1_unfold = feat_f1_unfold[data['b_ids'], data['j_ids']]

                # option: use coarse-level loftr feature as context: concat and linear
                if self.cat_c_feat:
                    feat_c_win = self.down_proj(torch.cat([feat_c0[data['b_ids'], data['i_ids']],
                                                        feat_c1[data['b_ids'], data['j_ids']]], 0))  # [2n, c]
                    feat_cf_win = self.merge_feat(torch.cat([
                        torch.cat([feat_f0_unfold, feat_f1_unfold], 0),  # [2n, ww, cf]
                        repeat(feat_c_win, 'n c -> n ww c', ww=W**2),  # [2n, ww, cf]
                    ], -1))
                    feat_f0_unfold, feat_f1_unfold = torch.chunk(feat_cf_win, 2, dim=0)

                # return feat_f0_unfold, feat_f1_unfold
                return feat_f0_unfold.float(), feat_f1_unfold.float()