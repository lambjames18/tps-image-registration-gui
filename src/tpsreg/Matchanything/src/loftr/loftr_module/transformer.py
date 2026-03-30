import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .linear_attention import LinearAttention, RoPELinearAttention, FullAttention, XAttention
from einops.einops import rearrange
from collections import OrderedDict
from .transformer_utils import TokenConfidence, MatchAssignment, filter_matches
from ..utils.coarse_matching import CoarseMatching
from ..utils.position_encoding import RoPEPositionEncodingSine
import numpy as np
from loguru import logger

PFLASH_AVAILABLE = False

class PANEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear',
                 pool_size=4,
                 bn=True,
                 xformer=False,
                 leaky=-1.0,
                 dw_conv=False,
                 scatter=False,
                 ):
        super(PANEncoderLayer, self).__init__()

        self.pool_size = pool_size
        self.dw_conv = dw_conv
        self.scatter = scatter
        if self.dw_conv:
            self.aggregate = nn.Conv2d(d_model, d_model, kernel_size=pool_size, padding=0, stride=pool_size, bias=False, groups=d_model)
        
        assert not self.scatter, 'buggy implemented here'
        self.dim = d_model // nhead
        self.nhead = nhead

        self.max_pool = torch.nn.MaxPool2d(kernel_size=self.pool_size, stride=self.pool_size)
        # multi-head attention
        if bn:
            method = 'dw_bn'
        else:
            method = 'dw'
        self.q_proj_conv = self._build_projection(d_model, d_model, method=method)
        self.k_proj_conv = self._build_projection(d_model, d_model, method=method)
        self.v_proj_conv = self._build_projection(d_model, d_model, method=method)
            
            # self.q_proj = nn.Linear(d_mosdel, d_model, bias=False)
            # self.k_proj = nn.Linear(d_model, d_model, bias=False)
            # self.v_proj = nn.Linear(d_model, d_model, bias=False)
        if xformer:
            self.attention = XAttention()
        else:
            self.attention = LinearAttention() if attention == 'linear' else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        if leaky > 0:
            self.mlp = nn.Sequential(
                nn.Linear(d_model*2, d_model*2, bias=False),
                nn.LeakyReLU(leaky, True),
                nn.Linear(d_model*2, d_model, bias=False),
            )
            
        else:
            self.mlp = nn.Sequential(
                nn.Linear(d_model*2, d_model*2, bias=False),
                nn.ReLU(True),
                nn.Linear(d_model*2, d_model, bias=False),
            )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # self.norm1 = nn.BatchNorm2d(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, C, H1, W1]
            source (torch.Tensor): [N, C, H2, W2]
            x_mask (torch.Tensor): [N, H1, W1] (optional) (L = H1*W1)
            source_mask (torch.Tensor): [N, H2, W2] (optional) (S = H2*W2)
        """
        bs = x.size(0)
        H1, W1 = x.size(-2), x.size(-1)
        H2, W2 = source.size(-2), source.size(-1)
        
        query, key, value = x, source, source

        if self.dw_conv:
            query = self.norm1(self.aggregate(query).permute(0,2,3,1)).permute(0,3,1,2)
        else:
            query = self.norm1(self.max_pool(query).permute(0,2,3,1)).permute(0,3,1,2)
        # only need to cal key or value...
        key = self.norm1(self.max_pool(key).permute(0,2,3,1)).permute(0,3,1,2)
        value = self.norm1(self.max_pool(value).permute(0,2,3,1)).permute(0,3,1,2)
        
        # After 0617 bnorm to prevent permute*6
        # query = self.norm1(self.max_pool(query))
        # key = self.norm1(self.max_pool(key))
        # value = self.norm1(self.max_pool(value))
        # multi-head attention
        query = self.q_proj_conv(query) # [N, C, H1//pool, W1//pool]
        key = self.k_proj_conv(key)
        value = self.v_proj_conv(value)
        
        C = query.shape[-3]
        
        ismask = x_mask is not None and source_mask is not None
        if bs == 1 or not ismask:
            if ismask:
                x_mask = self.max_pool(x_mask.float()).bool() # [N, H1//pool, W1//pool]
                source_mask = self.max_pool(source_mask.float()).bool()

                mask_h0, mask_w0 = x_mask[0].sum(-2)[0], x_mask[0].sum(-1)[0]
                mask_h1, mask_w1 = source_mask[0].sum(-2)[0], source_mask[0].sum(-1)[0]
                
                query = query[:, :, :mask_h0, :mask_w0]
                key = key[:, :, :mask_h1, :mask_w1]
                value = value[:, :, :mask_h1, :mask_w1]
            
            else:
                assert x_mask is None and source_mask is None
            
            # query = query.reshape(bs, -1, self.nhead, self.dim)  # [N, L, H, D]
            # key = key.reshape(bs, -1, self.nhead, self.dim)  # [N, S, H, D]
            # value = value.reshape(bs, -1, self.nhead, self.dim) # [N, S, H, D]
            if PFLASH_AVAILABLE: # N H L D
                query = rearrange(query, 'n (nhead d) h w -> n nhead (h w) d', nhead=self.nhead, d=self.dim)
                key = rearrange(key, 'n (nhead d) h w -> n nhead (h w) d', nhead=self.nhead, d=self.dim) 
                value = rearrange(value, 'n (nhead d) h w -> n nhead (h w) d', nhead=self.nhead, d=self.dim)
                
            else: # N L H D
                query = rearrange(query, 'n (nhead d) h w -> n (h w) nhead d', nhead=self.nhead, d=self.dim)  # [N, L, H, D]
                key = rearrange(key, 'n (nhead d) h w -> n (h w) nhead d', nhead=self.nhead, d=self.dim)  # [N, S, H, D]
                value = rearrange(value, 'n (nhead d) h w -> n (h w) nhead d', nhead=self.nhead, d=self.dim)  # [N, S, H, D]
            
            message = self.attention(query, key, value, q_mask=None, kv_mask=None)  # [N, L, H, D] or [N, H, L, D]
            
            if PFLASH_AVAILABLE: # N H L D
                message = rearrange(message, 'n nhead L d -> n L nhead d', nhead=self.nhead, d=self.dim)

            if ismask:
                message = message.view(bs, mask_h0, mask_w0, self.nhead, self.dim) 
                if mask_h0 != x_mask.size(-2):
                    message = torch.cat([message, torch.zeros(message.size(0), x_mask.size(-2)-mask_h0, x_mask.size(-1), self.nhead, self.dim, device=message.device, dtype=message.dtype)], dim=1)
                elif mask_w0 != x_mask.size(-1):
                    message = torch.cat([message, torch.zeros(message.size(0), x_mask.size(-2), x_mask.size(-1)-mask_w0, self.nhead, self.dim, device=message.device, dtype=message.dtype)], dim=2)
                # message = message.view(bs, -1, self.nhead*self.dim)  # [N, L, C]
                
            else:
                assert x_mask is None and source_mask is None
                

            message = self.merge(message.reshape(bs, -1, self.nhead*self.dim))  # [N, L, C]
            # message = message.reshape(bs, C, H1//self.pool_size, W1//self.pool_size) # [N, C, H, W] bug???
            message = rearrange(message, 'b (h w) c -> b c h w', h=H1//self.pool_size, w=W1//self.pool_size) # [N, C, H, W]
            
            if self.scatter:
                message = torch.repeat_interleave(message, self.pool_size, dim=-2)
                message = torch.repeat_interleave(message, self.pool_size, dim=-1)
                # message = self.aggregate(message)
                message = message * self.aggregate.weight.data.reshape(1, C, self.pool_size, self.pool_size).repeat(1,1,message.shape[-2]//self.pool_size,message.shape[-1]//self.pool_size)
            else:
                message = torch.nn.functional.interpolate(message, scale_factor=self.pool_size, mode='bilinear', align_corners=False) # [N, C, H1, W1]
            
            # message = self.norm1(message)

            # feed-forward network
            message = self.mlp(torch.cat([x, message], dim=1).permute(0, 2, 3, 1)) # [N, H1, W1, C]
            message = self.norm2(message).permute(0, 3, 1, 2) # [N, C, H1, W1]

            return x + message
        else:
            x_mask = self.max_pool(x_mask.float()).bool()
            source_mask = self.max_pool(source_mask.float()).bool()
            m_list = []
            for i in range(bs):
                mask_h0, mask_w0 = x_mask[i].sum(-2)[0], x_mask[i].sum(-1)[0]
                mask_h1, mask_w1 = source_mask[i].sum(-2)[0], source_mask[i].sum(-1)[0]
                
                q = query[i:i+1, :, :mask_h0, :mask_w0]
                k = key[i:i+1, :, :mask_h1, :mask_w1]
                v = value[i:i+1, :, :mask_h1, :mask_w1]

                if PFLASH_AVAILABLE: # N H L D
                    q = rearrange(q, 'n (nhead d) h w -> n nhead (h w) d', nhead=self.nhead, d=self.dim)
                    k = rearrange(k, 'n (nhead d) h w -> n nhead (h w) d', nhead=self.nhead, d=self.dim) 
                    v = rearrange(v, 'n (nhead d) h w -> n nhead (h w) d', nhead=self.nhead, d=self.dim)
                    
                else: # N L H D

                    q = rearrange(q, 'n (nhead d) h w -> n (h w) nhead d', nhead=self.nhead, d=self.dim)  # [N, L, H, D]
                    k = rearrange(k, 'n (nhead d) h w -> n (h w) nhead d', nhead=self.nhead, d=self.dim)  # [N, S, H, D]
                    v = rearrange(v, 'n (nhead d) h w -> n (h w) nhead d', nhead=self.nhead, d=self.dim)  # [N, S, H, D]
                
                m = self.attention(q, k, v, q_mask=None, kv_mask=None)  # [N, L, H, D]
                
                if PFLASH_AVAILABLE: # N H L D
                    m = rearrange(m, 'n nhead L d -> n L nhead d', nhead=self.nhead, d=self.dim)

                m = m.view(1, mask_h0, mask_w0, self.nhead, self.dim)
                if mask_h0 != x_mask.size(-2):
                    m = torch.cat([m, torch.zeros(1, x_mask.size(-2)-mask_h0, x_mask.size(-1), self.nhead, self.dim, device=m.device, dtype=m.dtype)], dim=1)
                elif mask_w0 != x_mask.size(-1):
                    m = torch.cat([m, torch.zeros(1, x_mask.size(-2), x_mask.size(-1)-mask_w0, self.nhead, self.dim, device=m.device, dtype=m.dtype)], dim=2)
                m_list.append(m)
            message = torch.cat(m_list, dim=0)
            

            message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
            # message = message.reshape(bs, C, H1//self.pool_size, W1//self.pool_size) # [N, C, H, W] bug???
            message = rearrange(message, 'b (h w) c -> b c h w', h=H1//self.pool_size, w=W1//self.pool_size) # [N, C, H, W]
            
            if self.scatter:
                message = torch.repeat_interleave(message, self.pool_size, dim=-2)
                message = torch.repeat_interleave(message, self.pool_size, dim=-1)
                # message = self.aggregate(message)
                # assert False
            else:
                message = torch.nn.functional.interpolate(message, scale_factor=self.pool_size, mode='bilinear', align_corners=False) # [N, C, H1, W1]
            
            # message = self.norm1(message)

            # feed-forward network
            message = self.mlp(torch.cat([x, message], dim=1).permute(0, 2, 3, 1)) # [N, H1, W1, C]
            message = self.norm2(message).permute(0, 3, 1, 2) # [N, C, H1, W1]

            return x + message

            
    def pro(self, x, source, x_mask=None, source_mask=None, profiler=None):
        """
        Args:
            x (torch.Tensor): [N, C, H1, W1]
            source (torch.Tensor): [N, C, H2, W2]
            x_mask (torch.Tensor): [N, H1, W1] (optional) (L = H1*W1)
            source_mask (torch.Tensor): [N, H2, W2] (optional) (S = H2*W2)
        """
        bs = x.size(0)
        H1, W1 = x.size(-2), x.size(-1)
        H2, W2 = source.size(-2), source.size(-1)
        
        query, key, value = x, source, source

        with profiler.profile("permute*6+norm1*3+max_pool*3"):
            if self.dw_conv:
                query = self.norm1(self.aggregate(query).permute(0,2,3,1)).permute(0,3,1,2)
            else:
                query = self.norm1(self.max_pool(query).permute(0,2,3,1)).permute(0,3,1,2)
            # only need to cal key or value...
            key = self.norm1(self.max_pool(key).permute(0,2,3,1)).permute(0,3,1,2)
            value = self.norm1(self.max_pool(value).permute(0,2,3,1)).permute(0,3,1,2)
        
        with profiler.profile("permute*6"):
            query = query.permute(0, 2, 3, 1)
            key = key.permute(0, 2, 3, 1)
            value = value.permute(0, 2, 3, 1)
            
            query = query.permute(0,3,1,2)
            key = key.permute(0,3,1,2)
            value = value.permute(0,3,1,2)
            
        # query = self.bnorm1(self.max_pool(query))
        # key = self.bnorm1(self.max_pool(key))
        # value = self.bnorm1(self.max_pool(value))
        # multi-head attention
        
        with profiler.profile("q_conv+k_conv+v_conv"):
            query = self.q_proj_conv(query) # [N, C, H1//pool, W1//pool]
            key = self.k_proj_conv(key)
            value = self.v_proj_conv(value)
        
        C = query.shape[-3]
        # TODO: Need to be consistent with bs=1 (where mask region do not in attention at all)
        if x_mask is not None and source_mask is not None:
            x_mask = self.max_pool(x_mask.float()).bool() # [N, H1//pool, W1//pool]
            source_mask = self.max_pool(source_mask.float()).bool()

            mask_h0, mask_w0 = x_mask[0].sum(-2)[0], x_mask[0].sum(-1)[0]
            mask_h1, mask_w1 = source_mask[0].sum(-2)[0], source_mask[0].sum(-1)[0]
            
            query = query[:, :, :mask_h0, :mask_w0]
            key = key[:, :, :mask_h1, :mask_w1]
            value = value[:, :, :mask_h1, :mask_w1]
            
            # mask_h0, mask_w0 = data['mask0'][0].sum(-2)[0], data['mask0'][0].sum(-1)[0]
            # mask_h1, mask_w1 = data['mask1'][0].sum(-2)[0], data['mask1'][0].sum(-1)[0]
            # C = feat_c0.shape[-3]
            # feat_c0 = feat_c0[:, :, :mask_h0, :mask_w0]
            # feat_c1 = feat_c1[:, :, :mask_h1, :mask_w1]


            # feat_c0 = feat_c0.reshape(-1, mask_h0, mask_w0, C)
            # feat_c1 = feat_c1.reshape(-1, mask_h1, mask_w1, C)
            # if mask_h0 != data['mask0'].size(-2):
            #     feat_c0 = torch.cat([feat_c0, torch.zeros(feat_c0.size(0), data['hw0_c'][0]-mask_h0, data['hw0_c'][1], C, device=feat_c0.device)], dim=1)
            # elif mask_w0 != data['mask0'].size(-1):
            #     feat_c0 = torch.cat([feat_c0, torch.zeros(feat_c0.size(0), data['hw0_c'][0], data['hw0_c'][1]-mask_w0, C, device=feat_c0.device)], dim=2)
                
            # if mask_h1 != data['mask1'].size(-2):
            #     feat_c1 = torch.cat([feat_c1, torch.zeros(feat_c1.size(0), data['hw1_c'][0]-mask_h1, data['hw1_c'][1], C, device=feat_c1.device)], dim=1)
            # elif mask_w1 != data['mask1'].size(-1):
            #     feat_c1 = torch.cat([feat_c1, torch.zeros(feat_c1.size(0), data['hw1_c'][0], data['hw1_c'][1]-mask_w1, C, device=feat_c1.device)], dim=2)

        
        else:
            assert x_mask is None and source_mask is None
        
        
        
        # query = query.reshape(bs, -1, self.nhead, self.dim)  # [N, L, H, D]
        # key = key.reshape(bs, -1, self.nhead, self.dim)  # [N, S, H, D]
        # value = value.reshape(bs, -1, self.nhead, self.dim) # [N, S, H, D]
        
        with profiler.profile("rearrange*3"):
            query = rearrange(query, 'n (nhead d) h w -> n (h w) nhead d', nhead=self.nhead, d=self.dim)  # [N, L, H, D]
            key = rearrange(key, 'n (nhead d) h w -> n (h w) nhead d', nhead=self.nhead, d=self.dim)  # [N, S, H, D]
            value = rearrange(value, 'n (nhead d) h w -> n (h w) nhead d', nhead=self.nhead, d=self.dim)  # [N, S, H, D]
        
        with profiler.profile("attention"):
            message = self.attention(query, key, value, q_mask=None, kv_mask=None)  # [N, L, H, D]
        
        if x_mask is not None and source_mask is not None:
            message = message.view(bs, mask_h0, mask_w0, self.nhead, self.dim) 
            if mask_h0 != x_mask.size(-2):
                message = torch.cat([message, torch.zeros(message.size(0), x_mask.size(-2)-mask_h0, x_mask.size(-1), self.nhead, self.dim, device=message.device, dtype=message.dtype)], dim=1)
            elif mask_w0 != x_mask.size(-1):
                message = torch.cat([message, torch.zeros(message.size(0), x_mask.size(-2), x_mask.size(-1)-mask_w0, self.nhead, self.dim, device=message.device, dtype=message.dtype)], dim=2)
            # message = message.view(bs, -1, self.nhead*self.dim)  # [N, L, C]
            
        else:
            assert x_mask is None and source_mask is None
            
        with profiler.profile("merge"):
            message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        # message = message.reshape(bs, C, H1//self.pool_size, W1//self.pool_size) # [N, C, H, W] bug???

        with profiler.profile("rearrange*1"):
            message = rearrange(message, 'b (h w) c -> b c h w', h=H1//self.pool_size, w=W1//self.pool_size) # [N, C, H, W]
        
        with profiler.profile("upsample"):
            if self.scatter:
                message = torch.repeat_interleave(message, self.pool_size, dim=-2)
                message = torch.repeat_interleave(message, self.pool_size, dim=-1)
                # message = self.aggregate(message)
                # assert False
            else:
                message = torch.nn.functional.interpolate(message, scale_factor=self.pool_size, mode='bilinear', align_corners=False) # [N, C, H1, W1]
        
        # message = self.norm1(message)

        # feed-forward network
        with profiler.profile("feed-forward_mlp+permute*2+norm2"):
            message = self.mlp(torch.cat([x, message], dim=1).permute(0, 2, 3, 1)) # [N, H1, W1, C]
            message = self.norm2(message).permute(0, 3, 1, 2) # [N, C, H1, W1]

        return x + message
    

    def _build_projection(self,
                          dim_in,
                          dim_out,
                          kernel_size=3,
                          padding=1,
                          stride=1,
                          method='dw_bn',
                          ):
        if method == 'dw_bn':
            proj = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(
                    dim_in,
                    dim_in,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=False,
                    groups=dim_in
                )),
                ('bn', nn.BatchNorm2d(dim_in)),
                # ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))
        elif method == 'avg':
            proj = nn.Sequential(OrderedDict([
                ('avg', nn.AvgPool2d(
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    ceil_mode=True
                )),
                # ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))
        elif method == 'linear':
            proj = None
        elif method == 'dw':
            proj = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(
                    dim_in,
                    dim_in,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=False,
                    groups=dim_in
                )),
                # ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))
        else:
            raise ValueError('Unknown method ({})'.format(method))

        return proj

class AG_RoPE_EncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear',
                 pool_size=4,
                 pool_size2=4,
                 xformer=False,
                 leaky=-1.0,
                 dw_conv=False,
                 dw_conv2=False,
                 scatter=False,
                 norm_before=True,
                 rope=False,
                 npe=None,
                 vit_norm=False,
                 dw_proj=False,
                 ):
        super(AG_RoPE_EncoderLayer, self).__init__()

        self.pool_size = pool_size
        self.pool_size2 = pool_size2
        self.dw_conv = dw_conv
        self.dw_conv2 = dw_conv2
        self.scatter = scatter
        self.norm_before = norm_before
        self.vit_norm = vit_norm
        self.dw_proj = dw_proj
        self.rope = rope
        if self.dw_conv and self.pool_size != 1:
            self.aggregate = nn.Conv2d(d_model, d_model, kernel_size=pool_size, padding=0, stride=pool_size, bias=False, groups=d_model)
        if self.dw_conv2 and self.pool_size2 != 1:
            self.aggregate2 = nn.Conv2d(d_model, d_model, kernel_size=pool_size2, padding=0, stride=pool_size2, bias=False, groups=d_model)
            
        self.dim = d_model // nhead
        self.nhead = nhead

        self.max_pool = torch.nn.MaxPool2d(kernel_size=self.pool_size2, stride=self.pool_size2)
        
        # multi-head attention
        if self.dw_proj:
            self.q_proj = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, stride=1, bias=False, groups=d_model)
            self.k_proj = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, stride=1, bias=False, groups=d_model)
            self.v_proj = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, stride=1, bias=False, groups=d_model)
        else:
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
        
        if self.rope:
            self.rope_pos_enc = RoPEPositionEncodingSine(d_model, max_shape=(256, 256), npe=npe, ropefp16=True)
        
        if xformer:
            self.attention = XAttention()
        else:
            self.attention = LinearAttention() if attention == 'linear' else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        if leaky > 0:
            if self.vit_norm:
                self.mlp = nn.Sequential(
                    nn.Linear(d_model, d_model*2, bias=False),
                    nn.LeakyReLU(leaky, True),
                    nn.Linear(d_model*2, d_model, bias=False),
                )
            else:
                self.mlp = nn.Sequential(
                    nn.Linear(d_model*2, d_model*2, bias=False),
                    nn.LeakyReLU(leaky, True),
                    nn.Linear(d_model*2, d_model, bias=False),
                )
            
        else:
            if self.vit_norm:
                self.mlp = nn.Sequential(
                    nn.Linear(d_model, d_model*2, bias=False),
                    nn.ReLU(True),
                    nn.Linear(d_model*2, d_model, bias=False),
                )
            else:
                self.mlp = nn.Sequential(
                    nn.Linear(d_model*2, d_model*2, bias=False),
                    nn.ReLU(True),
                    nn.Linear(d_model*2, d_model, bias=False),
                )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # self.norm1 = nn.BatchNorm2d(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, C, H1, W1]
            source (torch.Tensor): [N, C, H2, W2]
            x_mask (torch.Tensor): [N, H1, W1] (optional) (L = H1*W1)
            source_mask (torch.Tensor): [N, H2, W2] (optional) (S = H2*W2)
        """
        bs, C, H1, W1 = x.size()
        H2, W2 = source.size(-2), source.size(-1)
        

        if self.norm_before and not self.vit_norm:
            if self.pool_size == 1:
                query = self.norm1(x.permute(0,2,3,1)) # [N, H, W, C]
            elif self.dw_conv:
                query = self.norm1(self.aggregate(x).permute(0,2,3,1)) # [N, H, W, C]
            else:
                query = self.norm1(self.max_pool(x).permute(0,2,3,1)) # [N, H, W, C]
            if self.pool_size2 == 1:
                source = self.norm1(source.permute(0,2,3,1)) # [N, H, W, C]
            elif self.dw_conv2:
                source = self.norm1(self.aggregate2(source).permute(0,2,3,1)) # [N, H, W, C]
            else:
                source = self.norm1(self.max_pool(source).permute(0,2,3,1)) # [N, H, W, C]
        elif self.vit_norm:
            if self.pool_size == 1:
                query = self.norm1(x.permute(0,2,3,1)) # [N, H, W, C]
            elif self.dw_conv:
                query = self.aggregate(self.norm1(x.permute(0,2,3,1)).permute(0,3,1,2)).permute(0,2,3,1) # [N, H, W, C]
            else:
                query = self.max_pool(self.norm1(x.permute(0,2,3,1)).permute(0,3,1,2)).permute(0,2,3,1) # [N, H, W, C]
            if self.pool_size2 == 1:
                source = self.norm1(source.permute(0,2,3,1)) # [N, H, W, C]
            elif self.dw_conv2:
                source = self.aggregate2(self.norm1(source.permute(0,2,3,1)).permute(0,3,1,2)).permute(0,2,3,1) # [N, H, W, C]
            else:
                source = self.max_pool(self.norm1(source.permute(0,2,3,1)).permute(0,3,1,2)).permute(0,2,3,1) # [N, H, W, C]
        else:
            if self.pool_size == 1:
                query = x.permute(0,2,3,1) # [N, H, W, C]
            elif self.dw_conv:
                query = self.aggregate(x).permute(0,2,3,1) # [N, H, W, C]
            else:
                query = self.max_pool(x).permute(0,2,3,1) # [N, H, W, C]
            if self.pool_size2 == 1:
                source = source.permute(0,2,3,1) # [N, H, W, C]
            elif self.dw_conv2:
                source = self.aggregate2(source).permute(0,2,3,1) # [N, H, W, C]
            else:
                source = self.max_pool(source).permute(0,2,3,1) # [N, H, W, C]
        
        # projection
        if self.dw_proj:
            query = self.q_proj(query.permute(0,3,1,2)).permute(0,2,3,1)
            key = self.k_proj(source.permute(0,3,1,2)).permute(0,2,3,1)
            value = self.v_proj(source.permute(0,3,1,2)).permute(0,2,3,1)
        else:
            query, key, value = self.q_proj(query), self.k_proj(source), self.v_proj(source)
        
        # RoPE
        if self.rope:
            query = self.rope_pos_enc(query)
            if self.pool_size == 1 and self.pool_size2 == 4:
                key = self.rope_pos_enc(key, 4)
            else:
                key = self.rope_pos_enc(key)
        
        use_mask = x_mask is not None and source_mask is not None
        if bs == 1 or not use_mask:
            if use_mask:
                # downsample mask
                if self.pool_size ==1:
                    pass
                else:
                    x_mask = self.max_pool(x_mask.float()).bool() # [N, H1//pool, W1//pool]
                    
                if self.pool_size2 ==1:
                    pass
                else:
                    source_mask = self.max_pool(source_mask.float()).bool()

                mask_h0, mask_w0 = x_mask[0].sum(-2)[0], x_mask[0].sum(-1)[0]
                mask_h1, mask_w1 = source_mask[0].sum(-2)[0], source_mask[0].sum(-1)[0]
                
                query = query[:, :mask_h0, :mask_w0, :]
                key = key[:, :mask_h1, :mask_w1, :]
                value = value[:, :mask_h1, :mask_w1, :]
            else:
                assert x_mask is None and source_mask is None
            
            if PFLASH_AVAILABLE: # [N, H, W, C] -> [N, h, L, D]
                query = rearrange(query, 'n h w (nhead d) -> n nhead (h w) d', nhead=self.nhead, d=self.dim)
                key = rearrange(key, 'n h w (nhead d) -> n nhead (h w) d', nhead=self.nhead, d=self.dim)
                value = rearrange(value, 'n h w (nhead d) -> n nhead (h w) d', nhead=self.nhead, d=self.dim)
            else: # N L H D
                query = rearrange(query, 'n h w (nhead d) -> n (h w) nhead d', nhead=self.nhead, d=self.dim)
                key = rearrange(key, 'n h w (nhead d) -> n (h w) nhead d', nhead=self.nhead, d=self.dim)
                value = rearrange(value, 'n h w (nhead d) -> n (h w) nhead d', nhead=self.nhead, d=self.dim)
            
            message = self.attention(query, key, value, q_mask=None, kv_mask=None)  # [N, L, h, D] or [N, h, L, D]
            
            if PFLASH_AVAILABLE: # [N, h, L, D] -> [N, L, h, D]
                message = rearrange(message, 'n nhead L d -> n L nhead d', nhead=self.nhead, d=self.dim)

            if use_mask: # padding zero
                message = message.view(bs, mask_h0, mask_w0, self.nhead, self.dim)  # [N L h D]
                if mask_h0 != x_mask.size(-2):
                    message = torch.cat([message, torch.zeros(message.size(0), x_mask.size(-2)-mask_h0, x_mask.size(-1), self.nhead, self.dim, device=message.device, dtype=message.dtype)], dim=1)
                elif mask_w0 != x_mask.size(-1):
                    message = torch.cat([message, torch.zeros(message.size(0), x_mask.size(-2), x_mask.size(-1)-mask_w0, self.nhead, self.dim, device=message.device, dtype=message.dtype)], dim=2)
            else:
                assert x_mask is None and source_mask is None

            message = self.merge(message.reshape(bs, -1, self.nhead*self.dim))  # [N, L, C]
            message = rearrange(message, 'b (h w) c -> b c h w', h=H1//self.pool_size, w=W1//self.pool_size) # [N, C, H, W]
            
            if self.pool_size == 1:
                pass
            else:
                if self.scatter:
                    message = torch.repeat_interleave(message, self.pool_size, dim=-2)
                    message = torch.repeat_interleave(message, self.pool_size, dim=-1)
                    message = message * self.aggregate.weight.data.reshape(1, C, self.pool_size, self.pool_size).repeat(1,1,message.shape[-2]//self.pool_size,message.shape[-1]//self.pool_size)
                else:
                    message = torch.nn.functional.interpolate(message, scale_factor=self.pool_size, mode='bilinear', align_corners=False) # [N, C, H1, W1]
            
            if not self.norm_before and not self.vit_norm:
                message = self.norm1(message.permute(0,2,3,1)).permute(0,3,1,2) # [N, C, H, W]

            # feed-forward network
            if self.vit_norm:
                message_inter = (x + message)
                del x
                message = self.norm2(message_inter.permute(0, 2, 3, 1))
                message = self.mlp(message).permute(0, 3, 1, 2) # [N, C, H1, W1]
                return message_inter + message
            else:
                message = self.mlp(torch.cat([x, message], dim=1).permute(0, 2, 3, 1)) # [N, H1, W1, C]
                message = self.norm2(message).permute(0, 3, 1, 2) # [N, C, H1, W1]

                return x + message
        else: # mask with bs > 1
            if self.pool_size ==1:
                pass
            else:
                x_mask = self.max_pool(x_mask.float()).bool()
                    
            if self.pool_size2 ==1:
                pass
            else:
                source_mask = self.max_pool(source_mask.float()).bool()
            m_list = []
            for i in range(bs):
                mask_h0, mask_w0 = x_mask[i].sum(-2)[0], x_mask[i].sum(-1)[0]
                mask_h1, mask_w1 = source_mask[i].sum(-2)[0], source_mask[i].sum(-1)[0]
                
                q = query[i:i+1, :mask_h0, :mask_w0, :]
                k = key[i:i+1, :mask_h1, :mask_w1, :]
                v = value[i:i+1, :mask_h1, :mask_w1, :]

                if PFLASH_AVAILABLE: # [N, H, W, C] -> [N, h, L, D]
                    q = rearrange(q, 'n h w (nhead d) -> n nhead (h w) d', nhead=self.nhead, d=self.dim)
                    k = rearrange(k, 'n h w (nhead d) -> n nhead (h w) d', nhead=self.nhead, d=self.dim)
                    v = rearrange(v, 'n h w (nhead d) -> n nhead (h w) d', nhead=self.nhead, d=self.dim)
                else: # N L H D
                    q = rearrange(q, 'n h w (nhead d) -> n (h w) nhead d', nhead=self.nhead, d=self.dim)
                    k = rearrange(k, 'n h w (nhead d) -> n (h w) nhead d', nhead=self.nhead, d=self.dim)
                    v = rearrange(v, 'n h w (nhead d) -> n (h w) nhead d', nhead=self.nhead, d=self.dim)
                
                m = self.attention(q, k, v, q_mask=None, kv_mask=None) # [N, L, h, D] or [N, h, L, D]
                
                if PFLASH_AVAILABLE: # [N, h, L, D] -> [N, L, h, D]
                    m = rearrange(m, 'n nhead L d -> n L nhead d', nhead=self.nhead, d=self.dim)

                m = m.view(1, mask_h0, mask_w0, self.nhead, self.dim)
                if mask_h0 != x_mask.size(-2):
                    m = torch.cat([m, torch.zeros(1, x_mask.size(-2)-mask_h0, x_mask.size(-1), self.nhead, self.dim, device=m.device, dtype=m.dtype)], dim=1)
                elif mask_w0 != x_mask.size(-1):
                    m = torch.cat([m, torch.zeros(1, x_mask.size(-2), x_mask.size(-1)-mask_w0, self.nhead, self.dim, device=m.device, dtype=m.dtype)], dim=2)
                m_list.append(m)
            m = torch.cat(m_list, dim=0)
            
            m = self.merge(m.reshape(bs, -1, self.nhead*self.dim))  # [N, L, C]
            # m = m.reshape(bs, C, H1//self.pool_size, W1//self.pool_size) # [N, C, H, W] why this bug worked
            m = rearrange(m, 'b (h w) c -> b c h w', h=H1//self.pool_size, w=W1//self.pool_size) # [N, C, H, W]
            
            if self.pool_size == 1:
                pass
            else:
                if self.scatter:
                    m = torch.repeat_interleave(m, self.pool_size, dim=-2)
                    m = torch.repeat_interleave(m, self.pool_size, dim=-1)
                    m = m * self.aggregate.weight.data.reshape(1, C, self.pool_size, self.pool_size).repeat(1,1,m.shape[-2]//self.pool_size,m.shape[-1]//self.pool_size)
                else:
                    m = torch.nn.functional.interpolate(m, scale_factor=self.pool_size, mode='bilinear', align_corners=False) # [N, C, H1, W1]

            
            if not self.norm_before and not self.vit_norm:
                m = self.norm1(m.permute(0,2,3,1)).permute(0,3,1,2) # [N, C, H, W]

            # feed-forward network
            if self.vit_norm:
                m_inter = (x + m)
                del x
                m = self.norm2(m_inter.permute(0, 2, 3, 1))
                m = self.mlp(m).permute(0, 3, 1, 2) # [N, C, H1, W1]
                return m_inter + m
            else:
                m = self.mlp(torch.cat([x, m], dim=1).permute(0, 2, 3, 1)) # [N, H1, W1, C]
                m = self.norm2(m).permute(0, 3, 1, 2) # [N, C, H1, W1]

                return x + m

            return x + m

class AG_Conv_EncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear',
                 pool_size=4,
                 bn=True,
                 xformer=False,
                 leaky=-1.0,
                 dw_conv=False,
                 dw_conv2=False,
                 scatter=False,
                 norm_before=True,
                 ):
        super(AG_Conv_EncoderLayer, self).__init__()

        self.pool_size = pool_size
        self.dw_conv = dw_conv
        self.dw_conv2 = dw_conv2
        self.scatter = scatter
        self.norm_before = norm_before
        if self.dw_conv:
            self.aggregate = nn.Conv2d(d_model, d_model, kernel_size=pool_size, padding=0, stride=pool_size, bias=False, groups=d_model)
        if self.dw_conv2:
            self.aggregate2 = nn.Conv2d(d_model, d_model, kernel_size=pool_size, padding=0, stride=pool_size, bias=False, groups=d_model)
        self.dim = d_model // nhead
        self.nhead = nhead

        self.max_pool = torch.nn.MaxPool2d(kernel_size=self.pool_size, stride=self.pool_size)
        
        # multi-head attention
        if bn:
            method = 'dw_bn'
        else:
            method = 'dw'
        self.q_proj_conv = self._build_projection(d_model, d_model, method=method)
        self.k_proj_conv = self._build_projection(d_model, d_model, method=method)
        self.v_proj_conv = self._build_projection(d_model, d_model, method=method)
            
        if xformer:
            self.attention = XAttention()
        else:
            self.attention = LinearAttention() if attention == 'linear' else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        if leaky > 0:
            self.mlp = nn.Sequential(
                nn.Linear(d_model*2, d_model*2, bias=False),
                nn.LeakyReLU(leaky, True),
                nn.Linear(d_model*2, d_model, bias=False),
            )
            
        else:
            self.mlp = nn.Sequential(
                nn.Linear(d_model*2, d_model*2, bias=False),
                nn.ReLU(True),
                nn.Linear(d_model*2, d_model, bias=False),
            )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, C, H1, W1]
            source (torch.Tensor): [N, C, H2, W2]
            x_mask (torch.Tensor): [N, H1, W1] (optional) (L = H1*W1)
            source_mask (torch.Tensor): [N, H2, W2] (optional) (S = H2*W2)
        """
        bs = x.size(0)
        H1, W1 = x.size(-2), x.size(-1)
        H2, W2 = source.size(-2), source.size(-1)
        C = x.shape[-3]
        
        if self.norm_before:
            if self.dw_conv:
                query = self.norm1(self.aggregate(x).permute(0,2,3,1)).permute(0,3,1,2)
            else:
                query = self.norm1(self.max_pool(x).permute(0,2,3,1)).permute(0,3,1,2)
            if self.dw_conv2:
                source = self.norm1(self.aggregate2(source).permute(0,2,3,1)).permute(0,3,1,2)
            else:
                source = self.norm1(self.max_pool(source).permute(0,2,3,1)).permute(0,3,1,2)
        else:
            if self.dw_conv:
                query = self.aggregate(x)
            else:
                query = self.max_pool(x)
            if self.dw_conv2:
                source = self.aggregate2(source)
            else:
                source = self.max_pool(source)
        
        key, value = source, source

        query = self.q_proj_conv(query) # [N, C, H1//pool, W1//pool]
        key = self.k_proj_conv(key)
        value = self.v_proj_conv(value)

        use_mask = x_mask is not None and source_mask is not None
        if bs == 1 or not use_mask:
            if use_mask:
                x_mask = self.max_pool(x_mask.float()).bool() # [N, H1//pool, W1//pool]
                source_mask = self.max_pool(source_mask.float()).bool()

                mask_h0, mask_w0 = x_mask[0].sum(-2)[0], x_mask[0].sum(-1)[0]
                mask_h1, mask_w1 = source_mask[0].sum(-2)[0], source_mask[0].sum(-1)[0]
                
                query = query[:, :, :mask_h0, :mask_w0]
                key = key[:, :, :mask_h1, :mask_w1]
                value = value[:, :, :mask_h1, :mask_w1]
            
            else:
                assert x_mask is None and source_mask is None
            
            if PFLASH_AVAILABLE: # N H L D
                query = rearrange(query, 'n (nhead d) h w -> n nhead (h w) d', nhead=self.nhead, d=self.dim)
                key = rearrange(key, 'n (nhead d) h w -> n nhead (h w) d', nhead=self.nhead, d=self.dim) 
                value = rearrange(value, 'n (nhead d) h w -> n nhead (h w) d', nhead=self.nhead, d=self.dim)
                
            else: # N L H D
                query = rearrange(query, 'n (nhead d) h w -> n (h w) nhead d', nhead=self.nhead, d=self.dim)  # [N, L, H, D]
                key = rearrange(key, 'n (nhead d) h w -> n (h w) nhead d', nhead=self.nhead, d=self.dim)  # [N, S, H, D]
                value = rearrange(value, 'n (nhead d) h w -> n (h w) nhead d', nhead=self.nhead, d=self.dim)  # [N, S, H, D]
            
            message = self.attention(query, key, value, q_mask=None, kv_mask=None)  # [N, L, H, D] or [N, H, L, D]
            
            if PFLASH_AVAILABLE: # N H L D
                message = rearrange(message, 'n nhead L d -> n L nhead d', nhead=self.nhead, d=self.dim)

            if use_mask: # padding zero
                message = message.view(bs, mask_h0, mask_w0, self.nhead, self.dim)  # [N L H D]
                if mask_h0 != x_mask.size(-2):
                    message = torch.cat([message, torch.zeros(message.size(0), x_mask.size(-2)-mask_h0, x_mask.size(-1), self.nhead, self.dim, device=message.device, dtype=message.dtype)], dim=1)
                elif mask_w0 != x_mask.size(-1):
                    message = torch.cat([message, torch.zeros(message.size(0), x_mask.size(-2), x_mask.size(-1)-mask_w0, self.nhead, self.dim, device=message.device, dtype=message.dtype)], dim=2)
            else:
                assert x_mask is None and source_mask is None

            message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
            message = rearrange(message, 'b (h w) c -> b c h w', h=H1//self.pool_size, w=W1//self.pool_size) # [N, C, H, W]
            
            if self.scatter:
                message = torch.repeat_interleave(message, self.pool_size, dim=-2)
                message = torch.repeat_interleave(message, self.pool_size, dim=-1)
                message = message * self.aggregate.weight.data.reshape(1, C, self.pool_size, self.pool_size).repeat(1,1,message.shape[-2]//self.pool_size,message.shape[-1]//self.pool_size)
            else:
                message = torch.nn.functional.interpolate(message, scale_factor=self.pool_size, mode='bilinear', align_corners=False) # [N, C, H1, W1]
            
            if not self.norm_before:
                message = self.norm1(message.permute(0,2,3,1)).permute(0,3,1,2) # [N, C, H, W]

            # feed-forward network
            message = self.mlp(torch.cat([x, message], dim=1).permute(0, 2, 3, 1)) # [N, H1, W1, C]
            message = self.norm2(message).permute(0, 3, 1, 2) # [N, C, H1, W1]

            return x + message
        else: # mask with bs > 1
            x_mask = self.max_pool(x_mask.float()).bool()
            source_mask = self.max_pool(source_mask.float()).bool()
            m_list = []
            for i in range(bs):
                mask_h0, mask_w0 = x_mask[i].sum(-2)[0], x_mask[i].sum(-1)[0]
                mask_h1, mask_w1 = source_mask[i].sum(-2)[0], source_mask[i].sum(-1)[0]
                
                q = query[i:i+1, :, :mask_h0, :mask_w0]
                k = key[i:i+1, :, :mask_h1, :mask_w1]
                v = value[i:i+1, :, :mask_h1, :mask_w1]

                if PFLASH_AVAILABLE: # N H L D
                    q = rearrange(q, 'n (nhead d) h w -> n nhead (h w) d', nhead=self.nhead, d=self.dim)
                    k = rearrange(k, 'n (nhead d) h w -> n nhead (h w) d', nhead=self.nhead, d=self.dim) 
                    v = rearrange(v, 'n (nhead d) h w -> n nhead (h w) d', nhead=self.nhead, d=self.dim)
                    
                else: # N L H D
                    q = rearrange(q, 'n (nhead d) h w -> n (h w) nhead d', nhead=self.nhead, d=self.dim)  # [N, L, H, D]
                    k = rearrange(k, 'n (nhead d) h w -> n (h w) nhead d', nhead=self.nhead, d=self.dim)  # [N, S, H, D]
                    v = rearrange(v, 'n (nhead d) h w -> n (h w) nhead d', nhead=self.nhead, d=self.dim)  # [N, S, H, D]
                
                m = self.attention(q, k, v, q_mask=None, kv_mask=None)  # [N, L, H, D]
                
                if PFLASH_AVAILABLE: # N H L D
                    m = rearrange(m, 'n nhead L d -> n L nhead d', nhead=self.nhead, d=self.dim)

                m = m.view(1, mask_h0, mask_w0, self.nhead, self.dim)
                if mask_h0 != x_mask.size(-2):
                    m = torch.cat([m, torch.zeros(1, x_mask.size(-2)-mask_h0, x_mask.size(-1), self.nhead, self.dim, device=m.device, dtype=m.dtype)], dim=1)
                elif mask_w0 != x_mask.size(-1):
                    m = torch.cat([m, torch.zeros(1, x_mask.size(-2), x_mask.size(-1)-mask_w0, self.nhead, self.dim, device=m.device, dtype=m.dtype)], dim=2)
                m_list.append(m)
            m = torch.cat(m_list, dim=0)
            
            m = self.merge(m.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
            
            # m = m.reshape(bs, C, H1//self.pool_size, W1//self.pool_size) # [N, C, H, W] why this bug worked
            m = rearrange(m, 'b (h w) c -> b c h w', h=H1//self.pool_size, w=W1//self.pool_size) # [N, C, H, W]
            
            if self.scatter:
                m = torch.repeat_interleave(m, self.pool_size, dim=-2)
                m = torch.repeat_interleave(m, self.pool_size, dim=-1)
                m = m * self.aggregate.weight.data.reshape(1, C, self.pool_size, self.pool_size).repeat(1,1,m.shape[-2]//self.pool_size,m.shape[-1]//self.pool_size)
            else:
                m = torch.nn.functional.interpolate(m, scale_factor=self.pool_size, mode='bilinear', align_corners=False) # [N, C, H1, W1]
            
            if not self.norm_before:
                m = self.norm1(m.permute(0,2,3,1)).permute(0,3,1,2) # [N, C, H, W]

            # feed-forward network
            m = self.mlp(torch.cat([x, m], dim=1).permute(0, 2, 3, 1)) # [N, H1, W1, C]
            m = self.norm2(m).permute(0, 3, 1, 2) # [N, C, H1, W1]

            return x + m

    def _build_projection(self,
                          dim_in,
                          dim_out,
                          kernel_size=3,
                          padding=1,
                          stride=1,
                          method='dw_bn',
                          ):
        if method == 'dw_bn':
            proj = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(
                    dim_in,
                    dim_in,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=False,
                    groups=dim_in
                )),
                ('bn', nn.BatchNorm2d(dim_in)),
                # ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))
        elif method == 'avg':
            proj = nn.Sequential(OrderedDict([
                ('avg', nn.AvgPool2d(
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    ceil_mode=True
                )),
                # ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))
        elif method == 'linear':
            proj = None
        elif method == 'dw':
            proj = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(
                    dim_in,
                    dim_in,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=False,
                    groups=dim_in
                )),
                # ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))
        else:
            raise ValueError('Unknown method ({})'.format(method))

        return proj


class RoPELoFTREncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear',
                 rope=False,
                 token_mixer=None,
                 ):
        super(RoPELoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        if token_mixer is None:
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.rope = rope
        self.token_mixer = None
        if token_mixer is not None:
            self.token_mixer = token_mixer
            if token_mixer == 'dwcn':
                self.attention = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(
                        d_model,
                        d_model,
                        kernel_size=3,
                        padding=1,
                        stride=1,
                        bias=False,
                        groups=d_model
                    )),
                ]))
        elif self.rope:
            assert attention == 'linear'
            self.attention = RoPELinearAttention()

        if token_mixer is None:
            self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        if token_mixer is None:
            self.mlp = nn.Sequential(
                nn.Linear(d_model*2, d_model*2, bias=False),
                nn.ReLU(True),
                nn.Linear(d_model*2, d_model, bias=False),
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_model, bias=False),
                nn.ReLU(True),
                nn.Linear(d_model, d_model, bias=False),
            )
        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None, H=None, W=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, L, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        assert H*W == x.size(-2)

        # x = rearrange(x, 'n c h w -> n (h w) c')
        # source = rearrange(source, 'n c h w -> n (h w) c')
        query, key, value = x, source, source

        if self.token_mixer is not None:
            # multi-head attention
            m = self.norm1(x)
            m = rearrange(m, 'n (h w) c -> n c h w', h=H, w=W)
            m = self.attention(m)
            m = rearrange(m, 'n c h w -> n (h w) c')
            
            x = x + m
            x = x + self.mlp(self.norm2(x))
            return x
        else:
            # multi-head attention
            query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
            key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
            value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
            message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask, H=H, W=W)  # [N, L, (H, D)]
            message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
            message = self.norm1(message)

            # feed-forward network
            message = self.mlp(torch.cat([x, message], dim=2))
            message = self.norm2(message)

            return x + message

class LoFTREncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear',
                 xformer=False,
                 ):
        super(LoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        
        if xformer:
            self.attention = XAttention()
        else:
            self.attention = LinearAttention() if attention == 'linear' else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message

    def pro(self, x, source, x_mask=None, source_mask=None, profiler=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        with profiler.profile("proj*3"):
            query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
            key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
            value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        with profiler.profile("attention"):
            message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        with profiler.profile("merge"):
            message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        with profiler.profile("norm1"):
            message = self.norm1(message)

        # feed-forward network
        with profiler.profile("mlp"):
            message = self.mlp(torch.cat([x, message], dim=2))
        with profiler.profile("norm2"):
            message = self.norm2(message)

        return x + message

class PANEncoderLayer_cross(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear',
                 pool_size=4,
                 bn=True,
                 ):
        super(PANEncoderLayer_cross, self).__init__()

        self.pool_size = pool_size
        
        self.dim = d_model // nhead
        self.nhead = nhead

        self.max_pool = torch.nn.MaxPool2d(kernel_size=self.pool_size, stride=self.pool_size)
        # multi-head attention
        if bn:
            method = 'dw_bn'
        else:
            method = 'dw'
        self.qk_proj_conv = self._build_projection(d_model, d_model, method=method)
        self.v_proj_conv = self._build_projection(d_model, d_model, method=method)
            
            # self.q_proj = nn.Linear(d_mosdel, d_model, bias=False)
            # self.k_proj = nn.Linear(d_model, d_model, bias=False)
            # self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # self.norm1 = nn.BatchNorm2d(d_model)

    def forward(self, x1, x2, x1_mask=None, x2_mask=None):
        """
        Args:
            x (torch.Tensor): [N, C, H1, W1]
            source (torch.Tensor): [N, C, H2, W2]
            x_mask (torch.Tensor): [N, H1, W1] (optional) (L = H1*W1)
            source_mask (torch.Tensor): [N, H2, W2] (optional) (S = H2*W2)
        """
        bs = x1.size(0)
        H1, W1 = x1.size(-2) // self.pool_size, x1.size(-1) // self.pool_size
        H2, W2 = x2.size(-2) // self.pool_size, x2.size(-1) // self.pool_size
        
        query = self.norm1(self.max_pool(x1).permute(0,2,3,1)).permute(0,3,1,2)
        key = self.norm1(self.max_pool(x2).permute(0,2,3,1)).permute(0,3,1,2)
        v2 = self.norm1(self.max_pool(x2).permute(0,2,3,1)).permute(0,3,1,2)
        v1 = self.norm1(self.max_pool(x1).permute(0,2,3,1)).permute(0,3,1,2)
        
        # multi-head attention
        query = self.qk_proj_conv(query) # [N, C, H1//pool, W1//pool]
        key = self.qk_proj_conv(key)
        v2 = self.v_proj_conv(v2)
        v1 = self.v_proj_conv(v1)
        
        C = query.shape[-3]
        if x1_mask is not None and x2_mask is not None:
            x1_mask = self.max_pool(x1_mask.float()).bool() # [N, H1//pool, W1//pool]
            x2_mask = self.max_pool(x2_mask.float()).bool()

            mask_h1, mask_w1 = x1_mask[0].sum(-2)[0], x1_mask[0].sum(-1)[0]
            mask_h2, mask_w2 = x2_mask[0].sum(-2)[0], x2_mask[0].sum(-1)[0]
            
            query = query[:, :, :mask_h1, :mask_w1]
            key = key[:, :, :mask_h2, :mask_w2]
            v1 = v1[:, :, :mask_h1, :mask_w1]
            v2 = v2[:, :, :mask_h2, :mask_w2]
            x1_mask = x1_mask[:, :mask_h1, :mask_w1]
            x2_mask = x2_mask[:, :mask_h2, :mask_w2]
        
        else:
            assert x1_mask is None and x2_mask is None

        query = rearrange(query, 'n (nhead d) h w -> n (h w) nhead d', nhead=self.nhead, d=self.dim)  # [N, L, H, D]
        key = rearrange(key, 'n (nhead d) h w -> n (h w) nhead d', nhead=self.nhead, d=self.dim)  # [N, S, H, D]
        v2 = rearrange(v2, 'n (nhead d) h w -> n (h w) nhead d', nhead=self.nhead, d=self.dim)  # [N, S, H, D]
        v1 = rearrange(v1, 'n (nhead d) h w -> n (h w) nhead d', nhead=self.nhead, d=self.dim)  # [N, S, H, D]
        if x2_mask is not None or x1_mask is not None:
            x1_mask = x1_mask.flatten(-2)
            x2_mask = x2_mask.flatten(-2)
        
        
        QK = torch.einsum("nlhd,nshd->nlsh", query, key)
        with torch.autocast(enabled=False, device_type='cuda'):
            if x2_mask is not None or x1_mask is not None:
                # S1 = S2.transpose(-2,-3).masked_fill(~(x_mask[:, None, :, None] * source_mask[:, :, None, None]), -1e9) # float('-inf')
                QK = QK.float().masked_fill_(~(x1_mask[:, :, None, None] * x2_mask[:, None, :, None]), -1e9) # float('-inf')
        
    
            # Compute the attention and the weighted average
            softmax_temp = 1. / query.size(3)**.5  # sqrt(D)
            S1 = torch.softmax(softmax_temp * QK, dim=2)
            S2 = torch.softmax(softmax_temp * QK, dim=3)

        m1 = torch.einsum("nlsh,nshd->nlhd", S1, v2)
        m2 = torch.einsum("nlsh,nlhd->nshd", S2, v1)
        
        if x1_mask is not None and x2_mask is not None:
            m1 = m1.view(bs, mask_h1, mask_w1, self.nhead, self.dim) 
            if mask_h1 != H1:
                m1 = torch.cat([m1, torch.zeros(m1.size(0), H1-mask_h1, W1, self.nhead, self.dim, device=m1.device, dtype=m1.dtype)], dim=1)
            elif mask_w1 != W1:
                m1 = torch.cat([m1, torch.zeros(m1.size(0), H1, W1-mask_w1, self.nhead, self.dim, device=m1.device, dtype=m1.dtype)], dim=2)
        else:
            assert x1_mask is None and x2_mask is None
            
        m1 = self.merge(m1.reshape(bs, -1, self.nhead*self.dim))  # [N, L, C]
        m1 = rearrange(m1, 'b (h w) c -> b c h w', h=H1, w=W1) # [N, C, H, W]
        m1 = torch.nn.functional.interpolate(m1, scale_factor=self.pool_size, mode='bilinear', align_corners=False) # [N, C, H1, W1]
        # feed-forward network
        m1 = self.mlp(torch.cat([x1, m1], dim=1).permute(0, 2, 3, 1)) # [N, H1, W1, C]
        m1 = self.norm2(m1).permute(0, 3, 1, 2) # [N, C, H1, W1]

        if x1_mask is not None and x2_mask is not None:
            m2 = m2.view(bs, mask_h2, mask_w2, self.nhead, self.dim)
            if mask_h2 != H2:
                m2 = torch.cat([m2, torch.zeros(m2.size(0), H2-mask_h2, W2, self.nhead, self.dim, device=m2.device, dtype=m2.dtype)], dim=1)
            elif mask_w2 != W2:
                m2 = torch.cat([m2, torch.zeros(m2.size(0), H2, W2-mask_w2, self.nhead, self.dim, device=m2.device, dtype=m2.dtype)], dim=2)
        else:
            assert x1_mask is None and x2_mask is None
            
        m2 = self.merge(m2.reshape(bs, -1, self.nhead*self.dim))  # [N, L, C]
        m2 = rearrange(m2, 'b (h w) c -> b c h w', h=H2, w=W2) # [N, C, H, W]
        m2 = torch.nn.functional.interpolate(m2, scale_factor=self.pool_size, mode='bilinear', align_corners=False) # [N, C, H1, W1]
        # feed-forward network
        m2 = self.mlp(torch.cat([x2, m2], dim=1).permute(0, 2, 3, 1)) # [N, H1, W1, C]
        m2 = self.norm2(m2).permute(0, 3, 1, 2) # [N, C, H1, W1]
        
        return x1 + m1, x2 + m2

    def _build_projection(self,
                          dim_in,
                          dim_out,
                          kernel_size=3,
                          padding=1,
                          stride=1,
                          method='dw_bn',
                          ):
        if method == 'dw_bn':
            proj = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(
                    dim_in,
                    dim_in,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=False,
                    groups=dim_in
                )),
                ('bn', nn.BatchNorm2d(dim_in)),
                # ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))
        elif method == 'avg':
            proj = nn.Sequential(OrderedDict([
                ('avg', nn.AvgPool2d(
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    ceil_mode=True
                )),
                # ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))
        elif method == 'linear':
            proj = None
        elif method == 'dw':
            proj = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(
                    dim_in,
                    dim_in,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=False,
                    groups=dim_in
                )),
                # ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))
        else:
            raise ValueError('Unknown method ({})'.format(method))

        return proj

class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(LocalFeatureTransformer, self).__init__()
        
        self.full_config = config
        self.fine = False
        if 'coarse' not in config:
            self.fine = True # fine attention
        else:
            config = config['coarse']
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        self.pan = config['pan']
        self.bidirect = config['bidirection']
        # prune
        self.pool_size = config['pool_size']
        self.matchability = False
        self.depth_confidence = -1.0
        self.width_confidence = -1.0
        # self.depth_confidence = config['depth_confidence']
        # self.width_confidence = config['width_confidence']
        # self.matchability = self.depth_confidence > 0 or self.width_confidence > 0
        # self.thr = self.full_config['match_coarse']['thr']
        if not self.fine:
            # asy
            self.asymmetric = config['asymmetric']
            self.asymmetric_self = config['asymmetric_self']
            # aggregate
            self.aggregate = config['dwconv']
            # RoPE
            self.rope = config['rope']
            # absPE
            self.abspe = config['abspe']
            
        else:
            self.rope, self.asymmetric, self.asymmetric_self, self.aggregate = False, False, False, False
        if self.matchability:
            self.n_layers = len(self.layer_names) // 2
            assert self.n_layers == 4
            self.log_assignment = nn.ModuleList(
                [MatchAssignment(self.d_model) for _ in range(self.n_layers)])
            self.token_confidence = nn.ModuleList([
                TokenConfidence(self.d_model) for _ in range(self.n_layers-1)])
            
            self.CoarseMatching = CoarseMatching(self.full_config['match_coarse'])

        # self only
        # if self.rope:
        #     self_layer = RoPELoFTREncoderLayer(config['d_model'], config['nhead'], config['attention'], config['rope'], config['token_mixer'])
        #     self.layers = nn.ModuleList([copy.deepcopy(self_layer) for _ in range(len(self.layer_names))])
        
        if self.bidirect:
            assert config['xformer'] is False and config['pan'] is True
            self_layer = PANEncoderLayer(config['d_model'], config['nhead'], config['attention'], config['pool_size'], config['bn'], config['xformer'])
            cross_layer = PANEncoderLayer_cross(config['d_model'], config['nhead'], config['attention'], config['pool_size'], config['bn'])
            self.layers = nn.ModuleList([copy.deepcopy(self_layer) if _ == 'self' else copy.deepcopy(cross_layer) for _ in self.layer_names])
        else:
            if self.aggregate:
                if self.rope:
                    # assert config['npe'][0] == 832 and config['npe'][1] == 832 and config['npe'][2] == 832 and config['npe'][3] == 832
                    logger.info(f'npe trainH,trainW,testH,testW: {config["npe"][0]}, {config["npe"][1]}, {config["npe"][2]}, {config["npe"][3]}')
                    self_layer = AG_RoPE_EncoderLayer(config['d_model'], config['nhead'], config['attention'], config['pool_size'], config['pool_size2'],
                                                      config['xformer'], config['leaky'], config['dwconv'], config['dwconv2'], config['scatter'],
                                                      config['norm_before'], config['rope'], config['npe'], config['vit_norm'], config['rope_dwproj'])
                    cross_layer = AG_RoPE_EncoderLayer(config['d_model'], config['nhead'], config['attention'], config['pool_size'], config['pool_size2'],
                                                      config['xformer'], config['leaky'], config['dwconv'], config['dwconv2'], config['scatter'],
                                                      config['norm_before'], False, config['npe'], config['vit_norm'], config['rope_dwproj'])
                    self.layers = nn.ModuleList([copy.deepcopy(self_layer) if _ == 'self' else copy.deepcopy(cross_layer) for _ in self.layer_names])
                elif self.abspe:
                    logger.info(f'npe trainH,trainW,testH,testW: {config["npe"][0]}, {config["npe"][1]}, {config["npe"][2]}, {config["npe"][3]}')
                    self_layer = AG_RoPE_EncoderLayer(config['d_model'], config['nhead'], config['attention'], config['pool_size'], config['pool_size2'],
                                                      config['xformer'], config['leaky'], config['dwconv'], config['dwconv2'], config['scatter'],
                                                      config['norm_before'], False, config['npe'], config['vit_norm'], config['rope_dwproj'])
                    cross_layer = AG_RoPE_EncoderLayer(config['d_model'], config['nhead'], config['attention'], config['pool_size'], config['pool_size2'],
                                                      config['xformer'], config['leaky'], config['dwconv'], config['dwconv2'], config['scatter'],
                                                      config['norm_before'], False, config['npe'], config['vit_norm'], config['rope_dwproj'])
                    self.layers = nn.ModuleList([copy.deepcopy(self_layer) if _ == 'self' else copy.deepcopy(cross_layer) for _ in self.layer_names])
                    
                else:
                    encoder_layer = AG_Conv_EncoderLayer(config['d_model'], config['nhead'], config['attention'], config['pool_size'], config['bn'],
                                                      config['xformer'], config['leaky'], config['dwconv'], config['scatter'],
                                                      config['norm_before'])
                    self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
            else:
                encoder_layer = PANEncoderLayer(config['d_model'], config['nhead'], config['attention'], config['pool_size'], 
                                                config['bn'], config['xformer'], config['leaky'], config['dwconv'], config['scatter']) \
                                                if config['pan'] else LoFTREncoderLayer(config['d_model'], config['nhead'], 
                                                                                        config['attention'], config['xformer'])
                self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, mask0=None, mask1=None, data=None):
        """
        Args:
            feat0 (torch.Tensor): [N, C, H, W]
            feat1 (torch.Tensor): [N, C, H, W]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """
        # nchw for pan and n(hw)c for loftr
        assert self.d_model == feat0.size(1) or self.d_model == feat0.size(-1), "the feature number of src and transformer must be equal"
        H0, W0, H1, W1 = feat0.size(-2), feat0.size(-1), feat1.size(-2), feat1.size(-1)
        bs = feat0.shape[0]
        padding = False
        if bs == 1 and mask0 is not None and mask1 is not None and self.pan: # NCHW for pan
            mask_H0, mask_W0 = mask0.size(-2), mask0.size(-1)
            mask_H1, mask_W1 = mask1.size(-2), mask1.size(-1)
            mask_h0, mask_w0 = mask0[0].sum(-2)[0], mask0[0].sum(-1)[0]
            mask_h1, mask_w1 = mask1[0].sum(-2)[0], mask1[0].sum(-1)[0]
            
            #round to self.pool_size
            if self.pan:
                mask_h0, mask_w0, mask_h1, mask_w1 = mask_h0//self.pool_size*self.pool_size, mask_w0//self.pool_size*self.pool_size, mask_h1//self.pool_size*self.pool_size, mask_w1//self.pool_size*self.pool_size
            
            feat0 = feat0[:, :, :mask_h0, :mask_w0]
            feat1 = feat1[:, :, :mask_h1, :mask_w1]

            padding = True

        # rope self only
        # if self.rope:
        #     feat0, feat1 = rearrange(feat0, 'b c h w -> b (h w) c'), rearrange(feat1, 'b c h w -> b (h w) c')
        # prune
        if padding:
            l0, l1 = mask_h0 * mask_w0, mask_h1 * mask_w1
        else:
            l0, l1 = H0 * W0, H1 * W1
        do_early_stop = self.depth_confidence > 0
        do_point_pruning = self.width_confidence > 0
        if do_point_pruning:
            ind0 = torch.arange(0, l0, device=feat0.device)[None]
            ind1 = torch.arange(0, l1, device=feat0.device)[None]
            # We store the index of the layer at which pruning is detected.
            prune0 = torch.ones_like(ind0)
            prune1 = torch.ones_like(ind1)
        if do_early_stop:
            token0, token1 = None, None

        for i, (layer, name) in enumerate(zip(self.layers, self.layer_names)):
            if padding:
                mask0, mask1 = None, None
            if name == 'self':
                # if self.rope:
                #     feat0 = layer(feat0, feat0, mask0, mask1, H0, W0)
                #     feat1 = layer(feat1, feat1, mask0, mask1, H1, W1)
                if self.asymmetric:
                    assert False, 'not worked'
                    # feat0 = layer(feat0, feat0, mask0, mask1)
                    feat1 = layer(feat1, feat1, mask1, mask1)
                else:
                    feat0 = layer(feat0, feat0, mask0, mask0)
                    feat1 = layer(feat1, feat1, mask1, mask1)
            elif name == 'cross':
                if self.bidirect:
                    feat0, feat1 = layer(feat0, feat1, mask0, mask1)
                else:
                    if self.asymmetric or self.asymmetric_self:
                        assert False, 'not worked'
                        feat0 = layer(feat0, feat1, mask0, mask1)
                    else:
                        feat0 = layer(feat0, feat1, mask0, mask1)
                        feat1 = layer(feat1, feat0, mask1, mask0)
                
                if i == len(self.layer_names) - 1 and not self.training:
                    continue
                if self.matchability:
                    desc0, desc1 = rearrange(feat0, 'b c h w -> b (h w) c'), rearrange(feat1, 'b c h w -> b (h w) c')
                if do_early_stop:
                    token0, token1 = self.token_confidence[i//2](desc0, desc1)
                    if self.check_if_stop(token0, token1, i, l0+l1) and not self.training:
                        break
                if do_point_pruning:
                    scores0, scores1 = self.log_assignment[i//2].scores(desc0, desc1)
                    mask0 = self.get_pruning_mask(token0, scores0, i)
                    mask1 = self.get_pruning_mask(token1, scores1, i)
                    ind0, ind1 = ind0[mask0][None], ind1[mask1][None]
                    feat0, feat1 = desc0[mask0][None], desc1[mask1][None]
                    if feat0.shape[-2] == 0 or desc1.shape[-2] == 0:
                        break
                    prune0[:, ind0] += 1
                    prune1[:, ind1] += 1
                if self.training and self.matchability:
                    scores, _, matchability0, matchability1 = self.log_assignment[i//2](desc0, desc1)
                    m0_full = torch.zeros((bs, mask_h0 * mask_w0), device=matchability0.device, dtype=matchability0.dtype)
                    m0_full.scatter(1, ind0, matchability0.squeeze(-1))
                    if padding and self.d_model == feat0.size(1):
                        m0_full = m0_full.reshape(bs, mask_h0, mask_w0)
                        bs, c, mask_h0, mask_w0 = feat0.size()
                        if mask_h0 != mask_H0:
                            m0_full = torch.cat([m0_full, torch.zeros(bs, mask_H0-mask_h0, mask_w0, device=m0_full.device, dtype=m0_full.dtype)], dim=1)
                        elif mask_w0 != mask_W0:
                            m0_full = torch.cat([m0_full, torch.zeros(bs, mask_h0, mask_W0-mask_w0, device=m0_full.device, dtype=m0_full.dtype)], dim=2)
                        m0_full = m0_full.reshape(bs, mask_H0*mask_W0)
                    m1_full = torch.zeros((bs, mask_h1 * mask_w1), device=matchability0.device, dtype=matchability0.dtype)
                    m1_full.scatter(1, ind1, matchability1.squeeze(-1))
                    if padding and self.d_model == feat1.size(1):
                        m1_full = m1_full.reshape(bs, mask_h1, mask_w1)
                        bs, c, mask_h1, mask_w1 = feat1.size()
                        if mask_h1 != mask_H1:
                            m1_full = torch.cat([m1_full, torch.zeros(bs, mask_H1-mask_h1, mask_w1, device=m1_full.device, dtype=m1_full.dtype)], dim=1)
                        elif mask_w1 != mask_W1:
                            m1_full = torch.cat([m1_full, torch.zeros(bs, mask_h1, mask_W1-mask_w1, device=m1_full.device, dtype=m1_full.dtype)], dim=2)
                        m1_full = m1_full.reshape(bs, mask_H1*mask_W1)
                    data.update({'matchability0_'+str(i//2): m0_full, 'matchability1_'+str(i//2): m1_full})
                    m0, m1, mscores0, mscores1 = filter_matches(
                        scores, self.thr)
                    if do_point_pruning:
                        m0_ = torch.full((bs, l0), -1, device=m0.device, dtype=m0.dtype)
                        m1_ = torch.full((bs, l1), -1, device=m1.device, dtype=m1.dtype)
                        m0_[:, ind0] = torch.where(
                            m0 == -1, -1, ind1.gather(1, m0.clamp(min=0)))
                        m1_[:, ind1] = torch.where(
                            m1 == -1, -1, ind0.gather(1, m1.clamp(min=0)))
                        mscores0_ = torch.zeros((bs, l0), device=mscores0.device)
                        mscores1_ = torch.zeros((bs, l1), device=mscores1.device)
                        mscores0_[:, ind0] = mscores0
                        mscores1_[:, ind1] = mscores1
                        m0, m1, mscores0, mscores1 = m0_, m1_, mscores0_, mscores1_
                    if padding and self.d_model == feat0.size(1):
                        m0 = m0.reshape(bs, mask_h0, mask_w0)
                        bs, c, mask_h0, mask_w0 = feat0.size()
                        if mask_h0 != mask_H0:
                            m0 = torch.cat([m0, -torch.ones(bs, mask_H0-mask_h0, mask_w0, device=m0.device, dtype=m0.dtype)], dim=1)
                        elif mask_w0 != mask_W0:
                            m0 = torch.cat([m0, -torch.ones(bs, mask_h0, mask_W0-mask_w0, device=m0.device, dtype=m0.dtype)], dim=2)
                        m0 = m0.reshape(bs, mask_H0*mask_W0)
                    if padding and self.d_model == feat1.size(1):
                        m1 = m1.reshape(bs, mask_h1, mask_w1)
                        bs, c, mask_h1, mask_w1 = feat1.size()
                        if mask_h1 != mask_H1:
                            m1 = torch.cat([m1, -torch.ones(bs, mask_H1-mask_h1, mask_w1, device=m1.device, dtype=m1.dtype)], dim=1)
                        elif mask_w1 != mask_W1:
                            m1 = torch.cat([m1, -torch.ones(bs, mask_h1, mask_W1-mask_w1, device=m1.device, dtype=m1.dtype)], dim=2)
                        m1 = m1.reshape(bs, mask_H1*mask_W1)
                    data.update({'matches0_'+str(i//2): m0, 'matches1_'+str(i//2): m1})
                    conf = torch.zeros((bs, l0 * l1), device=scores.device, dtype=scores.dtype)
                    ind = ind0[...,None] * l1 + ind1[:,None,:]
                    # conf[ind.reshape(bs, -1)] = scores.reshape(bs, -1).exp()
                    conf.scatter(1, ind.reshape(bs, -1), scores.reshape(bs, -1).exp())
                    if padding and self.d_model == feat0.size(1):
                        conf = conf.reshape(bs, mask_h0, mask_w0, mask_h1, mask_w1)
                        bs, c, mask_h0, mask_w0 = feat0.size()
                        if mask_h0 != mask_H0:
                            conf = torch.cat([conf, torch.zeros(bs, mask_H0-mask_h0, mask_w0, mask_h1, mask_w1, device=conf.device, dtype=conf.dtype)], dim=1)
                        elif mask_w0 != mask_W0:
                            conf = torch.cat([conf, torch.zeros(bs, mask_h0, mask_W0-mask_w0, mask_h1, mask_w1, device=conf.device, dtype=conf.dtype)], dim=2)
                        bs, c, mask_h1, mask_w1 = feat1.size()
                        if mask_h1 != mask_H1:
                            conf = torch.cat([conf, torch.zeros(bs, mask_H0, mask_W0, mask_H1-mask_h1, mask_W1, device=conf.device, dtype=conf.dtype)], dim=3)
                        elif mask_w1 != mask_W1:
                            conf = torch.cat([conf, torch.zeros(bs, mask_H0, mask_W0, mask_H1, mask_W1-mask_w1, device=conf.device, dtype=conf.dtype)], dim=4)
                        conf = conf.reshape(bs, mask_H0*mask_W0, mask_H1*mask_W1)
                    data.update({'conf_matrix_'+str(i//2): conf})


                    
            else:
                raise KeyError

        if self.matchability and not self.training:
            scores, _, matchability0, matchability1 = self.log_assignment[i//2](desc0, desc1)
            conf = torch.zeros((bs, l0 * l1), device=scores.device, dtype=scores.dtype)
            ind = ind0[...,None] * l1 + ind1[:,None,:]
            # conf[ind.reshape(bs, -1)] = scores.reshape(bs, -1).exp()
            conf.scatter(1, ind.reshape(bs, -1), scores.reshape(bs, -1).exp())
            if padding and self.d_model == feat0.size(1):
                conf = conf.reshape(bs, mask_h0, mask_w0, mask_h1, mask_w1)
                bs, c, mask_h0, mask_w0 = feat0.size()
                if mask_h0 != mask_H0:
                    conf = torch.cat([conf, torch.zeros(bs, mask_H0-mask_h0, mask_w0, mask_h1, mask_w1, device=conf.device, dtype=conf.dtype)], dim=1)
                elif mask_w0 != mask_W0:
                    conf = torch.cat([conf, torch.zeros(bs, mask_h0, mask_W0-mask_w0, mask_h1, mask_w1, device=conf.device, dtype=conf.dtype)], dim=2)
                bs, c, mask_h1, mask_w1 = feat1.size()
                if mask_h1 != mask_H1:
                    conf = torch.cat([conf, torch.zeros(bs, mask_H0, mask_W0, mask_H1-mask_h1, mask_W1, device=conf.device, dtype=conf.dtype)], dim=3)
                elif mask_w1 != mask_W1:
                    conf = torch.cat([conf, torch.zeros(bs, mask_H0, mask_W0, mask_H1, mask_W1-mask_w1, device=conf.device, dtype=conf.dtype)], dim=4)
                conf = conf.reshape(bs, mask_H0*mask_W0, mask_H1*mask_W1)
            data.update({'conf_matrix': conf})
            data.update(**self.CoarseMatching.get_coarse_match(conf, data))
            # m0, m1, mscores0, mscores1 = filter_matches(
            #     scores, self.conf.filter_threshold)

            # matches, mscores = [], []
            # for k in range(b):
            #     valid = m0[k] > -1
            #     m_indices_0 = torch.where(valid)[0]
            #     m_indices_1 = m0[k][valid]
            #     if do_point_pruning:
            #         m_indices_0 = ind0[k, m_indices_0]
            #         m_indices_1 = ind1[k, m_indices_1]
            #     matches.append(torch.stack([m_indices_0, m_indices_1], -1))
            #     mscores.append(mscores0[k][valid])

            # # TODO: Remove when hloc switches to the compact format.
            # if do_point_pruning:
            #     m0_ = torch.full((b, m), -1, device=m0.device, dtype=m0.dtype)
            #     m1_ = torch.full((b, n), -1, device=m1.device, dtype=m1.dtype)
            #     m0_[:, ind0] = torch.where(
            #         m0 == -1, -1, ind1.gather(1, m0.clamp(min=0)))
            #     m1_[:, ind1] = torch.where(
            #         m1 == -1, -1, ind0.gather(1, m1.clamp(min=0)))
            #     mscores0_ = torch.zeros((b, m), device=mscores0.device)
            #     mscores1_ = torch.zeros((b, n), device=mscores1.device)
            #     mscores0_[:, ind0] = mscores0
            #     mscores1_[:, ind1] = mscores1
            #     m0, m1, mscores0, mscores1 = m0_, m1_, mscores0_, mscores1_

            # pred = {
            #     'matches0': m0,
            #     'matches1': m1,
            #     'matching_scores0': mscores0,
            #     'matching_scores1': mscores1,
            #     'stop': i+1,
            #     'matches': matches,
            #     'scores': mscores,
            # }
            
            # if do_point_pruning:
            #     pred.update(dict(prune0=prune0, prune1=prune1))
            # return pred


        if padding and self.d_model == feat0.size(1):
            bs, c, mask_h0, mask_w0 = feat0.size()
            if mask_h0 != mask_H0:
                feat0 = torch.cat([feat0, torch.zeros(bs, c, mask_H0-mask_h0, mask_W0, device=feat0.device, dtype=feat0.dtype)], dim=-2)
            elif mask_w0 != mask_W0:
                feat0 = torch.cat([feat0, torch.zeros(bs, c, mask_H0, mask_W0-mask_w0, device=feat0.device, dtype=feat0.dtype)], dim=-1)
            bs, c, mask_h1, mask_w1 = feat1.size()
            if mask_h1 != mask_H1:
                feat1 = torch.cat([feat1, torch.zeros(bs, c, mask_H1-mask_h1, mask_W1, device=feat1.device, dtype=feat1.dtype)], dim=-2)
            elif mask_w1 != mask_W1:
                feat1 = torch.cat([feat1, torch.zeros(bs, c, mask_H1, mask_W1-mask_w1, device=feat1.device, dtype=feat1.dtype)], dim=-1)

        return feat0, feat1
    
    def pro(self, feat0, feat1, mask0=None, mask1=None, profiler=None):
        """
        Args:
            feat0 (torch.Tensor): [N, C, H, W]
            feat1 (torch.Tensor): [N, C, H, W]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """

        assert self.d_model == feat0.size(1) or self.d_model == feat0.size(-1), "the feature number of src and transformer must be equal"
        with profiler.profile("LoFTR_transformer_attention"):
            for layer, name in zip(self.layers, self.layer_names):
                if name == 'self':
                    feat0 = layer.pro(feat0, feat0, mask0, mask0, profiler=profiler)
                    feat1 = layer.pro(feat1, feat1, mask1, mask1, profiler=profiler)
                elif name == 'cross':
                    feat0 = layer.pro(feat0, feat1, mask0, mask1, profiler=profiler)
                    feat1 = layer.pro(feat1, feat0, mask1, mask0, profiler=profiler)
                else:
                    raise KeyError

        return feat0, feat1

    def confidence_threshold(self, layer_index: int) -> float:
        """ scaled confidence threshold """
        threshold = 0.8 + 0.1 * np.exp(-4.0 * layer_index / self.n_layers)
        return np.clip(threshold, 0, 1)

    def get_pruning_mask(self, confidences: torch.Tensor, scores: torch.Tensor,
                         layer_index: int) -> torch.Tensor:
        """ mask points which should be removed """
        threshold = self.confidence_threshold(layer_index)
        if confidences is not None:
            scores = torch.where(
                confidences > threshold, scores, scores.new_tensor(1.0))
        return scores > (1 - self.width_confidence)

    def check_if_stop(self,
                      confidences0: torch.Tensor,
                      confidences1: torch.Tensor,
                      layer_index: int, num_points: int) -> torch.Tensor:
        """ evaluate stopping condition"""
        confidences = torch.cat([confidences0, confidences1], -1)
        threshold = self.confidence_threshold(layer_index)
        pos = 1.0 - (confidences < threshold).float().sum() / num_points
        return pos > self.depth_confidence
