import math
import torch
from torch import nn


class PositionEncodingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(256, 256), temp_bug_fix=True, npe=False):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
            temp_bug_fix (bool): As noted in this [issue](https://github.com/zju3dv/LoFTR/issues/41),
                the original implementation of LoFTR includes a bug in the pos-enc impl, which has little impact
                on the final performance. For now, we keep both impls for backward compatability.
                We will remove the buggy impl after re-training all variants of our released models.
        """
        super().__init__()

        pe = torch.zeros((d_model, *max_shape))
        y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)

        assert npe is not None
        if npe is not None:
            if isinstance(npe, bool):
                train_res_H, train_res_W, test_res_H, test_res_W = 832, 832, 832, 832
                print('loftr no npe!!!!', npe)
            else:
                print('absnpe!!!!', npe)
                train_res_H, train_res_W, test_res_H, test_res_W = npe[0], npe[1], npe[2], npe[3] # train_res_H, train_res_W, test_res_H, test_res_W
                y_position, x_position = y_position * train_res_H / test_res_H, x_position * train_res_W / test_res_W

        if temp_bug_fix:
            div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / (d_model//2)))
        else:  # a buggy implementation (for backward compatability only)
            div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / d_model//2))
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # [1, C, H, W]

    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
        return x + self.pe[:, :, :x.size(2), :x.size(3)]
    
class RoPEPositionEncodingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(256, 256), npe=None, ropefp16=True):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
            temp_bug_fix (bool): As noted in this [issue](https://github.com/zju3dv/LoFTR/issues/41),
                the original implementation of LoFTR includes a bug in the pos-enc impl, which has little impact
                on the final performance. For now, we keep both impls for backward compatability.
                We will remove the buggy impl after re-training all variants of our released models.
        """
        super().__init__()

        # pe = torch.zeros((d_model, *max_shape))
        # y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(-1)
        # x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(-1)
        i_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(-1) # [H, 1]
        j_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(-1) # [W, 1]
        
        assert npe is not None
        if npe is not None:
            train_res_H, train_res_W, test_res_H, test_res_W = npe[0], npe[1], npe[2], npe[3] # train_res_H, train_res_W, test_res_H, test_res_W
            i_position, j_position = i_position * train_res_H / test_res_H, j_position * train_res_W / test_res_W
        
        div_term = torch.exp(torch.arange(0, d_model//4, 1).float() * (-math.log(10000.0) / (d_model//4)))
        div_term = div_term[None, None, :]  # [1, 1, C//4]
        # pe[0::4, :, :] = torch.sin(x_position * div_term)
        # pe[1::4, :, :] = torch.cos(x_position * div_term)
        # pe[2::4, :, :] = torch.sin(y_position * div_term)
        # pe[3::4, :, :] = torch.cos(y_position * div_term)
        sin = torch.zeros(*max_shape, d_model//2, dtype=torch.float16 if ropefp16 else torch.float32)
        cos = torch.zeros(*max_shape, d_model//2, dtype=torch.float16 if ropefp16 else torch.float32)
        sin[:, :, 0::2] = torch.sin(i_position * div_term).half() if ropefp16 else torch.sin(i_position * div_term)
        sin[:, :, 1::2] = torch.sin(j_position * div_term).half() if ropefp16 else torch.sin(j_position * div_term)
        cos[:, :, 0::2] = torch.cos(i_position * div_term).half() if ropefp16 else torch.cos(i_position * div_term)
        cos[:, :, 1::2] = torch.cos(j_position * div_term).half() if ropefp16 else torch.cos(j_position * div_term)

        sin = sin.repeat_interleave(2, dim=-1)
        cos = cos.repeat_interleave(2, dim=-1)
        # self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # [1, H, W, C]
        self.register_buffer('sin', sin.unsqueeze(0), persistent=False)  # [1, H, W, C//2]
        self.register_buffer('cos', cos.unsqueeze(0), persistent=False)  # [1, H, W, C//2]
        
        i_position4 = i_position.reshape(64,4,64,4,1)[...,0,:]
        i_position4 = i_position4.mean(-3)
        j_position4 = j_position.reshape(64,4,64,4,1)[:,0,...]
        j_position4 = j_position4.mean(-2)
        sin4 = torch.zeros(max_shape[0]//4, max_shape[1]//4, d_model//2, dtype=torch.float16 if ropefp16 else torch.float32)
        cos4 = torch.zeros(max_shape[0]//4, max_shape[1]//4, d_model//2, dtype=torch.float16 if ropefp16 else torch.float32)
        sin4[:, :, 0::2] = torch.sin(i_position4 * div_term).half() if ropefp16 else torch.sin(i_position4 * div_term)
        sin4[:, :, 1::2] = torch.sin(j_position4 * div_term).half() if ropefp16 else torch.sin(j_position4 * div_term)
        cos4[:, :, 0::2] = torch.cos(i_position4 * div_term).half() if ropefp16 else torch.cos(i_position4 * div_term)
        cos4[:, :, 1::2] = torch.cos(j_position4 * div_term).half() if ropefp16 else torch.cos(j_position4 * div_term)
        sin4 = sin4.repeat_interleave(2, dim=-1)
        cos4 = cos4.repeat_interleave(2, dim=-1)
        self.register_buffer('sin4', sin4.unsqueeze(0), persistent=False)  # [1, H, W, C//2]
        self.register_buffer('cos4', cos4.unsqueeze(0), persistent=False)  # [1, H, W, C//2]

        

    def forward(self, x, ratio=1):
        """
        Args:
            x: [N, H, W, C]
        """
        if ratio == 4:
            return (x * self.cos4[:, :x.size(1), :x.size(2), :]) + (self.rotate_half(x) * self.sin4[:, :x.size(1), :x.size(2), :])
        else:
            return (x * self.cos[:, :x.size(1), :x.size(2), :]) + (self.rotate_half(x) * self.sin[:, :x.size(1), :x.size(2), :])
    
    def rotate_half(self, x):
        x = x.unflatten(-1, (-1, 2))
        x1, x2 = x.unbind(dim=-1)
        return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)