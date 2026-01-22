"""
Linear Transformer proposed in "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
Modified from: https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/linear_attention.py
"""

import torch
from torch.nn import Module, Dropout
import torch.nn.functional as F

# if hasattr(F, 'scaled_dot_product_attention'):
#     FLASH_AVAILABLE = True
# else: # v100
FLASH_AVAILABLE = False
# import xformers.ops
from ..utils.position_encoding import PositionEncodingSine, RoPEPositionEncodingSine
from einops.einops import rearrange
from loguru import logger


# flash_attn_func_ok = True
# try:
#     from flash_attn import flash_attn_func
# except ModuleNotFoundError:
#     flash_attn_func_ok = False


def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class LinearAttention(Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # set padded position to zero
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            values = values * kv_mask[:, :, None, None]

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        # queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()


class RoPELinearAttention(Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps
        self.RoPE = RoPEPositionEncodingSine(256, max_shape=(256, 256))

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    def forward(self, queries, keys, values, q_mask=None, kv_mask=None, H=None, W=None):
        """Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)
        nhead, d = Q.size(2), Q.size(3)
        # set padded position to zero
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            values = values * kv_mask[:, :, None, None]

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        # Q = Q / Q.size(1)
        # logger.info(f"Q: {Q.dtype}, K: {K.dtype}, values: {values.dtype}")

        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        # logger.info(f"Z_max: {Z.abs().max()}")
        Q = rearrange(Q, "n (h w) nhead d -> n h w (nhead d)", h=H, w=W)
        K = rearrange(K, "n (h w) nhead d -> n h w (nhead d)", h=H, w=W)
        Q, K = self.RoPE(Q), self.RoPE(K)
        # logger.info(f"Q_rope: {Q.abs().max()}, K_rope: {K.abs().max()}")
        Q = rearrange(Q, "n h w (nhead d) -> n (h w) nhead d", nhead=nhead, d=d)
        K = rearrange(K, "n h w (nhead d) -> n (h w) nhead d", nhead=nhead, d=d)
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        del K, values
        # logger.info(f"KV_max: {KV.abs().max()}")
        # queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length
        # Q = torch.einsum("nlhd,nlh->nlhd", Q, Z)
        # logger.info(f"QZ_max: {Q.abs().max()}")
        # queried_values = torch.einsum("nlhd,nhdv->nlhv", Q, KV) * v_length
        # logger.info(f"message_max: {queried_values.abs().max()}")
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()


class FullAttention(Module):
    def __init__(self, use_dropout=False, attention_dropout=0.1):
        super().__init__()
        self.use_dropout = use_dropout
        self.dropout = Dropout(attention_dropout)

    # @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """Multi-head scaled dot-product attention, a.k.a full attention.
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        # assert kv_mask is None
        # mask = torch.zeros(queries.size(0)*queries.size(2), queries.size(1), keys.size(1), device=queries.device)
        # mask.masked_fill(~(q_mask[:, :, None] * kv_mask[:, None, :]), float('-inf'))
        # if keys.size(1) % 8 != 0:
        #     mask = torch.cat([mask, torch.zeros(queries.size(0)*queries.size(2), queries.size(1), 8-keys.size(1)%8, device=queries.device)], dim=-1)
        # out = xformers.ops.memory_efficient_attention(queries, keys, values, attn_bias=mask[...,:keys.size(1)])
        # return out

        # N = queries.size(0)
        # list_q = [queries[i, :q_mask[i].sum, ...] for i in N]
        # list_k = [keys[i, :kv_mask[i].sum, ...] for i in N]
        # list_v = [values[i, :kv_mask[i].sum, ...] for i in N]
        # assert N == 1
        # out = xformers.ops.memory_efficient_attention(queries[:,:q_mask.sum(),...], keys[:,:kv_mask.sum(),...], values[:,:kv_mask.sum(),...])
        # out = torch.cat([out, torch.zeros(out.size(0), queries.size(1)-q_mask.sum(), queries.size(2), queries.size(3), device=queries.device)], dim=1)
        # return out
        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("nlhd,nshd->nlsh", queries, keys)
        if kv_mask is not None:
            QK.masked_fill_(
                ~(q_mask[:, :, None, None] * kv_mask[:, None, :, None]), -1e5
            )  # float('-inf')

        # Compute the attention and the weighted average
        softmax_temp = 1.0 / queries.size(3) ** 0.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=2)
        if self.use_dropout:
            A = self.dropout(A)

        queried_values = torch.einsum("nlsh,nshd->nlhd", A, values)

        return queried_values.contiguous()


class XAttention(Module):
    def __init__(self, use_dropout=False, attention_dropout=0.1):
        super().__init__()
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = Dropout(attention_dropout)

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """Multi-head scaled dot-product attention, a.k.a full attention.
        Args:
            if FLASH_AVAILABLE: # pytorch scaled_dot_product_attention
                queries: [N, H, L, D]
                keys: [N, H, S, D]
                values: [N, H, S, D]
            else:
                queries: [N, L, H, D]
                keys: [N, S, H, D]
                values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """

        assert q_mask is None and kv_mask is None, "already been sliced"
        if FLASH_AVAILABLE:
            # args = [x.half().contiguous() for x in [queries, keys, values]]
            # out = F.scaled_dot_product_attention(*args, attn_mask=mask).to(queries.dtype)
            args = [x.contiguous() for x in [queries, keys, values]]
            out = F.scaled_dot_product_attention(*args)
        else:
            # if flash_attn_func_ok:
            #     out = flash_attn_func(queries, keys, values)
            # else:
            QK = torch.einsum("nlhd,nshd->nlsh", queries, keys)

            # Compute the attention and the weighted average
            softmax_temp = 1.0 / queries.size(3) ** 0.5  # sqrt(D)
            A = torch.softmax(softmax_temp * QK, dim=2)

            out = torch.einsum("nlsh,nshd->nlhd", A, values)

            # out = xformers.ops.memory_efficient_attention(queries, keys, values)
        # out = xformers.ops.memory_efficient_attention(queries[:,:q_mask.sum(),...], keys[:,:kv_mask.sum(),...], values[:,:kv_mask.sum(),...])
        # out = torch.cat([out, torch.zeros(out.size(0), queries.size(1)-q_mask.sum(), queries.size(2), queries.size(3), device=queries.device)], dim=1)
        return out
