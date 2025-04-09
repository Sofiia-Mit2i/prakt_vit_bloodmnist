import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath

from .attention import WindowAttentionGlobal, window_partition, window_reverse
from .mlp import Mlp


class GCViTBlock(nn.Module):
    """
    GCViT block based on: "Hatamizadeh et al.,
    Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size_pre,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 attention=WindowAttentionGlobal,
                 norm_layer=nn.LayerNorm,
                 layer_scale=None,
                 use_rel_pos_bias=False
                 ):
        """
        Args:
            dim: feature size dimension.
            input_resolution: input image resolution.
            num_heads: number of attention head.
            window_size: window size.
            mlp_ratio: MLP ratio.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: drop path rate.
            act_layer: activation function.
            attention: attention block type.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
        """

        super().__init__()
        self.window_size = window_size
        self.norm1 = norm_layer(dim)

        self.attn = attention(dim,
                              num_heads=num_heads,
                              window_size=window_size,
                              window_size_pre=window_size_pre,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              attn_drop=attn_drop,
                              proj_drop=drop,
                              use_rel_pos_bias=use_rel_pos_bias
                              )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
        else:
            self.gamma1 = 1.0
            self.gamma2 = 1.0

    def forward(self, x, q_global):
        B, H, W, D, C = x.shape
        shortcut = x
        x = self.norm1(x)
        pad_w_l = pad_h_t = pad_d_f = 0
        pad_w_r = (self.window_size - W % self.window_size) % self.window_size
        pad_h_b = (self.window_size - H % self.window_size) % self.window_size
        pad_d_b = (self.window_size - D % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_d_f, pad_d_b, pad_w_l, pad_w_r, pad_h_t, pad_h_b)) #back to front
        _, Hp, Wp, Dp, _ = x.shape
        shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size * self.window_size, C)
        _, h, w, d = x_windows.shape
        attn_windows = self.attn(x_windows, q_global)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp, Dp)  # B H' W' D' C
        x = shifted_x
        if pad_w_r > 0 or pad_h_b > 0 or pad_d_b > 0:
            x = x[:, :H, :W, :D, :].contiguous()
        x = shortcut + self.drop_path(self.gamma1 * x)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x