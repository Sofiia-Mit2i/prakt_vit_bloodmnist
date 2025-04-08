import torch
import torch.nn as nn
from torch.nn.functional import interpolate

from .units.SE import SE 
from .units.permutation import _to_channel_last


class FeatExtract(nn.Module):
    """
    Feature extraction block based on: "Hatamizadeh et al.,                         Global Token Generator
    Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
    """

    def __init__(self, dim, keep_dim=False):
        """
        Args:
            dim: feature size dimension.
            keep_dim: bool argument for maintaining the resolution.
        """

        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1,
                      groups=dim, bias=False),
            nn.GELU(),
            SE(dim, dim),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
        )
        if not keep_dim:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.keep_dim = keep_dim

    def forward(self, x):
        x = x.contiguous()
        x = x + self.conv(x)
        if not self.keep_dim:
            x = self.pool(x)
        return x

class GlobalQueryGen(nn.Module):
    """
    Global query generator based on: "Hatamizadeh et al.,
    Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 image_resolution,
                 window_size,
                 num_heads):
        """
        Args:
            dim: feature size dimension.
            input_resolution: input image resolution.
            window_size: window size.
            num_heads: number of heads.

        For instance, repeating log(56/7) = 3 blocks, with input window dimension 56 and output window dimension 7 at
        down-sampling ratio 2. Please check Fig.5 of GC ViT paper for details.
        """

        super().__init__()
        if input_resolution == image_resolution//4:
            self.to_q_global = nn.Sequential(
                FeatExtract(dim, keep_dim=False),
                FeatExtract(dim, keep_dim=False),
                FeatExtract(dim, keep_dim=False),
            )

        elif input_resolution == image_resolution//8:
            self.to_q_global = nn.Sequential(
                FeatExtract(dim, keep_dim=False),
                FeatExtract(dim, keep_dim=False),
            )

        elif input_resolution == image_resolution//16:

            if window_size == input_resolution:
                self.to_q_global = nn.Sequential(
                    FeatExtract(dim, keep_dim=True)
                )

            else:
                self.to_q_global = nn.Sequential(
                    FeatExtract(dim, keep_dim=False)
                )

        elif input_resolution == image_resolution//32:
            self.to_q_global = nn.Sequential(
                FeatExtract(dim, keep_dim=True)
            )

        self.num_heads = num_heads
        self.N = window_size * window_size
        self.dim_head = dim // self.num_heads
        self.window_size = window_size

    def forward(self, x):
        x = self.to_q_global(x)
        B, C, H, W = x.shape
        if self.window_size != H or self.window_size !=W:
            x = interpolate(x, size=(self.window_size, self.window_size), mode='bicubic')
        x = _to_channel_last(x)
        x = x.reshape(B, 1, self.N, self.num_heads, self.dim_head).permute(0, 1, 3, 2, 4)
        return x