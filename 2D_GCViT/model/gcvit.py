import torch
import torch.nn as nn

from .embedding import PatchEmbed
from .layer import GCViTLayer


class GCViT(nn.Module):
    def __init__(self,
                 dim, #embedding dimension
                 depths, #tuple of ints, number of transformer blocks at each level
                 mlp_ratio,
                 num_heads,
                 window_size=(24, 24, 24, 24),
                 window_size_pre=(7, 7, 14, 7),
                 resolution=28,
                 drop_path_rate=0.2,
                 in_chans=3,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm,
                 layer_scale=None,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 pretrained=None,
                 use_rel_pos_bias=True,
                 **kwargs):
        super().__init__()

        self.num_levels = len(depths)
        self.embed_dim = dim
        self.num_features = [int(dim * 2 ** i) for i in range(self.num_levels)]
        self.mlp_ratio = mlp_ratio
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.patch_embed = PatchEmbed(in_chans=in_chans, dim=dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        for i in range(len(depths)):
            level = GCViTLayer(dim=int(dim * 2 ** i),
                               depth=depths[i],
                               num_heads=num_heads[i],
                               window_size=window_size[i],
                               window_size_pre=window_size_pre[i],
                               mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias,
                               qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                               norm_layer=norm_layer,
                               downsample=(i < len(depths) - 1),
                               layer_scale=layer_scale,
                               input_resolution=int(2 ** (-2 - i) * resolution),
                               image_resolution=resolution,
                               use_rel_pos_bias=use_rel_pos_bias)
            self.levels.append(level)

        # add a norm layer for each output
        self.out_indices = out_indices
        for i_layer in self.out_indices:
            layer = norm_layer(self.num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)


        for level in self.levels:
            for block in level.blocks:
                w_ = block.attn.window_size[0]
                relative_position_bias_table_pre = block.attn.relative_position_bias_table
                L1, nH1 = relative_position_bias_table_pre.shape
                L2 = (2 * w_ - 1) * (2 * w_ - 1)
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pre.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                    mode='bicubic')
                relative_position_bias_table_pretrained_resized = relative_position_bias_table_pretrained_resized.view(nH1, L2).permute(1, 0)
                block.attn.relative_position_bias_table = torch.nn.Parameter(relative_position_bias_table_pretrained_resized)

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x):
        outs = []
        for idx, level in enumerate(self.levels):
            x, xo = level(x)
            if idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(xo)
                outs.append(x_out.permute(0, 3, 1, 2).contiguous())
        return outs

    def forward(self, x):
        x = self.forward_embeddings(x)
        return self.forward_tokens(x)

    def forward_features(self, x):
        x = self.forward_embeddings(x)
        return self.forward_tokens(x)
