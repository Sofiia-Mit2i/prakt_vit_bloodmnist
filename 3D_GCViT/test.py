import torch
from torch import nn
from model.embedding import PatchEmbed
from model.layer import GCViTLayer
from model.gcvit import GCViT  # Assuming the model is in 'gcvit.py'

# Set some hyperparameters for testing
'''
dim = 64  # Embedding dimension
depths = [2, 2, 2, 2]  # Number of layers per level
mlp_ratio = 4  # MLP ratio
num_heads = [4, 8, 16, 32]  # Number of attention heads for each level
window_size = (24, 24, 24, 24)  # Assuming 4D window size for 3D data
window_size_pre = (7, 7, 14, 7)  # Predefined window sizes for the attention
resolution = 224  # Image resolution
drop_path_rate = 0.2  # Drop path rate
in_chans = 3  # Number of input channels (for RGB images)
out_indices = (0, 1, 2, 3)  # Output indices for each stage
use_rel_pos_bias = True  # Using relative position bias
'''

dim = 56    
depths = (2,2,2,2)
mlp_ratio = 2
num_heads = (4,4,4,4) 
window_size=(14, 14, 14, 14)
window_size_pre=(7, 7, 14, 7)
resolution=28
drop_path_rate=0.2
in_chans=3
qkv_bias=True
qk_scale=None
drop_rate=0.
attn_drop_rate=0.
norm_layer=nn.LayerNorm
layer_scale=None
out_indices=(0, 1, 2, 3)
frozen_stages=-1
pretrained=None
use_rel_pos_bias=True


# Create a dummy input tensor (e.g., batch size of 1, 3 channels, 224x224 resolution)
x = torch.randn(1, in_chans, resolution, resolution)

# Initialize the model
model = GCViT(
    dim=dim,
    depths=depths,
    mlp_ratio=mlp_ratio,
    num_heads=num_heads,
    window_size=window_size,
    window_size_pre=window_size_pre,
    resolution=resolution,
    drop_path_rate=drop_path_rate,
    in_chans=in_chans,
    out_indices=out_indices,
    use_rel_pos_bias=use_rel_pos_bias
)

# Print the model summary
print(model)

# Forward pass
output = model(x)

# Check the output shape
print("Output shapes for each level:")
for i, out in enumerate(output):
    print(f"Level {i} output shape: {out.shape}")
