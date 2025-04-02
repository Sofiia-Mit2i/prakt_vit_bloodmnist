import torch
from torch import nn
import einops

class InputEmbedding(nn.Module):
    def __init__(self, 
                 image_size=(28, 28, 28),      # (Depth, Height, Width)
                 patch_size=(7, 7, 7),        # 3D Patch size
                 n_channels=1,         	          # Input channels (Grayscale for FractureMNIST3D)
                 latent_size=256,     
#Reduce model complexity: Using latent_size=256 (instead of 512) can help reduce computational overhead.
                 dropout=0.1):         # Dropout probability
        super().__init__()

        # Ensure divisibility
        assert all(i % p == 0 for i, p in zip(image_size, patch_size)), 
        #Image dimensions must be divisible by patch size

        # Calculate number of patches
        self.num_patches = (image_size[0] // patch_size[0]) * \
                           (image_size[1] // patch_size[1]) * \
                           (image_size[2] // patch_size[2])
        print(self.num_patches)
        self.patch_size = patch_size
        print(self.patch_size)
        self.latent_size = latent_size
        print(self.latent_size)
        
        # 1. Patch Embedding using 3D convolution (instead of Linear projection)
        self.patch_embedding = nn.Conv3d(
            in_channels=n_channels,
            out_channels=latent_size,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # 2. Learnable Class Token
        self.class_token = nn.Parameter(torch.randn(1, 1, latent_size))
                   
        # 3. Positional Embedding for 3D patches
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, latent_size))
        
        # 4. Dropout
        self.dropout = nn.Dropout(dropout)
                  
    def forward(self, x):
        """
        Input: 
            x - Tensor of shape (batch_size, channels, depth, height, width)
        Output:
            embeddings - Tensor of shape (batch_size, num_patches+1, latent_size)
        """
        batch_size = x.shape[0]
        
        # 1. Apply 3D Conv to get patch embeddings
        patches = self.patch_embedding(x)  # Shape: (batch_size, latent_size, d', h', w')
        # Flatten spatial dimensions
        #patches = patches.flatten(2).transpose(1, 2)  # (B, num_patches, latent_size)
        #Each patch is treated like a small chunk of the 3D image (like a small cube of the CT scan).
        #Best when treating patches as small units that hold spatial information (good for medical images).
        # 2. Flatten patches into sequence
        
        patches = einops.rearrange(patches, "b c d h w -> b (d h w) c")
        
        #Instead of thinking in patches, the model learns from each small voxel separately.
        #Good when ViT needs to process every voxel independently, without considering local patch structure
        
        # 3. Add class token
        class_tokens = self.class_token.expand(batch_size, -1, -1)
        embeddings = torch.cat([class_tokens, patches], dim=1)
        
        # 4. Add positional embeddings
        embeddings += self.pos_embedding
        
        # 5. Apply dropout
        return self.dropout(embeddings)
