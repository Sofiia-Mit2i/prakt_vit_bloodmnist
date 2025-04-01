import torch
from torch import nn
import einops

class InputEmbedding(nn.Module):
    def __init__(self, 
                 image_size=28,        # Input image size (assumed square)
                 patch_size=7,         # Size of each image patch 
                 n_channels=3,         # Input channels (RGB)
                 latent_size=512,      # Embedding dimension
                 dropout=0.1):         # Dropout probability
        super().__init__()

        # Basic parameter validation
        assert image_size % patch_size == 0, "Image size must be divisible by patch size"

        # Calculate number of patches
        self.num_patches = (image_size // patch_size) ** 2
        print(num_patches)
        self.patch_size = patch_size
        print(patch_size)
        self.latent_size = latent_size
        print(latent_size)
        
        # 1. Patch Embedding Projection
        self.patch_embedding = nn.Linear(
            patch_size * patch_size * n_channels,
            latent_size
        )
        
        # 2. Learnable Class Token (global image representation)
        self.class_token = nn.Parameter(torch.randn(1, 1, latent_size))
                   
        # 3. Positional Embeddings (learnable position information)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, latent_size)
        )
        
        # 4. Dropout for regularization
        self.dropout = nn.Dropout(dropout)
                  
    def forward(self, x):
        """
        Input: 
            x - Tensor of shape (batch_size, channels, height, width)
        Output:
            embeddings - Tensor of shape (batch_size, num_patches+1, latent_size)
        """
        batch_size = x.shape[0]
        
        # 1. Split image into patches
        patches = einops.rearrange(
            x,
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=self.patch_size,
            p2=self.patch_size
        )
        
        # 2. Linear projection of patches
        embeddings = self.patch_embedding(patches)  # (b, n_patches, latent_size)
        
        # 3. Add class token to beginning of sequence
        class_tokens = self.class_token.expand(batch_size, -1, -1)
        embeddings = torch.cat([class_tokens, embeddings], dim=1)
        
        # 4. Add positional embeddings
        embeddings += self.pos_embedding
        
        # 5. Apply dropout
        return self.dropout(embeddings)
