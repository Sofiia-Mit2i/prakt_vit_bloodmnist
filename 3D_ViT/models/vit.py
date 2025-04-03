
# vision_transformer.py
import torch
from torch import nn
from encoder.input_embedding import InputEmbedding
from encoder.encoder_block import EncoderBlock

class VisionTransformer(nn.Module):
    def __init__(self,
                 image_size=28,           # Input image size (height/width)
                 patch_size=3,            # Size of image patches
                 n_channels=1,            # Input color channels
                 num_classes=3,           # FractureMNIST3D has 3 classes
                 latent_size=256,         # Embedding dimension
                 num_encoders=6,          # Number of transformer blocks
                 num_heads=12,             # Attention heads per encoder
                 dropout=0.1):           # Dropout probability
        super().__init__()

        # Validate parameters
        assert image_size % patch_size == 0, "Image size must be divisible by patch size"
        assert latent_size % num_heads == 0, "Latent size must be divisible by num_heads"

        # 1. Input Embedding Module
        self.embedding = InputEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            n_channels=n_channels,
            latent_size=latent_size,
            dropout=dropout
        )

        # 2. Transformer Encoder Stack
        self.encoders = nn.ModuleList([
            EncoderBlock(
                latent_size=latent_size,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_encoders)
        ])

        # 3. Classification Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(latent_size),
            nn.Linear(latent_size, latent_size//2),  # Feature compression, Gradual dimension reduction (256 → 128 → 8)
            nn.GELU(),                              # Activation
            nn.Dropout(dropout),
            nn.Linear(latent_size//2, num_classes)  # Final class projection
        )

    def forward(self, x):
        """
        Input: 
            x - Tensor of shape (batch_size, channels, height, width)
        Output:
            logits - Class predictions (batch_size, num_classes)
        """
        # 1. Create patch embeddings
        x = self.embedding(x)  # (B, num_patches+1, latent_size)

        # 2. Process through encoder stack
        for encoder in self.encoders:
            x = encoder(x)

        # 3. Extract class token (first in sequence)
        cls_token = x[:, 0]  # (B, latent_size)

        # 4. Classify using final features
        return self.classifier(cls_token)
