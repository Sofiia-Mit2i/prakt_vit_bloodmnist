import torch
from torch import nn

class EncoderBlock(nn.Module):
    def __init__(self, 
                 latent_size=256,    # Adjusted to match InputEmbedding
                 num_heads=8,       # Number of attention heads
                 dropout=0.1):      # Dropout probability
        super().__init__()
        
        # 1. Pre-Attention Layer Normalization
        self.norm1 = nn.LayerNorm(latent_size)
        
        # 2. Multi-Head Attention Mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=latent_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # Input format: (batch, seq, features)
        )
        
        # 3. Pre-MLP Layer Normalization
        self.norm2 = nn.LayerNorm(latent_size)
        
        # 4. Feed-Forward Network (MLP), Two linear layers with 4x dimension expansion
        self.mlp = nn.Sequential(
            nn.Linear(latent_size, latent_size * 4),  # Expand dimension
            nn.GELU(),                               # Activation
            nn.Dropout(dropout),                     # Regularization
            nn.Linear(latent_size * 4, latent_size), # Compress dimension
            nn.Dropout(dropout)                     # Final regularization
        )

    def forward(self, x):
        """
        Processing steps:
        1. Attention residual connection
        2. MLP residual connection
        
        Input shape: (batch_size, num_patches + 1, latent_size)
        Output shape: (batch_size, num_patches + 1, latent_size)
        """
        # --- Attention Sub-block ---
        x_norm = self.norm1(x)  # Pre-LN
        attn_output, _ = self.attention(x_norm, x_norm, x_norm)
        x = x + attn_output  # First residual connection

        # --- MLP Sub-block ---
        x_norm = self.norm2(x)  # Pre-LN
        ff_output = self.mlp(x_norm)
        x = x + ff_output  # Second residual connection

        return x
