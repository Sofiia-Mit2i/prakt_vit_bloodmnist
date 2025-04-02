import torch
from data.dataset import get_dataloaders
from encoder.input_embedding import InputEmbedding

def main():
    # Set the batch size
    batch_size = 16

    # Get data loaders
    train_loader, train_loader_at_eval, val_loader, test_loader = get_dataloaders(batch_size=batch_size)

    # Check the shape of a sample batch from the train_loader
    sample = next(iter(train_loader))
    images, targets = sample

    # Print the shapes to verify the data is being loaded and transformed correctly
    print("\n=== Data Verification ===")
    print("Input shape (should be [B, C, H, W]):", images.shape)
    print("Target shape (should be [B]):", targets.shape)
    # Print the shapes to verify the data is being loaded and transformed correctly

   
    print("\n=== Data Values ===")
    print("Sample values of the first image batch (normalized):", images[0].squeeze().numpy())
    print("Min pixel value:", images.min().item())
    print("Max pixel value:", images.max().item())
    print("Mean pixel value:", images.mean().item())
    print("Sample target values:", targets[:20].numpy())  # First 20 targets
     # Check some values for transformation correctness

    print("\n=== Testing Input Embedding ===")
    embedder = InputEmbedding(
        image_size=(28, 28, 28),  # FractureMNIST3D dimensions
        patch_size=(7, 7, 7),
        n_channels=1,
        latent_size=256
    )
    embeddings = embedder(images)
    # Verify output dimensions
    print("\nEmbedding output shape (should be [B, num_patches+1, latent_size]):", embeddings.shape)

    # Check embedding statistics
    print("\nEmbedding values analysis:")
    print(f"Min: {embeddings.min().item():.4f}  Max: {embeddings.max().item():.4f}")
    print(f"Mean: {embeddings.mean().item():.4f}  Std: {embeddings.std().item():.4f}")

    # Verify class token presence
    class_token = embeddings[:, 0, :]
    patch_tokens = embeddings[:, 1:, :]
    print("\nClass token stats:")
    print(f"Mean: {class_token.mean().item():.4f}  Std: {class_token.std().item():.4f}")

    # Verify positional embeddings are being applied
    print("\nFirst 5 positional embeddings (first element):")
    print(embedder.pos_embedding[0, :5, 0].detach().numpy())

    # Check for NaN/inf values
    assert not torch.isnan(embeddings).any(), "Embeddings contain NaN values!"
    assert not torch.isinf(embeddings).any(), "Embeddings contain inf values!"
    print("\nEmbedding sanity checks passed!")

    print("\n=== Testing Encoder Block ===")
    
    # Initialize encoder block
    encoder = EncoderBlock(
        latent_size=256,
        num_heads=8,
        dropout=0.1
    )
    
    # Process embeddings through encoder
    encoded_output = encoder(embeddings)
    
    # Verify output dimensions
    print("\nEncoder output shape (should match input):", encoded_output.shape)
    
    # Check value ranges
    print("\nEncoder output analysis:")
    print(f"Min: {encoded_output.min().item():.4f}  Max: {encoded_output.max().item():.4f}")
    print(f"Mean: {encoded_output.mean().item():.4f}  Std: {encoded_output.std().item():.4f}")
    
    # Check residual connections
    diff = (embeddings - encoded_output).abs().mean()
    print(f"\nAbsolute difference between input and output: {diff.item():.4f} (should be >0 but not too large)")
    
    # Check attention mechanism
    print("\nAttention mechanism verification:")
    attn_output, attn_weights = encoder.attention(encoder.norm1(embeddings)), None
    print("Attention output shape:", attn_output[0].shape if isinstance(attn_output, tuple) else attn_output.shape)
    
    # Verify MLP operations
    mlp_output = encoder.mlp(encoder.norm2(embeddings))
    print("MLP output shape:", mlp_output.shape)
    
    # Final sanity checks
    assert not torch.isnan(encoded_output).any(), "Encoder output contains NaN!"
    assert not torch.isinf(encoded_output).any(), "Encoder output contains inf!"
    print("\nAll encoder checks passed!")

    # Optional: Test gradient flow
    dummy_loss = encoded_output.mean()
    dummy_loss.backward()
    print("\nGradient flow test completed without errors")

if __name__ == "__main__":
    main()
