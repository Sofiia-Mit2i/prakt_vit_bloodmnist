import torch
from data.dataset import get_dataloaders

def main():
    # Set the batch size
    batch_size = 16

    # Get data loaders
    train_loader, train_loader_at_eval, val_loader, test_loader = get_dataloaders(batch_size=batch_size)

    # Check the shape of a sample batch from the train_loader
    sample = next(iter(train_loader))
    images, targets = sample

    # Print the shapes to verify the data is being loaded and transformed correctly
    print("Input shape (should be [B, C, H, W]):", images.shape)
    print("Target shape (should be [B]):", targets.shape)

    # Check some values for transformation correctness
    print("Sample values of the first image batch (normalized):", images[0].squeeze().numpy())
    print("Sample target values:", targets[:20].numpy())  # First 5 targets

    # Optionally, visualize a few images (if you have visualization libraries like matplotlib installed)
#import matplotlib.pyplot as plt
 #   grid_size = 4  # A small grid to visualize the first few images
  #  fig, axes = plt.subplots(1, grid_size, figsize=(12, 3))
   # for i in range(grid_size):
    #    axes[i].imshow(images[i].squeeze().numpy(), cmap='gray')
     #   axes[i].axis('off')
    #plt.show()

if __name__ == "__main__":
    main()
