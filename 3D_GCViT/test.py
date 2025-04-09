import torch
from torch.utils.data import DataLoader
from medmnist import FractureMNIST3D
import numpy as np

def print_dataset_stats(dataset, name):
    print(f"\n===== {name} Dataset Statistics =====")
    print(f"Number of samples: {len(dataset)}")
    
    # Get first sample
    img, label = dataset[0]
    
    # Print shapes and types
    print(f"\nImage shape: {img.shape} (C×D×H×W)")
    print(f"Label shape: {label.shape}")
    print(f"Image dtype: {img.dtype}")
    print(f"Label dtype: {label.dtype}")
    
    # Print value ranges
    print(f"\nImage value range: {img.min().item():.4f} to {img.max().item():.4f}")
    print(f"Unique labels: {torch.unique(torch.tensor([dataset[i][1] for i in range(len(dataset))]))}")
    
    # Print volumetric stats
    if len(img.shape) == 4:  # 3D data
        print("\nVolumetric statistics:")
        print(f"Depth (slices): {img.shape[1]}")
        print(f"Height: {img.shape[2]}")
        print(f"Width: {img.shape[3]}")
        print(f"Voxel count: {np.prod(img.shape)}")

def main():
    # Initialize dataset
    train_dataset = FractureMNIST3D(split='train', download=True)
    test_dataset = FractureMNIST3D(split='test', download=True)
    
    # Print dataset stats
    print_dataset_stats(train_dataset, "Training")
    print_dataset_stats(test_dataset, "Test")
    
    # Create DataLoader and check batch sizes
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4)
    
    # Check first batch
    train_batch = next(iter(train_loader))
    print("\n===== DataLoader Batch Check =====")
    print(f"Train batch images shape: {train_batch[0].shape} (B×C×D×H×W)")
    print(f"Train batch labels shape: {train_batch[1].shape}")
    print(f"Batch dtype: {train_batch[0].dtype}")
    
    # Verify normalization
    print("\nBatch value range:")
    print(f"Min: {train_batch[0].min().item():.4f}")
    print(f"Max: {train_batch[0].max().item():.4f}")
    print(f"Mean: {train_batch[0].mean().item():.4f}")
    print(f"Std: {train_batch[0].std().item():.4f}")

if __name__ == "__main__":
    print("====== FractureMNIST3D Data Inspection ======")
    main()