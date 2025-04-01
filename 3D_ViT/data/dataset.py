import torch
import nibabel as nib
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import FractureMNIST3D

def compute_normalization(dataset):
    """Compute mean and standard deviation for dataset normalization."""
    pixel_sum, pixel_sq_sum, num_pixels = 0.0, 0.0, 0

    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
    
    for images, _ in loader:
        pixel_sum += images.sum().item()
        pixel_sq_sum += (images**2).sum().item()
        num_pixels += images.numel()

    mean = pixel_sum / num_pixels
    std = (pixel_sq_sum / num_pixels - mean**2) ** 0.5
    print(f"Computed Mean: {mean}, Computed Std: {std}")
    
    return mean, std

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import FractureMNIST3D

def compute_normalization(dataset):
    """Compute mean and standard deviation for dataset normalization."""
    pixel_sum, pixel_sq_sum, num_pixels = 0.0, 0.0, 0

    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
    
    for images, _ in loader:
        pixel_sum += images.sum().item()
        pixel_sq_sum += (images**2).sum().item()
        num_pixels += images.numel()

    mean = pixel_sum / num_pixels
    std = (pixel_sq_sum / num_pixels - mean**2) ** 0.5
    print(f"Computed Mean: {mean}, Computed Std: {std}")
    
    return mean, std

def get_dataloaders(batch_size=16, num_workers=4, prefetch_factor=2, pin_memory=True):
    """Loads the FractureMNIST3D dataset and applies computed normalization."""
    
    # --- Load datasets first (without transform) to compute mean & std ---
    train_dataset_raw = FractureMNIST3D(split='train', download=True)
    
    # Compute normalization stats
    mean, std = compute_normalization(train_dataset_raw)

    # --- Define transforms with computed mean & std ---
    transform = transforms.Compose([
        transforms.Lambda(lambda x: torch.tensor(x).float().unsqueeze(0)),  # Convert to tensor and add channel dim
        transforms.Normalize(mean=[mean], std=[std])  # Normalize using computed values
    ])

    # --- Load datasets again with transforms ---
    train_dataset = FractureMNIST3D(split='train', download=True, transform=transform)
    val_dataset = FractureMNIST3D(split='val', download=True, transform=transform)
    test_dataset = FractureMNIST3D(split='test', download=True, transform=transform)

    # --- Create DataLoaders ---
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=prefetch_factor)
    train_loader_at_eval = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=prefetch_factor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=prefetch_factor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=prefetch_factor)

    # --- Print shape and stats ---
    images, labels = next(iter(train_loader))
    print(f"Images shape: {images.shape}, Labels shape: {labels.shape}")
    print(f"Sample labels: {labels[:5]}")  # First 5 labels

    return train_loader, train_loader_at_eval, val_loader, test_loader
