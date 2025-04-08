
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import FractureMNIST3D


class ToTensor4D:
    def __call__(self, pic):
        if pic.ndim == 4:
<<<<<<< HEAD
            tensor = torch.tensor(pic, dtype=torch.float32)
            # Reorder to PyTorch format [C, D, H, W]
            return tensor.permute(3, 0, 1, 2)
=======
            return torch.tensor(pic, dtype=torch.float32)
>>>>>>> 16ed084873d61d210826188e3820ba450ee604dd
        else:
            raise ValueError(f"pic should be 4 dimensional. Got {pic.ndim} dimensions.")
            
#def compute_normalization(dataset):
#    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
#    all_images = torch.cat([torch.tensor(img).unsqueeze(0) for img, _ in dataset], dim=0)  # Shape: (N, D, H, W)
#
#    # Compute statistics
#    min_val = all_images.min().item()
#    max_val = all_images.max().item()
#    mean_val = all_images.mean().item()
#    std_val = all_images.std().item()
#    median_val = all_images.median().item()
#    p25 = torch.quantile(all_images[:500], 0.25).item()  # Using a subset of the data
#    p75 = torch.quantile(all_images[:500], 0.75).item()
    
    # Print stats
#    print(f"Min: {min_val}, Max: {max_val}")
#    print(f"Mean: {mean_val}, Std: {std_val}")
#    print(f"Median: {median_val}, 25th Percentile: {p25}, 75th Percentile: {p75}")

#    return mean_val, std_val
class MedicalNormalize:
    def __init__(self, mean, std, pct_range=(0.5, 99.5)):
        self.mean = mean
        self.std = std
        self.pct_range = pct_range

    def __call__(self, x):
        # 1. Percentile-based clipping
        pct_low = np.percentile(x.numpy(), self.pct_range[0])
        pct_high = np.percentile(x.numpy(), self.pct_range[1])
        x = torch.clamp(x, pct_low, pct_high)
        
        # 2. Z-score normalization
        x = (x - self.mean) / self.std
        
        # 3. Scale to [-1, 1] range
        x = 2 * (x - x.min()) / (x.max() - x.min()) - 1
        return x

def compute_robust_stats(dataset, num_samples=500):
    """Compute robust statistics using random samples"""
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    samples = []    

    for i in indices:
        img, _ = dataset[i]
        # Convert numpy array to tensor and permute dimensions
        tensor_img = torch.tensor(img, dtype=torch.float32).permute(3, 0, 1, 2)
        samples.append(tensor_img)
    
    samples = torch.stack(samples)

    return {
        'mean': samples.mean().item(),
        'std': samples.std().item(),
        'median': samples.median().item(),
        'min': samples.min().item(),
        'max': samples.max().item()
    }

def get_dataloaders(batch_size=16, num_workers=4, prefetch_factor=2, pin_memory=True):
    """Loads the FractureMNIST3D dataset and applies computed normalization."""
     # --- Load datasets first (without transform) to compute mean & std ---
    train_dataset_raw = FractureMNIST3D(split='train', download=True)
    stats = compute_robust_stats(train_dataset_raw)


    # Compute normalization stats
    #mean_val, std_val = compute_normalization(train_dataset_raw)

    # --- Define transforms with computed mean & std ---
    #transform = transforms.Compose([
    #    ToTensor4D(), #Converts to tensor with ToTensor4D() (shape becomes [D, H, W, C])
    #  #  transforms.Lambda(lambda x: x.permute(3, 0, 1, 2)), #Uses permute to change to PyTorch's (C, D, H, W) format
    #   transforms.Normalize(mean=[mean_val], std=[std_val])  # Normalize using computed values
    #])

    # Define transforms
    base_transform = transforms.Compose([
        ToTensor4D(),
        MedicalNormalize(mean=stats['mean'], std=stats['std'])
    ])

    train_transform = transforms.Compose([
        base_transform,
        transforms.RandomApply([
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5)
        ], p=0.7),
        transforms.RandomErasing(p=0.1)
    ])
    # --- Load datasets again with transforms ---
    train_dataset = FractureMNIST3D(split='train', download=True, transform=train_transform)
    val_dataset = FractureMNIST3D(split='val', download=True,  transform=eval_transform)
    test_dataset = FractureMNIST3D(split='test', download=True, transform=eval_transform)

    # Handle class imbalance
    train_labels = [label for _, label in train_dataset]
    class_counts = np.bincount(train_labels)
    weights = 1. / class_counts[train_labels]
    sampler = WeightedRandomSampler(weights, len(weights))

    # --- Create DataLoaders ---
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=prefetch_factor)
    #train_loader_at_eval = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=prefetch_factor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=prefetch_factor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=prefetch_factor)

    # Verify shapes
    sample_img, sample_label = next(iter(train_loader))
    print(f"Final image shape: {sample_img.shape} (should be [B, C, D, H, W])")
    print(f"Label shape: {sample_label.shape}")
    
    return train_loader, val_loader, test_loader