<<<<<<< HEAD
import torch
import nibabel as nib
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import FractureMNIST3D

#def compute_normalization(dataset):
    #Compute mean and standard deviation for dataset normalization.
    #pixel_sum, pixel_sq_sum, num_pixels = 0.0, 0.0, 0

    #loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
    
    #for images, _ in loader:
    #    pixel_sum += images.sum().item()
    #    pixel_sq_sum += (images**2).sum().item()
    #    num_pixels += images.numel()

    #mean = pixel_sum / num_pixels
    #std = (pixel_sq_sum / num_pixels - mean**2) ** 0.5
    #print(f"Computed Mean: {mean}, Computed Std: {std}")
    
    #return mean, std
class ToTensor4D:
    def __call__(self, pic):
        if pic.ndim == 4:
            return torch.tensor(pic, dtype=torch.float32)
        else:
            raise ValueError(f"pic should be 4 dimensional. Got {pic.ndim} dimensions.")
            
def compute_normalization(dataset):
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
    all_images = torch.cat([torch.tensor(img).unsqueeze(0) for img, _ in dataset], dim=0)  # Shape: (N, D, H, W)

    # Compute statistics
    min_val = all_images.min().item()
    max_val = all_images.max().item()
    mean_val = all_images.mean().item()
    std_val = all_images.std().item()
    median_val = all_images.median().item()
    p25 = torch.quantile(all_images[:500], 0.25).item()  # Using a subset of the data
    p75 = torch.quantile(all_images[:500], 0.75).item()
    
    # Print stats
    print(f"Min: {min_val}, Max: {max_val}")
    print(f"Mean: {mean_val}, Std: {std_val}")
    print(f"Median: {median_val}, 25th Percentile: {p25}, 75th Percentile: {p75}")

    return mean_val, std_val

def get_dataloaders(batch_size=16, num_workers=4, prefetch_factor=2, pin_memory=True):
    """Loads the FractureMNIST3D dataset and applies computed normalization."""
    
    # --- Load datasets first (without transform) to compute mean & std ---
    train_dataset_raw = FractureMNIST3D(split='train', download=True)
    
    # Compute normalization stats
    mean_val, std_val = compute_normalization(train_dataset_raw)

    # --- Define transforms with computed mean & std ---
    #transform = transforms.Compose([
     #   ToTensor4D(), #Converts to tensor with ToTensor4D() (shape becomes [D, H, W, C])
      #  transforms.Lambda(lambda x: x.permute(3, 0, 1, 2)), #Uses permute to change to PyTorch's (C, D, H, W) format
       # transforms.Normalize(mean=[mean_val], std=[std_val])  # Normalize using computed values
    #])

    # --- Load datasets again with transforms ---
    train_dataset = FractureMNIST3D(split='train', download=True, transform=ToTensor4D())
    val_dataset = FractureMNIST3D(split='val', download=True,  transform=ToTensor4D())
    test_dataset = FractureMNIST3D(split='test', download=True, transform=ToTensor4D())

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
=======
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import FractureMNIST3D
import nibabel as nib

"""
Creates and returns DataLoaders for the BloodMNIST dataset.
- batch_size (int): Number of samples per training batch. Default: 32
- num_workers (int): Number of subprocesses for data loading. Default: 2
"""
def get_dataloaders(batch_size=16, num_workers=4):
        transform = transforms.Compose([
            transforms.ToTensor(),  
            transforms.Normalize(mean=[0.5], std=[0.5])  
        ])
        
        # --- Dataset Loading ---
        train_dataset = FractureMNIST3D(
        split='train', 
        download=True, 
        transform=transform,
#npz image
#
        target_transform=lambda x: torch.tensor(x).squeeze().long()
    )
        test_dataset = FractureMNIST3D(
        split='test', 
        download=True, 
        transform=transform,
        target_transform=lambda x: torch.tensor(x).squeeze().long()
    )
        
        
        
        # --- DataLoader Configuration ---
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        train_loader_at_eval = DataLoader(dataset=train_dataset, batch_size=2*batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(dataset=test_dataset, batch_size=2*batch_size, shuffle=False, num_workers=num_workers)

        # Add this to your dataset loading code
        sample = next(iter(train_loader))
        print("Input shape:", sample[0].shape)  # Should be [B, C, D, H, W]
        print("Target shape:", sample[1].shape)  # Should be [B]
        
        print(train_dataset)
        print("===================")
        print(test_dataset)
        print("===================")
        print(train_loader)

        train_dataset.montage(length=1)
        train_dataset.montage(length=10)

        return train_loader, train_loader_at_eval, test_loader

print(get_dataloaders())
>>>>>>> 295eab9 (erste Anpassungen für 3D)
