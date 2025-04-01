import torch
import nibabel as nib
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import FractureMNIST3D

def get_dataloaders(batch_size=16, num_workers=4, prefetch_factor=2, pin_memory=True):
        pixel_sum, pixel_sq_sum, num_pixels = 0.0, 0.0, 0
        for images, _ in train_loader:
            pixel_sum += images.sum().item()
            pixel_sq_sum += (images**2).sum().item()
            num_pixels += images.numel()
                
        mean = pixel_sum / num_pixels
        std = (pixel_sq_sum / num_pixels - mean**2) ** 0.5
        print("Computed Mean:", mean)
        print("Computed Std:", std)

        transform = transforms.Compose([
            transforms.Lambda(lambda x: torch.tensor(x).float().unsqueeze(0)), 
            transforms.Normalize(mean=[mean], std=[std])
        ])
#batch_size reduziert, Prefetching & pinned memory: Improve data loading speed. 
        # --- Dataset Loading ---
        train_dataset = FractureMNIST3D(split='train', download=True, transform=transform, target_transform=lambda x: torch.tensor(x).squeeze().long())
#a validation set added (MedMNIST has a predefined val split). 
        val_dataset = FractureMNIST3D(split='val', download=True, transform=transform,target_transform=lambda x: torch.tensor(x).squeeze().long())
        test_dataset = FractureMNIST3D(split='test', download=True, transform=transform, target_transform=lambda x: torch.tensor(x).squeeze().long())
        
        # --- DataLoader Configuration ---
#kein 2*batch_size mehr, wegen memory issues 
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=prefetch_factor)
        train_loader_at_eval = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=prefetch_factor)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=prefetch_factor)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=prefetch_factor)

        data_iter = iter(train_loader)  # Convert DataLoader into an iterator
        images, labels = next(data_iter)
        print("Images shape:", images.shape)  # Expected: (batch_size, C, D, H, W) for 3D
        print("Labels shape:", labels.shape)  # Expected: (batch_size,)
        min_pixel_value = float('inf')
        max_pixel_value = float('-inf')
        total_sum = 0  # Sum of all pixel values
        total_count = 0  # Total number of pixels
        for images, _ in train_loader:
            batch_min = images.min().item()  # Min value in batch
            batch_max = images.max().item()  # Max value in batch
    
            min_pixel_value = min(min_pixel_value, batch_min)
            max_pixel_value = max(max_pixel_value, batch_max)

            # Compute sum and count for average calculation
            total_sum += images.sum().item()
            total_count += images.numel()

            # Compute the average pixel value
            average_pixel_value = total_sum / total_count
        
            # Print results
        print("Overall Min Pixel Value:", min_pixel_value)
        print("Overall Max Pixel Value:", max_pixel_value)
        print("Overall Average Pixel Value:", average_pixel_value)
        print("Sample labels:", labels[:5])  # First 5 labels



        # Add this to your dataset loading code
        sample = next(iter(train_loader))
        print("Input shape:", sample[0].shape)  # Should be [B, C, H, W]
        print("Target shape:", sample[1].shape)  # Should be [B]
        
        print(train_dataset)
        print("===================")
        print(test_dataset)
        print("===================")
        print(train_loader)

        train_dataset.montage(length=1)
        train_dataset.montage(length=10)

        return train_loader, train_loader_at_eval, val_loader, test_loader
