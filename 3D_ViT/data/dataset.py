import torch
import nibabel as nib
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import FractureMNIST3D

def get_dataloaders(batch_size=16, num_workers=4, prefetch_factor=2, pin_memory=True):

        transform = transforms.Compose([
            lambda x: torch.tensor(x).float() / 255.0,  
            transforms.Normalize(mean=[0.5], std=[0.5])  
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
        print("Min pixel value:", images.min().item(), "Max pixel value:", images.max().item())
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
