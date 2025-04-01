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