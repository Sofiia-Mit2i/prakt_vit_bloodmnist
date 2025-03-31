import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import BloodMNIST

def get_dataloaders(BATCH_SIZE=32, num_workers=2):
"""
Creates and returns DataLoaders for the BloodMNIST dataset.
- batch_size (int): Number of samples per training batch. Default: 32
- num_workers (int): Number of subprocesses for data loading. Default: 2
"""
# preprocessing
        transform = transforms.Compose([
            # Convert PIL Image/numpy.ndarray to torch.Tensor and scale to [0, 1]
            transforms.ToTensor(),  
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
        # Normalize tensor values to range [-1, 1] using:
        # input[channel] = (input[channel] - mean[channel]) / std[channel]
        # For RGB images 3-channel normalization:
        ])
        
        # --- Dataset Loading ---
        train_dataset = BloodMNIST(split='train', download=True, transform=transform)
        test_dataset = BloodMNIST(split='test', download=True, transform=transform)
        
        
        
        # --- DataLoader Configuration ---
        train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
        train_loader_at_eval = DataLoader(dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False, num_workers=num_workers)

        return train_loader, train_loader_at_eval, test_loader
