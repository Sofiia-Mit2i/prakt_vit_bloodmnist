import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import FractureMNIST3D

class ToTensor4D:
    def __call__(self, pic):
        if pic.ndim == 4:
            return torch.tensor(pic, dtype=torch.float32)
        else:
            raise ValueError(f"pic should be 4 dimensional. Got {pic.ndim} dimensions.")


dataset = FractureMNIST3D(split='train',  transform=ToTensor4D(), download=True)
# Assuming `dataset` is your FractureMNIST3D dataset
# Convert all images to a single tensor for global statistics
all_images = torch.cat([img.unsqueeze(0) for img, _ in dataset], dim=0)  # Shape: (N, D, H, W)

# Compute statistics
min_val = all_images.min().item()
max_val = all_images.max().item()
mean_val = all_images.mean().item()
std_val = all_images.std().item()
median_val = all_images.median().item()
p25 = torch.quantile(all_images, 0.25).item()
p75 = torch.quantile(all_images, 0.75).item()

# Print stats
print(f"Min: {min_val}, Max: {max_val}")
print(f"Mean: {mean_val}, Std: {std_val}")
print(f"Median: {median_val}, 25th Percentile: {p25}, 75th Percentile: {p75}")

# Histogram
plt.hist(all_images.flatten().numpy(), bins=50, alpha=0.7, color='b')
plt.title("Pixel Intensity Distribution")
plt.xlabel("Intensity Value")
plt.ylabel("Frequency")
plt.show()
