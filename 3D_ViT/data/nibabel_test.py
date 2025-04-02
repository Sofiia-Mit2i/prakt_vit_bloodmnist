import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import FractureMNIST3D, INFO
import nibabel as nib


# Load dataset
train_dataset = FractureMNIST3D(split='train', download=True)

# Check the shape of a sample image
sample_image, sample_label = train_dataset[0]  # Get the first image and its label

# Print the shape of the image
print(f"Shape of the image: {sample_image.shape}")
print(f"Shape of the label: {sample_label.shape}")

print(INFO['fracturemnist3d'])  # Example for FractureMNIST3D
#npz image

print(type(sample_image))
#numpy.ndarray