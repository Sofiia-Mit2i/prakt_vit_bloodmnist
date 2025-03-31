# main.py
import torch
import logging
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from medmnist import BloodMNIST
from encoder.input_embedding import InputEmbedding
from encoder.encoder_block import EncoderBlock
from models.vit import VisionTransformer
from data.dataset import get_dataloaders
from training.trainer import ViTTrainer
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("vit_training.log"),
        logging.StreamHandler()
    ]
)

