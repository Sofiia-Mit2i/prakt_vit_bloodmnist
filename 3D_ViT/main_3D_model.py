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
from sklearn.metrics import roc_auc_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("vit_training.log"),
        logging.StreamHandler()
    ]
)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Configuration
    hyperparams = {
        'batch_size': 12,
        'num_workers': 4,
        'lr': 3e-4,
        'weight_decay': 0.01,
        'num_epochs': 15,
        'num_classes': 3  # FractureMNIST3D has 3 classes
    }
    
    try:
        # Data loading
        logging.info("Initializing data pipelines...")
        train_loader, train_loader_at_eval, val_loader, test_loader = get_dataloaders(
            batch_size=hyperparams['batch_size'],
            num_workers=hyperparams['num_workers']
        )
        
        # Model initialization
        logging.info("Building Vision Transformer...")
        model = VisionTransformer(
            image_size=28,
            patch_size=4,
            n_channels=1,
            num_classes=hyperparams['num_classes'],
            latent_size=256,
            num_encoders=6,
            num_heads=8,
            dropout=0.1
        )
        
        # Initialize trainer
        trainer = ViTTrainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            val_loader=val_loader,
            train_loader_at_eval=train_loader_at_eval,
            device=device,
            hyperparams=hyperparams,
            task='multi-class',
            data_flag='fracturemnist3d'
        )
        
        # Start training
        logging.info("Beginning training process...")
        trainer.train()
        
    except KeyboardInterrupt:
        logging.warning("Training interrupted by user")
    except Exception as e:
        logging.error(f"Main execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    logging.info("Starting BloodMNIST ViT Training Pipeline")
    main()
    logging.info("Training process completed successfully")
