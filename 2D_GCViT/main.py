# main.py
import torch
import logging
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from medmnist import BloodMNIST
from model.gcvit import GCViT
from data.dataset import get_dataloaders
from training.trainer import GCViTTrainer
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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Configuration
    hyperparams = {
        'batch_size': 128,
        'num_workers': 4,
        'lr': 3e-4,
        'weight_decay': 0.01,
        'num_epochs': 5,
        'num_classes': 8  # BloodMNIST has 8 classes
    }
    
    try:
        # Data loading
        logging.info("Initializing data pipelines...")
        train_loader, train_loader_at_eval, test_loader = get_dataloaders(
            batch_size=hyperparams['batch_size'],
            num_workers=hyperparams['num_workers']
        )
        
        # Model initialization
        logging.info("Building Global Context Vision Transformer...")
        model = GCViT(dim = 64,
                 depths = (1,1),
                 mlp_ratio = 4,
                 num_heads = (2,2),
                 out_indices=(0,1))
        
        # Initialize trainer
        trainer = GCViTTrainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            train_loader_at_eval=train_loader_at_eval,
            device=device,
            hyperparams=hyperparams,
            task='multi-class',
            data_flag='bloodmnist'
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
    logging.info("Starting BloodMNIST GCViT Training Pipeline")
    main()
    logging.info("Training process completed successfully")