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
        'num_epochs': 3,
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
        model = GCViT(dim = 56, #embedding dimension
                 depths = (2,2,2,2), #tuple of ints, number of transformer blocks at each level
                 mlp_ratio = 2, #multiplier for dim of mlp hidden layers
                 num_heads = (4,4,4,4), #tuple of ints, number of attention heads in each level
                 num_classes = 8,
                 window_size=(14, 14, 14, 14), #window size at each level, same length as depths
                 window_size_pre=(7, 7, 14, 7), #window size for preprocessing
                 resolution=28,
                 drop_path_rate=0.2,
                 in_chans=3,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm,
                 layer_scale=None,
                 out_indices=(0,1,2,3),
                 frozen_stages=-1,
                 pretrained=None,
                 use_rel_pos_bias=True)
        
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