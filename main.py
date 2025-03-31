# main.py
import torch
import logging
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from medmnist import BloodMNIST
from encoder/input_embedding import InputEmbedding
from encoder/encoder_block import EncoderBlock
from models/vit  import VisionTransformer
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

def get_dataloaders(batch_size=32, num_workers=2):
    """Modified dataset pipeline with error handling"""
    try:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # For 3 channels use [0.5]*3
        ])
        
        train_dataset = BloodMNIST(split='train', download=True, transform=transform)
        test_dataset = BloodMNIST(split='test', download=True, transform=transform)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=2*batch_size, 
            shuffle=False,
            num_workers=num_workers
        )
        
        return train_loader, test_loader
    
    except Exception as e:
        logging.error(f"Error in dataset pipeline: {str(e)}")
        raise

def train_model():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    hyperparams = {
        'batch_size': 64,
        'num_workers': 4,
        'lr': 3e-4,
        'weight_decay': 0.01,
        'num_epochs': 20,
        'num_classes': 8  # BloodMNIST has 8 blood cell types
    }
    
    try:
        # Data pipeline
        logging.info("Initializing data loaders...")
        train_loader, test_loader = get_dataloaders(
            batch_size=hyperparams['batch_size'],
            num_workers=hyperparams['num_workers']
        )
        
        # Model initialization
        logging.info("Building Vision Transformer...")
        model = VisionTransformer(
            image_size=28,
            patch_size=7,
            n_channels=3,  # BloodMNIST has RGB images
            num_classes=hyperparams['num_classes'],
            latent_size=256,
            num_encoders=6,
            num_heads=8,
            dropout=0.1
        ).to(device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=hyperparams['lr'],
            weight_decay=hyperparams['weight_decay']
        )
        
        # Training loop
        logging.info("Starting training...")
        best_acc = 0.0
        
        for epoch in range(hyperparams['num_epochs']):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            # Wrap train_loader in tqdm for progress bar
            train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{hyperparams['num_epochs']}", unit="batch")
            
            for images, labels in train_iter:
                images = images.to(device)
                labels = labels.squeeze().to(device)  # Remove extra dimension
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Statistics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Update progress bar
                train_iter.set_postfix({
                    'loss': loss.item(),
                    'acc': f"{100.*correct/total:.2f}%"
                })
            
            # Epoch statistics
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100. * correct / total
            logging.info(f"Epoch {epoch+1} - Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%")
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.squeeze().to(device)
                    
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            val_loss /= len(test_loader)
            val_acc = 100. * val_correct / val_total
            logging.info(f"Validation - Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), "best_vit_model.pth")
                logging.info(f"New best model saved with accuracy {best_acc:.2f}%")
        
        logging.info(f"Training complete. Best validation accuracy: {best_acc:.2f}%")
    
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        logging.info("Starting ViT training on BloodMNIST")
        train_model()
    except KeyboardInterrupt:
        logging.warning("Training interrupted by user")
    except Exception as e:
        logging.error(f"Critical error: {str(e)}")
