import logging
import torch
import numpy as np
from tqdm import tqdm
from torch import nn, optim

logger = logging.getLogger(__name__)

class ViTTrainer:
    def __init__(self, model, train_loader, test_loader, train_loader_at_eval,
                 device, hyperparams, task='multi-class', data_flag='bloodmnist'):
        self.model = model.to(device)
        self.device = device
        self.hyperparams = hyperparams
        self.task = task
        self.data_flag = data_flag
        
        # Data loaders
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_loader_at_eval = train_loader_at_eval
        
        # Initialize metrics
        self.best_metrics = {'auc': 0.0, 'acc': 0.0}
        
        # Configure components
        self._init_components()

    def _init_components(self):
        """Initialize loss function and optimizer"""
        if self.task == 'multi-label, binary-class':
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
            
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.hyperparams['lr'],
            weight_decay=self.hyperparams['weight_decay']
        )

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.hyperparams['num_epochs']}", unit="batch")
        
        for inputs, targets in progress_bar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device).squeeze().long()
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update tracking
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        return total_loss / len(self.train_loader)

    def evaluate(self, split):
        """Evaluate model on specified split"""
        self.model.eval()
        y_true = torch.tensor([])
        y_score = torch.tensor([])
        
        data_loader = self.train_loader_at_eval if split == 'train' else self.test_loader
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs).cpu()
                targets = targets.cpu()
                
                # Process targets
                targets = targets.squeeze().long()
                
                # Process outputs based on task
                if self.task == 'multi-label, binary-class':  
                    outputs = torch.sigmoid(outputs)
                else:
                    outputs = torch.softmax(outputs, dim=-1)
                
                y_true.extend(targets.numpy().tolist())
                y_score.extend(outputs.numpy().tolist())
    
            # Convert to numpy arrays
            y_true = np.array(y_true)
            y_score = np.array(y_score)
        
            # Compute metrics
            if self.task == 'multi-label, binary-class':
                auc_score = roc_auc_score(y_true, y_score, average='macro')
            else:
                auc_score = roc_auc_score(y_true, y_score, multi_class='ovr', average='macro')
        
            acc_score = accuracy_score(y_true, np.argmax(y_score, axis=1))
        
            metrics = (auc_score, acc_score) 
        
            logger.info(f"{split.upper()}  AUC: {metrics[0]:.3f}  ACC: {metrics[1]:.3f}")
            return metrics

    def save_checkpoint(self, metrics):
        if metrics[0] > self.best_metrics['auc']:
            self.best_metrics = {'auc': metrics[0], 'acc': metrics[1]}
            torch.save(self.model.state_dict(), "best_model.pth")
            logger.info(f"New best model saved with AUC: {metrics[0]:.3f}, ACC: {metrics[1]:.3f}")

    def train(self):
        try:
            logger.info("Starting training...")
            for epoch in range(self.hyperparams['num_epochs']):
                # Train one epoch
                train_loss = self.train_epoch(epoch)
                logger.info(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}")
                
                # Validate and save
                val_metrics = self.evaluate('test')
                self.save_checkpoint(val_metrics)
            
            # Final evaluation
            logger.info("==> Final Evaluation <==")
            self.evaluate('train')
            self.evaluate('test')
            
            logger.info(f"Training complete. Best model: AUC {self.best_metrics['auc']:.3f}, ACC {self.best_metrics['acc']:.3f}")
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
