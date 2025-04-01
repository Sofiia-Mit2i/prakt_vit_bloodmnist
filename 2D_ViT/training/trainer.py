import logging
import torch
import numpy as np
from tqdm import tqdm
from torch import nn, optim

import medmnist
from medmnist import INFO, Evaluator


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
        y_true = []
        y_score = []
        
        data_loader = self.train_loader_at_eval if split == 'train' else self.test_loader
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                #print(inputs.shape)
                #print(inputs.size())
                outputs = self.model(inputs).cpu()
                #print(outputs.size())
                targets = targets.cpu().squeeze().long()
                #print(targets.size())
                
                
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
        
            # MedMNIST Evaluator
            evaluator = Evaluator(self.data_flag, split)
            metrics = evaluator.evaluate(y_score)

            # Extract metrics
            auc_score, acc_score = metrics.AUC, metrics.ACC
        
            logger.info(f"{split.upper()}  AUC: {auc_score:.3f}  ACC: {acc_score:.3f}")
            return auc_score, acc_score

    def save_checkpoint(self, metrics):
        """Save model checkpoint if AUC improves"""
        auc_score, acc_score = metrics
        if auc_score > self.best_metrics['auc']:
            self.best_metrics = {'auc': auc_score, 'acc': acc_score}
            torch.save(self.model.state_dict(), "best_model.pth")
            logger.info(f"New best model saved with AUC: {auc_score:.3f}, ACC: {acc_score:.3f}")

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
