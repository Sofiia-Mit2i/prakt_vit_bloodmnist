import logging
import torch
import numpy as np
from tqdm import tqdm
from torch import nn, optim
from sklearn.metrics import roc_auc_score

import medmnist
from medmnist import INFO, Evaluator


logger = logging.getLogger(__name__)

class ViTTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, train_loader_at_eval,
                 device, hyperparams, task='multi-class', data_flag='fracturemnist3d'):
        self.model = model.to(device)
        self.device = device
        self.hyperparams = hyperparams
        self.task = task
        self.data_flag = data_flag
        
        # Data loaders
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_loader_at_eval = train_loader_at_eval
        self.val_loader = val_loader
        
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
        # Learning rate scheduling based on validation AUC
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max', 
            factor=0.5, 
            patience=3
        )

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Gradient clipping
            self.optimizer.step()
            
            # Track statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
            
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        train_acc = correct / total
        return total_loss / len(self.train_loader), train_acc

    def validate(self, loader):
        """Validate the model on the validation set"""
        self.model.eval()
        total_loss = 0.0
        all_targets, all_probs = [], []

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device).squeeze().long()
                outputs = self.model(inputs)

                # Compute loss
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

                # Convert to probabilities
                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        all_probs = np.concatenate(all_probs)
        all_targets = np.concatenate(all_targets)

        # Calculate validation metrics
        val_acc = 100. * (all_probs.argmax(1) == all_targets).mean()
        val_auc = roc_auc_score(all_targets, all_probs, multi_class='ovo')

        logger.info(f"Validation - Loss: {avg_loss:.4f}, Accuracy: {val_acc:.2f}%, AUC: {val_auc:.4f}")
        return avg_loss, val_acc, val_auc

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
                train_loss, train_acc = self.train_epoch(epoch)
                logger.info(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                
                # Validate and save
                val_loss, val_acc, val_auc = self.validate()
                
                # Adjust learning rate based on validation AUC
                self.scheduler.step(val_auc)
                
            
            # Final evaluation
            logger.info("==> Final Evaluation <==")
            self.validate('train')
            self.validate('test')
            
            logger.info(f"Training complete. Best model - AUC: {self.best_metrics['auc']:.4f}, ACC: {self.best_metrics['acc']:.4f}")
        
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
