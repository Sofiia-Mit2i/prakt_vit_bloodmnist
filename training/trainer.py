# training/trainer.py
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
        self.best_auc = 0.0
        self.best_acc = 0.0
        
        # Configure loss function
        if self.task == 'multi-label, binary-class':
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
            
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=hyperparams['lr'],
            weight_decay=hyperparams['weight_decay']
        )

    def _process_targets(self, outputs, targets):
        """Handle different task types and return proper loss"""
        if self.task == 'multi-label, binary-class':
            targets = targets.to(torch.float32)
            loss = self.criterion(outputs, targets)
            return loss, targets
        else:
            targets = targets.squeeze().long()
            loss = self.criterion(outputs, targets)
            return loss, targets

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_iter = tqdm(self.train_loader, 
                         desc=f"Epoch {epoch+1}/{self.hyperparams['num_epochs']}", 
                         unit="batch")
        
        for inputs, targets in train_iter:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Calculate loss
            loss, processed_targets = self._process_targets(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            
            # Calculate accuracy
            if self.task == 'multi-label, binary-class':
                preds = torch.sigmoid(outputs).round()
            else:
                preds = outputs.argmax(dim=1)
                
            train_correct += (preds == processed_targets).sum().item()
            train_total += processed_targets.size(0)
            
            # Update progress bar
            train_iter.set_postfix({
                'loss': loss.item(),
                'acc': f"{100.*train_correct/train_total:.2f}%"
            })
        
        return total_loss / len(self.train_loader), train_correct / train_total

    def evaluate(self, split='test'):
        """Evaluate model on specified split"""
        self.model.eval()
        y_true = []
        y_score = []
        
        data_loader = self.train_loader_at_eval if split == 'train' else self.test_loader
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                targets = targets.cpu()
                
                outputs = self.model(inputs).cpu()
                
                if self.task == 'multi-label, binary-class':
                    y_score.append(torch.sigmoid(outputs))
                    y_true.append(targets.float())
                else:
                    y_score.append(torch.softmax(outputs, dim=1))
                    y_true.append(targets.squeeze().long())
        
        y_true = torch.cat(y_true).numpy()
        y_score = torch.cat(y_score).numpy()
        
        # Get metrics from evaluator
        metrics = self._calculate_metrics(y_true, y_score, split)
        return metrics

    def _calculate_metrics(self, y_true, y_score, split):
        """Calculate metrics using your Evaluator class"""
        # Replace with your actual metrics calculation
        # Example implementation:
        from sklearn.metrics import accuracy_score, roc_auc_score
        
        if self.task == 'multi-label, binary-class':
            acc = accuracy_score(y_true, y_score.round())
            auc = roc_auc_score(y_true, y_score)
        else:
            acc = accuracy_score(y_true, np.argmax(y_score, axis=1))
            auc = roc_auc_score(y_true, y_score, multi_class='ovo')
            
        logger.info(f"{split.upper()}  AUC: {auc:.3f}  ACC: {acc:.3f}")
        return {'auc': auc, 'acc': acc}

    def save_checkpoint(self, metrics):
        if metrics['auc'] > self.best_auc:
            self.best_auc = metrics['auc']
            self.best_acc = metrics['acc']
            torch.save(self.model.state_dict(), "best_model.pth")
            logger.info(f"New best model saved with AUC: {self.best_auc:.3f}, ACC: {self.best_acc:.3f}")

    def train(self):
        try:
            logger.info("Starting training...")
            for epoch in range(self.hyperparams['num_epochs']):
                # Training phase
                train_loss, train_acc = self.train_epoch(epoch)
                logger.info(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f} | Acc: {train_acc:.3f}")
                
                # Evaluation phase
                train_metrics = self.evaluate('train')
                test_metrics = self.evaluate('test')
                
                # Save best model based on validation metrics
                self.save_checkpoint(test_metrics)
            
            logger.info(f"Training complete. Best model: AUC {self.best_auc:.3f}, ACC {self.best_acc:.3f}")
            return self.best_auc, self.best_acc
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
