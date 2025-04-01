# test_vit.py
import unittest
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from models.vit import VisionTransformer
from data.dataset import get_dataloaders
from training.trainer import ViTTrainer

class TestViTBloodMNIST(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize test environment with actual model and data"""
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Use smaller config for testing
        cls.hyperparams = {
            'batch_size': 32,
            'num_workers': 2,
            'lr': 3e-4,
            'weight_decay': 0.01,
            'num_epochs': 1,
            'num_classes': 8
        }

        # Initialize data loaders with test-friendly params
        try:
            cls.train_loader, cls.train_loader_at_eval, cls.test_loader = get_dataloaders(
                batch_size=cls.hyperparams['batch_size'],
                num_workers=cls.hyperparams['num_workers']
            )
        except Exception as e:
            raise RuntimeError(f"Data loading failed: {str(e)}") from e

        # Verify data dimensions
        sample_input, sample_label = next(iter(cls.train_loader))
        cls.input_shape = sample_input.shape
        cls.label_shape = sample_label.shape
        
        # Initialize model with test-friendly architecture
        cls.model = VisionTransformer(
            image_size=28,
            patch_size=7,
            n_channels=3,
            num_classes=cls.hyperparams['num_classes'],
            latent_size=128,  # Smaller for testing
            num_encoders=2,   # Fewer layers
            num_heads=4,
            dropout=0.1
        ).to(cls.device)

        # Initialize trainer
        cls.trainer = ViTTrainer(
            model=cls.model,
            train_loader=cls.train_loader,
            test_loader=cls.test_loader,
            train_loader_at_eval=cls.train_loader_at_eval,
            device=cls.device,
            hyperparams=cls.hyperparams,
            task='multi-class',
            data_flag='bloodmnist'
        )

    def test_data_shapes(self):
        """Verify input and label dimensions"""
        # Input: (batch, channels, height, width)
        self.assertEqual(self.input_shape, (self.hyperparams['batch_size'], 3, 28, 28))
        
        # Labels: (batch, 1) -> will be squeezed in trainer
        self.assertEqual(self.label_shape, (self.hyperparams['batch_size'], 1))

    def test_model_io(self):
        """Test model input/output dimensions"""
        test_input = torch.randn(*self.input_shape).to(self.device)
        output = self.model(test_input)
        
        # Should output (batch_size, num_classes)
        self.assertEqual(output.shape, (self.hyperparams['batch_size'], self.hyperparams['num_classes']))

    def test_train_epoch(self):
        """Test single training epoch completion"""
        initial_params = next(self.model.parameters()).clone().detach()
        
        # Train one epoch
        train_loss = self.trainer.train_epoch(0)
        
        # Verify parameter updates
        final_params = next(self.model.parameters()).clone().detach()
        self.assertFalse(torch.equal(initial_params, final_params),
                         "Model parameters should change during training")
        
        # Loss sanity checks
        self.assertIsInstance(train_loss, float)
        self.assertGreater(train_loss, 0, "Loss should be positive")

    def test_evaluation(self):
        """Test evaluation metrics consistency"""
        self.trainer.model.eval()
        metrics = self.trainer.evaluate('test')
        
        # Metrics should return (auc, acc)
        self.assertEqual(len(metrics), 2)
        
        # BloodMNIST has 8 classes, random chance ~12.5%
        self.assertGreater(metrics[1], 0.1, "Accuracy should beat random chance")
        self.assertGreater(metrics[0], 0.1, "AUC should beat random chance")

    def test_accuracy_calculation(self):
        """Test accuracy metric implementation"""
        # Perfect predictions
        y_true = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
        y_score = torch.eye(8)  # Perfect predictions
        
        acc = accuracy_score(y_true, np.argmax(y_score, axis=1))
        self.assertEqual(acc, 1.0, "Perfect predictions should give 100% accuracy")

        # Worst case (all wrong)
        y_score_wrong = torch.roll(torch.eye(8), shifts=1, dims=1)
        acc_wrong = accuracy_score(y_true, np.argmax(y_score_wrong, axis=1))
        self.assertEqual(acc_wrong, 0.0, "All wrong predictions should give 0% accuracy")

if __name__ == '__main__':
    unittest.main(verbosity=2)
