# hyperparam_search.py
import optuna
import torch
import torch.nn as nn
from model.gcvit import GCViT
from data.dataset import get_dataloaders
from training.trainer import GCViTTrainer
import logging

logging.basicConfig(level=logging.INFO)

def objective(trial):
    # Sample hyperparameters
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
    drop_path_rate = trial.suggest_uniform('drop_path_rate', 0.0, 0.3)
    mlp_ratio = trial.suggest_categorical('mlp_ratio', [2, 3, 4])
    
    num_epochs = 5  # Keep short for initial testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        train_loader, train_loader_eval, val_loader, test_loader = get_dataloaders(
            batch_size=batch_size,
            num_workers=2
        )

        model = GCViT(
            dim=56,
            depths=(2, 2, 2, 2),
            mlp_ratio=mlp_ratio,
            num_heads=(4, 4, 4, 4),
            num_classes=3,
            window_size=(7, 7, 7, 7),
            window_size_pre=(7, 7, 7, 7),
            resolution=28,
            drop_path_rate=drop_path_rate,
            in_chans=1,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            norm_layer=nn.LayerNorm,
            layer_scale=None,
            out_indices=(0, 1, 2, 3),
            frozen_stages=-1,
            pretrained=None,
            use_rel_pos_bias=True
        )

        trainer = GCViTTrainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            train_loader_at_eval=train_loader_eval,
            device=device,
            hyperparams={
                'batch_size': batch_size,
                'lr': lr,
                'weight_decay': weight_decay,
                'num_epochs': num_epochs,
                'num_classes': 3
            },
            task='multi-class',
            data_flag='fracturemnist3d',
        )

        logging.info(f"Trial starting with: batch_size={batch_size}, lr={lr:.2e}, wd={weight_decay:.2e}, drop_path={drop_path_rate}, mlp_ratio={mlp_ratio}")
        best_acc = trainer.train(return_best_val_AUC=True)
        return best_acc

    except Exception as e:
        logging.error(f"Trial failed: {e}")
        return 0.0  # Fail-safe: return worst possible score


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10, timeout=60*30)  # max 10 trials or 30 min

    print("\n✅ Best trial:")
    trial = study.best_trial
    for key, value in trial.params.items():
        print(f"  {key}: {value}")
    print(f"  Accuracy: {trial.value:.4f}")