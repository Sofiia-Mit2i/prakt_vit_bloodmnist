ACC and AUC values improvements

#data.dataset
1. Created MedicalNormalize class with:
    Percentile-based clipping (0.5-99.5%)
    Z-score normalization
    [-1, 1] range scaling
    Uses robust statistics from training set

'''
class MedicalNormalize:
    def __init__(self, mean, std, pct_range=(0.5, 99.5)):
        self.mean = mean
        self.std = std
        self.pct_range = pct_range

    def __call__(self, x):
        # 1. Percentile-based clipping
        pct_low = np.percentile(x.numpy(), self.pct_range[0])
        pct_high = np.percentile(x.numpy(), self.pct_range[1])
        x = torch.clamp(x, pct_low, pct_high)
        
        # 2. Z-score normalization
        x = (x - self.mean) / self.std
        
        # 3. Scale to [-1, 1] range
        x = 2 * (x - x.min()) / (x.max() - x.min()) - 1
        return x

def compute_robust_stats(dataset, num_samples=500):
    """Compute robust statistics using random samples"""
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    samples = torch.stack([dataset[i][0] for i in indices])
    
    return {
        'mean': samples.mean().item(),
        'std': samples.std().item(),
        'median': samples.median().item(),
        'pct_05': np.percentile(samples.numpy(), 0.5),
        'pct_995': np.percentile(samples.numpy(), 99.5)
    }
'''
2. 3D-Specific Augmentations:
    Added 3D-compatible transforms:
    Random affine transformations
    Multi-axis flipping
    Random erasing
    80% probability of applying augmentations
'''
 # Define transforms
    train_transform = transforms.Compose([
        ToTensor4D(),
        MedicalNormalize(mean=stats['mean'], std=stats['std']),
        transforms.RandomApply([
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5)
        ], p=0.8),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))
    ])
'''
3. Class Imbalance Handling:
    Implemented WeightedRandomSampler
    Automatic class weight calculation
'''
    train_labels = [label for _, label in train_dataset]
    class_counts = np.bincount(train_labels)
    weights = 1. / class_counts[train_labels]
    sampler = WeightedRandomSampler(weights, len(weights))
'''