from vit_pytorch import ViT

def get_vit(num_classes=8):
    model = ViT(
        image_size=28, patch_size=7, num_classes=num_classes,
        dim=64, depth=6, heads=8, mlp_dim=128,
        dropout=0.1, emb_dropout=0.1
    )
    return model
