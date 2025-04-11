def _to_channel_last(x):
    """
    Args:
        x: (B, C, D, H, W)

    Returns:
        x: (B, D, H, W, C)
    """
    return x.permute(0, 2, 3, 4, 1)


def _to_channel_first(x):
    """
    Args:
        x: (B, D, H, W, C)

    Returns:
        x: (B, C, D, H, W)
    """
    return x.permute(0, 4, 1, 2, 3)
