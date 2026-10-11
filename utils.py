"""
Small shared helpers: device selection, seeding, learning-rate schedule, image grid.
"""
import math

import numpy as np
import torch


def get_device():
    """Use the Apple GPU (MPS) or CUDA when available, otherwise the CPU."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def set_seed(seed):
    """Seed PyTorch (weight init, shuffling, noise) and NumPy (augmentation, mixup)."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def warmup_cosine(opt, warmup, total):
    """Per-epoch learning-rate multiplier: rises linearly to 1 over `warmup`
    epochs, then follows a cosine curve down to 0 at epoch `total`.
    Call scheduler.step() once at the end of every epoch."""
    return torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: (e + 1) / (warmup + 1) if e < warmup
        else 0.5 * (1 + math.cos(math.pi * (e - warmup) / max(total - warmup, 1))))


def save_grid(imgs, path, nrow=8):
    """Save a batch of single-channel images in [-1, 1] as one PNG grid."""
    from PIL import Image
    imgs = ((imgs.clamp(-1, 1) + 1) * 127.5).byte().cpu().numpy()[:, 0]
    n, h, w = imgs.shape
    grid = np.zeros((math.ceil(n / nrow) * h, nrow * w), dtype=np.uint8)
    for i in range(n):
        r, c = divmod(i, nrow)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = imgs[i]
    Image.fromarray(grid).save(path)
