"""
Data: pre-processing of the raw images, the patient-level split, PyTorch datasets
and the two data-augmentation settings used by the final models.

Pre-processed file (.npz) contents:
    images       uint8 (N, S, S)   grey-scale images resized to S x S
    labels       uint8 (N, 14)     multi-label targets, one 0/1 column per finding
    split        int8  (N,)        0 = train, 1 = validation, 2 = test
    names        (N,)              original file names
    label_names  (14,)             finding names, same order as config.LABELS
"""
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import config


# ----------------------------------------------------------------------------- pre-processing
def patient_split(patients, val_frac, test_frac, seed):
    """Randomly assign whole patients to train / validation / test.

    Splitting by patient (not by image) prevents leakage: images of the same
    person look alike, so they must never appear in two subsets.
    Returns an int8 array with 0 = train, 1 = validation, 2 = test per image."""
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(patients)))
    rng.shuffle(uniq)
    n_test = int(round(len(uniq) * test_frac))
    n_val = int(round(len(uniq) * val_frac))
    test_p = set(uniq[:n_test])
    val_p = set(uniq[n_test:n_test + n_val])
    return np.array([2 if p in test_p else 1 if p in val_p else 0 for p in patients], dtype=np.int8)


def prepare(size):
    """Resize every raw PNG to size x size grey-scale, build the 14-column label
    matrix from the 'Finding Labels' column (e.g. 'Effusion|Mass'), split by
    patient and save everything to data/processed/xray_<size>.npz."""
    import pandas as pd
    from PIL import Image

    df = pd.read_csv(config.LABELS_CSV)
    df = df[df['Image Index'].isin(set(os.listdir(config.IMG_DIR)))].reset_index(drop=True)
    labels = np.stack([df['Finding Labels'].str.split('|').apply(lambda s: c in s).values
                       for c in config.LABELS], axis=1).astype(np.uint8)

    images = np.zeros((len(df), size, size), dtype=np.uint8)
    for i, name in enumerate(df['Image Index']):
        img = Image.open(os.path.join(config.IMG_DIR, name)).convert('L')
        images[i] = np.asarray(img.resize((size, size), Image.BILINEAR))

    split = patient_split(df['Patient ID'].values, config.VAL_FRAC, config.TEST_FRAC, config.SPLIT_SEED)
    out = config.data_path(size)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, images=images, labels=labels, split=split,
                        names=df['Image Index'].values, label_names=np.array(config.LABELS))
    print(f'Saved {len(df)} images to {out} '
          f'(train/val/test = {[int((split == s).sum()) for s in range(3)]})')


def load_npz(path):
    """Return (images, labels, split, label_names) of a pre-processed file."""
    d = np.load(path, allow_pickle=True)
    return d['images'], d['labels'], d['split'], [str(n) for n in d['label_names']]


# ----------------------------------------------------------------------------- datasets
def to_tensor(img_u8):
    """uint8 image (H, W) -> float tensor (1, H, W) scaled to [-1, 1]."""
    return torch.from_numpy(img_u8).float().div(127.5).sub(1.0).unsqueeze(0)


class XrayDataset(Dataset):
    """Images and multi-label targets.

    augment=False: images are returned unchanged (validation / test).
    augment=True : light augmentation, used by the teacher, the GAN and most members:
        - random shift of up to 1/16 of the image size (edges are padded by
          repeating the border pixels);
        - random contrast (x 0.9-1.1) and brightness (+-0.1).
    No horizontal flip: the heart is on the left, a mirrored X-ray is anatomically wrong."""

    def __init__(self, images, labels, augment=False):
        self.x = images
        self.y = labels.astype(np.float32)
        self.augment = augment

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x = to_tensor(self.x[i])
        if self.augment:
            pad = x.shape[-1] // 16
            x = F.pad(x.unsqueeze(0), (pad,) * 4, mode='replicate').squeeze(0)
            dx, dy = np.random.randint(0, 2 * pad + 1, size=2)
            x = x[:, dy:dy + self.x.shape[1], dx:dx + self.x.shape[2]]
            x = (x * np.random.uniform(0.9, 1.1) + np.random.uniform(-0.1, 0.1)).clamp(-1, 1)
        return x, torch.from_numpy(self.y[i])


class StrongAugDataset(Dataset):
    """Strong augmentation, used by the 'strong' and 'chex10' members.

    Every time an image is loaded it gets a new random affine transform
        rotation +-10 degrees, scale 0.9-1.1, shift up to 1/16 of the image size
    (empty corners are filled with black), followed by a random contrast change
    (x 0.8-1.2 around the image mean) and brightness shift (+-0.1).
    Because the transform is redrawn every epoch, the network effectively sees a
    new variant of each image in every epoch."""

    def __init__(self, images, labels):
        self.x = images
        self.y = labels.astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as TF
        x = to_tensor(self.x[i])
        h = x.shape[-1]
        angle = float(np.random.uniform(-10, 10))
        shift = [int(v) for v in np.random.randint(-h // 16, h // 16 + 1, size=2)]
        scale = float(np.random.uniform(0.9, 1.1))
        x = TF.affine(x, angle=angle, translate=shift, scale=scale, shear=[0.0],
                      interpolation=InterpolationMode.BILINEAR, fill=[-1.0])
        m = x.mean()
        x = ((x - m) * np.random.uniform(0.8, 1.2) + m + np.random.uniform(-0.1, 0.1)).clamp(-1, 1)
        return x, torch.from_numpy(self.y[i])


def make_loader(ds, batch, train=False):
    """Training loaders shuffle and drop the last incomplete batch; evaluation
    loaders keep the original order so predictions line up with the labels."""
    return DataLoader(ds, batch_size=batch, shuffle=train, drop_last=train)
