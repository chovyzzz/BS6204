"""
Generate the labelled synthetic training images (step 4).

    python chest_xray/generate.py

For each of the 14 diseases, per_class label vectors are copied from random real
training cases that contain that disease (so rare diseases get as many images as
common ones, and co-occurring findings stay realistic). n_normal all-zero labels
are added. The generator from step 3 draws one 128x128 image per label, the image
is upsampled to 224x224 and everything is saved to data/processed/synthetic_224.npz.
These images are only ever added to the training set of the 'aug' members.
"""
import numpy as np
import torch
import torch.nn.functional as F

import config
from dataset import load_npz
from models import Generator
from utils import get_device


@torch.no_grad()
def generate():
    cfg = config.SYNTH_CFG
    device = get_device()
    _, Y, S, _ = load_npz(config.data_path(config.GAN_SIZE))
    Ytr = Y[S == 0]
    rng = np.random.default_rng(cfg['seed'])

    # label vectors: per_class per disease, then n_normal normal cases
    labels = [Ytr[rng.choice(np.where(Ytr[:, k] == 1)[0], cfg['per_class'], replace=True)]
              for k in range(config.N_CLS)]
    labels.append(np.zeros((cfg['n_normal'], config.N_CLS), dtype=Ytr.dtype))
    labels = np.concatenate(labels).astype(np.uint8)

    G = Generator(z_dim=config.GAN_CFG['z_dim']).to(device)
    G.load_state_dict(torch.load(f'{config.run_dir(config.GAN_DIR)}/last_G.pt', map_location=device))
    G.eval()
    torch.manual_seed(cfg['seed'])
    images = []
    for i in range(0, len(labels), 100):                     # batches of 100
        y = torch.from_numpy(labels[i:i + 100].astype(np.float32)).to(device)
        x = G(torch.randn(len(y), G.z_dim, device=device), y)
        x = F.interpolate(x, size=config.CLS_SIZE, mode='bilinear', align_corners=False)
        images.append(((x.clamp(-1, 1) + 1) * 127.5).round().byte().cpu().numpy()[:, 0])   # back to uint8
    np.savez_compressed(config.SYNTHETIC, images=np.concatenate(images), labels=labels)
    print(f'Saved {len(labels)} synthetic images to {config.SYNTHETIC}')


if __name__ == '__main__':
    generate()
