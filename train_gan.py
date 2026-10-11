"""
Train the DenseNet-guided conditional GAN (step 3).

    python chest_xray/train_gan.py

Each iteration has two updates.

Discriminator D (real images x with labels y, generated images G(z, y_fake)):
    loss_D = lambda_cls * weighted BCE(cls_head(x), y)          learn to recognise diseases
           + hinge(adv_head(x), adv_head(G(z, y_fake)))         tell real from fake

Generator G:
    loss_G = -adv_head(G(z, y_fake))                            look real to D
           + lambda_guide * BCE(teacher(G(z, y_fake)), y_fake)  show the requested diseases
The teacher is the frozen DenseNet from step 2. It is needed because D's own class
head never learns to recognise diseases well enough to tell G what a finding looks
like. Its loss is switched on only after guide_warmup epochs, once G already draws
realistic X-rays. Generated 128x128 images are upsampled to 224 for the teacher.

The labels y_fake are drawn from real training cases, so generated images keep
realistic combinations of co-occurring findings.

Outputs in runs/gan_guided/: last_G.pt (the generator used in step 4) and a grid
of 32 samples every 5 epochs (fake_epXXX.png) to monitor progress.
"""
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import config
from dataset import XrayDataset, load_npz, make_loader
from loss import class_weighted_bce, d_hinge_loss, g_adv_loss, label_loss
from models import DenseNetClassifier, Discriminator, Generator
from utils import get_device, save_grid, set_seed


def load_teacher(device):
    """Frozen DenseNet from step 2: evaluation mode, no gradients to its weights."""
    teacher = DenseNetClassifier(pretrained=False).to(device)
    teacher.load_state_dict(torch.load(os.path.join(config.run_dir(config.TEACHER), 'best.pt'),
                                       map_location=device))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def train_gan():
    cfg = config.GAN_CFG
    set_seed(cfg['seed'])
    device = get_device()
    out_dir = config.run_dir(config.GAN_DIR)
    os.makedirs(out_dir, exist_ok=True)

    X, Y, S, _ = load_npz(config.data_path(config.GAN_SIZE))
    train_dl = make_loader(XrayDataset(X[S == 0], Y[S == 0], augment=True), cfg['batch'], train=True)
    bce = class_weighted_bce(Y[S == 0], cfg['max_pos_weight'], device)

    D = Discriminator().to(device)
    opt_d = torch.optim.Adam(D.parameters(), lr=cfg['lr_d'], betas=(0.0, 0.9))
    teacher = load_teacher(device)
    G = Generator(z_dim=cfg['z_dim']).to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=cfg['lr_g'], betas=(0.0, 0.9))

    # pool of real label combinations to condition the generator on
    train_labels = torch.from_numpy(Y[S == 0].astype(np.float32))

    def sample_noise_and_labels(b):
        z = torch.randn(b, cfg['z_dim'], device=device)
        y = train_labels[torch.randint(len(train_labels), (b,))].to(device)
        return z, y

    fixed_z, fixed_y = sample_noise_and_labels(32)          # same inputs for every sample grid

    for ep in range(1, cfg['epochs'] + 1):
        t0 = time.time()
        stats = {'d_adv': 0., 'd_cls': 0., 'g_adv': 0., 'g_guide': 0.}
        n = 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            b = x.size(0)

            # ---- discriminator update ----
            adv_real, cls_real = D(x)
            loss_cls = bce(cls_real, y)
            z, y_fake = sample_noise_and_labels(b)
            with torch.no_grad():
                fake = G(z, y_fake)                          # G is not updated in this step
            adv_fake, _ = D(fake)
            loss_adv = d_hinge_loss(adv_real, adv_fake)
            opt_d.zero_grad(set_to_none=True)
            (cfg['lambda_cls'] * loss_cls + loss_adv).backward()
            opt_d.step()

            # ---- generator update ----
            z, y_fake = sample_noise_and_labels(b)
            fake = G(z, y_fake)
            adv_fake, _ = D(fake)
            loss_g = g_adv_loss(adv_fake)
            if ep > cfg['guide_warmup']:
                fake_224 = F.interpolate(fake, size=config.CLS_SIZE, mode='bilinear', align_corners=False)
                loss_guide = label_loss(teacher(fake_224), y_fake)
                loss_g = loss_g + cfg['lambda_guide'] * loss_guide
                stats['g_guide'] += loss_guide.item() * b
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()

            stats['d_adv'] += loss_adv.item() * b
            stats['d_cls'] += loss_cls.item() * b
            stats['g_adv'] += -adv_fake.mean().item() * b
            n += b

        print(f'epoch {ep:2d}/{cfg["epochs"]}  ' + '  '.join(f'{k} {v / n:.3f}' for k, v in stats.items())
              + f'  {time.time() - t0:.0f}s')
        if ep % 5 == 0 or ep == 1:
            G.eval()
            with torch.no_grad():
                save_grid(G(fixed_z, fixed_y), os.path.join(out_dir, f'fake_ep{ep:03d}.png'))
            G.train()
            torch.save(G.state_dict(), os.path.join(out_dir, 'last_G.pt'))


if __name__ == '__main__':
    train_gan()
