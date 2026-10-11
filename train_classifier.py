"""
Train one DenseNet-121 classifier (the teacher, or one ensemble member).

    python chest_xray/train_classifier.py improve/strong_s0     # retrain one member
    python chest_xray/train_classifier.py densenet              # retrain the teacher

The settings come from config.py (CLS_DEFAULTS updated with the run's own entry
in MEMBERS / TEACHER_CFG). Training:
    loss       class-weighted BCE (rare diseases weigh more), optional mixup
    optimiser  AdamW, linear warm-up then cosine decay, gradient clipping
    selection  after every epoch the mean validation AUC is computed and the best
               epoch is kept (best.pt)
Outputs in runs/<name>/:
    best.pt                     weights of the best validation epoch
    val_pred.npy, test_pred.npy sigmoid probabilities of that epoch (used by the ensemble)
    results.json                settings, validation curve and test AUC
"""
import json
import os
import sys
import time

import numpy as np
import torch

import config
from dataset import StrongAugDataset, XrayDataset, load_npz, make_loader
from loss import class_weighted_bce, mixup
from metrics import per_class_auc, predict
from models import DenseNetClassifier
from utils import get_device, set_seed, warmup_cosine


def settings(name):
    """Full settings of a run: the defaults overridden by its own entry."""
    own = config.TEACHER_CFG if name == config.TEACHER else config.MEMBERS[name]
    return {**config.CLS_DEFAULTS, **own}


def train_classifier(name):
    cfg = settings(name)
    set_seed(cfg['seed'])
    device = get_device()
    out_dir = config.run_dir(name)
    os.makedirs(out_dir, exist_ok=True)

    # ---- data: training set (+ GAN images if requested), validation, test ----
    X, Y, S, names = load_npz(config.data_path(config.CLS_SIZE))
    Xtr, Ytr = X[S == 0], Y[S == 0]
    if cfg['synthetic']:
        syn = np.load(config.SYNTHETIC)
        Xtr = np.concatenate([Xtr, syn['images']])
        Ytr = np.concatenate([Ytr, syn['labels'].astype(Ytr.dtype)])
    train_ds = StrongAugDataset(Xtr, Ytr) if cfg['strong_aug'] else XrayDataset(Xtr, Ytr, augment=True)
    train_dl = make_loader(train_ds, cfg['batch'], train=True)
    val_dl = make_loader(XrayDataset(X[S == 1], Y[S == 1]), cfg['batch'])
    test_dl = make_loader(XrayDataset(X[S == 2], Y[S == 2]), cfg['batch'])
    print(f'[{name}] device {device} | {len(Xtr)} training images | {cfg}')

    # ---- model, loss, optimiser ----
    model = DenseNetClassifier(pretrained=cfg['pretrained'], backbone=cfg['backbone']).to(device)
    # class weights use the real training labels only, so they do not change
    # when GAN images are added
    bce = class_weighted_bce(Y[S == 0], cfg['max_pos_weight'], device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    sched = warmup_cosine(opt, cfg['warmup'], cfg['epochs'])

    # ---- training loop ----
    best, history = -1.0, []
    for ep in range(1, cfg['epochs'] + 1):
        t0, total, n = time.time(), 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            if cfg['mixup'] > 0:
                x, y = mixup(x, y, cfg['mixup'])
            loss = bce(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip'])
            opt.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
        sched.step()

        _, val_auc = per_class_auc(*predict(model, val_dl, device), names)
        history.append({'epoch': ep, 'loss': total / n, 'val_auc': val_auc})
        print(f'  epoch {ep:2d}/{cfg["epochs"]}  loss {total / n:.3f}  val AUC {val_auc:.4f}  {time.time() - t0:.0f}s')
        if val_auc > best:
            best = val_auc
            torch.save(model.state_dict(), os.path.join(out_dir, 'best.pt'))

    # ---- reload the best epoch, save its predictions and the test result ----
    model.load_state_dict(torch.load(os.path.join(out_dir, 'best.pt'), map_location=device))
    for split, dl in (('val', val_dl), ('test', test_dl)):
        p, y = predict(model, dl, device)
        np.save(os.path.join(out_dir, f'{split}_pred.npy'), p)
    res, test_auc = per_class_auc(p, y, names)
    print(f'  best val AUC {best:.4f} | test AUC {test_auc:.4f}')
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump({'args': cfg, 'best_val_auc': best, 'test_auc': test_auc,
                   'test_auc_per_class': res, 'history': history}, f, indent=2)


if __name__ == '__main__':
    train_classifier(sys.argv[1])
