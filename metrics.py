"""
Evaluation helpers: model predictions and per-class ROC AUC.
"""
import math

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


@torch.no_grad()
def predict(model, loader, device):
    """Run a model over a loader and return (sigmoid probabilities, labels), both
    numpy arrays of shape (N, 14). The model is put back in training mode after."""
    model.eval()
    ps, ys = [], []
    for x, y in loader:
        ps.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
        ys.append(y.numpy())
    model.train()
    return np.concatenate(ps), np.concatenate(ys)


def per_class_auc(p, y, names):
    """ROC AUC of every class and their unweighted mean (the metric reported
    everywhere). A class without both positives and negatives in the subset has
    no AUC; it is stored as nan and left out of the mean."""
    res = {}
    for k, n in enumerate(names):
        res[n] = float(roc_auc_score(y[:, k], p[:, k])) if 0 < y[:, k].sum() < len(y) else float('nan')
    valid = [v for v in res.values() if not math.isnan(v)]
    return res, float(np.mean(valid))
