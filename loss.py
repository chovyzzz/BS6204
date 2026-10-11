"""
Loss functions.

Classification (DenseNet members, and the discriminator's class head):
    class_weighted_bce  binary cross-entropy where positives of rare diseases weigh more
    mixup               blends pairs of images and their labels (used by one member)

GAN:
    d_hinge_loss        discriminator: push real scores above +1 and fake scores below -1
    g_adv_loss          generator: make the discriminator score its images as real
    label_loss          generator: make the frozen teacher recognise the requested labels
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def class_weighted_bce(train_labels, max_w, device):
    """BCE-with-logits loss with one positive weight per class.

    weight_c = (#negatives_c / #positives_c), clipped to [1, max_w], computed on the
    real training labels only. Without it the network can reach a low loss by
    predicting "no disease" for rare findings such as Hernia (about 0.2% positive).
    The tensor is float32 explicitly because Apple's MPS backend has no float64."""
    pos = train_labels.sum(0).astype(np.float32)
    neg = len(train_labels) - pos
    w = torch.tensor(np.clip(neg / np.maximum(pos, 1), 1, max_w), dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=w)


def mixup(x, y, alpha):
    """Mixup: draw lam ~ Beta(alpha, alpha) and replace the batch by
    lam * batch + (1 - lam) * shuffled batch, for both images and labels.
    It discourages over-confident predictions and acts as a regulariser."""
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], lam * y + (1 - lam) * y[idx]


def d_hinge_loss(adv_real, adv_fake):
    """Hinge loss of the discriminator: mean(relu(1 - real)) + mean(relu(1 + fake))."""
    return F.relu(1 - adv_real).mean() + F.relu(1 + adv_fake).mean()


def g_adv_loss(adv_fake):
    """Adversarial loss of the generator: maximise the discriminator's score of fakes."""
    return -adv_fake.mean()


def label_loss(logits, y):
    """Plain (unweighted) multi-label BCE between a classifier's logits on generated
    images and the labels the generator was asked to draw."""
    return F.binary_cross_entropy_with_logits(logits, y)
