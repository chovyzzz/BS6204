"""
Network architectures.

    Generator           conditional DCGAN generator: (noise z, disease label y) -> 128x128 X-ray
    Discriminator       convolutional trunk with two heads: real/fake score and 14 class logits
    DenseNetClassifier  DenseNet-121 with a new 14-output layer (the ensemble members)

Note 1: the attribute names (fc, net, trunk, adv, cls, mean, std) are the keys
inside the saved .pt files; renaming them would break loading the trained models.
Note 2: this file must not be called model.py. TorchXRayVision contains the
absolute import `from model.utils import ...`, which would pick up our file
instead of its own package and fail when the CheXpert weights are loaded.
"""
import json
import math
import os

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

import config


class Generator(nn.Module):
    """Conditional DCGAN generator.

    Input : z (B, 128) standard normal noise and y (B, 14) 0/1 disease labels.
    1. z and y are concatenated into a 142-d vector, so the label decides
       *what* to draw and the noise decides *how* it looks.
    2. A fully connected layer maps it to a 512 x 4 x 4 feature map.
    3. Each transposed convolution (kernel 4, stride 2, padding 1) doubles the
       spatial size and halves the channels: 4 -> 8 -> 16 -> 32 -> 64 -> 128,
       512 -> 256 -> 128 -> 64 -> 32 -> 1, with BatchNorm + ReLU in between.
    4. Tanh maps the output to [-1, 1], the same range as the real images."""

    def __init__(self, img_size=config.GAN_SIZE, z_dim=128, n_cls=config.N_CLS, base=512):
        super().__init__()
        self.z_dim = z_dim
        self.base = base
        n_up = int(math.log2(img_size)) - 2                  # number of x2 steps from 4 to img_size
        self.fc = nn.Linear(z_dim + n_cls, base * 4 * 4)
        layers, ch = [], base
        for i in range(n_up):
            if i < n_up - 1:
                out = max(base // 2 ** (i + 1), 32)
                layers += [nn.ConvTranspose2d(ch, out, 4, 2, 1, bias=False),   # BN follows, so no bias
                           nn.BatchNorm2d(out), nn.ReLU(True)]
            else:
                out = 1
                layers += [nn.ConvTranspose2d(ch, 1, 4, 2, 1), nn.Tanh()]
            ch = out
        self.net = nn.Sequential(nn.BatchNorm2d(base), nn.ReLU(True), *layers)

    def forward(self, z, y):
        h = self.fc(torch.cat([z, y], 1)).view(-1, self.base, 4, 4)
        return self.net(h)


class Discriminator(nn.Module):
    """Two-headed discriminator.

    Trunk: per stage a 4x4 convolution with stride 2 (halves the size) and a 3x3
    convolution, each followed by LeakyReLU(0.2); channels 64 -> 128 -> 256 -> 512
    -> 512, image 128 -> 4. Global average pooling gives a 512-d feature vector.
    Heads:
        adv : linear 512 -> 1, real/fake score (trained with the hinge loss)
        cls : dropout 0.3 + linear 512 -> 14, disease logits (trained on real images)
    Spectral normalisation on the trunk and the adv head limits how fast the
    discriminator can change, which keeps GAN training stable.
    forward returns (adv score of shape (B,), class logits of shape (B, 14))."""

    def __init__(self, img_size=config.GAN_SIZE, n_cls=config.N_CLS, base=64, dropout=0.3):
        super().__init__()
        n_down = int(math.log2(img_size)) - 2
        layers, ch = [], 1
        for i in range(n_down):
            out = min(base * 2 ** i, 512)
            layers += [spectral_norm(nn.Conv2d(ch, out, 4, 2, 1)), nn.LeakyReLU(0.2, True),
                       spectral_norm(nn.Conv2d(out, out, 3, 1, 1)), nn.LeakyReLU(0.2, True)]
            ch = out
        self.trunk = nn.Sequential(*layers)
        self.adv = spectral_norm(nn.Linear(ch, 1))
        self.cls = nn.Sequential(nn.Dropout(dropout), nn.Linear(ch, n_cls))

    def forward(self, x):
        h = self.trunk(x).mean(dim=[2, 3])                   # global average pooling
        return self.adv(h).squeeze(1), self.cls(h)


class DenseNetClassifier(nn.Module):
    """DenseNet-121 with its classifier replaced by a linear layer 1024 -> 14.

    The output is one logit per disease; a sigmoid (not a softmax) turns each into
    an independent probability, because one image can show several diseases.
    Input: (B, 1, 224, 224) grey-scale images in [-1, 1].

    backbone='imagenet'  torchvision ImageNet weights. The grey image is mapped to
                         [0, 1], copied to 3 channels and normalised with the
                         ImageNet mean / std, the input format these weights expect.
    backbone='xrv-chex'  TorchXRayVision weights trained on CheXpert X-rays. These
                         expect 1 channel with pixel values in [-1024, 1024].
    pretrained=False     random initialisation (also used when loading a saved
                         model, whose weights then replace the initial ones)."""

    def __init__(self, n_cls=config.N_CLS, pretrained=True, backbone='imagenet'):
        super().__init__()
        self.backbone = backbone
        if backbone == 'imagenet':
            from torchvision import models as tv_models
            weights = tv_models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            self.net = tv_models.densenet121(weights=weights)
            self.net.classifier = nn.Linear(self.net.classifier.in_features, n_cls)
            self.register_buffer('mean', torch.tensor(config.IMAGENET_MEAN).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor(config.IMAGENET_STD).view(1, 3, 1, 1))
        elif backbone == 'xrv-chex':
            try:
                import torchxrayvision as xrv
            except ImportError as e:
                raise SystemExit(f'Cannot import torchxrayvision ({e}): pip install torchxrayvision')
            self.net = xrv.models.DenseNet(weights=config.XRV_CHEX if pretrained else None)
            self.net.op_threshs = None                       # drop its own output calibration
            self.net.classifier = nn.Linear(self.net.classifier.in_features, n_cls)
        else:
            raise ValueError(f'unknown backbone: {backbone}')

    def forward(self, x):
        if self.backbone == 'imagenet':
            x = (x + 1) / 2                                  # [-1, 1] -> [0, 1]
            x = x.repeat(1, 3, 1, 1)                         # grey -> 3 channels
            return self.net((x - self.mean) / self.std)
        return self.net.classifier(self.net.features2(x * 1024))


def load_classifier(run, device):
    """Load the trained DenseNet of a run folder (best.pt); the backbone type is
    read from its results.json."""
    folder = config.run_dir(run)
    with open(os.path.join(folder, 'results.json')) as f:
        backbone = json.load(f)['args'].get('backbone', 'imagenet')
    model = DenseNetClassifier(pretrained=False, backbone=backbone)
    model.load_state_dict(torch.load(os.path.join(folder, 'best.pt'), map_location=device))
    return model.to(device).eval()
