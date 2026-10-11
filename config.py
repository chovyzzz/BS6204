"""
Project configuration: file paths, class names and every hyper-parameter of the
final pipeline. All other modules read their settings from here, so this file is
the single place that documents how the final model was built.

Final model (test mean AUC 0.786)
---------------------------------
A weighted average of 7 DenseNet-121 classifiers (ENSEMBLE below). Three of them
were trained with extra images produced by a conditional GAN. The GAN is guided by
a frozen "teacher" DenseNet so that the generated images carry the requested
disease labels.

Pipeline (main.py runs the steps in this order):
    1. prepare   raw PNG -> data/processed/xray_128.npz (GAN) and xray_224.npz (DenseNet)
    2. teacher   DenseNet-121 that guides the GAN                  -> runs/densenet
    3. gan       DenseNet-guided conditional GAN                    -> runs/gan_guided
    4. generate  2800 labelled synthetic images (training set only) -> data/processed/synthetic_224.npz
    5. members   the 7 DenseNet-121 classifiers listed in MEMBERS   -> runs/<name>
    6. evaluate  weighted ensemble on the test set, ROC curves and confusion matrices
"""
import os

# ----------------------------------------------------------------------------- paths
PROJECT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT, 'data')
IMG_DIR = os.path.join(DATA_DIR, 'raw', 'images')                  # 5606 original PNG files
LABELS_CSV = os.path.join(DATA_DIR, 'raw', 'sample_labels.csv')    # Kaggle label table
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
SYNTHETIC = os.path.join(PROCESSED_DIR, 'synthetic_224.npz')       # GAN images (step 4)
RUNS = os.path.join(PROJECT, 'runs')                               # trained models and results


def data_path(size):
    """Pre-processed dataset of size x size images, e.g. data/processed/xray_224.npz."""
    return os.path.join(PROCESSED_DIR, f'xray_{size}.npz')


def run_dir(name):
    """Output folder of one training run, e.g. run_dir('improve/strong_s0')."""
    return os.path.join(RUNS, *name.split('/'))


# ----------------------------------------------------------------------------- data
# The 14 findings, in the same order as the label columns stored in the .npz files.
LABELS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
          'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule',
          'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
N_CLS = len(LABELS)

# Patient-level split: every image of a patient goes to the same subset, so the
# model is never tested on a patient it has seen. Result: 3942 / 547 / 1117 images.
SPLIT_SEED = 42
VAL_FRAC = 0.1
TEST_FRAC = 0.2

GAN_SIZE = 128      # image size used by the GAN
CLS_SIZE = 224      # image size used by DenseNet-121 (its ImageNet resolution)

# ImageNet normalisation used by the torchvision DenseNet weights.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Chest X-ray weights from TorchXRayVision trained on CheXpert only. Weights that
# include NIH ChestX-ray ("nih", "all") must not be used: our test set is NIH data.
XRV_CHEX = 'densenet121-res224-chex'

# ----------------------------------------------------------------------------- classifier
# Default DenseNet training settings; each member below overrides only what differs.
CLS_DEFAULTS = dict(
    epochs=6,
    batch=16,
    lr=1e-4,                # AdamW learning rate
    wd=1e-4,                # AdamW weight decay
    warmup=1,               # epochs of linear warm-up, then cosine decay to 0
    clip=1.0,               # gradient-norm clipping
    max_pos_weight=30.0,    # upper limit of the per-class positive weight in the loss
    pretrained=True,        # start from pretrained weights (False = random init)
    backbone='imagenet',    # 'imagenet' (torchvision) or 'xrv-chex' (CheXpert)
    strong_aug=False,       # False = light augmentation, True = strong augmentation
    mixup=0.0,              # mixup alpha, 0 = off
    synthetic=False,        # add the 2800 GAN images to the training set
)

# Teacher that guides the GAN (step 2). Light augmentation, 15 epochs.
TEACHER = 'densenet'
TEACHER_CFG = dict(seed=0, epochs=15)

# The 7 ensemble members (step 5). The folder names are kept from the experiments
# in which they were first trained, so that the saved models are reused.
MEMBERS = {
    # ImageNet weights + light augmentation + 2800 GAN images
    'augment/aug_s0':           dict(seed=0, synthetic=True),
    'augment/aug_s1':           dict(seed=1, synthetic=True),
    # ImageNet weights + light augmentation, trained longer (10 epochs)
    'augment/real_long_s1':     dict(seed=1, epochs=10),
    # ImageNet weights + strong augmentation
    'improve/strong_s0':        dict(seed=0, strong_aug=True),
    'improve/strong_s2':        dict(seed=2, strong_aug=True),
    # CheXpert weights + strong augmentation + mixup, 10 epochs
    'improve/chex10_s1':        dict(seed=1, epochs=10, backbone='xrv-chex', strong_aug=True, mixup=0.2),
    # Random initialisation + light augmentation (weak alone, but adds diversity)
    'scratch/scratch_light_s2': dict(seed=2, epochs=25, pretrained=False, lr=1e-3, warmup=2),
}

# Ensemble weights: how many times greedy selection on the validation set picked
# each member (see ensemble.greedy_select). The ensemble prediction is the
# weighted mean of the members' sigmoid probabilities (weights sum to 8).
ENSEMBLE = {
    'augment/aug_s1': 2,
    'augment/aug_s0': 1,
    'augment/real_long_s1': 1,
    'improve/strong_s0': 1,
    'improve/strong_s2': 1,
    'improve/chex10_s1': 1,
    'scratch/scratch_light_s2': 1,
}

# ----------------------------------------------------------------------------- GAN
GAN_DIR = 'gan_guided'
GAN_CFG = dict(
    seed=0,
    epochs=60,
    batch=32,
    z_dim=128,              # length of the random noise vector
    lr_d=2e-4,              # Adam, betas (0, 0.9), for the discriminator
    lr_g=2e-4,              # same for the generator
    lambda_cls=1.0,         # weight of the discriminator's classification loss on real images
    lambda_guide=1.0,       # weight of the teacher (label) loss for the generator
    guide_warmup=20,        # first 20 epochs: generator learns realistic X-rays only
    max_pos_weight=30.0,
)

# Synthetic training images (step 4): per_class images for each disease (label
# combinations copied from real training cases containing that disease) plus
# n_normal images with an all-zero label. Half normal, half abnormal, so that the
# classifier cannot learn "GAN image = disease".
SYNTH_CFG = dict(per_class=100, n_normal=1400, seed=0)
