# Chest X-ray 14-class multi-label classification (NIH ChestX-ray 5% sample)

Final model: a weighted ensemble of 7 DenseNet-121 classifiers, test mean AUC **0.786**.
A DenseNet-guided conditional GAN produces 2800 labelled images that are added to the
training set of two of the members.

## Files

```
chest_xray/
├── main.py               runs the whole pipeline (finished steps are skipped)
├── config.py             paths, class names and every setting of the final model
├── dataset.py            pre-processing, patient-level split, datasets, augmentation
├── models.py             Generator, Discriminator, DenseNetClassifier
├── loss.py               class-weighted BCE, mixup, GAN losses
├── metrics.py            predictions and per-class AUC
├── utils.py              device, seeding, learning-rate schedule, image grid
├── train_classifier.py   trains the teacher or one ensemble member
├── train_gan.py          trains the DenseNet-guided GAN
├── generate.py           generates the synthetic training images
├── ensemble.py           weighted ensemble; greedy selection used to choose the weights
├── visualize.py          ROC curves and confusion matrices
├── data/raw/             original PNG images and sample_labels.csv
├── data/processed/       xray_128.npz, xray_224.npz, synthetic_224.npz
└── runs/                 trained models, predictions, results and figures
```

## Run

```bash
PY=/Users/chovyzzz/bio-rl-recommender/.venv/bin/python
P=/Users/chovyzzz/bio-rl-recommender/archive/chest_xray

$PY $P/main.py             # full pipeline; with the saved models only step 6 runs
$PY $P/main.py evaluate    # test AUC, runs/roc/roc_all_in_one.* and confusion_matrices.*
$PY $P/train_classifier.py improve/strong_s0   # retrain one member
```

## Final ensemble

| Member | Weights | Training data | Augmentation | Epochs | Weight |
|---|---|---|---|---|---|
| augment/aug_s1 | ImageNet | real + GAN | light | 6 | 2/8 |
| augment/aug_s0 | ImageNet | real + GAN | light | 6 | 1/8 |
| augment/real_long_s1 | ImageNet | real | light | 10 | 1/8 |
| improve/strong_s0 | ImageNet | real | strong | 6 | 1/8 |
| improve/strong_s2 | ImageNet | real | strong | 6 | 1/8 |
| improve/chex10_s1 | CheXpert | real | strong + mixup | 10 | 1/8 |
| scratch/scratch_light_s2 | random | real | light | 25 | 1/8 |

Weights were chosen by greedy selection on the validation set (`ensemble.greedy_select`);
the test set was used only for the final evaluation.

## Data

Kaggle "NIH Chest X-ray sample" (5606 images), split by patient into
3942 train / 547 validation / 1117 test images (seed 42). No pretrained weights that
have seen NIH ChestX-ray data are used.
