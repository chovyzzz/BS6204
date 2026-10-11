"""
Reproduce the final model from the raw images.

    python chest_xray/main.py              # run every step; finished steps are skipped
    python chest_xray/main.py evaluate     # only step 6 (uses the saved predictions)

Steps (details in config.py):
    1. prepare   data/raw -> data/processed/xray_128.npz and xray_224.npz
    2. teacher   DenseNet-121 that guides the GAN               -> runs/densenet
    3. gan       DenseNet-guided conditional GAN                -> runs/gan_guided
    4. generate  2800 synthetic training images                 -> data/processed/synthetic_224.npz
    5. members   the 7 DenseNet-121 ensemble members            -> runs/<member>
    6. evaluate  weighted ensemble on the test set: per-disease AUC (runs/final_results.json),
                 ROC curves and confusion matrices (runs/roc/)
A step is skipped when its output already exists, so the saved models are reused.
Delete an output to redo its step.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import config                                   # noqa: E402
from dataset import prepare                     # noqa: E402
from ensemble import ensemble_predict, labels   # noqa: E402
from metrics import per_class_auc               # noqa: E402


def done(path):
    exists = os.path.exists(path)
    if exists:
        print(f'  already done: {os.path.relpath(path, HERE)}')
    return exists


def evaluate():
    """Step 6: test AUC of the final ensemble and the two figures."""
    from visualize import plot_confusion, plot_roc
    p_test, y_test = ensemble_predict('test'), labels('test')[0]
    p_val, (y_val, names) = ensemble_predict('val'), labels('val')
    res, mean = per_class_auc(p_test, y_test, names)

    print(f'\n{"Disease":20s}{"test AUC":>10s}{"positives":>11s}')
    for k, n in enumerate(names):
        print(f'{n:20s}{res[n]:>10.3f}{int(y_test[:, k].sum()):>11d}')
    print(f'{"Mean":20s}{mean:>10.3f}\n')
    with open(os.path.join(config.RUNS, 'final_results.json'), 'w') as f:
        json.dump({'members': config.ENSEMBLE, 'test_auc': mean, 'test_auc_per_class': res}, f, indent=2)

    plot_roc(p_test, y_test, names)
    plot_confusion(p_val, y_val, p_test, y_test, names)


def main():
    print('1. prepare')
    for size in (config.GAN_SIZE, config.CLS_SIZE):
        if not done(config.data_path(size)):
            prepare(size)

    print('2. teacher')
    if not done(os.path.join(config.run_dir(config.TEACHER), 'results.json')):
        from train_classifier import train_classifier
        train_classifier(config.TEACHER)

    print('3. gan')
    if not done(os.path.join(config.run_dir(config.GAN_DIR), 'last_G.pt')):
        from train_gan import train_gan
        train_gan()

    print('4. generate')
    if not done(config.SYNTHETIC):
        from generate import generate
        generate()

    print('5. members')
    for name in config.MEMBERS:
        if not done(os.path.join(config.run_dir(name), 'test_pred.npy')):
            from train_classifier import train_classifier
            train_classifier(name)

    print('6. evaluate')
    evaluate()


if __name__ == '__main__':
    if sys.argv[1:] == ['evaluate']:
        evaluate()
    else:
        main()
