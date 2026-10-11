"""
The final ensemble: a weighted average of the members' predicted probabilities.

ensemble_predict(split)
    Weighted mean of the saved sigmoid probabilities of the members in
    config.ENSEMBLE. This is the final model's prediction.

greedy_select(runs)
    How the weights in config.ENSEMBLE were chosen (Caruana et al., 2004), using
    the validation set only:
        1. start with the single model that has the best validation AUC;
        2. repeatedly add the model (possibly one already chosen) whose addition
           gives the largest validation AUC of the averaged prediction;
        3. stop when no addition improves the validation AUC by more than 1e-4.
    A model's weight is the number of times it was chosen. The test set is not
    used, so the test AUC of the result is an honest estimate. Our candidates were
    all trained DenseNet runs; the result is stored in config.ENSEMBLE.

    python chest_xray/ensemble.py improve/strong_s0 augment/aug_s1 ...   # rerun the selection
"""
import os
import sys

import numpy as np

import config
from dataset import load_npz
from metrics import per_class_auc


def labels(split):
    """Labels and class names of the validation ('val') or test ('test') subset."""
    _, Y, S, names = load_npz(config.data_path(config.CLS_SIZE))
    return Y[S == (1 if split == 'val' else 2)], names


def member_pred(run, split):
    """Saved sigmoid probabilities of one trained run on 'val' or 'test'."""
    return np.load(os.path.join(config.run_dir(run), f'{split}_pred.npy'))


def ensemble_predict(split='test'):
    """Weighted mean of the members' probabilities, shape (N, 14)."""
    weights = config.ENSEMBLE
    total = sum(weights.values())
    return sum(w * member_pred(run, split) for run, w in weights.items()) / total


def greedy_select(runs, max_steps=20):
    """Greedy forward selection with replacement on the validation set.
    Returns {run: number of times chosen}."""
    y, names = labels('val')
    preds = [member_pred(r, 'val') for r in runs]
    auc = lambda idx: per_class_auc(np.mean([preds[i] for i in idx], 0), y, names)[1]

    chosen = [int(np.argmax([auc([i]) for i in range(len(runs))]))]
    best = auc(chosen)
    for _ in range(max_steps):
        scores = [auc(chosen + [i]) for i in range(len(runs))]
        i = int(np.argmax(scores))
        if scores[i] <= best + 1e-4:
            break
        chosen.append(i)
        best = scores[i]
    print(f'validation AUC of the selected ensemble: {best:.4f}')
    return {runs[i]: chosen.count(i) for i in sorted(set(chosen), key=chosen.index)}


if __name__ == '__main__':
    print(greedy_select(sys.argv[1:]))
