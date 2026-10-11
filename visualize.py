"""
Figures of the final ensemble on the test set (saved as .png and .pdf in runs/roc/).

plot_roc(p, y, names)
    The ROC curves of all 14 diseases in one plot, plus the macro-average curve.
    Legend entries are sorted by AUC; 7 colours are used twice, solid lines for
    the 7 best diseases and dashed lines for the other 7.

plot_confusion(p_val, y_val, p_test, y_test, names)
    One 2x2 confusion matrix per disease. The task is multi-label (an image can
    have several diseases), so there is no single 14x14 matrix; each disease is
    its own yes/no decision. Turning a probability into yes/no needs a threshold:
    for every disease it is chosen on the validation set as the point that
    maximises Youden's J = sensitivity + specificity - 1, and then applied
    unchanged to the test set.
"""
import os

import numpy as np

import config

OUT_DIR = os.path.join(config.RUNS, 'roc')
HUES = ['#2a78d6', '#eb6834', '#1baf7a', '#e87ba4', '#008300', '#6250d6', '#e34948']
INK, INK2, GRID, CHANCE = '#0b0b0b', '#52514e', '#e7e6e2', '#c9c8c2'


def _pyplot():
    import matplotlib
    matplotlib.use('Agg')                                    # draw to files, no window
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2,
                         'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK})
    return plt


def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT_DIR, f'{name}.{ext}'), dpi=200, bbox_inches='tight', facecolor='white')
    print(f'saved {os.path.join(OUT_DIR, name)}.png / .pdf')


def plot_roc(p, y, names):
    from sklearn.metrics import roc_auc_score, roc_curve
    plt = _pyplot()
    rows = sorted(((roc_auc_score(y[:, k], p[:, k]), k, n) for k, n in enumerate(names)), reverse=True)

    fig, ax = plt.subplots(figsize=(9.6, 7.2))
    ax.plot([0, 1], [0, 1], ls=':', lw=1, color=CHANCE)                 # chance level
    grid = np.linspace(0, 1, 501)
    tprs = []
    for i, (auc, k, name) in enumerate(rows):
        fpr, tpr, _ = roc_curve(y[:, k], p[:, k])
        tprs.append(np.interp(grid, fpr, tpr))                         # for the macro average
        ax.plot(fpr, tpr, lw=1.6, color=HUES[i % 7], ls='-' if i < 7 else '--',
                label=f'{name} ({auc:.3f}, n+={int(y[:, k].sum())})')
    macro = np.mean(tprs, 0)
    macro[0] = 0
    ax.plot(grid, macro, lw=3, color=INK, label=f'Macro-average ({np.mean([r[0] for r in rows]):.3f})')

    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01); ax.set_aspect('equal')
    ax.set_xlabel('False positive rate (1 − specificity)')
    ax.set_ylabel('True positive rate (sensitivity)')
    ax.grid(color=GRID, lw=0.6); ax.tick_params(length=0)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.set_title('ROC curves of the final DenseNet-121 ensemble on the test set', loc='left', fontsize=12, pad=12)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9,
              title='Disease (AUC, test positives)', title_fontsize=9.5, handlelength=2.6)
    _save(fig, 'roc_all_in_one')


def youden_thresholds(p_val, y_val):
    """Per disease, the validation threshold that maximises sensitivity + specificity - 1."""
    from sklearn.metrics import roc_curve
    th = []
    for k in range(y_val.shape[1]):
        fpr, tpr, t = roc_curve(y_val[:, k], p_val[:, k])
        th.append(float(min(t[int(np.argmax(tpr - fpr))], 1.0)))
    return np.array(th)


def plot_confusion(p_val, y_val, p_test, y_test, names):
    from matplotlib.colors import LinearSegmentedColormap
    plt = _pyplot()
    th = youden_thresholds(p_val, y_val)
    cmap = LinearSegmentedColormap.from_list('blue', ['#f4f8fd', '#2a78d6'])

    fig, axes = plt.subplots(4, 4, figsize=(12.5, 13.2))
    print(f'{"Disease":20s}{"thr":>6s}{"TP":>5s}{"FN":>5s}{"FP":>5s}{"TN":>6s}{"sens":>7s}{"spec":>7s}{"PPV":>7s}')
    for k, (ax, name) in enumerate(zip(axes.flat, names)):
        truth, pred = y_test[:, k].astype(bool), p_test[:, k] >= th[k]
        # rows = true label (+, -), columns = prediction (+, -)
        cm = np.array([[np.sum(truth & pred), np.sum(truth & ~pred)],
                       [np.sum(~truth & pred), np.sum(~truth & ~pred)]])
        rate = cm / cm.sum(1, keepdims=True)                 # row percentages
        sens, spec = rate[0, 0], rate[1, 1]
        ppv = cm[0, 0] / max(cm[:, 0].sum(), 1)              # precision of positive predictions

        ax.imshow(rate, cmap=cmap, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                dark = rate[i, j] > 0.55
                ax.text(j, i - 0.08, f'{cm[i, j]}', ha='center', va='center', fontsize=13,
                        color='white' if dark else INK)
                ax.text(j, i + 0.22, f'{rate[i, j]:.0%}', ha='center', va='center', fontsize=8.5,
                        color='white' if dark else INK2)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['Pred +', 'Pred −'])
        ax.set_yticks([0, 1]); ax.set_yticklabels(['True +', 'True −'])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        star = ' *' if y_val[:, k].sum() < 5 else ''
        ax.set_title(f'{name}{star}', loc='left', fontsize=11, pad=20)
        ax.text(0, 1.04, f'thr {th[k]:.2f} · sens {sens:.2f} · spec {spec:.2f} · PPV {ppv:.2f}',
                transform=ax.transAxes, fontsize=8, color=INK2)
        print(f'{name:20s}{th[k]:>6.2f}{cm[0, 0]:>5d}{cm[0, 1]:>5d}{cm[1, 0]:>5d}{cm[1, 1]:>6d}'
              f'{sens:>7.2f}{spec:>7.2f}{ppv:>7.2f}')

    for ax in axes.flat[len(names):]:
        ax.axis('off')
    axes.flat[len(names)].text(
        0, 0.98, f'Test set ({len(y_test)} images), final ensemble.\n\n'
        'Rows: true label; columns: prediction.\nCell: count and row percentage, so\n'
        'top-left = sensitivity, bottom-right\n= specificity.\n\n'
        'Threshold per disease chosen on the\nvalidation set (max Youden J =\n'
        'sens + spec − 1); the test set was\nnot used to choose it.',
        transform=axes.flat[len(names)].transAxes, fontsize=9, va='top', color=INK2, linespacing=1.45)
    axes.flat[len(names) + 1].text(
        0, 0.98, 'PPV = precision: of the images\npredicted positive, the share that\nare truly positive. '
        'It is low for\nrare diseases even when sensitivity\nand specificity are good.\n\n'
        '* fewer than 5 validation positives:\nthreshold is unreliable.',
        transform=axes.flat[len(names) + 1].transAxes, fontsize=9, va='top', color=INK2, linespacing=1.45)
    fig.tight_layout(h_pad=2.6)
    _save(fig, 'confusion_matrices')
