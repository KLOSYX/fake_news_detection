import itertools
from typing import List

import matplotlib.pyplot as plt
import mplfonts
import numpy as np


def plot_confusion_matrix(
    cm,
    target_names: List[str],
    title="Confusion matrix",
    cmap="Blues",  # 这个地方设置混淆矩阵的颜色主题，这个主题看着就干净~
    normalize=True,
):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap("Blues")

    mplfonts.use_font("Noto Sans CJK SC")
    fig = plt.figure(figsize=(cm.shape[0], cm.shape[0]), dpi=120)
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            f"{cm[i, j]:0.4f}",
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label", size=15)
    plt.xlabel(f"Predicted label\naccuracy={accuracy:0.4f}; misclass={misclass:0.4f}", size=15)
    return fig
