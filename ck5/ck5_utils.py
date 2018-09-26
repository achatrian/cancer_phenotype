import numpy as np
from sklearn.metrics import confusion_matrix

# also have a look at pandas_ml confusion matrix
# https://stackoverflow.com/questions/2148543/how-to-write-a-confusion-matrix-in-python/30385488

def eval_classification(predictions, targets):
    r"""
    :param predictions:
    :param targets:
    :return:
    Computes metrics, works for single channel too
    y vs x : true vs predicted
    """
    cm = confusion_matrix(targets, predictions)
    #C0 = C.copy().fill_diagonal(np.zeros(C.shape[0]))
    d = cm.diagonal()

    acc = d.sum() / cm.sum()
    acc_cls = d / cm.sum(axis=1)
    tp = d.sum()
    dice = (2*tp) / (2*tp + (cm - np.diag(d)).sum())
    dice_cls = 2*d / (2*d + (cm - np.diag(d)).sum(axis=0) + (cm - np.diag(d)).sum(axis=0))
    return acc, acc_cls, dice, dice_cls