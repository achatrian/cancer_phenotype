import numpy as np
from sklearn.metrics import confusion_matrix
from torch import nn
import torch

# also have a look at pandas_ml confusion matrix
# https://stackoverflow.com/questions/2148543/how-to-write-a-confusion-matrix-in-python/30385488
EPS = 0.1


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

    acc = (d.sum() + EPS) / (cm.sum() + EPS)
    acc_cls = (d + EPS) / (cm.sum(axis=1) + EPS)
    tp = d.sum()
    dice = (2*tp + EPS) / (2*tp + (cm - np.diag(d)).sum() + EPS)
    dice_cls = (2*d + EPS) / (2*d + (cm - np.diag(d)).sum(axis=0) + (cm - np.diag(d)).sum(axis=0) + EPS)
    return acc, acc_cls, dice, dice_cls



class DiceLoss(nn.Module):
    def __init__(self, weight=None, num_class=3):
        super(DiceLoss, self).__init__()
        if num_class>1:
            self.sm = nn.Softmax()
        else:
            self.sm = nn.Sigmoid()

        if weight is None:
            self.weight = nn.Parameter(torch.tensor([1.0 for i in range(num_class)]).float(), requires_grad=False)
        else:
            self.weight = nn.Parameter(weight.float(), requires_grad=False)

    def forward(self, outputs, targets):
        targets_oh = torch.zeros_like(outputs)
        for i, t in enumerate(targets):
            targets_oh[i, int(t)] = 1.0
        return self.dice_loss(self.sm(outputs), targets_oh)

    def dice_loss(self, pred, target):
        r"""This definition generalize to real valued pred and target vector.
        Exact - for numpy arrays
        """

        smooth = 1.0
        intersection = (pred * target * self.weight).sum()
        w_pred = (pred * self.weight).sum()
        w_target = (target * self.weight).sum()
        return 1.0 - ((2. * intersection + smooth) / (w_pred + w_target + smooth)) + 0.0001

    #def tocuda(self):
        #self.weight = self.weight.cuda()

